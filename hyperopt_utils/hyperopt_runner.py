import ast
import copy
import multiprocessing
import os
import pickle
import queue
import subprocess
import threading
import time
from functools import wraps, partial

import numpy as np
import hyperopt as hyp


def hyperopt_wrapper(objective):
    """Convenience wrapper for objective function, prints the input arguments
    and how many trials have been run."""

    @wraps(objective)
    def tmp(*args, **kwargs):
        print(tmp.count)
        print(args, kwargs)
        tmp.count += 1

        return objective(*args, **kwargs)

    tmp.count = 0

    return tmp


def submit_and_monitor(command, run_time="3h", mem="4g", cores=1,
                       queue="serial", label=None):
    """Submits the given command to the cluster and waits for it to finish."""

    if label is None:
        label = int((time.time() * 1e5) % 1e10)
    tmpfile = "tmp_%s.sh" % label

    cmd = "sqsub -v -q %s -r %s --mpp=%s -n %d -o %s ./%s" % (
        queue, run_time, mem, cores, label, tmpfile)

    with open(tmpfile, "w") as f:
        f.write(command)

    mode = os.stat(tmpfile).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(tmpfile, mode)

    print("submitting:", cmd)

    # submit jobs
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                     shell=True)
    print(output)

    job_num = output.split()[-1]
    print("job_num", job_num)

    # monitor qstat to see if jobs are still running
    jobs_running = True
    while jobs_running:
        time.sleep(60)

        jobs_running = False
        try:
            output = subprocess.check_output("sqjobs", shell=True)
        except Exception as e:
            # sometimes sqjobs just fails for some reason, we'll just ignore
            # and continue
            print(e)
            continue

        for line in output.split("\n"):
            line_split = line.split()
            if (len(line_split) > 0 and line_split[0] == job_num and
                        line_split[2] != "D"):
                jobs_running = True
                break

    os.remove(tmpfile)

    print("job %s complete" % job_num)

    return job_num


def fmin_wrap(func, queue, *args, **kwargs):
    # wrapper function to call fmin and put the resulting
    # trials object in the results queue (so that we can get
    # results back from the subprocess)
    func(*args, **kwargs)
    queue.put(kwargs["trials"])


def optimize(objective, space, num_trials, output_file=None, init_trials=None,
             gamma=0.25, n_concurrent=15, lockstep=False, mode="thread"):
    """Optimizes the given objective over the search space, using a group
    of threads to run things in parallel."""

    assert n_concurrent > 0

    # set up the trials object that will store the results
    if init_trials is None:
        trials = hyp.Trials()
    else:
        with open(init_trials, "rb") as f:
            trials = pickle.load(f, encoding="bytes")

    n_trials = len(trials) + num_trials
    points_per = 1  # number of evaluation points per thread
    threads = [None for _ in range(n_concurrent)]
    results = [multiprocessing.Queue() for _ in range(n_concurrent)]
    crash_count = [0 for _ in range(n_concurrent)]
    in_progress = 0
    start = True

    # the algorithm that picks points in the search space
    algorithm = partial(hyp.tpe.suggest, gamma=gamma)

    print("init trials len", len(trials))

    begin = time.time()
    while len(trials) < n_trials:

        # iterate over all the threads, checking if they need to be started
        # up or closed down
        for i in range(n_concurrent):
            time.sleep(1.0)

            # check for start
            if (start and threads[i] is None and
                            len(trials) + in_progress < n_trials):
                print("starting %s %d" % (mode, i))

                os.environ["HYPEROPT_NUM"] = str(i)

                if mode == "thread":
                    sub_type = threading.Thread
                elif mode == "process":
                    sub_type = multiprocessing.Process
                else:
                    raise ValueError("Unknown mode")

                threads[i] = sub_type(
                    target=fmin_wrap,
                    args=(hyp.fmin, results[i], objective, space, algorithm,
                          len(trials) + points_per),
                    kwargs={
                        "rstate": np.random.RandomState(
                            np.random.randint(1e6)),
                        "trials": copy.deepcopy(trials),
                        "return_argmin": False})

                threads[i].daemon = True
                threads[i].start()
                in_progress += points_per
                crash_count[i] = 0

                if lockstep and np.all(np.not_equal(threads, None)):
                    # all threads have been started, so now don't start
                    # any more until they've all finished
                    start = False

            # check for finish
            elif threads[i] is not None:
                res = None
                try:
                    res = results[i].get(block=False)
                except queue.Empty:
                    if threads[i].is_alive():
                        # thread is still running, keep waiting
                        continue
                    else:
                        # we keep trying a few times (just in case it's
                        # being slow to show up in the queue)
                        if crash_count[i] > 2:
                            print("%s %d crashed" % (mode, i))
                        else:
                            print("%s %d possible crash? (%d)" %
                                  (mode, i, crash_count[i]))
                            crash_count[i] += 1
                            continue
                else:
                    # add results to trials
                    print("%s %d complete" % (mode, i))
                    trials.insert_trial_docs(res.trials[-points_per:])
                    trials.refresh()

                print("closing %s %d" % (mode, i))
                print("trials len", len(trials))
                print("successful trials",
                      len([t for t in trials.trials
                           if t["result"]["status"] == "ok"]))

                threads[i] = None
                in_progress -= points_per

                if output_file is not None:
                    with open(output_file, "wb") as f:
                        pickle.dump(trials, f)

                if lockstep and np.all(np.equal(threads, None)):
                    start = True

    end = time.time()
    print("run time:", end - begin)

    print("post trials len", len(trials))

    assert np.all(np.equal(threads, None))
    assert np.all([res.empty() for res in results])
    assert len(trials) - n_trials < points_per
    assert in_progress == 0

    return trials


def args2str(args):
    return str(args).replace(" ", "")


def str2args(args):
    return ast.literal_eval(args)
