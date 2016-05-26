import numpy as np
import hyperopt as hyp

from hyperopt_utils import hyperopt_runner

# the objective function hyperopt will run
@hyperopt_runner.hyperopt_wrapper
def objective(args):
    output = "test_%d" % np.random.randint(1e8)

    # the command that will actually be executed on the cluster
    cmd = "python -c print(%f**2) > %s" % (args["x"], output)

    # submit that command to the cluster and wait for it to finish
    hyperopt_runner.submit_and_monitor(cmd, mem="100m")

    # read in the results (however the 'cmd' function outputs them)
    with open(output) as f:
        y = float(f.readline())

    # compute the loss as appropriate for the objective
    return {"loss": (y - args["target"]) ** 2,
            "status": hyp.STATUS_OK}

# the parameter search space
space = {"x": hyp.hp.uniform("x", -3, 3),
         "target": 4}

# run hyperopt. here we're saying we want to run 10 trials, with 5 trials
# running simultaneously.
trials = hyperopt_runner.optimize(objective, space, 10, n_concurrent=5)

# print trials sorted by loss
results = sorted(trials.trials,
                 key=lambda t: t["result"]["loss"])

for i, t in enumerate(results):
    print(i, t["result"]["loss"], t["misc"]["vals"])