import pickle

import numpy as np
import matplotlib.pyplot as plt


def hyperopt_plots(filename, showpoints=True):
    with open(filename, "rb") as f:
        trials = pickle.load(f, encoding="bytes")

    fit_degree = 4
    top_n = 10

    trials = [t for t in trials.trials if t["result"]["status"] == "ok"]
    print("%d successful trials" % len(trials))

    bestest = sorted(trials, key=lambda t: (t["result"]["loss"]))

    for i, b in enumerate(bestest[:top_n]):
        print(i, b["result"]["loss"],
              2 * np.sqrt(b["result"]["loss_variance"])
              if "loss_variance" in b["result"] else None,
              b["misc"]["vals"])

    for k in trials[0]["misc"]["vals"].keys():
        plt.figure()
        data = np.asarray([(t["misc"]["vals"][k][0], t["result"]["loss"])
                           for t in trials if len(t["misc"]["vals"][k]) > 0])

        sort = data[:, 0].argsort()
        data = data[sort]

        if "loss_variance" in trials[0]["result"]:
            std = 2 * np.sqrt([t["result"]["loss_variance"] for t in trials
                               if len(t["misc"]["vals"][k]) > 0])
            std = std[sort]
        else:
            std = 0.0

        plt.errorbar(data[:, 0], data[:, 1], yerr=std,
                     fmt="o")
        coefs = np.polyfit(data[:, 0], data[:, 1], fit_degree - 1)
        plt.plot(data[:, 0], np.dot(np.power(np.tile(data[:, 0],
                                                     (fit_degree, 1)).T,
                                             np.arange(fit_degree)[None,
                                             ::-1]),
                                    coefs))

        for i, t in enumerate(bestest[:top_n]):
            if len(t["misc"]["vals"][k]) > 0:
                plt.scatter(t["misc"]["vals"][k][0], t["result"]["loss"],
                            marker="$%d$" % i, color="red", s=150)

        plt.title(k)
    # plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
    #         plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])

    plt.show()
