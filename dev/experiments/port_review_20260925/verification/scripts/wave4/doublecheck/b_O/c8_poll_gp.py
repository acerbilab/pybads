"""W4-28: does _poll_step_ return the GP object it was given, in every poll
of a few runs at levels 0, 1, 2 and with stobads?"""
import warnings

import hdr  # noqa
import numpy as np

from pybads import BADS

warnings.simplefilter("ignore")
stats = {"polls": 0, "same": 0}
orig = BADS._poll_step_


def poll(self, gp):
    out = orig(self, gp)
    stats["polls"] += 1
    stats["same"] += out[4] is gp
    return out


BADS._poll_step_ = poll
D = 3
lb, ub = -10 * np.ones((1, D)), 10 * np.ones((1, D))
plb, pub = -5 * np.ones((1, D)), 5 * np.ones((1, D))


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


for label, opts, noisy in [
    ("level 0", {}, 0),
    ("level 1", {"uncertainty_handling": True}, 1),
    (
        "level 2",
        {"uncertainty_handling": True, "specify_target_noise": True},
        2,
    ),
    ("stobads", {"uncertainty_handling": True, "stobads": True}, 1),
]:
    rng = np.random.default_rng(5)

    def f(x):
        v = rosen(x) + (0.5 * rng.normal() if noisy else 0.0)
        return (v, 0.5) if noisy == 2 else v

    stats.update(polls=0, same=0)
    o = {"display": "off", "random_seed": 5, "max_fun_evals": 200}
    o.update(opts)
    r = BADS(
        f, np.array([[1.5, -1.0, 0.5]]), lb, ub, plb, pub, options=o
    ).optimize()
    print(
        f"{label}: {stats['same']} of {stats['polls']} polls return the GP they were given; func_count {r['func_count']}"
    )
