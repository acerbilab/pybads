import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

rec = []
orig = gpt.local_gp_fitting


def wrapped(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    out = orig(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)
    if refit_flag:
        g = out[0]
        rec.append(
            (
                np.array(g.temporary_data["poll_scale"]).copy(),
                np.array(g.temporary_data["len_scale"]).copy(),
                optim_state["lb"].copy(),
                optim_state["ub"].copy(),
                optim_state["plb"].copy(),
                optim_state["pub"].copy(),
            )
        )
    return out


import pybads.bads.bads as bb

bb.local_gp_fitting = wrapped


def ell(x):
    x = np.atleast_2d(x)
    return float(np.sum((np.array([1.0, 10.0, 0.3]) * x) ** 2))


for case, (lb, ub, plb, pub) in {
    "unbounded": ([-np.inf] * 3, [np.inf] * 3, [-5] * 3, [5] * 3),
    "bounded": ([-20] * 3, [20] * 3, [-5] * 3, [5] * 3),
    "half lb<plb": ([-20] * 3, [np.inf] * 3, [-5] * 3, [5] * 3),
}.items():
    rec.clear()
    b = BADS(
        ell,
        np.array([[2.0, 2.0, 2.0]]),
        np.array([lb], float),
        np.array([ub], float),
        np.array([plb], float),
        np.array([pub], float),
        options={"random_seed": 3, "display": "off", "max_fun_evals": 150},
    )
    r = b.optimize()
    ps, ls, lbu, ubu, plbs, pubs = rec[-1]
    print(case, "fval", r["fval"], "n refits", len(rec))
    print("  optim_state lb", lbu, "ub", ubu, "plb", plbs, "pub", pubs)
    print("  len_scale", ls, "poll_scale", ps)
    print(
        "  poll_scale over refits (first dim):",
        [float(p[0]) for p, *_ in rec][:8],
    )
