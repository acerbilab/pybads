"""F1: effect of the swap on the reviewer's ellipsoid class, other seeds."""
import logging

import common  # noqa
import numpy as np

import pybads.bads.bads as bb
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)

orig_init = bb.BADS._init_optim_state_


def fixed_init(self):
    os_ = orig_init(self)
    os_["plb"], os_["pub"] = os_["pub"].copy(), os_["plb"].copy()
    return os_


R = np.linalg.qr(np.random.default_rng(3).standard_normal((3, 3)))[0]


def f_old(x):
    # axis-aligned ill-conditioned quadratic plus a mild quartic
    x = np.ravel(x)
    return float(
        x[0] ** 2 + (5 * x[1]) ** 2 + (30 * x[2]) ** 2 + 0.1 * x[0] ** 4
    )


D = 4
x0 = np.full((1, D), 1.5)


def f(x):
    x = np.ravel(x)
    return float(np.sum((10 ** (np.arange(4) / 3 * 2) * x) ** 2))


plb, pub = -3 * np.ones((1, D)), 3 * np.ones((1, D))
out = {}
for name, init in (("as is", orig_init), ("fixed", fixed_init)):
    bb.BADS._init_optim_state_ = init
    vals = []
    for seed in range(30, 36):
        b = BADS(
            f,
            x0,
            None,
            None,
            plb,
            pub,
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 150,
            },
        )
        r = b.optimize()
        vals.append(r["fval"])
    out[name] = np.log10(np.array(vals))
    print(name, "log10 fval:", np.round(out[name], 2))
bb.BADS._init_optim_state_ = orig_init
d = out["as is"] - out["fixed"]
print(
    "as is minus fixed (log10):",
    np.round(d, 2),
    "worse as is in",
    int(np.sum(d > 0)),
    "of",
    len(d),
)
