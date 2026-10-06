"""Consequence of the coordinate poll: default runs against runs whose poll
uses an LTMADS basis with entries bounded by mesh_size/search_mesh_size and
steps on the search mesh (a monkeypatched poll_mads_2n, for measurement
only), on a nonsmooth function whose descent direction is diagonal, and on
smooth ones."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS

orig = bm.poll_mads_2n


def ltmads(dim_x, poll_scale, search_mesh_size, mesh_size, rng=None):
    n = max(1.0, np.round(mesh_size / search_mesh_size))
    return orig(dim_x, poll_scale, mesh_size, search_mesh_size, rng=rng) / n


def ridge(x):  # coordinate directions fail on the ridge x1 = x2 > 0
    x = np.ravel(x)
    return float(10 * abs(x[0] - x[1]) + abs(x[0] + x[1]))


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ridge4(x):
    x = np.ravel(x)
    return float(10 * np.sum(np.abs(np.diff(x))) + abs(np.sum(x)))


for name, f, D, x0 in [
    ("ridge D2", ridge, 2, [1.5, 1.5]),
    ("ridge D4", ridge4, 4, [1.5] * 4),
    ("rosen D3", rosen, 3, [1.5] * 3),
]:
    for seed in [1, 2, 3]:
        res = []
        for variant in ["coded", "ltmads"]:
            bm.poll_mads_2n = orig if variant == "coded" else ltmads
            lb = -5 * np.ones((1, D))
            ub = 5 * np.ones((1, D))
            plb = -2 * np.ones((1, D))
            pub = 2 * np.ones((1, D))
            r = BADS(
                f,
                np.atleast_2d(x0),
                lb,
                ub,
                plb,
                pub,
                options={
                    "random_seed": seed,
                    "display": "off",
                    "max_fun_evals": 200,
                },
            ).optimize()
            res.append(
                "%s fval %.3g evals %d (%s)"
                % (
                    variant,
                    r["fval"],
                    r["func_count"],
                    r["message"].split(":")[1][:30],
                )
            )
        print(name, seed, " | ".join(res))
