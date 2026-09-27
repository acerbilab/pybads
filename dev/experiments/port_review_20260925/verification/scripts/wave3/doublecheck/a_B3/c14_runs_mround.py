"""F1: seeded runs with contraints_check as at 0d866e8 and with MATLAB's
round in its bins (the same code otherwise), side by side."""
import hdr  # noqa: F401
import numpy as np
from cc_variants import port_mround

import pybads.bads.bads as bads_module
import pybads.search.es_search as es_module
from pybads import BADS

port_cc = es_module.contraints_check


def edge_sphere(x):
    return float(np.sum((np.atleast_2d(x) + 1.0) ** 2))


def noisy_sphere(seed):
    r = np.random.default_rng(seed)
    return lambda x: float(np.sum(np.atleast_2d(x) ** 2)) + r.standard_normal()


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2)
    )


def configs(seed):
    return [
        (
            "edge_D2",
            edge_sphere,
            2.5 * np.ones(2),
            np.zeros(2),
            5 * np.ones(2),
            np.zeros(2),
            5 * np.ones(2),
            {},
        ),
        (
            "rosen_D3",
            rosen,
            np.zeros(3),
            -5 * np.ones(3),
            5 * np.ones(3),
            -2 * np.ones(3),
            2 * np.ones(3),
            {},
        ),
        (
            "noisy_D2",
            noisy_sphere(seed),
            2 * np.ones(2),
            -10 * np.ones(2),
            10 * np.ones(2),
            -5 * np.ones(2),
            5 * np.ones(2),
            {"uncertainty_handling": True},
        ),
    ]


for seed in (0, 1, 2):
    for i in range(3):
        out = []
        for fn in (port_cc, port_mround):
            es_module.contraints_check = fn
            bads_module.contraints_check = fn
            name, f, x0, lb, ub, plb, pub, extra = configs(seed)[i]
            r = BADS(
                f,
                x0,
                lb,
                ub,
                plb,
                pub,
                options={
                    "display": "off",
                    "max_fun_evals": 200,
                    "random_seed": seed,
                    **extra,
                },
            ).optimize()
            out.append((r["fval"], r["func_count"], np.asarray(r["x"]).copy()))
        (f0, n0, x0_), (f1, n1, x1_) = out
        same = np.array_equal(x0_, x1_) and n0 == n1 and f0 == f1
        print(
            f"{name} seed {seed}: 0d866e8 fval {f0:.6g} evals {n0}; MATLAB's round fval {f1:.6g} evals {n1}; identical {same}",
            flush=True,
        )
