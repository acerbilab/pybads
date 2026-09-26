import collections

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

cur = {"n": 0}
hist = collections.Counter()
orig_fit = gpyreg.GP.fit


def counted_fit(self, *a, **k):
    try:
        return orig_fit(self, *a, **k)
    except np.linalg.LinAlgError:
        cur["n"] += 1
        raise


gpyreg.GP.fit = counted_fit
orig_rob = gpt._robust_gp_fit_


def rob(*a, **k):
    cur["n"] = 0
    try:
        return orig_rob(*a, **k)
    finally:
        hist[cur["n"]] += 1


gpt._robust_gp_fit_ = rob


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ackley(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / D))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / D)
        + 20
        + np.e
    )


def step(x):
    return float(np.sum(np.floor(np.abs(np.atleast_2d(x)) * 4)))


raised = []
for name, f, D in [
    ("ell6", ell, 6),
    ("rosen3", rosen, 3),
    ("ackley3", ackley, 3),
    ("step2", step, 2),
]:
    for seed in [0, 1, 2]:
        b = BADS(
            f,
            np.full((1, D), 1.5),
            np.full((1, D), -10.0),
            np.full((1, D), 10.0),
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        try:
            r = b.optimize()
            msg = f"fval {r['fval']:.3g}"
        except Exception as e:
            msg = f"RAISED {type(e).__name__}: {e}"
            raised.append((name, seed, msg))
        print(
            name,
            seed,
            msg,
            "| failures-per-refit histogram so far:",
            dict(sorted(hist.items())),
        )
print("raised:", raised)
