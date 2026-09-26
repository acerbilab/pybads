"""How often default runs draw a starting point from _get_random_samples_from_priors_, and what it gives."""
import traceback
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

orig = gpt._get_random_samples_from_priors_
log = []


def spy(gp, rng=None):
    out = orig(gp, rng)
    caller = traceback.extract_stack()[-2].name
    pri = gp.get_priors()
    D = gp.D
    log.append(
        (
            caller,
            float(out[0, D]),
            float(pri["covariance_log_outputscale"][1][0][0]),
            float(gp.upper_bounds[D]),
            float(np.mean(out[0, :D])),
            float(pri["covariance_log_lengthscale"][1][0][0]),
        )
    )
    return out


gpt._get_random_samples_from_priors_ = spy
probs = {
    "rosen4": (
        lambda x: float(
            np.sum(
                100 * (np.asarray(x)[1:] - np.asarray(x)[:-1] ** 2) ** 2
                + (1 - np.asarray(x)[:-1]) ** 2
            )
        ),
        4,
        0.0,
        -2.0,
        2.0,
    ),
    "ackley6": (
        lambda x: float(
            -20 * np.exp(-0.2 * np.sqrt(np.mean(np.asarray(x) ** 2)))
            - np.exp(np.mean(np.cos(2 * np.pi * np.asarray(x))))
            + 20
            + np.e
        ),
        6,
        3.3,
        -5.0,
        5.0,
    ),
    "sphere10": (
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        10,
        0.5,
        -1.0,
        1.0,
    ),
}
for name, (f, D, x0v, pl, pu) in probs.items():
    for seed in (0, 1):
        log.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = BADS(
                f,
                np.full(D, x0v),
                np.full(D, -32.0),
                np.full(D, 32.0),
                np.full(D, pl),
                np.full(D, pu),
                options={
                    "random_seed": seed,
                    "max_fun_evals": 200,
                    "display": "off",
                },
            )
            b.optimize()
        print(
            f"{name} seed {seed}: prior draws {len(log)}",
            [
                (
                    c,
                    round(s, 2),
                    round(m, 2),
                    round(ub, 2),
                    round(l, 2),
                    round(lm, 2),
                )
                for c, s, m, ub, l, lm in log[:4]
            ],
        )
print(
    "tuple: (caller, drawn log sf, prior mu log sf, upper bound log sf, mean drawn log ell, prior mu log ell)"
)
