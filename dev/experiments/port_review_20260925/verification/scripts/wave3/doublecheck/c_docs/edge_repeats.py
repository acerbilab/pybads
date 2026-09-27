"""Repeated evaluations on a 2-D sphere with its minimum on a lower hard
bound (as the benchmark's edgesphere_D2: hard box [0, 10], plausible box
[1, 9], centre -1 in the first variable), 30 seeds, the PyBADS on
PYTHONPATH. A repeat is a call of the target at a point it was called at
before, the noise test's second call of x0 left out."""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
D = 2
tot = rep = runs = 0
for seed in range(30):
    rng = np.random.default_rng(1000 + seed)
    c = 1 + rng.random(D) * 8
    c[0] = -1.0
    x0 = 1 + rng.random(D) * 8
    calls = []

    def f(x):
        calls.append(tuple(np.ravel(x)))
        return float(np.sum((np.ravel(x) - c) ** 2))

    b = BADS(
        f,
        x0,
        lower_bounds=np.zeros(D),
        upper_bounds=np.full(D, 10.0),
        plausible_lower_bounds=np.ones(D),
        plausible_upper_bounds=np.full(D, 9.0),
        options={"display": "off", "random_seed": seed, "max_fun_evals": 200},
    )
    b.optimize()
    seq = calls[:1] + calls[2:]  # the noise test's second call of x0 left out
    n_rep = len(seq) - len(set(seq))
    tot += len(seq)
    rep += n_rep
    runs += n_rep > 0
print(f"repeats {rep} of {tot} evaluations, in {runs} of 30 runs", flush=True)
