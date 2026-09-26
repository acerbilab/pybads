"""C-F9 at default options: Actions shown at polls vs MATLAB's per-poll action."""
import common
import numpy as np
from v_misc_probe import Probe

cases = [
    (
        "rosen D=2",
        lambda x: float(
            np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        ),
        2,
        False,
    ),
    (
        "ackley D=3",
        lambda x: float(
            -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
            - np.exp(np.mean(np.cos(2 * np.pi * x)))
            + 20
            + np.e
        ),
        3,
        False,
    ),
    ("noisy sphere D=2", None, 2, True),
]
for name, f, D, noisy in cases:
    for seed in range(2):
        nrng = np.random.default_rng(seed)
        ff = (
            (
                lambda x: float(
                    np.sum(np.ravel(x) ** 2) + 0.5 * nrng.standard_normal()
                )
            )
            if noisy
            else f
        )
        b = Probe(
            ff,
            np.full(D, 0.5),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(display="off", random_seed=seed, max_fun_evals=200),
        )
        b.optimize()
        mism = [
            (p["it"], p["shown"], p["matlab"])
            for p in b.polls
            if p["shown"] != p["matlab"]
        ]
        print(
            f"{name} seed {seed}: polls {len(b.polls)}, Actions differing from MATLAB's {len(mism)}: {mism[:4]}"
        )
