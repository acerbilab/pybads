import warnings

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)


def run(D, mfe, noisy):
    rng = np.random.default_rng(0)

    def f(x):
        y = float(np.sum(np.atleast_2d(x) ** 2))
        return y + (rng.standard_normal() if noisy else 0.0)

    b = BADS(
        f,
        np.ones(D) * 0.5,
        -5 * np.ones(D),
        5 * np.ones(D),
        -2 * np.ones(D),
        2 * np.ones(D),
        options={"display": "off", "max_fun_evals": mfe, "random_seed": 1},
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = b.optimize()
    print(
        f"D={D} mfe={mfe} noisy={noisy}: func_count={r['func_count']} nfs={b.options['noise_final_samples']} mfe_after={b.options['max_fun_evals']} it={r['iterations']} yval_vec={None if r['yval_vec'] is None else r['yval_vec'].shape} nwarn={len(w)} msg={r['message'][:60]}"
    )


for mfe in [2, 3, 4, 5, 6]:
    run(2, mfe, False)
run(3, 4, False)
run(5, 7, False)
for mfe in [1, 2, 10, 25, 38, 60]:
    run(2, mfe, True)
