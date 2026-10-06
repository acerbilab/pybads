"""round_half_away's return type for a scalar; fsd of a noisy run that
ends within its first iteration with noise_final_samples = 0."""
import warnings

import hdr  # noqa
import numpy as np

from pybads import BADS
from pybads.rounding import round_half_away

warnings.simplefilter("ignore")
print(
    "round_half_away(2.5):",
    repr(round_half_away(2.5)),
    type(round_half_away(2.5)),
)
print("round_half_away([2.5]):", repr(round_half_away([2.5])))
D = 2
rng = np.random.default_rng(1)
f = lambda x: float(np.sum(np.ravel(x) ** 2)) + 0.7 * rng.normal()
for nfs, mfe in ((0, 38), (10, 38), (10, 33), (10, 34)):
    b = BADS(
        f,
        np.ones((1, D)),
        -10 * np.ones((1, D)),
        10 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        options={
            "display": "off",
            "random_seed": 2,
            "max_fun_evals": mfe,
            "uncertainty_handling": True,
            "noise_final_samples": nfs,
        },
    )
    r = b.optimize()
    print(
        f"noise_final_samples={nfs} max_fun_evals={mfe}: iterations {r['iterations']} func_count {r['func_count']} "
        f"fsd={r['fsd']} yval_vec={None if r['yval_vec'] is None else np.size(r['yval_vec'])}"
    )
