"""W4-26: the "done" call's optim_state at level 0 and at level 1 with
noise_final_samples = 0."""
import warnings

import hdr  # noqa
import numpy as np

from pybads import BADS

warnings.simplefilter("ignore")
D = 2
lb, ub = -10 * np.ones((1, D)), 10 * np.ones((1, D))
plb, pub = -5 * np.ones((1, D)), 5 * np.ones((1, D))
for label, opts, noisy in [
    ("level 0 (noise test)", {}, False),
    ("level 0 (uh=False)", {"uncertainty_handling": False}, False),
    (
        "level 1 nfs=0",
        {"uncertainty_handling": True, "noise_final_samples": 0},
        True,
    ),
]:
    rng = np.random.default_rng(0)
    f = lambda x: float(
        np.sum(np.ravel(x) ** 2 + 0.3 * np.cos(4 * np.ravel(x)))
    ) + (0.5 * rng.normal() if noisy else 0)
    cap = {}
    o = {
        "display": "off",
        "random_seed": 0,
        "max_fun_evals": 120,
        "output_fcn": lambda x, s, st: cap.__setitem__(st, (x, s)) or False,
    }
    o.update(opts)
    b = BADS(f, np.ones((1, D)), lb, ub, plb, pub, options=o)
    r = b.optimize()
    x, s = cap["done"]
    print(
        f"{label}: level {b.optim_state['uncertainty_handling_level']}; done x == result x: "
        f"{np.allclose(np.ravel(x), np.ravel(r['x']))}; state fval == result fval: {s['fval'] == r['fval']}; "
        f"fsd {s['fsd']} vs {r['fsd']}; state u == b.u: {np.allclose(np.ravel(s['u']), np.ravel(b.u))}"
    )
