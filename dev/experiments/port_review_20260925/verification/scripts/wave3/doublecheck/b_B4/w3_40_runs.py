"""W3-40: (a) the geometry suite's runs that crashed at W3-24, from the tree
given, capped at 200 evaluations; (b) a thin band along a coordinate, which
the coordinate poll can reach, at the revision of the tree given."""
import logging
import sys
from pathlib import Path

tree = sys.argv[1]
sys.path.insert(0, str(Path(tree) / "dev" / "scripts"))
import benchmark_targets as bt  # noqa: E402
import gpyreg  # noqa: E402
import numpy as np  # noqa: E402

import pybads  # noqa: E402
import pybads.bads.gaussian_process_train as gpt  # noqa: E402
from pybads import BADS  # noqa: E402

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.CRITICAL)

two = []
orig = gpt.local_gp_fitting


def fitting(gp, u, function_logger, *a, **k):
    X = function_logger.X[function_logger.X_flag]
    n_distinct = len(np.unique(X, axis=0))
    out = orig(gp, u, function_logger, *a, **k)
    if out[0].X is not None and len(np.unique(out[0].X, axis=0)) == 2:
        two.append((len(out[0].X), n_distinct))
    return out


import pybads.bads.bads as bm  # noqa: E402

bm.local_gp_fitting = fitting


def report(name, make):
    two.clear()
    try:
        b = make()
        r = b.optimize()
        print(
            f"{name}: {r['func_count']} evaluations, fval {r['fval']:.3g}; "
            f"rebuilds on two distinct points {len(two)} {two[:3]}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"{name}: {type(e).__name__}: {str(e)[:80]}; rebuilds on two "
            f"distinct points before {len(two)}",
            flush=True,
        )


if len(sys.argv) > 2 and sys.argv[2] == "geometry":
    for label, seed in (
        ("sphere_band_D2", 13),
        ("sphere_band_D2", 23),
        ("sphere_band_D3", 15),
    ):
        cfg = bt.find_config(label)
        prob = cfg.make(seed=seed, budget_scale=1.0)
        args, options = prob.bads_args()
        options["max_fun_evals"] = 200
        report(f"{label} seed {seed}", lambda: BADS(*args, options=options))
else:
    c = np.array([1.0, 0.0])
    for seed in range(4):
        report(
            f"band along x1, seed {seed}",
            lambda: BADS(
                lambda x: float(np.sum((np.ravel(x) - c) ** 2)),
                np.array([-1.0, 0.0]),
                -5 * np.ones(2),
                5 * np.ones(2),
                -3 * np.ones(2),
                3 * np.ones(2),
                non_box_cons=lambda X: np.abs(np.atleast_2d(X)[:, 1]) > 1e-3,
                options={
                    "display": "off",
                    "random_seed": seed,
                    "max_fun_evals": 100,
                },
            ),
        )
