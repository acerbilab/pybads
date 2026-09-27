"""One benchmark configuration of the default suite, at the revision whose
extracted tree is given (its benchmark_targets.py puts that tree first on
sys.path), with max_fun_evals capped: the final func_count, fval and x of
each seed, as JSON lines."""
import json
import logging
import sys
from pathlib import Path

tree, label, seeds, cap = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
)
sys.path.insert(0, str(Path(tree) / "dev" / "scripts"))
import benchmark_targets as bt  # noqa: E402
import gpyreg  # noqa: E402
import numpy as np  # noqa: E402

import pybads  # noqa: E402
from pybads import BADS  # noqa: E402

print("pybads", pybads.__file__, file=sys.stderr, flush=True)
print("gpyreg", gpyreg.__file__, file=sys.stderr, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
a, b = map(int, seeds.split("-"))
for seed in range(a, b + 1):
    cfg = bt.find_config(label)
    prob = cfg.make(seed=seed, budget_scale=1.0)
    args, options = prob.bads_args()
    options["max_fun_evals"] = min(cap, options.get("max_fun_evals", cap))
    res = BADS(*args, options=options).optimize()
    print(
        json.dumps(
            {
                "seed": seed,
                "func_count": int(res["func_count"]),
                "fval": float(res["fval"]),
                "x": [float(v) for v in res["x"]],
            }
        ),
        flush=True,
    )
