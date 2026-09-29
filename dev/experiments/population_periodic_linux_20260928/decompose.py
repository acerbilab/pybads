"""Split the error of the noisy configurations of the `periodic` suite
between their periodic variables and the other one, per arm, and count the
runs that end with a periodic coordinate on a bound. Run from the
repository root: python
dev/experiments/population_periodic_linux_20260928/decompose.py
[DIR], DIR holding the arms' directories `on/` and `off/` (default: this
experiment's directory)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "dev/scripts")
import benchmark_targets as bt  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
SEAM = 2 * (1 - np.cos(0.3))  # the largest periodic term at a bound
for label, D, noise in [
    ("periodic_D3_hetero", 3, "hetero"),
    ("periodic_D3_homo", 3, "homo"),
]:
    for arm in ("on", "off"):
        per, non, evals, on_bound = [], [], [], []
        for f in sorted(glob.glob(f"{R}/{arm}/{label}_seed*.json")):
            r = json.load(open(f))
            prob = bt.make_problem("periodic", D, noise=noise, seed=r["seed"])
            x = np.array(r["final"]["x"], dtype=float).ravel()
            z = x - prob.x_min
            h = (D + 1) // 2
            per.append(np.sum(2 * (1 - np.cos(z[:h]))))
            non.append(np.sum(z[h:] ** 2))
            evals.append(r["final"]["func_count"])
            on_bound.append(
                np.any(
                    np.isclose(x[:h], prob.lb[:h])
                    | np.isclose(x[:h], prob.ub[:h])
                )
            )
        per, non = np.array(per), np.array(non)
        print(
            f"{label:20s} {arm:3s} periodic part median {np.median(per):.3g}"
            f" [{np.percentile(per, 25):.3g}, {np.percentile(per, 75):.3g}]"
            f"  non-periodic median {np.median(non):.3g}"
            f" [{np.percentile(non, 25):.3g}, {np.percentile(non, 75):.3g}]"
            f"  evals {np.median(evals):.0f}  runs {len(per)}, with a"
            f" periodic coordinate on a bound {int(np.sum(on_bound))}"
        )
