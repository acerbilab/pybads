import glob
import json
import sys

import numpy as np

sys.path.insert(0, "dev/scripts")
import benchmark_targets as bt

R = "dev/scripts/runs/periodic_20260928"
for label, D, noise in [
    ("periodic_D3_hetero", 3, "hetero"),
    ("periodic_D3_homo", 3, "homo"),
]:
    for arm in ("on", "off"):
        per, non, evals = [], [], []
        for f in sorted(glob.glob(f"{R}/{arm}/{label}_seed*.json")):
            r = json.load(open(f))
            seed = r["seed"]
            prob = bt.make_problem("periodic", D, noise=noise, seed=seed)
            x = np.array(r["final"]["x"], dtype=float).ravel()
            z = x - prob.x_min
            h = (D + 1) // 2
            per.append(np.sum(2 * (1 - np.cos(z[:h]))))
            non.append(np.sum(z[h:] ** 2))
            evals.append(r["final"]["func_count"])
        per, non = np.array(per), np.array(non)
        print(
            f"{label:20s} {arm:3s} periodic part median {np.median(per):.3g} "
            f"[{np.percentile(per,25):.3g}, {np.percentile(per,75):.3g}]  "
            f"non-periodic median {np.median(non):.3g} [{np.percentile(non,25):.3g}, {np.percentile(non,75):.3g}]  evals {np.median(evals):.0f}"
        )
