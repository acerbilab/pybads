"""medians.py DIR...: median true_error and func_count per configuration, one column pair per population."""
import glob
import json
import os
import sys

import numpy as np

dirs = sys.argv[1:]
data = {}
for d in dirs:
    for p in glob.glob(os.path.join(d, "*_seed*.json")):
        r = json.load(open(p))
        data.setdefault(r["label"], {}).setdefault(d, []).append(
            (r["final"]["true_error"], r["final"]["func_count"])
        )
names = [os.path.basename(d.rstrip("/")) for d in dirs]
print(
    "| config | "
    + " | ".join(f"{n}: median error, evaluations" for n in names)
    + " |"
)
print("|---|" + "---|" * len(names))
for lab in sorted(data):
    cells = []
    for d in dirs:
        v = np.array(data[lab].get(d, []), dtype=float)
        cells.append(
            f"{np.median(v[:,0]):.3g}, {np.median(v[:,1]):.0f}"
            if len(v)
            else "-"
        )
    print(f"| {lab} | " + " | ".join(cells) + " |")
