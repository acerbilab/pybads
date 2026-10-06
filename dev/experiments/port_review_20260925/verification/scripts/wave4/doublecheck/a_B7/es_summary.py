import json
from collections import Counter

import numpy as np


def load(p):
    return {r["seed"]: r for r in map(json.loads, open(p))}


files = [
    "es_w421.jsonl",
    "es_w41_force948.jsonl",
    "es_w41.jsonl",
    "es_w41_force2.jsonl",
    "es_w41_force500.jsonl",
]
R = {f: load(f) for f in files}
for f, r in R.items():
    fc = np.array([r[s]["func_count"] for s in range(30)])
    err = np.array([r[s]["true_error"] for s in range(30)])
    ss = Counter(tuple(r[s]["sobol_seeds"]) for s in range(30))
    print(
        f"{f:24s} mean {fc.mean():.1f} range {fc.min()}-{fc.max()} distinct {len(set(fc))} sd {fc.std(ddof=1):.2f} counts {sorted(Counter(fc).items())} err median {np.median(err):.2g} max {err.max():.2g}; sobol seeds: {len(ss)} distinct{'' if len(ss) > 3 else ' ' + str(dict(ss))}"
    )
a, b = R["es_w421.jsonl"], R["es_w41_force948.jsonl"]
same = sum(
    a[s]["func_count"] == b[s]["func_count"]
    and a[s]["x"] == b[s]["x"]
    and a[s]["fval"] == b[s]["fval"]
    for s in range(30)
)
print(
    "W4-1 forced 948 (no draw) equals W4-21 in",
    same,
    "of 30 (func_count, x, fval)",
)
