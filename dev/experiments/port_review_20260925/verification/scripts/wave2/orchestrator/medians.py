"""med.py REF NEW [config...]: median error and evaluations per configuration, and the end criteria."""
import collections
import glob
import json
import os
import sys

import numpy as np

ref, new = sys.argv[1:3]
only = set(sys.argv[3:])


def load(d):
    out = collections.defaultdict(list)
    for p in glob.glob(os.path.join(d, "*_seed*.json")):
        r = json.load(open(p))
        out[r["label"]].append(r["final"])
    return out


a, b = load(ref), load(new)
for c in sorted(a):
    if only and c not in only:
        continue

    def stats(rs):
        e = np.median([r["true_error"] for r in rs])
        n = np.median([r["func_count"] for r in rs])
        ends = collections.Counter(
            r.get("exit", r.get("status", r.get("message", "?")))[:18]
            if isinstance(
                r.get("exit", r.get("status", r.get("message", "?"))), str
            )
            else r.get("status")
            for r in rs
        )
        return e, n, ends

    ea, na, xa = stats(a[c])
    eb, nb, xb = stats(b[c])
    print(
        f"{c:26s} err {ea:.2g} -> {eb:.2g}   evals {na:.0f} -> {nb:.0f}   ends {dict(xa)} -> {dict(xb)}"
    )
