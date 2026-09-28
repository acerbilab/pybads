"""The runs that the change moves, per configuration: paired by (label,
seed), the runs that differ in a field of ``final`` other than the wall
time, the median error and evaluations of each arm, the fraction solved,
and the paired log10 error ratio and change in evaluations over the
changed runs.

    python pairs.py REF_DIR NEW_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

FIELDS = (
    "x",
    "fval",
    "fsd",
    "true_error",
    "func_count",
    "iterations",
    "message",
    "crashed",
)
FLOOR = 1e-12


def load(d):
    out = {}
    for p in Path(d).glob("*_seed*.json"):
        r = json.loads(p.read_text())
        out[(r["label"], r["seed"])] = r
    return out


def same(a, b):
    return all(a["final"].get(k) == b["final"].get(k) for k in FIELDS)


def err(r):
    return r["final"]["true_error"]


def evals(r):
    return r["final"]["func_count"]


def solved(r):
    e = err(r)
    return not r["final"]["crashed"] and e is not None and e < r["tolerance"]


def main():
    ref, new = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(ref) & set(new))
    labels = sorted({k[0] for k in keys})
    print(
        "| config | pairs | changed | median error ref → new |"
        " median evaluations ref → new | solved ref → new |"
        " changed runs: median log10 error ratio | signed-rank p |"
        " median change in evaluations | crashed ref, new |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for label in labels:
        ks = [k for k in keys if k[0] == label]
        changed = [k for k in ks if not same(ref[k], new[k])]
        e_ref = np.median([err(ref[k]) for k in ks])
        e_new = np.median([err(new[k]) for k in ks])
        n_ref = np.median([evals(ref[k]) for k in ks])
        n_new = np.median([evals(new[k]) for k in ks])
        s_ref = np.mean([solved(ref[k]) for k in ks])
        s_new = np.mean([solved(new[k]) for k in ks])
        r = np.array(
            [
                np.log10((err(new[k]) + FLOOR) / (err(ref[k]) + FLOOR))
                for k in changed
            ]
        )
        dn = [evals(new[k]) - evals(ref[k]) for k in changed]
        med = f"{np.median(r):+.3f}" if len(r) else "—"
        w = (
            f"{wilcoxon(r).pvalue:.3f}"
            if len(r) >= 10 and np.any(r != 0)
            else "—"
        )
        dmed = f"{np.median(dn):+.1f}" if dn else "—"
        c_ref = sum(ref[k]["final"]["crashed"] for k in ks)
        c_new = sum(new[k]["final"]["crashed"] for k in ks)
        print(
            f"| `{label}` | {len(ks)} | {len(changed)} | {e_ref:.3g} → "
            f"{e_new:.3g} | {n_ref:g} → {n_new:g} | {s_ref:.2f} → "
            f"{s_new:.2f} | {med} | {w} | {dmed} | {c_ref}, {c_new} |"
        )


if __name__ == "__main__":
    main()
