"""Wall time per evaluation of the `periodic` suite: gpyreg's periodic
kernel before ("base") and after ("new") the circle map, and the same
problems without periodic variables ("off"), all run on one machine with
the same number of workers. Run from the repository root:

    python dev/experiments/periodic_kernel_linux_20260929/timing.py collect NEW BASE OFF OUT.json
    python dev/experiments/periodic_kernel_linux_20260929/timing.py table OUT.json

`collect` reads the records of three population directories and writes
one entry per run (configuration, seed, arm, evaluations, wall time);
`table` prints, per configuration, the median wall time per evaluation of
each arm, the ratios new/base (paired by seed, runs with the same number
of evaluations only, and all runs) and on/off, and the least-squares
slope of the time per evaluation on the number of evaluations."""
import glob
import json
import statistics as st
import sys
from collections import defaultdict

import numpy as np


def collect(dirs, out):
    rows = []
    for arm, d in zip(("new", "base", "off"), dirs):
        for f in sorted(glob.glob(f"{d}/*.json")):
            r = json.load(open(f))
            if "final" not in r:
                continue
            rows.append(
                {
                    "label": r["label"],
                    "seed": r["seed"],
                    "arm": arm,
                    "func_count": r["final"]["func_count"],
                    "wall_s": r["final"]["wall_s"],
                }
            )
    with open(out, "w") as f:
        json.dump(rows, f, indent=0)


def table(path):
    rows = json.load(open(path))
    by = defaultdict(dict)
    for r in rows:
        by[(r["label"], r["arm"])][r["seed"]] = r
    labels = sorted({r["label"] for r in rows})
    print(
        "| configuration | ms/eval base | ms/eval new | ms/eval off"
        " | new/base, paired (same evals: n, median) | new/base, paired (all)"
        " | base/off | new/off |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for lab in labels:
        b, n, o = by[(lab, "base")], by[(lab, "new")], by[(lab, "off")]
        pe = lambda a: [
            1e3 * r["wall_s"] / r["func_count"] for r in a.values()
        ]
        mb, mn, mo = st.median(pe(b)), st.median(pe(n)), st.median(pe(o))
        same = [
            n[s]["wall_s"] / b[s]["wall_s"]
            for s in b
            if s in n and n[s]["func_count"] == b[s]["func_count"]
        ]
        allr = [
            (n[s]["wall_s"] / n[s]["func_count"])
            / (b[s]["wall_s"] / b[s]["func_count"])
            for s in b
            if s in n
        ]
        print(
            f"| {lab} | {mb:.1f} | {mn:.1f} | {mo:.1f}"
            f" | {len(same)}, {st.median(same) if same else float('nan'):.2f}"
            f" | {st.median(allr):.2f} | {mb / mo:.2f} | {mn / mo:.2f} |"
        )
    print()
    print(
        "Time per evaluation against evaluations (least squares, ms per 100 evaluations):"
    )
    print(
        "| configuration | arm | slope | intercept at the median evals of off |"
    )
    print("|---|---|---|---|")
    for lab in labels:
        med_off = st.median(r["func_count"] for r in by[(lab, "off")].values())
        for arm in ("base", "new", "off"):
            a = by[(lab, arm)].values()
            x = np.array([r["func_count"] for r in a], float)
            y = np.array([1e3 * r["wall_s"] / r["func_count"] for r in a])
            k, c = np.polyfit(x, y, 1)
            print(f"| {lab} | {arm} | {100 * k:.2f} | {c + k * med_off:.1f} |")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2:5], sys.argv[5])
    else:
        table(sys.argv[2])
