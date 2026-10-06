"""Collect and tabulate the runs of timing_abba.py.

    python tabulate.py collect RAW_DIR > timing.json
    python tabulate.py table timing.json > timing.md

``collect`` keeps one row per run (arm, repetition, configuration, seed,
own and wall time, result, gpyreg's commit); ``table`` gives each run's own
time in the four runs of its configuration and seed, the ratio of the faster
run of arm B (gpyreg 3e56dce) to the faster of arm A (gpyreg 1.3.3), and the
median ratio of each configuration.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def collect(raw):
    rows = []
    for f in sorted(Path(raw).glob("*_rep*/*/summary.json")):
        d = json.loads(f.read_text())
        arm, rep = f.parent.parent.name.split("_rep")
        r, m = d["result"], d["meta"]
        rows.append(
            {
                "arm": arm,
                "rep": int(rep),
                "label": d["label"],
                "seed": d["seed"],
                "own_s": r["own_s"],
                "wall_s": r["wall_s"],
                "func_count": r["func_count"],
                "fval": r["fval"],
                "gpyreg": m["gpyreg_source"]["git"]["sha"],
                "pybads": m["pybads_source"]["git"]["sha"],
                "started": m["started"],
            }
        )
    json.dump(rows, sys.stdout, indent=1)
    print()


def table(path):
    rows = json.loads(Path(path).read_text())
    runs = defaultdict(dict)
    for r in rows:
        runs[(r["label"], r["seed"])][(r["arm"], r["rep"])] = r
    ratios = defaultdict(list)
    print(
        "| configuration | seed | A rep 1 | A rep 2 | B rep 1 | B rep 2 |"
        " B / A | same result |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for (label, seed), v in sorted(runs.items()):
        own = {k: x["own_s"] for k, x in v.items()}
        a = min(own[("A", 1)], own[("A", 2)])
        b = min(own[("B", 1)], own[("B", 2)])
        ratios[label].append(b / a)
        same = len({(x["func_count"], x["fval"]) for x in v.values()}) == 1
        print(
            f"| {label} | {seed} | {own[('A', 1)]:.2f} | {own[('A', 2)]:.2f}"
            f" | {own[('B', 1)]:.2f} | {own[('B', 2)]:.2f} | {b / a:.3f}"
            f" | {'yes' if same else 'no'} |"
        )
    print()
    print("| configuration | median B / A |")
    print("|---|---|")
    meds = {k: statistics.median(v) for k, v in ratios.items()}
    for label, med in sorted(meds.items(), key=lambda t: t[1]):
        print(f"| {label} | {med:.3f} |")
    every = [x for v in ratios.values() for x in v]
    print()
    print(
        f"Own time in seconds; B / A is the faster run of B over the faster"
        f" of A. Median over all {len(every)} pairs:"
        f" {statistics.median(every):.3f}; configuration medians"
        f" {min(meds.values()):.3f} to {max(meds.values()):.3f}."
    )


if __name__ == "__main__":
    {"collect": collect, "table": table}[sys.argv[1]](sys.argv[2])
