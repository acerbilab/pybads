"""Collect and tabulate the runs of timing_ab.py.

    python tabulate.py collect RAW_DIR > timing.json
    python tabulate.py table timing.json > timing.md

``collect`` keeps one row per run (arm, configuration, seed, wall time,
target's time, own time, evaluations, error, the machine's load, the
commits of PyBADS and gpyreg). The own time is the wall time of
``optimize()`` less the target's evaluations, computed the same way for
both arms (PyBADS 1.1.0 has no stage times). ``table`` gives, per
configuration, the medians of each arm over the seeds and two ratios of B
(PyBADS 1ecfeb61, gpyreg 1.4.0) to A (PyBADS 1.1.0, gpyreg 1.3.3): the
ratio of the medians, and the median of the ratios of the runs of one seed,
with a 95 % bootstrap interval. The two arms of one seed share the problem,
start point and noise, not the run: the code differs, so the runs differ in
their evaluations as in their time.
"""

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BOOT = 10000


def collect(raw):
    rows = []
    for f in sorted(Path(raw).glob("[AB]/*/summary.json")):
        d = json.loads(f.read_text())
        machine = f.parent / "machine.json"
        busy = (
            json.loads(machine.read_text())["busy"]
            if machine.exists()
            else None
        )
        r, m = d["result"], d["meta"]
        rows.append(
            {
                "arm": f.parent.parent.name,
                "label": d["label"],
                "seed": d["seed"],
                "wall_s": r["wall_s"],
                "target_s": r["target_s"],
                "own_s": r["wall_s"] - r["target_s"],
                "func_count": r["func_count"],
                "true_error": r["true_error"],
                "busy": busy,
                "pybads": m["pybads_source"]["git"]["sha"],
                "gpyreg": m["gpyreg_source"]["git"]["sha"],
                "started": m["started"],
            }
        )
    json.dump(rows, sys.stdout, indent=1)
    print()


def boot_median(xs, rng):
    meds = sorted(
        statistics.median(rng.choices(xs, k=len(xs))) for _ in range(BOOT)
    )
    return meds[int(0.025 * BOOT)], meds[int(0.975 * BOOT) - 1]


def table(path):
    rows = json.loads(Path(path).read_text())
    runs = defaultdict(dict)
    for r in rows:
        runs[r["label"]][(r["arm"], r["seed"])] = r
    rng = random.Random(0)
    med = statistics.median

    print(
        "| configuration | seeds | own A (s) | own B (s) | B / A of medians"
        " | median of B / A per seed [95 % CI] | B faster | evals A | evals B"
        " | own per eval A (ms) | own per eval B (ms) | error A | error B |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    total = {"A": 0.0, "B": 0.0}
    for label in sorted(runs):
        v = runs[label]
        seeds = sorted(s for (arm, s) in v if arm == "A" and ("B", s) in v)
        a = [v[("A", s)] for s in seeds]
        b = [v[("B", s)] for s in seeds]
        for arm, xs in (("A", a), ("B", b)):
            total[arm] += sum(x["own_s"] for x in xs)
        oa, ob = med(x["own_s"] for x in a), med(x["own_s"] for x in b)
        ratios = [y["own_s"] / x["own_s"] for x, y in zip(a, b)]
        lo, hi = boot_median(ratios, rng)
        faster = sum(q < 1 for q in ratios)
        print(
            f"| {label} | {len(seeds)} | {oa:.2f} | {ob:.2f} | {ob / oa:.3f}"
            f" | {med(ratios):.3f} [{lo:.3f}, {hi:.3f}]"
            f" | {faster}/{len(seeds)}"
            f" | {med(x['func_count'] for x in a):.0f}"
            f" | {med(x['func_count'] for x in b):.0f}"
            f" | {med(1e3 * x['own_s'] / x['func_count'] for x in a):.1f}"
            f" | {med(1e3 * x['own_s'] / x['func_count'] for x in b):.1f}"
            f" | {med(x['true_error'] for x in a):.3g}"
            f" | {med(x['true_error'] for x in b):.3g} |"
        )
    busy = [r["busy"] for r in rows if r["busy"] is not None]
    print()
    print(
        f"Own time: the wall time of `optimize()` less the target's"
        f" evaluations. Summed over every run: A {total['A']:.0f} s,"
        f" B {total['B']:.0f} s, B / A {total['B'] / total['A']:.3f}."
    )
    if busy:
        print(
            f"Load of the whole machine during a run (all logical CPUs):"
            f" median {100 * med(busy):.1f} %, largest"
            f" {100 * max(busy):.1f} %."
        )


if __name__ == "__main__":
    {"collect": collect, "table": table}[sys.argv[1]](sys.argv[2])
