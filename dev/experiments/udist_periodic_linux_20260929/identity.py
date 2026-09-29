"""Compare two populations of ``population.py`` run by run: each record,
its timings (``wall_s`` and ``stage_times``) and ``meta`` left out, must be
equal. Prints, per
configuration, the runs that are identical and the median of the paired
ratios of the wall times, NEW over BASE, and exits 1 unless every run of
BASE has an identical run in NEW.

    python identity.py BASE_DIR NEW_DIR
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(directory):
    records = {}
    for path in sorted(Path(directory).glob("*_seed*.json")):
        record = json.loads(path.read_text())
        records[(record["label"], record["seed"])] = record
    return records


def comparable(record):
    record = dict(record)
    record.pop("meta", None)
    record["final"] = {
        k: v
        for k, v in record["final"].items()
        if k not in ("wall_s", "stage_times")
    }
    return record


def main(base_dir, new_dir):
    base, new = load(base_dir), load(new_dir)
    same = defaultdict(int)
    total = defaultdict(int)
    ratios = defaultdict(list)
    for key, record in base.items():
        label = key[0]
        total[label] += 1
        other = new.get(key)
        if other is None:
            continue
        if comparable(record) == comparable(other):
            same[label] += 1
        ratios[label].append(
            other["final"]["wall_s"] / record["final"]["wall_s"]
        )
    print("| Configuration | Identical runs | Median wall time ratio |")
    print("|---|---|---|")
    for label in total:
        print(
            f"| {label} | {same[label]} of {total[label]} | "
            f"{statistics.median(ratios[label]):.2f} |"
        )
    n_same, n_total = sum(same.values()), sum(total.values())
    print(f"\n{n_same} of {n_total} runs identical, timings aside")
    return 0 if n_same == n_total else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
