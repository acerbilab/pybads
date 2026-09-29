"""The time that two populations of ``population.py`` spend in the GP's
rebuilds (``local_gp_fitting``, whose empirical prior takes ``udist`` of
the training inputs), paired by run: per configuration, the median ratio of
the own time of the stages ``*/gp_rebuild``, NEW over BASE, their median
share of BASE's own time (the target's evaluations left out), the median
time saved as a share of it, and the median ratio of the wall times.

    python speed.py BASE_DIR NEW_DIR
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def rebuild(record):
    paths = record["final"]["stage_times"]["paths"]
    return sum(t for path, t in paths.items() if path.endswith("gp_rebuild"))


def own(record):
    top = record["final"]["stage_times"]["top_level"]
    return sum(t for stage, t in top.items() if stage != "target")


def main(base_dir, new_dir):
    rows = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(base_dir).glob("*_seed*.json")):
        base = json.loads(path.read_text())
        new = json.loads((Path(new_dir) / path.name).read_text())
        row = rows[base["label"]]
        row["ratio"].append(rebuild(new) / rebuild(base))
        row["share"].append(rebuild(base) / own(base))
        row["saved"].append((rebuild(base) - rebuild(new)) / own(base))
        row["wall"].append(new["final"]["wall_s"] / base["final"]["wall_s"])
    print(
        "| Configuration | Rebuilds' own time, NEW / BASE | Their share of "
        "BASE's own time | Saved, share of BASE's own time | Wall time, "
        "NEW / BASE |"
    )
    print("|---|---|---|---|---|")
    for label, row in rows.items():
        med = {k: statistics.median(v) for k, v in row.items()}
        print(
            f"| {label} | {med['ratio']:.2f} | {100 * med['share']:.0f} % | "
            f"{100 * med['saved']:.1f} % | {med['wall']:.2f} |"
        )


if __name__ == "__main__":
    main(*sys.argv[1:3])
