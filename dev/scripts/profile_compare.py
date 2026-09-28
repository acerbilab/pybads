"""Compare two ``profile_suite.py`` campaigns, BASE and NEW.

Reads the ``aggregate.json`` of each campaign (``profile_suite.py
--aggregate DIR`` writes it) and pairs their runs by configuration, seed
and mode. For the plain runs of each configuration it prints the medians
over the seeds of the ratio NEW / BASE of the wall time, the own time
(``total_time`` less the target's evaluations) and each top-level stage,
the ratio of a control stage, and whether each pair ran the same trajectory
(the same returned point, value and number of evaluations). The control is
a stage the change under test does not reach (``--control``, ``gp_init`` by
default, one GP fit on the initial design): a control ratio far from 1
means that the machine ran at another speed during that configuration, and
its row measures nothing. A campaign of a PyBADS without stage timers has
no stages: its rows compare the wall and own times alone.

For the cProfile runs present in both it prints the median seconds of each
bucket, their ratio and the calls, and the median time per call of the
buckets of ``PER_CALL``.

Example, from the repository root::

    python dev/scripts/profile_compare.py dev/scripts/runs/profile/base \\
        dev/scripts/runs/profile/new
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from profile_suite import GP_TRAINING, TOP_LEVEL  # noqa: E402

PER_CALL = (
    "ESSearchHedge.__call__",
    "acq_fcn_lcb",
    "contraints_check",
    "local_gp_fitting",
    "GP.fit",
    "GP.predict",
    "GP.update",
    "RationalQuadraticARD.compute",
)


def load(path):
    rows = json.loads((Path(path) / "aggregate.json").read_text())
    return {
        (r["label"], r["seed"], r["mode"]): r for r in rows if not r["probe"]
    }


def same_trajectory(b, n):
    return (
        b["x"] == n["x"]
        and b["fval"] == n["fval"]
        and b["func_count"] == n["func_count"]
    )


def stage_seconds(row, key):
    """The seconds of a top-level stage, of ``"gp_training"`` (every fit,
    failed tries, retries and fallbacks, wherever they happen), or of a
    leaf or a path; None without stage times."""
    if row["top_level"] is None:
        return None
    if key == "gp_training":
        return sum(row["leaf"].get(k, {"s": 0.0})["s"] for k in GP_TRAINING)
    for view in ("top_level", "leaf", "paths"):
        if key in row[view]:
            return row[view][key]["s"]
    return 0.0


def _ratio(b, n):
    if b is None or n is None or not b > 0:
        return None
    return n / b


def _median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def pairs(base, new, mode):
    """``{label: [(base_row, new_row), ...]}`` over the seeds present in
    both campaigns."""
    out = {}
    for key in sorted(base):
        label, seed, run_mode = key
        if run_mode == mode and key in new:
            out.setdefault(label, []).append((base[key], new[key]))
    return out


def compare_plain(base, new, control):
    groups = pairs(base, new, "plain")
    if not groups:
        return
    stages = list(TOP_LEVEL) + ["gp_training"]
    print(
        "## Plain runs: medians over the seeds of NEW / BASE"
        f" (control: {control})\n"
    )
    header = ["configuration", "seeds", "wall", "own"] + stages
    header += ["control", "trajectory"]
    print("| " + " | ".join(header) + " |")
    print("|---" * len(header) + "|")
    for label, group in groups.items():
        cells = [label, str(len(group))]
        for key in ("wall_s", "own_s"):
            cells.append(
                _fmt(_median([_ratio(b[key], n[key]) for b, n in group]))
            )
        for key in stages + [control]:
            cells.append(
                _fmt(
                    _median(
                        [
                            _ratio(
                                stage_seconds(b, key), stage_seconds(n, key)
                            )
                            for b, n in group
                        ]
                    )
                )
            )
        differ = [b["seed"] for b, n in group if not same_trajectory(b, n)]
        cells.append("same" if not differ else f"differs, seeds {differ}")
        print("| " + " | ".join(cells) + " |")
    print()
    total_b = sum(b["wall_s"] for g in groups.values() for b, _ in g)
    total_n = sum(n["wall_s"] for g in groups.values() for _, n in g)
    print(
        f"Wall time of the paired plain runs: {total_b:.1f} s -> {total_n:.1f}"
        f" s (ratio {total_n / total_b:.3f})\n"
    )
    missing = sorted(
        f"{k[0]} seed {k[1]}" for k in base if k[2] == "plain" and k not in new
    )
    if missing:
        print("Not in NEW: " + ", ".join(missing) + "\n")


def compare_cprof(base, new):
    groups = pairs(base, new, "cprof")
    groups = {
        label: [(b, n) for b, n in group if b["buckets"] and n["buckets"]]
        for label, group in groups.items()
    }
    groups = {label: group for label, group in groups.items() if group}
    if not groups:
        return
    labels = list(groups)
    buckets = list(next(iter(groups.values()))[0][0]["buckets"])
    print(
        "## cProfile: bucket seconds, BASE -> NEW (median ratio)" " [calls]\n"
    )
    print("| bucket | " + " | ".join(labels) + " |")
    print("|---" * (1 + len(labels)) + "|")
    for bucket in buckets:
        cells = [bucket]
        for label in labels:
            group = groups[label]
            bs = _median([b["buckets"][bucket]["s"] for b, _ in group])
            ns = _median([n["buckets"][bucket]["s"] for _, n in group])
            ratio = _median(
                [
                    _ratio(
                        b["buckets"][bucket]["s"], n["buckets"][bucket]["s"]
                    )
                    for b, n in group
                ]
            )
            bc = _median([b["buckets"][bucket]["calls"] for b, _ in group])
            nc = _median([n["buckets"][bucket]["calls"] for _, n in group])
            calls = (
                _fmt(bc, 0) if bc == nc else f"{_fmt(bc, 0)}->{_fmt(nc, 0)}"
            )
            cells.append(f"{_fmt(bs)}->{_fmt(ns)} ({_fmt(ratio)}) [{calls}]")
        print("| " + " | ".join(cells) + " |")
    print()
    print("Time per call, ms, BASE -> NEW (median ratio)\n")
    print("| bucket | " + " | ".join(labels) + " |")
    print("|---" * (1 + len(labels)) + "|")
    for bucket in PER_CALL:
        if bucket not in buckets:
            continue
        cells = [bucket]
        for label in labels:
            group = groups[label]

            def per_call(row):
                b = row["buckets"][bucket]
                return 1e3 * b["s"] / b["calls"] if b["calls"] else None

            pb = _median([per_call(b) for b, _ in group])
            pn = _median([per_call(n) for _, n in group])
            ratio = _median(
                [_ratio(per_call(b), per_call(n)) for b, n in group]
            )
            cells.append(f"{_fmt(pb, 3)}->{_fmt(pn, 3)} ({_fmt(ratio)})")
        print("| " + " | ".join(cells) + " |")
    print()
    for label, group in groups.items():
        differ = [b["seed"] for b, n in group if not same_trajectory(b, n)]
        verdict = "same" if not differ else f"differs, seeds {differ}"
        print(f"- {label} (cProfile runs): trajectory {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="campaign directory of the baseline")
    ap.add_argument("new", help="campaign directory of the change")
    ap.add_argument(
        "--control",
        default="gp_init",
        help="a stage the change does not reach: a top-level stage, a leaf"
        " or a path (default gp_init)",
    )
    args = ap.parse_args(argv)
    base, new = load(args.base), load(args.new)
    compare_plain(base, new, args.control)
    compare_cprof(base, new)


if __name__ == "__main__":
    main()
