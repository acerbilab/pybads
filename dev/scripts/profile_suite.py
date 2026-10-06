"""Run ``profile_run.py`` over a suite and seeds, and aggregate the runs.

Each (configuration, seed, mode) runs in its own subprocess, one at a time,
with one BLAS thread (as ``population.py`` runs), each writing a log of its
own in the campaign directory. The modes are ``plain`` (the stage times)
and ``cprof`` (the same run under cProfile, for the buckets); ``both`` runs
every plain run first. A run whose directory holds a ``summary.json`` is
skipped, so a campaign resumes with the same ``--out``; a relative
``--out``, and a relative entry of ``PYTHONPATH`` (a gpyreg clone), is
taken from the working directory of the call. At the end, or
with ``--aggregate DIR`` alone, it writes ``aggregate.json`` (one row per
run) and ``aggregate.md`` (medians over the seeds of each configuration)
in the campaign directory. It exits 1 when a run fails or leaves no
``summary.json``, or when the campaign directory holds no run.

``--probe CONFIG`` runs one configuration plain before and after the
campaign (seed 0, tags ``probe_start_...`` and ``probe_end_...``) and
prints the ratio of their wall times: a machine that slows down during a
campaign shows there.

Examples, from the repository root::

    python -u dev/scripts/profile_suite.py --suite profile --seeds 0-2 \\
        --mode both --out dev/scripts/runs/profile/<campaign> \\
        > dev/scripts/runs/profile_suite_$(date +%s).log 2>&1
    python dev/scripts/profile_suite.py --aggregate \\
        dev/scripts/runs/profile/<campaign>

Arguments after ``--`` go to every ``profile_run.py``, as
``-- --options '{"max_fun_evals": 300}'``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_targets as bt  # noqa: E402
from harness import (  # noqa: E402
    jsonable,
    parse_seeds,
    single_thread_env,
    write_json,
)
from profile_run import DEFAULT_CAMPAIGNS  # noqa: E402

PROFILE_RUN = HERE / "profile_run.py"

# The top-level stages in the order of a run, and the leaves reported
TOP_LEVEL = (
    "init",
    "gp_init",
    "search",
    "poll",
    "history",
    "reestimate",
    "final_samples",
    "output_fcn",
    "loop",
)
LEAVES = (
    "search_es",
    "gp_fit",
    "gp_fit_failed",
    "gp_fit_retry",
    "gp_fit_fallback",
    "gp_rebuild",
    "gp_update",
    "target_from_gp",
)
GP_TRAINING = ("gp_fit", "gp_fit_failed", "gp_fit_retry", "gp_fit_fallback")


def run_one(label, seed, mode, out_dir, extra, tag=None):
    """Run one configuration in a child process, in the working directory
    ``REPO_ROOT``; ``out_dir`` is absolute, so that the run's directory, its
    log and the skip check are in the same place whatever the caller's
    working directory. A run is done when its ``summary.json`` exists."""
    tag = tag or f"{label}_seed{seed}_{mode}"
    summary = out_dir / tag / "summary.json"
    if summary.exists():
        print(f"[suite] skip {tag} (summary.json exists)", flush=True)
        return True
    log = out_dir / f"{tag}.log"
    cmd = [
        sys.executable,
        "-u",
        str(PROFILE_RUN),
        "--config",
        label,
        "--seed",
        str(seed),
        "--out",
        str(out_dir),
        "--tag",
        tag,
    ]
    if mode == "cprof":
        cmd.append("--cprofile")
    cmd += extra
    print(
        f"[suite] start {tag} at {time.strftime('%H:%M:%S')} -> {log.name}",
        flush=True,
    )
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(
            cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT
        )
    ok = rc == 0 and summary.exists()
    if ok:
        status = "done"
    elif rc == 0:
        status = f"FAILED: no {summary}"
    else:
        status = f"FAILED rc={rc}"
    print(f"[suite] {status} {tag} in {time.time() - t0:.0f} s", flush=True)
    return ok


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------


def run_row(summary, tag):
    """One run of a campaign, as ``aggregate.json`` holds it."""
    r = summary["result"]
    row = {
        "tag": tag,
        "label": summary["label"],
        "seed": summary["seed"],
        "mode": summary["mode"],
        "probe": tag.startswith("probe_"),
        "wall_s": r["wall_s"],
        "total_time": r["total_time"],
        "target_s": r["target_s"],
        "own_s": r["own_s"],
        "func_count": r["func_count"],
        "iterations": r["iterations"],
        "x": r["x"],
        "fval": r["fval"],
        "true_error": r["true_error"],
        "message": r["message"],
        "top_level": None,
        "leaf": None,
        "paths": None,
        "residual_s": None,
        "buckets": None,
        "commit": summary["meta"]["git"],
        "threads": summary["meta"]["threads"],
    }
    stages = summary.get("stages")
    if stages:
        for view in ("top_level", "leaf", "paths"):
            row[view] = {
                key: {"s": v["s"], "calls": v["calls"]}
                for key, v in stages[view].items()
            }
        row["residual_s"] = stages["residual_s"]
    if summary.get("buckets"):
        row["buckets"] = {
            b["label"]: {"s": b["cumtime"], "calls": b["calls"]}
            for b in summary["buckets"]
        }
    return row


def load_rows(out_dir):
    rows = []
    for path in sorted(Path(out_dir).glob("*/summary.json")):
        rows.append(run_row(json.loads(path.read_text()), path.parent.name))
    return rows


def finite_median(values):
    """The median of the finite values; None without any."""
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def fmt(v, nd=1):
    """A table's cell: a float to ``nd`` decimals, "-" for None."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _share(row, view, key):
    """The seconds of ``key`` in ``row[view]`` as a percentage of the own
    time; None without stage times."""
    if row[view] is None or not row["own_s"]:
        return None
    return 100 * row[view].get(key, {"s": 0.0})["s"] / row["own_s"]


def _groups(rows, mode):
    groups = {}
    for row in rows:
        if row["mode"] == mode and not row["probe"]:
            groups.setdefault(row["label"], []).append(row)
    return groups


def aggregate_text(rows, name):
    lines = [f"# Profile campaign {name}", ""]
    commits = {
        (r["commit"]["sha"], r["commit"]["dirty"], json.dumps(r["threads"]))
        for r in rows
    }
    lines += [
        "Commits, dirty flags and thread variables of the runs: "
        + "; ".join(sorted(f"{c[0]} dirty={c[1]} {c[2]}" for c in commits)),
        "",
    ]
    plain = _groups(rows, "plain")
    if plain:
        lines += [
            "## Plain runs: medians over the seeds",
            "",
            "Own time: `total_time` less the target's evaluations. Stage"
            " columns: % of the own time, each top-level stage with the"
            " stages nested in it; GP training: every hyperparameter fit,"
            " failed tries, retries and fallbacks included, wherever it"
            " happens. Residual: the largest |total_time - stages - target|"
            " over the seeds.",
            "",
            "| configuration | seeds | wall s | own s | ms/eval | evals |"
            " iters | error | "
            + " | ".join(TOP_LEVEL)
            + " | GP training | failed fits | ES search | residual s |",
            "|---" * (8 + len(TOP_LEVEL) + 4) + "|",
        ]
        for label, group in plain.items():
            timed = [r for r in group if r["top_level"] is not None]
            cells = [
                label,
                str(len(group)),
                fmt(finite_median([r["wall_s"] for r in group]), 2),
                fmt(finite_median([r["own_s"] for r in group]), 2),
                fmt(
                    finite_median(
                        [1e3 * r["own_s"] / r["func_count"] for r in group]
                    ),
                    1,
                ),
                fmt(finite_median([r["func_count"] for r in group]), 0),
                fmt(finite_median([r["iterations"] for r in group]), 0),
                fmt(finite_median([r["true_error"] for r in group]), 3),
            ]
            for key in TOP_LEVEL:
                cells.append(
                    fmt(
                        finite_median(
                            [_share(r, "top_level", key) for r in timed]
                        )
                    )
                )
            gp_training = [
                sum(_share(r, "leaf", key) for key in GP_TRAINING)
                for r in timed
            ]
            cells.append(fmt(finite_median(gp_training)))
            cells.append(
                fmt(
                    finite_median(
                        [_share(r, "leaf", "gp_fit_failed") for r in timed]
                    )
                )
            )
            cells.append(
                fmt(
                    finite_median(
                        [_share(r, "leaf", "search_es") for r in timed]
                    )
                )
            )
            residuals = [abs(r["residual_s"]) for r in timed]
            cells.append(f"{max(residuals):.1e}" if residuals else "-")
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "### Leaves: % of the own time (medians over the seeds;"
            " entries in parentheses)",
            "",
            "| configuration | " + " | ".join(LEAVES) + " |",
            "|---" * (1 + len(LEAVES)) + "|",
        ]
        for label, group in plain.items():
            timed = [r for r in group if r["leaf"] is not None]
            cells = [label]
            for key in LEAVES:
                share = finite_median([_share(r, "leaf", key) for r in timed])
                calls = finite_median(
                    [r["leaf"].get(key, {"calls": 0})["calls"] for r in timed]
                )
                cells.append(f"{fmt(share)} ({fmt(calls, 0)})")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    cprof = _groups(rows, "cprof")
    if cprof:
        labels = list(cprof)
        buckets = list(next(iter(cprof.values()))[0]["buckets"] or {})
        lines += [
            "## cProfile buckets: % of the profiled `optimize()` (medians"
            " over the seeds; outermost calls in parentheses)",
            "",
            "| bucket | " + " | ".join(labels) + " |",
            "|---" * (1 + len(labels)) + "|",
        ]
        for bucket in buckets:
            cells = [bucket]
            for label in labels:
                group = [r for r in cprof[label] if r["buckets"]]
                shares = [
                    100
                    * r["buckets"][bucket]["s"]
                    / r["buckets"]["BADS.optimize"]["s"]
                    for r in group
                ]
                calls = [r["buckets"][bucket]["calls"] for r in group]
                cells.append(
                    f"{fmt(finite_median(shares))} ({fmt(finite_median(calls), 0)})"
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "| profiled wall s | "
            + " | ".join(
                fmt(finite_median([r["wall_s"] for r in cprof[label]]), 2)
                for label in labels
            )
            + " |",
            "",
        ]
    probes = [r for r in rows if r["probe"]]
    if probes:
        lines += ["## Speed probe", ""]
        for r in probes:
            lines.append(f"- {r['tag']}: wall {r['wall_s']:.2f} s")
        lines.append("")
    return "\n".join(lines)


def aggregate(out_dir):
    """Write ``aggregate.json`` and ``aggregate.md`` of the runs in
    ``out_dir``; return the number of runs."""
    out_dir = Path(out_dir)
    rows = load_rows(out_dir)
    write_json(out_dir / "aggregate.json", jsonable(rows))
    text = aggregate_text(rows, out_dir.name)
    (out_dir / "aggregate.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"[suite] wrote {out_dir / 'aggregate.md'}", flush=True)
    return len(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1 :]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", default="profile", choices=list(bt.SUITES))
    ap.add_argument(
        "--mode", default="both", choices=["plain", "cprof", "both"]
    )
    ap.add_argument("--seeds", default="0", help='as "0-2" or "0,3,5-7"')
    ap.add_argument(
        "--only", default=None, help="comma-separated configuration labels"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="campaign directory (default: a new one under"
        " dev/scripts/runs/profile/)",
    )
    ap.add_argument(
        "--aggregate",
        type=Path,
        default=None,
        metavar="DIR",
        help="only aggregate the runs of a campaign directory",
    )
    ap.add_argument(
        "--probe",
        default=None,
        metavar="CONFIG",
        help="run this configuration plain before and after the campaign",
    )
    args = ap.parse_args(argv)

    if args.aggregate:
        if not aggregate(args.aggregate.resolve()):
            print(f"[suite] no runs in {args.aggregate}", flush=True)
            return 1
        return 0

    single_thread_env()  # inherited by the runs
    # absolute, from the caller's directory: the runs start in REPO_ROOT,
    # where a relative entry (a gpyreg clone) would name another directory,
    # or none, and the runs would import the installed gpyreg without notice
    if os.environ.get("PYTHONPATH"):
        os.environ["PYTHONPATH"] = os.pathsep.join(
            os.path.abspath(p)
            for p in os.environ["PYTHONPATH"].split(os.pathsep)
            if p
        )
    out_dir = (
        args.out or DEFAULT_CAMPAIGNS / f"campaign_{int(time.time())}"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = bt.suite_configs(args.suite)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = sorted(wanted - {c.label for c in cfgs})
        if unknown:
            sys.exit(f"not in suite {args.suite!r}: {', '.join(unknown)}")
        cfgs = [c for c in cfgs if c.label in wanted]
    seeds = parse_seeds(args.seeds)
    modes = ["plain", "cprof"] if args.mode == "both" else [args.mode]
    print(
        f"[suite] {len(cfgs)} configurations x {len(seeds)} seeds x"
        f" {modes} -> {out_dir}",
        flush=True,
    )
    t0 = time.time()
    ok = True
    probe = {}
    if args.probe:
        tag = f"probe_start_{args.probe}"
        ok &= run_one(args.probe, 0, "plain", out_dir, extra, tag=tag)
    for mode in modes:
        for cfg in cfgs:
            for seed in seeds:
                ok &= run_one(cfg.label, seed, mode, out_dir, extra)
    if args.probe:
        tag = f"probe_end_{args.probe}"
        ok &= run_one(args.probe, 0, "plain", out_dir, extra, tag=tag)
        for when in ("start", "end"):
            summary = out_dir / f"probe_{when}_{args.probe}" / "summary.json"
            if summary.exists():
                probe[when] = json.loads(summary.read_text())["result"][
                    "wall_s"
                ]
    print(
        f"[suite] campaign finished in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    if len(probe) == 2:
        print(
            f"[suite] speed probe {args.probe}: {probe['start']:.2f} s at"
            f" the start, {probe['end']:.2f} s at the end (ratio"
            f" {probe['end'] / probe['start']:.2f})",
            flush=True,
        )
    if not aggregate(out_dir):
        print(f"[suite] FAILED: no runs in {out_dir}", flush=True)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
