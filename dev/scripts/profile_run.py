"""Profile one PyBADS run: where its time goes.

Runs one configuration of ``benchmark_targets.py`` at one seed, set up as
``population.py`` sets up a run (the same problem, start point, noise
stream and options, the same evaluations made before the run for a
configuration of the ``warmstart`` suite, and the same rule for the PyBADS
it imports: the checkout that holds this script, put first on
``sys.path``), and records:

* the result: the wall time of ``optimize()``, its ``total_time``, the
  target's time, ``overhead``, the evaluations, the iterations, the error
  against the target's minimum and the message;
* the stage times that the run stores (``optim_state["stage_times"]``):
  each second of the run is charged to one stage, the innermost one open,
  and the target's evaluations to the pseudo-stage ``target``. The stages
  are tabulated by path (as ``search/gp_rebuild/gp_fit``, exclusive of the
  stages nested in it, and inclusive of them), by top-level stage and by
  leaf (the last name of the path, summed over the places it occurs:
  ``gp_fit`` is every hyperparameter fit), with the number of entries of
  each, and checked: ``total_time`` less the stages and the target is the
  residual, a few microseconds;
* one row per iteration, from the cumulative snapshots of
  ``iteration_history["timer"]``, and a last row for what follows the last
  iteration (the final re-estimation and samples of a noisy run);
* with ``--cprofile``, a cProfile of ``optimize()`` alone: ``profile.prof``
  (for ``pstats`` or ``snakeviz``), ``profile.txt`` (the top 40 functions
  by cumulative and by own time), and ``buckets``, the cumulative time and
  the number of calls of a curated list of functions (``BUCKETS``). The
  profiler slows the run down, the more so where the calls are many and
  short: take the stage times from a run without it.

A PyBADS without stage timers (a commit from before them) stores no stage
times: the summary then holds the result and, with ``--cprofile``, the
buckets alone.

Output goes to ``DIR/<tag>/`` (``--out DIR``, by default a new campaign
directory under ``dev/scripts/runs/profile/``; ``--tag`` by default
``<label>_seed<seed>_<plain|cprof>``): ``summary.json``, with the
provenance of ``population.py``'s records and, as those records hold it,
the kind, number of rows and digest of the evaluations made before the run
(``precomputed``), and the cProfile files. The earlier run that makes
those evaluations is neither timed nor profiled.

Examples, from the repository root::

    python -u dev/scripts/profile_run.py --config ellipsoid_D3 --seed 0
    python -u dev/scripts/profile_run.py --config rosenbrock_D6 --seed 1 \\
        --cprofile --out dev/scripts/runs/profile/adhoc
    python -u dev/scripts/profile_run.py --config ellipsoid_D10 --seed 0 \\
        --options '{"max_fun_evals": 500}' --tag ellipsoid_D10_short

The BLAS threads are those of the environment, which ``summary.json``
records; ``profile_suite.py`` runs each configuration with one thread, as
``population.py`` does.
"""

import argparse
import cProfile
import io
import json
import platform
import pstats
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# The package of this checkout, whichever checkout is installed.
sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_targets as bt  # noqa: E402
import population as pp  # noqa: E402

DEFAULT_CAMPAIGNS = HERE / "runs" / "profile"
TARGET = "target"

# (label, file path suffix, function name): the cumulative time of the
# code objects whose file ends with the suffix (with forward slashes) and
# whose name is the function name, summed.
BUCKETS = [
    ("BADS.optimize", "pybads/bads/bads.py", "optimize"),
    ("_search_step_", "pybads/bads/bads.py", "_search_step_"),
    ("ESSearchHedge.__call__", "pybads/search/search_hedge.py", "__call__"),
    (
        "contraints_check",
        "pybads/function_logger/constraints_check.py",
        "contraints_check",
    ),
    (
        "acq_fcn_lcb",
        "pybads/acquisition_functions/acq_fcn_lcb.py",
        "acq_fcn_lcb",
    ),
    ("_poll_step_", "pybads/bads/bads.py", "_poll_step_"),
    (
        "local_gp_fitting",
        "pybads/bads/gaussian_process_train.py",
        "local_gp_fitting",
    ),
    (
        "_robust_gp_fit_",
        "pybads/bads/gaussian_process_train.py",
        "_robust_gp_fit_",
    ),
    ("GP.fit", "gpyreg/gaussian_process.py", "fit"),
    ("GP.predict", "gpyreg/gaussian_process.py", "predict"),
    ("GP.update", "gpyreg/gaussian_process.py", "update"),
    # The kernel of every PyBADS GP is the rational-quadratic ARD one: the
    # other kernels' `compute`, in the same file, are never called
    (
        "RationalQuadraticARD.compute",
        "gpyreg/covariance_functions.py",
        "compute",
    ),
    ("cholesky", "scipy/linalg/_decomp_cholesky.py", "cholesky"),
    ("solve_triangular", "scipy/linalg/_basic.py", "solve_triangular"),
    ("SliceSampler.sample", "gpyreg/slice_sample.py", "sample"),
    (
        "IterationHistory.record",
        "pybads/utils/iteration_history.py",
        "record",
    ),
    ("deepcopy", "/copy.py", "deepcopy"),
]


# --------------------------------------------------------------------------
# Stage times
# --------------------------------------------------------------------------


def _add(table, key, seconds, calls=0):
    entry = table.setdefault(key, {"s": 0.0, "calls": 0})
    entry["s"] += seconds
    entry["calls"] += calls


def stage_views(stored):
    """The views of ``optim_state["stage_times"]``: ``paths`` (exclusive
    seconds and entries of each path), ``inclusive`` (the seconds of each
    path with the paths nested in it, and its entries), ``top_level`` (the
    inclusive seconds and the entries of each top-level stage) and ``leaf``
    (the exclusive seconds and the entries of each last name, over all its
    paths). ``target`` is a top-level key and a leaf, without entries."""
    seconds, calls = stored["seconds"], stored["calls"]
    paths, inclusive, top, leaf = {}, {}, {}, {}
    for path, s in seconds.items():
        n = int(calls.get(path, 0))
        _add(paths, path, s, n)
        _add(leaf, path.rsplit("/", 1)[-1], s, n)
        parts = path.split("/")
        for k in range(1, len(parts) + 1):
            prefix = "/".join(parts[:k])
            _add(inclusive, prefix, s, n if k == len(parts) else 0)
        _add(top, parts[0], s, n if len(parts) == 1 else 0)
    return {
        "paths": paths,
        "inclusive": inclusive,
        "top_level": top,
        "leaf": leaf,
    }


def per_iteration(bads, stored):
    """One row per recorded iteration: the seconds of each top-level stage
    during it (the difference of two cumulative snapshots of
    ``iteration_history["timer"]``), its evaluations so far, its incumbent's
    ``fval`` and its mesh size; then a row ``"end"`` for the time after the
    last snapshot."""
    history = bads.iteration_history
    snapshots = history["timer"]
    if snapshots is None:
        return []
    rows = []
    previous = {}
    for i, snapshot in enumerate(snapshots):
        if snapshot is None:
            continue
        top = pp.top_level_seconds(snapshot["seconds"])
        row = {"iter": i}
        for k in sorted(set(top) | set(previous)):
            row[k] = top.get(k, 0.0) - previous.get(k, 0.0)
        row["func_count"] = int(history["func_count"][i])
        row["fval"] = float(history["fval"][i])
        row["mesh_size"] = float(history["mesh_size"][i])
        rows.append(row)
        previous = top
    final = pp.top_level_seconds(stored["seconds"])
    row = {"iter": "end"}
    for k in sorted(set(final) | set(previous)):
        row[k] = final.get(k, 0.0) - previous.get(k, 0.0)
    rows.append(row)
    return rows


# --------------------------------------------------------------------------
# cProfile
# --------------------------------------------------------------------------


def buckets(stats):
    """Cumulative time and outermost calls of each ``BUCKETS`` entry; a
    recursive function's cumulative time counts its outermost calls only,
    as cProfile's does."""
    out = []
    for label, suffix, name in BUCKETS:
        cum, calls, where = 0.0, 0, []
        for (path, line, func), (cc, nc, tt, ct, _) in stats.stats.items():
            path = path.replace("\\", "/")
            if func == name and path.endswith(suffix):
                cum += ct
                calls += cc
                where.append(f"{path.rsplit('/', 1)[-1]}:{line}")
        out.append(
            {"label": label, "cumtime": cum, "calls": calls, "where": where}
        )
    return out


def profile_text(stats, n=40):
    buf = io.StringIO()
    stats.stream = buf
    stats.sort_stats("cumulative").print_stats(n)
    stats.sort_stats("tottime").print_stats(n)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--config",
        required=True,
        help="a configuration label of benchmark_targets.py --list",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--options",
        default=None,
        help="JSON dict merged into the run's options",
    )
    ap.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="multiplies the configuration's budget, as in population.py",
    )
    ap.add_argument(
        "--cprofile",
        action="store_true",
        help="profile optimize() with cProfile",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="campaign directory (default: a new one under"
        " dev/scripts/runs/profile/)",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="run directory name (default <label>_seed<k>_<plain|cprof>)",
    )
    return ap.parse_args(argv)


def _print_table(title, table, total):
    print(f"[profile_run] {title}", flush=True)
    for key, entry in sorted(table.items(), key=lambda kv: -kv[1]["s"]):
        share = 100 * entry["s"] / total if total else float("nan")
        calls = entry["calls"] or ""
        print(
            f"    {key:44s} {entry['s']:9.3f} s {share:6.1f}% {calls:>7}",
            flush=True,
        )


def main(argv=None):
    args = parse_args(argv)
    cfg = bt.find_config(args.config)
    prob = cfg.make(seed=args.seed, budget_scale=args.budget_scale)
    bads_args, options = prob.bads_args()
    options.update(json.loads(args.options) if args.options else {})
    requested = pp.jsonable(options)

    mode = "cprof" if args.cprofile else "plain"
    out = (
        args.out or DEFAULT_CAMPAIGNS / f"adhoc_{int(time.time())}"
    ).resolve()
    tag = args.tag or f"{cfg.label}_seed{args.seed}_{mode}"
    run_dir = out / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[profile_run] {cfg.label} seed {args.seed} -> {run_dir}", flush=True
    )
    started = pp._now()

    from pybads import BADS

    bads = BADS(*bads_args, options=options, **prob.bads_kwargs())
    prof = cProfile.Profile() if args.cprofile else None
    t0 = time.perf_counter()
    if prof is not None:
        prof.enable()
    try:
        res = bads.optimize()
    finally:
        if prof is not None:
            prof.disable()
    wall = time.perf_counter() - t0

    x = np.asarray(res["x"], dtype=float).ravel()
    target_s = float(bads.function_logger.total_fun_eval_time)
    total_time = float(res["total_time"])
    result = {
        "wall_s": wall,
        "total_time": total_time,
        "target_s": target_s,
        "own_s": total_time - target_s,
        "overhead": float(res["overhead"]),
        "func_count": int(res["func_count"]),
        "iterations": int(res["iterations"]),
        "x": x.tolist(),
        "fval": float(np.asarray(res["fval"]).item()),
        "fsd": float(np.asarray(res["fsd"]).item()),
        "true_error": prob.f_true(x) - prob.f_min,
        "message": str(res["message"]),
    }
    summary = {
        "label": cfg.label,
        "seed": args.seed,
        "mode": mode,
        "problem": prob.name,
        "D": prob.D,
        "noise": prob.noise,
        "precomputed": pp.precomputed_summary(cfg, prob),
        "requested_options": requested,
        "result": result,
        "stages": None,
        "per_iteration": [],
        "buckets": None,
        "meta": {
            "git": pp.git_info(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": pp.pkg_version("scipy"),
            "pybads": pp.pkg_version("pybads"),
            "pybads_source": pp.module_source("pybads"),
            "gpyreg": pp.pkg_version("gpyreg"),
            "gpyreg_source": pp.module_source("gpyreg"),
            "threads": pp.thread_env(),
            "cprofile": bool(args.cprofile),
            "started": started,
            "finished": pp._now(),
        },
    }

    print(
        f"[profile_run] wall {wall:.2f} s, total_time {total_time:.2f} s,"
        f" target {target_s:.2f} s, {result['func_count']} evaluations,"
        f" {result['iterations']} iterations,"
        f" error {result['true_error']:.3g}: {result['message']}",
        flush=True,
    )
    stored = bads.optim_state.get("stage_times")
    if stored:
        views = stage_views(stored)
        residual = total_time - sum(stored["seconds"].values())
        summary["stages"] = dict(views, residual_s=residual)
        summary["per_iteration"] = per_iteration(bads, stored)
        print(
            "[profile_run] stages (s, % of total_time, entries);"
            f" total_time - stages - target = {residual:.2e} s",
            flush=True,
        )
        _print_table("top level", views["top_level"], total_time)
        _print_table("by leaf", views["leaf"], total_time)
        _print_table("by path (exclusive)", views["paths"], total_time)
    else:
        print(
            "[profile_run] no stage times: a PyBADS without them", flush=True
        )

    if prof is not None:
        stats = pstats.Stats(prof)
        stats.dump_stats(str(run_dir / "profile.prof"))
        (run_dir / "profile.txt").write_text(
            profile_text(stats), encoding="utf-8"
        )
        summary["buckets"] = buckets(stats)
        opt = summary["buckets"][0]["cumtime"]
        print(
            "[profile_run] cProfile buckets (cumulative s, % of optimize,"
            " outermost calls):",
            flush=True,
        )
        for b in summary["buckets"]:
            share = 100 * b["cumtime"] / opt if opt else float("nan")
            print(
                f"    {b['label']:30s} {b['cumtime']:9.3f} {share:6.1f}%"
                f" {b['calls']:>9d}",
                flush=True,
            )

    pp._write_json(run_dir / "summary.json", pp.jsonable(summary))
    print(f"[profile_run] done -> {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
