"""Populations of seeded PyBADS runs of a benchmark suite: run them,
summarize one, compare two.

Targets and suites come from ``benchmark_targets.py``. Sub-commands, from
the repository root::

    python -u dev/scripts/population.py run --suite default --seeds 0-29 \
        --out dev/scripts/runs/population/<name>
    python dev/scripts/population.py summary DIR
    python dev/scripts/population.py compare REF NEW [--alpha 0.05]
    python dev/scripts/population.py compare REF --split

``run`` executes each (configuration, seed) in a fresh spawned process, one
process at a time by default (``--workers``), with one BLAS thread and a
headless matplotlib. It skips every run whose record exists, so an
interrupted population resumes and a finished one extends to more seeds.
A run that raises inside BADS is an outcome: its record holds the exception.
A failure of the harness itself (before BADS is constructed) writes
``<label>_seed<seed>.error.txt`` instead, so the run is retried on resume.
``--only`` restricts the suite to some labels, ``--options`` merges a JSON
dict into every run's options, ``--budget-scale`` multiplies every
configuration's budget.

Each run writes ``<label>_seed<seed>.json``: the configuration, the seed,
the requested and the effective options, ``final`` (the returned point,
``fval``, ``fsd``, ``true_error = f_true(x) - f_min``, ``func_count``,
``iterations``, ``message``, ``wall_s``, ``crashed``, ``exception`` and
``min_noise_var``) and ``meta`` (the provenance: git state, versions, the
source and commit of the imported gpyreg, thread variables, start and end
times). ``min_noise_var`` is the smallest training noise variance ``sn2``
over the GPs of ``iteration_history["gp"]`` and their hyperparameter
samples: the quantity gpyreg compares with ``1e-6`` to choose its low-noise
representation of the posterior (``gaussian_process.py``, where
``__core_computation`` sets ``L_chol``).

``summary`` tabulates each configuration (median and interquartile range of
``true_error`` and ``func_count``, the fraction solved, the crash count) and
writes ``summary.md``. ``compare`` tests each configuration and metric
(``true_error``, ``func_count``) with a two-sample Kolmogorov-Smirnov test
where both sides have at least 3 runs, and ``true_error`` also with a
Wilcoxon signed-rank test on ``log10(true_error + 1e-12)`` paired by seed
(both populations share each seed's start point and noise stream); the
p-values of all tests form one Holm family at ``--alpha``. A configuration
whose crash count rises from zero is flagged too. It prints the effect
sizes (the median paired log10 error ratio with a bootstrap 95% interval,
the difference in fraction solved) and exits 1 on any flag. ``--split``
compares the even and the odd seeds of one population with the KS tests
alone: the null check.
"""

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# The package of this checkout, whichever checkout is installed: the records
# are labelled with this checkout's commit.
sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_targets as bt  # noqa: E402

METRICS = ("true_error", "func_count")
LOG_FLOOR = 1e-12  # added to true_error before log10
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# Options whose value after the run is recorded: those BADS rewrites for a
# noisy target, those a suite configuration sets, and the run's settings.
EFFECTIVE_OPTION_KEYS = (
    "max_fun_evals",
    "max_iter",
    "uncertainty_handling",
    "specify_target_noise",
    "noise_size",
    "noise_final_samples",
    "tol_fun",
    "tol_mesh",
    "tol_stall_iters",
    "n_train_min",
    "n_train_max",
    "random_seed",
    "display",
)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_info(cwd=REPO_ROOT):
    """Short commit and dirty flag (tracked files only) of a repository."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=cwd,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"sha": sha, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"sha": None, "dirty": None}


def module_source(name):
    """Where the imported package ``name`` loads from, and its commit.

    ``git`` is the commit and dirty state of the repository that tracks the
    package directory, so a checkout or a worktree placed on ``PYTHONPATH``
    is identified; an installed copy under site-packages reports ``git`` as
    None.
    """
    try:
        path = Path(importlib.import_module(name).__file__).resolve().parent
    except Exception:  # noqa: BLE001
        return None
    git = None
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "__init__.py"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        git = git_info(cwd=path)
    except Exception:  # noqa: BLE001
        pass
    return {"path": str(path), "git": git}


def thread_env():
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


def jsonable(v):
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------------
# One run (executed in a spawned process)
# --------------------------------------------------------------------------


def record_path(out_dir, label, seed):
    return Path(out_dir) / f"{label}_seed{seed}.json"


def error_path(out_dir, label, seed):
    return Path(out_dir) / f"{label}_seed{seed}.error.txt"


def min_noise_var(bads):
    """Smallest training noise variance over the recorded GPs.

    For each GP of ``iteration_history["gp"]`` (one per poll iteration;
    ``None`` slots skipped) and each of its hyperparameter samples, the
    noise function evaluated at the GP's training data, as gpyreg does
    before it chooses between a Cholesky factor (``min(sn2) >= 1e-6``) and
    the low-noise representation. ``None`` when no GP was recorded.
    """
    try:
        gps = bads.iteration_history["gp"]
    except Exception:  # noqa: BLE001
        return None
    if gps is None:
        return None
    best = np.inf
    for gp in gps:
        if gp is None or gp.posteriors is None or gp.X is None:
            continue
        cov_N = gp.covariance.hyperparameter_count(gp.X.shape[1])
        noise_N = gp.noise.hyperparameter_count()
        for post in gp.posteriors:
            if post is None:
                continue
            hyp = np.asarray(post.hyp, dtype=float).ravel()
            sn2 = gp.noise.compute(
                hyp[cov_N : cov_N + noise_N], gp.X, gp.y, gp.s2
            )
            best = min(best, float(np.min(sn2)))
    return None if best == np.inf else best


def _final(prob, bads, res, exc, wall):
    crashed = exc is not None
    out = {
        "x": None,
        "fval": None,
        "fsd": None,
        "true_error": None,
        "func_count": None,
        "iterations": None,
        "message": None,
        "wall_s": wall,
        "crashed": crashed,
        "exception": exc,
        "min_noise_var": None,
    }
    if res is not None:
        x = np.asarray(res["x"], dtype=float).ravel()
        out.update(
            x=x.tolist(),
            fval=float(np.asarray(res["fval"]).item()),
            fsd=float(np.asarray(res["fsd"]).item()),
            true_error=prob.f_true(x) - prob.f_min,
            func_count=int(res["func_count"]),
            iterations=int(res["iterations"]),
            message=str(res["message"]),
        )
    elif bads is not None:  # crashed during optimize(): what the run reached
        try:
            out["func_count"] = int(bads.function_logger.func_count)
            out["iterations"] = int(bads.optim_state["iter"])
        except Exception:  # noqa: BLE001
            pass
    if bads is not None:
        try:
            out["min_noise_var"] = min_noise_var(bads)
        except Exception:  # noqa: BLE001  (keep the run; the field stays None)
            pass
    return out


def _write_json(path, obj):
    """Write through a temporary file, so that an interrupted write never
    leaves a record that ``run`` would take as done."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def run_task(label, seed, extra_options, budget_scale, out_dir):
    """Run one (configuration, seed); write its record; return a row."""
    out_dir = Path(out_dir)
    started = _now()
    try:
        cfg = bt.find_config(label)
        prob = cfg.make(seed=seed, budget_scale=budget_scale)
        args, options = prob.bads_args()
        options.update(extra_options or {})
        requested = jsonable(options)
        from pybads import BADS
    except Exception:  # noqa: BLE001
        error_path(out_dir, label, seed).write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return {"label": label, "seed": seed, "status": "error"}

    bads = res = exc = None
    t0 = time.perf_counter()
    try:
        bads = BADS(*args, options=options)
        res = bads.optimize()
    except Exception as e:  # noqa: BLE001
        exc = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
    wall = time.perf_counter() - t0
    final = _final(prob, bads, res, exc, wall)
    effective = None
    if bads is not None:
        keys = list(EFFECTIVE_OPTION_KEYS) + [
            k for k in requested if k not in EFFECTIVE_OPTION_KEYS
        ]
        effective = {k: jsonable(bads.options.get(k)) for k in keys}
    record = {
        "label": label,
        "seed": seed,
        "problem": prob.name,
        "D": prob.D,
        "noise": prob.noise,
        "unbounded": cfg.unbounded,
        "x0": prob.x0.tolist(),
        "f_min": prob.f_min,
        "tolerance": prob.tolerance,
        "budget": cfg.budget,
        "budget_scale": budget_scale,
        "requested_options": requested,
        "effective_options": effective,
        "final": jsonable(final),
        "meta": {
            "git": git_info(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": pkg_version("scipy"),
            "pybads": pkg_version("pybads"),
            "pybads_source": module_source("pybads"),
            "gpyreg": pkg_version("gpyreg"),
            "gpyreg_source": module_source("gpyreg"),
            "threads": thread_env(),
            "started": started,
            "finished": _now(),
        },
    }
    _write_json(record_path(out_dir, label, seed), record)
    error_path(out_dir, label, seed).unlink(missing_ok=True)  # a retried run
    return {
        "label": label,
        "seed": seed,
        "status": "crash" if exc else "ok",
        "wall_s": wall,
        "true_error": final["true_error"],
        "func_count": final["func_count"],
        "exception": None
        if exc is None
        else f"{exc['type']}: {exc['message']}",
    }


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def parse_seeds(spec):
    """``"0-29"``, ``"0,3,5-7"`` -> sorted list of ints."""
    seeds = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return sorted(set(seeds))


def _progress(k, n, r, t0):
    tag = f"{r['label']}_seed{r['seed']}"
    elapsed = time.time() - t0
    eta = elapsed / k * (n - k)
    clock = f"[{elapsed / 60:.1f} min, eta {eta / 60:.1f} min]"
    if r["status"] == "error":
        return f"[population] {k}/{n} ERROR {tag:34s} (see .error.txt) {clock}"
    if r["status"] == "crash":
        return (
            f"[population] {k}/{n} CRASH {tag:34s} {r['wall_s']:6.1f} s"
            f"  {r['exception'][:80]}  {clock}"
        )
    return (
        f"[population] {k}/{n} ok    {tag:34s} {r['wall_s']:6.1f} s"
        f"  err={r['true_error']:.3g} evals={r['func_count']}  {clock}"
    )


def cmd_run(args):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    bt.single_thread_env()  # inherited by the spawned processes
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = bt.suite_configs(args.suite)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = sorted(wanted - {c.label for c in cfgs})
        if unknown:
            sys.exit(f"not in suite {args.suite!r}: {', '.join(unknown)}")
        cfgs = [c for c in cfgs if c.label in wanted]
    extra = json.loads(args.options) if args.options else {}
    seeds = parse_seeds(args.seeds)
    tasks = [
        (c.label, s)
        for c in cfgs
        for s in seeds
        if not record_path(out_dir, c.label, s).exists()
    ]
    n_done = len(cfgs) * len(seeds) - len(tasks)
    print(
        f"[population] {len(tasks)} runs ({n_done} already done),"
        f" {args.workers} worker(s) -> {out_dir}",
        flush=True,
    )
    if not tasks:
        return 0
    counts = {"ok": 0, "crash": 0, "error": 0}
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        max_tasks_per_child=1,
    ) as ex:
        futs = {
            ex.submit(
                run_task, label, seed, extra, args.budget_scale, str(out_dir)
            ): (label, seed)
            for label, seed in tasks
        }
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001  (the task or its process)
                label, seed = futs[fut]
                error_path(out_dir, label, seed).write_text(
                    f"{type(e).__name__}: {e}\n", encoding="utf-8"
                )
                r = {"label": label, "seed": seed, "status": "error"}
            counts[r["status"]] += 1
            print(_progress(k, len(tasks), r, t0), flush=True)
    print(
        f"[population] done: {counts['ok']} ok, {counts['crash']} crashed,"
        f" {counts['error']} harness errors,"
        f" {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return 1 if counts["error"] else 0


# --------------------------------------------------------------------------
# Loading, summary
# --------------------------------------------------------------------------


def load_population(d):
    """``{label: entry}`` from the records of a directory.

    Each entry holds ``seeds`` (int array), ``crashed`` (bool array), one
    float array per metric (NaN for a crashed run), ``tolerance``, the
    ``records`` themselves and ``errors``, the number of harness error files.
    """
    pop = {}
    for p in sorted(Path(d).glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        e = pop.setdefault(r["label"], {"records": [], "errors": 0})
        e["records"].append(r)
    for p in Path(d).glob("*.error.txt"):
        if p.with_name(p.name.replace(".error.txt", ".json")).exists():
            continue  # retried since
        label = p.name.rsplit("_seed", 1)[0]
        pop.setdefault(label, {"records": [], "errors": 0})["errors"] += 1
    for e in pop.values():
        recs = sorted(e["records"], key=lambda r: r["seed"])
        e["records"] = recs
        e["seeds"] = np.array([r["seed"] for r in recs], dtype=int)
        e["crashed"] = np.array(
            [bool(r["final"]["crashed"]) for r in recs], dtype=bool
        )
        for m in METRICS + ("wall_s", "min_noise_var"):
            e[m] = np.array(
                [
                    np.nan
                    if r["final"]["crashed"] or r["final"].get(m) is None
                    else r["final"][m]
                    for r in recs
                ],
                dtype=float,
            )
        e["tolerance"] = recs[0]["tolerance"] if recs else np.nan
    return pop


def solved_fraction(e):
    """Fraction of runs with ``true_error < tolerance``; a crash counts as
    unsolved."""
    n = len(e["seeds"])
    if n == 0:
        return np.nan
    err = e["true_error"]
    return float(np.sum(np.isfinite(err) & (err < e["tolerance"])) / n)


def _med_iqr(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "-"
    q = np.percentile(x, [25, 50, 75])
    return f"{q[1]:.3g} [{q[0]:.3g}, {q[2]:.3g}]"


def provenance(pop):
    """The distinct (commit, dirty, gpyreg source) combinations of a
    population's records, with counts: more than one means a mixed
    population."""
    combos = {}
    for e in pop.values():
        for r in e["records"]:
            m = r.get("meta", {})
            g = m.get("git") or {}
            src = m.get("gpyreg_source") or {}
            gg = src.get("git") or {}
            key = (
                f"pybads {g.get('sha')}{' (dirty)' if g.get('dirty') else ''},"
                f" gpyreg {m.get('gpyreg')} from {src.get('path')}"
                f" at {gg.get('sha')}{' (dirty)' if gg.get('dirty') else ''}"
            )
            combos[key] = combos.get(key, 0) + 1
    return combos


def summary_text(d):
    pop = load_population(d)
    lines = [f"# Population {Path(d).name}", ""]
    for key, n in provenance(pop).items():
        lines.append(f"- {n} runs: {key}")
    lines += [
        "",
        "| config | runs | crashed | true_error | func_count | solved |"
        " tolerance | wall s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in sorted(pop):
        e = pop[label]
        err = f" (+{e['errors']} harness errors)" if e["errors"] else ""
        lines.append(
            f"| {label} | {len(e['seeds'])}{err} | {int(e['crashed'].sum())}"
            f" | {_med_iqr(e['true_error'])} | {_med_iqr(e['func_count'])}"
            f" | {solved_fraction(e):.2f} | {e['tolerance']:g}"
            f" | {_med_iqr(e['wall_s'])} |"
        )
    lines += [
        "",
        "true_error, func_count and wall time: median [interquartile range]"
        " over the runs that did not crash; solved: fraction of all runs"
        " with true_error below the tolerance.",
    ]
    return "\n".join(lines)


def cmd_summary(args):
    text = summary_text(args.dir)
    print(text)
    (Path(args.dir) / "summary.md").write_text(text + "\n", encoding="utf-8")
    return 0


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def holm(pvals, alpha=0.05):
    """Holm's step-down procedure: ``(reject, adjusted p-values)``."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(
        1.0, np.maximum.accumulate((m - np.arange(m)) * p[order])
    )
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= alpha, adj


def _log_err(x):
    return np.log10(np.maximum(x, 0.0) + LOG_FLOOR)


def paired_log_ratios(a, b):
    """``log10(new) - log10(ref)`` of ``true_error`` over the seeds present
    and not crashed in both, in seed order."""
    common = np.intersect1d(a["seeds"], b["seeds"])
    ia = np.searchsorted(a["seeds"], common)
    ib = np.searchsorted(b["seeds"], common)
    xa, xb = a["true_error"][ia], b["true_error"][ib]
    ok = np.isfinite(xa) & np.isfinite(xb)
    return _log_err(xb[ok]) - _log_err(xa[ok])


def x0_mismatches(a, b):
    """Number of seeds present in both whose recorded start points differ:
    a pairing by seed assumes the same targets and streams on both sides."""
    xa = {r["seed"]: r.get("x0") for r in a["records"]}
    return sum(
        1
        for r in b["records"]
        if r["seed"] in xa and xa[r["seed"]] != r.get("x0")
    )


def wilcoxon_p(d):
    """Two-sided signed-rank p-value; 1 when every difference is zero."""
    from scipy import stats

    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = stats.wilcoxon(d)
    return float(r.statistic), float(r.pvalue)


def bootstrap_median_ci(d, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Median of ``d`` with a percentile bootstrap 95% interval."""
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(d)), float(lo), float(hi)


def _ks(x, y):
    """Two-sample KS test, without scipy's warning when it falls back from
    the exact p-value to the asymptotic one."""
    from scipy import stats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return stats.ks_2samp(x, y)


def ks_threshold(n1, n2, level):
    """Smallest two-sample KS statistic with p-value at most ``level`` for
    samples of sizes ``n1`` and ``n2`` (scanned on shifted grids)."""
    x = (np.arange(n1) + 0.5) / n1
    y0 = (np.arange(n2) + 0.5) / n2
    best = None
    for s in np.linspace(0.0, 1.0, 1001):
        r = _ks(x, y0 + s)
        if r.pvalue <= level and (best is None or r.statistic < best):
            best = r.statistic
    return best


def compare_populations(ref, new, alpha=0.05, paired=True, crash_flag=True):
    """Tests, effect sizes and flags of two populations.

    Returns ``(text, flagged)``, ``flagged`` being the set of flagged
    configuration labels.
    """
    labels = sorted(set(ref) & set(new))
    tests = []  # dicts: label, test, metric, n_ref, n_new, stat, p
    for label in labels:
        a, b = ref[label], new[label]
        for m in METRICS:
            x = a[m][np.isfinite(a[m])]
            y = b[m][np.isfinite(b[m])]
            if len(x) >= 3 and len(y) >= 3:
                ks = _ks(x, y)
                tests.append(
                    dict(
                        label=label,
                        test="KS",
                        metric=m,
                        n_ref=len(x),
                        n_new=len(y),
                        stat=float(ks.statistic),
                        p=float(ks.pvalue),
                    )
                )
        if paired:
            d = paired_log_ratios(a, b)
            if len(d) >= 3:
                w, p = wilcoxon_p(d)
                tests.append(
                    dict(
                        label=label,
                        test="signed-rank",
                        metric="log10 true_error",
                        n_ref=len(d),
                        n_new=len(d),
                        stat=w,
                        p=p,
                    )
                )
    reject, adj = holm([t["p"] for t in tests], alpha)
    flagged = {t["label"] for t, rj in zip(tests, reject) if rj}
    crash_rise = []
    if crash_flag:
        for label in labels:
            c_ref = int(ref[label]["crashed"].sum())
            c_new = int(new[label]["crashed"].sum())
            if c_ref == 0 and c_new > 0:
                crash_rise.append(label)
    flagged |= set(crash_rise)

    lines = [
        "| config | test | metric | n ref | n new | statistic | p |"
        " p Holm | |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, rj, pa in zip(tests, reject, adj):
        lines.append(
            f"| {t['label']} | {t['test']} | {t['metric']} | {t['n_ref']} |"
            f" {t['n_new']} | {t['stat']:.3g} | {t['p']:.3g} | {pa:.3g} |"
            f" {'FLAG' if rj else 'ok'} |"
        )
    lines += [
        "",
        "| config | pairs | median log10 error ratio (new/ref) [95% CI] |"
        " solved ref | solved new | change | crashed ref | crashed new |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in labels:
        a, b = ref[label], new[label]
        if paired:
            d = paired_log_ratios(a, b)
            med, lo, hi = bootstrap_median_ci(d)
            ratio = f"{med:+.3f} [{lo:+.3f}, {hi:+.3f}]" if len(d) else "-"
            n_pairs = len(d)
        else:
            ratio, n_pairs = "-", "-"
        s_ref, s_new = solved_fraction(a), solved_fraction(b)
        lines.append(
            f"| {label} | {n_pairs} | {ratio} | {s_ref:.2f} | {s_new:.2f} |"
            f" {s_new - s_ref:+.2f} | {int(a['crashed'].sum())} |"
            f" {int(b['crashed'].sum())} |"
        )
    only_ref = sorted(set(ref) - set(new))
    only_new = sorted(set(new) - set(ref))
    if only_ref or only_new:
        lines += ["", f"Only in REF: {only_ref}; only in NEW: {only_new}."]
    if paired:
        unpaired = {
            label: n
            for label in labels
            if (n := x0_mismatches(ref[label], new[label]))
        }
        if unpaired:
            lines += [
                "",
                "WARNING: seeds whose start points differ between REF and"
                f" NEW (the pairing does not hold): {unpaired}.",
            ]
    ks_sizes = [(t["n_ref"], t["n_new"]) for t in tests if t["test"] == "KS"]
    if ks_sizes:
        n1, n2 = max(set(ks_sizes), key=ks_sizes.count)
        level = alpha / len(tests)
        thr = ks_threshold(n1, n2, level)
        thr_txt = (
            "no KS statistic reaches it"
            if thr is None
            else f"that is a KS statistic of at least {thr:.3f}"
        )
        lines += [
            "",
            f"Holm family: {len(tests)} tests at alpha {alpha}. A flag needs"
            f" p <= {level:.2g} for the first step; for {n1} vs {n2} runs"
            f" {thr_txt}.",
        ]
    if crash_rise:
        lines += ["", f"Crash count rising from zero: {crash_rise}."]
    verdict = (
        f"{len(flagged)} configuration(s) flagged: {sorted(flagged)}"
        if flagged
        else f"no configuration flagged ({len(tests)} tests)"
    )
    lines += ["", f"**{verdict}**"]
    return "\n".join(lines), flagged


def split_population(pop):
    """Even and odd seeds of a population, as two populations."""
    even, odd = {}, {}
    for label, e in pop.items():
        for target, parity in ((even, 0), (odd, 1)):
            mask = e["seeds"] % 2 == parity
            sub = {
                k: (v[mask] if isinstance(v, np.ndarray) else v)
                for k, v in e.items()
            }
            sub["records"] = [r for r, keep in zip(e["records"], mask) if keep]
            target[label] = sub
    return even, odd


def cmd_compare(args):
    if args.split:
        pop = load_population(args.ref)
        ref, new = split_population(pop)
        title = f"# Null check: even vs odd seeds of {Path(args.ref).name}"
        prov = [("REF", pop)]
        text, flagged = compare_populations(
            ref, new, alpha=args.alpha, paired=False, crash_flag=False
        )
    else:
        ref, new = load_population(args.ref), load_population(args.new)
        title = (
            f"# Compare {Path(args.ref).name} (REF) and"
            f" {Path(args.new).name} (NEW)"
        )
        prov = [("REF", ref), ("NEW", new)]
        text, flagged = compare_populations(ref, new, alpha=args.alpha)
    head = [title, ""]
    for name, pop in prov:
        for key, n in provenance(pop).items():
            head.append(f"- {name}, {n} runs: {key}")
    print("\n".join(head) + "\n\n" + text, flush=True)
    return 1 if flagged else 0


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a population")
    r.add_argument("--suite", default="default")
    r.add_argument("--seeds", default="0-29", help='e.g. "0-29" or "0,2,5-9"')
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--only", default=None, help="comma-separated labels")
    r.add_argument("--options", default=None, help="JSON merged into runs")
    r.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="multiply every configuration's budget",
    )
    r.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel BADS processes (default 1: one heavy process)",
    )
    s = sub.add_parser("summary", help="tabulate a population")
    s.add_argument("dir", type=Path)
    c = sub.add_parser("compare", help="compare two populations")
    c.add_argument("ref", type=Path)
    c.add_argument("new", nargs="?", default=None, type=Path)
    c.add_argument(
        "--split", action="store_true", help="even vs odd seeds of REF"
    )
    c.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "summary":
        return cmd_summary(args)
    if args.split == (args.new is not None):
        ap.error("compare takes either NEW or --split")
    return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
