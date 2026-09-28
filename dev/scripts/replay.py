"""Exact step-by-step replay of short seeded PyBADS runs on one machine:
record a trace of each run, and compare two recordings step by step.

Sub-commands, from the repository root::

    python -u dev/scripts/replay.py record [--out DIR] [--configs a,b]
    python dev/scripts/replay.py check BASE NEW [--force]
    python dev/scripts/replay.py check DIR
    python dev/scripts/replay.py report DIR

``record`` runs each configuration (default: ``DEFAULT_CONFIGS`` at seed 0,
at 50 D evaluations, ``--budget-scale 0.1`` of the suite's 500 D) in a fresh
child process, built as ``population.py`` builds its runs, and writes one
trace per run under ``--out`` (default
``dev/scripts/runs/replay/<sha>_<time>/``): ``<label>_seed<seed>.npz`` and
its ``.json`` sidecar. ``--repeat N`` records each run N times in the same
process, the later ones as ``..._rep1`` and on. The default set takes about
40 s. Before starting the children it pins the numerical platform: one BLAS
thread (``--threads``) and, on x86_64, OpenBLAS's Haswell kernels
(``OPENBLAS_CORETYPE``, ``--coretype``), so that the kernels do not follow
the CPU's own detection; ``--no-pin`` leaves the environment as the shell
set it. ``--options`` merges a JSON dict into every run's options.

A trace captures, with no change to the package:

- every call of the target: the point (original space), the value, the SD
  that the target returns (NaN when none), the stage (``init``: the start,
  the noise test and the initial design; ``search``; ``poll``;
  ``between``: outside a step, the final samples of a noisy run), the
  iteration (``optim_state["iter"]``, -1 in the initialization), and a
  digest of the state of the run's generator ``bads.rng`` at the call;
- every search and poll step (``BADS._search_step_``, ``_poll_step_`` and
  ``_update_search_stats_`` wrapped on the instance): its iteration, the
  evaluations before and after it, the search method that the hedge chose,
  the outcome (a search's status, ``empty`` for an empty search set; a
  poll's ``success``, ``moved`` or ``refine``), ``mesh_size_integer`` after
  it and the generator's digest;
- every GP computation that ``bads.py`` calls (``init_and_train_gp``,
  ``local_gp_fitting`` and ``add_and_update_gp``, wrapped in the namespace
  of ``pybads.bads.bads``): its stage (``between``: the re-evaluation of
  a noisy run's history), iteration, evaluations so far, training-set
  size, refit flag, exit flag, the hyperparameters it leaves (every sample)
  and the generator's digest;
- ``iteration_history`` after the run (``u``, ``x``, ``yval``, ``fval``,
  ``fsd``, the mesh sizes, ``func_count`` and ``gp_hyp_full``), and the
  whole ``OptimizeResult`` (its timings and version recorded apart, and not
  compared);
- the platform key (OS, libc, machine, CPU model, Python, NumPy's BLAS
  build, NumPy and SciPy versions, each loaded OpenBLAS's kernel and thread
  count, ``OPENBLAS_CORETYPE`` and the thread variables) and the
  provenance (the git commit and dirty flag of the checkout that holds this
  script, the versions and source paths of NumPy, SciPy, gpyreg and PyBADS,
  gpyreg's commit, the pinning).

A private name that the tool wraps or reads and that the package no longer
has stops the recording with its name.

``check BASE NEW`` pairs the runs of two recordings by name; ``check DIR``
pairs each repeat of DIR with the first recording of its run. It refuses
(exit status 2) traces whose platform keys differ, unless ``--force``, and
reports, per run, exact (bitwise) identity per stream (``evals``,
``steps``, ``fits``, ``history``, ``result``) and
otherwise the first divergence: the evaluation, its iteration and stage on
each side, ``|dx|`` and ``|dy|`` there, whether the generator's states
agree there (they agree when the same draws came before, so that a value
moved; they differ when a draw was added or skipped, a changed branch), the
first step that differs, the earliest GP computation whose hyperparameters
differ, and, as diagnostics only, the horizons: the leading evaluations
whose points and values agree to 1e-12 and to 1e-8. Exit status 0 when
every run is identical, 1 otherwise.

``report DIR`` tabulates the runs of one recording and prints its platform
key and provenance.

The comparison is exact, so it holds only for one machine, one set of
versions, one BLAS kernel and one thread count: another kernel or another
number of threads rounds a few operations differently, and a run parts
within a few dozen evaluations, into different decisions. To check a
change that must move nothing, record at the parent commit with the
``replay.py`` of a worktree at it (``benchmark_targets.py`` puts the
checkout that holds it first on ``sys.path``), record at the change, and
``check`` the two. ``--repeat 2`` and ``check DIR`` find the first
computation at which two runs of one process differ.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# benchmark_targets puts this checkout first on sys.path
import benchmark_targets as bt  # noqa: E402
from population import (  # noqa: E402
    git_info,
    jsonable,
    module_source,
    parse_seeds,
    pkg_version,
)

DEFAULT_CONFIGS = (
    "sphere_D2",
    "ellipsoid_D3",
    "rosenbrock_D6",
    "sphere_D3_homo",
    "sphere_D3_hetero",
    "sphere_nonbox_D3",
    "ellipsoid_D3_unbounded",
    "logsphere_D3",
)
DEFAULT_BUDGET_SCALE = 0.1  # of the suites' 500 D: 50 D evaluations
DEFAULT_RUNS = REPO_ROOT / "dev" / "scripts" / "runs" / "replay"
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
PINNED_CORETYPE = "Haswell"  # the x86_64 OpenBLAS kernels of the recordings
HORIZON_TOLS = (1e-12, 1e-8)
STREAMS = ("evals", "steps", "fits", "history", "result")
PREFIX = {
    "evals": "eval_",
    "steps": "step_",
    "fits": "fit_",
    "history": "hist_",
    "result": "res_",
}
HISTORY_KEYS = (
    "u",
    "x",
    "yval",
    "fval",
    "fsd",
    "mesh_size",
    "search_mesh_size",
    "func_count",
)
RESULT_ARRAYS = ("x", "yval_vec", "ysd_vec")
RESULT_SCALARS = (
    "fval",
    "fsd",
    "func_count",
    "iterations",
    "mesh_size",
    "success",
    "status",
    "message",
    "target_type",
    "problem_type",
    "random_seed",
    "algorithm",
)
RESULT_UNCOMPARED = ("total_time", "overhead", "version")

# The private names of the package that a recording wraps or reads; the
# steps also read `poll_moved` and `search_es_hedge`, which `optimize()`
# sets
BADS_METHODS = ("_search_step_", "_poll_step_", "_update_search_stats_")
MODULE_FUNCTIONS = (
    "init_and_train_gp",
    "local_gp_fitting",
    "add_and_update_gp",
)
BADS_ATTRIBUTES = (
    "rng",
    "optim_state",
    "function_logger",
    "mesh_size_integer",
    "iteration_history",
)


class MissingName(RuntimeError):
    """A private name that the recording needs is missing from PyBADS."""


# --------------------------------------------------------------------------
# Platform and provenance
# --------------------------------------------------------------------------


def _cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            keys = ("model name", "Hardware", "CPU part")
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                key, _, value = line.partition(":")
                if key.strip() in keys:
                    return value.strip()
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or None


def _blas_build():
    """NumPy's BLAS and LAPACK as built (name, version, configuration)."""
    try:
        cfg = np.show_config(mode="dicts")["Build Dependencies"]
    except Exception:  # noqa: BLE001
        return None
    return {
        k: {
            f: cfg[k].get(f)
            for f in ("name", "version", "openblas configuration")
        }
        for k in ("blas", "lapack")
        if k in cfg
    }


def _openblas_libraries():
    """The OpenBLAS libraries that the process has loaded."""
    paths = set()
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text().splitlines():
            p = line.split()[-1] if line.split() else ""
            if "openblas" in Path(p).name.lower():
                paths.add(p)
        return sorted(paths)
    for mod in ("numpy", "scipy"):  # no /proc: the bundled libraries
        root = Path(__import__(mod).__file__).resolve().parent
        for d in (root.parent / f"{mod}.libs", root / ".dylibs", root):
            if d.is_dir():
                paths.update(
                    str(p) for p in d.glob("*openblas*") if p.is_file()
                )
    return sorted(paths)


def _openblas_runtime():
    """Kernel (core name) and thread count of each loaded OpenBLAS."""
    import ctypes

    import scipy.linalg  # noqa: F401  (loads SciPy's own OpenBLAS)

    out = []
    for path in _openblas_libraries():
        entry = {"library": Path(path).name, "corename": None, "threads": None}
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            out.append(entry)
            continue
        for field, func, restype in (
            ("corename", "get_corename", ctypes.c_char_p),
            ("threads", "get_num_threads", ctypes.c_int),
        ):
            for prefix in ("scipy_openblas_", "openblas_"):
                for suffix in ("64_", "", "_64"):
                    f = getattr(lib, prefix + func + suffix, None)
                    if f is None:
                        continue
                    f.restype = restype
                    value = f()
                    entry[field] = (
                        value.decode() if isinstance(value, bytes) else value
                    )
                    break
                if entry[field] is not None:
                    break
        out.append(entry)
    return out


def platform_key():
    """What must be the same for two runs of one commit to repeat exactly;
    ``check`` refuses traces whose keys differ."""
    import scipy

    return {
        "os": platform.system(),
        "libc": " ".join(platform.libc_ver()).strip() or None,
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_build": _blas_build(),
        "openblas_runtime": _openblas_runtime(),
        "env": {
            k: os.environ.get(k) for k in ("OPENBLAS_CORETYPE",) + THREAD_VARS
        },
    }


def provenance():
    """What identifies the code that ran; reported, never refused."""
    import scipy

    return {
        "git": git_info(),
        "checkout": str(REPO_ROOT),
        "pybads": pkg_version("pybads"),
        "pybads_source": module_source("pybads"),
        "gpyreg": pkg_version("gpyreg"),
        "gpyreg_source": module_source("gpyreg"),
        "numpy_path": str(Path(np.__file__).resolve().parent),
        "scipy_path": str(Path(scipy.__file__).resolve().parent),
        "python_executable": sys.executable,
        "kernel": platform.release(),
    }


# --------------------------------------------------------------------------
# Recording (in the child process)
# --------------------------------------------------------------------------


def _require(obj, name, where):
    if not hasattr(obj, name):
        raise MissingName(f"replay: {where} has no {name!r}")
    return getattr(obj, name)


class _bookkeeping:
    """Turn a key or an attribute that the recording reads and that the
    package no longer has into ``MissingName``, so that it is not taken
    for a crash of the run."""

    def __init__(self, where):
        self.where = where

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and issubclass(
            exc_type, (KeyError, AttributeError)
        ):
            raise MissingName(f"replay: {self.where}: {exc!r}") from exc
        return False


def _float(v):
    return float("nan") if v is None else float(np.asarray(v).item())


def _hyp(gp):
    """The GP's hyperparameters, every sample, as a 2-D array."""
    try:
        return np.atleast_2d(
            np.asarray(gp.get_hyperparameters(as_array=True), dtype=float)
        )
    except Exception:  # noqa: BLE001  (a GP without hyperparameters)
        return np.zeros((0, 0))


class Recorder:
    """What one run does, captured from its target and from the package's
    functions, wrapped for the run's duration."""

    def __init__(self):
        self.bads = None
        self.stage = "init"
        self.evals = []
        self.steps = []
        self.fits = []
        self._search_status = None

    # state of the run ------------------------------------------------------

    def rng_digest(self):
        rng = getattr(self.bads, "rng", None)
        if rng is None:
            return ""
        state = json.dumps(
            rng.bit_generator.state, sort_keys=True, default=str
        )
        return hashlib.blake2b(state.encode(), digest_size=8).hexdigest()

    def iteration(self):
        if self.bads is None:
            return -1
        return int(_require(self.bads, "optim_state", "BADS")["iter"])

    def func_count(self):
        if self.bads is None:
            return 0
        return int(self.bads.function_logger.func_count)

    # the target ------------------------------------------------------------

    def wrap_target(self, fun):
        def target(x):
            out = fun(x)
            y, sd = out if isinstance(out, tuple) else (out, None)
            self.evals.append(
                (
                    np.array(x, dtype=float).ravel(),
                    _float(y),
                    _float(sd),
                    self.stage,
                    self.iteration(),
                    self.rng_digest(),
                )
            )
            return out

        return target

    # the steps -------------------------------------------------------------

    def attach(self, bads):
        """Wrap the steps on the instance ``bads``."""
        self.bads = bads
        for name in BADS_ATTRIBUTES:
            _require(bads, name, "BADS")
        search = _require(bads, "_search_step_", "BADS")
        poll = _require(bads, "_poll_step_", "BADS")
        stats = _require(bads, "_update_search_stats_", "BADS")

        def search_step(gp):
            fc = self.func_count()
            self._search_status = None
            self.stage = "search"
            try:
                out = search(gp)
            finally:
                self.stage = "between"
            with _bookkeeping("the search step"):
                hedge = getattr(bads, "search_es_hedge", None)
                method = getattr(hedge, "chosen_search_fun", None)
                outcome = (
                    "empty" if out[0] is None else str(self._search_status)
                )
                self._step(
                    "search", fc, str(method[0]) if method else "", outcome
                )
            return out

        def poll_step(gp):
            with _bookkeeping("the poll step"):
                fc = self.func_count()
                n_success = len(bads.optim_state["u_success"])
            self.stage = "poll"
            try:
                out = poll(gp)
            finally:
                self.stage = "between"
            with _bookkeeping("the poll step"):
                if len(bads.optim_state["u_success"]) > n_success:
                    outcome = "success"
                else:
                    outcome = "moved" if bads.poll_moved else "refine"
                self._step("poll", fc, "", outcome)
            return out

        def update_search_stats(search_status, search_dist):
            self._search_status = search_status
            return stats(search_status, search_dist)

        bads._search_step_ = search_step
        bads._poll_step_ = poll_step
        bads._update_search_stats_ = update_search_stats

    def _step(self, kind, fc_before, method, outcome):
        self.steps.append(
            (
                kind,
                self.iteration(),
                fc_before,
                self.func_count(),
                method,
                outcome,
                int(self.bads.mesh_size_integer),
                self.rng_digest(),
            )
        )

    # the GP ----------------------------------------------------------------

    def wrap_gp_functions(self, module):
        """Wrap the GP functions in ``module``'s namespace; return a
        function that restores them."""
        originals = {
            name: _require(module, name, module.__name__)
            for name in MODULE_FUNCTIONS
        }

        def wrap(name, fun):
            def wrapped(*args, **kwargs):
                out = fun(*args, **kwargs)
                gp = out[0] if isinstance(out, tuple) else out
                refit = -1
                exit_flag = float("nan")
                if name == "local_gp_fitting":
                    flag = (
                        kwargs["refit_flag"]
                        if "refit_flag" in kwargs
                        else args[6]
                    )
                    refit = int(bool(flag))
                    exit_flag = float(out[1])
                self._fit(name, gp, refit, exit_flag)
                return out

            return wrapped

        for name, fun in originals.items():
            setattr(module, name, wrap(name, fun))

        def restore():
            for name, fun in originals.items():
                setattr(module, name, fun)

        return restore

    def _fit(self, kind, gp, refit, exit_flag):
        X = getattr(gp, "X", None)
        tmp = getattr(gp, "temporary_data", {}) or {}
        self.fits.append(
            (
                kind,
                self.stage,
                self.iteration(),
                self.func_count(),
                -1 if X is None else int(np.shape(X)[0]),
                refit,
                exit_flag,
                int(bool(tmp.get("needs_rebuild", False))),
                _hyp(gp),
                self.rng_digest(),
            )
        )

    # arrays ----------------------------------------------------------------

    def arrays(self, D):
        a = {}
        ev = self.evals
        a["eval_x"] = np.array([e[0] for e in ev], dtype=float).reshape(-1, D)
        a["eval_y"] = np.array([e[1] for e in ev], dtype=float)
        a["eval_sd"] = np.array([e[2] for e in ev], dtype=float)
        a["eval_stage"] = np.array([e[3] for e in ev], dtype="U8")
        a["eval_iter"] = np.array([e[4] for e in ev], dtype=np.int64)
        a["eval_rng"] = np.array([e[5] for e in ev], dtype="U16")
        st = self.steps
        for i, (key, dtype) in enumerate(
            (
                ("kind", "U8"),
                ("iter", np.int64),
                ("fc_before", np.int64),
                ("fc_after", np.int64),
                ("method", "U16"),
                ("outcome", "U16"),
                ("mesh_int", np.int64),
                ("rng", "U16"),
            )
        ):
            a[f"step_{key}"] = np.array([s[i] for s in st], dtype=dtype)
        fi = self.fits
        for i, (key, dtype) in enumerate(
            (
                ("kind", "U20"),
                ("stage", "U8"),
                ("iter", np.int64),
                ("fc", np.int64),
                ("n_train", np.int64),
                ("refit", np.int64),
                ("exit_flag", float),
                ("needs_rebuild", np.int64),
            )
        ):
            a[f"fit_{key}"] = np.array([f[i] for f in fi], dtype=dtype)
        a.update(_ragged("fit_hyp", [f[8] for f in fi]))
        a["fit_rng"] = np.array([f[9] for f in fi], dtype="U16")
        return a


def _ragged(prefix, blocks):
    """A list of 2-D arrays as one flat array, the offsets and the shapes."""
    blocks = [np.atleast_2d(np.asarray(b, dtype=float)) for b in blocks]
    sizes = [b.size for b in blocks]
    return {
        f"{prefix}_flat": (
            np.concatenate([b.ravel() for b in blocks])
            if blocks
            else np.zeros(0)
        ),
        f"{prefix}_off": np.concatenate(([0], np.cumsum(sizes))).astype(
            np.int64
        ),
        f"{prefix}_shape": np.array(
            [b.shape for b in blocks], dtype=np.int64
        ).reshape(-1, 2),
    }


def _block(arrays, prefix, i):
    off = arrays[f"{prefix}_off"]
    shape = arrays[f"{prefix}_shape"][i]
    return arrays[f"{prefix}_flat"][off[i] : off[i + 1]].reshape(shape)


def _history_arrays(bads, D):
    ih = bads.iteration_history
    a = {}
    for key in HISTORY_KEYS:
        v = ih.get(key)
        rows = [] if v is None else list(v)
        width = D if key in ("u", "x") else 1
        a[f"hist_{key}"] = np.array(
            [
                np.full(width, np.nan)
                if r is None
                else np.asarray(r, dtype=float).ravel()
                for r in rows
            ],
            dtype=float,
        ).reshape(-1, width)
    hyp = ih.get("gp_hyp_full")
    a.update(
        _ragged(
            "hist_hyp",
            [
                np.zeros((0, 0)) if h is None else h
                for h in ([] if hyp is None else list(hyp))
            ],
        )
    )
    return a


def _result_parts(res):
    arrays, scalars, uncompared = {}, {}, {}
    if res is None:
        return arrays, None, None
    for key in RESULT_ARRAYS:
        v = res.get(key)
        arrays[f"res_{key}"] = (
            np.zeros(0) if v is None else np.asarray(v, dtype=float).ravel()
        )
        scalars[f"{key}_is_none"] = v is None
    for key in RESULT_SCALARS:
        scalars[key] = jsonable(res.get(key))
    for key in RESULT_UNCOMPARED:
        uncompared[key] = jsonable(res.get(key))
    return arrays, scalars, uncompared


def record_run(label, seed, budget_scale, extra_options=None):
    """Run one configuration with every capture attached; return the
    trace's arrays and sidecar."""
    import pybads.bads.bads as bads_module
    from pybads import BADS

    for name in BADS_METHODS:
        _require(BADS, name, "pybads.bads.bads.BADS")
    cfg = bt.find_config(label)
    prob = cfg.make(seed=seed, budget_scale=budget_scale)
    args, options = prob.bads_args()
    options.update(extra_options or {})
    requested = jsonable(options)
    rec = Recorder()
    args = (rec.wrap_target(args[0]),) + tuple(args[1:])
    restore = rec.wrap_gp_functions(bads_module)
    bads = res = crash = None
    t0 = time.perf_counter()
    try:
        bads = BADS(*args, options=options)
        rec.attach(bads)
        res = bads.optimize()
    except MissingName:
        raise
    except Exception as e:  # noqa: BLE001  (a crash is an outcome)
        crash = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        restore()
    wall = time.perf_counter() - t0
    arrays = rec.arrays(prob.D)
    if bads is not None:
        arrays.update(_history_arrays(bads, prob.D))
    res_arrays, res_scalars, res_uncompared = _result_parts(res)
    arrays.update(res_arrays)
    sidecar = {
        "label": label,
        "seed": seed,
        "budget_scale": budget_scale,
        "D": prob.D,
        "requested_options": requested,
        "result": res_scalars,
        "result_uncompared": res_uncompared,
        "crash": crash,
        "counts": {
            "evals": len(rec.evals),
            "steps": len(rec.steps),
            "fits": len(rec.fits),
            "iterations": None if res is None else int(res["iterations"]),
        },
        "wall_s": wall,
    }
    return arrays, sidecar


def trace_name(label, seed, rep=0):
    return f"{label}_seed{seed}" + (f"_rep{rep}" if rep else "")


def write_trace(out_dir, name, arrays, sidecar):
    out_dir = Path(out_dir)
    np.savez_compressed(out_dir / f"{name}.npz", **arrays)
    tmp = out_dir / f"{name}.json.tmp"
    tmp.write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    os.replace(tmp, out_dir / f"{name}.json")


def _cmd_child(args):
    """Record one configuration ``--repeat`` times in this process."""
    extra = json.loads(args.options) if args.options else {}
    key = platform_key()
    prov = provenance()
    for rep in range(args.repeat):
        name = trace_name(args.label, args.seed, rep)
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            arrays, sidecar = record_run(
                args.label, args.seed, args.budget_scale, extra
            )
        except MissingName as e:
            print(str(e), flush=True)
            return 3
        sidecar.update(
            name=name,
            repeat=rep,
            platform=key,
            provenance=dict(prov, started=started, pin=json.loads(args.pin)),
        )
        write_trace(args.out, name, arrays, sidecar)
        c = sidecar["counts"]
        status = (
            "CRASH " + sidecar["crash"]["type"] if sidecar["crash"] else ""
        )
        print(
            f"[replay] {name:34s} {c['evals']:4d} evaluations,"
            f" {c['steps']:3d} steps, {c['fits']:4d} GP computations,"
            f" {sidecar['wall_s']:5.1f} s {status}",
            flush=True,
        )
    return 0


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------


def pinned_env(threads, coretype, no_pin):
    """The environment of the children, and what was pinned."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    if no_pin:
        return env, {"threads": None, "coretype": None, "no_pin": True}
    for k in THREAD_VARS:
        env[k] = str(threads)
    if coretype is None and platform.machine().lower() in (
        "x86_64",
        "amd64",
    ):
        coretype = PINNED_CORETYPE
    if coretype:
        env["OPENBLAS_CORETYPE"] = coretype
    else:
        env.pop("OPENBLAS_CORETYPE", None)
    return env, {"threads": threads, "coretype": coretype, "no_pin": False}


def default_out_dir():
    g = git_info()
    tag = (g["sha"] or "nogit") + ("-dirty" if g["dirty"] else "")
    return DEFAULT_RUNS / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}"


def cmd_record(args):
    labels = [s.strip() for s in args.configs.split(",") if s.strip()]
    for label in labels:
        bt.find_config(label)  # an unknown label fails before any run
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out) if args.out else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    taken = [
        trace_name(label, seed, rep)
        for label in labels
        for seed in seeds
        for rep in range(args.repeat)
        if (out_dir / f"{trace_name(label, seed, rep)}.json").exists()
    ]
    if taken:
        sys.exit(
            f"replay: {out_dir} already holds {', '.join(taken)};"
            " record into a new directory"
        )
    env, pin = pinned_env(args.threads, args.coretype, args.no_pin)
    print(
        f"[replay] {len(labels) * len(seeds)} runs x {args.repeat}"
        f" -> {out_dir}  (pin: {pin})",
        flush=True,
    )
    t0 = time.time()
    failed = []
    for label in labels:
        for seed in seeds:
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "_record",
                "--label",
                label,
                "--seed",
                str(seed),
                "--budget-scale",
                repr(args.budget_scale),
                "--repeat",
                str(args.repeat),
                "--out",
                str(out_dir),
                "--pin",
                json.dumps(pin),
            ]
            if args.options:
                cmd += ["--options", args.options]
            if subprocess.run(cmd, env=env).returncode != 0:
                failed.append(trace_name(label, seed))
    print(f"[replay] done in {time.time() - t0:.0f} s -> {out_dir}")
    if failed:
        print(f"[replay] FAILED: {', '.join(failed)}")
        return 1
    return 0


# --------------------------------------------------------------------------
# Loading and comparing
# --------------------------------------------------------------------------


class Trace:
    def __init__(self, json_path):
        json_path = Path(json_path)
        self.path = json_path
        self.meta = json.loads(json_path.read_text(encoding="utf-8"))
        self.name = self.meta.get("name", json_path.stem)
        with np.load(json_path.with_suffix(".npz")) as z:
            self.arrays = {k: z[k] for k in z.files}

    def stream(self, stream):
        p = PREFIX[stream]
        return {k: v for k, v in self.arrays.items() if k.startswith(p)}

    @property
    def n_evals(self):
        return len(self.arrays.get("eval_y", ()))


def load_dir(d):
    """The traces of a recording, by name."""
    return {
        t.name: t for t in (Trace(p) for p in sorted(Path(d).glob("*.json")))
    }


def _same(a, b):
    """Bitwise identity of two arrays (shape, and every value, NaN equal to
    NaN, 0.0 different from -0.0)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.all(~_differs(a, b)))
    return bool(np.array_equal(a, b))


def _differs(a, b):
    """Elementwise: the float arrays ``a`` and ``b`` (same shape) differ."""
    both_nan = np.isnan(a) & np.isnan(b)
    same = (a == b) & (np.signbit(a) == np.signbit(b))
    return ~(same | both_nan)


def _stream_same(base, new, stream):
    a, b = base.stream(stream), new.stream(stream)
    if set(a) != set(b):
        return False
    if not all(_same(a[k], b[k]) for k in a):
        return False
    if stream == "result":
        return base.meta.get("result") == new.meta.get("result") and (
            (base.meta.get("crash") or {}).get("message")
            == (new.meta.get("crash") or {}).get("message")
        )
    return True


def _row_differs(a, b, n):
    """Per row of the first ``n``: does row i of ``a`` differ from ``b``'s."""
    a, b = np.asarray(a)[:n], np.asarray(b)[:n]
    if a.dtype.kind == "f":
        d = _differs(a, b)
    else:
        d = a != b
    return d.reshape(n, -1).any(axis=1) if n else np.zeros(0, bool)


def first_eval_divergence(base, new):
    """The first evaluation at which the two runs differ, or None."""
    A, B = base.arrays, new.arrays
    na, nb = base.n_evals, new.n_evals
    n = min(na, nb)
    mask = np.zeros(n, dtype=bool)
    for key in ("eval_x", "eval_y", "eval_sd", "eval_stage", "eval_iter"):
        mask |= _row_differs(A[key], B[key], n)
    rng_differs = _row_differs(A["eval_rng"], B["eval_rng"], n)
    bad = np.flatnonzero(mask)
    if len(bad) == 0:
        if na == nb:
            k_rng = np.flatnonzero(rng_differs)
            if len(k_rng) == 0:
                return None
            k = int(k_rng[0])  # same points and values, other draws
        else:
            return {"k": n, "short": True, "n_base": na, "n_new": nb}
    else:
        k = int(bad[0])
    return {
        "k": k,
        "short": False,
        "n_base": na,
        "n_new": nb,
        "iter": (int(A["eval_iter"][k]), int(B["eval_iter"][k])),
        "stage": (str(A["eval_stage"][k]), str(B["eval_stage"][k])),
        "dx": float(np.max(np.abs(A["eval_x"][k] - B["eval_x"][k]))),
        "dy": float(abs(A["eval_y"][k] - B["eval_y"][k])),
        "rng_agree": bool(A["eval_rng"][k] == B["eval_rng"][k]),
        "rng_agree_before": bool(k == 0 or not rng_differs[k - 1]),
    }


def horizon(base, new, tol):
    """The leading evaluations whose points and values agree to ``tol``
    (relative, with ``tol`` as an absolute floor)."""
    A, B = base.arrays, new.arrays
    n = min(base.n_evals, new.n_evals)
    ok = np.isclose(
        A["eval_x"][:n], B["eval_x"][:n], rtol=tol, atol=tol, equal_nan=True
    ).all(axis=1) & np.isclose(
        A["eval_y"][:n], B["eval_y"][:n], rtol=tol, atol=tol, equal_nan=True
    )
    bad = np.flatnonzero(~ok)
    return int(bad[0]) if len(bad) else n


def first_step_divergence(base, new):
    A, B = base.arrays, new.arrays
    na, nb = len(A["step_kind"]), len(B["step_kind"])
    n = min(na, nb)
    keys = ("kind", "iter", "fc_before", "fc_after", "method", "outcome")
    keys += ("mesh_int", "rng")
    for i in range(n):
        diff = [k for k in keys if A[f"step_{k}"][i] != B[f"step_{k}"][i]]
        if diff:
            return {
                "i": i,
                "fields": diff,
                "base": {k: A[f"step_{k}"][i].item() for k in keys},
                "new": {k: B[f"step_{k}"][i].item() for k in keys},
            }
    if na != nb:
        return {"i": n, "fields": ["count"], "n_base": na, "n_new": nb}
    return None


def first_fit_divergence(base, new, hyp_only=True):
    """The earliest GP computation whose hyperparameters differ (or, with
    ``hyp_only=False``, in which anything recorded differs)."""
    A, B = base.arrays, new.arrays
    na, nb = len(A["fit_kind"]), len(B["fit_kind"])
    keys = ("kind", "stage", "iter", "fc", "n_train", "refit", "needs_rebuild")
    for i in range(min(na, nb)):
        ha, hb = _block(A, "fit_hyp", i), _block(B, "fit_hyp", i)
        hyp_diff = not _same(ha, hb)
        other = [k for k in keys if A[f"fit_{k}"][i] != B[f"fit_{k}"][i]]
        if hyp_diff or (not hyp_only and other):
            out = {
                "i": i,
                "hyp_differ": hyp_diff,
                "fields": other,
                "base": {k: A[f"fit_{k}"][i].item() for k in keys},
                "new": {k: B[f"fit_{k}"][i].item() for k in keys},
                "dhyp": None,
            }
            if hyp_diff and ha.shape == hb.shape and ha.size:
                out["dhyp"] = float(np.nanmax(np.abs(ha - hb)))
            return out
    if na != nb:
        return {"i": min(na, nb), "count": (na, nb)}
    return None


def compare_traces(base, new):
    """Everything ``check`` reports about one pair of runs."""
    streams = {s: _stream_same(base, new, s) for s in STREAMS}
    out = {
        "name": new.name,
        "identical": all(streams.values()),
        "streams": streams,
        "n_evals": (base.n_evals, new.n_evals),
        "n_steps": (
            len(base.arrays["step_kind"]),
            len(new.arrays["step_kind"]),
        ),
        "n_fits": (len(base.arrays["fit_kind"]), len(new.arrays["fit_kind"])),
        "crash": (
            (base.meta.get("crash") or {}).get("type"),
            (new.meta.get("crash") or {}).get("type"),
        ),
    }
    if out["identical"]:
        return out
    out["eval"] = first_eval_divergence(base, new)
    out["step"] = first_step_divergence(base, new)
    out["fit_hyp"] = first_fit_divergence(base, new)
    out["fit_any"] = first_fit_divergence(base, new, hyp_only=False)
    out["horizons"] = {tol: horizon(base, new, tol) for tol in HORIZON_TOLS}
    rb, rn = base.meta.get("result") or {}, new.meta.get("result") or {}
    out["final"] = {
        k: (rb.get(k), rn.get(k)) for k in ("fval", "func_count", "iterations")
    }
    return out


def _it(i):
    """An iteration as BADS displays it (from 1; 0 is the initialization)."""
    return i + 1


def format_comparison(c):
    lines = []
    name = c["name"]
    if c["identical"]:
        ne, ns, nf = c["n_evals"][0], c["n_steps"][0], c["n_fits"][0]
        return [
            f"{name:34s} identical  ({ne} evaluations, {ns} steps,"
            f" {nf} GP computations)"
        ]
    pad = " " * 36
    e = c["eval"]
    if e is None:
        lines.append(f"{name:34s} DIFFERS, with identical evaluations")
    elif e["short"]:
        lines.append(
            f"{name:34s} DIFFERS: the first {e['k']} evaluations are"
            f" identical; base has {e['n_base']}, new {e['n_new']}"
        )
    else:
        stage = e["stage"][0]
        if e["stage"][1] != stage:
            stage += f" (new: {e['stage'][1]})"
        it = f"iteration {_it(e['iter'][0])}"
        if e["iter"][1] != e["iter"][0]:
            it += f" (new: {_it(e['iter'][1])})"
        rng = (
            "generator states agree"
            if e["rng_agree"]
            else "generator states differ"
            + (
                ""
                if e["rng_agree_before"]
                else " (already at the previous evaluation)"
            )
        )
        lines.append(
            f"{name:34s} PARTED at evaluation {e['k']} ({it}, {stage}):"
            f" |dx| {e['dx']:.3g}, |dy| {e['dy']:.3g}; {rng}"
        )
    f = c["fit_hyp"]
    if f is None:
        lines.append(
            pad + "GP hyperparameters: identical in every computation"
        )
    elif "count" in f:
        lines.append(
            pad + f"GP hyperparameters: identical in the first {f['i']}"
            f" computations; base has {f['count'][0]}, new {f['count'][1]}"
        )
    else:
        b = f["base"]
        dh = "" if f["dhyp"] is None else f", max |dhyp| {f['dhyp']:.3g}"
        ntr = f"{b['n_train']}"
        if f["new"]["n_train"] != b["n_train"]:
            ntr += f" (new: {f['new']['n_train']})"
        lines.append(
            pad + f"first GP hyperparameters that differ: computation"
            f" {f['i']}, {b['kind']} ({b['stage']}, iteration"
            f" {_it(b['iter'])}, after {b['fc']} evaluations,"
            f" {ntr} training points){dh}"
        )
    fa = c["fit_any"]
    if fa is not None and fa != f and "fields" in fa and fa["fields"]:
        lines.append(
            pad + f"first GP computation that differs: {fa['i']}"
            f" ({', '.join(fa['fields'])})"
        )
    s = c["step"]
    if s is not None:
        if "base" in s:
            b, n = s["base"], s["new"]
            lines.append(
                pad + f"first step that differs: {s['i']}, {b['kind']} of"
                f" iteration {_it(b['iter'])}: "
                + ", ".join(f"{k} {b[k]!r} -> {n[k]!r}" for k in s["fields"])
            )
        else:
            lines.append(
                pad + f"steps: the first {s['i']} agree; base has"
                f" {s['n_base']}, new {s['n_new']}"
            )
    h = c["horizons"]
    lines.append(
        pad
        + "horizons: "
        + ", ".join(f"{h[t]} evaluations at {t:g}" for t in HORIZON_TOLS)
    )
    lines.append(
        pad
        + "streams: "
        + ", ".join(
            f"{s} {'same' if ok else 'differ'}"
            for s, ok in c["streams"].items()
        )
    )
    fin = c["final"]
    lines.append(
        pad
        + "final: "
        + ", ".join(f"{k} {a!r} vs {b!r}" for k, (a, b) in fin.items())
    )
    if any(c["crash"]):
        lines.append(pad + f"crash: base {c['crash'][0]}, new {c['crash'][1]}")
    return lines


def platform_differences(a, b, prefix=""):
    """The fields in which two platform keys differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += platform_differences(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += platform_differences(x, y, f"{prefix[:-1]}[{i}].")
        return out
    return [] if a == b else [f"{prefix[:-1]}: {a!r} vs {b!r}"]


def _describe(meta):
    p, v = meta.get("platform", {}), meta.get("provenance", {})
    rt = p.get("openblas_runtime") or []
    kernels = sorted({(r.get("corename"), r.get("threads")) for r in rt})
    g = v.get("git") or {}
    gp = (v.get("gpyreg_source") or {}).get("git") or {}
    return (
        f"{p.get('os')} {p.get('machine')}, {p.get('cpu')}, numpy"
        f" {p.get('numpy')}, scipy {p.get('scipy')}, OpenBLAS kernels"
        f" {kernels}",
        f"commit {g.get('sha')}{' (dirty)' if g.get('dirty') else ''},"
        f" gpyreg {v.get('gpyreg')} at"
        f" {(v.get('gpyreg_source') or {}).get('path')}"
        f" ({gp.get('sha')}{', dirty' if gp.get('dirty') else ''})",
    )


def pair_traces(base_dir, new_dir=None):
    """Pairs of traces to compare, and the names found on one side only."""
    if new_dir is None:  # the repeats within one recording
        traces = load_dir(base_dir)
        pairs, lonely = [], []
        for name, t in traces.items():
            if t.meta.get("repeat", 0):
                first = trace_name(t.meta["label"], t.meta["seed"])
                if first in traces:
                    pairs.append((traces[first], t))
                else:
                    lonely.append(name)
        return pairs, lonely
    base, new = load_dir(base_dir), load_dir(new_dir)
    pairs = [(base[n], new[n]) for n in base if n in new]
    lonely = sorted(set(base) ^ set(new))
    return pairs, lonely


def cmd_check(args):
    pairs, lonely = pair_traces(args.base, args.new)
    if not pairs:
        print("replay: no runs to compare")
        return 2
    # the platform keys
    mismatched = {}  # each difference, with the runs that show it
    for b, n in pairs:
        for d in platform_differences(b.meta["platform"], n.meta["platform"]):
            mismatched.setdefault(d, []).append(n.name)
    base_desc = _describe(pairs[0][0].meta)
    new_desc = _describe(pairs[0][1].meta)
    print(f"base: {args.base}\n  {base_desc[1]}\n  {base_desc[0]}")
    print(f"new:  {args.new or args.base}\n  {new_desc[1]}\n  {new_desc[0]}")
    if mismatched:
        print("platform keys differ:")
        for d, names in mismatched.items():
            where = "every run" if len(names) == len(pairs) else names
            print(f"  {d}  ({where})")
        if not args.force:
            print(
                "replay: refused; runs repeat exactly only on one platform"
                " (--force compares them all the same)"
            )
            return 2
    else:
        print("platform: identical")
    print()
    n_same = 0
    for b, n in pairs:
        c = compare_traces(b, n)
        n_same += c["identical"]
        for line in format_comparison(c):
            print(line)
    for name in lonely:
        print(f"{name:34s} on one side only")
    print(
        f"\n{len(pairs)} runs compared: {n_same} identical,"
        f" {len(pairs) - n_same} differ"
        + (f"; {len(lonely)} on one side only" if lonely else "")
    )
    return 0 if n_same == len(pairs) and not lonely else 1


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def cmd_report(args):
    traces = load_dir(args.dir)
    if not traces:
        print(f"replay: no traces in {args.dir}")
        return 2
    first = next(iter(traces.values()))
    desc = _describe(first.meta)
    print(f"{args.dir}\n  {desc[1]}\n  {desc[0]}")
    print(f"  pin: {first.meta.get('provenance', {}).get('pin')}")
    print(f"  platform env: {first.meta['platform'].get('env')}")
    for t in traces.values():
        d = platform_differences(first.meta["platform"], t.meta["platform"])
        if d:
            print(f"  {t.name}: platform differs from {first.name}: {d}")
    print()
    head = (
        f"{'run':34s} {'evals':>5s} {'iter':>4s} {'search':>9s}"
        f" {'poll':>7s} {'GP':>9s} {'fval':>11s} {'wall':>6s}  end"
    )
    print(head)
    print("-" * len(head))
    for t in traces.values():
        A, m = t.arrays, t.meta
        sk, so = A["step_kind"], A["step_outcome"]
        n_search = int(np.sum(sk == "search"))
        n_search_ok = int(np.sum((sk == "search") & (so == "success")))
        n_poll = int(np.sum(sk == "poll"))
        n_poll_ok = int(np.sum((sk == "poll") & (so == "success")))
        n_refit = int(np.sum(A["fit_refit"] == 1))
        res = m.get("result") or {}
        end = (
            f"CRASH {m['crash']['type']}"
            if m.get("crash")
            else str(res.get("message", ""))
            .replace("Optimization terminated: ", "")
            .strip()[:40]
        )
        fval = res.get("fval")
        print(
            f"{t.name:34s} {t.n_evals:5d}"
            f" {str(m['counts'].get('iterations')):>4s}"
            f" {n_search_ok:3d}/{n_search:<5d} {n_poll_ok:2d}/{n_poll:<4d}"
            f" {n_refit:3d}/{len(A['fit_kind']):<5d}"
            f" {'' if fval is None else format(fval, '.4g'):>11s}"
            f" {m['wall_s']:5.1f}s  {end}"
        )
    print(
        "\nsearch and poll: successes/steps; GP: refits/computations;"
        " iter: as BADS reports it"
    )
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record traces of seeded runs")
    r.add_argument("--out", default=None)
    r.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    r.add_argument("--seeds", default="0", help="e.g. 0, 0-2 or 0,3")
    r.add_argument("--budget-scale", type=float, default=DEFAULT_BUDGET_SCALE)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--options", default=None, help="JSON dict of options")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument(
        "--coretype",
        default=None,
        help=f"OPENBLAS_CORETYPE (default on x86_64: {PINNED_CORETYPE})",
    )
    r.add_argument(
        "--no-pin",
        action="store_true",
        help="leave the BLAS threads and kernel as the environment sets them",
    )
    c = sub.add_parser("check", help="compare two recordings step by step")
    c.add_argument("base")
    c.add_argument("new", nargs="?", default=None)
    c.add_argument("--force", action="store_true")
    p = sub.add_parser("report", help="tabulate one recording")
    p.add_argument("dir")
    k = sub.add_parser("_record", help=argparse.SUPPRESS)
    k.add_argument("--label", required=True)
    k.add_argument("--seed", type=int, required=True)
    k.add_argument("--budget-scale", type=float, required=True)
    k.add_argument("--repeat", type=int, default=1)
    k.add_argument("--out", required=True)
    k.add_argument("--pin", default="{}")
    k.add_argument("--options", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return {
        "record": cmd_record,
        "check": cmd_check,
        "report": cmd_report,
        "_record": _cmd_child,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
