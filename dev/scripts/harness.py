"""The harness that the developer tools share: how they build a run of a
configuration of the benchmark, the thread settings of their runs, what
their records keep of the code and the platform that ran, and the form of
those records.

- ``build_run`` builds one run of a configuration of
  ``benchmark_targets.py``, as every tool gives it to ``BADS``: the
  configuration's problem at a seed and a budget scale, the tool's extra
  options merged into its options, and the evaluations made before the run
  when the configuration has them, with what a record keeps of them
  (``precomputed_summary``).
- ``THREAD_VARS`` names the variables that set the number of BLAS and
  OpenMP threads, which the tools set to one for their runs
  (``single_thread_env``) and record (``thread_env``, ``platform_key``).
- ``code_meta`` identifies the code that ran: the commit of the checkout
  that holds the tool, Python and the platform, the versions of NumPy,
  SciPy, PyBADS and gpyreg, and the source and commit of the PyBADS and
  the gpyreg imported (``module_source``; ``module_identity`` is the same
  in the layout of the oracles' fixtures). ``git_info`` gives a
  checkout's commit and whether its tracked files under a directory have
  uncommitted changes.
- ``platform_key`` identifies the numerical platform: what must be the same
  for two runs of one commit to repeat bit for bit.
- ``parse_seeds``, ``jsonable``, ``write_json`` and ``timestamp``: the
  seeds a tool is given, and the JSON of its records.

The module imports NumPy only inside its functions, so that a script can
set the thread variables through it before NumPy loads its BLAS, and it
leaves ``sys.path`` as it is: ``build_run`` imports
``benchmark_targets.py``, which puts the checkout that holds it first on
``sys.path``, only when it is given a configuration's label.

A tool copied into a worktree at an older commit, as ``replay.py`` is to
record the parent of a change, is copied with this module, and builds the
runs of that commit's ``benchmark_targets.py``, whose problems before
2822c561 have no evaluations made before the run: ``build_run`` gives
their runs none.
"""

import dataclasses
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# The variables that set the number of BLAS and OpenMP threads: OpenMP,
# OpenBLAS, MKL, and Accelerate on macOS
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# --------------------------------------------------------------------------
# Threads
# --------------------------------------------------------------------------


def single_thread_env():
    """One BLAS thread (every variable of ``THREAD_VARS``) and a headless
    matplotlib, in the environment of this process and of the processes
    that it starts after the call; the threads of this process only if
    NumPy has not loaded its BLAS yet."""
    for k in THREAD_VARS:
        os.environ[k] = "1"
    os.environ["MPLBACKEND"] = "Agg"


def thread_env():
    """The variables of ``THREAD_VARS`` as the process sees them (None for
    one that is not set)."""
    return {k: os.environ.get(k) for k in THREAD_VARS}


# --------------------------------------------------------------------------
# Seeds and records
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


def jsonable(v):
    """``v`` with NumPy scalars and arrays, tuples and dict keys made JSON
    types; ``repr`` for any other object."""
    import numpy as np

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


def write_json(path, obj):
    """Write through a temporary file, so that an interrupted write never
    leaves a record that a tool resuming its runs would take as done."""
    path = Path(path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def timestamp():
    """The local time, as the records' ``started`` and ``finished``."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------------
# The code that ran
# --------------------------------------------------------------------------


def pkg_version(name):
    """The installed distribution's version; None when it is not
    installed."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(args, cwd):
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_info(cwd=REPO_ROOT, exclude=(), describe=False):
    """The short commit of the checkout that holds ``cwd``, with ``git
    describe`` if ``describe``, and whether its tracked files under ``cwd``,
    but the paths of ``exclude`` (relative to ``cwd``), have uncommitted
    changes; None values outside a checkout."""
    try:
        info = {"sha": _git(["rev-parse", "--short", "HEAD"], cwd)}
        if describe:
            info["describe"] = _git(
                ["describe", "--tags", "--long", "--always"], cwd
            )
        excluded = [f":!{p}" for p in exclude]
        status = ["status", "--porcelain", "--untracked-files=no"]
        info["dirty"] = bool(_git([*status, "--", ".", *excluded], cwd))
        return info
    except Exception:  # noqa: BLE001
        keys = ("sha", "describe", "dirty") if describe else ("sha", "dirty")
        return dict.fromkeys(keys)


def _package_git(init, describe):
    """``git_info`` of the directory of a package's ``__init__.py``, when a
    checkout tracks the file; None otherwise, as for an installed copy
    under site-packages."""
    try:
        _git(["ls-files", "--error-unmatch", init.name], init.parent)
    except Exception:  # noqa: BLE001
        return None
    return git_info(init.parent, describe=describe)


def module_source(name):
    """Where the package ``name``, imported, loads from, and its commit:
    ``{"path", "git"}``, or None when it does not import.

    ``git`` is the ``git_info`` of the package's directory in the checkout
    that tracks it, which identifies the code that ran: for pybads the
    checkout of the tool, which ``benchmark_targets.py`` puts first on
    ``sys.path``, and for gpyreg a clone placed on ``PYTHONPATH`` or the
    editable checkout; None for an installed copy under site-packages.
    """
    try:
        init = Path(importlib.import_module(name).__file__).resolve()
    except Exception:  # noqa: BLE001
        return None
    return {"path": str(init.parent), "git": _package_git(init, False)}


def module_identity(module, dist):
    """``module_source`` of an imported module in the layout of the oracles'
    fixtures: ``{"source", "git", "installed_version"}``, ``git`` with
    ``git describe``, and the version of the installed distribution
    ``dist``."""
    init = Path(module.__file__).resolve()
    return {
        "source": str(init.parent),
        "git": _package_git(init, True),
        "installed_version": pkg_version(dist),
    }


def code_meta():
    """What identifies the code that ran, as the records' ``meta`` opens:
    the ``git_info`` of the checkout that holds the tool, Python, the
    platform, the versions of NumPy, SciPy, PyBADS and gpyreg, and the
    ``module_source`` of the PyBADS and the gpyreg imported (the version of
    each is that of the installed distribution, which a checkout on
    ``sys.path`` does not change)."""
    import numpy as np

    return {
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": pkg_version("scipy"),
        "pybads": pkg_version("pybads"),
        "pybads_source": module_source("pybads"),
        "gpyreg": pkg_version("gpyreg"),
        "gpyreg_source": module_source("gpyreg"),
    }


# --------------------------------------------------------------------------
# The numerical platform
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


def _numpy_cpu_features():
    """The CPU features that NumPy's dispatch uses, as enabled at run time
    (``NPY_DISABLE_CPU_FEATURES`` removes some)."""
    for path in (
        "numpy._core._multiarray_umath",
        "numpy.core._multiarray_umath",
    ):
        try:
            module = __import__(path, fromlist=["__cpu_features__"])
            features = module.__cpu_features__
        except (ImportError, AttributeError):
            continue
        return sorted(k for k, v in features.items() if v)
    return None


def _blas_build():
    """NumPy's BLAS and LAPACK as built (name, version, configuration)."""
    import numpy as np

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
    """What must be the same for two runs of one commit to repeat bit for
    bit: the system, its C library, the CPU and the features of it that
    NumPy dispatches to, Python, NumPy, SciPy, BLAS as built and as loaded
    (its kernel and threads), and the environment variables that choose
    them (``OPENBLAS_CORETYPE`` and ``THREAD_VARS``)."""
    import numpy as np
    import scipy

    return {
        "os": platform.system(),
        "libc": " ".join(platform.libc_ver()).strip() or None,
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "numpy_cpu_features": _numpy_cpu_features(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_build": _blas_build(),
        "openblas_runtime": _openblas_runtime(),
        "env": {
            k: os.environ.get(k) for k in ("OPENBLAS_CORETYPE",) + THREAD_VARS
        },
    }


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Run:
    """One run of a configuration of the benchmark, as the tools give it to
    ``BADS``: ``BADS(*args, options=options, **kwargs)``."""

    cfg: object  # the benchmark_targets.Config
    prob: object  # its benchmark_targets.Problem at the run's seed
    args: tuple  # the target, x0, the four bounds and non_box_cons
    options: dict  # the problem's options, the tool's extra ones merged
    kwargs: dict  # the evaluations made before the run, if any
    requested: dict  # jsonable(options), as the records keep them
    precomputed: Optional[dict]  # precomputed_summary(cfg, prob)


def build_run(config, seed, budget_scale=1.0, extra_options=None):
    """The ``Run`` of ``config`` (a ``benchmark_targets.Config``, or its
    label) at ``seed``: its problem, whose start point and noise come from
    the seed and whose budget is the configuration's times
    ``budget_scale``, with ``extra_options`` merged into its options. A
    configuration given evaluations made before its runs makes them here,
    by an earlier run of BADS (``benchmark_targets.earlier_evaluations``),
    before a tool wraps anything of the package for the run."""
    if isinstance(config, str):
        import benchmark_targets as bt

        config = bt.find_config(config)
    prob = config.make(seed=seed, budget_scale=budget_scale)
    args, options = prob.bads_args()
    options.update(extra_options or {})
    # a benchmark_targets.py from before 2822c561 gives no run evaluations
    # made before it
    kwargs = prob.bads_kwargs() if hasattr(prob, "bads_kwargs") else {}
    return Run(
        cfg=config,
        prob=prob,
        args=tuple(args),
        options=options,
        kwargs=kwargs,
        requested=jsonable(options),
        precomputed=precomputed_summary(config, prob),
    )


def precomputed_summary(cfg, prob):
    """What a record keeps of the evaluations made before a run: their
    kind (``Config.precomputed``), their number of rows and the first 16
    hex digits of the SHA-256 digest of their arrays, by which
    ``population.py compare`` and ``replay.py check`` check that two
    recordings gave a seed's run the same; None without them."""
    import numpy as np

    precomputed = getattr(prob, "precomputed", None)
    if precomputed is None:
        return None
    digest = hashlib.sha256()
    for a in precomputed:
        digest.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return {
        "kind": cfg.precomputed,
        "rows": int(len(precomputed[1])),
        "digest": digest.hexdigest()[:16],
    }
