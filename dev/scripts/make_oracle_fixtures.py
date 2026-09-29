"""Generate or check the oracle fixtures of ``pybads/testing/oracles/``.

Runs the recipes of ``pybads/testing/oracles/_recipes.py`` (short seeded
PyBADS runs on small synthetic targets), takes each run's state at the
start of a chosen search or poll step as plain arrays (``_state.py``), adds
a candidate set, computes every oracle (``_oracles.py``) on the *rebuilt*
state and stores the portable outputs as the references beside the state,
in ``pybads/testing/oracles/fixtures/<recipe>.npz`` and ``.json``. The
tests (``test_oracles.py``) rebuild each state and recompute.

    python dev/scripts/make_oracle_fixtures.py --list
    python dev/scripts/make_oracle_fixtures.py --check
    python dev/scripts/make_oracle_fixtures.py --check --report FILE.json
    python dev/scripts/make_oracle_fixtures.py --check --exact
    python dev/scripts/make_oracle_fixtures.py --dump DIR
    python dev/scripts/make_oracle_fixtures.py --check --exact --against DIR
    python dev/scripts/make_oracle_fixtures.py --rebaseline ORACLE \\
        --reason "..."
    python dev/scripts/make_oracle_fixtures.py --write --reason "..."

The fixtures store the portable outputs alone (``_oracles.platform_bound``),
those that another platform reproduces within their tolerances. The
platform-bound outputs, a GP refit, a whole ES search step and the outputs
through the solve of an ill-conditioned GP, reproduce only on one machine,
with one set of libraries and one BLAS setting: they are compared only
between two commits there, through ``--dump`` and ``--against``.

``--check`` recomputes every oracle, compares each stored output with its
reference under its tolerance, and exits 1 on a failure, a missing
reference or an oracle that raises. It reports, per fixture and in all, the
outputs it compared and the platform-bound outputs, which the fixtures do
not store, and lists the options of a stored state that the code no longer
has (they are dropped when the state is rebuilt). ``--exact`` compares bit
for bit, and only under the platform key (:func:`platform_key`) of the
machine that computed the references: elsewhere it refuses, and names the
fields that differ. ``--report`` writes every comparison with its deviation
to a JSON file: the measurement of the tolerance floors under other BLAS
settings.

``--dump DIR`` writes every output of every oracle, the platform-bound ones
included, as this checkout computes them, with the platform key, to
``DIR``. ``--check --exact --against DIR`` compares every output exactly
with the dump, and refuses a dump made under another platform key: dumped
with the script of a worktree at the parent commit and checked at the
change, on one machine with one BLAS setting, it is the gate for a change
that must move nothing. It reports the outputs it compared and those that
the dump lacks (an oracle or a view that the parent commit does not have);
an output in the dump that this checkout does not compute is a failure.

``--rebaseline ORACLE --reason TEXT`` recomputes one oracle, in every view
of each fixture, from the stored states and replaces its references alone
(or adds them, for a new oracle), for a change that moves that oracle on
purpose. It records the reason, the date, the commit, the platform key and
the largest change of each output in the fixture's ``meta["rebaselined"]``,
and checks that every other array is unchanged, that the new references
reproduce and that the other oracles still pass. It works on any machine; a
platform-bound oracle has no references to replace.

``--write --reason TEXT`` reruns the recipes and replaces every fixture,
references included: a new baseline, for a change of the recipes or of the
snapshot's contents, never a way to make a failing oracle pass. It refuses
a checkout with uncommitted changes outside the fixtures, so that the
fixtures name the commit of the code that wrote them. It checks the margins
that the decisions of the stored outputs need (the training set's radius
and its last point, the Sto-BADS outcomes, the hedge's choices), so that
rounding within the tolerances cannot flip them.

The environment's BLAS threads default to one (``OMP_NUM_THREADS`` and its
kin, unless set). PyBADS comes from the checkout that holds this script,
which it puts first on ``sys.path``; gpyreg from ``PYTHONPATH`` or the
installed one, which the platform key identifies by its source and, for a
checkout, its commit.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib.metadata import PackageNotFoundError, version  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gpyreg  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

import pybads  # noqa: E402
from pybads import BADS  # noqa: E402
from pybads.search.grid_functions import udist  # noqa: E402
from pybads.testing.oracles._oracles import (  # noqa: E402
    DEFAULT_SEED,
    DRAW_UNIFORM,
    GP_CONDITION_MAX,
    NOISE_FLOOR_CONDITION,
    ORACLES,
    PLATFORM_BOUND,
    ScriptedGenerator,
    case_name,
    compare,
    format_rows,
    gp_condition_bound,
    improvement_inputs,
    noise_floor_hyperparameters,
    oracle_cases,
    platform_bound,
    portable_outputs,
    training_set_cases,
    view_oracles,
)
from pybads.testing.oracles._recipes import (  # noqa: E402
    NON_BOX_CONS,
    RECIPES,
    make_target,
)
from pybads.testing.oracles._state import (  # noqa: E402
    SCHEMA_VERSION,
    build_state,
    decode,
    encode,
    load_arrays,
    load_snapshot,
    load_tree,
    save_snapshot,
    snapshot_files,
    snapshot_from_bads,
    snapshot_names,
)

FIXTURES = REPO_ROOT / "pybads" / "testing" / "oracles" / "fixtures"
CANDIDATE_SEED = 20260928
# The smallest relative margin that a decision computed from GP-free
# outputs needs; a decision computed from outputs through the GP's solve
# needs their tolerance (`decision_margin`)
MARGIN = 1e-8
# The sizes tried for the reduced training sets of `gp_training_set`
SMALL_TRAINING_SIZES = (24, 23, 25, 22, 26, 21, 27, 20, 28)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# --------------------------------------------------------------------------
# Provenance and platform
# --------------------------------------------------------------------------


def _git(args, cwd):
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_info(cwd=REPO_ROOT, exclude=()):
    """Short commit, ``git describe`` and dirty flag (tracked files, but
    the paths of ``exclude``) of a checkout; ``None`` values outside
    one."""
    try:
        excluded = [f":!{p}" for p in exclude]
        return {
            "sha": _git(["rev-parse", "--short", "HEAD"], cwd),
            "describe": _git(
                ["describe", "--tags", "--long", "--always"], cwd
            ),
            "dirty": bool(
                _git(
                    [
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                        "--",
                        ".",
                        *excluded,
                    ],
                    cwd,
                )
            ),
        }
    except Exception:  # noqa: BLE001
        return {"sha": None, "describe": None, "dirty": None}


def checkout_info():
    """The commit of the checkout that runs, the fixtures' own changes
    aside."""
    fixtures = FIXTURES.relative_to(REPO_ROOT).as_posix()
    return git_info(REPO_ROOT, exclude=(fixtures,))


def module_identity(module, dist):
    """A module's source directory, with its commit when it is a git
    checkout that tracks the module, and the installed distribution's
    version."""
    init = Path(module.__file__).resolve()
    source = init.parent
    try:
        installed = version(dist)
    except PackageNotFoundError:
        installed = None
    try:
        _git(["ls-files", "--error-unmatch", init.name], source)
        git = git_info(source)
    except Exception:  # noqa: BLE001  (not in a checkout that tracks it)
        git = None
    return {
        "source": str(source),
        "git": git,
        "installed_version": installed,
    }


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
    for mod in (np, scipy):  # no /proc: the bundled libraries
        root = Path(mod.__file__).resolve().parent
        name = mod.__name__
        for d in (root.parent / f"{name}.libs", root / ".dylibs", root):
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
    """What must be the same for the platform-bound outputs to reproduce
    exactly: the system, its C library, the CPU and the features of it
    that NumPy dispatches to, Python, NumPy, SciPy, BLAS as built and as
    loaded (its kernel and threads), the environment variables that choose
    them, and gpyreg, by its source and commit."""
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
        "gpyreg": module_identity(gpyreg, "gpyreg"),
    }


def key_differences(a, b):
    """The fields in which two platform keys differ."""
    a, b = a or {}, b or {}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# --------------------------------------------------------------------------
# Runs and snapshots
# --------------------------------------------------------------------------


class _Captured(Exception):
    pass


def run_recipe(recipe):
    """Run ``recipe`` up to the start of the first search (or poll) step
    that begins with at least ``capture_evals`` evaluations; returns the
    ``BADS`` object there, a copy of the GP that the step receives, and the
    user options of the run."""
    options = dict(recipe.options)
    options["display"] = "off"
    lb, ub, plb, pub = recipe.bounds
    nbc = NON_BOX_CONS[recipe.non_box_cons] if recipe.non_box_cons else None
    bads = BADS(
        make_target(recipe),
        recipe.x0.copy(),
        lb,
        ub,
        plb,
        pub,
        non_box_cons=nbc,
        options=options,
    )
    captured = {}
    method = f"_{recipe.capture_step}_step_"
    step = getattr(bads, method)

    def capture(gp):
        if bads.function_logger.func_count >= recipe.capture_evals:
            captured["gp"] = copy.deepcopy(gp)
            raise _Captured
        return step(gp)

    setattr(bads, method, capture)
    try:
        bads.optimize()
    except _Captured:
        return bads, captured["gp"], options
    raise RuntimeError(
        f"{recipe.name}: the run ended before a {recipe.capture_step} step"
        f" with {recipe.capture_evals} evaluations"
    )


def make_candidates(state, seed):
    """The candidate set of a snapshot, in the transformed space: training
    inputs, the incumbent's poll points, points near it at the mesh's scale,
    points in the plausible box and points near the hard bounds."""
    rng = np.random.default_rng(seed)
    gp, os_ = state["gp"], state["optim_state"]
    u = np.atleast_2d(np.array(state["incumbent"]["u"], dtype=float))
    D = u.shape[1]
    ms = float(os_["mesh_size"])
    ps = np.ravel(gp.temporary_data["poll_scale"])
    poll = np.vstack(
        [u + s * ms * ps * np.eye(D)[i] for i in range(D) for s in (1, -1)]
    )
    near = u + 4 * ms * rng.standard_normal((32, D))
    lo = np.maximum(os_["plb"], os_["lb"])
    hi = np.minimum(os_["pub"], os_["ub"])
    box = lo + (hi - lo) * rng.random((16, D))
    lb = np.where(np.isfinite(os_["lb"]), os_["lb"], lo - 2 * (hi - lo))
    ub = np.where(np.isfinite(os_["ub"]), os_["ub"], hi + 2 * (hi - lo))
    edge = lb + (ub - lb) * rng.choice([0.02, 0.98], size=(6, D))
    return np.vstack([gp.X[:16], poll, near, box, edge])


def check_consistency(bads, state):
    """The rebuilt state is the run's: the transformer's bounds, the
    logger's rows and the GP's training set."""
    vt, os_ = state["var_transf"], state["optim_state"]
    for key in ("lb", "ub", "plb", "pub"):
        assert np.array_equal(getattr(vt, key), os_[key]), key
    fl = state["logger"]
    n = fl.X_max_idx + 1
    live = bads.function_logger
    assert np.array_equal(fl.X[:n], live.X[:n])
    assert np.array_equal(fl.Y[:n], live.Y[:n])
    gp = state["gp"]
    for x in gp.X:
        assert np.any(np.all(fl.X[:n] == x, axis=1)), "a GP input not logged"
    assert not state["dropped_options"], state["dropped_options"]


def training_set_margin(state, ref):
    """The smallest relative margin of the choices of the ``gp_training_set``
    oracle: of each distance from the radius, and of the gap between the
    last point taken and the first left out."""
    gp, fl, os_ = state["gp"], state["logger"], state["optim_state"]
    n = fl.X_max_idx + 1
    radius2 = (
        state["options"]["gp_radius"] * gp.temporary_data["effective_radius"]
    ) ** 2
    margin = np.inf
    for name, center, _ in training_set_cases(state):
        dist = udist(
            fl.X[:n],
            center,
            gp.temporary_data["len_scale"],
            os_["lb"],
            os_["ub"],
            os_["scale"],
            os_["periodic_vars"],
        )
        dist = np.min(np.atleast_2d(dist), axis=1)
        margin = min(margin, np.min(np.abs(dist - radius2)) / radius2)
        ntrain = int(ref[f"{name}_ntrain"])
        s = np.sort(dist)
        if ntrain < s.size:
            margin = min(margin, (s[ntrain] - s[ntrain - 1]) / s[ntrain])
    return margin


def choose_small_training_size(tree, arrays):
    """Set ``tree["inputs"]["small_n_train_max"]``, the size of the reduced
    training sets of the ``gp_training_set`` oracle, to the first of
    ``SMALL_TRAINING_SIZES`` whose choices have margins: points at equal
    distances, such as the mirror images on the mesh, are common, and a
    boundary between them would be decided by rounding."""
    for size in SMALL_TRAINING_SIZES:
        tree["inputs"]["small_n_train_max"] = size
        snap = decode(tree, arrays)
        ref = ORACLES["gp_training_set"](build_state(snap), DEFAULT_SEED)
        if training_set_margin(build_state(snap), ref) > MARGIN:
            return size
    raise RuntimeError("no reduced training set with margins")


def decision_margin(name, key, terms=1):
    """The margin that a decision computed from the output ``key`` of the
    oracle ``name`` needs, as a sum of ``terms`` entries of that output
    below 1 in magnitude: ``MARGIN`` for a GP-free output; for an output
    through the GP's solve, the most that its tolerance lets such a sum
    move."""
    orc = ORACLES[name]
    if not orc.depends_on_gp(key):
        return MARGIN
    rtol, atol = orc.tolerance(key)
    return terms * (rtol + atol)


def check_margins(snap, outputs, seed):
    """Refuse a snapshot on which a decision of a stored output lies within
    its margin of its threshold, where rounding within the tolerances could
    flip it: the training sets' choices, the outcomes of Sto-BADS's rule
    and the hedge's choices, in each view. ``outputs`` holds every output
    by case."""
    state = build_state(snap)
    margin = training_set_margin(state, outputs["gp_training_set"])
    assert margin > MARGIN, f"a training set's choice at its bound ({margin})"
    options = state["options"]
    (f_base, f_new, s_base, s_new), frames = improvement_inputs(state)
    power = options["stobads_frame_size_scaling_power"]
    for gamma in (1.96, 1.5):
        for frame in frames:
            eps = np.sqrt(s_base**2 + s_new**2)
            bound = gamma * eps * frame**power
            mu = f_base - f_new
            pos = bound > 0
            rel = np.abs(np.abs(mu[pos]) - bound[pos]) / bound[pos]
            assert np.all(rel > MARGIN), "a Sto-BADS outcome at its bound"
    for case, name, view in oracle_cases(snap):
        if name != "hedge":
            continue
        out = outputs[case]
        for label in ("", "gamma0_"):
            key = f"{label}prob"
            if platform_bound(snap, view, name, key):
                continue
            draws = out[f"{label}draws"]
            assert np.all(draws[:, 0] == DRAW_UNIFORM)
            assert np.all(draws[:, 1] == 1)
            rng = ScriptedGenerator(seed)
            uniforms = np.array([rng.random() for _ in range(len(draws))])
            cums = np.cumsum(out[key], axis=1)
            gap = np.min(np.abs(cums - uniforms[:, None]))
            need = decision_margin(name, key, terms=cums.shape[1])
            assert gap > need, f"{case}: a choice at its bound ({gap})"


def compute_outputs(snap, cases):
    """Every output of the cases ``(case, name, view)``, by case, each
    computed on a state rebuilt for it, as the tests do."""
    seed = snap["meta"]["oracle_seed"]
    return {
        case: ORACLES[name](build_state(snap, view), seed)
        for case, name, view in cases
    }


def write(recipes, reason):
    here = checkout_info()
    if here["dirty"] is not False:
        sys.exit(
            f"{REPO_ROOT} has uncommitted changes outside the fixtures:"
            " commit them first, so that the fixtures name the commit of"
            " the code that wrote them"
        )
    FIXTURES.mkdir(parents=True, exist_ok=True)
    key = platform_key()
    for recipe in recipes:
        t0 = time.perf_counter()
        bads, gp, user_options = run_recipe(recipe)
        fl = bads.function_logger
        meta = {
            "recipe": recipe.name,
            "note": recipe.note,
            "D": recipe.D,
            "target": recipe.target,
            "target_args": recipe.target_args,
            "x0": recipe.x0.tolist(),
            "bounds": [b.tolist() for b in recipe.bounds],
            "user_options": user_options,
            "non_box_cons": recipe.non_box_cons,
            "capture": {
                "step": recipe.capture_step,
                "capture_evals": recipe.capture_evals,
                "func_count": int(fl.func_count),
                "iter": int(bads.optim_state["iter"]),
                "search_count": int(bads.optim_state["search_count"]),
            },
            "oracle_seed": DEFAULT_SEED,
            "candidate_seed": CANDIDATE_SEED,
            # The checkout that wrote the fixture, by commit and by
            # `git describe` (the installed version's metadata can be stale)
            "pybads": here,
            "platform_key": key,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
        }
        arrays, tree = snapshot_from_bads(bads, gp, meta)
        state = build_state(decode(tree, arrays))
        check_consistency(bads, state)
        bound = gp_condition_bound(state["gp"], state["logger"])
        tree["meta"]["gp_condition_bound"] = {"stored": bound}
        if bound > GP_CONDITION_MAX:
            hyp = noise_floor_hyperparameters(state["gp"], state["logger"])
            tree["inputs"]["noise_floor_hyp"] = encode(
                hyp, "inputs/noise_floor_hyp", arrays
            )
            floored = build_state(decode(tree, arrays), "noise_floor")
            bound = gp_condition_bound(floored["gp"], floored["logger"])
            assert bound <= NOISE_FLOOR_CONDITION * (1 + 1e-9), bound
            tree["meta"]["gp_condition_bound"]["noise_floor"] = bound
        cand = make_candidates(state, CANDIDATE_SEED)
        tree["cand"] = {"X": encode(cand, "cand/X", arrays)}
        choose_small_training_size(tree, arrays)
        snap = decode(tree, arrays)
        cases = oracle_cases(snap)
        outputs = compute_outputs(snap, cases)
        check_margins(snap, outputs, DEFAULT_SEED)
        for case, name, view in cases:
            out = portable_outputs(snap, view, name, outputs[case])
            tree["ref"][case] = encode(out, f"ref/{case}", arrays)
        path = FIXTURES / recipe.name
        save_snapshot(path, arrays, tree)
        tally = check_one(path, exact=True)
        if tally["failures"]:
            raise RuntimeError(
                f"{recipe.name}: the round trip failed: {tally['failures']}"
            )
        size = sum(p.stat().st_size for p in snapshot_files(path))
        print(
            f"[write] {recipe.name:18s} {fl.func_count:3d} evaluations,"
            f" iteration {bads.optim_state['iter']}, {size / 1024:.0f} KB,"
            f" {tally['compared']} outputs stored,"
            f" {tally['bound']} platform-bound,"
            f" {time.perf_counter() - t0:.1f} s",
            flush=True,
        )


# --------------------------------------------------------------------------
# Check
# --------------------------------------------------------------------------


def reference_keys(meta):
    """The platform key under which each oracle's references were computed:
    that of its last rebaseline, or else of the fixture's generation."""
    keys = {name: meta["platform_key"] for name in ORACLES}
    for entry in meta.get("rebaselined", []):
        keys[entry["oracle"]] = entry["platform_key"]
    return keys


def check_one(path, exact=False, verbose=False, report=None, against=None):
    """Recompute every oracle of one fixture, in every view, and compare
    with its stored references (under the tolerances, or exactly), or with
    a dump (``against``, a directory written by :func:`dump`, every output
    exactly). Returns a tally: the outputs compared, the platform-bound ones
    (not stored), those absent from the dump, the options of the stored
    state that the code no longer has, and the failures."""
    snap = load_snapshot(path)
    meta = snap["meta"]
    tally = {
        "compared": 0,
        "bound": 0,
        "not_in_dump": [],
        "dropped_options": build_state(snap)["dropped_options"],
        "failures": [],
    }
    if against is not None:
        ref_all = load_dump(against, path.name)
        if ref_all is None:
            tally["failures"].append(("*", "the dump has no such fixture"))
            return tally
        cases = oracle_cases(snap, stored_only=False)
    else:
        ref_all = snap["ref"]
        cases = oracle_cases(snap, stored_only=False)
        stored = {c for c, _, _ in oracle_cases(snap)}
        for case in sorted(set(ref_all) - stored):
            tally["failures"].append((case, "not a case of the oracles"))
    known = {c for c, _, _ in cases}
    for case in sorted(set(ref_all) - known):
        if against is not None:
            tally["failures"].append((case, "in the dump, not computed"))
    for case, name, view in cases:
        orc = ORACLES[name]
        try:
            out = orc(build_state(snap, view), meta["oracle_seed"])
        except Exception as err:  # noqa: BLE001
            tally["failures"].append((case, f"{type(err).__name__}: {err}"))
            print(f"  [{path.name}] {case} raised {err!r}", flush=True)
            continue
        if against is not None:
            ref = ref_all.get(case, {})
            absent = sorted(set(out) - set(ref))
            tally["not_in_dump"] += [f"{case}/{k}" for k in absent]
            out = {k: v for k, v in out.items() if k in ref}
            if not out and not ref:
                continue
            tolerance = (0.0, 0.0)
        else:
            portable = portable_outputs(snap, view, name, out)
            tally["bound"] += len(out) - len(portable)
            if name in PLATFORM_BOUND:
                continue
            if case not in ref_all:
                tally["failures"].append((case, "no reference"))
                continue
            ref, out = ref_all[case], portable
            tolerance = (0.0, 0.0) if exact else orc.tolerance
        rows = compare(ref, out, tolerance)
        tally["compared"] += len(rows)
        if report is not None:
            for key, a, r, ok in rows:
                report.append(
                    {
                        "snapshot": path.name,
                        "case": case,
                        "oracle": name,
                        "view": view,
                        "key": key,
                        "class": orc.tolerance_class(key),
                        "platform_bound": platform_bound(
                            snap, view, name, key
                        ),
                        "max_abs": a,
                        "max_scaled": r,
                        "ok": ok,
                    }
                )
        if verbose or not all(r[3] for r in rows):
            print(f"  [{path.name}] {case}\n{format_rows(rows)}", flush=True)
        if not all(r[3] for r in rows):
            tally["failures"].append((case, [r[0] for r in rows if not r[3]]))
    if verbose and tally["not_in_dump"]:
        print(f"  [{path.name}] not in the dump: {tally['not_in_dump']}")
    return tally


def check(names, exact, verbose, report_path, against=None):
    here = platform_key()
    if against is not None:
        made = json.loads((Path(against) / "dump.json").read_text())
        differ = key_differences(made.get("platform_key"), here)
        if differ:
            sys.exit(
                f"the dump in {against} was made under another platform key"
                f" (it differs in {', '.join(differ)}): dump again on this"
                " machine, with this BLAS setting"
            )
    elif exact:
        refused = []
        for name in names:
            meta = load_tree(FIXTURES / name)["meta"]
            for oracle, key in sorted(reference_keys(meta).items()):
                if oracle in PLATFORM_BOUND:
                    continue
                differ = key_differences(key, here)
                if differ:
                    refused.append(f"{name}/{oracle} ({', '.join(differ)})")
        if refused:
            sys.exit(
                "--exact compares bit for bit with references computed under"
                " another platform key, so it would fail on rounding alone:"
                f" {'; '.join(refused[:4])}"
                f"{'; ...' if len(refused) > 4 else ''}. Compare two commits"
                " on this machine instead: --dump DIR at the parent commit,"
                " then --check --exact --against DIR at the change"
            )
    report = [] if report_path else None
    totals = {"compared": 0, "bound": 0, "not_in_dump": 0}
    failures = {}
    for name in names:
        tally = check_one(FIXTURES / name, exact, verbose, report, against)
        for k in totals:
            v = tally[k]
            totals[k] += len(v) if isinstance(v, list) else v
        state = "ok" if not tally["failures"] else f"FAIL {tally['failures']}"
        extra = (
            f", {len(tally['not_in_dump'])} not in the dump"
            if against is not None
            else f", {tally['bound']} platform-bound not stored"
        )
        dropped = (
            f"; options dropped: {', '.join(tally['dropped_options'])}"
            if tally["dropped_options"]
            else ""
        )
        print(
            f"[check] {name:18s} {state} ({tally['compared']} compared"
            f"{extra}){dropped}",
            flush=True,
        )
        if tally["failures"]:
            failures[name] = tally["failures"]
    if report_path:
        Path(report_path).write_text(
            json.dumps(
                {
                    "platform_key": here,
                    "exact": exact,
                    "against": None if against is None else str(against),
                    "rows": report,
                },
                indent=1,
                default=float,
            ),
            encoding="utf-8",
        )
    how = "exactly " if exact or against is not None else ""
    if against is not None:
        skipped = (
            f"{totals['not_in_dump']} outputs not in the dump (an oracle or"
            " a view that the dump's commit does not have)"
        )
        what = f"with the dump in {against}"
    else:
        skipped = (
            f"{totals['bound']} platform-bound outputs are not stored (the"
            " gate for those is --dump and --against)"
        )
        what = "with the stored references"
    print(
        f"[check] {len(names) - len(failures)} of {len(names)} fixtures pass:"
        f" {totals['compared']} outputs compared {how}{what}; {skipped}"
    )
    if totals["compared"] == 0:
        failures["*"] = "no output compared"
    return failures


def dump(names, out_dir):
    """Write every output of every oracle, in every view, the platform-bound
    ones included, recomputed from each fixture's state, to
    ``out_dir/<fixture>.npz``, with the platform key and the commit in
    ``out_dir/dump.json``: the reference of ``--check --against``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        snap = load_snapshot(FIXTURES / name)
        outputs = compute_outputs(snap, oracle_cases(snap, stored_only=False))
        arrays = {
            f"{case}/{k}": v
            for case, out in outputs.items()
            for k, v in out.items()
        }
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        print(f"[dump] {name}: {len(arrays)} outputs", flush=True)
    (out_dir / "dump.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "platform_key": platform_key(),
                "pybads": dict(checkout_info(), source=str(REPO_ROOT)),
                "names": names,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def load_dump(out_dir, name):
    """The outputs of a dump for the fixture ``name``, by case, or ``None``
    if the dump lacks it."""
    path = Path(out_dir) / f"{name}.npz"
    if not path.exists():
        return None
    ref = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            case, key = k.split("/", 1)
            ref.setdefault(case, {})[key] = z[k]
    return ref


# --------------------------------------------------------------------------
# Rebaseline
# --------------------------------------------------------------------------


def _finite(v):
    return float(v) if np.isfinite(v) else repr(float(v))


def rebaseline(names, oracle_name, reason):
    """Replace one oracle's references in each fixture, in every view,
    from its stored state; see the module's docstring."""
    if oracle_name in PLATFORM_BOUND:
        sys.exit(
            f"{oracle_name} is platform-bound: the fixtures store none of its"
            " outputs, and --dump and --against compare them"
        )
    orc = ORACLES[oracle_name]
    key = platform_key()
    pending = []
    for name in names:
        path = FIXTURES / name
        tree, arrays = load_tree(path), load_arrays(path)
        snap = decode(tree, arrays)
        cases = [c for c in oracle_cases(snap) if c[1] == oracle_name]
        outputs = compute_outputs(snap, cases)
        news, change = {}, {}
        for case, _, view in cases:
            new = portable_outputs(snap, view, oracle_name, outputs[case])
            old = snap["ref"].get(case)
            if old is None:
                change[case] = "added"
            else:
                rows = compare(old, new, orc.tolerance)
                print(f"  [{name}] {case}, old against new")
                print(format_rows(rows))
                change[case] = {r[0]: _finite(r[1]) for r in rows}
            news[case] = new
        pending.append((name, path, tree, arrays, news, change))

    for name, path, tree, arrays, news, change in pending:
        prefixes = tuple(f"ref/{case}/" for case in news)
        before = {k: v.copy() for k, v in arrays.items()}
        for k in [k for k in arrays if k.startswith(prefixes)]:
            del arrays[k]
        for case, new in news.items():
            tree["ref"][case] = encode(new, f"ref/{case}", arrays)
        tree["meta"].setdefault("rebaselined", []).append(
            {
                "oracle": oracle_name,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pybads": checkout_info(),
                "platform_key": key,
                "reason": reason,
                "max_abs_change": change,
            }
        )
        # Through temporary files outside the fixtures directory, which the
        # tests read, renamed into place
        tmp = FIXTURES.parent / (path.name + ".rewrite-tmp")
        save_snapshot(tmp, arrays, tree)
        for src, dst in zip(snapshot_files(tmp), snapshot_files(path)):
            os.replace(src, dst)
        after = load_arrays(path)
        for k, value in before.items():
            if not k.startswith(prefixes):
                assert np.array_equal(value, after[k], equal_nan=True), k
        assert {k for k in after if not k.startswith(prefixes)} == {
            k for k in before if not k.startswith(prefixes)
        }
        snap = load_snapshot(path)
        cases = [c for c in oracle_cases(snap) if c[1] == oracle_name]
        again = compute_outputs(snap, cases)
        for case, _, view in cases:
            new = portable_outputs(snap, view, oracle_name, again[case])
            rows = compare(snap["ref"][case], new, (0.0, 0.0))
            if not all(r[3] for r in rows):
                raise RuntimeError(f"{name}: {case} does not reproduce")
        tally = check_one(path)
        if tally["failures"]:
            raise RuntimeError(
                f"{name}: the oracles fail after: {tally['failures']}"
            )
        print(
            f"[rebaseline] {name:18s} {oracle_name} replaced"
            f" ({', '.join(news)})",
            flush=True,
        )


# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--rebaseline", metavar="ORACLE")
    mode.add_argument("--dump", metavar="DIR")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--against", metavar="DIR")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--report", metavar="PATH")
    ap.add_argument("--reason")
    ap.add_argument(
        "--only", help="comma-separated recipe names (default: all)"
    )
    args = ap.parse_args(argv)

    names = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only
        else None
    )
    if args.against and not (args.check and args.exact):
        ap.error("--against goes with --check --exact")
    if args.list:
        for recipe in RECIPES.values():
            print(f"{recipe.name:18s} {recipe.note}")
        for name, orc in ORACLES.items():
            bound = " (platform-bound)" if name in PLATFORM_BOUND else ""
            views = [
                v for v in ("stored", "noise_floor") if name in view_oracles(v)
            ]
            print(f"  oracle {name}{bound}, views {views}: {orc.tol}")
        return 0
    if args.write:
        if not args.reason:
            ap.error("--write needs --reason")
        recipes = [RECIPES[n] for n in (names or RECIPES)]
        write(recipes, args.reason)
        return 0
    stored = snapshot_names(FIXTURES)
    names = names or stored
    if args.rebaseline:
        if not args.reason:
            ap.error("--rebaseline needs --reason")
        if args.rebaseline not in ORACLES:
            ap.error(f"unknown oracle {args.rebaseline!r}")
        rebaseline(names, args.rebaseline, args.reason)
        return 0
    key = platform_key()
    print(
        f"[check] gpyreg {key['gpyreg']['source']},"
        f" pybads {Path(pybads.__file__).resolve().parent},"
        f" BLAS env {key['env']},"
        f" OpenBLAS {[e['corename'] for e in key['openblas_runtime']]}",
        flush=True,
    )
    if args.dump:
        dump(names, args.dump)
        return 0
    failures = check(
        names, args.exact, args.verbose, args.report, args.against
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
