"""Generate or check the oracle fixtures of ``pybads/testing/oracles/``.

Runs the recipes of ``pybads/testing/oracles/_recipes.py`` (short seeded
PyBADS runs on small synthetic targets), takes each run's state at the
start of a chosen search or poll step as plain arrays (``_state.py``), adds
a candidate set, computes every oracle (``_oracles.py``) on the *rebuilt*
state and stores the outputs as the references beside the state, in
``pybads/testing/oracles/fixtures/<recipe>.npz`` and ``.json``. The tests
(``test_oracles.py``) rebuild each state and recompute.

    python dev/scripts/make_oracle_fixtures.py --list
    python dev/scripts/make_oracle_fixtures.py --check
    python dev/scripts/make_oracle_fixtures.py --check --exact
    python dev/scripts/make_oracle_fixtures.py --check --report FILE.json
    python dev/scripts/make_oracle_fixtures.py --dump DIR
    python dev/scripts/make_oracle_fixtures.py --check --exact --against DIR
    python dev/scripts/make_oracle_fixtures.py --rebaseline ORACLE \\
        --reason "..."
    python dev/scripts/make_oracle_fixtures.py --write --reason "..."

``--check`` recomputes every stored oracle and compares it with its
reference under its tolerances, and exits 1 on a failure or a missing
reference. ``--check --exact`` compares bit for bit: the references equal
the numerics of the machine and the BLAS setting that generated them, so on
that machine, with one BLAS thread, it is the gate for a change that must
move nothing. Where the platform key differs from the fixture's
(``_oracles.platform_key``: the system, the CPU, the libraries, the BLAS
threads and kernel), ``--check`` skips the platform-bound outputs, as the
tests do, unless ``PYBADS_ORACLES_ALL`` is set. ``--report`` writes every
comparison, with its deviation, to a JSON file: the measurement of the
tolerance floors under other BLAS settings.

On another machine, ``--dump DIR`` writes every oracle's outputs as this
checkout computes them, and ``--check --exact --against DIR`` compares with
that dump instead of the references, every output included, and refuses a
dump made under another platform key: a dump made with the script of a
worktree at the parent commit, and the check at the change, with the same
BLAS setting, gate a change that must move nothing on any one machine.

``--rebaseline ORACLE --reason TEXT`` recomputes one oracle from the stored
states and replaces its references alone (or adds them, for a new oracle),
for a change that moves that oracle on purpose; it records the reason, the
date, the commit and the largest change of each output in the fixture's
``meta["rebaselined"]``, and checks that every other array is unchanged,
that the new references reproduce and that the other oracles still pass.
Off the generating platform, it refuses an oracle with platform-bound
outputs.

``--write --reason TEXT`` reruns the recipes and replaces every fixture,
references included: a new baseline, for a change of the recipes or of the
snapshot's contents, never a way to make a failing oracle pass. It checks
the margins that the decisions of the portable oracles need (the training
set's radius and its last point, the Sto-BADS outcomes, the hedge's choice),
so that rounding on another platform cannot flip them.

The environment's BLAS threads default to one (``OMP_NUM_THREADS`` and its
kin, unless set). PyBADS comes from the checkout that holds this script,
which it puts first on ``sys.path``; gpyreg from ``PYTHONPATH`` or the
installed one, whose version and checkout's commit the fixture records.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib.metadata import PackageNotFoundError, version  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import json  # noqa: E402

import gpyreg  # noqa: E402
import numpy as np  # noqa: E402

import pybads  # noqa: E402
from pybads import BADS  # noqa: E402
from pybads.search.grid_functions import udist  # noqa: E402
from pybads.testing.oracles._oracles import (  # noqa: E402
    DEFAULT_SEED,
    DRAW_UNIFORM,
    GP_CONDITION_MAX,
    ORACLES,
    PLATFORM_BOUND,
    ScriptedGenerator,
    compare,
    comparison,
    format_rows,
    gp_condition_bound,
    improvement_inputs,
    platform_key,
    training_set_cases,
)
from pybads.testing.oracles._recipes import (  # noqa: E402
    NON_BOX_CONS,
    RECIPES,
    make_target,
)
from pybads.testing.oracles._state import (  # noqa: E402
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
# The smallest relative margin that a decision of a portable oracle needs
MARGIN = 1e-8
# The sizes tried for the reduced training sets of `gp_training_set`
SMALL_TRAINING_SIZES = (24, 23, 25, 22, 26, 21, 27, 20, 28)


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_info(cwd=REPO_ROOT):
    """Short commit and dirty flag (tracked files) of a checkout."""
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


def same_platform(meta):
    return platform_key() == meta.get("platform_key") or bool(
        os.environ.get("PYBADS_ORACLES_ALL")
    )


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


def check_margins(state, refs, seed):
    """Refuse a snapshot on which a decision of a portable oracle lies
    within ``MARGIN`` (relative) of its threshold, where rounding on
    another platform could flip it: the training sets' choices, the
    outcomes of Sto-BADS's rule and the hedge's choices."""
    margin = training_set_margin(state, refs["gp_training_set"])
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
    for label in ("", "gamma0_"):
        draws = refs["hedge"][f"{label}draws"]
        assert np.all(draws[:, 0] == DRAW_UNIFORM) and np.all(draws[:, 1] == 1)
        rng = ScriptedGenerator(seed)
        uniforms = np.array([rng.random() for _ in range(len(draws))])
        cums = np.cumsum(refs["hedge"][f"{label}prob"], axis=1)
        gap = np.min(np.abs(cums - uniforms[:, None]))
        assert gap > MARGIN, f"the hedge's choice at its bound ({gap})"


def compute_references(snap, names=None):
    seed = snap["meta"]["oracle_seed"]
    return {
        name: ORACLES[name](build_state(snap), seed)
        for name in (names or ORACLES)
    }


def write(recipes, reason):
    FIXTURES.mkdir(parents=True, exist_ok=True)
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
            "git": git_info(),
            "versions": {
                p: pkg_version(p)
                for p in ("pybads", "gpyreg", "numpy", "scipy")
            },
            # The commit of the gpyreg checkout that ran (None for an
            # installed copy)
            "gpyreg_git": git_info(Path(gpyreg.__file__).resolve().parent),
            "platform_key": platform_key(),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
        }
        arrays, tree = snapshot_from_bads(bads, gp, meta)
        state = build_state(decode(tree, arrays))
        check_consistency(bads, state)
        tree["meta"]["gp_condition_bound"] = gp_condition_bound(
            state["gp"], state["logger"]
        )
        cand = make_candidates(state, CANDIDATE_SEED)
        tree["cand"] = {"X": encode(cand, "cand/X", arrays)}
        choose_small_training_size(tree, arrays)
        snap = decode(tree, arrays)
        refs = compute_references(snap)
        check_margins(build_state(snap), refs, DEFAULT_SEED)
        for name, out in refs.items():
            tree["ref"][name] = encode(out, f"ref/{name}", arrays)
        path = FIXTURES / recipe.name
        save_snapshot(path, arrays, tree)
        bad = check_one(path, exact=True)
        if bad:
            raise RuntimeError(f"{recipe.name}: the round trip failed: {bad}")
        size = sum(p.stat().st_size for p in snapshot_files(path))
        print(
            f"[write] {recipe.name:18s} {fl.func_count:3d} evaluations,"
            f" iteration {bads.optim_state['iter']}, {size / 1024:.0f} KB,"
            f" {time.perf_counter() - t0:.1f} s",
            flush=True,
        )


# --------------------------------------------------------------------------
# Check
# --------------------------------------------------------------------------


def check_one(path, exact=False, verbose=False, report=None, against=None):
    """Recompute every oracle of one fixture and compare with its
    references, or with a dump (``against``, a directory written by
    :func:`dump` on this platform); returns the failures. A registered
    oracle without a reference is a failure; the platform-bound outputs are
    skipped off the generating platform (``_oracles.comparison``), and
    compared with a dump in full."""
    snap = load_snapshot(path)
    meta = snap["meta"]
    same = same_platform(meta)
    if against is not None:
        snap = dict(snap, ref=load_dump(against, path.name))
        same = True
    bad = []
    for name in sorted(set(ORACLES) - set(snap["ref"])):
        bad.append((name, "no reference"))
    for name in sorted(set(snap["ref"]) - set(ORACLES)):
        bad.append((name, "not a registered oracle"))
    for name in sorted(set(snap["ref"]) & set(ORACLES)):
        orc = ORACLES[name]
        skip, tolerance = comparison(name, snap, same, exact)
        bound, _ = comparison(name, snap, False)
        ref = {k: v for k, v in snap["ref"][name].items() if k not in skip}
        if not ref:
            if verbose:
                print(f"  [{path.name}] {name}: platform-bound, skipped")
            continue
        out = orc(build_state(snap), meta["oracle_seed"])
        out = {k: v for k, v in out.items() if k not in skip}
        rows = compare(ref, out, tolerance)
        if report is not None:
            for key, a, r, ok in rows:
                report.append(
                    {
                        "snapshot": path.name,
                        "oracle": name,
                        "key": key,
                        "class": orc.tolerance_class(key),
                        "platform_bound": key in bound,
                        "max_abs": a,
                        "max_scaled": r,
                        "ok": ok,
                    }
                )
        if verbose and skip:
            print(f"  [{path.name}] {name}: skipped {sorted(skip)}")
        if verbose or not all(r[3] for r in rows):
            print(f"  [{path.name}] {name}\n{format_rows(rows)}", flush=True)
        if not all(r[3] for r in rows):
            bad.append((name, [r[0] for r in rows if not r[3]]))
    return bad


def check(names, exact, verbose, report_path, against=None):
    if against is not None:
        made = json.loads((Path(against) / "dump.json").read_text())
        if made["platform_key"] != platform_key():
            sys.exit(
                f"the dump in {against} was made on another platform or BLAS"
                f" setting: {made['platform_key']}"
            )
    report = [] if report_path else None
    failures = {}
    for name in names:
        bad = check_one(FIXTURES / name, exact, verbose, report, against)
        print(f"[check] {name:18s} {'ok' if not bad else 'FAIL ' + str(bad)}")
        if bad:
            failures[name] = bad
    if report_path:
        Path(report_path).write_text(
            json.dumps(
                {
                    "platform_key": platform_key(),
                    "exact": exact,
                    "rows": report,
                },
                indent=1,
                default=float,
            ),
            encoding="utf-8",
        )
    return failures


def dump(names, out_dir):
    """Write every oracle's outputs, recomputed from each fixture's state,
    to ``out_dir/<fixture>.npz``, with the platform key and the commit in
    ``out_dir/dump.json``: the reference of ``--check --against``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        snap = load_snapshot(FIXTURES / name)
        arrays = {}
        for oracle, orc in ORACLES.items():
            out = orc(build_state(snap), snap["meta"]["oracle_seed"])
            arrays.update({f"{oracle}/{k}": v for k, v in out.items()})
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        print(f"[dump] {name}", flush=True)
    (out_dir / "dump.json").write_text(
        json.dumps(
            {
                "platform_key": platform_key(),
                "git": git_info(),
                "names": names,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def load_dump(out_dir, name):
    """The outputs of a dump for the fixture ``name``, by oracle."""
    ref = {}
    with np.load(Path(out_dir) / f"{name}.npz", allow_pickle=False) as z:
        for k in z.files:
            oracle, key = k.split("/", 1)
            ref.setdefault(oracle, {})[key] = z[k]
    return ref


# --------------------------------------------------------------------------
# Rebaseline
# --------------------------------------------------------------------------


def _finite(v):
    return float(v) if np.isfinite(v) else repr(float(v))


def rebaseline(names, oracle_name, reason):
    """Replace one oracle's references in each fixture, from its stored
    state; see the module's docstring."""
    orc = ORACLES[oracle_name]
    prefix = f"ref/{oracle_name}/"
    pending = []
    for name in names:
        path = FIXTURES / name
        tree, arrays = load_tree(path), load_arrays(path)
        snap = decode(tree, arrays)
        meta = snap["meta"]
        out = orc(build_state(snap), meta["oracle_seed"])
        bound = oracle_name in PLATFORM_BOUND or (
            meta["gp_condition_bound"] > GP_CONDITION_MAX
            and any(orc.depends_on_gp(k) for k in out)
        )
        if bound and platform_key() != meta["platform_key"]:
            sys.exit(
                f"{name}: {oracle_name} has platform-bound outputs and this"
                " is not the platform that generated the fixture"
            )
        old = snap["ref"].get(oracle_name)
        if old is None:
            change = "added"
        else:
            rows = compare(old, out, orc.tolerance)
            print(f"  [{name}] {oracle_name}, old against new")
            print(format_rows(rows))
            change = {r[0]: _finite(r[1]) for r in rows}
        pending.append((name, path, tree, arrays, out, change))

    for name, path, tree, arrays, out, change in pending:
        before = {k: v.copy() for k, v in arrays.items()}
        for key in [k for k in arrays if k.startswith(prefix)]:
            del arrays[key]
        tree["ref"][oracle_name] = encode(out, f"ref/{oracle_name}", arrays)
        tree["meta"].setdefault("rebaselined", []).append(
            {
                "oracle": oracle_name,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git": git_info(),
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
        for key, value in before.items():
            if not key.startswith(prefix):
                assert np.array_equal(value, after[key], equal_nan=True), key
        assert {k for k in after if not k.startswith(prefix)} == {
            k for k in before if not k.startswith(prefix)
        }
        snap = load_snapshot(path)
        again = orc(build_state(snap), snap["meta"]["oracle_seed"])
        rows = compare(snap["ref"][oracle_name], again, (0.0, 0.0))
        if not all(r[3] for r in rows):
            raise RuntimeError(f"{name}: {oracle_name} does not reproduce")
        bad = check_one(path)
        if bad:
            raise RuntimeError(f"{name}: other oracles fail after: {bad}")
        print(f"[rebaseline] {name:18s} {oracle_name} replaced", flush=True)


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
    if args.list:
        for recipe in RECIPES.values():
            print(f"{recipe.name:18s} {recipe.note}")
        for name, orc in ORACLES.items():
            bound = " (platform-bound)" if name in PLATFORM_BOUND else ""
            print(f"  oracle {name}{bound}: {orc.tol}")
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
    print(
        f"[check] gpyreg {Path(gpyreg.__file__).resolve().parent},"
        f" pybads {Path(pybads.__file__).resolve().parent},"
        f" BLAS env {platform_key()['blas_env']}",
        flush=True,
    )
    if args.dump:
        dump(names, args.dump)
        return 0
    failures = check(
        names, args.exact, args.verbose, args.report, args.against
    )
    print(
        f"[check] {len(names) - len(failures)} of {len(names)} fixtures"
        f" pass{' exactly' if args.exact else ''}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
