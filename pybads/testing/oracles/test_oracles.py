"""The oracles: PyBADS's components recomputed on stored states, against
the references stored with them.

Every fixture under ``fixtures/`` is the state of a short seeded run at the
start of a search or poll step (``_recipes.py``), saved as plain arrays by
``dev/scripts/make_oracle_fixtures.py``, with the portable outputs of each
oracle (``_oracles.py``) computed from it, in each view of the state. The
tests rebuild the state through the public constructors, recompute every
oracle and compare under the tolerance class of each output, on every
platform. A failure means that the numerics moved. Never loosen a
tolerance, or regenerate the fixtures, to make a change pass: when a change
moves an oracle on purpose, replace that oracle's references alone with
``make_oracle_fixtures.py --rebaseline ORACLE --reason "..."``. The
platform-bound outputs (``_oracles.platform_bound``) are not stored; the
script's ``--dump`` and ``--check --exact --against`` compare them between
two commits on one machine.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pybads.rng import get_rng
from pybads.testing.oracles._oracles import (
    DEFAULT_SEED,
    DRAW_INTEGERS,
    DRAW_NORMAL,
    DRAW_PERMUTATION,
    DRAW_UNIFORM,
    GP_CONDITION_MAX,
    NOISE_FLOOR_CONDITION,
    ORACLES,
    PLATFORM_BOUND,
    TOLERANCES,
    ScriptedGenerator,
    compare,
    format_rows,
    gp_condition_bound,
    logger_rows,
    oracle_cases,
    platform_bound,
    portable_outputs,
    view_oracles,
)
from pybads.testing.oracles._recipes import RECIPES
from pybads.testing.oracles._state import (
    SCHEMA_VERSION,
    build_state,
    load_snapshot,
    load_tree,
    snapshot_names,
    snapshot_views,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NAMES = snapshot_names(FIXTURES) if FIXTURES.exists() else []
STORED = sorted(set(ORACLES) - PLATFORM_BOUND)


@pytest.fixture(scope="module")
def snapshots():
    return {name: load_snapshot(FIXTURES / name) for name in NAMES}


def test_fixtures_present():
    assert set(NAMES) == set(RECIPES), f"fixtures under {FIXTURES}"


@pytest.mark.parametrize("oracle", STORED)
@pytest.mark.parametrize("name", NAMES)
def test_oracle(snapshots, name, oracle):
    """The oracle in each view of the snapshot that computes it: its
    portable outputs, which are exactly those stored, within their
    tolerances of the references."""
    snap = snapshots[name]
    orc = ORACLES[oracle]
    compared = 0
    for case, name_, view in oracle_cases(snap):
        if name_ != oracle:
            continue
        assert case in snap["ref"], f"{name}: no reference for {case}"
        out = orc(build_state(snap, view), snap["meta"]["oracle_seed"])
        out = portable_outputs(snap, view, oracle, out)
        rows = compare(snap["ref"][case], out, orc.tolerance)
        assert all(r[3] for r in rows), f"{name}/{case}:\n{format_rows(rows)}"
        compared += len(rows)
    assert compared > 0, f"{name}/{oracle}: no output stored"


def test_fixtures_hold_the_portable_cases(snapshots):
    """A fixture stores the cases of its views, those of the oracles that
    are not platform-bound, and no other."""
    for name, snap in snapshots.items():
        assert set(snap["ref"]) == {c for c, _, _ in oracle_cases(snap)}
        assert load_tree(FIXTURES / name)["schema"] == SCHEMA_VERSION


def test_platform_bound_outputs():
    """Every output of a platform-bound oracle is bound, and so are the
    outputs through the GP's solve in a view whose GP is ill-conditioned
    (and those alone): the GP-free outputs of such a view are stored."""
    keys = ["fmu", "fs2", "small_worst_rows", "gamma0_g", "prob", "state_B"]
    for well in (True, False):
        c = GP_CONDITION_MAX / 10 if well else GP_CONDITION_MAX * 10
        snap = {"meta": {"gp_condition_bound": {"stored": c}}}
        for name, orc in ORACLES.items():
            bound = {
                k for k in keys if platform_bound(snap, "stored", name, k)
            }
            if name in PLATFORM_BOUND:
                assert bound == set(keys), name
            elif well:
                assert bound == set(), name
            else:
                assert bound == {k for k in keys if orc.depends_on_gp(k)}
    hedge = ORACLES["hedge"]
    assert hedge.depends_on_gp("gamma0_chosen")
    assert not hedge.depends_on_gp("chosen")
    assert ORACLES["gp_predict"].depends_on_gp("fmu")
    assert ORACLES["gp_training_set"].depends_on_gp("small_worst_fmu")
    assert not ORACLES["gp_training_set"].depends_on_gp("small_worst_rows")


def test_views_and_cases():
    """The noise-floor view computes the oracles with outputs through the
    GP's solve, but the platform-bound ones; the stored view computes every
    oracle, and the fixtures store the cases of the portable ones."""
    assert set(view_oracles("stored")) == set(ORACLES)
    assert set(view_oracles("noise_floor")) == {
        "gp_predict",
        "acq_lcb",
        "gp_training_set",
        "hedge",
    }
    snap = {"inputs": {"noise_floor_hyp": np.zeros(3)}}
    assert snapshot_views(snap) == ["stored", "noise_floor"]
    cases = {c for c, _, _ in oracle_cases(snap)}
    assert "gp_predict@noise_floor" in cases and "gp_predict" in cases
    assert not cases & PLATFORM_BOUND
    everything = {c for c, _, _ in oracle_cases(snap, stored_only=False)}
    assert everything - cases == PLATFORM_BOUND
    assert snapshot_views({"inputs": {}}) == ["stored"]


def test_states_rebuild(snapshots):
    """The rebuilt GP's training inputs are rows of the rebuilt log, its
    posterior is finite, the transformer gives the stored bounds, and the
    log holds the evaluations of the capture. A snapshot whose GP is
    ill-conditioned has the noise-floor view, whose GP is not, and differs
    from the stored one in its noise alone."""
    for name, snap in snapshots.items():
        state = build_state(snap)
        gp, fl, os_ = state["gp"], state["logger"], state["optim_state"]
        rows = logger_rows(gp.X, gp.y, fl)
        assert rows.size == gp.X.shape[0], name
        assert all(np.all(np.isfinite(p.alpha)) for p in gp.posteriors), name
        for key in ("lb", "ub", "plb", "pub"):
            np.testing.assert_allclose(
                getattr(state["var_transf"], key), os_[key], rtol=1e-12
            )
        assert fl.func_count == snap["meta"]["capture"]["func_count"], name
        bounds = snap["meta"]["gp_condition_bound"]
        ill = bounds["stored"] > GP_CONDITION_MAX
        assert ("noise_floor" in snapshot_views(snap)) == ill, name
        if not ill:
            continue
        floored = build_state(snap, "noise_floor")
        bound = gp_condition_bound(floored["gp"], floored["logger"])
        assert bound <= NOISE_FLOOR_CONDITION * (1 + 1e-9), name
        np.testing.assert_allclose(bound, bounds["noise_floor"], rtol=1e-9)
        hyp = gp.get_hyperparameters(as_array=True)
        raised = floored["gp"].get_hyperparameters(as_array=True)
        cov_N = gp.covariance.hyperparameter_count(gp.D)
        assert np.all(raised[:, cov_N] > hyp[:, cov_N]), name
        assert np.array_equal(
            np.delete(raised, cov_N, 1), np.delete(hyp, cov_N, 1)
        )


def test_stored_options_the_code_lacks_are_dropped(snapshots):
    """An option of a stored state that the option files no longer define
    is left out of the rebuilt options, and listed."""
    snap = copy.deepcopy(snapshots[NAMES[0]])
    snap["options"]["an_option_since_removed"] = 3
    snap["meta"]["user_options"]["another_one"] = True
    state = build_state(snap)
    assert state["dropped_options"] == [
        "an_option_since_removed",
        "another_one",
    ]
    assert "an_option_since_removed" not in state["options"]
    assert build_state(snapshots[NAMES[0]])["dropped_options"] == []


def test_missing_state_key_names_the_remedy(snapshots):
    """A key that the code reads and a stored state lacks raises a
    ``KeyError`` that names the key and where its default goes, in copies
    of the state too."""
    state = build_state(snapshots[NAMES[0]])
    for d in (state["optim_state"], state["gp"].temporary_data):
        for obj in (d, copy.deepcopy(d)):
            with pytest.raises(KeyError, match="a_new_key.*STATE_DEFAULTS"):
                obj["a_new_key"]
            assert obj.get("a_new_key") is None


def test_another_schema_is_refused(tmp_path):
    name = NAMES[0]
    tree = load_tree(FIXTURES / name)
    tree["schema"] = SCHEMA_VERSION + 1
    (tmp_path / f"{name}.json").write_text(json.dumps(tree))
    with pytest.raises(ValueError, match="schema"):
        load_tree(tmp_path / name)


def test_tolerance_classes():
    """The tolerances, which the fixtures' measured floors set (the
    docstring of ``_oracles.py``): changing one is a decision, not a fix."""
    assert TOLERANCES == {
        "exact": (0.0, 0.0),
        "gp_free": (1e-10, 1e-13),
        "linalg": (1e-9, 1e-13),
        "gp_mean": (1e-4, 1e-10),
        "gp_var": (1e-3, 1e-8),
    }
    for name in PLATFORM_BOUND:
        assert ORACLES[name].tol == {"default": "exact"}, name
        assert not ORACLES[name].has_gp_outputs(), name


# --------------------------------------------------------------------------
# The prescribed draws
# --------------------------------------------------------------------------


def test_scripted_generator_passes_through_get_rng():
    rng = ScriptedGenerator(0)
    assert get_rng(rng) is rng


def test_scripted_generator_stream_is_pinned():
    """The draws are exact arithmetic on PCG64's raw stream: these values
    hold on every platform and NumPy version. A change here moves every
    oracle that draws."""
    rng = ScriptedGenerator(DEFAULT_SEED)
    assert rng.random(3).tolist() == [
        0.5304086977153835,
        0.054355376851938586,
        0.31971194795947255,
    ]
    assert rng.standard_normal() == -0.21130840232014858


def test_scripted_generator_draws_and_log():
    rng = ScriptedGenerator(1)
    u = rng.random((4, 2))
    assert u.shape == (4, 2) and np.all((u >= 0) & (u < 1))
    z = rng.normal(size=(3, 2))
    assert z.shape == (3, 2) and np.all(np.abs(z) <= 6)
    k = rng.integers(1, 3, 5)
    assert k.dtype == np.int64 and set(k.tolist()) <= {1, 2}
    assert rng.integers(1, np.float64(4.0), size=(2, 2)).shape == (2, 2)
    p = rng.permutation(6)
    assert sorted(p.tolist()) == list(range(6))
    rows = np.arange(12.0).reshape(4, 3)
    q = rng.permutation(rows)
    assert sorted(q[:, 0].tolist()) == [0.0, 3.0, 6.0, 9.0]
    assert np.isscalar(rng.random())
    log = rng.log_array()
    assert log[:, 0].tolist() == [
        DRAW_UNIFORM,
        DRAW_NORMAL,
        DRAW_INTEGERS,
        DRAW_INTEGERS,
        DRAW_PERMUTATION,
        DRAW_PERMUTATION,
        DRAW_UNIFORM,
    ]
    assert log[:, 1].tolist() == [8, 6, 5, 4, 6, 4, 1]
    assert log[2, 2:].tolist() == [1.0, 3.0]
    again = ScriptedGenerator(1)
    assert np.array_equal(again.random((4, 2)), u)


@pytest.mark.parametrize("method", ["choice", "uniform", "shuffle", "spawn"])
def test_scripted_generator_refuses_unscripted_draws(method):
    with pytest.raises(NotImplementedError, match=method):
        getattr(ScriptedGenerator(0), method)
