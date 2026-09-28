"""The oracles: PyBADS's components recomputed on stored states, against
the references stored with them.

Every fixture under ``fixtures/`` is the state of a short seeded run at the
start of a search or poll step (``_recipes.py``), saved as plain arrays by
``dev/scripts/make_oracle_fixtures.py``, with the outputs of each oracle
(``_oracles.py``) computed from it. The tests rebuild the state through the
public constructors, recompute every oracle and compare under the tolerance
class of each output. A failure means that the numerics moved. Never loosen
a tolerance, or regenerate the fixtures, to make a change pass: when a
change moves an oracle on purpose, replace that oracle's references alone
with ``make_oracle_fixtures.py --rebaseline ORACLE --reason "..."``. The
platform-bound outputs (those of the platform-bound oracles, and those that
go through an ill-conditioned GP's solve: ``_oracles.comparison``) are
compared exactly where the platform key matches the fixture's, and skipped
elsewhere (set ``PYBADS_ORACLES_ALL=1`` to force them); on the machine that
generated the fixtures, ``make_oracle_fixtures.py --check --exact``
compares every output exactly.
"""

import os
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
    ORACLES,
    PLATFORM_BOUND,
    TOLERANCES,
    ScriptedGenerator,
    compare,
    comparison,
    format_rows,
    logger_rows,
    platform_key,
)
from pybads.testing.oracles._recipes import RECIPES
from pybads.testing.oracles._state import (
    build_state,
    load_snapshot,
    snapshot_names,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NAMES = snapshot_names(FIXTURES) if FIXTURES.exists() else []


@pytest.fixture(scope="module")
def snapshots():
    return {name: load_snapshot(FIXTURES / name) for name in NAMES}


def test_fixtures_present():
    assert set(NAMES) == set(RECIPES), f"fixtures under {FIXTURES}"


@pytest.mark.parametrize("oracle", sorted(ORACLES))
@pytest.mark.parametrize("name", NAMES)
def test_oracle(snapshots, name, oracle):
    snap = snapshots[name]
    assert oracle in snap["ref"], f"{name}: no reference for {oracle}"
    here, there = platform_key(), snap["meta"]["platform_key"]
    same = here == there or bool(os.environ.get("PYBADS_ORACLES_ALL"))
    skip, tolerance = comparison(oracle, snap, same)
    ref = {k: v for k, v in snap["ref"][oracle].items() if k not in skip}
    if not ref:
        pytest.skip(
            f"{oracle} is platform-bound: the fixture's platform is {there},"
            f" this one {here}; set PYBADS_ORACLES_ALL=1 to force it"
        )
    out = ORACLES[oracle](build_state(snap), snap["meta"]["oracle_seed"])
    out = {k: v for k, v in out.items() if k not in skip}
    rows = compare(ref, out, tolerance)
    assert all(r[3] for r in rows), f"{name}/{oracle}:\n{format_rows(rows)}"


def test_platform_bound_outputs():
    """Every output of a platform-bound oracle is bound, and so are the
    outputs through the GP's solve on a snapshot whose GP is ill-conditioned
    (and those alone): the GP-free outputs of such a snapshot are compared
    everywhere."""
    names = {"gp_predict", "acq_lcb", "gp_training_set", "hedge"}
    keys = dict.fromkeys(
        ["fmu", "fs2", "small_worst_rows", "gamma0_g", "prob", "state_B"]
    )
    for well in (True, False):
        bound = GP_CONDITION_MAX / 10 if well else GP_CONDITION_MAX * 10
        for name in sorted(names | PLATFORM_BOUND):
            snap = {"meta": {"gp_condition_bound": bound}, "ref": {name: keys}}
            skip, tolerance = comparison(name, snap, same_platform=False)
            orc = ORACLES[name]
            if name in PLATFORM_BOUND:
                assert skip == set(keys), name
            elif well:
                assert skip == set(), name
            else:
                assert skip == {k for k in keys if orc.depends_on_gp(k)}
            none, tolerance = comparison(name, snap, same_platform=True)
            assert none == set()
            for k in skip:
                assert tolerance(k) == (0.0, 0.0)
    hedge = ORACLES["hedge"]
    assert hedge.depends_on_gp("gamma0_chosen")
    assert not hedge.depends_on_gp("chosen")
    assert ORACLES["gp_predict"].depends_on_gp("fmu")
    assert ORACLES["gp_training_set"].depends_on_gp("small_worst_fmu")
    assert not ORACLES["gp_training_set"].depends_on_gp("small_worst_rows")


def test_fixtures_hold_only_registered_oracles(snapshots):
    for name, snap in snapshots.items():
        assert set(snap["ref"]) == set(ORACLES), name


def test_states_rebuild(snapshots):
    """The rebuilt GP's training inputs are rows of the rebuilt log, its
    posterior is finite, the transformer gives the stored bounds, and the
    log holds the evaluations of the capture."""
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
