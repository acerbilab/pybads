"""A search that leaves no candidate: the ES search returns an empty set, and
the search step counts it as a failed search, as MATLAB BADS does. A later
generation of the ES that leaves no candidate adds none, and the search
returns the best of the earlier ones."""

import logging

import numpy as np
import pytest

import pybads.search.es_search as es_search_module
from pybads import BADS
from pybads.function_examples import rosenbrocks_fcn
from pybads.search.search_hedge import ESSearchHedge

D = 3


def _noisy_sphere(noise_seed=0):
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        return float(np.sum(np.atleast_2d(x) ** 2)) + rng.standard_normal()

    return fun


def _run(**options):
    opts = {
        "display": "off",
        "max_fun_evals": 80,
        "random_seed": 3,
        "uncertainty_handling": True,
    }
    opts.update(options)
    return BADS(
        _noisy_sphere(),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    ).optimize()


@pytest.mark.parametrize("stobads", [False, True], ids=["bads", "stobads"])
def test_empty_search_set_is_a_failed_search(monkeypatch, stobads):
    """The second search returns no candidate: it counts as a failure, and
    the run carries on."""
    calls = {"n": 0}
    statuses = []
    original_hedge = ESSearchHedge.__call__
    original_stats = BADS._update_search_stats_

    def hedge(self, *args, **kwargs):
        calls["n"] += 1
        us, z = original_hedge(self, *args, **kwargs)
        if calls["n"] == 2:
            return np.empty((0, D)), np.empty(0)
        return us, z

    def stats(self, search_status, search_dist):
        statuses.append(search_status)
        return original_stats(self, search_status, search_dist)

    monkeypatch.setattr(ESSearchHedge, "__call__", hedge)
    monkeypatch.setattr(BADS, "_update_search_stats_", stats)
    result = _run(stobads=stobads)
    assert calls["n"] > 2
    assert statuses[1] == "failure"
    assert np.isfinite(result["fval"])


def test_es_search_without_candidates_returns_an_empty_set(monkeypatch):
    """When the constraint check removes every candidate, the ES search
    returns an empty set, and every search of the run fails."""
    returned = []
    original_call = es_search_module.ESSearch.__call__

    def no_candidates(u_new, *args, **kwargs):
        return np.empty((0, np.atleast_2d(u_new).shape[1]))

    def call(self, *args, **kwargs):
        out = original_call(self, *args, **kwargs)
        returned.append(out)
        return out

    monkeypatch.setattr(es_search_module, "contraints_check", no_candidates)
    monkeypatch.setattr(es_search_module.ESSearch, "__call__", call)
    result = _run()
    assert len(returned) > 0
    for us, z in returned:
        assert us.shape == (0, D)
    assert np.isfinite(result["fval"])


def test_empty_search_set_decays_the_hedge_gains(monkeypatch):
    """An empty search set updates the search hedge as a failed search: every
    gain decays, with no reward, as MATLAB BADS's acqPortfolio gives its
    chosen search a reward of 0."""
    calls = {"n": 0}
    gains = {}
    original_hedge = ESSearchHedge.__call__
    original_stats = BADS._update_search_stats_

    def hedge(self, *args, **kwargs):
        calls["n"] += 1
        us, z = original_hedge(self, *args, **kwargs)
        if calls["n"] == 3:
            gains["before"] = self.g.copy()
            return np.empty((0, D)), np.empty(0)
        return us, z

    def stats(self, search_status, search_dist):
        if calls["n"] == 3 and "after" not in gains:
            gains["after"] = self.search_es_hedge.g.copy()
            gains["decay"] = self.search_es_hedge.decay
        return original_stats(self, search_status, search_dist)

    monkeypatch.setattr(ESSearchHedge, "__call__", hedge)
    monkeypatch.setattr(BADS, "_update_search_stats_", stats)
    result = _run()
    assert calls["n"] > 3
    assert np.all(np.isfinite(gains["before"]))
    assert np.any(gains["before"] != 0)
    assert np.array_equal(gains["after"], gains["decay"] * gains["before"])
    assert np.isfinite(result["fval"])


def _initial_state(**options):
    """A BADS object and its GP after the initial design, for a unit call of
    the ES search."""
    bads = BADS(
        rosenbrocks_fcn,
        np.zeros((1, D)),
        -20 * np.ones((1, D)),
        20 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        options={"random_seed": 0, "display": "off", **options},
    )
    bads.options["fun_eval_start"] = 10
    gp, _, _, _ = bads._init_optimization_()
    return bads, gp


@pytest.mark.parametrize("n_search_iter", [2, 3])
def test_es_search_keeps_the_candidates_of_an_emptied_generation(
    monkeypatch, caplog, n_search_iter
):
    """When the constraints reject every candidate of the second generation,
    the ES search keeps those of the first and returns the best of them, as
    MATLAB's searchES does; a third generation is reproduced from them, at
    an unchanged scale. The emptied generation is logged at DEBUG, since
    MATLAB's searchES says nothing of it."""
    bads, gp = _initial_state(n_search_iter=n_search_iter)
    caplog.set_level(logging.DEBUG, logger="BADS")
    generations = []
    original_lcb = es_search_module.acq_fcn_lcb

    def lcb(u, *args, **kwargs):
        out = original_lcb(u, *args, **kwargs)
        generations.append((u.copy(), np.ravel(out[0]).copy()))
        return out

    calls = {"n": 0}

    def non_box_cons(x):
        # Every candidate of the second generation violates the constraint
        calls["n"] += 1
        return np.full(len(x), 1.0 if calls["n"] == 2 else 0.0)

    monkeypatch.setattr(es_search_module, "acq_fcn_lcb", lcb)
    mu = int(bads.options["n_search"] / n_search_iter)
    search_es = es_search_module.ESSearchWM(
        mu, mu, bads.options, rng=np.random.default_rng(0)
    )
    us, z = search_es(
        bads.u,
        None,
        None,
        bads.function_logger,
        gp,
        bads.optim_state,
        True,
        non_box_cons,
    )

    sizes = [len(u) for u, _ in generations]
    assert len(sizes) == n_search_iter
    assert sizes[0] > 0 and sizes[1] == 0
    assert all(size > 0 for size in sizes[2:])
    U = np.vstack([u for u, _ in generations])
    Z = np.concatenate([z for _, z in generations])
    assert np.shape(us) == (D,)
    assert z == np.min(Z)
    assert np.any(np.all(U[Z == z] == us, axis=1))
    assert search_es.scale == bads.options["es_start"]
    records = [
        record
        for record in caplog.records
        if "No candidate left in generation 2 of the search"
        in record.getMessage()
    ]
    assert [record.levelno for record in records] == [logging.DEBUG]
