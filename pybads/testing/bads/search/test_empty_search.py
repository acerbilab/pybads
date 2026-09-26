"""A search that leaves no candidate: the ES search returns an empty set, and
the search step counts it as a failed search, as MATLAB BADS does."""

import numpy as np
import pytest

import pybads.search.es_search as es_search_module
from pybads import BADS
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
