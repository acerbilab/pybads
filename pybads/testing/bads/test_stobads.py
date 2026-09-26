"""The poll of Sto-BADS (`stobads=True`), after the rule of Sto-MADS: a poll
succeeds if some polled point succeeds, and fails for certain only if every
point does."""

import numpy as np
import pytest

from pybads import BADS

D = 3


def _noisy_sphere(noise_seed=0):
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        return float(np.sum(np.atleast_2d(x) ** 2)) + rng.standard_normal()

    return fun


def _scripted_first_poll(monkeypatch, outcomes):
    """Replace the Sto-BADS rule by `outcomes`, point after point, in the
    first poll, and record that poll: the estimates the rule received, the
    mesh size integer before and after, and the estimates the incumbent
    moved to."""
    state = {"in_poll": False, "estimates": [], "first": None, "moves": []}
    original_poll = BADS._poll_step_
    original_rule = BADS._sto_success_improvement_
    original_move = BADS._update_incumbent_

    def poll(self, gp):
        first = state["first"] is None
        state["in_poll"] = first
        before = self.mesh_size_integer
        try:
            return original_poll(self, gp)
        finally:
            if first:
                state["first"] = {
                    "estimates": list(state["estimates"]),
                    "mesh_before": before,
                    "mesh_after": self.mesh_size_integer,
                    "moves": list(state["moves"]),
                }
            state["in_poll"] = False

    def rule(self, f_base, f_new, *args):
        if state["in_poll"]:
            state["estimates"].append(f_new)
            return outcomes[min(len(state["estimates"]), len(outcomes)) - 1]
        return original_rule(self, f_base, f_new, *args)

    def move(self, u_new, yval_new, fval_new, fsd_new):
        if state["in_poll"]:
            state["moves"].append(fval_new)
        return original_move(self, u_new, yval_new, fval_new, fsd_new)

    monkeypatch.setattr(BADS, "_poll_step_", poll)
    monkeypatch.setattr(BADS, "_sto_success_improvement_", rule)
    monkeypatch.setattr(BADS, "_update_incumbent_", move)
    return state


def _run(opp_stobads):
    return BADS(
        _noisy_sphere(),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={
            "display": "off",
            "max_fun_evals": 80,
            "random_seed": 3,
            "uncertainty_handling": True,
            "stobads": True,
            "opp_stobads": opp_stobads,
            "complete_poll": True,
            # Below its maximum (0), so that a successful poll can expand it
            "init_mesh_size_integer": -2,
        },
    ).optimize()


@pytest.mark.parametrize("opp_stobads", [False, True])
def test_success_then_failures_is_a_successful_poll(monkeypatch, opp_stobads):
    """A success at the second point, then certain failures: the poll
    succeeds, moves the incumbent to the successful point and expands the
    mesh."""
    state = _scripted_first_poll(monkeypatch, [-1, 1, -1])
    _run(opp_stobads)
    poll = state["first"]
    assert len(poll["estimates"]) >= 3
    assert poll["moves"] == [poll["estimates"][1]]
    assert poll["mesh_after"] == poll["mesh_before"] + 1


@pytest.mark.parametrize(
    "opp_stobads, moves", [(False, False), (True, True)], ids=["off", "on"]
)
def test_uncertain_poll_moves_only_with_opp_stobads(
    monkeypatch, opp_stobads, moves
):
    """No success and some uncertain point: the mesh contracts, and the
    incumbent moves only with `opp_stobads`."""
    state = _scripted_first_poll(monkeypatch, [-1, 0, -1])
    _run(opp_stobads)
    poll = state["first"]
    assert len(poll["estimates"]) >= 3
    assert bool(poll["moves"]) == moves
    assert poll["mesh_after"] < poll["mesh_before"]


def test_certain_failure_does_not_move(monkeypatch):
    state = _scripted_first_poll(monkeypatch, [-1])
    _run(opp_stobads=True)
    poll = state["first"]
    assert poll["moves"] == []
    assert poll["mesh_after"] < poll["mesh_before"]
