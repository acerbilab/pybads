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


def _scripted_first_poll(monkeypatch, outcomes, improvements=None):
    """Replace the Sto-BADS rule by `outcomes`, and the improvement of each
    polled point over the incumbent by `improvements` if given, point after
    point, in the first poll, and record that poll: the estimates the rule
    received, the mesh size integer before and after, the estimates the
    incumbent moved to, the incumbent before and after, and whether the
    poll was marked as moved."""
    state = {
        "in_poll": False,
        "estimates": [],
        "improvements": [],
        "first": None,
        "moves": [],
    }
    original_poll = BADS._poll_step_
    original_rule = BADS._sto_success_improvement_
    original_improvement = BADS._eval_improvement_
    original_move = BADS._update_incumbent_

    def poll(self, gp):
        first = state["first"] is None
        state["in_poll"] = first
        before = self.mesh_size_integer
        u_before = self.u.copy()
        try:
            return original_poll(self, gp)
        finally:
            if first:
                state["first"] = {
                    "estimates": list(state["estimates"]),
                    "improvements": list(state["improvements"]),
                    "mesh_before": before,
                    "mesh_after": self.mesh_size_integer,
                    "moves": list(state["moves"]),
                    "u_before": u_before,
                    "u_after": self.u.copy(),
                    "marked": self.poll_moved,
                }
            state["in_poll"] = False

    def improvement(self, *args):
        z = original_improvement(self, *args)
        if state["in_poll"]:
            if improvements is not None:
                n = len(state["improvements"])
                z = np.array([improvements[min(n, len(improvements) - 1)]])
            state["improvements"].append(float(np.ravel(z)[0]))
        return z

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
    monkeypatch.setattr(BADS, "_eval_improvement_", improvement)
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
    """No success and some uncertain point, which improves on the
    incumbent: the mesh contracts, and the incumbent moves to that point,
    and the poll is marked as moved, only with `opp_stobads`."""
    state = _scripted_first_poll(
        monkeypatch, [-1, 0, -1], improvements=[-1.0, 0.5, -1.0]
    )
    _run(opp_stobads)
    poll = state["first"]
    assert len(poll["estimates"]) >= 3
    assert poll["moves"] == ([poll["estimates"][1]] if moves else [])
    assert (not np.array_equal(poll["u_after"], poll["u_before"])) == moves
    assert poll["marked"] == moves
    assert poll["mesh_after"] < poll["mesh_before"]


def test_uncertain_poll_without_improvement_does_not_move(monkeypatch):
    """No success, some uncertain point, and every polled point estimated
    worse than the incumbent: with `opp_stobads`, the incumbent stays, and
    the poll is not marked as moved, so the next searches do not rebuild
    the local GP for it."""
    state = _scripted_first_poll(monkeypatch, [0], improvements=[-1.0])
    _run(opp_stobads=True)
    poll = state["first"]
    assert len(poll["estimates"]) >= 3
    assert poll["moves"] == []
    assert np.array_equal(poll["u_after"], poll["u_before"])
    assert not poll["marked"]
    assert poll["mesh_after"] < poll["mesh_before"]


def test_certain_failure_does_not_move(monkeypatch):
    state = _scripted_first_poll(monkeypatch, [-1])
    _run(opp_stobads=True)
    poll = state["first"]
    assert poll["moves"] == []
    assert poll["mesh_after"] < poll["mesh_before"]
