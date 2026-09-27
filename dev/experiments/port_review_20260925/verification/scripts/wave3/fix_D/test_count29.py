import numpy as np
import pytest

from pybads import BADS

D = 3


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _make_bads(fun=_sphere, **options):
    opts = {"display": "off", "max_fun_evals": 80, "random_seed": 3}
    opts.update(options)
    return BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    )


def test_rebuilds_of_the_local_gp_after_a_move(monkeypatch):
    """A move of the incumbent asks for a rebuild of the local GP, as MATLAB
    BADS empties the posterior: after a poll that moves the incumbent, each
    search of the next rounds rebuilds it, until a poll that does not move
    (`bads.m:1049`, at the end of every pass); otherwise the first search of
    a round rebuilds it, as the poll does at its first step, and a later
    search only after a search that moved the incumbent, or for a refit. On
    Rosenbrock's function some polls move (on the sphere, none)."""
    import pybads.bads.bads as bads_module

    def rosenbrock(x):
        x = np.ravel(x)
        return float(
            np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        )

    original_fitting = bads_module.local_gp_fitting
    steps = []
    refits = []

    def fitting(*args, **kwargs):
        refits.append(args[6])  # refit_flag
        return original_fitting(*args, **kwargs)

    def step(original, kind):
        def wrapper(self, gp):
            first = self.optim_state["search_count"] == 0
            u_best, n_fits = self.u_best.copy(), len(refits)
            out = original(self, gp)
            moved = not np.array_equal(u_best, self.u_best)
            steps.append((kind, first, moved, refits[n_fits:]))
            return out

        return wrapper

    monkeypatch.setattr(bads_module, "local_gp_fitting", fitting)
    monkeypatch.setattr(
        BADS, "_search_step_", step(BADS._search_step_, "search")
    )
    monkeypatch.setattr(BADS, "_poll_step_", step(BADS._poll_step_, "poll"))
    _make_bads(rosenbrock, max_fun_evals=100).optimize()

    # The searches that only a poll's move asks to rebuild the local GP, and
    # those that nothing asks to
    after_poll_move, unasked = 0, 0
    poll_moved = None
    search_moved = False
    for kind, first, moved, step_refits in steps:
        if kind == "poll":
            poll_moved = moved
        elif poll_moved is not None:
            if first or search_moved or poll_moved:
                assert len(step_refits) == 1
            else:
                assert all(step_refits)  # a refit only
            if not first and not search_moved and not any(step_refits):
                if poll_moved:
                    after_poll_move += 1
                else:
                    unasked += 1
        search_moved = kind == "search" and moved
    print(
        "COUNTS",
        after_poll_move,
        unasked,
        sum(1 for s in steps if s[0] == "poll" and s[2]),
    )
