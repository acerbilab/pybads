"""Sto-BADS with opp_stobads on a flat noisy target: after a poll whose
uncertain outcome 'moves' the incumbent to itself, count the later searches
of the round that rebuild the local GP only because the poll is marked as
moved (reset_gp set, not the first search of a round, no refit, no failed
rebuild)."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
import pybads.bads.bads as bm
from pybads import BADS

st = {
    "self_move_poll": False,
    "extra_rebuilds": 0,
    "self_moves": 0,
    "in_poll": False,
}
op, os_, om = BADS._poll_step_, BADS._search_step_, BADS._update_incumbent_
orig_refit = BADS._is_gp_refit_time_


def poll(self, gp):
    st["in_poll"] = True
    st["self_move_poll"] = False
    try:
        return op(self, gp)
    finally:
        st["in_poll"] = False


def move(self, u, y, f, s):
    if (
        st["in_poll"]
        and np.array_equal(np.ravel(u), np.ravel(self.u))
        and f == self.fval
    ):
        st["self_move_poll"] = True
        st["self_moves"] += 1
    return om(self, u, y, f, s)


def refit_time(self, *a, **k):
    out = orig_refit(self, *a, **k)
    st["last_refit"] = out[0]
    return out


def search(self, gp):
    pre = (
        self.reset_gp,
        self.optim_state["search_count"],
        gp.temporary_data.get("needs_rebuild", False),
        gp.temporary_data.get("needs_refit", False),
    )
    out = os_(self, gp)
    if (
        st["self_move_poll"]
        and pre[0]
        and pre[1] > 0
        and not pre[2]
        and not pre[3]
        and not st.get("last_refit", False)
    ):
        st["extra_rebuilds"] += 1
    return out


(
    BADS._poll_step_,
    BADS._search_step_,
    BADS._update_incumbent_,
    BADS._is_gp_refit_time_,
) = (poll, search, move, refit_time)
for seed in [0, 1, 2]:
    st.update(extra_rebuilds=0, self_moves=0)
    D = 3
    nrng = np.random.default_rng(seed + 1000)
    fun = (
        lambda x: float(0.01 * np.sum(np.atleast_2d(x) ** 2))
        + nrng.standard_normal()
    )
    r = BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={
            "display": "off",
            "random_seed": seed,
            "max_fun_evals": 200,
            "uncertainty_handling": True,
            "stobads": True,
            "opp_stobads": True,
        },
    ).optimize()
    print(
        f"seed {seed}: polls moved to the incumbent itself {st['self_moves']}, searches rebuilt only for that mark {st['extra_rebuilds']}",
        flush=True,
    )
