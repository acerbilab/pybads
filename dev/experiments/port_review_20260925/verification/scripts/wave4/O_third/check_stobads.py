"""Sto-BADS (stobads=True, opp_stobads=True): count the poll outcomes, and
the polls whose 'move' (uncertain outcome, sto_poll == 0) goes to the
incumbent itself because no polled point had a positive improvement; also
the search moves on an uncertain outcome to a point estimated worse."""

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)

from pybads import BADS

rec = {
    "in_poll": False,
    "poll_moves": 0,
    "poll_self_moves": 0,
    "polls": 0,
    "outcomes": [],
    "search_uncertain_worse_moves": 0,
    "search_moves": 0,
    "in_search": False,
    "rebuild_after_self_move": 0,
}

orig_poll = BADS._poll_step_
orig_search = BADS._search_step_
orig_move = BADS._update_incumbent_
orig_rule = BADS._sto_success_improvement_


def poll(self, gp):
    rec["in_poll"] = True
    rec["polls"] += 1
    rec["cur"] = []
    try:
        out = orig_poll(self, gp)
    finally:
        rec["in_poll"] = False
    rec["outcomes"].append(max(rec["cur"]) if rec["cur"] else None)
    return out


def search(self, gp):
    rec["in_search"] = True
    try:
        return orig_search(self, gp)
    finally:
        rec["in_search"] = False


def rule(self, f_base, f_new, s_base, s_new, frame_size):
    out = orig_rule(self, f_base, f_new, s_base, s_new, frame_size)
    if rec["in_poll"]:
        rec["cur"].append(out)
    if rec["in_search"]:
        rec["last_search"] = (out, f_base - f_new)
    return out


def move(self, u_new, yval_new, fval_new, fsd_new):
    if rec["in_poll"]:
        rec["poll_moves"] += 1
        if (
            np.array_equal(np.ravel(u_new), np.ravel(self.u))
            and fval_new == self.fval
        ):
            rec["poll_self_moves"] += 1
    if rec["in_search"]:
        rec["search_moves"] += 1
        out, mu = rec.get("last_search", (None, None))
        if out == 0 and mu is not None and mu < 0:
            rec["search_uncertain_worse_moves"] += 1
    return orig_move(self, u_new, yval_new, fval_new, fsd_new)


BADS._poll_step_ = poll
BADS._search_step_ = search
BADS._sto_success_improvement_ = rule
BADS._update_incumbent_ = move

for seed in [0, 1, 2]:
    for k in (
        "poll_moves",
        "poll_self_moves",
        "polls",
        "search_uncertain_worse_moves",
        "search_moves",
    ):
        rec[k] = 0
    rec["outcomes"] = []
    D = 3
    nrng = np.random.default_rng(seed + 1000)

    def fun(x):
        return float(np.sum(np.atleast_2d(x) ** 2)) + nrng.standard_normal()

    b = BADS(
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
    )
    r = b.optimize()
    oc = rec["outcomes"]
    print(
        f"seed {seed}: fval={r['fval']:.4g} x={np.round(r['x'], 3)} evals={r['func_count']}",
        flush=True,
    )
    print(
        "  polls",
        rec["polls"],
        "outcomes (1/0/-1):",
        oc.count(1),
        oc.count(0),
        oc.count(-1),
        "| poll moves",
        rec["poll_moves"],
        "of which to the incumbent itself",
        rec["poll_self_moves"],
        "| search moves",
        rec["search_moves"],
        "of which uncertain and worse",
        rec["search_uncertain_worse_moves"],
        flush=True,
    )
