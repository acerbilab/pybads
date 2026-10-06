"""F1: Sto-BADS with opp_stobads. Count polls whose outcome is uncertain
(best Sto outcome 0) and whose 'move' goes to the incumbent itself; count the
search rebuilds of the local GP that nothing but that mark asks for (no refit,
not the first search of a round, no failed-rebuild marker, no search move
since the last rebuild); and compare the run with a variant in which such a
poll is not marked as moved.

Rebuild causes are read from the caller's frame of local_gp_fitting."""
import sys

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)

import pybads.bads.bads as bm
from pybads import BADS

ST = {}


def reset_state():
    ST.update(
        in_poll=False,
        poll_sto=[],
        uncertain_polls=0,
        self_move_polls=0,
        real_uncertain_moves=0,
        last_poll_self_move=False,
        search_move_pending=False,
        search_rebuilds=0,
        extra_search_rebuilds=0,
        poll_rebuilds=0,
    )


orig_lgf = bm.local_gp_fitting


def lgf(gp, u, *a, **k):
    fr = sys._getframe(1)
    name = fr.f_code.co_name
    if name == "_search_step_":
        L = fr.f_locals
        self = L["self"]
        ST["search_rebuilds"] += 1
        refit = bool(L["refit_flag"])
        first = self.optim_state["search_count"] == 0
        nr = bool(gp.temporary_data.get("needs_rebuild", False))
        if (
            not refit
            and not first
            and not nr
            and self.reset_gp
            and not ST["search_move_pending"]
            and ST["last_poll_self_move"]
        ):
            ST["extra_search_rebuilds"] += 1
    elif name == "_poll_step_":
        ST["poll_rebuilds"] += 1
    ST["search_move_pending"] = False
    return orig_lgf(gp, u, *a, **k)


bm.local_gp_fitting = lgf

orig_upd = BADS._update_incumbent_


def upd(self, u_new, y, f, s):
    name = sys._getframe(1).f_code.co_name
    if name == "_search_step_":
        ST["search_move_pending"] = True
    if name == "_poll_step_":
        same = np.array_equal(np.ravel(u_new), np.ravel(self.u)) and (
            f == self.fval
        )
        ST["_poll_same"] = same
    return orig_upd(self, u_new, y, f, s)


BADS._update_incumbent_ = upd

orig_sto = BADS._sto_success_improvement_


def sto(self, *a, **k):
    out = orig_sto(self, *a, **k)
    if ST["in_poll"]:
        ST["poll_sto"].append(out)
    return out


BADS._sto_success_improvement_ = sto

orig_poll = BADS._poll_step_
FIX = {"on": False}


def poll(self, gp):
    ST["in_poll"] = True
    ST["poll_sto"] = []
    ST["_poll_same"] = None
    try:
        out = orig_poll(self, gp)
    finally:
        ST["in_poll"] = False
    best = max(ST["poll_sto"]) if ST["poll_sto"] else None
    self_move = False
    if best == 0:
        ST["uncertain_polls"] += 1
        if ST["_poll_same"]:
            ST["self_move_polls"] += 1
            self_move = True
        elif ST["_poll_same"] is False:
            ST["real_uncertain_moves"] += 1
    if self_move and FIX["on"]:
        self.poll_moved = False
    ST["last_poll_self_move"] = self_move and not FIX["on"]
    return out


BADS._poll_step_ = poll


def flat(seed):
    rng = np.random.default_rng(10_000 + seed)
    return lambda x: float(0.01 * np.sum(np.asarray(x) ** 2)) + float(
        rng.standard_normal()
    )


def noisy_quad2(seed):
    rng = np.random.default_rng(20_000 + seed)
    return lambda x: float(
        np.sum((np.asarray(x) - 0.5) ** 2 * np.array([1.0, 0.1]))
    ) + 0.5 * float(rng.standard_normal())


def lvl2(seed):
    rng = np.random.default_rng(30_000 + seed)

    def f(x):
        sd = 1.0
        return (
            float(0.02 * np.sum(np.asarray(x) ** 2))
            + sd * float(rng.standard_normal()),
            sd,
        )

    return f


CASES = [
    ("flat D=3 level1", flat, 3, {"uncertainty_handling": True}),
    ("aniso D=2 level1", noisy_quad2, 2, {"uncertainty_handling": True}),
    (
        "flat D=3 level2",
        lvl2,
        3,
        {"uncertainty_handling": True, "specify_target_noise": True},
    ),
]

for label, mk, D, extra in CASES:
    for seed in [3, 4, 5]:
        res = {}
        for fix in [False, True]:
            FIX["on"] = fix
            reset_state()
            opts = {
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 200,
                "stobads": True,
                "opp_stobads": True,
            }
            opts.update(extra)
            b = BADS(
                mk(seed),
                3.0 * np.ones(D),
                -50 * np.ones(D),
                50 * np.ones(D),
                -10 * np.ones(D),
                10 * np.ones(D),
                options=opts,
            )
            r = b.optimize()
            res[fix] = (r, dict(ST))
        (r0, s0), (r1, s1) = res[False], res[True]
        print(
            f"{label} seed {seed}: uncertain polls {s0['uncertain_polls']}, "
            f"self-moves {s0['self_move_polls']}, real uncertain moves "
            f"{s0['real_uncertain_moves']}; search rebuilds {s0['search_rebuilds']} "
            f"(extra, only for the mark: {s0['extra_search_rebuilds']}) vs "
            f"{s1['search_rebuilds']} unmarked",
            flush=True,
        )
        print(
            f"    as is:    x={np.round(r0['x'], 3)} fval={r0['fval']:.4f} "
            f"fsd={r0['fsd']:.4f} iters={r0['iterations']} fevals={r0['func_count']}",
            flush=True,
        )
        print(
            f"    unmarked: x={np.round(r1['x'], 3)} fval={r1['fval']:.4f} "
            f"fsd={r1['fsd']:.4f} iters={r1['iterations']} fevals={r1['func_count']}",
            flush=True,
        )
