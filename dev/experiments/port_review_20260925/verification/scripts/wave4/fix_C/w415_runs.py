"""W4-15: the Sto-BADS runs of the O verifier's v_f1_selfmove.py (whose
first and third cases contain v_f1_bitwise.py's three runs: flat D=3,
level 1 seeds 3 and 4, level 2 seed 5), without altering them: counts the
uncertain polls, the self-moves, the moves of an uncertain poll to a polled
point, the polls marked as moved and the searches' rebuilds of the local
GP, and saves every evaluated point and value, for a comparison of two
versions of PyBADS (argv[1]: the output .npz)."""
import sys

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)

import pybads.bads.bads as bm
from pybads import BADS

ST = {}
orig_lgf = bm.local_gp_fitting


def lgf(gp, u, *a, **k):
    name = sys._getframe(1).f_code.co_name
    if name == "_search_step_":
        ST["search_rebuilds"] += 1
    return orig_lgf(gp, u, *a, **k)


bm.local_gp_fitting = lgf
orig_upd = BADS._update_incumbent_


def upd(self, u_new, y, f, s):
    if sys._getframe(1).f_code.co_name == "_poll_step_":
        ST["_poll_same"] = np.array_equal(
            np.ravel(u_new), np.ravel(self.u)
        ) and (f == self.fval)
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


def poll(self, gp):
    ST["in_poll"] = True
    ST["poll_sto"] = []
    ST["_poll_same"] = None
    try:
        out = orig_poll(self, gp)
    finally:
        ST["in_poll"] = False
    best = max(ST["poll_sto"]) if ST["poll_sto"] else None
    if best == 0:
        ST["uncertain_polls"] += 1
        if ST["_poll_same"]:
            ST["self_moves"] += 1
        elif ST["_poll_same"] is False:
            ST["real_uncertain_moves"] += 1
        if self.poll_moved:
            ST["uncertain_marked"] += 1
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
        return (
            float(0.02 * np.sum(np.asarray(x) ** 2))
            + float(rng.standard_normal()),
            1.0,
        )

    return f


CASES = [
    ("flat_D3_level1", flat, 3, {"uncertainty_handling": True}),
    ("aniso_D2_level1", noisy_quad2, 2, {"uncertainty_handling": True}),
    (
        "flat_D3_level2",
        lvl2,
        3,
        {"uncertainty_handling": True, "specify_target_noise": True},
    ),
]
save = {}
for label, mk, D, extra in CASES:
    for seed in [3, 4, 5]:
        ST.update(
            in_poll=False,
            poll_sto=[],
            uncertain_polls=0,
            self_moves=0,
            real_uncertain_moves=0,
            uncertain_marked=0,
            search_rebuilds=0,
        )
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
        fl = b.function_logger
        key = f"{label}_seed{seed}"
        save[key + "_X"] = fl.X[: fl.Xn + 1].copy()
        save[key + "_Y"] = fl.Y[: fl.Xn + 1].copy()
        save[key + "_res"] = np.concatenate(
            [np.ravel(r["x"]), [r["fval"], r["fsd"], r["func_count"]]]
        )
        print(
            f"{key}: uncertain polls {ST['uncertain_polls']}, self-moves "
            f"{ST['self_moves']}, moves to a polled point "
            f"{ST['real_uncertain_moves']}, marked as moved "
            f"{ST['uncertain_marked']}; search rebuilds "
            f"{ST['search_rebuilds']}; x={np.round(r['x'], 3)} "
            f"fval={r['fval']:.4f} fsd={r['fsd']:.4f} "
            f"iters={r['iterations']} fevals={r['func_count']}",
            flush=True,
        )
np.savez(sys.argv[1], **save)
