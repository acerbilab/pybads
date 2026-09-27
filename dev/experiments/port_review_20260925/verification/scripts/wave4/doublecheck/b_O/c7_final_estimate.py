"""W4-26 (the "done" call's optim_state and iteration_history at the chosen
iterate) and W4-30 (fsd of a noisy run stopped in its initialization)."""
import warnings

import hdr  # noqa
import numpy as np

from pybads import BADS

warnings.simplefilter("ignore")
D = 2
lb, ub = -10 * np.ones((1, D)), 10 * np.ones((1, D))
plb, pub = -5 * np.ones((1, D)), 5 * np.ones((1, D))
x0 = np.ones((1, D))


def make_fun(level, seed, sd=0.7):
    rng = np.random.default_rng(seed + 100)

    def fun(x):
        f = float(np.sum(np.ravel(x) ** 2)) + sd * rng.normal()
        return (f, sd) if level == 2 else f

    return fun


def run(level, seed, max_fun_evals, extra=None, stop_at=None):
    cap = {}

    def out(x, s, st):
        cap[st] = s
        return st == stop_at

    o = {
        "display": "off",
        "random_seed": seed,
        "max_fun_evals": max_fun_evals,
        "output_fcn": out,
    }
    if level == 1:
        o["uncertainty_handling"] = True
    elif level == 2:
        o["uncertainty_handling"] = True
        o["specify_target_noise"] = True
    if extra:
        o.update(extra)
    b = BADS(make_fun(level, seed), x0, lb, ub, plb, pub, options=o)
    r = b.optimize()
    return b, r, cap


print("-- W4-26")
for level in (1, 2):
    for seed, mfe in ((0, 120), (1, 150), (2, 38), (3, 40)):
        b, r, cap = run(level, seed, mfe)
        s = cap["done"]
        u_res = b.var_transf(r["x"]) if False else b.u
        ok_state = (
            np.allclose(np.ravel(s["u"]), np.ravel(b.u))
            and s["fval"] == r["fval"]
            and s["fsd"] == r["fsd"]
            and s["yval"] == b.yval
        )
        H = b.iteration_history
        us = np.array(
            [np.ravel(u) for u in H.get("u")[: b.optim_state["iter"] + 1]]
        )
        idx = [
            i for i in range(len(us)) if np.array_equal(us[i], np.ravel(b.u))
        ]
        # the chosen iterate: MATLAB skips iterate 0 unless it is the only one
        ok_hist = any(
            H.get("fval")[i] == r["fval"] and H.get("fsd")[i] == r["fsd"]
            for i in idx
        )
        print(
            f"level {level} seed {seed} mfe {mfe}: iterations {r['iterations']}, func_count "
            f"{r['func_count']}, n_final {np.size(r['yval_vec'])}; done-state == result: {ok_state}; "
            f"history at iterates {idx} holds result fval/fsd: {ok_hist}"
        )

print("-- W4-30: stopped at init")
for level in (1, 2):
    for ns in (None, 2.5, [2.5, 1.0]):
        extra = {} if ns is None else {"noise_size": ns}
        b, r, cap = run(level, 0, 200, extra, stop_at="init")
        print(
            f"level {level} noise_size={ns!r}: iterations {r['iterations']} fsd={r['fsd']} "
            f"yval_vec={r['yval_vec']} ysd_vec={r['ysd_vec']} S at incumbent="
            f"{b.function_logger.S[np.argmin(b.function_logger.Y[:b.function_logger.Xn+1])] if level == 2 else None}"
        )
# level 1 through the noise test (uncertainty_handling left empty)
cap = {}
b = BADS(
    make_fun(1, 0),
    x0,
    lb,
    ub,
    plb,
    pub,
    options={
        "display": "off",
        "random_seed": 0,
        "output_fcn": lambda x, s, st: st == "init",
    },
)
r = b.optimize()
print(
    f"noise test finds noise, stopped at init: level {b.optim_state['uncertainty_handling_level']} "
    f"fsd={r['fsd']} yval_vec={r['yval_vec']}"
)
print("-- a noisy run whose initialization uses up the budget")
for level in (1, 2):
    for mfe in (3, 10):
        b, r, cap = run(level, 0, mfe)
        print(
            f"level {level} max_fun_evals {mfe}: iterations {r['iterations']} func_count {r['func_count']} "
            f"fsd={r['fsd']} yval_vec={r['yval_vec']} msg={r['message'][:60]!r}"
        )
