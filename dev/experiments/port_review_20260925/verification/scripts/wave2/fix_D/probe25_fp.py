"""The fingerprint's three noisy runs: the moves after the re-estimation,
and the steps that follow them."""
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
orig_re = BADS._re_evaluate_history_
orig_s = BADS._search_step_
orig_p = BADS._poll_step_
log = []
pend = {}


def re(self, gp):
    orig_re(self, gp)
    h = self.iteration_history
    pend["e"] = (
        self.optim_state["iter"],
        h.get("fval").astype(float),
        [np.ravel(u) for u in h.get("u")],
        self.function_logger.func_count,
    )


def chk(self, step):
    if "e" in pend:
        it, fv, u, fc = pend.pop("e")
        moved = self.fval != fv[it]
        log.append((it, fc, step, bool(moved)))


def s(self, gp):
    chk(self, "search")
    return orig_s(self, gp)


def p(self, gp):
    chk(self, "poll")
    return orig_p(self, gp)


BADS._re_evaluate_history_ = re
BADS._search_step_ = s
BADS._poll_step_ = p
g = np.random.default_rng(0)


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        log.clear()
        pend.clear()
        r = BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        ).optimize()
        if noisy:
            print(
                seed,
                r["iterations"],
                r["func_count"],
                log,
                "left:",
                pend.get("e", (None,))[0],
            )
