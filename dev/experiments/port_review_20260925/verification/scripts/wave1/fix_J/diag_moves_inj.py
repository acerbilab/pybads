"""Moves of the end-of-iteration choice while the re-estimate fails at row 1
and at the last row (the test's injection), over seeds and budgets."""
import sys

import gpyreg as gpr
import numpy as np

from pybads import BADS
from pybads.testing.bads.test_gp_update_failures import (
    Injector,
    _make_bads,
    _noisy_sphere,
)

budget = int(sys.argv[1])
seeds = range(int(sys.argv[2]))
orig = {m: getattr(gpr.GP, m) for m in ("update", "set_hyperparameters")}
orig_re = BADS._re_evaluate_history_
orig_eval = BADS._eval_improvement_

for seed in seeds:
    state = {"armed": False, "row": 0, "rows": 0}

    def should_fail(call):
        if not state["armed"] or call.caller != "_re_evaluate_history_":
            return False
        row = state["row"]
        state["row"] += 1
        return row in (1, state["rows"] - 1)

    inj = Injector(should_fail)
    for m in orig:
        setattr(gpr.GP, m, inj.wrap(m))
    stats = {"choices": 0, "nan_moves": 0, "zmax": []}

    def re_evaluate(self, gp):
        rows = self.iteration_history.get("u").shape[0]
        state.update(armed=rows >= 3, row=0, rows=rows)
        try:
            orig_re(self, gp)
        finally:
            state["armed"] = False

    def evaluate(self, f_base, f_new, s_base, s_new, q):
        z = orig_eval(self, f_base, f_new, s_base, s_new, q)
        if sys._getframe(1).f_code.co_name == "optimize" and np.ndim(f_new):
            stats["choices"] += 1
            zz = z[1:]
            m = np.nanmax(zz)
            stats["zmax"].append(round(float(m), 4))
            if np.any(np.isnan(zz)) and m > 1e-3:
                stats["nan_moves"] += 1
        return z

    BADS._re_evaluate_history_ = re_evaluate
    BADS._eval_improvement_ = evaluate
    try:
        r = _make_bads(
            _noisy_sphere(),
            max_fun_evals=budget,
            seed=seed,
            uncertainty_handling=True,
            noise_final_samples=0,
        ).optimize()
    finally:
        BADS._re_evaluate_history_ = orig_re
        BADS._eval_improvement_ = orig_eval
        for m, f in orig.items():
            setattr(gpr.GP, m, f)
    print(
        seed,
        stats["choices"],
        stats["nan_moves"],
        stats["zmax"],
        len(inj.failed),
    )
