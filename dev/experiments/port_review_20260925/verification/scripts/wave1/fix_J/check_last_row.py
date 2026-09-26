"""At each call of `_re_evaluate_history_` whose loop runs, is the last row
of the history the incumbent's (u, yval, fval, fsd)? Over noisy runs at
levels 1 and 2, seeds 0-9, with and without failures of the re-estimate's
rebuild (every 3rd rebuild of the re-estimate fails)."""
import sys

import gpyreg as gpr
import numpy as np

from pybads import BADS
from pybads.testing.bads.test_gp_update_failures import (
    Injector,
    _make_bads,
    _noisy_sphere,
    _noisy_sphere_with_sd,
)

orig = {m: getattr(gpr.GP, m) for m in ("update", "set_hyperparameters")}
orig_re = BADS._re_evaluate_history_
counts = {}
bad = []


def re_evaluate(self, gp):
    line = sys._getframe(1).f_lineno
    runs = self.optim_state["last_re_eval"] != self.function_logger.func_count
    h = self.iteration_history
    if runs:
        n = h.get("u").shape[0]
        same = (
            np.array_equal(h.get("u")[-1], np.ravel(self.u))
            and h.get("yval")[-1] == float(self.yval)
            and h.get("fval")[-1] == self.fval
            and h.get("fsd")[-1] == self.fsd
            and n - 1 == self.optim_state["iter"]
        )
        counts[line] = counts.get(line, 0) + 1
        if not same:
            bad.append((line, n))
    return orig_re(self, gp)


BADS._re_evaluate_history_ = re_evaluate
for inject in (False, True):
    for level in (1, 2):
        for seed in range(10):
            n_calls = {"k": 0}

            def should_fail(call):
                if inject and call.caller == "_re_evaluate_history_":
                    n_calls["k"] += 1
                    return n_calls["k"] % 3 == 0
                return False

            inj = Injector(should_fail)
            for m in orig:
                setattr(gpr.GP, m, inj.wrap(m))
            opts = {"uncertainty_handling": True}
            fun = _noisy_sphere(seed)
            if level == 2:
                opts["specify_target_noise"] = True
                fun = _noisy_sphere_with_sd(seed)
            try:
                r = _make_bads(
                    fun, max_fun_evals=150, seed=seed, **opts
                ).optimize()
                ok = np.isfinite(r["fval"])
            except Exception as e:
                ok = type(e).__name__
            finally:
                for m, f in orig.items():
                    setattr(gpr.GP, m, f)
            print(
                inject,
                level,
                seed,
                "finite" if ok is True else ok,
                len(inj.failed),
            )
print("calls by line:", counts)
print("mismatches:", bad)
