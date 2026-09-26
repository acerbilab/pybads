"""Which local_gp_fitting calls are refits, in the whole-run and search
scenarios of test_gp_update_failures.py that inject local failures."""
import gpyreg as gpr
import numpy as np

import pybads.bads.bads as bads_module
import pybads.testing.bads.test_gp_update_failures as t

original_local = bads_module.local_gp_fitting


def run(should_fail, max_fun_evals, **extra):
    injector = t.Injector(should_fail)
    saved = {m: getattr(gpr.GP, m) for m in ("update", "set_hyperparameters")}
    for m in saved:
        setattr(gpr.GP, m, injector.wrap(m))
    calls = []

    def spy(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
        n0 = injector.counts["local_gp_fitting"]
        f0 = len(injector.failed)
        out = original_local(
            gp, u, fl, options, optim_state, ih, refit_flag, rng=rng
        )
        calls.append(
            (
                refit_flag,
                injector.counts["local_gp_fitting"] - n0,
                len(injector.failed) - f0,
                out[1],
            )
        )
        return out

    bads_module.local_gp_fitting = spy
    try:
        make_fun, options = t.LEVELS[1]
        t._make_bads(
            make_fun(), max_fun_evals=max_fun_evals, **options, **extra
        ).optimize()
    finally:
        bads_module.local_gp_fitting = original_local
        for m, f in saved.items():
            setattr(gpr.GP, m, f)
    print("failed:", [c[:3] for c in injector.failed])
    print(
        "local calls with a failure (refit, eligible calls, failures, exit flag):"
    )
    print("  ", [c for c in calls if c[2] > 0 or c[3] == -2])


print("--- double_local_level1")
run(lambda call: call.site == "local_gp_fitting" and call.n in (5, 6), 150)

for sto in (False, True):
    print(f"--- search after failed rebuild, stobads={sto}, one local failure")
    state = {"armed": False, "local": 0}

    def should_fail(call):
        if call.caller != "_search_step_":
            return False
        if not state["armed"] and call.site == "add_and_update_gp":
            if call.n >= 3:
                state["armed"] = True
                return True
        elif state["armed"] and call.site == "local_gp_fitting":
            if state["local"] < 1:
                state["local"] += 1
                return True
        return False

    run(should_fail, 100, **({"stobads": True} if sto else {}))
