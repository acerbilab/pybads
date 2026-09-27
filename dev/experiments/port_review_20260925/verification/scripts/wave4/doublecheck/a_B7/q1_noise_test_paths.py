"""PI's question on W4-6's completion (46af65a): does every path of
_init_mesh_ set optim_state["n_noise_test"] before a reader needs it, and
does the fit schedule's budget then count what n_eff and
eff_starting_points count? Hooks _get_gp_training_options and records, at
every call, the count, func_count, the log's points and n_eff."""
import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

orig = gpt._get_gp_training_options
calls = []


def hooked(
    optim_state,
    iteration_history,
    options,
    hyp_dict,
    gp_s_N,
    function_logger,
    second_fit=False,
):
    fl = function_logger
    calls.append(
        dict(
            n_noise_test=optim_state.get("n_noise_test", "MISSING"),
            func_count=fl.func_count,
            points=fl.Xn + 1,
            n_eff=float(np.sum(fl.n_evals[fl.X_flag])),
            esp=optim_state["eff_starting_points"],
            mfe=options["max_fun_evals"],
            ntm=options["n_train_max"],
        )
    )
    return orig(
        optim_state,
        iteration_history,
        options,
        hyp_dict,
        gp_s_N,
        fl,
        second_fit,
    )


gpt._get_gp_training_options = hooked

D = 2


def det(x):
    x = np.ravel(x)
    return float(np.sum(x**2))


def make_noisy(seed):
    rng = np.random.default_rng(seed)

    def f(x):
        return float(np.sum(np.ravel(x) ** 2)) + rng.normal()

    return f


def make_he(seed):
    rng = np.random.default_rng(seed)

    def f(x):
        return float(np.sum(np.ravel(x) ** 2)) + 0.5 * rng.normal(), 0.5

    return f


n_fun = {"n": 0}


def counting(f):
    def g(x):
        n_fun["n"] += 1
        return f(x)

    return g


cases = [
    ("det, uh None", det, {}),
    ("det, uh False", det, {"uncertainty_handling": False}),
    ("noisy, uh True", make_noisy(1), {"uncertainty_handling": True}),
    ("noisy found by test", make_noisy(1), {}),
    ("specify_target_noise", make_he(1), {"specify_target_noise": True}),
    ("det, mfe=1", det, {"max_fun_evals": 1}),
    ("noisy found, mfe=1", make_noisy(1), {"max_fun_evals": 1}),
    ("det, mfe=2", det, {"max_fun_evals": 2}),
    ("noisy found, mfe=2", make_noisy(1), {"max_fun_evals": 2}),
    ("det, mfe=3", det, {"max_fun_evals": 3}),
    ("noisy found, mfe=3", make_noisy(1), {"max_fun_evals": 3}),
    ("det, mfe=5", det, {"max_fun_evals": 5}),
    (
        "det, output_fcn stops at init",
        det,
        {"output_fcn": lambda x, s, st: st == "init"},
    ),
    (
        "noisy found, output_fcn stops at init",
        make_noisy(1),
        {"output_fcn": lambda x, s, st: st == "init"},
    ),
    ("det, fun_eval_start=0", det, {"fun_eval_start": 0}),
]
for name, f, opts in cases:
    calls.clear()
    n_fun["n"] = 0
    options = {"display": "off", "random_seed": 3, "max_fun_evals": 60}
    options.update(opts)
    bads = BADS(
        counting(f),
        np.array([0.7, -0.4]),
        -5 * np.ones(D),
        5 * np.ones(D),
        -2 * np.ones(D),
        2 * np.ones(D),
        options=options,
    )
    try:
        res = bads.optimize()
        out = f"fc={res['func_count']} calls_of_fun={n_fun['n']} it={res['iterations']}"
    except Exception as e:
        out = f"RAISED {type(e).__name__}: {e}"
    st = bads.optim_state
    print(
        f"--- {name}: level={st.get('uncertainty_handling_level')} "
        f"n_noise_test={st.get('n_noise_test', 'MISSING')} {out}"
    )
    bad = []
    for c in calls:
        # every evaluation but the noise test adds one to a row's count,
        # and each row is one point
        if c["n_noise_test"] == "MISSING":
            bad.append(("missing", c))
            continue
        if (
            c["func_count"] - c["n_noise_test"] != c["n_eff"]
            or c["n_eff"] != c["points"]
        ):
            bad.append(("count", c))
    print(f"   {len(calls)} schedule calls; mismatches: {len(bad)}")
    if calls:
        print("   first:", calls[0])
        print("   last: ", calls[-1])
    for b in bad[:3]:
        print("   MISMATCH", b)
