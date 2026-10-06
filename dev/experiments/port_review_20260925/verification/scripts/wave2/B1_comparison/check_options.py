import logging
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)


def sph(x):
    return float(np.sum(np.ravel(x) ** 2))


def run(label, opts, optimize=True, D=2):
    o = {"random_seed": 0, "display": "off"}
    o.update(opts)
    try:
        b = BADS(
            sph,
            x0=np.ones(D),
            lower_bounds=-5 * np.ones(D),
            upper_bounds=5 * np.ones(D),
            plausible_lower_bounds=-2 * np.ones(D),
            plausible_upper_bounds=2 * np.ones(D),
            options=o,
        )
    except Exception as e:
        print(f"[{label}] BADS(): {type(e).__name__}: {str(e)[:110]}")
        return None
    info = {k: b.options[k] for k in opts if k in b.options}
    if not optimize:
        print(
            f"[{label}] created; options={info} level={b.optim_state['uncertainty_handling_level']} logflag={b.var_transf.apply_log_t.ravel()} logger level={logging.getLogger('BADS').level}"
        )
        return b
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = b.optimize()
        print(
            f"[{label}] ran: func_count={r['func_count']} iterations={r['iterations']} fval={r['fval']:.3g} msg={r['message'][:60]}"
        )
    except Exception as e:
        print(f"[{label}] optimize(): {type(e).__name__}: {str(e)[:110]}")
    return b


# user None for options with non-None defaults (MATLAB: empty -> default)
for k in (
    "max_fun_evals",
    "tol_mesh",
    "tol_fun",
    "max_iter",
    "noise_final_samples",
    "nonlinear_scaling",
    "display",
    "fun_eval_start",
    "search_n_try",
):
    run(
        f"{k}=None",
        {k: None, "max_fun_evals": 40} if k != "max_fun_evals" else {k: None},
    )
# string expressions (MATLAB evaluates them)
run("max_fun_evals='50*D'", {"max_fun_evals": "50*D"})
run("nonlinear_scaling='off'", {"nonlinear_scaling": "off"}, optimize=False)
run(
    "uncertainty_handling='off'",
    {"uncertainty_handling": "off"},
    optimize=False,
)
run("uncertainty_handling=0", {"uncertainty_handling": 0}, optimize=False)
run(
    "specify_target_noise=None", {"specify_target_noise": None}, optimize=False
)
# values MATLAB refuses
run("max_fun_evals=0", {"max_fun_evals": 0})
run("max_fun_evals=-5", {"max_fun_evals": -5})
run("max_fun_evals=30.5", {"max_fun_evals": 30.5})
run(
    "improvement_quantile=0.9 (MATLAB warns)",
    {"improvement_quantile": 0.9, "max_fun_evals": 40},
)
# display levels
for d in ("final", "notify", "off", "iter", "none", "OFF"):
    b = run(f"display={d!r}", {"display": d}, optimize=False)
# misspelt option
run("misspelt", {"max_fun_eval": 10}, optimize=False)
import os

# .ini descriptions truncated by ':' / '='
from pybads.bads.options import _read_config_file

root = os.path.dirname(pybads.__file__)
for p in ("basic", "advanced"):
    arr = _read_config_file(f"{root}/bads/option_configs/{p}_bads_options.ini")
    for k, v, d in arr:
        if k in (
            "noise_size",
            "stobads_frame_size_scaling_power",
            "periodic_vars",
            "output_fcn",
            "random_seed",
        ):
            print(f"description[{k}] = {d!r}")
