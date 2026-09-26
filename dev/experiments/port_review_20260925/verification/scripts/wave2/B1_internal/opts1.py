from common import *

base = {"display": "off", "random_seed": 0, "max_fun_evals": 40}
x0 = np.array([[0.3, -0.2]])
lb = np.array([[-2.0, -2.0]])
ub = np.array([[2.0, 2.0]])


def run(
    extra,
    ret=lambda r, b: (r["func_count"], r["iterations"], r["message"][:60]),
):
    def f():
        b = BADS(quad, x0.copy(), lb, ub, options=dict(base, **extra))
        r = b.optimize()
        return ret(r, b)

    return f


trycall("f_vals given", run({"f_vals": [quad(x0)]}))
for k, v in [
    ("max_fun_evals", None),
    ("max_iter", None),
    ("tol_mesh", None),
    ("tol_fun", None),
    ("noise_final_samples", None),
    ("nonlinear_scaling", None),
    ("fun_eval_start", None),
    ("display", None),
    ("uncertainty_handling", None),
    ("noise_size", None),
    ("specify_target_noise", None),
    ("tol_stall_iters", None),
    ("poll_mesh_multiplier", None),
    ("search_n_try", None),
    ("cache_size", None),
    ("init_mesh_size_integer", None),
    ("complete_poll", None),
    ("accelerate_mesh", None),
]:
    trycall(f"{k}=None", run({k: v}))
# expressions
trycall("max_fun_evals='20*D'", run({"max_fun_evals": "20*D"}))
trycall("max_iter='2*D'", run({"max_iter": "2*D"}))
# strings for booleans
trycall(
    "uncertainty_handling='off'",
    run(
        {"uncertainty_handling": "off"},
        lambda r, b: (
            b.optim_state["uncertainty_handling_level"],
            r["target_type"],
        ),
    ),
)
trycall(
    "uncertainty_handling=0",
    run(
        {"uncertainty_handling": 0},
        lambda r, b: (
            b.optim_state["uncertainty_handling_level"],
            r["target_type"],
        ),
    ),
)
trycall(
    "nonlinear_scaling='off' (level)",
    lambda: BADS(
        quad,
        np.array([[1.0, 0.0]]),
        np.array([[1e-3, -1]]),
        np.array([[1e3, 1]]),
        np.array([[0.1, -0.5]]),
        np.array([[100, 0.5]]),
        options={"display": "off", "nonlinear_scaling": "off"},
    ).var_transf.apply_log_t,
)
trycall(
    "nonlinear_scaling=False",
    lambda: BADS(
        quad,
        np.array([[1.0, 0.0]]),
        np.array([[1e-3, -1]]),
        np.array([[1e3, 1]]),
        np.array([[0.1, -0.5]]),
        np.array([[100, 0.5]]),
        options={"display": "off", "nonlinear_scaling": False},
    ).var_transf.apply_log_t,
)
# float for integer options
trycall("max_fun_evals=40.0", run({"max_fun_evals": 40.0}))
trycall(
    "noise_final_samples=2.0 noisy",
    run({"noise_final_samples": 2.0, "uncertainty_handling": True}),
)
# unknown option
trycall("unknown option", run({"maxfunevals": 10}))
# advanced option derived from a user option (tol_noise from tol_fun)
b = BADS(quad, x0.copy(), lb, ub, options={"display": "off", "tol_fun": 1e-2})
print(
    "tol_noise with tol_fun=1e-2:",
    b.options["tol_noise"],
    " hedge_beta:",
    b.options["hedge_beta"],
)
b = BADS(quad, x0.copy(), lb, ub, options={"display": "off"})
print(
    "default tol_noise:",
    b.options["tol_noise"],
    "hedge_beta:",
    b.options["hedge_beta"],
)
# descriptions
print("descr noise_size:", repr(b.options.descriptions["noise_size"]))
print("descr periodic_vars:", repr(b.options.descriptions["periodic_vars"]))
print(
    "descr stobads_frame_size_scaling_power:",
    repr(b.options.descriptions["stobads_frame_size_scaling_power"]),
)
print("descr n_basis:", repr(b.options.descriptions["n_basis"]))
