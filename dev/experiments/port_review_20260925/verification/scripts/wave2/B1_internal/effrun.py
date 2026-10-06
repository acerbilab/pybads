from common import *


def f(x):
    x = np.ravel(x)
    return float((np.log(x[0]) - np.log(0.05)) ** 2 + x[1] ** 2)


b = BADS(
    f,
    np.array([0.05, 0.0]),
    np.array([1e-3, -1.0]),
    np.array([1e3, 1.0]),
    np.array([1e-2, -0.5]),
    np.array([1e2, 0.5]),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 100},
)
print(
    "x0 after setup:",
    b.x0,
    "plb:",
    b.optim_state["plb_orig"],
    "log:",
    b.var_transf.apply_log_t,
)
r = b.optimize()
print(
    "first evaluated point:",
    b.function_logger.X_orig[0],
    "f:",
    b.function_logger.Y_orig[0]
    if hasattr(b.function_logger, "Y_orig")
    else None,
)
print(
    "result x:",
    r["x"],
    "fval:",
    r["fval"],
    "func_count:",
    r["func_count"],
    "f at user x0:",
    f([0.05, 0.0]),
)
