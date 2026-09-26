from common import *

x0 = np.array([0.3, -0.2])
lb = np.array([-2.0, -2.0])
ub = np.array([2.0, 2.0])
b = BADS(
    quad,
    x0.copy(),
    lb,
    ub,
    options={"display": "off", "random_seed": 0, "max_fun_evals": 20},
)
r = b.optimize()
print(
    "fsd",
    repr(r["fsd"]),
    "fval",
    repr(r["fval"]),
    "x",
    repr(r["x"]),
    "x0",
    repr(r["x0"]),
    "mesh",
    repr(r["mesh_size"]),
    "iterations",
    repr(r["iterations"]),
    "func_count",
    repr(r["func_count"]),
)
print(
    "evaluated start point:",
    b.function_logger.X_orig[0],
    " result x0:",
    r["x0"],
)
print(
    "overhead",
    r["overhead"],
    "total_time",
    r["total_time"],
    "version",
    r["version"],
)
# tol_mesh placement on powers of two
for k in range(1, 41):
    t = 2.0**-k
    v = 2.0 ** np.ceil(np.log(t) / np.log(2.0))
    if v != t:
        print("tol_mesh 2^-%d placed at" % k, v)
