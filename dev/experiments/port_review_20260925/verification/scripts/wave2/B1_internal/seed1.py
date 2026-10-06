from common import *

x0 = np.array([[0.3, -0.2]])
lb = np.array([[-2.0, -2.0]])
ub = np.array([[2.0, 2.0]])


def mk(seed):
    b = BADS(
        quad,
        x0.copy(),
        lb,
        ub,
        options={"display": "off", "random_seed": seed},
    )
    return b


for seed in [
    True,
    False,
    -1,
    2**70,
    np.float32(3.0),
    np.float64(3.0),
    3.0,
    3.5,
    float("nan"),
    float("inf"),
    [1, 2],
    (1, 2),
    np.array([5]),
    np.uint8(7),
    "3",
    np.random.SeedSequence(3),
    np.random.PCG64(3),
]:

    def f():
        b = mk(seed)
        return (
            b.optim_state["random_seed"],
            type(b.optim_state["random_seed"]).__name__,
            b.rng.integers(1000),
        )

    trycall(f"seed={seed!r}", f)
# does bool True give the same stream as 1?
print("True vs 1:", mk(True).rng.integers(1000), mk(1).rng.integers(1000))
# change of option after construction has no effect
b = mk(5)
b.options["random_seed"] = 6
print("after change:", b.rng.integers(1000), mk(5).rng.integers(1000))
# The result reports the seed
b = BADS(
    quad,
    x0.copy(),
    lb,
    ub,
    options={"display": "off", "random_seed": 9.0, "max_fun_evals": 20},
)
r = b.optimize()
print("result random_seed:", r["random_seed"], type(r["random_seed"]).__name__)
print("result keys:", sorted(dict.keys(r)))
trycall("result status", lambda: r["status"])
trycall("result .status", lambda: r.status)
print("success:", r["success"], "message:", r["message"])
print(
    "x:",
    r["x"],
    r["x"].shape,
    "x0:",
    r["x0"],
    "fval:",
    r["fval"],
    type(r["fval"]).__name__,
    "fsd:",
    r["fsd"],
    "iterations:",
    r["iterations"],
    "func_count:",
    r["func_count"],
    "mesh:",
    r["mesh_size"],
)
print(
    "problem_type:",
    r["problem_type"],
    "target_type:",
    r["target_type"],
    "yval_vec:",
    r["yval_vec"],
    "ysd_vec:",
    r["ysd_vec"],
)
print("fval equals logged f at x?", quad(r["x"]), r["fval"])
