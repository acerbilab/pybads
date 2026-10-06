from common import *

nbc = lambda x: np.sum(np.atleast_2d(x) ** 2, axis=1) > 1
fails = 0
for s in range(20):
    try:
        BADS(
            quad,
            None,
            np.array([-2.0, -2.0]),
            np.array([2.0, 2.0]),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            non_box_cons=nbc,
            options={"display": "off", "random_seed": s},
        )
    except ValueError as e:
        fails += 1
print("random x0 with non_box_cons: refused in", fails, "of 20 seeds")
trycall(
    "non_box_cons returning a scalar",
    lambda: BADS(
        quad,
        np.array([0.1, 0.1]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        non_box_cons=lambda x: np.sum(x**2) > 100,
        options={"display": "off"},
    ),
)
trycall(
    "non_box_cons returning python bool",
    lambda: BADS(
        quad,
        np.array([0.1, 0.1]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        non_box_cons=lambda x: bool(np.sum(x**2) > 100),
        options={"display": "off"},
    ),
)
