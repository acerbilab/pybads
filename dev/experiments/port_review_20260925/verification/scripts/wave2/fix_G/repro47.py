import numpy as np

import pybads
from pybads.variable_transformer import VariableTransformer

print(pybads.__file__)


def show(name, vt):
    x = np.array([[10.0, 3.0]])
    print(
        name,
        vt.apply_log_t.tolist(),
        [
            a.dtype.name
            for a in (vt.orig_lb, vt.orig_ub, vt.orig_plb, vt.orig_pub)
        ],
    )
    print(
        "  u bounds",
        np.round(np.concatenate([vt.lb, vt.ub, vt.plb, vt.pub]), 6).tolist(),
    )
    print(
        "  transform",
        vt(x).tolist(),
        "inverse",
        vt.inverse_transf(vt(x)).tolist(),
    )


for dtype in (int, float):
    lb, ub = np.array([[1, -10]], dtype=dtype), np.array(
        [[1000, 10]], dtype=dtype
    )
    plb, pub = np.array([[2, -5]], dtype=dtype), np.array(
        [[500, 5]], dtype=dtype
    )
    show(
        f"{dtype.__name__}, all given",
        VariableTransformer(2, lb, ub, plb, pub),
    )
    show(
        f"{dtype.__name__}, plausible omitted", VariableTransformer(2, lb, ub)
    )

# shapes and types for other inputs, which the fix must keep
for name, args in {
    "numpy float scalars": (
        np.float64(1.0),
        np.float64(1000.0),
        np.float64(2.0),
        np.float64(500.0),
    ),
    "numpy int scalars": (
        np.int64(1),
        np.int64(1000),
        np.int64(2),
        np.int64(500),
    ),
    "1-D arrays": (
        np.array([1.0, -10.0]),
        np.array([1000.0, 10.0]),
        np.array([2.0, -5.0]),
        np.array([500.0, 5.0]),
    ),
    "float32": tuple(
        np.array(a, dtype=np.float32)
        for a in ([[1, -10]], [[1000, 10]], [[2, -5]], [[500, 5]])
    ),
}.items():
    try:
        vt = VariableTransformer(2, *args)
        print(
            name,
            "ok",
            vt.lb.shape,
            vt.orig_lb.shape,
            vt.orig_lb.dtype,
            np.round(
                np.concatenate([np.atleast_2d(vt.lb), np.atleast_2d(vt.ub)]), 6
            ).tolist(),
        )
    except Exception as e:
        print(name, "raised", type(e).__name__, e)
