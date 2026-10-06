"""W3-31 and W3-39: which values of improvement_quantile and
accelerate_mesh_steps BADS refuses or accepts when it is created, at the
revision on PYTHONPATH."""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)


def make(opt, val):
    return BADS(
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        np.array([0.5, 0.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={"display": "off", opt: val},
    )


values = {
    "improvement_quantile": [
        0.5,
        0.25,
        1e-6,
        0.9,
        np.float64(0.3),
        np.float32(0.3),
        np.array([0.3]),
        np.array(0.3),
        0,
        1,
        -0.2,
        1.5,
        np.nan,
        np.inf,
        True,
        False,
        np.True_,
        None,
        "0.3",
        "3",
        1 + 0j,
        0.3 + 0j,
        np.array([0.3, 0.4]),
    ],
    "accelerate_mesh_steps": [
        3,
        1,
        10,
        3.0,
        np.int64(3),
        np.int32(2),
        np.float64(3.0),
        np.uint8(4),
        0,
        -1,
        2.5,
        0.0,
        np.inf,
        np.nan,
        True,
        False,
        np.True_,
        None,
        "3",
        np.array(3),
        np.array([3]),
        3 + 0j,
        10**30,
        1e300,
    ],
}
for opt, vals in values.items():
    print(f"--- {opt}", flush=True)
    for v in vals:
        try:
            b = make(opt, v)
            got = b.options[opt]
            print(
                f"{v!r:>28} ({type(v).__name__}): accepted -> "
                f"{got!r} ({type(got).__name__})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"{v!r:>28} ({type(v).__name__}): {type(e).__name__}: "
                f"{str(e)[:90]}",
                flush=True,
            )
