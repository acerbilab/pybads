"""The real-valued check still refuses complex inputs; float32 inputs."""
import warnings

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
f = lambda x: float(np.sum(np.atleast_2d(x) ** 2))
o = {"display": "off", "random_seed": 0}
b = (-5 * np.ones(2), 5 * np.ones(2), -2 * np.ones(2), 2 * np.ones(2))
try:
    BADS(f, np.array([0.5 + 1j, 0.2]), *b, options=o)
    print("complex x0: accepted")
except ValueError as err:
    print("complex x0: ValueError", str(err).split()[0:6])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    bads = BADS(f, np.array([0.5 + 0j, 0.2]), *b, options=o)
    print(
        "complex x0, zero imaginary part: accepted, x0",
        bads.x0,
        bads.x0.dtype,
        [str(x.category.__name__) for x in w],
    )
bads = BADS(
    f,
    np.array([0.5, 0.2], dtype=np.float32),
    *[x.astype(np.float32) for x in b],
    options=o,
)
print(
    "float32:",
    bads.x0.dtype,
    bads.optim_state["lb_orig"].dtype,
    bads.optim_state["plb"].dtype,
)
