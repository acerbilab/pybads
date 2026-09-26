import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
try:
    b = BADS(
        lambda x: 0.0,
        x0=np.array([0.0003]),
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
        plausible_lower_bounds=np.array([0.0001]),
        plausible_upper_bounds=np.array([0.0005]),
        options={"display": "off", "random_seed": 0},
    )
    print("accepted", b.var_transf.orig_plb, b.var_transf.orig_pub)
except Exception as e:
    print(type(e).__name__, str(e).strip()[:150])
