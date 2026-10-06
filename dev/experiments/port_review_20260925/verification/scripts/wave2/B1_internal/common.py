import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)


def quad(x):
    x = np.atleast_1d(x).ravel()
    return float(np.sum(x**2))


def trycall(name, f):
    try:
        r = f()
        print(f"[{name}] OK ->", r)
        return r
    except Exception as e:
        print(f"[{name}] {type(e).__name__}: {str(e)[:300]}")
