import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ellipsoid(x):
    x = np.ravel(x)
    D = len(x)
    return float(np.sum((10 ** (np.arange(D) / max(D - 1, 1)) * x) ** 2))


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


def box(D):
    return (-5 * np.ones(D), 5 * np.ones(D), -2 * np.ones(D), 2 * np.ones(D))
