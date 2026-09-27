import numpy as np

import pybads
from pybads.testing.bads.test_bads_seed import D, _initial_design
from pybads.testing.init_functions.test_init_sobol import _design

print(pybads.__file__)
print(
    "BADS: seeds 42 and 43 give the same design:",
    np.array_equal(
        _initial_design(42, np.ones(D) * 4),
        _initial_design(43, np.ones(D) * 4),
    ),
)
print(
    "BADS: interior starts, seed 42, same design:",
    np.array_equal(
        _initial_design(42, np.ones(D) * 4),
        _initial_design(42, np.array([-5.0, 0.0, 10.0])),
    ),
)
print(
    "init_sobol: rng 0 and 1 give the same design:",
    np.array_equal(_design(3, 3, rng=0)[0], _design(3, 3, rng=1)[0]),
)
