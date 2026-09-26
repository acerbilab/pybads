"""Does BADS(...) or optimize() write into the arrays the user passed?"""
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)

_check = BADS._bounds_check_
_returned = []


def _recording_check(self, *args, **kwargs):
    out = _check(self, *args, **kwargs)
    _returned[:] = out
    return out


BADS._bounds_check_ = _recording_check


def f(x):
    x = np.atleast_2d(x)
    return float((np.log10(x[0, 0]) - 1.0) ** 2 + 0.01 * x[0, 1] ** 2)


for kind, make in [
    ("float 1-D", lambda v: np.array(v, dtype=float)),
    ("float 2-D", lambda v: np.array([v], dtype=float)),
    ("int 1-D", lambda v: np.array(v, dtype=int)),
    ("int 2-D", lambda v: np.array([v], dtype=int)),
]:
    # a log-scaled variable and a linear one
    user = [make(v) for v in ([5, 3], [1, -10], [1000, 10], [2, -5], [500, 5])]
    kept = [u.copy() for u in user]
    for plausible in (True, False):
        args = user if plausible else user[:3]
        bads = BADS(
            f,
            *args,
            options={"display": "off", "random_seed": 0, "max_fun_evals": 30},
        )
        after_init = all(np.array_equal(u, k) for u, k in zip(user, kept))
        aliases = [
            name
            for name in (
                "x0",
                "lower_bounds",
                "upper_bounds",
                "plausible_lower_bounds",
                "plausible_upper_bounds",
            )
            if any(np.shares_memory(getattr(bads, name), u) for u in user)
        ]
        from_check = [
            i
            for i, r in enumerate(_returned)
            if any(np.shares_memory(r, u) for u in user)
        ]
        bads.optimize()
        after_run = all(np.array_equal(u, k) for u, k in zip(user, kept))
        print(
            f"{kind:10s} plausible={plausible!s:5s} unchanged after "
            f"BADS(...): {after_init}, after optimize(): {after_run}, "
            f"attributes sharing memory with user arrays: {aliases}, "
            f"outputs of _bounds_check_ that do: {from_check}"
        )
