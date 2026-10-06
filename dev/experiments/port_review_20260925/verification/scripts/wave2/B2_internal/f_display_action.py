import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


class P(BADS):
    def _display_function_log_(self, iteration, method):
        if method in ("Refine grid", "Successful poll"):
            shown = "".join(self.logging_action[-1])
            self.rows = getattr(self, "rows", [])
            self.rows.append(
                (
                    iteration,
                    method,
                    shown,
                    self.gp_refitted_flag,
                    self.last_skipped == self.optim_state["iter"],
                )
            )
        return super()._display_function_log_(iteration, method)


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


D = 3
b = P(
    rosen,
    np.zeros((1, D)),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options=dict(random_seed=1, display="off", max_fun_evals=200),
)
b.optimize()
bad = [
    r for r in b.rows if ("Train" in r[2]) != r[3] or ("Skip" in r[2]) != r[4]
]
print(
    "poll lines:",
    len(b.rows),
    "lines whose action does not match the poll's refit/skip:",
    len(bad),
)
for r in bad[:6]:
    print(
        "  iter",
        r[0],
        r[1],
        "shown action:",
        repr(r[2]),
        "refit this iteration:",
        r[3],
        "skip:",
        r[4],
    )
