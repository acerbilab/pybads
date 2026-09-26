"""improvement_quantile at the ends of (0, 1): MATLAB's EvalImprovement
raises; _eval_improvement_ computes sigma * (+-inf) + mu."""
import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
from pybads import BADS

f = lambda x: np.sum(np.ravel(x) ** 2)
for q in (1.0, 0.0):
    b = BADS(
        f,
        np.ones(2) * 2,
        -10 * np.ones(2),
        10 * np.ones(2),
        -5 * np.ones(2),
        5 * np.ones(2),
        options=dict(
            improvement_quantile=q,
            random_seed=0,
            display="off",
            max_fun_evals=100,
        ),
    )
    print(
        "q =",
        q,
        "improvement (level 0):",
        b._eval_improvement_(1.0, 0.5, 0.0, 0.0, q),
    )
    r = b.optimize()
    print(
        "   fval",
        r["fval"],
        "x",
        np.round(r["x"], 3),
        "nfev",
        r["func_count"],
        r["message"],
    )
