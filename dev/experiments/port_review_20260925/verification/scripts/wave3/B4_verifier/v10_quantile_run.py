"""C-F7: a run with improvement_quantile outside (0, 1)."""
import numpy as np
from vhdr import box, sphere

from pybads import BADS

D = 2
lb, ub, plb, pub = box(D)
for q in (0.0, 1.0, 1.5):
    try:
        r = BADS(
            sphere,
            1.5 * np.ones(D),
            lb,
            ub,
            plb,
            pub,
            options=dict(
                random_seed=0,
                display="off",
                max_fun_evals=100,
                improvement_quantile=q,
            ),
        ).optimize()
        print(
            f"q={q}: ran, fval={r['fval']:.4g} nfev={r['func_count']} x={np.round(r['x'], 3)} msg={r['message'][:60]}"
        )
    except Exception as e:
        print(f"q={q}:", type(e).__name__, e)
r = BADS(
    sphere,
    1.5 * np.ones(D),
    lb,
    ub,
    plb,
    pub,
    options=dict(random_seed=0, display="off", max_fun_evals=100),
).optimize()
print(f"q=0.5 (default): fval={r['fval']:.4g} nfev={r['func_count']}")
