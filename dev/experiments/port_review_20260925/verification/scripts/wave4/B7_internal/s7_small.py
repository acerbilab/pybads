import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads import BADS

D = 2
lb = -5 * np.ones(D)
ub = 5 * np.ones(D)
plb = -2 * np.ones(D)
pub = 2 * np.ones(D)
f = lambda x: float(np.sum((np.asarray(x) - 0.3) ** 2))
for mfe, nbc in [(2, None), (2, lambda x: np.sum(x**2, axis=1) > 20)]:
    try:
        b = BADS(
            f,
            np.array([1.2, -0.7]),
            lb,
            ub,
            plb,
            pub,
            non_box_cons=nbc,
            options={"random_seed": 0, "display": "off", "max_fun_evals": mfe},
        )
        r = b.optimize()
        print(
            f"max_fun_evals {mfe}, nbc {nbc is not None}: func_count {r['func_count']}, rows {b.function_logger.Xn+1}, status {r['status']}",
            flush=True,
        )
    except Exception as e:
        print(
            f"max_fun_evals {mfe}, nbc {nbc is not None}: {type(e).__name__}: {str(e)[:200]}",
            flush=True,
        )

# a start at the lower plausible bound (u0 = -1): the seed's cast
b = BADS(
    f,
    np.array([-2.0, 0.5]),
    lb,
    ub,
    plb,
    pub,
    options={"random_seed": 0, "display": "off"},
)
print("u0 at plb:", b.u, "->", b.u[:11].astype(np.uint64))
b2 = BADS(
    f,
    np.array([-1.9, 0.5]),
    lb,
    ub,
    plb,
    pub,
    options={"random_seed": 0, "display": "off"},
)
b._init_mesh_()
b2._init_mesh_()
print(
    "design for x0 at plb:\n",
    b.function_logger.X[1 : b.function_logger.Xn + 1],
)
print(
    "design for x0 inside:\n",
    b2.function_logger.X[1 : b2.function_logger.Xn + 1],
)
