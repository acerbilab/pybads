import io
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


D = 2
args = (
    sphere,
    np.full((1, D), 0.7),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
)
# sloppy_improvement=False
try:
    r = BADS(
        *args,
        options=dict(
            sloppy_improvement=False,
            random_seed=0,
            display="off",
            max_fun_evals=100,
        ),
    ).optimize()
    print("sloppy False ok", r["fval"])
except Exception as e:
    print("sloppy False raised:", type(e).__name__, e)


# max_iter: count polls vs searches in the last iteration
class P(BADS):
    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        s.log = []

    def _search_step_(s, gp):
        s.log.append(("S", s.optim_state["iter"]))
        return super()._search_step_(gp)

    def _poll_step_(s, gp):
        s.log.append(("P", s.optim_state["iter"]))
        return super()._poll_step_(gp)


for mi in (1, 2, 3):
    b = P(*args, options=dict(max_iter=mi, random_seed=0, display="off"))
    r = b.optimize()
    npoll = sum(1 for k, _ in b.log if k == "P")
    last = [k for k, i in b.log if i == b.optim_state["iter"]]
    print(
        f"max_iter={mi}: iterations={r['iterations']}, polls={npoll}, steps in last iteration={last}, msg={r['message']!r}"
    )

# display levels
for disp in ("final", "notify", "iter", "off"):
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    lg = logging.getLogger("BADS")
    lg.addHandler(h)
    b = BADS(
        *args, options=dict(display=disp, random_seed=0, max_fun_evals=40)
    )
    b.optimize()
    lg.removeHandler(h)
    lines = [l for l in stream.getvalue().splitlines() if l.strip()]
    print(f"display={disp!r}: {len(lines)} lines logged; first: {lines[:1]}")
