"""B1 verifier, K1 / I-F1 / C-F2: with the half-bounds test made per
variable (in memory: the source of BADS._bounds_check_ recompiled with the
condition replaced, no file touched), does a problem that mixes a bounded
and an unbounded variable run? 200 evaluations at most, seed 1; the
optimum at (0.3, 2.0)."""
import inspect
import textwrap

import common  # noqa: F401
import numpy as np

import pybads.bads.bads as bads_mod
from pybads import BADS

src = textwrap.dedent(inspect.getsource(BADS._bounds_check_))
import re

pat = re.compile(
    r"if \(\s*np\.any\(np\.isfinite\(lower_bounds\)\)\s*and np\.any\(np\.invert\(np\.isfinite\(upper_bounds\)\)\)\s*or np\.any\(np\.invert\(np\.isfinite\(lower_bounds\)\)\)\s*and np\.any\(np\.isfinite\(upper_bounds\)\)\s*\):"
)
assert len(pat.findall(src)) == 1, "condition not found"
src = pat.sub(
    "if np.any(np.isfinite(lower_bounds) != np.isfinite(upper_bounds)):", src
)
ns = {}
exec(src, bads_mod.__dict__, ns)
patched = ns["_bounds_check_"]


def fun(x):
    x = np.asarray(x).ravel()
    return float((x[0] - 0.3) ** 2 + 0.1 * (x[1] - 2.0) ** 2)


lb, ub = np.array([0.0, -np.inf]), np.array([1.0, np.inf])
plb, pub = np.array([0.1, -3.0]), np.array([0.9, 3.0])
for variant in ("as is", "per-variable test"):
    BADS._bounds_check_ = (
        BADS.__dict__["_bounds_check_"] if variant == "as is" else patched
    )
    try:
        b = BADS(
            fun,
            np.array([0.5, 0.0]),
            lb,
            ub,
            plb,
            pub,
            options={"display": "off", "random_seed": 1, "max_fun_evals": 200},
        )
        r = b.optimize()
        print(
            f"{variant}: ran; x {np.round(r['x'], 4)} fval {r['fval']:.3g} "
            f"evals {r['func_count']} message {r['message'][:50]!r}; "
            f"optim_state plb/pub {b.optim_state['plb'].ravel()} "
            f"{b.optim_state['pub'].ravel()}, lb/ub u "
            f"{b.optim_state['lb'].ravel()} {b.optim_state['ub'].ravel()}"
        )
    except Exception as e:
        print(f"{variant}: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
