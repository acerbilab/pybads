"""Effect of the cumulative bound nudge vs MATLAB's rule (bound += nudge[1] = 0) on default deterministic runs."""
import inspect
import logging
import textwrap

import common  # noqa
import numpy as np

from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

logging.getLogger("BADS").setLevel(logging.ERROR)

src = textwrap.dedent(inspect.getsource(gpt._robust_gp_fit_))
old = "noise_bound = (noise_bound[0] + noise_nudge, noise_bound[1])"
assert old in src
src_m = src.replace(
    old, "noise_bound = (noise_bound[0] + np.ravel(nudge)[1], noise_bound[1])"
).replace("def _robust_gp_fit_(", "def _robust_gp_fit_matlab_(")
exec(compile(src_m, "patched", "exec"), gpt.__dict__)
orig = gpt._robust_gp_fit_


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


runs = [("rosen3", rosen, 3, s) for s in (41, 42, 43)] + [
    ("ell4", ell, 4, s) for s in (40, 41, 42)
]
for name, fun, D, seed in runs:
    out = []
    for variant in ("as is", "matlab bound"):
        gpt._robust_gp_fit_ = (
            orig if variant == "as is" else gpt._robust_gp_fit_matlab_
        )
        b = BADS(
            fun,
            0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
            -5 * np.ones((1, D)),
            5 * np.ones((1, D)),
            -2 * np.ones((1, D)),
            2 * np.ones((1, D)),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        r = b.optimize()
        out.append((r["fval"], r["func_count"]))
    gpt._robust_gp_fit_ = orig
    print(
        f"{name} seed {seed}: as is fval {out[0][0]:.3e} ({out[0][1]} evals) | MATLAB bound fval {out[1][0]:.3e} ({out[1][1]} evals)"
        f" | log10 ratio {np.log10(out[0][0]/out[1][0]):+.2f}"
    )
