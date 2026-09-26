"""B1 verifier, C-F4 / I-F12 (search_factor_min): the search factor in
default runs, as PyBADS runs them and with MATLAB's floor
(bads.m:1366, max(SearchFactorMin, factor*SearchScaleFailure)) applied in
memory by wrapping BADS._update_search_stats_. 200 evaluations, seed 1."""

import common  # noqa: F401
import numpy as np

from pybads import BADS

orig = BADS._update_search_stats_


def floored(self, search_status, search_dist):
    out = orig(self, search_status, search_dist)
    # orig multiplied by sqrt(0.5) on failure and may have reset to 1
    if search_status == "failure" and self.optim_state["search_factor"] != 1:
        self.optim_state["search_factor"] = max(
            self.options["search_factor_min"],
            self.optim_state["search_factor"],
        )
    return out


def rosen(x):
    x = np.asarray(x).ravel()
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ellip(x):
    x = np.asarray(x).ravel()
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / (D - 1)) * x) ** 2))


def run(fun, D, floor):
    BADS._update_search_stats_ = floored if floor else orig
    b = BADS(
        fun,
        np.full(D, -1.5) if fun is rosen else np.full(D, 1.0),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options={"display": "off", "random_seed": 1, "max_fun_evals": 200},
    )
    r = b.optimize()
    lsf = np.exp(np.array(b.optim_state["search_stats"]["log_search_factor"]))
    return r, lsf


print(
    f"{'problem':16s} {'variant':8s} {'searches':>8s} {'<0.5':>6s} "
    f"{'min factor':>10s} {'fval':>12s} {'evals':>6s}"
)
for name, fun, D in [
    ("rosenbrock D=2", rosen, 2),
    ("ellipsoid D=6", ellip, 6),
    ("rosenbrock D=6", rosen, 6),
]:
    for floor in (False, True):
        r, lsf = run(fun, D, floor)
        print(
            f"{name:16s} {'floor' if floor else 'pybads':8s} {lsf.size:8d} "
            f"{np.mean(lsf < 0.5 - 1e-12):6.0%} {lsf.min():10.4f} "
            f"{r['fval']:12.5g} {r['func_count']:6d}"
        )
