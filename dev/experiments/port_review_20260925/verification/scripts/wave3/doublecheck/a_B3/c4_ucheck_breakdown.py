"""W3-1: in seeded runs, the calls of contraints_check whose output changes
with MATLAB's round in the bins, split into (i) calls where the candidates
removed as already evaluated differ and (ii) calls where only the merging of
candidates that share a bin differs; with the search mesh size at the call."""
import hdr  # noqa: F401
import numpy as np
from ucheck_ref import mround

import pybads.bads.bads as bads_module
import pybads.search.es_search as es_module
from pybads import BADS

port_cc = es_module.contraints_check


def removed_as_evaluated(U, lb, ub, tol_mesh, fl, proj, rounding):
    """Indices (rows of the deduplicated, projected U) whose bin holds a
    logged point."""
    if proj:
        V = np.maximum(np.minimum(U, ub), lb)
    else:
        idx = np.any(U > ub, axis=1) | np.any(U < lb, axis=1)
        V = U[~idx]
    _, first = np.unique(V, axis=0, return_index=True)
    V = V[np.sort(first)]
    t = tol_mesh / 2
    b1 = rounding(V / t) + 0.0
    b2 = {tuple(r) for r in rounding(fl.X[: fl.X_max_idx + 1] / t) + 0.0}
    return {i for i, r in enumerate(b1) if tuple(r) in b2}


rec = []


def wrap(site):
    def cc(U, lb, ub, tol_mesh, fl, proj=True, non_box_cons=None):
        out = port_cc(U, lb, ub, tol_mesh, fl, proj, non_box_cons)
        from cc_variants import port_mround

        alt = port_mround(U, lb, ub, tol_mesh, fl, proj, non_box_cons)
        if not (out.shape == alt.shape and np.array_equal(out, alt)):
            ra = removed_as_evaluated(U, lb, ub, tol_mesh, fl, proj, np.round)
            rb = removed_as_evaluated(U, lb, ub, tol_mesh, fl, proj, mround)
            rec.append((site, ra != rb, len(ra), len(rb), len(ra ^ rb)))
        return out

    return cc


es_module.contraints_check = wrap("es")
bads_module.contraints_check = wrap("bads.py")


def edge_sphere(x):
    return float(np.sum((np.atleast_2d(x) + 1.0) ** 2))


def noisy_sphere(seed):
    r = np.random.default_rng(seed)
    return lambda x: float(np.sum(np.atleast_2d(x) ** 2)) + r.standard_normal()


for name, f, x0, lb, ub, plb, pub, opts in [
    (
        "edge_D2",
        edge_sphere,
        2.5 * np.ones(2),
        np.zeros(2),
        5 * np.ones(2),
        np.zeros(2),
        5 * np.ones(2),
        {"random_seed": 0},
    ),
    (
        "edge_D1",
        edge_sphere,
        np.array([2.5]),
        np.zeros(1),
        5 * np.ones(1),
        np.zeros(1),
        5 * np.ones(1),
        {"random_seed": 0},
    ),
    (
        "noisy_D2",
        noisy_sphere(0),
        2 * np.ones(2),
        -10 * np.ones(2),
        10 * np.ones(2),
        -5 * np.ones(2),
        5 * np.ones(2),
        {"random_seed": 0, "uncertainty_handling": True},
    ),
]:
    rec.clear()
    b = BADS(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"display": "off", "max_fun_evals": 200, **opts},
    )
    b.optimize()
    n_rem = sum(1 for r in rec if r[1])
    n_sym = sum(r[4] for r in rec)
    print(
        f"{name}: calls whose output changes with MATLAB's round {len(rec)}; of them, "
        f"the candidates removed as evaluated differ in {n_rem} (candidates removed by one rounding only: {n_sym})",
        flush=True,
    )
