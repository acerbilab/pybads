"""Reach at default options of F3 (prior sampler) and F4 (Cholesky noise
inflation), in default level-0 runs. Counts:
- calls of _get_random_samples_from_priors_, by caller, and the log sf of the
  start that the draw produces after averaging and clipping;
- factorizations in __training_cholesky, and those returned with sn2_mult>1;
- GP states handed on (after each local_gp_fitting and add_and_update_gp)
  whose posterior carries sn2_mult > 1.
"""

import traceback

import gpyreg as gpr
import numpy as np
from common import ackley, bads_mod, gpt, rosenbrock, sphere

from pybads import BADS

GP = gpr.GP
orig_chol = GP.__dict__["_GP__training_cholesky"].__func__
orig_sampler = gpt._get_random_samples_from_priors_
orig_fit = bads_mod.local_gp_fitting
orig_add = bads_mod.add_and_update_gp


def run(fun, D, seed, x0, lb, ub, plb, pub, label):
    chol = []
    samp = []
    states = []

    def chol_w(K, sn2, L_chol, sn2_mult=1):
        try:
            out = orig_chol(K, sn2, L_chol, sn2_mult)
        except Exception:
            chol.append(np.inf)
            raise
        chol.append(out[2] / sn2_mult)
        return out

    def samp_w(gp, rng=None):
        caller = traceback.extract_stack()[-2].name
        h = orig_sampler(gp, rng)
        samp.append((caller, float(h[0][gp.D])))  # log sf of the draw
        return h

    def state(gp):
        m = (
            [p.sn2_mult or 1 for p in gp.posteriors]
            if gp.posteriors is not None
            else [np.nan]
        )
        states.append(max(m))

    def fit_w(*a, **k):
        out = orig_fit(*a, **k)
        state(out[0])
        return out

    def add_w(*a, **k):
        out = orig_add(*a, **k)
        state(out)
        return out

    GP._GP__training_cholesky = staticmethod(chol_w)
    gpt._get_random_samples_from_priors_ = samp_w
    bads_mod.local_gp_fitting = fit_w
    bads_mod.add_and_update_gp = add_w
    try:
        res = BADS(
            fun,
            x0,
            lb,
            ub,
            plb,
            pub,
            options=dict(display="off", random_seed=seed, max_fun_evals=200),
        ).optimize()
    finally:
        GP._GP__training_cholesky = staticmethod(orig_chol)
        gpt._get_random_samples_from_priors_ = orig_sampler
        bads_mod.local_gp_fitting = orig_fit
        bads_mod.add_and_update_gp = orig_add
    c = np.array(chol)
    s = np.array(states, dtype=float)
    ub_sf = np.log(1e6 * 1e-3 / 1e-6)
    print(f"--- {label}: fval={res['fval']:.3g} n={res['func_count']}")
    print(
        f"   factorizations {len(c)}: inflated {int(np.sum((c>1)&np.isfinite(c)))}"
        f" (max x{np.nanmax(np.where(np.isfinite(c), c, np.nan)):.0e}),"
        f" failed after 10 attempts {int(np.sum(~np.isfinite(c)))}"
    )
    vals, cnt = np.unique(s[np.isfinite(s)], return_counts=True)
    print(
        f"   GP states handed on {len(s)}: sn2_mult distribution "
        f"{dict(zip(vals.tolist(), cnt.tolist()))}"
    )
    callers = {}
    for cl, v in samp:
        callers.setdefault(cl, []).append(v)
    for cl, v in callers.items():
        v = np.array(v)
        print(
            f"   prior sampler from {cl}: {len(v)} draws, log sf drawn "
            f"{np.round(v[:4], 1)}...; share above the upper bound "
            f"{ub_sf:.1f}: {np.mean(v > ub_sf):.2f}"
        )
    if not samp:
        print("   prior sampler: not called")


if __name__ == "__main__":
    cases = [
        ("sphere D=2 s0", sphere, 2, 0, -5, 5, -2, 2),
        ("rosenbrock D=2 s1", rosenbrock, 2, 1, -5, 5, -2, 2),
        ("rosenbrock D=4 s0", rosenbrock, 4, 0, -5, 5, -2, 2),
        ("ackley D=6 s0", ackley, 6, 0, -32, 32, -8, 8),
    ]
    for label, f, D, seed, lb, ub, plb, pub in cases:
        x0 = np.random.default_rng(100 + seed).uniform(plb, pub, D)
        run(
            f,
            D,
            seed,
            x0,
            np.full(D, lb * 1.0),
            np.full(D, ub * 1.0),
            np.full(D, plb * 1.0),
            np.full(D, pub * 1.0),
            label,
        )
