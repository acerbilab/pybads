"""Runtime checks in short seeded runs (<= 200 evaluations):
1. after every refit of local_gp_fitting, the geometry in temporary_data
   equals MATLAB gpupdate.m's transcription on the GP's hyperparameters;
2. every update_hedge equals acqPortfolio.m 'upd' on the same inputs;
3. every _eval_improvement_ call: which caller, and whether its base is the
   incumbent (search, poll) as in bads.m."""

import copy
import inspect

import gpyreg
import numpy as np
from scipy.special import erfc

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)

import pybads.bads.bads as bads_mod
from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge

stats = {"refits": 0, "geom_maxdiff": 0.0, "hedge": 0, "hedge_maxdiff": 0.0}
callers = {}

orig_lgf = bads_mod.local_gp_fitting


def matlab_geometry(gp, optim_state, options):
    hyp = gp.get_hyperparameters()[0]
    logell = np.asarray(hyp["covariance_log_lengthscale"]).ravel()
    D = logell.size
    lenscale = np.exp(logell) if D > 1 else 1.0
    ll = options["gp_rescale_poll"] * logell
    ll = np.exp(ll - np.mean(ll))
    ub = np.ravel(optim_state["ub"]).copy()
    lb = np.ravel(optim_state["lb"]).copy()
    pub = np.ravel(optim_state["pub"])
    plb = np.ravel(optim_state["plb"])
    ub[~np.isfinite(ub)] = pub[~np.isfinite(ub)]
    lb[~np.isfinite(lb)] = plb[~np.isfinite(lb)]
    ll = np.minimum(
        np.maximum(ll, optim_state["search_mesh_size"]),
        (ub - lb) / optim_state["scale"],
    )
    alpha = float(np.exp(np.ravel(hyp["covariance_log_shape"])[0]))
    effr = np.sqrt(alpha * (np.exp(1 / alpha) - 1))
    return lenscale, ll, effr


def lgf(
    gp,
    current_point,
    function_logger,
    options,
    optim_state,
    ih,
    refit_flag,
    rng=None,
):
    before = {
        k: copy.deepcopy(gp.temporary_data.get(k))
        for k in ("len_scale", "poll_scale", "effective_radius")
    }
    out = orig_lgf(
        gp,
        current_point,
        function_logger,
        options,
        optim_state,
        ih,
        refit_flag,
        rng=rng,
    )
    gp2 = out[0]
    td = gp2.temporary_data
    if refit_flag and not td.get("needs_rebuild", False):
        stats["refits"] += 1
        ls, ps, er = matlab_geometry(gp2, optim_state, options)
        D = np.size(ps)
        if D > 1:
            stats["geom_maxdiff"] = max(
                stats["geom_maxdiff"],
                np.max(np.abs(np.ravel(td["len_scale"]) - ls)),
            )
        stats["geom_maxdiff"] = max(
            stats["geom_maxdiff"],
            np.max(np.abs(np.ravel(td["poll_scale"]) - ps)),
        )
        stats["geom_maxdiff"] = max(
            stats["geom_maxdiff"],
            float(np.max(np.abs(np.ravel(td["effective_radius"]) - er))),
        )
    elif not refit_flag:
        for k, v in before.items():
            assert np.array_equal(np.ravel(v), np.ravel(td.get(k))), k
    return out


bads_mod.local_gp_fitting = lgf

orig_upd = ESSearchHedge.update_hedge


def upd(self, u_search, fval_old, f, fs, gp, mesh_size):
    g0 = self.g.copy()
    orig_upd(self, u_search, fval_old, f, fs, gp, mesh_size)
    if u_search is None:
        gm = self.decay * g0
    else:
        gm = g0.copy()
        for i in range(self.n_funs):
            if i == self.chosen_hedge.item():
                fH, fsH = f, fs
            else:
                fH, fsH = 0.0, 1.0
            if fsH == 0:
                er = max(0, fval_old - fH)
            elif np.isfinite(fH) and np.isfinite(fsH) and fsH > 0:
                gz = (fval_old - fH) / fsH
                er = fsH * (
                    gz * 0.5 * erfc(-gz / np.sqrt(2))
                    + np.exp(-0.5 * gz**2) / np.sqrt(2 * np.pi)
                )
            else:
                er = 0
            gm[i] = self.decay * g0[i] + er / self.phat[i] / mesh_size
    stats["hedge"] += 1
    stats["hedge_maxdiff"] = max(
        stats["hedge_maxdiff"], float(np.max(np.abs(self.g - gm)))
    )


ESSearchHedge.update_hedge = upd

orig_ei = BADS._eval_improvement_


def ei(self, f_base, f_new, s_base, s_new, q):
    caller = inspect.stack()[1]
    key = (caller.function, caller.lineno)
    rec = callers.setdefault(key, {"n": 0, "base_is_incumbent": 0, "q": set()})
    rec["n"] += 1
    rec["q"].add(q)
    if np.isscalar(f_base) or np.size(f_base) == 1:
        if np.all(
            np.asarray(f_base, float) == np.asarray(self.fval, float)
        ) and np.all(np.asarray(s_base, float) == np.asarray(self.fsd, float)):
            rec["base_is_incumbent"] += 1
    return orig_ei(self, f_base, f_new, s_base, s_new, q)


BADS._eval_improvement_ = ei


def run(D, noisy, seed, max_fe, extra=None):
    nrng = np.random.default_rng(seed + 1000)
    scales = np.logspace(0, 2, D)

    def fun(x):
        x = np.atleast_2d(x)
        f = float(np.sum((scales * x) ** 2))
        if noisy:
            f += nrng.standard_normal()
        return f

    opts = {"display": "off", "random_seed": seed, "max_fun_evals": max_fe}
    if noisy:
        opts["uncertainty_handling"] = True
    if extra:
        opts.update(extra)
    b = BADS(
        fun,
        np.ones(D) * 0.5,
        -5 * np.ones(D),
        5 * np.ones(D),
        -2 * np.ones(D),
        2 * np.ones(D),
        options=opts,
    )
    r = b.optimize()
    return r


for D, noisy, extra in [
    (3, False, None),
    (3, True, None),
    (1, False, None),
    (2, True, {"improvement_quantile": 0.2}),
]:
    for k in stats:
        stats[k] = 0 if isinstance(stats[k], int) else 0.0
    callers.clear()
    r = run(D, noisy, 1, 150, extra)
    print(
        f"D={D} noisy={noisy} extra={extra}: fval={r['fval']:.4g} evals={r['func_count']}",
        flush=True,
    )
    print("  ", stats, flush=True)
    for key, rec in sorted(callers.items(), key=lambda kv: kv[0][1]):
        print("   ", key, rec, flush=True)
