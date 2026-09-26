"""Instrumented default runs (at most 200 evaluations each) to measure, for
the poll of slice B4:
 (a) evaluations of points already in the function log (within tol_mesh/2),
     by stage (contraints_check keeps them; MATLAB's uCheck drops them);
 (b) stop decisions of the poll that differ between the top D+1 PIs
     (PyBADS) and the top D (MATLAB's pless);
 (c) stop decisions (after a good poll) whose target hyperparameters differ
     from the current GP's, and whether MATLAB's UpdateTarget (the current
     posterior with the kernel and mean of hyp_best) flips the decision;
 (d) searches that MATLAB would rebuild after a moved poll (its sticky
     pollmoved_flag) and whose nearest-neighbour set differs from the GP's.
"""
import sys

import gpyreg
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import erfc

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS
from pybads.bads.gaussian_process_train import get_grid_search_neighbors
from pybads.function_logger import FunctionLogger

STAGE = ["init"]
S = {}


def reset_stats():
    S.clear()
    S.update(
        dict(
            evals={"init": 0, "search": 0, "poll": 0},
            dups={"init": 0, "search": 0, "poll": 0},
            dup_in_train=0,
            rebuilds=0,
            inflated=0,
            inflated_with_dup=0,
            stop_calls=0,
            stop_relevant=0,
            flip_D=0,
            flip_D_details=[],
            hyp_diff=0,
            flip_hybrid=0,
            hyb_dtarget=[],
            sticky_checks=0,
            sticky_diff=0,
            gz_shapes=set(),
            py_stops=0,
            m_stops=0,
            flip_Donly=0,
            pollsets=0,
            pollsets_with_eval=0,
        )
    )


# (a) duplicates
orig_fl_call = FunctionLogger.__call__


def fl_call(self, x, record_duplicate_data=True):
    if record_duplicate_data:
        st = STAGE[0]
        S["evals"][st] += 1
        if self.X_max_idx >= 0:
            tol = 1e-6 / 2
            u1 = np.round(np.ravel(x) / tol)
            u2 = np.round(self.X[: self.X_max_idx + 1] / tol)
            if np.any(np.all(u2 == u1, axis=1)):
                S["dups"][st] += 1
    return orig_fl_call(self, x, record_duplicate_data)


FunctionLogger.__call__ = fl_call

orig_lgf = gpt.local_gp_fitting


def lgf(gp, *a, **k):
    out = orig_lgf(gp, *a, **k)
    g = out[0]
    S["rebuilds"] += 1
    X = g.X
    dup = len(np.unique(np.round(X / 5e-7), axis=0)) < X.shape[0]
    S["dup_in_train"] += dup
    m = g.posteriors[0].sn2_mult
    if m is not None and m > 1:
        S["inflated"] += 1
        S["inflated_with_dup"] += dup
    return out


import pybads.bads.bads as bm

bm.local_gp_fitting = lgf
orig_cc = bm.contraints_check


def cc(U, lb, ub, tol_mesh, fl, proj=True, nbc=None):
    out = orig_cc(U, lb, ub, tol_mesh, fl, proj, nbc)
    if STAGE[0] == "poll" and not proj and out.size > 0:
        S["pollsets"] += 1
        u2 = np.round(fl.X[: fl.X_max_idx + 1] / (tol_mesh / 2))
        u1 = np.round(out / (tol_mesh / 2))
        if any(np.any(np.all(u2 == r, axis=1)) for r in u1):
            S["pollsets_with_eval"] += 1
    return out


bm.contraints_check = cc


def decide(self, good, unrel, p, poll_count):
    tol = self.options["tol_poi"]
    if good:
        return True if unrel else bool(p > 1 - tol)
    return bool(
        not unrel
        and (
            self.options["consecutive_skipping"]
            or self.last_skipped < self.optim_state["iter"] - 1
        )
        and poll_count >= self.options["min_failed_poll_steps"]
        and p > 1 - tol
    )


def hybrid_predict(gp, hyp, x):
    D = gp.X.shape[1]
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    mean_N = gp.mean.hyperparameter_count(D)
    hyp = np.ravel(hyp)
    post = gp.posteriors[0]
    x = np.atleast_2d(x)
    kss = gp.covariance.compute(hyp[:cov_N], x, compute_diag=True)[:, 0]
    Ks = gp.covariance.compute(hyp[:cov_N], gp.X, x)
    m = np.ravel(
        gp.mean.compute(hyp[cov_N + noise_N : cov_N + noise_N + mean_N], x)
    )
    mu = m + (Ks.T @ post.alpha)[:, 0]
    if post.L_chol:
        V = solve_triangular(post.L, post.sW * Ks, trans=1)
        s2 = kss - np.sum(V * V, 0)
    else:
        s2 = kss + np.sum(Ks * (post.L @ Ks), 0)
    return mu, np.maximum(s2, 0)


orig_stop = BADS._is_poll_stop_


def is_poll_stop(self, good, unrel, p_less, poll_count):
    fr = sys._getframe(1)
    L = fr.f_locals
    D = self.D
    S["stop_calls"] += 1
    gamma_z = L["gamma_z"]
    S["gz_shapes"].add(tuple(np.shape(gamma_z)))
    if np.all(np.isfinite(gamma_z)):
        # PyBADS as written (np.sort on an (n, 1) array sorts nothing)
        fpi_py = np.sort(0.5 * erfc(-gamma_z / np.sqrt(2)))[::-1]
        p_py = np.prod(1 - fpi_py[: D + 1])
        assert np.isclose(p_py, p_less), (p_py, p_less)
        # MATLAB: fpi sorted descending, the top min(D, n)
        fpi = np.sort(np.ravel(0.5 * erfc(-gamma_z / np.sqrt(2))))[::-1]
        pD = np.prod(1 - fpi[:D])
        pD1 = np.prod(1 - fpi[: D + 1])
        if good and not unrel:
            S["stop_relevant"] += 1
            dpy = decide(self, good, unrel, p_py, poll_count)
            dm = decide(self, good, unrel, pD, poll_count)
            dsorted = decide(self, good, unrel, pD1, poll_count)
            S["py_stops"] += dpy
            S["m_stops"] += dm
            if dpy != dm:
                S["flip_D"] += 1
                S["flip_D_details"].append(
                    (
                        int(poll_count),
                        len(fpi),
                        "py" if dpy else "matlab",
                        float(1 - p_py),
                        float(1 - pD),
                    )
                )
            if dsorted != dm:
                S["flip_Donly"] += 1
            gp = L["gp"]
            hb = np.atleast_2d(L["gp_poll_hyp_best"])
            hc = gp.get_hyperparameters(as_array=True)
            if not np.allclose(hb, hc):
                S["hyp_diff"] += 1
                mu_c, s2_c = hybrid_predict(gp, hc, L["u_poll_best"])
                mu_p, s2_p = gp.predict(np.atleast_2d(L["u_poll_best"]))
                assert np.allclose(mu_c, mu_p.ravel()) and np.allclose(
                    s2_c, s2_p.ravel(), atol=1e-10
                )
                mu_h, s2_h = hybrid_predict(gp, hb, L["u_poll_best"])
                ft_h = mu_h - self.optim_state["sd_level"] * np.sqrt(
                    s2_h + self.options["tol_fun"] ** 2
                )
                ft_py = self.optim_state["f_target"]
                S["hyb_dtarget"].append(
                    (float(ft_py), float(ft_h[0]), float(L["fs"].min()))
                )
                gz = (
                    ft_h - self.sufficient_improvement - np.ravel(L["f_mu"])
                ) / np.ravel(L["fs"])
                if np.all(np.isfinite(gz)):
                    fh = np.sort(0.5 * erfc(-gz / np.sqrt(2)))[::-1]
                    ph = np.prod(1 - fh[:D])
                else:
                    ph = 0.0
                if decide(self, good, unrel, ph, poll_count) != dm:
                    S["flip_hybrid"] += 1
    return orig_stop(self, good, unrel, p_less, poll_count)


BADS._is_poll_stop_ = is_poll_stop

orig_search = BADS._search_step_


def search(self, gp):
    if (
        getattr(self, "_sticky", False)
        and self.optim_state["search_count"] > 0
        and not self.reset_gp
        and not gp.temporary_data.get("needs_rebuild", False)
    ):
        S["sticky_checks"] += 1
        nt = self.optim_state.get("ntrain")
        X, y, _ = get_grid_search_neighbors(
            self.function_logger, self.u, gp, self.options, self.optim_state
        )
        self.optim_state["ntrain"] = nt
        a = set(map(tuple, np.round(X, 12).tolist()))
        b = set(map(tuple, np.round(gp.X, 12).tolist()))
        if a != b or X.shape != gp.X.shape:
            S["sticky_diff"] += 1
    STAGE[0] = "search"
    out = orig_search(self, gp)
    STAGE[0] = "other"
    return out


BADS._search_step_ = search

orig_poll = BADS._poll_step_


def poll(self, gp):
    STAGE[0] = "poll"
    out = orig_poll(self, gp)
    STAGE[0] = "other"
    self._sticky = self.reset_gp
    return out


BADS._poll_step_ = poll


def rosen(x):
    x = np.atleast_2d(x)
    return np.sum(
        100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1
    )


def ellipsoid(x):
    x = np.ravel(x)
    return np.sum((x / np.arange(1, len(x) + 1) ** 2) ** 2)


def ackley(x):
    x = np.ravel(x)
    D = len(x)
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / D))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / D)
        + 20
        + np.e
    )


problems = []
for D in (2, 4):
    problems.append(("rosenbrock", rosen, D))
problems.append(("ellipsoid", ellipsoid, 3))
problems.append(("ellipsoid", ellipsoid, 6))
problems.append(("ackley", ackley, 3))


def run(name, f, D, seed, noisy=False):
    reset_stats()
    STAGE[0] = "init"
    lb = -100 * np.ones(D)
    ub = 100 * np.ones(D)
    plb = -8 * np.ones(D)
    pub = 12 * np.ones(D)
    x0 = 4 * np.ones(D)
    fun = f
    opts = dict(random_seed=seed, display="off", max_fun_evals=200)
    if noisy:
        rng = np.random.default_rng(seed + 1000)
        fun = lambda x: f(x) + rng.normal()
        opts["uncertainty_handling"] = True
    b = BADS(fun, x0, lb, ub, plb, pub, options=opts)
    r = b.optimize()
    return r


tot = {}
for noisy in (False, True):
    for name, f, D in problems:
        for seed in range(2):
            r = run(name, f, D, seed, noisy)
            key = f"{name}_D{D}_{'noisy' if noisy else 'det'}"
            print(
                f"{key} seed={seed} fval={float(r['fval']):.4g} nfev={r['func_count']} "
                f"evals={S['evals']} dups={S['dups']} rebuilds={S['rebuilds']} dup_in_train={S['dup_in_train']} "
                f"inflated={S['inflated']} infl_w_dup={S['inflated_with_dup']} "
                f"pollsets={S['pollsets']} w_eval={S['pollsets_with_eval']} gz={S['gz_shapes']} stop_rel={S['stop_relevant']} py_stops={S['py_stops']} m_stops={S['m_stops']} flip={S['flip_D']} flipDonly={S['flip_Donly']} {S['flip_D_details'][:4]} "
                f"hyp_diff={S['hyp_diff']} flip_hyb={S['flip_hybrid']} "
                f"sticky_checks={S['sticky_checks']} sticky_diff={S['sticky_diff']}",
                flush=True,
            )
            if S["hyb_dtarget"]:
                print(
                    "   targets (py, matlab-hybrid, min fs):",
                    [
                        tuple(round(v, 5) for v in t)
                        for t in S["hyb_dtarget"][:4]
                    ],
                )
