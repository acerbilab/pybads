"""Instrumented default runs (<= 200 evaluations each), measuring:
 A  (I-F3/C-F1/K6) stop decisions after a good poll with a reliable GP where
    the port's p_less and MATLAB's (sorted, top D) decide differently;
 B  (K2/C-F4) those decisions where gp_poll_hyp_best differs from the GP's
    hyperparameters, and whether MATLAB's hybrid target, or the current GP's
    own prediction, would decide differently from the port's target;
 C  (I-F6/C-F2) poll sets holding logged points, and poll evaluations of
    logged points;
 D  (C-F5) searches that MATLAB's sticky pollmoved_flag would rebuild and the
    port does not, and whether the rebuilt training set / prediction differs;
 E  (K4) target computations where optim_state["fval"] != self.fval;
 F  (K7) calls of poll_mads_2n per poll (the refill branch needs > 1);
 G  (I-F9) stop calls after a good poll with a zero predictive SD.
"""
import copy
import sys
import time

import numpy as np
from scipy.special import erfc
from vhdr import box, ellipsoid, rosen
from vhybrid import hybrid_predict

import pybads.bads.bads as bm
from pybads import BADS

S = {}


def reset():
    S.clear()
    S.update(
        A_rel=0,
        A_flip=0,
        A_flip_sortonly=0,
        A_flip_Donly=0,
        A_nleft_gt=0,
        A_topmissed=0,
        B_hypdiff=0,
        B_flip_hyb=0,
        B_flip_cur=0,
        B_ex=[],
        C_sets=0,
        C_sets_logged=0,
        C_evals=0,
        C_evals_logged=0,
        D_checks=0,
        D_setdiff=0,
        D_preddiff=0,
        E_calls=0,
        E_stale=0,
        F_polls=0,
        F_multi=0,
        G_zero=0,
        stage="init",
        polln=0,
    )


def logged(fl, u, tol=1e-6):
    u2 = np.round(fl.X[: fl.X_max_idx + 1] / (tol / 2))
    return bool(
        np.any(np.all(u2 == np.round(np.ravel(u) / (tol / 2)), axis=1))
    )


def pl_port(fpi, D):
    f = np.sort(fpi)[::-1]
    return np.prod(1 - f[0 : min(D + 1, len(f))])


def pl_mat(fpi, D):
    f = np.sort(np.ravel(fpi))[::-1]
    return np.prod(1 - f[: min(D, len(f))])


def pl_sorted_D1(fpi, D):
    f = np.sort(np.ravel(fpi))[::-1]
    return np.prod(1 - f[: min(D + 1, len(f))])


orig_stop = bm.BADS._is_poll_stop_


def stop(self, good, unrel, p_less, poll_count):
    L = sys._getframe(1).f_locals
    D, tol = self.D, self.options["tol_poi"]
    fs = L["fs"]
    f_mu = L["f_mu"]
    SI = self.sufficient_improvement
    if good and np.any(fs == 0):
        S["G_zero"] += 1
    if good and not unrel and np.all(np.isfinite(L["gamma_z"])):
        S["A_rel"] += 1
        fpi = 0.5 * erfc(-L["gamma_z"] / np.sqrt(2))
        assert np.isclose(pl_port(fpi, D), p_less)
        dp = p_less > 1 - tol
        dm = pl_mat(fpi, D) > 1 - tol
        S["A_flip"] += dp != dm
        S["A_flip_sortonly"] += (pl_sorted_D1(fpi, D) > 1 - tol) != dp
        S["A_flip_Donly"] += (pl_sorted_D1(fpi, D) > 1 - tol) != dm
        if len(fpi) > D + 1:
            S["A_nleft_gt"] += 1
            S["A_topmissed"] += int(
                np.argmax(np.ravel(fpi)) < len(fpi) - (D + 1)
            )
        gp = L["gp"]
        hb = np.atleast_2d(L["gp_poll_hyp_best"])
        hc = gp.get_hyperparameters(as_array=True)
        if not np.allclose(hb, hc):
            S["B_hypdiff"] += 1
            sdl, tf = self.optim_state["sd_level"], self.options["tol_fun"]
            u = L["u_poll_best"]
            mh, s2h = hybrid_predict(gp, hb, u)
            mc, s2c = gp.predict(np.atleast_2d(u))
            alts = {
                "hyb": mh.item() - sdl * np.sqrt(s2h.item() + tf**2),
                "cur": mc.item() - sdl * np.sqrt(s2c.item() + tf**2),
            }
            for k, ft in alts.items():
                gz = (ft - SI - f_mu) / fs
                d = (
                    (pl_port(0.5 * erfc(-gz / np.sqrt(2)), D) > 1 - tol)
                    if np.all(np.isfinite(gz))
                    else True
                )
                S["B_flip_" + k] += d != dp
            if len(S["B_ex"]) < 3:
                S["B_ex"].append(
                    (
                        round(self.optim_state["f_target"], 6),
                        round(alts["hyb"], 6),
                        round(alts["cur"], 6),
                    )
                )
    return orig_stop(self, good, unrel, p_less, poll_count)


bm.BADS._is_poll_stop_ = stop

orig_cc = bm.contraints_check


def cc(U, lb, ub, tol_mesh, fl, proj=True, nbc=None):
    out = orig_cc(U, lb, ub, tol_mesh, fl, proj, nbc)
    if S["stage"] == "poll" and not proj and out.size:
        S["C_sets"] += 1
        S["C_sets_logged"] += any(logged(fl, r, tol_mesh) for r in out)
    return out


bm.contraints_check = cc

orig_pm = bm.poll_mads_2n


def pm(*a, **k):
    S["polln"] += 1
    return orig_pm(*a, **k)


bm.poll_mads_2n = pm

orig_fl_call = bm.FunctionLogger.__call__


def fl_call(self, x, record_duplicate_data=True):
    if S["stage"] == "poll" and record_duplicate_data:
        S["C_evals"] += 1
        S["C_evals_logged"] += logged(self, x)
    return orig_fl_call(self, x, record_duplicate_data)


bm.FunctionLogger.__call__ = fl_call

orig_tgt = bm.BADS._get_target_from_gp_


def tgt(self, u, gp, hyp_best):
    S["E_calls"] += 1
    S["E_stale"] += not np.isclose(
        float(np.ravel(self.optim_state["fval"])[0]),
        float(np.ravel(self.fval)[0]),
    )
    return orig_tgt(self, u, gp, hyp_best)


bm.BADS._get_target_from_gp_ = tgt

orig_search = bm.BADS._search_step_


def search(self, gp):
    if (
        getattr(self, "_sticky", False)
        and self.optim_state["search_count"] > 0
        and not self.reset_gp
        and not gp.temporary_data.get("needs_rebuild", False)
    ):
        S["D_checks"] += 1
        g2 = copy.deepcopy(gp)
        os2 = copy.deepcopy(self.optim_state)
        g2, _ = bm.local_gp_fitting(
            g2,
            self.u,
            self.function_logger,
            self.options,
            os2,
            self.iteration_history,
            False,
            rng=np.random.default_rng(0),
        )
        a = {tuple(np.round(r, 12)) for r in gp.X}
        b = {tuple(np.round(r, 12)) for r in g2.X}
        S["D_setdiff"] += a != b
        m1, _ = gp.predict(np.atleast_2d(self.u))
        m2, _ = g2.predict(np.atleast_2d(self.u))
        S["D_preddiff"] += not np.isclose(
            m1.item(), m2.item(), rtol=1e-6, atol=1e-10
        )
    S["stage"] = "search"
    out = orig_search(self, gp)
    S["stage"] = "other"
    return out


bm.BADS._search_step_ = search

orig_poll = bm.BADS._poll_step_


def poll(self, gp):
    S["stage"] = "poll"
    S["polln"] = 0
    out = orig_poll(self, gp)
    S["stage"] = "other"
    S["F_polls"] += 1
    S["F_multi"] += S["polln"] > 1
    self._sticky = self.reset_gp
    return out


bm.BADS._poll_step_ = poll


def run(name, f, D, seed, noisy):
    reset()
    lb, ub, plb, pub = box(D)
    fun = f
    opts = dict(random_seed=seed, display="off", max_fun_evals=200)
    if noisy:
        nrng = np.random.default_rng(1000 + seed)
        fun = lambda x: f(x) + nrng.normal()
        opts["uncertainty_handling"] = True
    t = time.time()
    r = BADS(fun, 1.5 * np.ones(D), lb, ub, plb, pub, options=opts).optimize()
    s = {k: v for k, v in S.items() if k not in ("stage", "polln")}
    print(
        f"{name} D={D} lvl={1 if noisy else 0} seed={seed} fval={r['fval']:.4g} nfev={r['func_count']} t={time.time()-t:.0f}s\n   {s}",
        flush=True,
    )


for noisy in (False, True):
    for name, f, D, seeds in (
        ("rosen", rosen, 4, (1, 2)),
        ("ellipsoid", ellipsoid, 6, (1,)),
    ):
        for seed in seeds:
            run(name, f, D, seed, noisy)
print("np.geterr() after the runs:", np.geterr())
