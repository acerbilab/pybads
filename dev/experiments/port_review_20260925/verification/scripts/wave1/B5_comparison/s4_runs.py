"""Instrumented default runs.

1. At every call of _is_gp_refit_time_, compare the port's verdict with a
   transcription of MATLAB's gppredcheck/IsRefitTime on the same stats, with
   the SD the port stores (latent) and with the predictive SD of the
   observation (MATLAB's ys, latent + noise).
2. Count the rebuilds of the local GP (local_gp_fitting) by caller, and
   those that MATLAB's rule (rebuild when post is empty: count == 0, refit,
   or a reset since the last rebuild) would not make.
3. mode "clear_reset": emulate MATLAB's empty-post semantics by clearing
   reset_gp after each rebuild in the search or poll step.
"""
import sys
import types

import gpyreg
import matlab_ref as M
import numpy as np

import pybads
import pybads.bads.bads as bads_module
from pybads.bads.bads import BADS

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)

ALPHA = 1e-6
_orig_lgf = bads_module.local_gp_fitting
_orig_acq = bads_module.acq_fcn_lcb
_orig_is_refit = BADS._is_gp_refit_time_
_orig_save = BADS._save_gp_stats_
_orig_record = BADS._record_gp_refit_


class B(BADS):
    @property
    def reset_gp(self):
        return self._reset

    @reset_gp.setter
    def reset_gp(self, v):
        self._reset = bool(v)
        if v:
            self._pending = True


def run(fun, x0, lb, ub, plb, pub, seed, mode, max_fe=200, extra=None):
    log = types.SimpleNamespace(
        rebuilds={"search": 0, "poll": 0, "other": 0},
        extra_rebuilds={"search": 0, "poll": 0},
        poll_extra_with_new_points=0,
        calls=0,
        dis_py_vs_mat_latent_unrel=0,
        dis_py_vs_mat_latent_refit=0,
        dis_mat_latent_vs_true_unrel=0,
        dis_mat_latent_vs_true_refit=0,
        dis_py_vs_mat_true_refit=0,
        dis_py_vs_mat_true_unrel=0,
        n_hist={},
        ratio=[],
        last_acq=None,
        ys_true=[],
    )

    def acq(xi, fc, gp, *a, **k):
        z, f_mu, fs = _orig_acq(xi, fc, gp, *a, **k)
        _, ys2 = gp.predict(xi, add_noise=True)
        log.last_acq = (fs.ravel().copy(), np.sqrt(ys2).ravel().copy())
        return z, f_mu, fs

    def save(self, fval, ymu, ys):
        fs_all, ys_all = log.last_acq
        idx = np.flatnonzero(fs_all == ys)
        yt = ys_all[idx[0]] if idx.size else np.nan
        log.ys_true.append(yt)
        log.ratio.append(yt / ys if ys > 0 else np.inf)
        return _orig_save(self, fval, ymu, ys)

    def record(self):
        log.ys_true = []
        return _orig_record(self)

    def is_refit(self, alpha):
        st = self.gp_stats
        if st.get("iter_gp") is None:
            f = m = s = []
        else:
            f = [float(v) for v in st.get("fval")]
            m = [float(v) for v in st.get("ymu")]
            s = [float(v) for v in st.get("ys")]
        n = len(f)
        log.n_hist[n] = log.n_hist.get(n, 0) + 1
        fc = self.function_logger.func_count
        last = self.optim_state["lastfitgp"]
        mrt = self.options["min_refit_time"]

        def mat(sds):
            try:
                u = M.gppredcheck(f, m, sds, alpha)
            except Exception:
                u = True
            r = M.is_refit_time(last, fc, n, u, self.D, mrt)
            return r, (False if r else u)

        r_lat, u_lat = mat(s)
        r_true, u_true = mat(list(log.ys_true))
        r_py, u_py = _orig_is_refit(self, alpha)
        r_py, u_py = bool(r_py), bool(u_py)
        log.calls += 1
        log.dis_py_vs_mat_latent_unrel += u_py != u_lat
        log.dis_py_vs_mat_latent_refit += r_py != r_lat
        log.dis_mat_latent_vs_true_unrel += u_lat != u_true
        log.dis_mat_latent_vs_true_refit += r_lat != r_true
        log.dis_py_vs_mat_true_refit += r_py != r_true
        log.dis_py_vs_mat_true_unrel += u_py != u_true
        return r_py, u_py

    def lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
        fr = sys._getframe(1)
        name = fr.f_code.co_name
        self = fr.f_locals.get("self")
        caller = {"_search_step_": "search", "_poll_step_": "poll"}.get(
            name, "other"
        )
        is_step_rebuild = caller in (
            "search",
            "poll",
        ) and gp is fr.f_locals.get("gp")
        if (
            caller == "search"
            and "new_gp" in fr.f_locals
            and fr.f_locals.get("u_search") is not None
            and not np.array_equal(u, self.u)
        ):
            is_step_rebuild = (
                False  # the noisy rebuild around the search point
            )
        if is_step_rebuild:
            log.rebuilds[caller] += 1
            if caller == "search":
                cnt = optim_state["search_count"]
            else:
                cnt = fr.f_locals["poll_count"]
            matlab_would = (
                refit_flag
                or cnt == 0
                or gp.temporary_data.get("needs_rebuild", False)
                or self._pending
            )
            if not matlab_would:
                log.extra_rebuilds[caller] += 1
                if caller == "poll":
                    n_before = gp.X.shape[0]
        else:
            log.rebuilds["other"] += 1
        out = _orig_lgf(
            gp, u, fl, options, optim_state, ih, refit_flag, rng=rng
        )
        if is_step_rebuild:
            self._pending = False
            if mode == "clear_reset":
                self._reset = False
        return out

    bads_module.local_gp_fitting = lgf
    bads_module.acq_fcn_lcb = acq
    BADS._is_gp_refit_time_ = is_refit
    BADS._save_gp_stats_ = save
    BADS._record_gp_refit_ = record
    try:
        opts = {"display": "off", "random_seed": seed, "max_fun_evals": max_fe}
        if extra:
            opts.update(extra)
        b = B(fun, x0, lb, ub, plb, pub, options=opts)
        b._pending = False
        res = b.optimize()
    finally:
        bads_module.local_gp_fitting = _orig_lgf
        bads_module.acq_fcn_lcb = _orig_acq
        BADS._is_gp_refit_time_ = _orig_is_refit
        BADS._save_gp_stats_ = _orig_save
        BADS._record_gp_refit_ = _orig_record
    return res, log


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1)) * x) ** 2))


class NoisySphere:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        return float(np.sum(np.ravel(x) ** 2) + self.rng.standard_normal())


if __name__ == "__main__":
    probs = {
        "rosen3": (rosen, 3),
        "ellip4": (ellip, 4),
    }
    for name, (f, D) in probs.items():
        for seed in range(3):
            x0 = (
                np.full((1, D), -1.5)
                if name == "rosen3"
                else np.full((1, D), 2.0)
            )
            lb, ub = np.full((1, D), -5.0), np.full((1, D), 5.0)
            plb, pub = np.full((1, D), -3.0), np.full((1, D), 3.0)
            for mode in ("as_is", "clear_reset"):
                res, log = run(f, x0, lb, ub, plb, pub, seed, mode)
                print(
                    f"{name} seed={seed} {mode:11s} fval={res['fval']:.3e} "
                    f"fevals={res['func_count']} iters={res['iterations']} "
                    f"rebuilds={log.rebuilds} extra(not in MATLAB)={log.extra_rebuilds}"
                )
                if mode == "as_is":
                    r = np.array(log.ratio)
                    print(
                        f"   refit-time calls={log.calls}; verdict differs: "
                        f"unreliable py-vs-MATLAB(same SD)={log.dis_py_vs_mat_latent_unrel}, "
                        f"refit py-vs-MATLAB(same SD)={log.dis_py_vs_mat_latent_refit}; "
                        f"MATLAB latent-vs-predictive SD: unrel={log.dis_mat_latent_vs_true_unrel}, "
                        f"refit={log.dis_mat_latent_vs_true_refit}; "
                        f"py-vs-MATLAB(predictive SD): unrel={log.dis_py_vs_mat_true_unrel} refit={log.dis_py_vs_mat_true_refit}"
                    )
                    print(
                        f"   ys_pred/fs_latent: median={np.median(r):.3g} "
                        f"q90={np.quantile(r, 0.9):.3g} max={np.max(r):.3g}; "
                        f"n-stats histogram (n: calls) {dict(sorted(log.n_hist.items())[:6])}"
                    )
    D = 3
    for seed in range(2):
        for mode in ("as_is", "clear_reset"):
            res, log = run(
                NoisySphere(100 + seed),
                np.full((1, D), 2.0),
                np.full((1, D), -5.0),
                np.full((1, D), 5.0),
                np.full((1, D), -3.0),
                np.full((1, D), 3.0),
                seed,
                mode,
                extra={"uncertainty_handling": True},
            )
            print(
                f"noisy-sphere3 seed={seed} {mode:11s} fval={res['fval']:.3e} "
                f"fevals={res['func_count']} rebuilds={log.rebuilds} extra={log.extra_rebuilds}"
            )
            if mode == "as_is":
                r = np.array(log.ratio)
                print(
                    f"   refit-time calls={log.calls}; unreliable py-vs-MATLAB(same SD)={log.dis_py_vs_mat_latent_unrel}, "
                    f"refit py-vs-MATLAB(same SD)={log.dis_py_vs_mat_latent_refit}; "
                    f"MATLAB latent-vs-predictive SD: unrel={log.dis_mat_latent_vs_true_unrel}, refit={log.dis_mat_latent_vs_true_refit}"
                )
                print(
                    f"   ys_pred/fs_latent: median={np.median(r):.3g} q90={np.quantile(r, 0.9):.3g} max={np.max(r):.3g}"
                )
