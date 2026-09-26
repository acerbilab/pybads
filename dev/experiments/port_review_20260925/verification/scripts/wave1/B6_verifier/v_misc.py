"""Small checks: F7 (effective radius), F9 (upper_gp_length_factor), F8
(fit_lik), F10 (mean functions), C-F6 (gp_cov_prior), C-F7 (ddof), F6/C-F4
(flat initial design)."""

import sys

import numpy as np
from common import bads_mod, gpt, rosenbrock, sphere
from scipy.optimize import brentq

from pybads import BADS

part = sys.argv[1] if len(sys.argv) > 1 else "all"


def bads(fun, D, opts, x0=None, lb=-5, ub=5, plb=-2, pub=2):
    x0 = np.full(D, 0.7) if x0 is None else x0
    o = dict(display="off", random_seed=0, max_fun_evals=200)
    o.update(opts)
    return BADS(
        fun,
        x0,
        np.full(D, lb * 1.0),
        np.full(D, ub * 1.0),
        np.full(D, plb * 1.0),
        np.full(D, pub * 1.0),
        options=o,
    )


if part in ("all", "F7"):
    print(
        "== F7: effective radius r_e = sqrt(a (exp(1/a) - 1)) "
        "(gaussian_process_train.py:504; MATLAB gpupdate.m:317)"
    )
    rq = lambda r, a: (1 + r**2 / (2 * a)) ** (-a)  # gpyreg = GPML covRQard
    for la in (-5, -2, -1, 0, 1, 3, 5):
        a = np.exp(la)
        r_e = np.sqrt(a * np.expm1(1 / a)) if la > -5 else np.inf
        k_sqrt2 = rq(np.sqrt(2) * r_e, a) if np.isfinite(r_e) else np.nan
        print(
            f"   log a={la:3d}: r_e={r_e:.4g}; k(sqrt2*r_e)={k_sqrt2:.6f} "
            f"(e^-1={np.exp(-1):.6f}); k(r_e)={rq(r_e, a):.3f}"
        )
    # MATLAB's Matern constants (gpupdate.m:319-325): root of k = e^-1, / sqrt 2
    m3 = lambda x: (1 + np.sqrt(3) * x) * np.exp(-np.sqrt(3) * x) - np.exp(-1)
    m5 = lambda x: (1 + np.sqrt(5) * x + 5 / 3 * x**2) * np.exp(
        -np.sqrt(5) * x
    ) - np.exp(-1)
    m1 = lambda x: np.exp(-x) - np.exp(-1)
    print(
        f"   matern1 root/sqrt2={brentq(m1, .1, 5)/np.sqrt(2):.15f} "
        f"(MATLAB 1/sqrt(2)={1/np.sqrt(2):.15f})"
    )
    print(
        f"   matern3 root/sqrt2={brentq(m3, .1, 5)/np.sqrt(2):.15f} "
        f"(MATLAB 0.876179713323485)"
    )
    print(
        f"   matern5 root/sqrt2={brentq(m5, .1, 5)/np.sqrt(2):.15f} "
        f"(MATLAB 0.918524648109253)"
    )
    se = lambda x: np.exp(-(x**2) / 2) - np.exp(-1)
    print(
        f"   SE (GPML exp(-r^2/2)) root/sqrt2={brentq(se, .1, 5)/np.sqrt(2):.6f}"
        " (MATLAB 'otherwise' 1)"
    )

if part in ("all", "F9"):
    print("== F9: upper_gp_length_factor")
    orig = gpt._gp_hyp
    got = {}

    def w(*a, **k):
        gp, h, n = orig(*a, **k)
        got["b"] = gp.get_bounds()["covariance_log_lengthscale"]
        return gp, h, n

    gpt._gp_hyp = w
    try:
        for f in (0, 0.05, 5.0):
            bads(
                sphere, 2, {"upper_gp_length_factor": f, "max_fun_evals": 8}
            ).optimize()
            print(
                f"   factor {f}: length-scale bounds {np.round(got['b'], 3)}"
            )
    finally:
        gpt._gp_hyp = orig

if part in ("all", "F8"):
    print("== F8: fit_lik=False")
    try:
        bads(sphere, 2, {"fit_lik": False}).optimize()
        print("   ran")
    except Exception as e:
        print("   raises", type(e).__name__, "-", e)

if part in ("all", "F10"):
    print("== F10: gp_mean_fun")
    for name in ("se", "negquadse", "zero"):
        try:
            r = bads(
                sphere, 2, {"gp_mean_fun": name, "max_fun_evals": 60}
            ).optimize()
            print(f"   {name}: ran, fval={r['fval']:.3g}")
        except Exception as e:
            print(f"   {name}: raises {type(e).__name__} - {e}")
    x0 = np.random.default_rng(101).uniform(-2, 2, 2)
    try:
        r = BADS(
            rosenbrock,
            x0,
            np.full(2, -5.0),
            np.full(2, 5.0),
            np.full(2, -2.0),
            np.full(2, 2.0),
            options=dict(
                display="off",
                random_seed=1,
                max_fun_evals=200,
                gp_mean_fun="negquad",
            ),
        ).optimize()
        print(f"   negquad rosenbrock D=2 s1: ran, fval={r['fval']:.3g}")
    except Exception as e:
        print(f"   negquad rosenbrock D=2 s1: raises {type(e).__name__} - {e}")

if part in ("all", "CF6"):
    print("== C-F6: gp_cov_prior")
    orig = bads_mod.local_gp_fitting
    for val in ("iso", "ard", "foo"):
        seen = set()

        def w(*a, **k):
            out = orig(*a, **k)
            p = out[0].get_priors()["covariance_log_lengthscale"]
            seen.add(tuple(np.round(np.ravel(p[1][0]), 3)))
            return out

        bads_mod.local_gp_fitting = w
        try:
            bads(
                sphere, 2, {"gp_cov_prior": val, "max_fun_evals": 60}
            ).optimize()
            print(
                f"   {val}: ran; distinct length-scale prior centres over "
                f"rebuilds: {len(seen)} e.g. {sorted(seen)[:3]}"
            )
        except Exception as e:
            print(f"   {val}: raises {type(e).__name__} - {e}")
        finally:
            bads_mod.local_gp_fitting = orig

if part in ("all", "CF7"):
    print("== C-F7: log std with ddof 0 vs ddof 1, 0.5*log(N/(N-1))")
    for N in (6, 10, 25, 50, 110, 200):
        print(f"   N={N}: {0.5*np.log(N/(N-1)):.4f} (prior SD 2)")

if part in ("all", "F6"):
    print("== F6 / C-F4: a target flat on the initial design")

    def plateau(x):
        x = np.atleast_1d(x)
        d2 = np.sum((x - 3.0) ** 2)
        return float(d2) if d2 < 0.25 else 1000.0

    for D in (2, 3):
        try:
            r = bads(
                plateau, D, {}, x0=np.full(D, 0.3), lb=-5, ub=5, plb=-1, pub=1
            ).optimize()
            print(f"   D={D}: ran, fval={r['fval']:.3g}")
        except Exception as e:
            print(f"   D={D}: raises {type(e).__name__} - {e}")
