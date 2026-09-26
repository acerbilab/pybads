import warnings

import gpyreg as gpr
import numpy as np
import scipy.stats as st
from gpyreg.f_min_fill import f_min_fill

print("gpyreg.__file__ =", gpr.__file__)
X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
for side in (1, -1):
    for z in (37, 37.6, 38, 60):
        lower, upper = (z, z + 10) if side > 0 else (-z - 10, -z)
        gp = gpr.GP(
            D=1,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )
        m0 = lower + 0.5 if side > 0 else upper - 0.5
        hyp = np.array([0.0, 0.0, np.log(0.1), m0])
        gp.update(X_new=X, y_new=np.sin(X) + m0, hyp=hyp[None, :])
        b = {
            n: (-np.inf, np.inf)
            for n in [
                "covariance_log_lengthscale",
                "covariance_log_outputscale",
                "noise_log_scale",
                "mean_const",
            ]
        }
        b["mean_const"] = (lower, upper)
        gp.set_bounds(b)
        pr = {n: None for n in b}
        pr["mean_const"] = ("gaussian", (0.0, 1.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gp.set_priors(pr)
            lp = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
            lpg = gp.log_posterior(hyp, compute_grad=True)
        hprior = {
            "mu": np.array([0.0]),
            "sigma": np.array([1.0]),
            "df": np.array([0.0]),
            "a": np.array([np.nan]),
            "b": np.array([np.nan]),
        }
        LB, UB = np.array([lower]), np.array([upper])
        Xd, _ = f_min_fill(
            lambda x: 0.0,
            np.array([[m0]]),
            LB,
            UB,
            LB,
            UB,
            hprior,
            64,
            rng=np.random.default_rng(0),
        )
        # reference: Mills asymptotic
        zz = z
        ref = (
            -(zz**2) / 2
            - np.log(zz)
            - 0.5 * np.log(2 * np.pi)
            + np.log(
                1 - 1 / zz**2 + 3 / zz**4 - 15 / zz**6 + 105 / zz**8
            )
        )
        ref_lp = st.norm.logpdf(m0) - ref
        print(
            f"side {side:+d} z {z:5}: mass {gp.normalization_constants[3]:.3e} lp {lp:.12g} ref {ref_lp:.12g} relerr {abs(lp-ref_lp)/abs(ref_lp):.1e} grad finite {np.all(np.isfinite(lpg[1]))} warnings {len(w)}; design finite {np.all(np.isfinite(Xd))} in [{Xd.min():.4f}, {Xd.max():.4f}]"
        )
