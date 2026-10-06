import gpyreg as gpr
import numpy as np
from gpyreg import gaussian_process as gp_module

print("gpyreg.__file__ =", gpr.__file__)
rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, size=(20, 1))
y = np.sin(3 * X) + 0.1 * rng.standard_normal((20, 1))
rec = []
real = gp_module.f_min_fill


def recording(*a, **k):
    X0, y0 = real(*a, **k)
    rec.append(y0.copy())
    return X0, y0


gp_module.f_min_fill = recording
for switch in (False, True):
    for opts_N in (1, 2):
        g = gpr.GP(
            D=1,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
            raise_on_cholesky_failure=switch,
        )
        b = {
            "covariance_log_lengthscale": (-2.0, 2.0),
            "covariance_log_outputscale": (-2.0, 14.0),
            "noise_log_scale": (np.log(2e-3), 0.0),
            "mean_const": (-5.0, 5.0),
        }
        g.set_bounds(b)
        rec.clear()
        try:
            hyp, res, _ = g.fit(
                X,
                y,
                options={"n_samples": 0, "init_N": 64, "opts_N": opts_N},
                rng=np.random.default_rng(0),
            )
            y0 = rec[0]
            print(
                switch,
                opts_N,
                "ok",
                np.round(hyp, 3),
                "inf:",
                np.sum(np.isinf(y0)),
                "finite:",
                np.sum(np.isfinite(y0)),
                "mult",
                g.posteriors[0].sn2_mult,
            )
        except Exception as e:
            y0 = rec[0] if rec else None
            print(
                switch,
                opts_N,
                "raised",
                type(e).__name__,
                e,
                None if y0 is None else np.sum(np.isinf(y0)),
            )
