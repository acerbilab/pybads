import gpyreg as gpr
import numpy as np
from gpyreg import gaussian_process as gp_module

print("gpyreg.__file__ =", gpr.__file__)
designs = []
real_fill = gp_module.f_min_fill


def rec(*a, **k):
    X0, y0 = real_fill(*a, **k)
    designs.append(y0.copy())
    return X0, y0


gp_module.f_min_fill = rec
rng = np.random.default_rng(5)
X2 = 1.0 + 0.05 * rng.standard_normal((45, 2))
y2 = (100 * (X2[:, 1] - X2[:, 0] ** 2) ** 2 + (1 - X2[:, 0]) ** 2)[
    :, None
] + 1e3
for switch in (False, True):
    gp = gpr.GP(
        D=2,
        covariance=gpr.covariance_functions.RationalQuadraticARD(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        raise_on_cholesky_failure=switch,
    )
    gp.set_bounds(
        {
            "covariance_log_lengthscale": (
                np.full(2, np.log(1e-6)),
                np.full(2, np.log(20.0)),
            ),
            "covariance_log_outputscale": (np.log(1e-3), 20.0),
            "covariance_log_shape": (np.array([-5.0]), np.array([5.0])),
            "noise_log_scale": (np.log(1e-3) - 1, 5.0),
            "mean_const": (np.array([-1e4]), np.array([1e4])),
        }
    )
    designs.clear()
    try:
        hyp, res, _ = gp.fit(
            X2,
            y2,
            options={"n_samples": 0, "init_N": 256, "opts_N": 2},
            rng=np.random.default_rng(2),
        )
        print(
            switch,
            "ok",
            np.round(hyp[0], 3),
            "nlZ",
            res.fun,
            "design inf",
            int(np.sum(np.isinf(designs[0]))),
            "sn2_mult",
            gp.posteriors[0].sn2_mult,
        )
    except np.linalg.LinAlgError as e:
        print(
            switch,
            "raised",
            e,
            "design inf",
            int(np.sum(np.isinf(designs[0]))) if designs else None,
        )
