import gpyreg as gpr
import numpy as np

print("gpyreg.__file__ =", gpr.__file__)
rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, size=(20, 1))
y = np.sin(3 * X)


def gp():
    return gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )


for log_sf in [4, 6, 8, 10, 12, 14]:
    for log_sn in [np.log(2e-3), np.log(1e-8)]:
        g = gp()
        hyp = np.array([[0.0, log_sf, log_sn, 0.0]])
        try:
            g.update(X_new=X, y_new=y, hyp=hyp)
            print(
                log_sf,
                round(log_sn, 2),
                g.posteriors[0].sn2_mult,
                g.posteriors[0].L_chol,
            )
        except Exception as e:
            print(log_sf, round(log_sn, 2), "raises", type(e).__name__)
