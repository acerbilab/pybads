"""gpyreg GP.predict (as PyBADS calls it, add_noise=False) vs MATLAB mygp's fmu, fs2 (latent), both noise regimes."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import gpyreg as gpr
from check_kernel import covRQard_fast

rng = np.random.default_rng(3)
worst = {}
for level in (0, 2):
    for lognoise in (-2.0, -7.5):
        w = [0, 0]
        for t in range(20):
            D = int(rng.integers(1, 5))
            n = int(rng.integers(5, 30))
            m = 7
            X = rng.uniform(-1, 1, (n, D))
            y = np.sum(X**2, 1, keepdims=True)
            Xs = rng.uniform(-1.5, 1.5, (m, D))
            s = rng.uniform(0.01, 0.3, (n, 1)) if level == 2 else None
            gp = gpr.GP(
                D=D,
                covariance=gpr.covariance_functions.RationalQuadraticARD(),
                mean=gpr.mean_functions.ConstantMean(),
                noise=gpr.noise_functions.GaussianNoise(
                    constant_add=True, user_provided_add=(level == 2)
                ),
            )
            hyp = np.concatenate(
                [
                    rng.uniform(-1, 0.5, D),
                    [rng.uniform(-0.5, 1)],
                    [rng.uniform(-2, 2)],
                    [lognoise],
                    [rng.normal()],
                ]
            )
            gp.update(
                X_new=X,
                y_new=y,
                s2_new=(s**2 if s is not None else None),
                hyp=hyp[None, :],
            )
            mu, s2 = gp.predict(Xs)
            # MATLAB: infExact posterior, mygp prediction (fmu, fs2), lik called with s=[] at test points
            K, _ = covRQard_fast(hyp[: D + 2], X)
            Ks, _ = covRQard_fast(hyp[: D + 2], X, Xs)
            sn2 = np.exp(2 * hyp[D + 2]) + (
                s.ravel() ** 2 if s is not None else 0
            )
            C = K + np.diag(np.broadcast_to(sn2, (n,)))
            alpha = np.linalg.solve(C, y - hyp[-1])
            fmu = hyp[-1] + Ks.T @ alpha
            fs2 = np.maximum(
                np.exp(2 * hyp[D]) - np.sum(Ks * np.linalg.solve(C, Ks), 0), 0
            )
            w[0] = max(w[0], np.max(np.abs(mu.ravel() - fmu.ravel())))
            w[1] = max(
                w[1], np.max(np.abs(s2.ravel() - fs2)) / np.exp(2 * hyp[D])
            )
        worst[(level, lognoise)] = w
print("max |mu diff|, max rel |s2 diff| by (level, log noise):", worst)
