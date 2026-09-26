"""gpyreg's log posterior for the GP PyBADS builds vs a transcription of MATLAB's
infPrior_fast(infExact_fastrobust, likGaussHe, meanConst, covRQard_fast, priorGauss), and prctile1 vs hazen."""

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import gpyreg as gpr
from check_kernel import covRQard_fast  # transcription


def solve_chol(L, B):
    return np.linalg.solve(L, np.linalg.solve(L.T, B))


def matlab_nlZ(hyp, D, X, y, s, prior_mu, prior_s2):
    """hyp = [log ell (D), log sf, log alpha, log sn, mean]; s = target SDs or None."""
    n = X.shape[0]
    hc = hyp[: D + 2]
    hl = hyp[D + 2]
    hm = hyp[D + 3]
    K, dK = covRQard_fast(hc, X)
    m = hm * np.ones((n, 1))
    sn2_base = np.exp(2 * hl)
    sn2 = sn2_base + s**2 if s is not None else sn2_base
    sn2v = np.atleast_1d(sn2).ravel()
    Lchol = np.min(sn2v) >= 1e-6
    if Lchol:
        if np.isscalar(sn2) or np.size(sn2) == 1 and s is None:
            sn2div = float(sn2v[0])
            sn2_mat = np.eye(n)
        else:
            sn2div = np.min(sn2v)
            sn2_mat = np.diag(sn2v / sn2div)
        M = K / sn2div + sn2_mat
    else:
        M = K + (np.diag(sn2v) if sn2v.size > 1 else sn2v[0] * np.eye(n))
    L = np.linalg.cholesky(M).T
    sl = sn2div if Lchol else 1.0
    alpha = solve_chol(L, y - m) / sl
    nlZ = (
        ((y - m).T @ alpha).item() / 2
        + np.sum(np.log(np.diag(L)))
        + n * np.log(2 * np.pi * sl) / 2
    )
    Q = solve_chol(L, np.eye(n)) / sl - alpha @ alpha.T
    d = np.zeros(D + 4)
    for i in range(D + 2):
        d[i] = np.sum(Q * dK[:, :, i]) / 2
    if s is None:
        d[D + 2] = sn2_base * np.trace(Q)
    else:
        d[D + 2] = sn2_base * np.trace(
            Q
        )  # infExact_fastrobust.m:128 (same formula with s)
    d[D + 3] = -np.sum(alpha)
    # priorGauss contributions
    lp = (
        -((hyp - prior_mu) ** 2) / (2 * prior_s2)
        - np.log(2 * np.pi * prior_s2) / 2
    )
    dlp = -(hyp - prior_mu) / prior_s2
    return nlZ - lp.sum(), d - dlp


rng = np.random.default_rng(0)
for level in (0, 2):
    worst = [0, 0]
    for trial in range(30):
        D = int(rng.integers(1, 5))
        n = int(rng.integers(5, 30))
        X = rng.uniform(-1, 1, size=(n, D))
        y = (np.sum(X**2, 1) + 0.1 * rng.normal(size=n)).reshape(-1, 1)
        s = rng.uniform(0.05, 0.5, size=(n, 1)) if level == 2 else None
        gp = gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.RationalQuadraticARD(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(
                constant_add=True, user_provided_add=(level == 2)
            ),
        )
        mu = np.concatenate(
            [
                rng.normal(-1, 1, D),
                [rng.normal()],
                [1.0],
                [np.log(0.03)],
                [np.median(y)],
            ]
        )
        sd = np.concatenate([np.full(D, 2.0), [2.0], [1.0], [1.0], [0.5]])
        gp.set_priors(
            {
                "covariance_log_lengthscale": ("gaussian", (mu[:D], sd[:D])),
                "covariance_log_outputscale": (
                    "gaussian",
                    (mu[D : D + 1], sd[D : D + 1]),
                ),
                "covariance_log_shape": (
                    "gaussian",
                    (mu[D + 1 : D + 2], sd[D + 1 : D + 2]),
                ),
                "noise_log_scale": (
                    "gaussian",
                    (mu[D + 2 : D + 3], sd[D + 2 : D + 3]),
                ),
                "mean_const": ("gaussian", (mu[D + 3 :], sd[D + 3 :])),
            }
        )
        # MATLAB-like unbounded (so gpyreg's normalization constants are 1)
        gp.set_bounds(
            {
                k: (np.full(v, -np.inf), np.full(v, np.inf))
                for k, v in [
                    ("covariance_log_lengthscale", D),
                    ("covariance_log_outputscale", 1),
                    ("covariance_log_shape", 1),
                    ("noise_log_scale", 1),
                    ("mean_const", 1),
                ]
            }
        )
        hyp = np.concatenate(
            [
                rng.uniform(-1.5, 0.5, D),
                [rng.uniform(-1, 1)],
                [rng.uniform(-2, 2)],
                [rng.uniform(-4, -1)],
                [rng.normal()],
            ]
        )
        gp.update(
            X_new=X,
            y_new=y,
            s2_new=(s**2 if s is not None else None),
            hyp=hyp[None, :],
            compute_posterior=False,
        )
        lpost, dlpost = gp.log_posterior(hyp, compute_grad=True)
        mn, md = matlab_nlZ(hyp, D, X, y, s, mu, sd**2)
        worst[0] = max(worst[0], abs(-lpost - mn) / max(1, abs(mn)))
        worst[1] = max(
            worst[1], np.max(np.abs(-dlpost - md)) / max(1, np.max(np.abs(md)))
        )
    print(
        f"level {level}: max rel diff nlZ+prior {worst[0]:.2e}, gradient {worst[1]:.2e}"
    )


def prctile1(x, p):
    x = np.sort(np.asarray(x).ravel())
    n = x.size
    if p == 50:
        return (
            x[(n + 1) // 2 - 1] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2
        )
    r = (p / 100) * n
    k = int(np.floor(r + 0.5))
    if k < 1:
        return x[0]
    if k >= n:
        return x[-1]
    r = r - k
    return (0.5 - r) * x[k - 1] + (0.5 + r) * x[k]


worst = 0
for trial in range(2000):
    n = int(rng.integers(1, 60))
    x = rng.normal(size=n)
    p = float(rng.choice([90, 50, 15.87, rng.uniform(0, 100)]))
    worst = max(
        worst, abs(prctile1(x, p) - np.percentile(x, p, method="hazen"))
    )
print("prctile1 vs numpy hazen, max abs diff:", worst)
