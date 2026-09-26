"""Q1: gpyreg RationalQuadraticARD vs an independent transcription of GPML covRQard,
and its derivatives vs central finite differences."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.covariance_functions import RationalQuadraticARD


def sq_dist(a, b):  # GPML sq_dist: columns are points (D x n), returns n x m
    return (
        np.sum(a**2, 0)[:, None] + np.sum(b**2, 0)[None, :] - 2 * a.T @ b
    ).clip(min=0)


def covRQard(hyp, x, z=None, i=None):
    """Transcription of GPML v3.6 cov/covRQard.m (x: n x D)."""
    n, D = x.shape
    ell = np.exp(hyp[:D])
    sf2 = np.exp(2 * hyp[D])
    alpha = np.exp(hyp[D + 1])
    xeqz = z is None
    if xeqz:
        D2 = sq_dist(np.diag(1 / ell) @ x.T, np.diag(1 / ell) @ x.T)
    else:
        D2 = sq_dist(np.diag(1 / ell) @ x.T, np.diag(1 / ell) @ z.T)
    if i is None:
        return sf2 * (1 + 0.5 * D2 / alpha) ** (-alpha)
    if i < D:
        zz = x if xeqz else z
        return (
            sf2
            * (1 + 0.5 * D2 / alpha) ** (-alpha - 1)
            * sq_dist(x[:, i][None, :] / ell[i], zz[:, i][None, :] / ell[i])
        )
    if i == D:
        return 2 * sf2 * (1 + 0.5 * D2 / alpha) ** (-alpha)
    K = 1 + 0.5 * D2 / alpha
    return sf2 * K ** (-alpha) * (0.5 * D2 / K - alpha * np.log(K))


rng = np.random.default_rng(0)
cov = RationalQuadraticARD()
worst = dict(K=0, Ks=0, diag=0, dK_vs_gpml=0, dK_vs_fd=0)
for trial in range(50):
    D = int(rng.integers(1, 7))
    N = int(rng.integers(2, 25))
    M = int(rng.integers(1, 10))
    X = rng.normal(size=(N, D)) * rng.uniform(0.1, 3)
    Xs = rng.normal(size=(M, D))
    hyp = np.concatenate(
        [rng.normal(-0.5, 1.5, D), [rng.normal(0, 2)], [rng.uniform(-5, 5)]]
    )
    K, dK = cov.compute(hyp, X, compute_grad=True)
    Kg = covRQard(hyp, X)
    worst["K"] = max(
        worst["K"],
        np.max(np.abs(K - Kg) / np.maximum(1e-300, np.abs(Kg).max())),
    )
    Ks = cov.compute(hyp, X, Xs)
    worst["Ks"] = max(
        worst["Ks"],
        np.max(np.abs(Ks - covRQard(hyp, X, Xs)) / np.abs(Kg).max()),
    )
    kd = cov.compute(hyp, Xs, compute_diag=True)
    worst["diag"] = max(
        worst["diag"],
        np.max(np.abs(kd[:, 0] - np.exp(2 * hyp[D]))) / np.exp(2 * hyp[D]),
    )
    scale = np.abs(Kg).max()
    for i in range(D + 2):
        worst["dK_vs_gpml"] = max(
            worst["dK_vs_gpml"],
            np.max(np.abs(dK[:, :, i] - covRQard(hyp, X, i=i))) / scale,
        )
        h = 1e-6
        e = np.zeros_like(hyp)
        e[i] = h
        fd = (cov.compute(hyp + e, X) - cov.compute(hyp - e, X)) / (2 * h)
        worst["dK_vs_fd"] = max(
            worst["dK_vs_fd"], np.max(np.abs(dK[:, :, i] - fd)) / scale
        )
print("max relative discrepancies over 50 random cases:")
for k, v in worst.items():
    print(f"  {k}: {v:.3e}")
# Order of hyperparameter blocks in the GP vector PyBADS builds
gp = gpyreg.GP(
    D=3,
    covariance=cov,
    mean=gpyreg.mean_functions.ConstantMean(),
    noise=gpyreg.noise_functions.GaussianNoise(constant_add=True),
)
print(
    "hyperparameter layout:",
    [(k, len(v)) for k, v in gp.get_hyperparameters()[0].items()],
)
