"""Q1: gpyreg RationalQuadraticARD vs a transcription of GPML covRQard_fast.m."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.covariance_functions import RationalQuadraticARD


def sq_dist(a, b=None):
    # GPML sq_dist on column-major (D x n) arrays, with mean subtraction
    if b is None:
        mu = a.mean(axis=1, keepdims=True)
        a = a - mu
        b = a
    else:
        n, m = a.shape[1], b.shape[1]
        mu = (m / (n + m)) * b.mean(axis=1, keepdims=True) + (
            n / (n + m)
        ) * a.mean(axis=1, keepdims=True)
        a = a - mu
        b = b - mu
    C = (a * a).sum(0)[:, None] + ((b * b).sum(0)[None, :] - 2 * a.T @ b)
    return np.maximum(C, 0)


def covRQard_fast(hyp, x, z=None):
    n, D = x.shape
    ell = np.exp(hyp[:D])
    sf2 = np.exp(2 * hyp[D])
    alpha = np.exp(hyp[D + 1])
    if z is None:
        D2 = sq_dist(np.diag(1 / ell) @ x.T)
    else:
        D2 = sq_dist(np.diag(1 / ell) @ x.T, np.diag(1 / ell) @ z.T)
    M = 1 + 0.5 * D2 / alpha
    K = sf2 * M ** (-alpha)
    if z is not None:
        return K, None
    # sq_dist_fast(x,[],ell): per-dimension squared distances divided by ell^2
    a = (x / ell)[None, :, :]  # 1 x n x D
    b = (x / ell)[:, None, :]  # n x 1 x D
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    C = np.maximum(a * a + (b * b - 2 * a * b), 0)
    dK = np.zeros((n, n, D + 2))
    dK[:, :, :D] = (K / M)[:, :, None] * C
    dK[:, :, D] = 2 * K
    dK[:, :, D + 1] = K * (0.5 * D2 / M - alpha * np.log(M))
    return K, dK


rng = np.random.default_rng(0)
k = RationalQuadraticARD()
maxerr = {"K": 0, "dK": 0, "Kxz": 0, "fd": 0, "diag": 0}
for trial in range(200):
    D = int(rng.integers(1, 8))
    n = int(rng.integers(2, 25))
    m = int(rng.integers(1, 10))
    X = rng.normal(size=(n, D)) * rng.uniform(0.1, 3)
    Z = rng.normal(size=(m, D))
    hyp = np.concatenate(
        [rng.uniform(-2, 2, D), [rng.uniform(-2, 3)], [rng.uniform(-5, 5)]]
    )
    K, dK = k.compute(hyp, X, compute_grad=True)
    Km, dKm = covRQard_fast(hyp, X)
    scale = np.exp(2 * hyp[D])
    maxerr["K"] = max(maxerr["K"], np.max(np.abs(K - Km)) / scale)
    maxerr["dK"] = max(maxerr["dK"], np.max(np.abs(dK - dKm)) / scale)
    Kxz = k.compute(hyp, X, Z)
    Kxzm, _ = covRQard_fast(hyp, X, Z)
    maxerr["Kxz"] = max(maxerr["Kxz"], np.max(np.abs(Kxz - Kxzm)) / scale)
    Kd = k.compute(hyp, X, compute_diag=True)
    maxerr["diag"] = max(maxerr["diag"], np.max(np.abs(Kd.ravel() - scale)))
    # central finite differences of gpyreg K
    h = 1e-6
    for j in range(D + 2):
        e = np.zeros_like(hyp)
        e[j] = h
        fd = (k.compute(hyp + e, X) - k.compute(hyp - e, X)) / (2 * h)
        maxerr["fd"] = max(
            maxerr["fd"], np.max(np.abs(fd - dK[:, :, j])) / scale
        )
print("max relative errors over 200 random cases:", maxerr)
# parameterization check: hyp = [log ell, log sf, log alpha]; value at a known point
hyp = np.array([np.log(2.0), np.log(3.0), np.log(0.5)])
X = np.array([[0.0], [1.0]])
K = k.compute(hyp, X)
print(
    "K(0,1) gpyreg:",
    K[0, 1],
    " expected 9*(1+0.25/(2*0.5))^-0.5 =",
    9 * (1 + 0.25 / (2 * 0.5)) ** -0.5,
)
