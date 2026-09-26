"""My Python transcriptions of MATLAB BADS utilities at 74919c0 (written for this verification)."""
import numpy as np
from scipy.special import gammaincinv
from scipy.stats import norm


def kurtosis_biased(x):
    x = np.asarray(x, float)
    m = x.mean()
    return np.mean((x - m) ** 4) / np.mean((x - m) ** 2) ** 2


def swtest(x, alpha):
    """utils/swtest.m: returns (H, p)."""
    x = np.sort(np.asarray(x, float).ravel())
    n = x.size
    mtilde = norm.ppf((np.arange(1, n + 1) - 3 / 8) / (n + 1 / 4))
    weights = np.zeros(n)
    if kurtosis_biased(x) > 3:
        weights = mtilde / np.sqrt(mtilde @ mtilde)
        W = (weights @ x) ** 2 / ((x - x.mean()) @ (x - x.mean()))
        nu = np.log(n)
        u1 = np.log(nu) - nu
        u2 = np.log(nu) + 2 / nu
        mu = -1.2725 + 1.0521 * u1
        sigma = 1.0308 - 0.26758 * u2
        z = (np.log(1 - W) - mu) / sigma
        p = 1 - norm.cdf(z)
    else:
        c = mtilde / np.sqrt(mtilde @ mtilde)
        u = 1 / np.sqrt(n)
        P1 = [-2.706056, 4.434685, -2.071190, -0.147981, 0.221157, c[n - 1]]
        P2 = [-3.582633, 5.682633, -1.752461, -0.293762, 0.042981, c[n - 2]]
        P3 = [-0.0006714, 0.0250540, -0.39978, 0.54400]
        P4 = [-0.0020322, 0.0627670, -0.77857, 1.38220]
        P5 = [0.00389150, -0.083751, -0.31082, -1.5861]
        P6 = [0.00303020, -0.082676, -0.48030]
        P7 = [0.459, -2.273]
        weights[n - 1] = np.polyval(P1, u)
        weights[0] = -weights[n - 1]
        if n > 5:
            weights[n - 2] = np.polyval(P2, u)
            weights[1] = -weights[n - 2]
            count = 3
            phi = (
                mtilde @ mtilde
                - 2 * mtilde[n - 1] ** 2
                - 2 * mtilde[n - 2] ** 2
            ) / (1 - 2 * weights[n - 1] ** 2 - 2 * weights[n - 2] ** 2)
        else:
            count = 2
            phi = (mtilde @ mtilde - 2 * mtilde[n - 1] ** 2) / (
                1 - 2 * weights[n - 1] ** 2
            )
        if n == 3:
            weights[0] = 1 / np.sqrt(2)
            weights[n - 1] = -weights[0]
            phi = 1
        weights[count - 1 : n - count + 1] = mtilde[
            count - 1 : n - count + 1
        ] / np.sqrt(phi)
        W = (weights @ x) ** 2 / ((x - x.mean()) @ (x - x.mean()))
        if 4 <= n <= 11:
            mu = np.polyval(P3, n)
            sigma = np.exp(np.polyval(P4, n))
            gam = np.polyval(P7, n)
            s = -np.log(gam - np.log(1 - W))
        elif n > 11:
            mu = np.polyval(P5, np.log(n))
            sigma = np.exp(np.polyval(P6, np.log(n)))
            s = np.log(1 - W)
        else:
            mu, sigma, s = 0.0, 1.0, 0.0
        p = 1 - norm.cdf((s - mu) / sigma)
        if n == 3:
            p = 6 / np.pi * (np.arcsin(np.sqrt(W)) - np.arcsin(np.sqrt(3 / 4)))
    return int(alpha >= p), p


def gppredcheck(fval, ymu, ys, alpha):
    """utils/gppredcheck.m on stats with last = len(fval)."""
    n = len(fval)
    if n == 0:
        return 1
    z = (np.asarray(fval, float) - np.asarray(ymu, float)) / np.asarray(
        ys, float
    )
    if np.any(np.isnan(z)):
        return 1
    n = z.size
    if n < 3:
        chi2inv = lambda x, v: 2 * gammaincinv(v / 2, x)
        plo, phi = chi2inv(alpha / 2, n), chi2inv(1 - alpha / 2, n)
        t = np.sum(z**2)
        return int(t < plo or t > phi or np.any(np.isnan([plo, phi])))
    return swtest(z, alpha)[0]


def is_refit_time(
    funccount, nvars, lastfitgp, min_refit_time, fval, ymu, ys, alpha
):
    """bads.m IsRefitTime: returns (refit, unrel)."""
    try:
        unrel = gppredcheck(fval, ymu, ys, alpha)
    except Exception:
        unrel = 1
    refitperiod = max(10, 2 * nvars) if funccount < 200 else 5 * nvars
    refit = (
        lastfitgp < funccount - min_refit_time
        and (len(fval) >= refitperiod or unrel)
        and funccount > nvars
    )
    if refit:
        unrel = 0
    return bool(refit), bool(unrel)


def prctile1(x, p):
    x = np.sort(np.asarray(x, float).ravel())
    n = x.size
    r = p / 100 * n
    k = int(np.floor(r + 0.5))
    if k < 1:
        return x[0]
    if k >= n:
        return x[-1]
    r = r - k
    return (0.5 - r) * x[k - 1] + (0.5 + r) * x[k]
