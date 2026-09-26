"""Python transcriptions of MATLAB BADS functions (74919c0), for comparison.

gppredcheck.m, swtest.m, IsRefitTime (bads.m:1223-1254).
"""
import numpy as np
from scipy.special import erfc, gammaincinv
from scipy.stats import norm


def swtest(x, alpha=0.05):
    """utils/swtest.m, line by line. Returns (H, pValue, W)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if x.size < 3:
        raise ValueError("Sample vector X must have at least 3 valid obs.")
    normcdf = lambda z: 0.5 * erfc(-z / np.sqrt(2))
    x = np.sort(x)
    n = x.size
    mtilde = norm.ppf((np.arange(1, n + 1) - 3 / 8) / (n + 1 / 4))
    weights = np.zeros(n)
    # MATLAB kurtosis (Statistics Toolbox), flag=1 (biased): m4/m2^2
    xc = x - x.mean()
    kurt = np.mean(xc**4) / np.mean(xc**2) ** 2
    if kurt > 3:
        weights = mtilde / np.sqrt(mtilde @ mtilde)
        W = (weights @ x) ** 2 / (xc @ xc)
        nu = np.log(n)
        u1 = np.log(nu) - nu
        u2 = np.log(nu) + 2 / nu
        mu = -1.2725 + 1.0521 * u1
        sigma = 1.0308 - 0.26758 * u2
        newSF = np.log(1 - W)
        z = (newSF - mu) / sigma
        p = 1 - normcdf(z)
        branch = "SF"
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
        # weights(count : n-count+1) = mtilde(count : n-count+1)/sqrt(phi)
        weights[count - 1 : n - count + 1] = mtilde[
            count - 1 : n - count + 1
        ] / np.sqrt(phi)
        W = (weights @ x) ** 2 / (xc @ xc)
        newn = np.log(n)
        if 4 <= n <= 11:
            mu = np.polyval(P3, n)
            sigma = np.exp(np.polyval(P4, n))
            gam = np.polyval(P7, n)
            newSW = -np.log(gam - np.log(1 - W))
        elif n > 11:
            mu = np.polyval(P5, newn)
            sigma = np.exp(np.polyval(P6, newn))
            newSW = np.log(1 - W)
        else:  # n == 3
            mu, sigma, newSW = 0.0, 1.0, 0.0
        z = (newSW - mu) / sigma
        p = 1 - normcdf(z)
        if n == 3:
            p = 6 / np.pi * (np.arcsin(np.sqrt(W)) - np.arcsin(np.sqrt(3 / 4)))
        branch = "SW"
    H = alpha >= p
    return bool(H), p, W, branch


def gppredcheck(fval, ymu, ys, alpha):
    """utils/gppredcheck.m: h = 1 means the GP is unreliable."""
    n = len(fval)
    if n == 0:
        return True
    z = (np.asarray(fval, float) - np.asarray(ymu, float)) / np.asarray(
        ys, float
    )
    if np.any(np.isnan(z)):
        return True
    n = z.size
    if n < 3:
        chi2inv = lambda x, v: 2 * gammaincinv(v / 2, x)
        plo = chi2inv(alpha / 2, n)
        phi = chi2inv(1 - alpha / 2, n)
        t = np.sum(z**2)
        return bool(t < plo or t > phi or np.isnan(plo) or np.isnan(phi))
    h, _, _, _ = swtest(z, alpha)
    return h


def is_refit_time(lastfitgp, funccount, n_stats, unrel, nvars, min_refit_time):
    """IsRefitTime (bads.m:1236-1244), given gppredcheck's verdict."""
    refitperiod = max(10, nvars * 2) if funccount < 200 else nvars * 5
    return (
        lastfitgp < funccount - min_refit_time
        and (n_stats >= refitperiod or unrel)
        and funccount > nvars
    )
