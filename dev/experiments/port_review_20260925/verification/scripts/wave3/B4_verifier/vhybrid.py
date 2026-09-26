"""MATLAB's UpdateTarget prediction (bads.m:1299-1302 -> gppred.m:39-47 ->
mygp.m:122-123, 146-187): the posterior (alpha, L, sW) of the GP as it is,
with Ks, kss and the mean of the hyperparameters `hyp`."""
import numpy as np
from scipy.linalg import solve_triangular


def hybrid_predict(gp, hyp, x):
    x = np.atleast_2d(x)
    D = gp.X.shape[1]
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    mean_N = gp.mean.hyperparameter_count(D)
    hyp = np.ravel(hyp)
    post = gp.posteriors[0]
    kss = gp.covariance.compute(hyp[:cov_N], x, compute_diag=True)[:, 0]
    Ks = gp.covariance.compute(hyp[:cov_N], gp.X, x)
    ms = np.ravel(
        gp.mean.compute(hyp[cov_N + noise_N : cov_N + noise_N + mean_N], x)
    )
    fmu = ms + (Ks.T @ post.alpha)[:, 0]
    if post.L_chol:
        V = solve_triangular(post.L, post.sW * Ks, trans=1)
        fs2 = kss - np.sum(V * V, 0)
    else:
        # L holds the negative inverse of K + sn2 I (mygp.m:181-184,
        # kss + sum(Ks.*(L*Ks))); the same quantity, computed through the
        # Cholesky factor of that matrix to avoid the rounding of the inverse
        F = gp._GP__low_noise_factor(0)
        V = solve_triangular(F, Ks, trans=1)
        fs2 = kss - np.sum(V * V, 0)
    return fmu, np.maximum(fs2, 0)
