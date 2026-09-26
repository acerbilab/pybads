import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads.search.grid_functions import force_to_grid, grid_units
from pybads.variable_transformer import VariableTransformer
from pybads.variable_transformer.variables_transformer import (
    maskindex as py_maskindex,
)

warnings.simplefilter("ignore")
REALMAX = np.finfo(float).max


def m_maskindex(v, idx):
    v = np.array(v, dtype=float, copy=True)
    v[..., ~idx] = 0
    return v


class MTrinfo:
    """Transcription of utils/transvars.m (create), 'dir' and 'inv'."""

    def __init__(self, nvars, lb, ub, plb, pub, logflag):
        lb, ub, plb, pub = [
            np.asarray(a, float).ravel() for a in (lb, ub, plb, pub)
        ]
        logct = np.asarray(logflag, float).ravel().copy()
        assert np.all(np.isfinite(np.r_[plb, pub]))
        assert np.all((lb <= plb) & (plb < pub) & (pub <= ub))
        for i in np.where(np.isnan(logct))[0]:
            logct[i] = float(
                np.all(np.array([lb[i], ub[i], plb[i], pub[i]]) > 0)
                and (pub[i] / plb[i] >= 10)
            )
        logct = logct.astype(bool)
        self.logct = logct
        self.old = dict(
            lb=lb.copy(), ub=ub.copy(), plb=plb.copy(), pub=pub.copy()
        )
        L, U, PL, PU = lb.copy(), ub.copy(), plb.copy(), pub.copy()
        L[logct] = np.log(L[logct])
        U[logct] = np.log(U[logct])
        PL[logct] = np.log(PL[logct])
        PU[logct] = np.log(PU[logct])
        mu = 0.5 * (PL + PU)
        gamma = 0.5 * (PU - PL)
        z = lambda x: m_maskindex((x - mu) / gamma, ~logct)
        zlog = lambda x: m_maskindex(
            (np.log(np.abs(x) + (x == 0)) - mu) / gamma, logct
        )
        s = logct.sum()
        if s == 0:
            g = z
            ginv = lambda y: gamma * y + mu
        elif s == nvars:
            g = zlog
            ginv = lambda y: np.minimum(REALMAX, np.exp(gamma * y + mu))
        else:
            g = lambda x: z(x) + zlog(x)
            ginv = lambda y: m_maskindex(gamma * y + mu, ~logct) + m_maskindex(
                np.minimum(REALMAX, np.exp(gamma * y + mu)), logct
            )
        lbt = lb.copy()
        lbt[~np.isfinite(lb)] = -1 / np.sqrt(np.spacing(1.0))
        ubt = ub.copy()
        ubt[~np.isfinite(ub)] = 1 / np.sqrt(np.spacing(1.0))
        ubt[~np.isfinite(ub) & logct] = 1e6
        t = [
            np.all(np.abs(ginv(g(v)) - v) < 1e-6) for v in (lbt, ubt, plb, pub)
        ]
        assert all(t), "Cannot invert"
        self.g, self.ginv = g, ginv
        self.lb, self.ub, self.plb, self.pub = g(lb), g(ub), g(plb), g(pub)
        self.mu, self.gamma = mu, gamma

    def dir(self, x):
        return np.minimum(np.maximum(self.g(x), self.lb), self.ub)

    def inv(self, y):
        return np.minimum(
            np.maximum(self.ginv(y), self.old["lb"]), self.old["ub"]
        )


def compare(label, lb, ub, plb, pub, logflag=None, N=2000, seed=1):
    D = lb.size
    lf = np.full(D, np.nan) if logflag is None else logflag
    M = MTrinfo(D, lb, ub, plb, pub, lf)
    P = VariableTransformer(
        D,
        lb.reshape(1, -1).copy(),
        ub.reshape(1, -1).copy(),
        plb.reshape(1, -1).copy(),
        pub.reshape(1, -1).copy(),
        lf.reshape(1, -1).copy(),
    )
    rng = np.random.default_rng(seed)
    # inputs: inside hard bounds (wide), plus plausible, plus bound points
    lo = np.where(np.isfinite(lb), lb, plb - 10 * (pub - plb))
    hi = np.where(np.isfinite(ub), ub, pub + 10 * (pub - plb))
    X = lo + (hi - lo) * rng.random((N, D))
    X = np.vstack(
        [
            X,
            lb,
            ub,
            plb,
            pub,
            np.where(np.isfinite(lb), lb, -1e300),
            np.where(np.isfinite(ub), ub, 1e300),
        ]
    )
    ym = M.dir(X)
    yp = P(X)
    Y = rng.uniform(-3, 3, (N, D))
    Y = np.vstack(
        [
            Y,
            M.lb,
            M.ub,
            -np.ones(D),
            np.ones(D),
            np.full(D, -np.inf),
            np.full(D, np.inf),
        ]
    )
    xm = M.inv(Y)
    xp = P.inverse_transf(Y)
    # grid_units row path and 1-row path
    gu = grid_units(X[:5], P)
    gu1 = grid_units(X[:1], P)
    rt = P.inverse_transf(P(X[:N]))
    ok_log = np.array_equal(M.logct, P.apply_log_t.ravel())

    def same(a, b):
        return np.array_equal(
            np.nan_to_num(a, nan=7.7), np.nan_to_num(b, nan=7.7)
        )

    print(
        f"[{label}] log m={M.logct.astype(int)} p={P.apply_log_t.ravel().astype(int)} eq={ok_log}; "
        f"bounds eq={same(M.lb,P.lb.ravel()) and same(M.ub,P.ub.ravel()) and same(M.plb,P.plb.ravel()) and same(M.pub,P.pub.ravel())} "
        f"lb_u={P.lb.ravel()} ub_u={P.ub.ravel()}; dir max|diff|={np.nanmax(np.abs(np.where(np.isfinite(ym), ym-yp, 0))):.3g} "
        f"exact={same(ym, yp)}; inv max|diff|={np.nanmax(np.abs(np.where(np.isfinite(xm), xm-xp, 0))):.3g} exact={same(xm,xp)}; "
        f"grid_units==dir {np.array_equal(gu, yp[:5])} {np.array_equal(gu1, yp[:1])}; roundtrip max rel err={np.max(np.abs(rt-X[:N])/np.maximum(1,np.abs(X[:N]))):.2e}"
    )


compare(
    "linear finite",
    np.array([-10.0, 0.0, -3.0]),
    np.array([10.0, 5.0, 7.0]),
    np.array([-5.0, 1.0, -1.0]),
    np.array([5.0, 4.0, 2.0]),
)
compare(
    "linear inf",
    np.array([-np.inf, -np.inf]),
    np.array([np.inf, np.inf]),
    np.array([-5.0, 1.0]),
    np.array([5.0, 4.0]),
)
compare(
    "mixed log/lin",
    np.array([0.01, -10.0, 1.0]),
    np.array([1000.0, 10.0, 50.0]),
    np.array([0.1, -5.0, 2.0]),
    np.array([100.0, 5.0, 40.0]),
)
compare(
    "all log",
    np.array([0.01, 1.0]),
    np.array([1000.0, 50.0]),
    np.array([0.1, 2.0]),
    np.array([100.0, 40.0]),
)
compare(
    "log ub=inf (MATLAB only)",
    np.array([0.01, -np.inf]),
    np.array([np.inf, np.inf]),
    np.array([0.1, -1.0]),
    np.array([100.0, 1.0]),
)
compare(
    "log flag forced 0",
    np.array([0.01, 1.0]),
    np.array([1000.0, 50.0]),
    np.array([0.1, 2.0]),
    np.array([100.0, 40.0]),
    logflag=np.zeros(2),
)
compare(
    "ratio exactly 10",
    np.array([1.0]),
    np.array([10.0]),
    np.array([1.0]),
    np.array([10.0]),
)
# 1-D input to the transformer (shape (D,))
P = VariableTransformer(
    2,
    np.array([[-10.0, 0.01]]),
    np.array([[10.0, 1000.0]]),
    np.array([[-5.0, 0.1]]),
    np.array([[5.0, 100.0]]),
)
print(
    "1-D input dir shape:",
    P(np.array([1.0, 2.0])).shape,
    " inv shape:",
    P.inverse_transf(np.array([0.1, 0.2])).shape,
)
try:
    print(grid_units(np.array([1.0, 2.0]), P))
except Exception as e:
    print("grid_units 1-D input:", type(e).__name__, e)
# maskindex on 1-D
try:
    print(py_maskindex(np.array([1.0, 2.0]), np.array([True, False])))
except Exception as e:
    print("maskindex 1-D:", type(e).__name__, e)
# scalar apply_log_t branch
try:
    VariableTransformer(
        2,
        np.array([[-10.0, 0.01]]),
        np.array([[10.0, 1000.0]]),
        None,
        None,
        apply_log_t=0,
    )
    print("scalar apply_log_t OK")
except Exception as e:
    print("scalar apply_log_t:", type(e).__name__, e)
