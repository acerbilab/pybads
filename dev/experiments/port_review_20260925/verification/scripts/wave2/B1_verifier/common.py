"""Shared helpers of the B1 verifier's checks: the import banner and a
Python transcription of MATLAB BADS's setup of the bounds (boundscheck.m,
setupvars.m:6-9, 58-99, transvars.m create/dir/inv, force2grid.m)."""
import logging

import gpyreg
import numpy as np

import pybads

print("pybads.__file__ =", pybads.__file__)
print("gpyreg.__file__ =", gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)

EPS = np.spacing(1.0)
REALMAX = np.finfo(np.float64).max


def mround(x):
    """MATLAB's round: halves away from zero."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def m_transvars(nvars, lb, ub, plb, pub, logct=None):
    """transvars.m, create branch (lines 76-183); returns a dict."""
    lb = np.atleast_1d(np.asarray(lb, float)).ravel().copy()
    ub = np.atleast_1d(np.asarray(ub, float)).ravel().copy()
    plb = np.atleast_1d(np.asarray(plb, float)).ravel().copy()
    pub = np.atleast_1d(np.asarray(pub, float)).ravel().copy()
    if lb.size == 1:
        lb = lb * np.ones(nvars)
    if ub.size == 1:
        ub = ub * np.ones(nvars)
    if plb.size == 1:
        plb = plb * np.ones(nvars)
    if pub.size == 1:
        pub = pub * np.ones(nvars)
    assert np.all(np.isfinite(np.r_[plb, pub])), "PLB/PUB not finite"
    assert np.all((lb <= plb) & (plb < pub) & (pub <= ub)), "order"
    if logct is None:
        logct = np.full(nvars, np.nan)
    logct = np.asarray(logct, float).ravel().copy()
    for i in np.where(np.isnan(logct))[0]:
        logct[i] = float(
            np.all(np.array([lb[i], ub[i], plb[i], pub[i]]) > 0)
            and (pub[i] / plb[i] >= 10)
        )
    logct = logct.astype(bool)
    t = dict(
        logct=logct,
        olb=lb.copy(),
        oub=ub.copy(),
        oplb=plb.copy(),
        opub=pub.copy(),
    )
    tl, tu, tpl, tpu = lb.copy(), ub.copy(), plb.copy(), pub.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tl[logct] = np.log(tl[logct])
        tu[logct] = np.log(tu[logct])
        tpl[logct] = np.log(tpl[logct])
        tpu[logct] = np.log(tpu[logct])
    mu = 0.5 * (tpl + tpu)
    gamma = 0.5 * (tpu - tpl)

    def g(x):
        x = np.asarray(x, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (x - mu) / gamma
            zl = (np.log(np.abs(x) + (x == 0)) - mu) / gamma
        return np.where(logct, zl, z)

    def ginv(y):
        y = np.asarray(y, float)
        with np.errstate(over="ignore"):
            lin = gamma * y + mu
            lg = np.minimum(REALMAX, np.exp(gamma * y + mu))
        return np.where(logct, lg, lin)

    # self-test (lines 169-178)
    lbt = lb.copy()
    lbt[~np.isfinite(lb)] = -1 / np.sqrt(EPS)
    ubt = ub.copy()
    ubt[~np.isfinite(ub)] = 1 / np.sqrt(EPS)
    ubt[~np.isfinite(ub) & logct] = 1e6
    NumEps = 1e-6
    tests = [
        np.all(np.abs(ginv(g(lbt)) - lbt) < NumEps),
        np.all(np.abs(ginv(g(ubt)) - ubt) < NumEps),
        np.all(np.abs(ginv(g(plb)) - plb) < NumEps),
        np.all(np.abs(ginv(g(pub)) - pub) < NumEps),
    ]
    if not all(tests):
        raise AssertionError(
            "Cannot invert the transform to obtain the identity at the "
            "provided boundaries."
        )
    t.update(g=g, ginv=ginv, lb=g(lb), ub=g(ub), plb=g(plb), pub=g(pub))
    return t


def m_dir(x, t):
    return np.minimum(np.maximum(t["g"](x), t["lb"]), t["ub"])


def m_inv(u, t):
    return np.minimum(np.maximum(t["ginv"](u), t["olb"]), t["oub"])


def m_setup(x0, LB, UB, PLB=None, PUB=None, rand_u=None, mesh=2.0**-10):
    """boundscheck.m + setupvars.m (6-9, 58-99), with NonlinearScaling on.
    Returns (u0, x0, trinfo). rand_u: the uniform draw in [0,1]^D used when
    x0 is not finite (setupvars.m:83)."""
    x0 = np.atleast_1d(np.asarray(x0, float)).ravel()
    nvars = x0.size
    LB = np.atleast_1d(
        np.asarray(-np.inf if LB is None else LB, float)
    ).ravel()
    UB = np.atleast_1d(np.asarray(np.inf if UB is None else UB, float)).ravel()
    if LB.size == 1:
        LB = LB * np.ones(nvars)
    if UB.size == 1:
        UB = UB * np.ones(nvars)
    if PLB is not None:
        PLB = np.atleast_1d(np.asarray(PLB, float)).ravel()
        if PLB.size == 1:
            PLB = PLB * np.ones(nvars)
    if PUB is not None:
        PUB = np.atleast_1d(np.asarray(PUB, float)).ravel()
        if PUB.size == 1:
            PUB = PUB * np.ones(nvars)
    if PLB is None:
        PLB = LB.copy()
    if PUB is None:
        PUB = UB.copy()
    for v in (LB, UB, PLB, PUB):
        if v.size != nvars:
            raise AssertionError("sizes")
    if not np.all(np.isfinite(np.r_[PLB, PUB])):
        raise AssertionError("Plausible interval bounds need to be finite.")
    if not np.all((LB <= PLB) & (PLB < PUB) & (PUB <= UB)):
        raise AssertionError("Bound vectors do not respect the order")
    t = m_transvars(nvars, LB, UB, PLB, PUB)
    tLB, tUB, tPLB, tPUB = t["lb"], t["ub"], t["plb"], t["pub"]
    if np.any(~np.isfinite(x0)):
        u0 = rand_u * (tPUB - tPLB) + tPLB
        u0 = mesh * mround(u0 / mesh)
        x0 = m_inv(u0, t)
    else:
        u0 = mesh * mround(m_dir(x0, t) / mesh)
    u0 = np.where(u0 < tLB, u0 + mesh, u0)
    u0 = np.where(u0 > tUB, u0 - mesh, u0)
    if not np.all((x0 <= UB) & (x0 >= LB) & (u0 <= tUB) & (u0 >= tLB)):
        raise AssertionError("Initial starting point is not within bounds")
    return u0, x0, t


def py_setup(x0, lb, ub, plb=None, pub=None, options=None, nbc=None):
    """Construct a BADS object and return (u0, x0 after checks, bads)."""
    from pybads import BADS

    opts = {"display": "off", "random_seed": 0}
    if options:
        opts.update(options)
    b = BADS(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        x0,
        lb,
        ub,
        plb,
        pub,
        non_box_cons=nbc,
        options=opts,
    )
    return b.optim_state["u"].ravel(), b.x0.ravel(), b


def fmt(a):
    return np.array2string(np.asarray(a, float).ravel(), precision=6)
