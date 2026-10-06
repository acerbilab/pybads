from common import *

from pybads.variable_transformer import VariableTransformer
from pybads.variable_transformer.variables_transformer import maskindex

rng = np.random.default_rng(0)


def ref_forward(x, lb, ub, plb, pub, logf):
    # independent transcription of the specification: log where flagged, then affine map plb->-1, pub->1
    x = np.atleast_2d(x).astype(float)
    y = x.copy()
    L = logf.ravel()
    tp = lambda v: np.where(L, np.log(v), v)
    lp, up = tp(plb.ravel()), tp(pub.ravel())
    mu = 0.5 * (lp + up)
    g = 0.5 * (up - lp)
    with np.errstate(all="ignore"):
        y = (np.where(L, np.log(np.abs(x)), x) - mu) / g
    return y


def ref_inverse(u, plb, pub, logf):
    L = logf.ravel()
    tp = lambda v: np.where(L, np.log(v), v)
    lp, up = tp(plb.ravel()), tp(pub.ravel())
    mu = 0.5 * (lp + up)
    g = 0.5 * (up - lp)
    v = g * u + mu
    return np.where(L, np.exp(v), v)


cases = {
    "linear finite": (
        np.array([[-5.0, 0.0, -3]]),
        np.array([[5.0, 10.0, 3]]),
        np.array([[-2.0, 1.0, -1]]),
        np.array([[3.0, 9.0, 2]]),
    ),
    "all log": (
        np.array([[1e-3, 1.0, 0.1]]),
        np.array([[1e3, 1e4, 50]]),
        np.array([[1e-2, 2.0, 0.5]]),
        np.array([[1e2, 5e3, 20]]),
    ),
    "mixed log": (
        np.array([[1e-3, -4.0, -np.inf]]),
        np.array([[1e3, 4.0, np.inf]]),
        np.array([[1e-2, -1.0, -2]]),
        np.array([[1e2, 3.0, 5]]),
    ),
    "infinite all": (
        np.full((1, 3), -np.inf),
        np.full((1, 3), np.inf),
        np.array([[-1.0, -3.0, 0.0]]),
        np.array([[1.0, 2.0, 10.0]]),
    ),
    "ratio exactly 10": (
        np.array([[0.5, 1.0]]),
        np.array([[20.0, 100.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[10.0, 9.99]]),
    ),
}
for name, (lb, ub, plb, pub) in cases.items():
    D = lb.shape[1]
    vt = VariableTransformer(
        D,
        lb.copy(),
        ub.copy(),
        plb.copy(),
        pub.copy(),
        np.full((1, D), np.nan),
    )
    logf = vt.apply_log_t
    print(
        f"== {name}: log={logf.ravel()}, T(lb)={vt.lb.ravel()}, T(ub)={vt.ub.ravel()}, T(plb)={vt.plb.ravel()}, T(pub)={vt.pub.ravel()}"
    )
    # random points inside the hard box (finite part), or around plausible if infinite
    lo = np.where(np.isfinite(lb), lb, plb - 10 * (pub - plb)).ravel()
    hi = np.where(np.isfinite(ub), ub, pub + 10 * (pub - plb)).ravel()
    X = np.empty((1000, D))
    for d in range(D):
        if logf.ravel()[d]:
            X[:, d] = np.exp(rng.uniform(np.log(lo[d]), np.log(hi[d]), 1000))
        else:
            X[:, d] = rng.uniform(lo[d], hi[d], 1000)
    U = vt(X)
    Uref = ref_forward(X, lb, ub, plb, pub, logf)
    X2 = vt.inverse_transf(U)
    print(
        "  fwd max abs err vs ref:",
        np.max(np.abs(U - Uref)),
        " roundtrip max rel err:",
        np.max(np.abs(X2 - X) / np.maximum(1, np.abs(X))),
    )
    Ur = rng.uniform(-3, 3, (1000, D))
    Xr = vt.inverse_transf(Ur)
    Xref = np.clip(ref_inverse(Ur, plb, pub, logf), lb, ub)
    print(
        "  inv max rel err vs ref:",
        np.max(np.abs(Xr - Xref) / np.maximum(1, np.abs(Xref))),
        " fwd(inv) err:",
        np.max(np.abs(vt(Xr) - np.clip(Ur, vt.lb, vt.ub))),
    )
    # plausible bounds map onto -1, 1
    print("  plb->", vt(plb).ravel(), " pub->", vt(pub).ravel())
    # 1-D input
    print(
        "  1-D input shapes: fwd",
        vt(X[0]).shape,
        " inv",
        vt.inverse_transf(U[0]).shape,
    )

# maskindex
v = np.arange(6.0).reshape(2, 3)
print("maskindex:", maskindex(v, np.array([[True, False, True]])))
# large finite bounds: absolute tolerance of the self-test
for lbv in [-1e8, -1e10, -1e12]:
    trycall(
        f"linear lb={lbv}",
        lambda: VariableTransformer(
            1,
            np.array([[lbv]]),
            np.array([[1.0]]),
            np.array([[0.1]]),
            np.array([[0.7]]),
            np.full((1, 1), np.nan),
        ).lb,
    )
for ubv in [1e8, 1e10, 1e12]:
    trycall(
        f"log ub={ubv}",
        lambda: VariableTransformer(
            1,
            np.array([[1e-3]]),
            np.array([[ubv]]),
            np.array([[1.0]]),
            np.array([[1e6]]),
            np.full((1, 1), np.nan),
        ).ub,
    )
# same through BADS
trycall(
    "BADS lb=-1e12 plausible [0.1,0.7]",
    lambda: BADS(
        quad,
        np.array([[0.3, 0.3]]),
        np.array([[-1e12, -1]]),
        np.array([[1, 1]]),
        np.array([[0.1, -0.5]]),
        np.array([[0.7, 0.5]]),
        options={"display": "off", "random_seed": 0},
    ).var_transf.lb,
)
trycall(
    "BADS log ub=1e12",
    lambda: BADS(
        quad,
        np.array([[10.0, 0.3]]),
        np.array([[1e-3, -1]]),
        np.array([[1e12, 1]]),
        np.array([[1.0, -0.5]]),
        np.array([[1e6, 0.5]]),
        options={"display": "off", "random_seed": 0},
    ).var_transf.ub,
)
