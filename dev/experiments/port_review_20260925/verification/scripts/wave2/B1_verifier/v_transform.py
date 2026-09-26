"""B1 verifier, I-F5: the self-test of the transform (absolute tolerance
1e-6) on bounds of large magnitude, for PyBADS's VariableTransformer and a
transcription of MATLAB's transvars.m (NumEps = 1e-6, lines 30, 169-178).
Also: bit-identity of the two transforms on the accepted cases."""
import numpy as np
from common import m_transvars

from pybads.variable_transformer import VariableTransformer

rng = np.random.default_rng(12345)


def py_ok(lb, ub, plb, pub):
    try:
        VariableTransformer(
            1,
            np.array([[lb]]),
            np.array([[ub]]),
            np.array([[plb]]),
            np.array([[pub]]),
            np.full((1, 1), np.nan),
        )
        return True
    except ValueError:
        return False


def m_ok(lb, ub, plb, pub):
    try:
        m_transvars(1, lb, ub, plb, pub)
        return True
    except AssertionError:
        return False


print("case                      N   PY refused  MAT refused  disagree")
for label, gen in [
    ("log, ub ~ 1e8", lambda: (1e-3, 10 ** rng.uniform(7.7, 8.3), 1.0, 10.0)),
    ("log, ub ~ 1e9", lambda: (1e-3, 10 ** rng.uniform(8.7, 9.3), 1.0, 10.0)),
    (
        "lin, |lb|=ub ~ 1e10",
        lambda: (-(b := 10 ** rng.uniform(9.5, 10.5)), b, -2.0, 2.0),
    ),
    (
        "lin, |lb|=ub ~ 1e11",
        lambda: (-(b := 10 ** rng.uniform(10.5, 11.5)), b, -2.0, 2.0),
    ),
    (
        "lin, |lb|=ub ~ 1e6 (ctrl)",
        lambda: (-(b := 10 ** rng.uniform(5.5, 6.5)), b, -2.0, 2.0),
    ),
]:
    npy = nm = dis = 0
    N = 200
    for _ in range(N):
        a = gen()
        p, m = py_ok(*a), m_ok(*a)
        npy += not p
        nm += not m
        dis += p != m
    print(f"{label:24s} {N:4d} {npy:8d} {nm:11d} {dis:9d}")

# the reviewer's single case
a = (-9.53e10, 9.53e10, -2.06, -0.74)
print(
    "\nlb=-9.53e10, ub=9.53e10, plb=-2.06, pub=-0.74: PY ok =",
    py_ok(*a),
    " MAT ok =",
    m_ok(*a),
)
# the size of the round-trip error at lb, relative to |lb|
mu, gam = 0.5 * (a[2] + a[3]), 0.5 * (a[3] - a[2])
err = abs(gam * ((a[0] - mu) / gam) + mu - a[0])
print(
    f"  |ginv(g(lb)) - lb| = {err:.3g}, relative {err / abs(a[0]):.3g}, "
    f"spacing(lb) = {np.spacing(abs(a[0])):.3g}"
)

# bit-identity of the maps on accepted cases
X = rng.uniform(-3, 3, size=(2000, 3))
lb = np.array([[-10.0, 1e-3, -np.inf]])
ub = np.array([[10.0, 1e3, np.inf]])
plb = np.array([[-2.0, 1e-2, -1.0]])
pub = np.array([[3.0, 1e2, 1.0]])
vt = VariableTransformer(3, lb, ub, plb, pub, np.full((1, 3), np.nan))
t = m_transvars(3, lb, ub, plb, pub)
Xo = vt.inverse_transf(X)
Xm = np.minimum(np.maximum(t["ginv"](X), t["olb"]), t["oub"])
Uo = vt(Xo)
Um = np.minimum(np.maximum(t["g"](Xm), t["lb"]), t["ub"])
print(
    "\nmixed lin/log/unbounded, 2000 points: inverse identical:",
    np.array_equal(Xo, Xm),
    " direct identical:",
    np.array_equal(Uo, Um),
    " log flags:",
    vt.apply_log_t.ravel(),
    t["logct"],
)
