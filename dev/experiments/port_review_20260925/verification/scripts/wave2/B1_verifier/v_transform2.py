"""B1 verifier, I-F5 continued: asymmetric plausible boxes, linear
variables with |bounds| ~ 1e10-1e12; PyBADS against MATLAB's transvars.m."""
import numpy as np
from common import m_transvars
from v_transform import m_ok, py_ok

rng = np.random.default_rng(7)
for e in (8, 10, 11, 12):
    npy = nm = dis = 0
    N = 200
    for _ in range(N):
        b = 10 ** rng.uniform(e - 0.5, e + 0.5)
        p1, p2 = np.sort(rng.uniform(-3, 3, 2))
        a = (-b, b, p1, p2)
        p, m = py_ok(*a), m_ok(*a)
        npy += not p
        nm += not m
        dis += p != m
    print(
        f"lin |b|~1e{e}: N={N} PY refused={npy} MAT refused={nm} disagree={dis}"
    )
a = (-9.53e10, 9.53e10, -2.06, -0.74)
mu, gam = 0.5 * (a[2] + a[3]), 0.5 * (a[3] - a[2])
for x in (a[0], a[1]):
    err = abs(gam * ((x - mu) / gam) + mu - x)
    print(
        f"x={x:.3g}: |ginv(g(x)) - x| = {err:.3g} (tolerance 1e-6), "
        f"spacing = {np.spacing(abs(x)):.3g}"
    )
