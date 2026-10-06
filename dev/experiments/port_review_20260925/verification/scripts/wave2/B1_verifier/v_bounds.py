"""B1 verifier: the bounds check against a transcription of MATLAB's
boundscheck.m + setupvars.m. Cases: half/mixed bounds (I-F1, C-F2, K1),
scalar bounds (I-F2, C-F5), effective bounds and the plausible-box
expansion (I-F3, I-F4, C-F1)."""
import numpy as np
from common import fmt, m_setup, py_setup

inf = np.inf


def run(name, x0, lb, ub, plb=None, pub=None):
    print(f"\n== {name}: x0={x0} lb={lb} ub={ub} plb={plb} pub={pub}")
    try:
        u, x, b = py_setup(x0, lb, ub, plb, pub)
        vt = b.var_transf
        print(
            "  PY  accepted: x0 ->",
            fmt(x),
            " plb_orig ->",
            fmt(vt.orig_plb),
            " pub_orig ->",
            fmt(vt.orig_pub),
        )
        print(
            "      log:",
            vt.apply_log_t.ravel(),
            " u0 =",
            fmt(u),
            " u(lb) =",
            fmt(vt.lb),
            " u(ub) =",
            fmt(vt.ub),
        )
    except Exception as e:
        print("  PY  refused:", type(e).__name__, str(e).split("\n")[0][:90])
    try:
        mu, mx, t = m_setup(
            x0 if x0 is not None else np.full(np.size(plb), np.nan),
            lb,
            ub,
            plb,
            pub,
            rand_u=np.full(np.size(plb), 0.5),
        )
        print(
            "  MAT accepted: x0 ->",
            fmt(mx),
            " plb_orig ->",
            fmt(t["oplb"]),
            " pub_orig ->",
            fmt(t["opub"]),
        )
        print(
            "      log:",
            t["logct"],
            " u0 =",
            fmt(mu),
            " u(lb) =",
            fmt(t["lb"]),
            " u(ub) =",
            fmt(t["ub"]),
        )
    except Exception as e:
        print("  MAT refused:", type(e).__name__, str(e)[:90])


# Half bounds and mixed bounds
run(
    "mixed bounded/unbounded",
    [0.5, 0.5],
    [0, -inf],
    [1, inf],
    [0.1, -1],
    [0.9, 1],
)
run("half-bounded (lb=0, ub=inf)", [0.5], [0.0], [inf], [0.1], [2.0])
run(
    "all unbounded (control)",
    [0.5, 0.5],
    [-inf, -inf],
    [inf, inf],
    [-1, -1],
    [1, 1],
)
run(
    "all bounded (control)", [0.5, 0.5], [0, 0], [1, 1], [0.1, 0.1], [0.9, 0.9]
)

# Scalar bounds
run("scalar bounds, D=3", [0.5, 0.5, 0.5], -1.0, 1.0)
run("scalar bounds and plausible, D=3", [0.0, 0.0, 0.0], -5.0, 5.0, -2.0, 2.0)
run("scalar bounds, D=1 (control)", [0.5], -1.0, 1.0)

# Effective bounds
run("plb omitted, [-5,5]", [0.0], [-5.0], [5.0])
run("plb omitted, [1,10] (MATLAB log)", [2.0], [1.0], [10.0])
run("plb omitted, [1e-3,1e3]", [1.0], [1e-3], [1e3])
run(
    "x0 at user plb, [1e-3,1e3], plb=1e-2",
    [1e-2],
    [1e-3],
    [1e3],
    [1e-2],
    [1e2],
)
run("x0 = lb = -5, plb = -2", [-5.0], [-5.0], [5.0], [-2.0], [2.0])
run(
    "narrow box near lb: [0,1], plb=1e-4, pub=5e-4",
    [2e-4],
    [0.0],
    [1.0],
    [1e-4],
    [5e-4],
)
run(
    "asymmetric: [-1000,1], plb=0, pub=0.99",
    [0.5],
    [-1000.0],
    [1.0],
    [0.0],
    [0.99],
)
run("x0=None, plb omitted, [-5,5]", None, [-5.0], [5.0], [-5.0], [5.0])

# I-F3: x0 outside the plausible box
for x0 in (0.005, 0.02, 1.0, 9.0, 9.995):
    run(f"x0={x0} outside [2,8] in [0,10]", [x0], [0.0], [10.0], [2.0], [8.0])
