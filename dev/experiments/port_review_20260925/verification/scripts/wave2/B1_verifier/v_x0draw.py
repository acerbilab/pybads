"""B1 verifier, K5: the random x0 of PyBADS (bads.py:258-274, then
gridized in _init_optim_state_) against a transcription of MATLAB's
setupvars.m:79-85 (u0 = rand.*(PUB-PLB)+PLB in the transformed box,
force2grid, x0 = origunits(u0)), with the same uniform numbers. Bounds
chosen so that the effective bounds move nothing."""
import numpy as np
from common import m_setup, py_setup

lb = np.array([1e-3, -10.0, 0.5])
ub = np.array([1e3, 10.0, 50.0])
plb = np.array([1e-2, -3.0, 1.0])
pub = np.array([1e2, 5.0, 20.0])
same_u0 = same_draw = 0
N = 200
logs = None
u_all = []
for s in range(N):
    upy, xpy, b = py_setup(None, lb, ub, plb, pub, options={"random_seed": s})
    ru = np.random.default_rng(s).random(size=(1, 3)).ravel()
    umat, xmat, t = m_setup(np.full(3, np.nan), lb, ub, plb, pub, rand_u=ru)
    logs = b.var_transf.apply_log_t.ravel(), t["logct"]
    same_u0 += np.array_equal(upy, umat)
    # the ungridized draw of PyBADS, mapped back to u
    same_draw += np.allclose(
        b.var_transf(b.x0).ravel(),
        ru * (t["pub"] - t["plb"]) + t["plb"],
        rtol=0,
        atol=1e-12,
    )
    u_all.append(b.var_transf(b.x0).ravel())
u_all = np.array(u_all)
print("log flags PY / MAT:", logs)
print(
    f"seeds {N}: gridized u0 identical {same_u0}; ungridized draw equal "
    f"to MATLAB's u within 1e-12: {same_draw}"
)
print(
    "u of the draws: min",
    np.round(u_all.min(0), 3),
    "max",
    np.round(u_all.max(0), 3),
    "mean",
    np.round(u_all.mean(0), 3),
)
print(
    "x of the draws, variable 1 (log): share below 1 (the log midpoint):",
    np.mean(
        np.array(
            [(np.exp(u * np.log(1e2 / 1e-2) / 2 + 0)) for u in u_all[:, 0]]
        )
        < 1
    ),
)
print(
    "result x0 is the ungridized draw; the evaluated point is the "
    "gridized one: e.g. seed 0 x0 =",
    py_setup(None, lb, ub, plb, pub, options={"random_seed": 0})[2].x0.ravel(),
    "evaluated x =",
    py_setup(None, lb, ub, plb, pub, options={"random_seed": 0})[2]
    .var_transf.inverse_transf(
        np.atleast_2d(
            py_setup(None, lb, ub, plb, pub, options={"random_seed": 0})[0]
        )
    )
    .ravel(),
)
