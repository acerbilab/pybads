import copy

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.search.search_hedge as sh
from pybads import BADS
from pybads.search.es_search import ESSearchWM

captured = []
orig_call = sh.ESSearchHedge.__call__


def cap(self, u, lb, ub, fl, gp, optim_state):
    captured.append(
        (
            u.copy(),
            copy.deepcopy(gp),
            copy.deepcopy(optim_state),
            self.options_dict,
        )
    )
    return orig_call(self, u, lb, ub, fl, gp, optim_state)


sh.ESSearchHedge.__call__ = cap


def rosen(x):
    x = np.atleast_2d(x)
    return np.sum(
        100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2
    )


D = 3
b = BADS(
    rosen,
    np.full(D, -1.5),
    np.full(D, -20.0),
    np.full(D, 20.0),
    np.full(D, -5.0),
    np.full(D, 5.0),
    options={"random_seed": 3, "display": "off", "max_fun_evals": 100},
)
b.optimize()


def sqrtsigma_matlab(u, U, Y, mesh_size, sumrule=1):
    """Transcription of searchES.m method 1 (lines 41-84), no periodic variables."""
    jit = mesh_size
    frac = 0.5
    mu = frac * U.shape[0]
    k = int(np.floor(mu))
    w = np.log(mu + 0.5) - np.log(np.arange(1, k + 1))
    w = w / w.sum()
    index = np.argsort(Y, kind="stable")
    Ubest = U[index[:k]]
    S = Ubest - u
    C = np.sum(w) * (
        S.T @ S
    )  # ucov.m: sum(bsxfun(@times,weights,ushift'*ushift),3)
    lam, E = np.linalg.eigh(C)
    lam = np.maximum(0, lam) + jit**2
    lam = lam / lam.sum() if sumrule else lam / lam.max()
    return np.diag(np.sqrt(lam)) @ E.T


rel = []
for u, gp, st, opts in captured[::4]:
    s = ESSearchWM(2048, 2048, opts)
    Sp = s._initialize_(u, gp, st, 1)
    Sm = sqrtsigma_matlab(u, gp.X, gp.y.ravel(), st["mesh_size"])
    Cp, Cm = Sp.T @ Sp, Sm.T @ Sm
    rel.append(np.linalg.norm(Cp - Cm) / np.linalg.norm(Cm))
    n = gp.X.shape[0]
print("training set sizes:", sorted({c[1].X.shape[0] for c in captured}))
print(
    "relative Frobenius difference of the search covariance (Python vs MATLAB transcription):"
)
print(np.round(rel, 3))
# Show that using floor(mu) points in Python's own code would make them equal
u, gp, st, opts = captured[-1]
Y = gp.y.ravel()
mu = 0.5 * gp.X.shape[0]
print(
    "rows used: Python",
    len(np.argsort(Y)[0 : int(np.floor(mu + 1))]),
    " MATLAB",
    int(np.floor(mu)),
)


# Control: the transcription with floor(mu)+1 rows reproduces Python
def sqrtsigma_kplus1(u, U, Y, mesh_size):
    mu = 0.5 * U.shape[0]
    k = int(np.floor(mu))
    index = np.argsort(Y)
    S = U[index[: k + 1]] - u
    lam, E = np.linalg.eigh(S.T @ S)
    lam = np.maximum(0, lam) + mesh_size**2
    lam = lam / lam.sum()
    return np.diag(np.sqrt(lam)) @ E.T


ok = []
for u, gp, st, opts in captured[::4]:
    Sp = ESSearchWM(2048, 2048, opts)._initialize_(u, gp, st, 1)
    Sk = sqrtsigma_kplus1(u, gp.X, gp.y.ravel(), st["mesh_size"])
    ok.append(np.allclose(Sp.T @ Sp, Sk.T @ Sk))
print("control (floor(mu)+1 rows) equals Python in all states:", all(ok))


# Size of the weighting that both sides drop: weighted vs unweighted, floor(mu) rows, normalized as the search does
def cov_norm(u, U, Y, mesh_size, weighted):
    mu = 0.5 * U.shape[0]
    k = int(np.floor(mu))
    w = np.log(mu + 0.5) - np.log(np.arange(1, k + 1))
    w = w / w.sum()
    S = U[np.argsort(Y, kind="stable")[:k]] - u
    C = S.T @ (w[:, None] * S) if weighted else S.T @ S
    lam, E = np.linalg.eigh(C)
    lam = np.maximum(0, lam) + mesh_size**2
    lam = lam / lam.sum()
    return E @ np.diag(lam) @ E.T


d = []
for u, gp, st, opts in captured[::4]:
    Cw = cov_norm(u, gp.X, gp.y.ravel(), st["mesh_size"], True)
    Cu = cov_norm(u, gp.X, gp.y.ravel(), st["mesh_size"], False)
    d.append(np.linalg.norm(Cw - Cu) / np.linalg.norm(Cu))
print(
    "weighted vs unweighted normalized covariance, relative Frobenius difference:",
    np.round(d, 3),
)
