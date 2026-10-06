"""F2 (ucov ignores weights; MATLAB's too) and F3 (floor(mu)+1 rows)."""
import numpy as np
import scipy.linalg
import vhdr  # noqa
from capture import capture_states

from pybads.search.es_search import ESSearchWM, ucov


def ucov_matlab(U, u0, w):
    """utils/ucov.m, non-periodic: C = sum(bsxfun(@times, weights(1,1,:), ushift'*ushift), 3)."""
    S = U - u0
    M = S.T @ S
    return (
        np.sum(np.asarray(w).reshape(-1, 1, 1) * M[None, :, :], axis=0)
        if np.size(w)
        else M
    )


def sqrtsigma_matlab(U, Y, u, mesh_size, sumrule=True, weighted=False):
    """searchES.m method 1 (74919c0), lines 55-84."""
    mu = 0.5 * U.shape[0]
    w = np.log(mu + 0.5) - np.log(np.arange(1, int(np.floor(mu)) + 1))
    w = w / w.sum()
    index = np.argsort(Y, kind="stable")
    Ubest = U[index[: int(np.floor(mu))]]
    if weighted:
        S = Ubest - u
        C = (S * w[:, None]).T @ S
    else:
        C = ucov_matlab(Ubest, u, w)
    lam, E = scipy.linalg.eigh(C)
    lam = np.maximum(0, lam) + mesh_size**2
    lam = lam / lam.sum() if sumrule else lam / lam.max()
    return np.diag(np.sqrt(lam)) @ E.T


rng = np.random.default_rng(0)
U = rng.normal(size=(9, 3))
u0 = rng.normal(size=3)
w = rng.random(5)
w /= w.sum()
S = U - u0
print(
    "ucov(PyBADS) == S^T S:",
    np.allclose(ucov(U, u0, w, np.ones(3), -np.ones(3), np.ones(3)), S.T @ S),
)
print(
    "ucov(MATLAB transcription) == S^T S:",
    np.allclose(ucov_matlab(U, u0, w), S.T @ S),
)
print(
    "weighted sum_i w_i s_i s_i^T == S^T S:",
    np.allclose((S[:5] * w[:, None]).T @ S[:5], S.T @ S),
)

states = capture_states(D=3, seed=0, max_fun_evals=120)
print(f"captured {len(states)} search states (rosenbrock D=3, seed 0)")
rel_f3 = []
rel_w = []
nsel = []
for st in states:
    gp, os_ = st["gp"], st["optim_state"]
    es = ESSearchWM(2048, 2048, st["options"], rng=np.random.default_rng(1))
    es.mesh_size = os_["mesh_size"]
    py = es._initialize_(st["u"], gp, os_, True)
    U, Y = gp.X, gp.y.ravel()
    ml = sqrtsigma_matlab(U, Y, st["u"], os_["mesh_size"])
    mlw = sqrtsigma_matlab(U, Y, st["u"], os_["mesh_size"], weighted=True)
    Cpy, Cml, Cmlw = (
        py.T @ py,
        ml.T @ ml,
        mlw.T @ mlw,
    )  # covariance of the draws
    rel_f3.append(np.linalg.norm(Cpy - Cml) / np.linalg.norm(Cml))
    rel_w.append(np.linalg.norm(Cmlw - Cml) / np.linalg.norm(Cml))
    mu = 0.5 * U.shape[0]
    nsel.append((U.shape[0], int(np.floor(mu)), int(np.floor(mu + 1))))
print(
    "(n_train, MATLAB rows floor(mu), PyBADS rows floor(mu+1)) at first 4 states:",
    nsel[:4],
)
print(
    "rel. Frobenius diff of normalized search covariance, PyBADS vs MATLAB: min %.3f median %.3f max %.3f"
    % (min(rel_f3), np.median(rel_f3), max(rel_f3))
)
print(
    "rel. diff if the weights were applied (MATLAB rows): min %.3f median %.3f max %.3f"
    % (min(rel_w), np.median(rel_w), max(rel_w))
)
# Python with floor(mu)+1 rows transcribed reproduces the port exactly?
st = states[-1]
gp = st["gp"]
U, Y = gp.X, gp.y.ravel()
mu = 0.5 * U.shape[0]
idx = np.argsort(Y)[: int(np.floor(mu + 1))]
C = ucov_matlab(U[idx], st["u"], np.ones(1))
lam, E = scipy.linalg.eigh(C)
lam = np.maximum(0, lam) + st["optim_state"]["mesh_size"] ** 2
lam /= lam.sum()
es = ESSearchWM(2048, 2048, st["options"])
es.mesh_size = st["optim_state"]["mesh_size"]
print(
    "port == transcription with floor(mu)+1 rows:",
    np.allclose(
        es._initialize_(st["u"], gp, st["optim_state"], True),
        np.diag(np.sqrt(lam)) @ E.T,
    ),
)
