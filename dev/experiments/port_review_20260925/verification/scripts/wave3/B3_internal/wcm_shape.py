import gpyreg
import numpy as np
import scipy.linalg

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.search.es_search as es
from pybads import BADS

rec = []
orig_init = es.ESSearchWM._initialize_


def init(self, u, gp, optim_state, sum_rule):
    S_port = orig_init(self, u, gp, optim_state, sum_rule)
    U = gp.X
    Y = gp.y.flatten()
    mu = 0.5 * U.shape[0]
    w = np.log(mu + 0.5) - np.log(np.arange(1, np.floor(mu + 1)))
    w /= w.sum()
    d = U[np.argsort(Y)[: len(w)]] - u
    C = d.T @ (w[:, None] * d)
    lam, E = scipy.linalg.eigh(C)
    lam = np.maximum(0, lam) + optim_state["mesh_size"] ** 2
    lam /= lam.sum()
    S_fix = np.diag(np.sqrt(lam)) @ E.T
    Cp = S_port.T @ S_port
    Cf = S_fix.T @ S_fix
    ep = np.linalg.eigvalsh(Cp)
    ef = np.linalg.eigvalsh(Cf)
    vp = np.linalg.eigh(Cp)[1][:, -1]
    vf = np.linalg.eigh(Cf)[1][:, -1]
    rec.append(
        (
            U.shape[0],
            optim_state["mesh_size"],
            ep.min() / ep.max(),
            ef.min() / ef.max(),
            abs(vp @ vf),
        )
    )
    return S_port


es.ESSearchWM._initialize_ = init
rng0 = np.random.default_rng(12345)
Q, _ = np.linalg.qr(rng0.normal(size=(4, 4)))
f = lambda x: float(
    np.sum(
        10.0 ** (4 * np.arange(4) / 3)
        * (Q @ (np.asarray(x).ravel() - 0.7)) ** 2
    )
)
D = 4
b = BADS(
    f,
    np.full(D, -1.5),
    np.full(D, -8.0),
    np.full(D, 8.0),
    np.full(D, -4.0),
    np.full(D, 4.0),
    options={"random_seed": 0, "max_fun_evals": 200, "display": "off"},
)
r = b.optimize()
print(
    "n_train, mesh, min/max eig (port), min/max eig (weighted), |cos| of leading directions"
)
for t in rec[:: max(1, len(rec) // 12)]:
    print("%3d %.4g %.3g %.3g %.3f" % t)
