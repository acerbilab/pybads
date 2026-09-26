import copy
import warnings

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.search.search_hedge as sh
from pybads import BADS
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb
from pybads.search.es_search import ESSearchWM

captured = []
orig_call = sh.ESSearchHedge.__call__


def cap(self, u, lb, ub, fl, gp, optim_state):
    if not captured:
        captured.append(
            (
                u.copy(),
                copy.deepcopy(fl),
                copy.deepcopy(gp),
                copy.deepcopy(optim_state),
                self.options_dict,
            )
        )
    return orig_call(self, u, lb, ub, fl, gp, optim_state)


sh.ESSearchHedge.__call__ = cap
f = lambda x: np.sum(np.ravel(x) ** 2)
D = 3
b = BADS(
    f,
    np.full(D, 3.0),
    np.full(D, -20.0),
    np.full(D, 20.0),
    np.full(D, -5.0),
    np.full(D, 5.0),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 30},
)
b.optimize()
sh.ESSearchHedge.__call__ = orig_call
u, fl, gp, st, opts = captured[0]

# (a) LCB on an empty candidate set
try:
    z, fmu, fs = acq_fcn_lcb(np.empty((0, D)), 10, gp)
    print("(a) acq_fcn_lcb on empty set: z shape", z.shape)
except Exception as e:
    print("(a) acq_fcn_lcb on empty set raises:", type(e).__name__, e)

# (b) ES whose second-iteration offspring are all infeasible
calls = [0]


def nbc(X):
    calls[0] += 1
    return (
        np.zeros(len(X)) if calls[0] == 1 else np.ones(len(X))
    )  # 1st call: all feasible; later: none


s = ESSearchWM(2048, 2048, opts, np.random.default_rng(0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    us, z = s(u, None, None, fl, gp, st, 1, nbc)
print(
    "(b) ES with all iteration-2 offspring infeasible returns shape",
    np.shape(us),
    "(MATLAB: the best of iteration 1)",
)

# (c) a scalar Python float as the LCB parameter
try:
    acq_fcn_lcb(np.atleast_2d(u), 10, gp, 1.0)
    print("(c) float sqrt_beta accepted")
except Exception as e:
    print("(c) acq_fcn_lcb(..., sqrt_beta=1.0) raises:", type(e).__name__, e)

# (d) hedge_gamma = 0: update_hedge slices the point's coordinates
opts0 = dict(opts)
opts0["hedge_gamma"] = 0.0
h = sh.ESSearchHedge(
    opts0["search_method"], opts0, None, rng=np.random.default_rng(0)
)
h(u, None, None, fl, gp, st)
us1 = np.atleast_2d(u)[0] + 0.01
print(
    "(d) chosen",
    h.chosen_hedge,
    "u_hedge slices for i_hedge=0,1:",
    [us1[np.minimum(i, len(us1) - 1) :].shape for i in range(2)],
)
try:
    h.update_hedge(us1, 1.0, 0.9, 0.0, gp, st["mesh_size"])
    print("(d) update_hedge with gamma=0 ran; g =", h.g)
except Exception as e:
    print("(d) update_hedge with gamma=0 raises:", type(e).__name__, e)
