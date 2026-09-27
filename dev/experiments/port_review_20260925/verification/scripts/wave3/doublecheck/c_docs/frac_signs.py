"""At n_search_iter = 4 and 5, record for every scale update of the ES search
MATLAB's fraction of new candidates (0d866e8's count) and, on the same pool,
the count of 1.1.0 (z_idx[0:ntest+1] > nold); report how often each is above
0.2 (the scale grows) and how often 1.1.0's grows where MATLAB's shrinks."""
import logging

import gpyreg
import numpy as np

import pybads
import pybads.search.es_search as es
from pybads import BADS
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb
from pybads.function_logger.constraints_check import contraints_check
from pybads.search.grid_functions import force_to_grid

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
rec = []


def call(
    self,
    u,
    lb,
    ub,
    func_logger,
    gp,
    optim_state,
    sum_rule=True,
    non_box_cons=None,
):
    # 0d866e8's ESSearch.__call__, with both counts recorded
    self.mesh_size = optim_state["mesh_size"]
    self.search_factor = optim_state["search_factor"]
    self.search_mesh_size = optim_state["search_mesh_size"]
    U = gp.X
    nvars = U.shape[1]
    self.sqrt_sigma = self._initialize_(u, gp, optim_state, sum_rule)
    self.sqrt_sigma = self.mesh_size * self.search_factor * self.sqrt_sigma
    N = int(self.mu)
    u_new = u + self.vec * (self.rng.normal(size=(N, nvars)) @ self.sqrt_sigma)
    us = np.empty((min(u_new.shape[0], self.lamb), nvars))
    z = np.empty((us.shape[0], 1))
    for i in range(self.n_search_iter):
        u_new = force_to_grid(u_new, self.search_mesh_size)
        u_new = contraints_check(
            u_new,
            optim_state["lb_search"],
            optim_state["ub_search"],
            optim_state["tol_mesh"],
            func_logger,
            True,
            non_box_cons,
        )
        z_new, _, _ = acq_fcn_lcb(
            u_new, func_logger.func_count, gp, self.search_acq_fcn[1]
        )
        z_new = z_new.flatten()
        nold_110 = us.shape[0]
        nold = us.shape[0] if i > 0 else 0
        if i == 0:
            us_c, z_c = u_new.copy(), z_new.copy()
        else:
            us_c = np.append(us_c, u_new, axis=0)
            z_c = np.append(z_c, z_new, axis=0)
        N = min(us_c.shape[0], self.lamb)
        z_idx = np.argsort(z_c, kind="stable")
        ntest = min(u_new.shape[0], nold)
        n_new = np.sum(z_idx[0:ntest] >= us_c.shape[0] - u_new.shape[0])
        ntest_110 = min(u_new.shape[0], nold_110)
        n_new_110 = np.sum(z_idx[0 : ntest_110 + 1] > nold_110)
        z = z_c[z_idx[0:N]]
        us = us_c[z_idx[0:N]]
        if us.shape[0] == 0:
            break
        if i < self.n_search_iter - 1:
            if i > 0 and ntest > 0:
                frac = n_new / ntest
                rec.append(
                    (self.n_search_iter, i + 1, frac, n_new_110 / ntest_110)
                )
                self.scale = self.scale * np.exp(self.es_beta * (frac - 0.2))
            mask = self._get_selection_idx_mask_(us.shape[0], self.lamb)
            ll = min(self.lamb, us.shape[0])
            u_new = (
                us[mask[0:ll]]
                + (self.rng.normal(size=(ll, nvars)) @ self.sqrt_sigma)
                * self.scale
            )
    if us.shape[0] == 0:
        return us, z
    return us[0], z[0]


es.ESSearch.__call__ = call
for nsi in (3, 4, 5):
    for D, seed in ((3, 0), (6, 1)):
        sc = np.arange(1, D + 1, dtype=float)
        b = BADS(
            lambda x: float(np.sum(sc * (np.atleast_2d(x) - 0.3) ** 2)),
            np.full(D, 2.0),
            lower_bounds=np.full(D, -5.0),
            upper_bounds=np.full(D, 5.0),
            plausible_lower_bounds=np.full(D, -4.0),
            plausible_upper_bounds=np.full(D, 4.0),
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 150,
                "n_search_iter": nsi,
            },
        )
        b.optimize()
r = np.array(rec)
for nsi in (3, 4, 5):
    for gen in range(2, nsi):
        m = (r[:, 0] == nsi) & (r[:, 1] == gen)
        if not m.any():
            continue
        f, f110 = r[m, 2], r[m, 3]
        print(
            f"n_search_iter {nsi}, generation {gen}: updates {m.sum()}; MATLAB's fraction median "
            f"{np.median(f):.3f} (below 0.2 in {np.sum(f < 0.2)}), 1.1.0's median {np.median(f110):.3f} "
            f"(below 0.2 in {np.sum(f110 < 0.2)}); 1.1.0 grows where MATLAB shrinks in "
            f"{np.sum((f < 0.2) & (f110 > 0.2))}; the two counts differ in {np.sum(f != f110)}",
            flush=True,
        )
