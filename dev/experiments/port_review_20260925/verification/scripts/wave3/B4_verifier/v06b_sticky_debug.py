"""C-F5: why the rebuild MATLAB's sticky flag forces predicts differently at
level 1 with the same set of training inputs."""
import copy

import numpy as np
from vhdr import box, ellipsoid

import pybads.bads.bads as bm
from pybads import BADS

orig_search = bm.BADS._search_step_


def search(self, gp):
    if (
        getattr(self, "_sticky", False)
        and self.optim_state["search_count"] > 0
        and not self.reset_gp
        and not gp.temporary_data.get("needs_rebuild", False)
    ):
        g2 = copy.deepcopy(gp)
        os2 = copy.deepcopy(self.optim_state)
        g2, _ = bm.local_gp_fitting(
            g2,
            self.u,
            self.function_logger,
            self.options,
            os2,
            self.iteration_history,
            False,
            rng=np.random.default_rng(0),
        )
        A = np.array(sorted(map(tuple, np.round(gp.X, 12))))
        B = np.array(sorted(map(tuple, np.round(g2.X, 12))))
        m1, v1 = gp.predict(np.atleast_2d(self.u))
        m2, v2 = g2.predict(np.atleast_2d(self.u))
        print(
            f"search_count={self.optim_state['search_count']} n_old={len(A)} n_new={len(B)} "
            f"same multiset={A.shape == B.shape and np.allclose(A, B)} "
            f"same hyp={np.allclose(gp.get_hyperparameters(as_array=True), g2.get_hyperparameters(as_array=True))} "
            f"sn2_mult old/new={gp.posteriors[0].sn2_mult}/{g2.posteriors[0].sn2_mult} "
            f"mu old/new={m1.item():.6g}/{m2.item():.6g} sd old/new={np.sqrt(v1.item()):.3g}/{np.sqrt(v2.item()):.3g}"
        )
    return orig_search(self, gp)


bm.BADS._search_step_ = search
orig_poll = bm.BADS._poll_step_


def poll(self, gp):
    out = orig_poll(self, gp)
    self._sticky = self.reset_gp
    return out


bm.BADS._poll_step_ = poll

D = 6
lb, ub, plb, pub = box(D)
nrng = np.random.default_rng(1001)
BADS(
    lambda x: ellipsoid(x) + nrng.normal(),
    1.5 * np.ones(D),
    lb,
    ub,
    plb,
    pub,
    options=dict(
        random_seed=1,
        display="off",
        max_fun_evals=200,
        uncertainty_handling=True,
    ),
).optimize()
