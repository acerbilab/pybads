"""I-F5: at level 0 the poll's target is predicted at u_poll_best from a GP
that does not hold it; I-F9: zero predictive SDs in the poll; K10: at a
failed poll, the accelerated reduction reads the entry iter - 3 of a history
that holds exactly iter entries (0..iter-1), MATLAB's iterList(iter_m - 3)
with iter_m = iter + 1."""
import sys

import numpy as np
from vhdr import box, ellipsoid, rosen

import pybads.bads.bads as bm
from pybads import BADS

S = {}
orig_stop = bm.BADS._is_poll_stop_


def stop(self, good, unrel, p_less, poll_count):
    L = sys._getframe(1).f_locals
    S["steps"] += 1
    S["zero_sd_steps"] += bool(np.any(L["fs"] == 0))
    if good:
        u = L["u_poll_best"]
        gp = L["gp"]
        in_gp = bool(np.any(np.all(np.isclose(gp.X, u, atol=1e-12), axis=1)))
        S["good"].append(
            (
                in_gp,
                float(L["y_poll_best"]),
                self.optim_state["f_target_mu"],
                self.optim_state["f_target"],
            )
        )
    return orig_stop(self, good, unrel, p_less, poll_count)


bm.BADS._is_poll_stop_ = stop

orig_poll = bm.BADS._poll_step_


def poll(self, gp):
    it = self.optim_state["iter"]
    fv = self.iteration_history.get("fval")
    n_hist = 0 if fv is None else int(np.sum([v is not None for v in fv]))
    out = orig_poll(self, gp)
    if it >= self.options["accelerate_mesh_steps"]:
        S["accel"].append((it, n_hist))
    return out


bm.BADS._poll_step_ = poll

for name, f, D, seed in (
    ("rosen", rosen, 3, 1),
    ("rosen", rosen, 3, 2),
    ("ellipsoid", ellipsoid, 4, 2),
):
    S.update(steps=0, zero_sd_steps=0, good=[], accel=[])
    lb, ub, plb, pub = box(D)
    r = BADS(
        f,
        1.5 * np.ones(D),
        lb,
        ub,
        plb,
        pub,
        options=dict(random_seed=seed, display="off", max_fun_evals=200),
    ).optimize()
    g = S["good"]
    print(
        f"{name} D={D} seed={seed} fval={r['fval']:.3g} nfev={r['func_count']}: poll steps={S['steps']} "
        f"with a zero SD={S['zero_sd_steps']}; stop calls after a good poll={len(g)}, of which u_poll_best "
        f"not in the GP={sum(not a for a, *_ in g)}"
    )
    for in_gp, y, mu, ft in g[:4]:
        print(
            f"     in GP={in_gp} observed y={y:.4g} predicted mu={mu:.4g} target={ft:.4g}"
        )
    print(
        "   accelerated-reduction checks (iter, entries in history):",
        S["accel"][:6],
        "all entries == iter:",
        all(a == b for a, b in S["accel"]),
    )
