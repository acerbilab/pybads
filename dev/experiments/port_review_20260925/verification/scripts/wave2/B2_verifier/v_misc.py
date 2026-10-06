"""C-F9 Actions column; K8 poll's returned GP; I-F9 max_iter; I-F10 first
iterations' incumbent value; C-F10 iterations at init; I-F8/C-F13 final message
and yval_vec shape."""
import logging

import common
import numpy as np

from pybads import BADS


class H(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, rec):
        self.msgs.append(rec.getMessage())


class Probe(BADS):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.polls = []
        self.searches = 0
        self.same_gp = []

    def _search_step_(self, gp):
        self.searches += 1
        return super()._search_step_(gp)

    def _poll_step_(self, gp):
        f0, s0 = self.fval, self.fsd
        out = super()._poll_step_(gp)
        self.same_gp.append(out[-1] is gp)
        act = []
        if self.gp_refitted_flag:
            act.append(
                "Train" + (" (failed)" if self.gp_exit_flag < 0 else "")
            )
        if self.last_skipped == self.optim_state["iter"]:
            act.append("skip" if act else "Skip")
        self.polls.append(
            dict(
                it=self.optim_state["iter"],
                shown=self.logging_action[-1],
                matlab=", ".join(act) if act else "",
                fval_in=f0,
                fsd_in=s0,
            )
        )
        return out


def mk(f, D, **o):
    oo = dict(display="off", random_seed=0, max_fun_evals=200)
    oo.update(o)
    return Probe(
        f,
        np.full(D, 0.5),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=oo,
    )


rosen = lambda x: float(
    np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
)
# C-F9 and K8
for so in [dict(), dict(search_n_try=0)]:
    b = mk(rosen, 3, **so)
    r = b.optimize()
    mism = [
        (p["it"], p["shown"], p["matlab"])
        for p in b.polls
        if p["shown"] != p["matlab"]
    ]
    print(
        f"C-F9 {so}: polls {len(b.polls)}, Actions shown != MATLAB's action: {len(mism)}; first: {mism[:5]}"
    )
    print(
        f"K8 {so}: poll returned the same GP object in {sum(b.same_gp)} of {len(b.same_gp)} polls"
    )
# I-F9 max_iter
for mi in [1, 2, 3]:
    b = mk(rosen, 3, max_iter=mi)
    r = b.optimize()
    print(
        f"I-F9 max_iter={mi}: polls {len(b.polls)}, searches {b.searches}, iterations {r['iterations']}, msg {r['message'][-30:]}"
    )
# I-F10 first iterations of a noisy run
nrng = np.random.default_rng(0)
fn = lambda x: float(np.sum(np.ravel(x) ** 2) + 0.5 * nrng.standard_normal())
b = mk(fn, 2, uncertainty_handling=True)
r = b.optimize()
fl = b.function_logger
raw_min = float(np.min(fl.Y[: b.optim_state["eff_starting_points"]]))
print(
    f"I-F10: design raw minimum {raw_min:.4f}; incumbent (fval, fsd) entering polls 1-3: "
    f"{[(round(p['fval_in'], 4), round(p['fsd_in'], 4)) for p in b.polls[:3]]}"
)
# C-F10 iterations when the run ends in its initialization
for uh in [None, True]:
    b = mk(
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        2,
        max_fun_evals=1,
        uncertainty_handling=uh,
    )
    r = b.optimize()
    print(
        f"C-F10 max_fun_evals=1 uncertainty_handling={uh}: iterations {r['iterations']} (MATLAB: iter = 1 before the loop, bads.m:482)"
    )
# I-F8 / C-F13
for label, o, stn in [
    (
        "level 1, nfs=1",
        dict(noise_final_samples=1, uncertainty_handling=True),
        False,
    ),
    (
        "level 1, nfs=0",
        dict(noise_final_samples=0, uncertainty_handling=True),
        False,
    ),
    (
        "level 1, nfs=3",
        dict(noise_final_samples=3, uncertainty_handling=True),
        False,
    ),
    (
        "level 2, nfs=1",
        dict(noise_final_samples=1, specify_target_noise=True),
        True,
    ),
]:
    nrng = np.random.default_rng(1)
    ff = (
        (
            lambda x: (
                float(np.sum(np.ravel(x) ** 2) + 0.5 * nrng.standard_normal()),
                0.5,
            )
        )
        if stn
        else fn
    )
    b = mk(ff, 2, max_fun_evals=120, display="iter", **o)
    h = H()
    b.logger.addHandler(h)
    b.logger.propagate = False
    r = b.optimize()
    yv = r["yval_vec"]
    last = [m for m in h.msgs if "function value at minimum" in m]
    print(
        f"{label}: yval_vec shape {None if yv is None else np.shape(yv)}; last incumbent yval {b.optim_state['yval_vec']} ; "
        f"returned fval {r['fval']:.4f} fsd {r['fsd']:.4f}; message: {last}"
    )
