"""K2 / C-F4: the port's target prediction under hyp_best (a posterior
recomputed under hyp_best) against MATLAB's (the current posterior with
hyp_best's kernel and mean), on GPs and hyperparameters of a real run."""
import copy

import numpy as np
from vhdr import box, rosen
from vhybrid import hybrid_predict

from pybads import BADS

D = 3
lb, ub, plb, pub = box(D)
b = BADS(
    rosen,
    1.5 * np.ones(D),
    lb,
    ub,
    plb,
    pub,
    options=dict(random_seed=2, display="off", max_fun_evals=150),
)
r = b.optimize()
H = b.iteration_history
gps = H.get("gp")
hyps = H.get("gp_hyp_full")
us = H.get("u")
n = len([g for g in gps if g is not None])
print("iterations recorded:", n)
# emulation check: with the GP's own hyperparameters it reproduces gp.predict
g = gps[n - 1]
mu_h, s2_h = hybrid_predict(g, g.get_hyperparameters(as_array=True), us[n - 1])
mu_p, s2_p = g.predict(np.atleast_2d(us[n - 1]))
print(
    "emulation == gp.predict with own hyp:",
    np.allclose(mu_h, mu_p.ravel()),
    np.allclose(s2_h, s2_p.ravel(), atol=1e-12),
)
sd_level, tol_fun = b.optim_state["sd_level"], b.options["tol_fun"]
print(
    f"{'iter':>4} {'hyp from':>8} {'port mu':>11} {'port s':>10} {'matlab mu':>11} {'matlab s':>10} {'port tgt':>11} {'matlab tgt':>11}"
)
for k in range(n - 1, max(n - 7, 0), -1):
    g = gps[k]
    for j in (k - 1, k - 3):
        if j < 0:
            continue
        hb = np.atleast_2d(hyps[j])
        if np.allclose(hb, g.get_hyperparameters(as_array=True)):
            continue
        tmp = copy.deepcopy(g)
        tmp.set_hyperparameters(hb)
        mu_p, s2_p = tmp.predict(np.atleast_2d(us[k]))
        mu_m, s2_m = hybrid_predict(g, hb, us[k])
        tp = mu_p.item() - sd_level * np.sqrt(s2_p.item() + tol_fun**2)
        tm = mu_m.item() - sd_level * np.sqrt(s2_m.item() + tol_fun**2)
        print(
            f"{k:>4} {j:>8} {mu_p.item():>11.4g} {np.sqrt(s2_p.item()):>10.3g} {mu_m.item():>11.4g} {np.sqrt(s2_m.item()):>10.3g} {tp:>11.4g} {tm:>11.4g}"
        )
print(
    "observed yval at those iterates:",
    [round(float(v), 5) for v in H.get("yval")[max(n - 7, 0) : n]],
)
