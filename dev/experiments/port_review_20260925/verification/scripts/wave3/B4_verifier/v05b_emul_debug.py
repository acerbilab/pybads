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
b.optimize()
H = b.iteration_history
for k, g in enumerate(H.get("gp")):
    if g is None:
        continue
    p = g.posteriors[0]
    u = H.get("u")[k]
    mu_h, s2_h = hybrid_predict(g, g.get_hyperparameters(as_array=True), u)
    mu_p, s2_p = g.predict(np.atleast_2d(u))
    print(
        k,
        "L_chol",
        p.L_chol,
        "sn2_mult",
        p.sn2_mult,
        "s2 emul",
        s2_h,
        "s2 gp",
        s2_p.ravel(),
        "mu equal",
        np.allclose(mu_h, mu_p.ravel()),
    )
