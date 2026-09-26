import gpyreg
import numpy as np

import pybads.bads.bads as bads_module
from pybads import BADS

print(bads_module.__file__, gpyreg.__file__)
predictions = []
orig_acq = bads_module.acq_fcn_lcb
orig_save = BADS._save_gp_stats_


def spy_acq(xi, fc, gp, *a, **k):
    out = orig_acq(xi, fc, gp, *a, **k)
    predictions.append((xi, gp, out))
    return out


cnt = [0]


def spy_save(self, fval, ymu, ys):
    xi, gp, (z, f_mu, fs) = predictions[-1]
    i = np.argmin(z)
    _, f2 = gp.predict(xi[i : i + 1])
    _, y2 = gp.predict(xi[i : i + 1], add_noise=True)
    nl = gp.get_hyperparameters()[0]["noise_log_scale"].item()
    mult = [p.sn2_mult for p in gp.posteriors]
    if cnt[0] < 15 or abs(y2.item() - f2.item() - np.exp(2 * nl)) > 1e-12:
        print(
            cnt[0],
            "ys",
            ys,
            "lat",
            np.sqrt(f2.item()),
            "noise",
            np.exp(nl),
            "y2-f2",
            y2.item() - f2.item(),
            "exp2nl",
            np.exp(2 * nl),
            "mult",
            mult,
            "s2",
            None if gp.s2 is None else gp.s2.ravel()[:3],
        )
    cnt[0] += 1
    return orig_save(self, fval, ymu, ys)


bads_module.acq_fcn_lcb = spy_acq
BADS._save_gp_stats_ = spy_save
D = 3
b = BADS(
    lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
    np.ones(D) * 4,
    -100 * np.ones(D),
    100 * np.ones(D),
    -8 * np.ones(D),
    12 * np.ones(D),
    options={"display": "off", "max_fun_evals": 60, "random_seed": 0},
)
b.optimize()
print("noise hyp names", b.gp_stats.keys() if False else "")
