"""Level 0: after a poll point improves, the poll's target is predicted at
that point by a GP that does not hold it. Compare the prediction with the
observed value, and count the poll stops that follow."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS

in_poll = {"v": False}
orig_poll = bm.BADS._poll_step_


def poll(self, gp):
    in_poll["v"] = True
    try:
        return orig_poll(self, gp)
    finally:
        in_poll["v"] = False


bm.BADS._poll_step_ = poll

rec = []
orig_t = bm.BADS._get_target_from_gp_


def tgt(self, u, gp, hyp_best):
    out = orig_t(self, u, gp, hyp_best)
    if in_poll["v"]:
        u2 = np.atleast_2d(u)
        in_gp = np.any(np.all(np.abs(gp.X - u2) < 1e-12, axis=1))
        fl = self.function_logger
        n = fl.X_max_idx + 1
        m = np.all(np.abs(fl.X[:n] - u2) < 1e-12, axis=1)
        y = fl.Y[:n][m][-1] if np.any(m) else np.nan
        mu, s2 = gp.predict(u2)
        rec.append(
            (
                bool(in_gp),
                np.ravel(out[0])[0],
                np.ravel(y)[0],
                np.sqrt(np.ravel(s2)[0]),
                np.ravel(out[2])[0],
                np.ravel(self.fval)[0],
                float(self.sufficient_improvement),
            )
        )
    return out


bm.BADS._get_target_from_gp_ = tgt

stops = {"good": 0}
orig_stop = bm.BADS._is_poll_stop_


def stop(self, good, cal, p_less, count):
    r = orig_stop(self, good, cal, p_less, count)
    if r and good:
        stops["good"] += 1
    return r


bm.BADS._is_poll_stop_ = stop


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / (D - 1)) * x) ** 2))


for name, f, D in [("rosen", rosen, 3), ("ellip", ellip, 4)]:
    rec.clear()
    stops["good"] = 0
    lb = -5 * np.ones((1, D))
    ub = 5 * np.ones((1, D))
    plb = -2 * np.ones((1, D))
    pub = 2 * np.ones((1, D))
    x0 = np.full((1, D), 1.5)
    b = BADS(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": 1, "display": "off", "max_fun_evals": 200},
    )
    r = b.optimize()
    outside = [x for x in rec if not x[0]]
    print(
        name,
        "poll target calls:",
        len(rec),
        "at a point outside the GP:",
        len(outside),
        "stops after a good poll:",
        stops["good"],
    )
    for x in outside[:12]:
        in_gp, mu, y, s, ft, fval, si = x
        print(
            "   pred %.4g  observed %.4g  pred-obs %.3g  sd %.3g  target %.4g  incumbent %.4g  suff.impr %.3g"
            % (mu, y, mu - y, s, ft, fval, si)
        )
