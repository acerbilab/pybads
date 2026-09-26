"""Level 0: runs with the poll's target at a polled point that the GP lacks
predicted by a GP that holds the point (a copy of the GP updated with the
observation), against the runs as coded."""
import copy

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS

MODE = {"fix": False}
ctx = {"poll": False}
cnt = {}
orig_poll = bm.BADS._poll_step_


def poll(self, gp):
    ctx["poll"] = True
    try:
        return orig_poll(self, gp)
    finally:
        ctx["poll"] = False


bm.BADS._poll_step_ = poll
orig_t = bm.BADS._get_target_from_gp_


def tgt(self, u, gp, hyp_best):
    u2 = np.atleast_2d(u)
    if (
        MODE["fix"]
        and ctx["poll"]
        and not np.any(np.all(np.abs(gp.X - u2) < 1e-12, axis=1))
    ):
        fl = self.function_logger
        n = fl.X_max_idx + 1
        m = np.all(np.abs(fl.X[:n] - u2) < 1e-12, axis=1)
        if np.any(m):
            g2 = copy.deepcopy(gp)
            g2.update(X_new=u2, y_new=np.atleast_2d(fl.Y[:n][m][-1]))
            cnt["n"] = cnt.get("n", 0) + 1
            return orig_t(self, u, g2, hyp_best)
    return orig_t(self, u, gp, hyp_best)


bm.BADS._get_target_from_gp_ = tgt


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / (D - 1)) * x) ** 2))


for name, f, D in [
    ("rosen", rosen, 3),
    ("ellip", ellip, 4),
    ("ellip", ellip, 6),
]:
    for seed in [1, 2, 3]:
        out = []
        for fix in [False, True]:
            MODE["fix"] = fix
            cnt.clear()
            lb = -5 * np.ones((1, D))
            ub = 5 * np.ones((1, D))
            plb = -2 * np.ones((1, D))
            pub = 2 * np.ones((1, D))
            r = BADS(
                f,
                np.full((1, D), 1.5),
                lb,
                ub,
                plb,
                pub,
                options={
                    "random_seed": seed,
                    "display": "off",
                    "max_fun_evals": 200,
                },
            ).optimize()
            out.append(
                "%s: fval %.4g evals %d %s"
                % (
                    "with point" if fix else "coded",
                    r["fval"],
                    r["func_count"],
                    dict(cnt),
                )
            )
        print(name, D, seed, " | ".join(out))
