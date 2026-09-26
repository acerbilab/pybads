import collections

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.bads as bb
from pybads import BADS

cnt = collections.Counter()
orig = bb.BADS._is_poll_stop_


def stop(self, certain_good_poll, do_gp_calibration, p_less, poll_count):
    it = self.gp_stats.get("iter_gp")
    n = 0 if it is None else len(it)
    out = orig(self, certain_good_poll, do_gp_calibration, p_less, poll_count)
    if certain_good_poll:
        cnt["good-poll stop checks"] += 1
        if n in (1, 2):
            cnt[f"n={n}: flagged unreliable"] += bool(do_gp_calibration)
            if do_gp_calibration:
                cnt[
                    f"n={n}: stopped because flagged, PoI rule would have continued"
                ] += not (p_less > 1 - self.options["tol_poi"])
    return out


bb.BADS._is_poll_stop_ = stop


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, f, D in [
    ("rosen2", rosen, 2),
    ("ell4", ell, 4),
    ("rosen4", rosen, 4),
]:
    cnt.clear()
    b = BADS(
        f,
        np.full((1, D), 1.5),
        np.full((1, D), -10.0),
        np.full((1, D), 10.0),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        options={"random_seed": 0, "display": "off", "max_fun_evals": 200},
    )
    b.optimize()
    print(name, dict(cnt))
