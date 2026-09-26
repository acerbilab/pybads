"""F6 (internal) / F7 part 1 (comparison): does reading n = 1 as 'no stats' change poll-stop decisions?"""
import logging

import common  # noqa
import numpy as np
from matlab_transcriptions import gppredcheck

from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)


class B(BADS):
    def _is_poll_stop_(self, good, unrel, p_less, poll_count):
        st = self.gp_stats
        n = 0 if st.get("iter_gp") is None else len(st.get("iter_gp"))
        dec = super()._is_poll_stop_(good, unrel, p_less, poll_count)
        if n == 1:
            fv = np.array(st.get("fval"), float)
            mu = np.array(st.get("ymu"), float)
            s = np.array(st.get("ys"), float)
            s[np.isclose(0.0, s)] = 1e-6
            unrel_m = bool(
                gppredcheck(fv, mu, s, self.options["normalpha_level"])
            )
            saved = self.last_skipped
            dec_m = super()._is_poll_stop_(good, unrel_m, p_less, poll_count)
            self.last_skipped = saved if not dec else self.last_skipped
            self._n1.append((good, unrel, unrel_m, dec, dec_m))
        return dec


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


tot = []
for name, fun, D, seed in [
    ("rosen3", rosen, 3, 100),
    ("ell4", ell, 4, 100),
    ("ell2", ell, 2, 100),
    ("rosen2", rosen, 2, 100),
]:
    b = B(
        fun,
        0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options={"random_seed": seed, "display": "off", "max_fun_evals": 200},
    )
    b._n1 = []
    b.optimize()
    L = b._n1
    print(
        f"{name} seed {seed}: poll-stop checks at n=1: {len(L)}; python unreliable {sum(x[1] for x in L)}, MATLAB-rule unreliable {sum(x[2] for x in L)}; "
        f"with a good poll {sum(x[0] for x in L)}; decisions that differ {sum(x[3] != x[4] for x in L)}"
    )
