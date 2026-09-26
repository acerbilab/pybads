"""K2 / C-F4 and I-F3 / C-F1 at level 1 (noise inferred), wide box:
stop decisions after a good poll with a reliable GP; those where
gp_poll_hyp_best differs from the GP's hyperparameters; decision flips under
MATLAB's hybrid target and under the current GP's own prediction; and flips
of MATLAB's p_less rule."""
import sys

import numpy as np
from scipy.special import erfc
from vhdr import ellipsoid, rosen
from vhybrid import hybrid_predict

import pybads.bads.bads as bm
from pybads import BADS


def ackley(x):
    x = np.ravel(x)
    D = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / D))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / D)
        + 20
        + np.e
    )


def pl_port(fpi, D):
    f = np.sort(fpi)[::-1]
    return np.prod(1 - f[0 : min(D + 1, len(f))])


def pl_mat(fpi, D):
    f = np.sort(np.ravel(fpi))[::-1]
    return np.prod(1 - f[: min(D, len(f))])


S = {}
orig_stop = bm.BADS._is_poll_stop_


def stop(self, good, unrel, p_less, poll_count):
    L = sys._getframe(1).f_locals
    D, tol = self.D, self.options["tol_poi"]
    if good and not unrel and np.all(np.isfinite(L["gamma_z"])):
        S["rel"] += 1
        fpi = 0.5 * erfc(-L["gamma_z"] / np.sqrt(2))
        dp = p_less > 1 - tol
        S["flip_pless"] += dp != (pl_mat(fpi, D) > 1 - tol)
        gp = L["gp"]
        hb = np.atleast_2d(L["gp_poll_hyp_best"])
        hc = gp.get_hyperparameters(as_array=True)
        if not np.allclose(hb, hc):
            S["hypdiff"] += 1
            sdl, tf, SI = (
                self.optim_state["sd_level"],
                self.options["tol_fun"],
                self.sufficient_improvement,
            )
            u = L["u_poll_best"]
            mh, s2h = hybrid_predict(gp, hb, u)
            mc, s2c = gp.predict(np.atleast_2d(u))
            ft = {
                "hyb": mh.item() - sdl * np.sqrt(s2h.item() + tf**2),
                "cur": mc.item() - sdl * np.sqrt(s2c.item() + tf**2),
            }
            for k, v in ft.items():
                gz = (v - SI - L["f_mu"]) / L["fs"]
                d = (
                    (pl_port(0.5 * erfc(-gz / np.sqrt(2)), D) > 1 - tol)
                    if np.all(np.isfinite(gz))
                    else True
                )
                S["flip_" + k] += d != dp
            S["ex"].append(
                (
                    round(self.optim_state["f_target"], 4),
                    round(ft["hyb"], 4),
                    round(ft["cur"], 4),
                )
            )
    return orig_stop(self, good, unrel, p_less, poll_count)


bm.BADS._is_poll_stop_ = stop

tot = dict(rel=0, hypdiff=0, flip_hyb=0, flip_cur=0, flip_pless=0)
for name, f, D, seed in (
    ("rosen", rosen, 2, 0),
    ("rosen", rosen, 2, 1),
    ("ackley", ackley, 3, 0),
    ("ellipsoid", ellipsoid, 3, 0),
    ("rosen", rosen, 4, 0),
):
    S.update(rel=0, hypdiff=0, flip_hyb=0, flip_cur=0, flip_pless=0, ex=[])
    nrng = np.random.default_rng(1000 + seed)
    fun = lambda x, f=f: f(x) + nrng.normal()
    r = BADS(
        fun,
        4 * np.ones(D),
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=dict(
            random_seed=seed,
            display="off",
            max_fun_evals=200,
            uncertainty_handling=True,
        ),
    ).optimize()
    print(
        f"{name} D={D} seed={seed} fval={r['fval']:.4g}: relevant={S['rel']} hyp differs={S['hypdiff']} "
        f"flip hybrid={S['flip_hyb']} flip current-GP={S['flip_cur']} flip p_less rule={S['flip_pless']} "
        f"(port, hybrid, current) targets={S['ex'][:3]}",
        flush=True,
    )
    for k in tot:
        tot[k] += S[k]
print("total:", tot)
