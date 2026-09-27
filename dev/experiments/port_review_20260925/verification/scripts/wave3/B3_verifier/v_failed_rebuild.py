"""B3-K2: the search after a failed rebuild of the local GP."""
import gpyreg as gpr
import numpy as np
import vhdr  # noqa

import pybads.bads.bads as bads_mod
import pybads.search.es_search as es_mod
from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge

TARGET = 5  # the search step whose rebuild fails
st = {"search": 0, "in_step": False, "fail": False}
orig_step, orig_local = BADS._search_step_, bads_mod.local_gp_fitting
orig_update, orig_seth = gpr.GP.update, gpr.GP.set_hyperparameters
orig_hedge, orig_lcb = ESSearchHedge.__call__, es_mod.acq_fcn_lcb
info = {}


def step(self, gp):
    st["search"] += 1
    st["in_step"] = True
    try:
        return orig_step(self, gp)
    finally:
        st["in_step"] = False


def local(gp, u, *a, **k):
    first = st["in_step"] and st["search"] == TARGET and "done" not in info
    if first:
        st["fail"] = True
        info["done"] = True
    try:
        out = orig_local(gp, u, *a, **k)
    finally:
        st["fail"] = False
    if first:
        info["markers"] = {
            m: out[0].temporary_data.get(m, False)
            for m in ("needs_rebuild", "needs_refit")
        }
        info["exit"] = out[1]
    return out


def upd(self, *a, **k):
    if (
        st["fail"] and not a and set(k) == {"hyp"}
    ):  # local_gp_fitting's gp.update(hyp=hyp_gp)
        raise np.linalg.LinAlgError("injected")
    return orig_update(self, *a, **k)


def seth(self, *a, **k):
    if (
        st["fail"] and len(a) == 1 and not k
    ):  # the retry gp.set_hyperparameters(old_hyp_gp)
        raise np.linalg.LinAlgError("injected")
    return orig_seth(self, *a, **k)


gens = []


def lcb(u, *a, **k):
    out = orig_lcb(u, *a, **k)
    gens.append((u.copy(), np.ravel(out[0]).copy()))
    return out


def hedge(self, u, lb, ub, fl, gp, os_):
    gens.clear()
    us, z = orig_hedge(self, u, lb, ub, fl, gp, os_)
    if st["search"] == TARGET:
        info["u"] = np.array(u).copy()
        info["gp_marked"] = gp.temporary_data.get("needs_rebuild", False)
        m, s2 = gp.predict(np.atleast_2d(u))
        info["gp_finite"] = bool(
            np.isfinite(m).all() and np.isfinite(s2).all()
        )
        info["py_point"] = np.array(us).copy()
        info["matlab_point"] = gens[0][0][
            0
        ].copy()  # z == 0 everywhere: stable sort keeps uCheck's order
        info["z_py"] = float(np.ravel(z)[0])
        info["z_at_matlab_point"] = float(gens[0][1][0])
        info["mesh"] = os_["search_mesh_size"]
    return us, z


BADS._search_step_ = step
bads_mod.local_gp_fitting = local
gpr.GP.update, gpr.GP.set_hyperparameters = upd, seth
ESSearchHedge.__call__ = hedge
es_mod.acq_fcn_lcb = lcb

D = 3
res = BADS(
    lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
    np.full(D, 1.5),
    np.full(D, -5.0),
    np.full(D, 5.0),
    np.full(D, -2.0),
    np.full(D, 2.0),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 80},
).optimize()
print(
    "failed rebuild at search",
    TARGET,
    "-> exit flag",
    info["exit"],
    "markers",
    info["markers"],
)
print(
    "GP handed to the ES is the restored one (marked):",
    info["gp_marked"],
    "| its predictions are finite:",
    info["gp_finite"],
)
h = info["mesh"]
print(
    "port evaluates the LCB minimizer of the restored GP: offset from u (grid units)",
    ((info["py_point"] - info["u"]) / h).round(1).tolist(),
    "LCB",
    round(info["z_py"], 4),
)
print(
    "MATLAB (z = 0 for all) would evaluate the lexicographically first candidate: offset",
    ((info["matlab_point"] - info["u"]) / h).round(1).tolist(),
    "LCB under the restored GP",
    round(info["z_at_matlab_point"], 4),
)
print("run finished: fval", res["fval"], "evals", res["func_count"])
