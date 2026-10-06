"""I-F6 / C-F2: evaluations of points already in the function log (within
tol_mesh/2), by stage, in default runs; MATLAB's uCheck would drop them."""
import numpy as np
from vhdr import box, rosen

import pybads.bads.bads as bm
from pybads import BADS

S = {"stage": "init"}
orig_call = bm.FunctionLogger.__call__


def call(self, x, record_duplicate_data=True):
    if record_duplicate_data and self.X_max_idx >= 0:
        st = S["stage"]
        S.setdefault("n_" + st, 0)
        S["n_" + st] += 1
        u2 = np.round(self.X[: self.X_max_idx + 1] / 5e-7)
        if np.any(np.all(u2 == np.round(np.ravel(x) / 5e-7), axis=1)):
            S.setdefault("re_" + st, []).append(
                np.round(np.ravel(x), 4).tolist()
            )
    return orig_call(self, x, record_duplicate_data)


bm.FunctionLogger.__call__ = call
for meth, st in (("_search_step_", "search"), ("_poll_step_", "poll")):
    orig = getattr(bm.BADS, meth)

    def wrap(self, gp, orig=orig, st=st):
        S["stage"] = st
        try:
            return orig(self, gp)
        finally:
            S["stage"] = "other"

    setattr(bm.BADS, meth, wrap)

runs = [(3, s, False, box(3), 1.5) for s in (1, 2)] + [
    (
        D,
        s,
        True,
        (
            -100 * np.ones(D),
            100 * np.ones(D),
            -8 * np.ones(D),
            12 * np.ones(D),
        ),
        4.0,
    )
    for D, s in ((2, 0), (2, 1), (4, 0))
]
for D, seed, noisy, (lb, ub, plb, pub), x0 in runs:
    for k in [k for k in S if k != "stage"]:
        del S[k]
    S["stage"] = "init"
    opts = dict(random_seed=seed, display="off", max_fun_evals=200)
    fun = rosen
    if noisy:
        nrng = np.random.default_rng(1000 + seed)
        fun = lambda x: rosen(x) + nrng.normal()
        opts["uncertainty_handling"] = True
    r = BADS(fun, x0 * np.ones(D), lb, ub, plb, pub, options=opts).optimize()
    print(
        f"rosen D={D} lvl={int(noisy)} seed={seed} fval={r['fval']:.4g}: evals poll={S.get('n_poll',0)} "
        f"search={S.get('n_search',0)}; re-evaluations poll={len(S.get('re_poll',[]))} {S.get('re_poll',[])[:3]} "
        f"search={len(S.get('re_search',[]))}",
        flush=True,
    )
