"""B3-K6 / comparison F10 / B3-K13: an empty search set in _search_step_."""
import numpy as np
import vhdr  # noqa

import pybads.search.es_search as es_mod
from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge

orig_cc = es_mod.contraints_check
orig_hedge = ESSearchHedge.__call__
orig_step = BADS._search_step_
orig_impr = BADS._eval_improvement_
orig_stats = BADS._update_search_stats_
EMPTY = {2, 3}
state = {"n": 0, "empty": False}
log = []


def cc(U, *a, **k):
    out = orig_cc(U, *a, **k)
    return out[:0] if state["empty"] else out


def hedge(self, *a, **k):
    state["n"] += 1
    state["empty"] = state["n"] in EMPTY
    try:
        return orig_hedge(self, *a, **k)
    finally:
        state["empty"] = False


def step(self, gp):
    g0 = (
        None if self.search_es_hedge is None else self.search_es_hedge.g.copy()
    )
    sf0 = self.optim_state["search_factor"]
    out = orig_step(self, gp)
    if state["n"] in EMPTY or state["n"] - 1 in EMPTY:
        log.append(
            dict(
                call=state["n"],
                u_search=out[0],
                f_sd=out[3],
                f_sd_type=type(out[3]).__name__,
                dist=out[1],
                sf=(sf0, self.optim_state["search_factor"]),
                g=(g0, self.search_es_hedge.g.copy()),
                count=self.optim_state["search_count"],
                fsd_inc=self.fsd,
            )
        )
    return out


impr = []


def ev(self, f_base, f_new, s_base, s_new, q):
    z = orig_impr(self, f_base, f_new, s_base, s_new, q)
    impr.append((state["n"], float(np.ravel(z)[0]), s_base, s_new))
    return z


stat = []


def st(self, s, d):
    stat.append((state["n"], s))
    return orig_stats(self, s, d)


es_mod.contraints_check = cc
ESSearchHedge.__call__ = hedge
BADS._search_step_ = step
BADS._eval_improvement_ = ev
BADS._update_search_stats_ = st

D = 3
for label, opts, fun in (
    ("level 0, defaults", {}, lambda x: float(np.sum(np.atleast_2d(x) ** 2))),
    (
        "level 1, improvement_quantile=0.75",
        {"uncertainty_handling": True, "improvement_quantile": 0.75},
        (
            lambda r: (
                lambda x: float(np.sum(np.atleast_2d(x) ** 2))
                + r.standard_normal()
            )
        )(np.random.default_rng(0)),
    ),
):
    state["n"] = 0
    log.clear()
    impr.clear()
    stat.clear()
    o = {"display": "off", "random_seed": 0, "max_fun_evals": 80}
    o.update(opts)
    res = BADS(
        fun,
        np.full(D, 2.0),
        np.full(D, -10.0),
        np.full(D, 10.0),
        np.full(D, -3.0),
        np.full(D, 3.0),
        options=o,
    ).optimize()
    print(
        f"--- {label}: run finished, fval={res['fval']:.4g}, evals={res['func_count']}, searches={state['n']}"
    )
    for e in log:
        print(
            f"  search #{e['call']}: u_search={None if e['u_search'] is None else 'point'}, returned f_sd={e['f_sd']!r} ({e['f_sd_type']}), "
            f"dist={e['dist']!r}, search_factor {e['sf'][0]:.4f}->{e['sf'][1]:.4f} (count {e['count']}), "
            f"hedge g unchanged={np.array_equal(e['g'][0], e['g'][1]) if e['g'][0] is not None else 'n/a'}, incumbent fsd={e['fsd_inc']:.3g}"
        )
    print(
        "  improvement computed at the empty searches:",
        [(n, round(z, 4)) for n, z, sb, sn in impr if n in EMPTY],
    )
    print(
        "  status at the empty searches:", [s for n, s in stat if n in EMPTY]
    )
