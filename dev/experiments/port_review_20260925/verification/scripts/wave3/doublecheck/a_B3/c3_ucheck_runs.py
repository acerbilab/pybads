"""W3-1 in seeded runs: at every call of contraints_check (the initial
design, the search step, the ES, the poll), its output against (a) the same
code with MATLAB's round in its bins and (b) uCheck.m transcribed; and at
every ES search, the point returned with each variant from the same state of
the generator."""

import hdr  # noqa: F401
import numpy as np
from cc_variants import matlab_ucheck, port_mround

import pybads.bads.bads as bads_module
import pybads.search.es_search as es_module
from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge

port_cc = es_module.contraints_check
stats = {}


def key(site):
    return stats.setdefault(
        site,
        {
            "calls": 0,
            "fine": 0,
            "diff_round": 0,
            "diff_matlab": 0,
            "removed_diff": 0,
        },
    )


def wrap(site):
    def cc(U, lb, ub, tol_mesh, fl, proj=True, non_box_cons=None):
        out = port_cc(U, lb, ub, tol_mesh, fl, proj, non_box_cons)
        s = key(site)
        s["calls"] += 1
        a = port_mround(U, lb, ub, tol_mesh, fl, proj, non_box_cons)
        b = matlab_ucheck(U, lb, ub, tol_mesh, fl, proj, non_box_cons)
        if not (out.shape == a.shape and np.array_equal(out, a)):
            s["diff_round"] += 1
        if not (out.shape == b.shape and np.array_equal(out, b)):
            s["diff_matlab"] += 1
        return out

    return cc


es_stats = {"calls": 0, "diff_round": 0, "diff_matlab": 0}
orig_hedge_call = ESSearchHedge.__call__


def hedge_call(self, *args, **kwargs):
    state = self.rng.bit_generator.state
    g = self.g.copy()
    count = self.count
    results = {}
    for name, fn in [("round", port_mround), ("matlab", matlab_ucheck)]:
        self.rng.bit_generator.state = state
        self.g = g.copy()
        self.count = count
        es_module.contraints_check = fn
        results[name] = orig_hedge_call(self, *args, **kwargs)
    es_module.contraints_check = wrap("es")
    self.rng.bit_generator.state = state
    self.g = g.copy()
    self.count = count
    out = orig_hedge_call(self, *args, **kwargs)
    es_stats["calls"] += 1
    for name in results:
        us = results[name][0]
        if not (
            np.shape(us) == np.shape(out[0]) and np.array_equal(us, out[0])
        ):
            es_stats["diff_" + name] += 1
    return out


ESSearchHedge.__call__ = hedge_call
bads_module.contraints_check = wrap("bads.py")
es_module.contraints_check = wrap("es")


def edge_sphere(x):
    return float(np.sum((np.atleast_2d(x) + 1.0) ** 2))


def sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def noisy_sphere(seed):
    r = np.random.default_rng(seed)
    return lambda x: float(np.sum(np.atleast_2d(x) ** 2)) + r.standard_normal()


configs = []
for seed in (0, 1):
    configs.append(
        (
            "edge_D1",
            edge_sphere,
            np.array([2.5]),
            np.zeros(1),
            5 * np.ones(1),
            np.zeros(1),
            5 * np.ones(1),
            {"random_seed": seed},
        )
    )
    configs.append(
        (
            "edge_D2",
            edge_sphere,
            2.5 * np.ones(2),
            np.zeros(2),
            5 * np.ones(2),
            np.zeros(2),
            5 * np.ones(2),
            {"random_seed": seed},
        )
    )
    configs.append(
        (
            "sphere_D3",
            sphere,
            2 * np.ones(3),
            -10 * np.ones(3),
            10 * np.ones(3),
            -5 * np.ones(3),
            5 * np.ones(3),
            {"random_seed": seed},
        )
    )
    configs.append(
        (
            "noisy_D2",
            noisy_sphere(seed),
            2 * np.ones(2),
            -10 * np.ones(2),
            10 * np.ones(2),
            -5 * np.ones(2),
            5 * np.ones(2),
            {"random_seed": seed, "uncertainty_handling": True},
        )
    )

for name, f, x0, lb, ub, plb, pub, opts in configs:
    stats.clear()
    for k in es_stats:
        es_stats[k] = 0
    opts = {"display": "off", "max_fun_evals": 200, **opts}
    b = BADS(f, x0, lb, ub, plb, pub, options=opts)
    r = b.optimize()
    print(
        f"{name} seed {opts['random_seed']}: fval {r['fval']:.3g} evals {r['func_count']} "
        f"mesh_size end {b.mesh_size:.3g}",
        flush=True,
    )
    for site, s in stats.items():
        print(
            f"   {site}: calls {s['calls']}, output differs with MATLAB's round {s['diff_round']}, "
            f"from uCheck.m {s['diff_matlab']}",
            flush=True,
        )
    print(
        f"   ES searches {es_stats['calls']}: point returned differs with MATLAB's round "
        f"{es_stats['diff_round']}, with uCheck.m {es_stats['diff_matlab']}",
        flush=True,
    )
