"""K4: in _re_evaluate_history_, whose len_scale / effective_radius choose the
neighbours of iterate i: the working GP's (MATLAB: gpstruct passed in, only hyp
swapped, gpupdate.m:85-97 without refit) or the stored GP i's?"""
import copy

import common
import numpy as np

import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

CTX = dict(active=False, calls=[])
orig = gpt.get_grid_search_neighbors


def patched(function_logger, u, gp, options, optim_state):
    out = orig(function_logger, u, gp, options, optim_state)
    if CTX["active"]:
        CTX["calls"].append(
            (
                np.array(gp.temporary_data["len_scale"]).copy(),
                float(np.ravel(gp.temporary_data["effective_radius"])[0]),
                u.copy(),
                out[0].copy(),
            )
        )
    return out


gpt.get_grid_search_neighbors = patched


class Probe(BADS):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stats = dict(
            iterates=0,
            same_as_working=0,
            same_as_stored=0,
            stored_geom_differs=0,
            set_differs=0,
            n_pts=[],
        )

    def _re_evaluate_history_(self, gp):
        if self.optim_state["last_re_eval"] == self.function_logger.func_count:
            return super()._re_evaluate_history_(gp)
        CTX.update(active=True, calls=[])
        wl = np.array(gp.temporary_data["len_scale"]).copy()
        wr = float(np.ravel(gp.temporary_data["effective_radius"])[0])
        stored = self.iteration_history.get("gp")
        super()._re_evaluate_history_(gp)
        CTX["active"] = False
        st = self.stats
        for i, (ls, er, u, Xn) in enumerate(CTX["calls"]):
            st["iterates"] += 1
            st["n_pts"].append(len(Xn))
            st["same_as_working"] += bool(np.array_equal(ls, wl) and er == wr)
            sg = stored[i]
            sl = np.array(sg.temporary_data["len_scale"])
            sr = float(np.ravel(sg.temporary_data["effective_radius"])[0])
            st["same_as_stored"] += bool(np.array_equal(ls, sl) and er == sr)
            if not (np.array_equal(sl, wl) and sr == wr):
                st["stored_geom_differs"] += 1
                o = copy.deepcopy(self.optim_state)
                Xs, _, _ = orig(self.function_logger, u, sg, self.options, o)
                a = {tuple(r) for r in np.round(Xn, 12)}
                bset = {tuple(r) for r in np.round(Xs, 12)}
                st["set_differs"] += a != bset


for seed in range(3):
    nrng = np.random.default_rng(40 + seed)
    f = lambda x: float(
        np.sum((np.ravel(x) - 0.2) ** 2) + 0.3 * nrng.standard_normal()
    )
    b = Probe(
        f,
        np.array([1.2, -0.8, 0.5]),
        np.full(3, -5.0),
        np.full(3, 5.0),
        np.full(3, -2.0),
        np.full(3, 2.0),
        options=dict(
            uncertainty_handling=True,
            max_fun_evals=200,
            random_seed=seed,
            display="off",
        ),
    )
    r = b.optimize()
    s = b.stats
    print(
        f"seed {seed}: func_count {r['func_count']}, iterate rebuilds {s['iterates']}, geometry = working GP's {s['same_as_working']}, "
        f"= stored GP's {s['same_as_stored']}, stored geometry differs from working {s['stored_geom_differs']}, "
        f"of which neighbour set would differ {s['set_differs']}; training-set sizes {min(s['n_pts'])}-{max(s['n_pts'])}"
    )
