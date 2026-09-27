"""Capture (u, gp, optim_state, func_logger) at each search call of a seeded run."""
import copy

import numpy as np

from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def capture_states(D=3, seed=0, max_fun_evals=120, fun=rosen, max_states=20):
    states = []
    orig = ESSearchHedge.__call__

    def call(self, u, lb, ub, func_logger, gp, optim_state):
        if len(states) < max_states:
            states.append(
                dict(
                    u=np.array(u, dtype=float).copy(),
                    gp=copy.deepcopy(gp),
                    optim_state=copy.deepcopy(optim_state),
                    func_logger=copy.deepcopy(func_logger),
                    options=self.options_dict,
                )
            )
        return orig(self, u, lb, ub, func_logger, gp, optim_state)

    ESSearchHedge.__call__ = call
    try:
        BADS(
            fun,
            np.zeros(D),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": max_fun_evals,
            },
        ).optimize()
    finally:
        ESSearchHedge.__call__ = orig
    return states
