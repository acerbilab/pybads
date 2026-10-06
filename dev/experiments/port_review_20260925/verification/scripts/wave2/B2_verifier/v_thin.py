"""K6: a thin feasible band as non_box_cons. Course of the run: the initial
design the constraint leaves, the GP it is trained on, and the termination."""
import logging

import common
import numpy as np

import pybads.bads.bads as bmod
from pybads import BADS


class Probe(BADS):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trace = []

    def _poll_step_(self, gp):
        n0 = self.function_logger.func_count
        msi = self.mesh_size_integer
        out = super()._poll_step_(gp)
        self.trace.append(
            (
                "poll",
                self.optim_state["iter"],
                self.function_logger.func_count - n0,
                msi,
                self.mesh_size_integer,
                None if gp.X is None else len(gp.X),
            )
        )
        return out

    def _search_step_(self, gp):
        n0 = self.function_logger.func_count
        out = super()._search_step_(gp)
        self.trace.append(
            (
                "search",
                self.optim_state["iter"],
                self.function_logger.func_count - n0,
                None,
                None,
                len(gp.X),
            )
        )
        return out


class H(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, rec):
        self.msgs.append((rec.levelname, rec.getMessage()))


for D, seed, noisy in [
    (2, 0, False),
    (2, 1, False),
    (3, 0, False),
    (2, 0, True),
]:
    nrng = np.random.default_rng(seed)
    f = (
        (
            lambda x: float(
                np.sum((np.ravel(x) - 0.3) ** 2) + 0.1 * nrng.standard_normal()
            )
        )
        if noisy
        else (lambda x: float(np.sum((np.ravel(x) - 0.3) ** 2)))
    )
    nbc = (
        lambda X: np.abs(np.atleast_2d(X)[:, 0] - np.atleast_2d(X)[:, 1])
        - 0.005
    )
    b = Probe(
        f,
        np.full(D, 0.5),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        nbc,
        options=dict(
            display="full",
            random_seed=seed,
            max_fun_evals=200,
            uncertainty_handling=noisy or None,
        ),
    )
    h = H()
    b.logger.addHandler(h)
    b.logger.propagate = False
    try:
        r = b.optimize()
        res = f"func_count {r['func_count']}, iterations {r['iterations']}, x {np.round(r['x'], 4)}, fval {r['fval']:.4g}, msg: {r['message']}"
    except Exception as e:
        res = f"{type(e).__name__}: {e}"
    gp_n = [t[5] for t in b.trace]
    print(
        f"D={D} seed={seed} noisy={noisy}: design points kept {b.optim_state.get('eff_starting_points', 0) - 1}; "
        f"steps {[(t[0][0], t[1], t[2]) for t in b.trace][:12]}; GP sizes {gp_n[:12]}"
    )
    print("    ", res)
    warn = [
        m
        for lv, m in h.msgs
        if lv in ("WARNING", "ERROR")
        or "wrong" in m.lower()
        or "fail" in m.lower()
    ]
    print("     warnings/failures:", sorted(set(warn))[:6])
