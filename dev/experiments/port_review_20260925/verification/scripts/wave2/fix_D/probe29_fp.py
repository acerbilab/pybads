"""The fingerprint's six runs: at each poll, the iteration, whether it
failed, and the improvement over the iterate accelerate_mesh_steps back."""
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
orig = BADS._poll_step_


def poll(self, gp):
    it = self.optim_state["iter"]
    m0 = self.mesh_size_integer
    out = orig(self, gp)
    steps = self.options["accelerate_mesh_steps"]
    imp = None
    if it >= steps:
        imp = np.ravel(
            self._eval_improvement_(
                self.iteration_history.get("fval")[it - steps],
                self.fval,
                self.iteration_history.get("fsd")[it - steps],
                self.fsd,
                self.options["improvement_quantile"],
            )
        )[0]
    self.plog.append((it, m0 - self.mesh_size_integer, imp))
    return out


BADS._poll_step_ = poll


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


g = np.random.default_rng(0)


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        b = BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        )
        b.plog = []
        r = b.optimize()
        print(
            noisy,
            seed,
            r["iterations"],
            [
                (i, d, None if imp is None else round(imp, 4))
                for i, d, imp in b.plog
            ],
        )
