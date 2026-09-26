"""I-F13 / C-F3: the accelerated mesh reduction's guard. At each failed poll,
evaluate MATLAB's condition (iter_m > steps, iter_m = iter_py + 1) beside the
port's (iter_py > steps), with the same base index."""
import common
import numpy as np

from pybads import BADS


class Probe(BADS):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log = []

    def _poll_step_(self, gp):
        msi0 = self.mesh_size_integer
        it = self.optim_state["iter"]
        steps = self.options["accelerate_mesh_steps"]
        out = super()._poll_step_(gp)
        dec = msi0 - self.mesh_size_integer
        if dec > 0:  # failed poll
            py_guard = it > steps
            m_guard = (it + 1) > steps
            m_accel = None
            if m_guard:
                idx = it - steps
                fb = self.iteration_history.get("fval")[idx]
                sb = self.iteration_history.get("fsd")[idx]
                imp = self._eval_improvement_(
                    fb,
                    self.fval,
                    sb,
                    self.fsd,
                    self.options["improvement_quantile"],
                )
                m_accel = bool(imp < self.options["tol_fun"])
            self.log.append((it, dec, py_guard, m_guard, m_accel))
        return out


def run(f, x0, D, seed, **opts):
    o = dict(display="off", random_seed=seed, max_fun_evals=200)
    o.update(opts)
    b = Probe(
        f,
        x0,
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=o,
    )
    r = b.optimize()
    return b, r


cases = [
    (
        "sphere D=2 x0=0",
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        np.zeros(2),
        2,
    ),
    (
        "sphere D=2 x0=1",
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        np.ones(2),
        2,
    ),
    (
        "rosen D=2",
        lambda x: float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2),
        np.array([-1.0, 1.0]),
        2,
    ),
    (
        "ellipsoid D=4",
        lambda x: float(np.sum((10.0 ** np.arange(4) * np.ravel(x)) ** 2)),
        np.ones(4),
        4,
    ),
]
for name, f, x0, D in cases:
    for seed in range(2):
        b, r = run(f, x0, D, seed)
        fails = [(it, dec) for it, dec, *_ in b.log]
        diffs = [e for e in b.log if e[3] and not e[2]]
        print(
            f"{name} seed {seed}: iterations {r['iterations']}, failed polls at 0-based iters "
            f"{[it for it, _ in fails][:14]}; iteration where only MATLAB tests: "
            f"{[(e[0], 'MATLAB would halve again' if e[4] else 'no extra halving') for e in diffs]}"
        )
