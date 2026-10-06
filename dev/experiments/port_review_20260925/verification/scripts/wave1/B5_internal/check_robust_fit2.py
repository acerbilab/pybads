import copy

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS


def f(x):
    x = np.atleast_2d(x)
    return float(np.sum((np.array([1.0, 10.0, 0.3]) * x) ** 2))


b = BADS(
    f,
    np.array([[2.0, 2.0, 2.0]]),
    np.array([[-20.0] * 3]),
    np.array([[20.0] * 3]),
    np.array([[-5.0] * 3]),
    np.array([[5.0] * 3]),
    options={"random_seed": 1, "display": "off", "max_fun_evals": 60},
)
b.optimize()
gp_train = {
    "init_N": 8,
    "opts_N": 1,
    "n_samples": 0,
    "init_method": "rand",
    "tol_opt": 1e-5,
}
orig_fit = gpyreg.GP.fit
for n_fail in [0, 1, 2]:
    state = {"n": 0}

    def fail_k(self, *a, **k):
        state["n"] += 1
        lb = float(np.ravel(self.get_bounds()["noise_log_scale"][0])[0])
        print(
            f"   call {state['n']}: noise lower bound {lb:.4f}, n_train {len(a[1])}"
        )
        if state["n"] <= n_fail:
            raise np.linalg.LinAlgError("injected")
        return orig_fit(self, *a, **k)

    gpyreg.GP.fit = fail_k
    gp2 = copy.deepcopy(b.iteration_history["gp"][b.optim_state["iter"]])
    hyp = gp2.get_hyperparameters(as_array=True)
    out = gpt._robust_gp_fit_(
        gp2,
        gp2.X,
        gp2.y,
        gp2.s2,
        hyp,
        gp_train,
        b.optim_state,
        b.options,
        np.random.default_rng(0),
    )
    print(
        f"n_fail={n_fail}: exit flag {out[3]}, fitted noise {float(out[0].get_hyperparameters()[0]['noise_log_scale'][0]):.4f}, returned gp n_train {out[0].X.shape[0]}"
    )
