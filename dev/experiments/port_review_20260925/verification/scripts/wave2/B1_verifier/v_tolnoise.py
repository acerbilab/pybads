"""B1 verifier, K4 / C-F3: tol_noise = eps*tol_fun (PyBADS) against
sqrt(eps)*TolFun (MATLAB, bads.m:195), on two nearly deterministic
targets: a sum of 1000 terms added in a random order (rounding only), and
the sphere + 1 with N(0, 1e-12) jitter. D = 2, 100 evaluations, seed 5;
the MATLAB threshold is set through the tol_noise option."""
import common  # noqa: F401
import numpy as np

from pybads import BADS

D = 2
bnd = (np.full(D, -5.0), np.full(D, 5.0), np.full(D, -2.0), np.full(D, 2.0))
w = np.random.default_rng(0).uniform(0.5, 1.5, 1000) / 1000


def make_targets():
    r1 = np.random.default_rng(101)
    r2 = np.random.default_rng(202)

    def shuffled(x):
        x = np.asarray(x).ravel()
        terms = w * (1.0 + np.sum(x**2))
        return float(sum(terms[r1.permutation(terms.size)]))

    def jitter(x):
        x = np.asarray(x).ravel()
        return 1.0 + float(np.sum(x**2)) + 1e-12 * r2.normal()

    return {"shuffled sum": shuffled, "jitter 1e-12": jitter}


for thr_name, thr in [
    ("PyBADS eps*tol_fun", None),
    ("MATLAB sqrt(eps)*tol_fun", np.sqrt(np.spacing(1.0)) * 1e-3),
]:
    for name, fun in make_targets().items():
        o = {"display": "off", "random_seed": 5, "max_fun_evals": 100}
        if thr is not None:
            o["tol_noise"] = thr
        b = BADS(fun, np.full(D, 1.3), *bnd, options=o)
        y1 = fun(np.full(D, 0.7))
        y2 = fun(np.full(D, 0.7))
        r = b.optimize()
        print(
            f"{thr_name:26s} {name:13s} |repeat diff| {abs(y1 - y2):.2e} "
            f"threshold {b.options['tol_noise']:.2e} -> level "
            f"{b.optim_state['uncertainty_handling_level']}, "
            f"evals {r['func_count']}, fval-1 {r['fval'] - 1:.2e}, "
            f"|x| {np.linalg.norm(r['x']):.2e}, fsd {r['fsd']:.2e}"
        )
