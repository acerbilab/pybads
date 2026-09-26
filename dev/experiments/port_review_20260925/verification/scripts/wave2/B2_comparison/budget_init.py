import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)


def run(fun, D, opts, label):
    x0 = np.full(D, 0.5)
    lb = np.full(D, -5.0)
    ub = np.full(D, 5.0)
    plb = np.full(D, -2.0)
    pub = np.full(D, 2.0)
    o = dict(display="off", random_seed=0)
    o.update(opts)
    b = BADS(fun, x0, lb, ub, plb, pub, options=o)
    r = b.optimize()
    print(
        f"{label}: max_fun_evals(user)={opts.get('max_fun_evals')} func_count={r['func_count']} "
        f"target={r['target_type']} iterations={r['iterations']} "
        f"noise_final_samples(after)={b.options['noise_final_samples']} "
        f"max_fun_evals(after)={b.options['max_fun_evals']} "
        f"eff_starting_points={b.optim_state['eff_starting_points']} yval_vec={r['yval_vec']}\n   msg={r['message']}"
    )


sphere = lambda x: float(np.sum(x**2))
nrng = np.random.default_rng(1)
noisy = lambda x: float(np.sum(x**2) + nrng.normal())
run(sphere, 2, dict(max_fun_evals=3), "det D=2")
run(sphere, 3, dict(max_fun_evals=4), "det D=3")
run(noisy, 2, dict(max_fun_evals=25), "noisy D=2 (test infers level 1)")
nrng = np.random.default_rng(1)
run(noisy, 2, dict(max_fun_evals=40), "noisy D=2 max 40")
