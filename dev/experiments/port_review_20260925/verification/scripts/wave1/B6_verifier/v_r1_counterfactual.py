"""B6-R1 consequence: default level-0 runs with the port's local_gp_fitting
against a copy that writes the noise prior back (MATLAB's update).
Same seed, same target, 200 evaluations at most."""

import time

import numpy as np
from common import (
    ackley,
    bads_mod,
    patched_local_gp_fitting,
    rosenbrock,
    sphere,
)

from pybads import BADS

orig = bads_mod.local_gp_fitting
fixed = patched_local_gp_fitting(True)


def one(fun, D, seed, variant):
    bads_mod.local_gp_fitting = fixed if variant == "matlab" else orig
    fitted = []
    inner = bads_mod.local_gp_fitting

    def wrapped(*a, **k):
        out = inner(*a, **k)
        if a[6]:
            fitted.append(
                float(
                    np.ravel(
                        out[0].get_hyperparameters()[0]["noise_log_scale"]
                    )[0]
                )
            )
        return out

    bads_mod.local_gp_fitting = wrapped
    try:
        x0 = np.full(D, 0.7)
        lb, ub = np.full(D, -5.0), np.full(D, 5.0)
        plb, pub = np.full(D, -2.0), np.full(D, 2.0)
        t = time.time()
        res = BADS(
            fun,
            x0,
            lb,
            ub,
            plb,
            pub,
            options=dict(display="off", random_seed=seed, max_fun_evals=200),
        ).optimize()
        dt = time.time() - t
    finally:
        bads_mod.local_gp_fitting = orig
    fitted = np.array(fitted)
    at_lb = np.mean(np.isclose(fitted, np.log(1e-3) - 1, atol=1e-3))
    return res["fval"], res["func_count"], res["iterations"], at_lb, dt


if __name__ == "__main__":
    cases = [
        (sphere, 2, "sphere"),
        (rosenbrock, 3, "rosenbrock"),
        (ackley, 4, "ackley"),
    ]
    for fun, D, name in cases:
        for seed in (0, 1):
            a = one(fun, D, seed, "port")
            b = one(fun, D, seed, "matlab")
            print(
                f"{name} D={D} seed={seed}: port fval={a[0]:.3e} "
                f"n={a[1]} it={a[2]} refits-at-noise-LB={a[3]:.2f} "
                f"({a[4]:.0f}s) | with update fval={b[0]:.3e} n={b[1]} "
                f"it={b[2]} refits-at-noise-LB={b[3]:.2f} ({b[4]:.0f}s)"
            )
