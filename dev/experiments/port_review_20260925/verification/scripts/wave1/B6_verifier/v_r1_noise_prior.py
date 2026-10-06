"""B6-R1: is the noise prior's centre updated at each rebuild?

Wrap pybads.bads.bads.local_gp_fitting; after each call record the noise
prior that the GP holds, the mesh size, and MATLAB's centre
log(NoiseSize(1)) + MeshNoiseMultiplier*log(MeshSize) (gpdefBads.m:207).
"""

import numpy as np
from common import bads_mod, noisy, noisy_with_sd, rosenbrock, sphere

from pybads import BADS

orig = bads_mod.local_gp_fitting


def run(fun, D, level_opts, seed, label):
    rec = []

    def wrapped(
        gp,
        current_point,
        function_logger,
        options,
        optim_state,
        iteration_history,
        refit_flag,
        rng=None,
    ):
        out = orig(
            gp,
            current_point,
            function_logger,
            options,
            optim_state,
            iteration_history,
            refit_flag,
            rng=rng,
        )
        g = out[0]
        pr = g.get_priors()["noise_log_scale"]
        if options.get("specify_target_noise"):
            ns = options["tol_fun"]
        else:
            ns = np.ravel(options["noise_size"])[0]
        matlab_mu = np.log(ns) + options["mesh_noise_multiplier"] * np.log(
            optim_state["mesh_size"]
        )
        hyp = g.get_hyperparameters()[0]["noise_log_scale"]
        rec.append(
            (
                optim_state["mesh_size"],
                float(np.ravel(pr[1][0])[0]),
                float(np.ravel(pr[1][1])[0]),
                float(matlab_mu),
                float(np.ravel(hyp)[0]),
                bool(refit_flag),
            )
        )
        return out

    bads_mod.local_gp_fitting = wrapped
    try:
        x0 = np.full(D, 0.7)
        lb, ub = np.full(D, -5.0), np.full(D, 5.0)
        plb, pub = np.full(D, -2.0), np.full(D, 2.0)
        opts = dict(display="off", random_seed=seed, max_fun_evals=200)
        opts.update(level_opts)
        res = BADS(fun, x0, lb, ub, plb, pub, options=opts).optimize()
    finally:
        bads_mod.local_gp_fitting = orig
    rec = np.array(rec, dtype=float)
    ms = rec[:, 0]
    print(
        f"--- {label}: {len(rec)} rebuilds, fval={res['fval']:.4g}, "
        f"func_count={res['func_count']}"
    )
    print(
        "   distinct prior centres held by the GP:",
        np.unique(np.round(rec[:, 1], 6)),
    )
    print("   distinct prior SDs:", np.unique(np.round(rec[:, 2], 6)))
    print(
        f"   MATLAB centre range: [{rec[:,3].min():.3f}, "
        f"{rec[:,3].max():.3f}]; mesh range [{ms.min():.3g}, {ms.max():.3g}]"
    )
    print(
        f"   largest |PyBADS - MATLAB centre|: "
        f"{np.max(np.abs(rec[:,1]-rec[:,3])):.3f}"
    )
    fit = rec[rec[:, 5] == 1]
    if len(fit):
        print(
            f"   fitted log sn at refits: min {fit[:,4].min():.3f}, "
            f"median {np.median(fit[:,4]):.3f}, max {fit[:,4].max():.3f}"
        )
    return rec


if __name__ == "__main__":
    run(sphere, 2, {}, 0, "level 0, sphere D=2")
    run(rosenbrock, 3, {}, 1, "level 0, rosenbrock D=3")
    run(
        noisy(sphere, 1.0, 3),
        2,
        {"uncertainty_handling": True},
        3,
        "level 1, noisy sphere D=2 (sd 1)",
    )
    run(
        noisy_with_sd(sphere, 0.5, 4),
        2,
        {"uncertainty_handling": True, "specify_target_noise": True},
        4,
        "level 2, sphere D=2 (sd 0.5 given)",
    )
