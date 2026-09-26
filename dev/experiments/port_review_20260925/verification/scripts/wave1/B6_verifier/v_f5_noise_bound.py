"""F5 (internal): the noise upper bound log SD 5 (SD 148; MATLAB
gpdefBads.m:161 has the same) against a target whose noise SD is 500."""

import numpy as np
from common import bads_mod

from pybads import BADS

orig = bads_mod.local_gp_fitting


def run(noise_size, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda x: float(
        1e4 * np.sum(np.atleast_1d(x) ** 2) + 500 * rng.standard_normal()
    )
    fits = []

    def w(*a, **k):
        out = orig(*a, **k)
        if a[6]:
            g = out[0]
            fits.append(
                (
                    float(
                        np.ravel(
                            g.get_hyperparameters()[0]["noise_log_scale"]
                        )[0]
                    ),
                    g.get_priors()["noise_log_scale"][1][0][0],
                    g.upper_bounds[g.D + 2],
                )
            )
        return out

    bads_mod.local_gp_fitting = w
    opts = dict(
        display="off",
        random_seed=seed,
        max_fun_evals=200,
        uncertainty_handling=True,
    )
    if noise_size is not None:
        opts["noise_size"] = noise_size
    try:
        r = BADS(
            f,
            np.full(2, 0.7),
            np.full(2, -5.0),
            np.full(2, 5.0),
            np.full(2, -2.0),
            np.full(2, 2.0),
            options=opts,
        ).optimize()
    finally:
        bads_mod.local_gp_fitting = orig
    F = np.array(fits)
    print(
        f"noise_size={noise_size}: x={np.round(r['x'], 3)} "
        f"fval={r['fval']:.4g} refits={len(F)}; prior centre "
        f"{F[0,1]:.2f}, upper bound {F[0,2]:.1f}; fitted log sn "
        f"{np.round(F[:,0], 2)}"
    )


print("true log SD of the noise:", round(np.log(500), 2))
run(500.0)
run(None)
