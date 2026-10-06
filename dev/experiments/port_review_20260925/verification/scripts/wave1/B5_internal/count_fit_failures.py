import collections

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.bads as bb
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

stats = collections.Counter()
orig_fit = gpyreg.GP.fit


def counted_fit(self, *a, **k):
    stats["fit_calls"] += 1
    try:
        return orig_fit(self, *a, **k)
    except np.linalg.LinAlgError:
        stats["fit_linalg"] += 1
        raise


gpyreg.GP.fit = counted_fit
orig_rob = gpt._robust_gp_fit_


def rob(*a, **k):
    out = orig_rob(*a, **k)
    stats[f"robust_exit_{out[3]}"] += 1
    return out


gpt._robust_gp_fit_ = rob
orig_prior = gpt._get_random_samples_from_priors_


def pri(*a, **k):
    stats["prior_sampler_calls"] += 1
    return orig_prior(*a, **k)


gpt._get_random_samples_from_priors_ = pri
orig_lgf = gpt.local_gp_fitting


def lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    out = orig_lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)
    stats["rebuilds"] += 1
    if refit_flag:
        stats["refits"] += 1
        if np.any(optim_state.get("second_fit", False)):
            stats["second_fits"] += 1
    if out[1] == -2:
        stats["rebuild_exit_-2"] += 1
    return out


bb.local_gp_fitting = lgf


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


def noisy_sphere_factory(seed):
    r = np.random.default_rng(seed)
    return lambda x: float(np.sum(np.atleast_2d(x) ** 2) + r.normal())


cases = [
    ("rosen D=2", rosen, 2, None),
    ("ell D=4", ell, 4, None),
    ("noisy sphere D=3", "noisy", 3, None),
]
for name, fun, D, _ in cases:
    for seed in [0, 1, 2]:
        stats.clear()
        f = noisy_sphere_factory(100 + seed) if fun == "noisy" else fun
        x0 = np.full((1, D), 1.5)
        b = BADS(
            f,
            x0,
            np.full((1, D), -10.0),
            np.full((1, D), 10.0),
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        try:
            r = b.optimize()
            res = f"fval {r['fval']:.3g}"
        except Exception as e:
            res = f"RAISED {type(e).__name__}: {e}"
        print(f"{name} seed {seed}: {res} | {dict(stats)}")
