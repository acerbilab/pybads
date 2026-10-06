"""O-K1. (A) Reachability: count the hyperparameter samples that the GP
holds after every local_gp_fitting and at every _robust_gp_fit_ entry, in
runs at default options (level 0, level 1) and with double_refit=True and
gp_samples=5 (level 1). (B) With a GP forced to hold two samples, compare the
geometry of local_gp_fitting with a transcription of gpupdate.m:283-315
(hypweight = 1/2 each)."""
import copy

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
import pybads.bads.bads as bm
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

REC = {"after": [], "robust_in": []}
orig_lgf = bm.local_gp_fitting
orig_rob = gpt._robust_gp_fit_


def lgf(*a, **k):
    gp, flag = orig_lgf(*a, **k)
    REC["after"].append(len(gp.get_hyperparameters()))
    er = gp.temporary_data.get("effective_radius")
    REC.setdefault("er_size", []).append(np.size(er))
    return gp, flag


def rob(gp, X, Y, s2, hyp_gp, *a, **k):
    REC["robust_in"].append(np.atleast_2d(hyp_gp).shape[0])
    out = orig_rob(gp, X, Y, s2, hyp_gp, *a, **k)
    REC.setdefault("robust_out", []).append(np.atleast_2d(out[1]).shape[0])
    return out


bm.local_gp_fitting = lgf
gpt._robust_gp_fit_ = rob

rng_noise = np.random.default_rng(7)


def f_det(x):
    x = np.asarray(x)
    return float(np.sum(x**2 * np.array([1.0, 4.0, 0.25])))


def f_noisy(x):
    return f_det(x) + float(rng_noise.standard_normal())


import os

PARTS = os.environ.get("PARTS", "AB")
for label, fun, extra in (
    []
    if "A" not in PARTS
    else [
        ("level 0, default", f_det, {}),
        ("level 1, default", f_noisy, {"uncertainty_handling": True}),
        (
            "level 1, double_refit, gp_samples=5",
            f_noisy,
            {
                "uncertainty_handling": True,
                "double_refit": True,
                "gp_samples": 5,
            },
        ),
    ]
):
    for key in list(REC):
        REC[key] = []
    opts = {"display": "off", "random_seed": 11, "max_fun_evals": 150}
    opts.update(extra)
    b = BADS(
        fun,
        np.array([2.0, -1.5, 3.0]),
        -10 * np.ones(3),
        10 * np.ones(3),
        -5 * np.ones(3),
        5 * np.ones(3),
        options=opts,
    )
    b.optimize()
    print(
        f"(A) {label}: local_gp_fitting calls {len(REC['after'])}; samples after: "
        f"{sorted(set(REC['after']))}; effective_radius sizes {sorted(set(REC['er_size']))}; "
        f"_robust_gp_fit_ calls {len(REC['robust_in'])}, starts at entry {sorted(set(REC['robust_in']))}, "
        f"rows returned {sorted(set(REC.get('robust_out', [])))}",
        flush=True,
    )

# (B) two samples forced
bm.local_gp_fitting = orig_lgf
gpt._robust_gp_fit_ = orig_rob
b = BADS(
    f_det,
    np.array([2.0, -1.5, 3.0]),
    -10 * np.ones(3),
    10 * np.ones(3),
    -5 * np.ones(3),
    5 * np.ones(3),
    options={"display": "off", "random_seed": 11, "max_fun_evals": 60},
)
b.optimize()
gp = copy.deepcopy(b.iteration_history.get("gp")[b.optim_state["iter"]])
PERT = np.array([0.4, -0.3, 0.2])


def rob2(gp, X, Y, s2, hyp_gp, *a, **k):
    gp, hyp, res, ef = orig_rob(gp, X, Y, s2, hyp_gp, *a, **k)
    h2 = hyp.copy()
    d = gp.hyperparameters_to_dict(h2)
    d[0]["covariance_log_lengthscale"] = (
        d[0]["covariance_log_lengthscale"] + PERT
    )
    d[0]["covariance_log_shape"] = d[0]["covariance_log_shape"] + 1.0
    h2 = gp.hyperparameters_from_dict(d)
    hyp2 = np.vstack([hyp, h2])
    gp.set_hyperparameters(hyp2, compute_posterior=False)
    return gp, hyp2, res, ef


gpt._robust_gp_fit_ = rob2
os_ = copy.deepcopy(b.optim_state)
gp2, ef = gpt.local_gp_fitting(
    gp,
    b.u,
    b.function_logger,
    b.options,
    os_,
    b.iteration_history,
    True,
    rng=np.random.default_rng(0),
)
hyps = gp2.get_hyperparameters()
print("(B) samples held:", len(hyps), flush=True)
D = 3
w = np.full(len(hyps), 1 / len(hyps))
logl = np.array([h["covariance_log_lengthscale"] for h in hyps]).T  # D x N
ll = b.options["gp_rescale_poll"] * logl
ll_m = np.exp(np.sum(w * (ll - np.mean(ll)), axis=1))
ub = os_["ub"].copy()
lb = os_["lb"].copy()
ub[~np.isfinite(ub)] = os_["pub"][~np.isfinite(ub)]
lb[~np.isfinite(lb)] = os_["plb"][~np.isfinite(lb)]
ll_m = np.minimum(
    np.maximum(ll_m, os_["search_mesh_size"]), (ub - lb) / os_["scale"]
)
alpha = np.array([np.exp(h["covariance_log_shape"]).item() for h in hyps])
a_m = np.sum(w * alpha)
er_m = np.sqrt(a_m * (np.exp(1 / a_m) - 1))
len_m = np.sum(w[None, :] * np.exp(logl), axis=1)
print(
    "(B) poll_scale PyBADS:",
    np.ravel(gp2.temporary_data["poll_scale"]),
    flush=True,
)
print("(B) poll_scale MATLAB:", np.ravel(ll_m), flush=True)
print(
    "(B) effective_radius PyBADS:",
    gp2.temporary_data["effective_radius"],
    flush=True,
)
print("(B) effective_radius MATLAB:", er_m, flush=True)
print(
    "(B) len_scale PyBADS:",
    gp2.temporary_data["len_scale"],
    " MATLAB:",
    len_m,
    flush=True,
)
