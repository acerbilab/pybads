"""W3-4, W3-5, W3-8, W3-9, W3-15: ESSearch.__call__ against a transcription
of MATLAB BADS's searchES.m and ESupdate.m (74919c0), on the same inputs and
the same random draws. To compare the ES's own logic, the transcription uses
the port's force_to_grid, contraints_check and LCB, and the port's
eigendecomposition (scipy.linalg.eigh) for MATLAB's eig."""
import hdr  # noqa: F401
import numpy as np
import scipy

import pybads.search.es_search as es_module
from pybads import BADS
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb
from pybads.function_examples import rosenbrocks_fcn
from pybads.function_logger.constraints_check import contraints_check
from pybads.search.es_search import ESSearchELL, ESSearchWM, ucov
from pybads.search.grid_functions import force_to_grid


def esupdate_selectmask(mu, lam):
    tot = mu + lam
    s = 1.0 / np.sqrt(np.arange(1, tot + 1))
    w = np.ceil(s / np.sum(s) * lam).astype(int)
    nonzero = np.sum(w > 0)
    while np.sum(w) - lam > nonzero:
        w = np.maximum(0, w - 1)
        nonzero = np.sum(w > 0)
    delta = np.sum(w) - lam
    last = np.flatnonzero(w > 0)[-1] + 1
    w[max(1, last - delta + 1) - 1 : last] -= 1
    cw = np.cumsum(w) - w + 1
    idx = np.zeros(np.max(cw), dtype=int)
    idx[cw - 1] = 1
    return np.cumsum(idx[:-1])  # 1-based


def searchES_matlab(
    method,
    u,
    gp,
    optim_state,
    options,
    rng,
    non_box_cons,
    fl,
    nan_minmax=False,
):
    """searchES.m, methods 1 (ES-wcm) and 2 (ES-ell), sumrule = 1."""
    MeshSize = optim_state["mesh_size"]
    SearchFactor = optim_state["search_factor"]
    U = gp.X
    Y = gp.y.ravel()
    nvars = u.size
    if method == 1:
        jit = MeshSize
        frac = 0.5
        mu = frac * U.shape[0]
        weights = np.log(mu + 0.5) - np.log(
            np.arange(1, int(np.floor(mu)) + 1)
        )
        weights = weights / np.sum(weights)
        index = np.argsort(Y, kind="stable")  # sort(Y,'ascend')
        Ubest = U[index[: int(np.floor(mu))]]
        C = ucov(
            Ubest,
            u,
            weights,
            optim_state["ub"],
            optim_state["lb"],
            optim_state["scale"],
            optim_state["periodic_vars"],
        )
        lam, E = scipy.linalg.eigh(C)
        lam = np.maximum(0, lam) + jit**2
        lam = lam / np.sum(lam)
        sqrtsigma = np.diag(np.sqrt(lam)) @ E.T
    else:
        r = gp.temporary_data["poll_scale"]
        r = r / np.sqrt(np.sum(r**2))
        sqrtsigma = np.diag(r)
    sqrtsigma = MeshSize * SearchFactor * sqrtsigma
    N = int(options["n_search"] / options["n_search_iter"])  # es.mu
    lam_es = N
    w = options["poll_mesh_multiplier"] ** np.array([-1.0, 0.0])
    ns = np.diff(np.round(np.linspace(0, N, w.size + 1)).astype(int))
    v = np.concatenate([w[i] * np.ones(ns[i]) for i in range(w.size)])[:, None]
    unew = u + v * (rng.normal(size=(N, nvars)) @ sqrtsigma)
    scale = options["es_start"]
    us = np.zeros((0, nvars))
    zold = np.zeros(0)
    hist = []
    Niter = options["n_search_iter"]
    for i in range(1, Niter + 1):
        unew = force_to_grid(unew, optim_state["search_mesh_size"])
        if nan_minmax:  # MATLAB's min/max ignore NaN
            unew = np.fmax(
                np.fmin(unew, optim_state["ub_search"]),
                optim_state["lb_search"],
            )
        unew = contraints_check(
            unew,
            optim_state["lb_search"],
            optim_state["ub_search"],
            optim_state["tol_mesh"],
            fl,
            True,
            non_box_cons,
        )
        z, _, _ = acq_fcn_lcb(
            unew, fl.func_count, gp, options["search_acq_fcn"][1]
        )
        z = np.ravel(z)
        nold = us.shape[0]
        us = np.vstack([us, unew])
        z = np.concatenate([zold, z])
        Nsel = min(us.shape[0], lam_es)
        index = np.argsort(z, kind="stable")  # MATLAB's sort, stable
        z = z[index]
        ntest = min(unew.shape[0], nold)
        nnew = int(np.sum(index[:ntest] + 1 > nold))  # 1-based index > nold
        zold = z[:Nsel]
        us = us[index[:Nsel]]
        if i < Niter:
            with np.errstate(invalid="ignore", divide="ignore"):
                frac_ = nnew / ntest if ntest > 0 else np.nan  # 0/0 = NaN
            if i > 1:
                scale = scale * np.exp(options["es_beta"] * (frac_ - 0.2))
            if us.shape[0] > 0:
                mask = esupdate_selectmask(us.shape[0], lam_es)
            else:
                mask = np.zeros(0, dtype=int)
            ll = min(lam_es, us.shape[0])
            unew = (
                us[mask[:ll] - 1]
                + (rng.normal(size=(ll, nvars)) @ sqrtsigma) * scale
            )
        hist.append((ntest, nnew, scale))
    if us.shape[0] > 0:
        return us[0], zold[0], scale, hist
    return us, zold, scale, hist


def state(D=3, seed=0, **options):
    b = BADS(
        rosenbrocks_fcn,
        np.zeros((1, D)),
        -20 * np.ones((1, D)),
        20 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        options={"random_seed": seed, "display": "off", **options},
    )
    b.options["fun_eval_start"] = 10
    gp, _, _, _ = b._init_optimization_()
    return b, gp


# Record the port's ntest/n_new per generation
orig_call = es_module.ESSearch.__call__

agree = total = 0
report = []
for n_iter in (2, 3, 4, 5):
    for seed in range(4):
        for method in (1, 2):
            for sf in (1.0, 4.0):
                b, gp = state(seed=seed, n_search_iter=n_iter)
                b.optim_state["search_factor"] = sf
                mu = int(b.options["n_search"] / n_iter)
                cls = ESSearchWM if method == 1 else ESSearchELL
                s = cls(
                    mu, mu, b.options, rng=np.random.default_rng(100 + seed)
                )
                us_p, z_p = s(
                    b.u,
                    None,
                    None,
                    b.function_logger,
                    gp,
                    b.optim_state,
                    True,
                    None,
                )
                us_m, z_m, scale_m, hist = searchES_matlab(
                    method,
                    b.u,
                    gp,
                    b.optim_state,
                    b.options,
                    np.random.default_rng(100 + seed),
                    None,
                    b.function_logger,
                )
                ok = (
                    np.array_equal(us_p, us_m)
                    and np.array_equal(np.ravel(z_p), np.ravel(z_m))
                    and np.isclose(s.scale, scale_m, rtol=0, atol=0)
                )
                total += 1
                agree += ok
                if not ok:
                    report.append(
                        (
                            n_iter,
                            seed,
                            method,
                            sf,
                            us_p,
                            us_m,
                            s.scale,
                            scale_m,
                        )
                    )
print(
    f"ES search against searchES.m (no emptied generation): same point, value and scale in {agree}/{total}"
)
for r in report[:5]:
    print("  differs:", r)

# An emptied generation: a constraint rejects every candidate of generation k
for n_iter, k in [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3)]:
    res = []
    for impl in ("port", "matlab"):
        b, gp = state(seed=0, n_search_iter=n_iter)
        calls = {"n": 0}

        def nbc(x):
            calls["n"] += 1
            return np.full(len(x), 1.0 if calls["n"] == k else 0.0)

        mu = int(b.options["n_search"] / n_iter)
        if impl == "port":
            s = ESSearchWM(mu, mu, b.options, rng=np.random.default_rng(7))
            us, z = s(
                b.u,
                None,
                None,
                b.function_logger,
                gp,
                b.optim_state,
                True,
                nbc,
            )
            res.append((us, np.ravel(z), s.scale))
        else:
            us, z, sc, hist = searchES_matlab(
                1,
                b.u,
                gp,
                b.optim_state,
                b.options,
                np.random.default_rng(7),
                nbc,
                b.function_logger,
                nan_minmax=True,
            )
            res.append((us, np.ravel(z), sc))
    (up, zp, sp), (um, zm, sm) = res
    print(
        f"n_search_iter {n_iter}, generation {k} emptied: port point {np.round(up, 6).tolist()} scale {sp:.4g}; "
        f"searchES.m point {np.round(um, 6).tolist()} scale {sm}; same point {np.array_equal(up, um)}"
    )

# W3-5's numbers: offspring of parents 0-5 at mu = lambda = 2048, and the mask sums
b, gp = state()
s = ESSearchWM(2048, 2048, b.options, rng=np.random.default_rng(0))
m = s._get_selection_idx_mask_(2048, 2048)
mm = esupdate_selectmask(2048, 2048)
print(
    "mask == selectmask - 1:",
    np.array_equal(m, mm - 1),
    "; sum 0-based",
    int(m.sum()),
    "; 1-based",
    int(mm.sum()),
)
print(
    "offspring of parents 0-5:",
    np.bincount(m)[:6].tolist(),
    "; old shifted mask would give",
    np.bincount(np.concatenate([[0], mm[:-1]]))[:6].tolist() if True else None,
)
bad = 0
for mu_ in list(range(1, 80)) + [100, 683, 1024, 1365, 2048, 4096]:
    for lam_ in list(range(1, 80)) + [100, 683, 1024, 1365, 2048, 4096]:
        if not np.array_equal(
            s._get_selection_idx_mask_(mu_, lam_),
            esupdate_selectmask(mu_, lam_) - 1,
        ):
            bad += 1
print("mask mismatches over (mu, lambda) grid:", bad)
