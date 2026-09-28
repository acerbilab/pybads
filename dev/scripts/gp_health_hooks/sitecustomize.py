"""Counters of the GP layer's numerical health, per PyBADS run.

Put this directory first on ``PYTHONPATH`` and set ``GP_HEALTH_OUT`` to a
directory: every Python process then wraps gpyreg's ``GP`` at startup and,
when it ran a ``population.py`` task, writes
``<label>_seed<seed>.json`` there at exit. Without ``GP_HEALTH_OUT`` it
counts nothing. It shadows any other ``sitecustomize`` (the one of the Linux
reference's container, ``/usr/lib/python3.11/sitecustomize.py``, is
empty).

The wrappers observe and never change what they wrap: they draw no random
number, set no attribute of the GP, and return what the wrapped method
returned. A population run with them gives the records of one without
them, which ``same_fields.py`` checks. What is counted:

- ``chol``: each factorization of the training covariance
  (``__training_cholesky``), by the outermost public method of ``GP`` that
  led to it (``fit``, ``update``, ``set_hyperparameters``, ``predict``, or
  ``direct:<PyBADS function>`` for PyBADS's direct calls of
  ``_GP__gp_obj_fun``), its kind (``nlZ``: an evaluation of the objective;
  ``post``: a posterior; ``lowfactor``: the factor that ``predict``
  rebuilds in the low-noise representation) and its representation
  (``L_chol``): the calls, those that multiplied the noise by ten at least
  once, those that failed ten times and raised, and a histogram of the
  number of failed attempts.
- ``posteriors``: after each outermost ``fit``, ``update`` and
  ``set_hyperparameters`` that returned, by PyBADS caller, whether a
  posterior keeps a noise multiplier above one (the inflated noise that
  the hyperparameters do not show), with examples.
- ``predict``: by PyBADS caller chain, the level of uncertainty handling
  and ``add_noise``, the calls, the points, the points whose returned
  variance is exactly 0, and the calls on a GP whose posterior keeps an
  inflated noise.
- ``zero_sd``: for the latent variances returned as exactly 0, the value
  before gpyreg's clamp at 0, recomputed from the first posterior as
  ``predict`` computes it, relative to the prior variance ``kss``; the
  distance to the nearest training input in length scales; the ratio of the
  signal variance to the effective noise; examples with the mesh size.
- ``log_prior``: the log prior's NaN and -inf values by outermost method,
  with the hyperparameters whose Gaussian prior lies outside their bounds
  or whose normalization constant is 0; ``obj_fun``: the NaN values of the
  objective; ``priors_outside``: per ``fit``, the blocks (``cov``,
  ``noise``, ``mean``) with a prior centre outside the bounds.
- ``train_n``: the training-set sizes of 3 or fewer points (all and
  distinct) after each outermost ``fit``, ``update`` and
  ``set_hyperparameters``, and the smallest size of the run.
- ``exceptions``: what the outermost public methods raised, by method and
  type: what PyBADS's guards catch; ``robust_fit``: the fits that
  ``_robust_gp_fit_`` makes, by its try index and outcome (a refit whose
  every try fails has ten failures and no ``ok``); ``fit_seconds``: the
  wall time of the fits that returned and of those that raised.

``raised`` counts a factorization that raised, after ten attempts, or
after one where the GP raises on a failed factorization; the last entry of
``fails_hist`` counts the same calls.

One knob changes results, for experiments: with
``GP_FORCE_RAISE_ON_CHOLESKY_FAILURE=1``, every ``GP`` is constructed with
``raise_on_cholesky_failure=True`` (gpyreg after 1.3.3; an older gpyreg
stops the process at startup), whatever the caller passes.
"""

import os

if os.environ.get("GP_FORCE_RAISE_ON_CHOLESKY_FAILURE") == "1":
    import functools
    import inspect

    from gpyreg.gaussian_process import GP as _GP

    if (
        "raise_on_cholesky_failure"
        not in inspect.signature(_GP.__init__).parameters
    ):
        # SystemExit, which site.py does not catch as it catches an
        # exception of sitecustomize
        raise SystemExit(
            "GP_FORCE_RAISE_ON_CHOLESKY_FAILURE: this gpyreg's GP has no "
            "raise_on_cholesky_failure"
        )
    _orig_gp_init = _GP.__init__

    @functools.wraps(_orig_gp_init)
    def _gp_init(self, *args, **kwargs):
        kwargs["raise_on_cholesky_failure"] = True
        _orig_gp_init(self, *args, **kwargs)

    _GP.__init__ = _gp_init

if os.environ.get("GP_HEALTH_OUT"):  # noqa: C901
    import atexit
    import collections
    import inspect
    import json
    import math
    import sys
    import time

    import numpy as np
    import scipy.linalg
    from gpyreg import gaussian_process as _gpmod
    from gpyreg.gaussian_process import GP

    _OUT = os.environ["GP_HEALTH_OUT"]
    _MAX_EXAMPLES = 25

    def _counter():
        return collections.defaultdict(int)

    _S = {
        "calls": _counter(),
        "exceptions": _counter(),
        "chol": collections.defaultdict(
            lambda: {
                "calls": 0,
                "inflated": 0,
                "raised": 0,
                "fails_hist": [0] * 11,
            }
        ),
        "posteriors": collections.defaultdict(
            lambda: {"returns": 0, "inflated": 0, "max_mult": 1.0}
        ),
        "posterior_examples": [],
        "predict": collections.defaultdict(
            lambda: {
                "calls": 0,
                "points": 0,
                "zero_points": 0,
                "calls_with_zero": 0,
                "calls_inflated_post": 0,
            }
        ),
        "zero_sd": {
            "points": 0,
            "raw_negative": 0,
            "raw_zero": 0,
            "raw_positive": 0,
            "raw_rel_log10_hist": _counter(),
            "dist_hist": _counter(),
            "snr_log10_hist": _counter(),
            "low_noise_repr": 0,
            "examples": [],
        },
        "log_prior": collections.defaultdict(
            lambda: {"calls": 0, "nan": 0, "neginf": 0}
        ),
        "log_prior_nan_examples": [],
        "obj_fun": collections.defaultdict(
            lambda: {"calls": 0, "nan": 0, "neginf_or_inf": 0}
        ),
        "priors_outside": collections.defaultdict(
            lambda: {"fits": 0, "outside": 0, "zero_norm": 0}
        ),
        "priors_outside_examples": [],
        "train_n": {
            "small_all": _counter(),
            "small_distinct": _counter(),
            "min_all": None,
            "min_distinct": None,
        },
        "robust_fit": _counter(),
        "fit_seconds": {"ok": 0.0, "raised": 0.0},
        "level": None,
        "hook_errors": [],
    }
    _CTX = []  # the outermost public method in progress, repeated
    _CORE = []  # nlZ / post / lowfactor
    _SUPPRESS = [False]
    _RUN = {}
    _PYBADS_DIR = [None]
    _BADS = [None]

    def _hook_error(where, e):
        if len(_S["hook_errors"]) < 20:
            _S["hook_errors"].append(f"{where}: {type(e).__name__}: {e}")

    def _pybads_dir():
        if _PYBADS_DIR[0] is None:
            mod = sys.modules.get("pybads")
            if mod is not None and getattr(mod, "__file__", None):
                _PYBADS_DIR[0] = os.path.dirname(mod.__file__) + os.sep
        return _PYBADS_DIR[0]

    def _find_run_and_bads(frame):
        """The population task's label and seed, and the BADS object, from
        the frames of the stack (read once each)."""
        pdir = _pybads_dir()
        f = frame
        while f is not None and (not _RUN or _BADS[0] is None):
            code = f.f_code
            if (
                not _RUN
                and code.co_name == "run_task"
                and code.co_filename.endswith("population.py")
            ):
                loc = f.f_locals
                _RUN["label"] = loc.get("label")
                _RUN["seed"] = loc.get("seed")
            if (
                _BADS[0] is None
                and pdir is not None
                and code.co_filename.startswith(pdir)
            ):
                obj = f.f_locals.get("self")
                if type(obj).__name__ == "BADS":
                    _BADS[0] = obj
            f = f.f_back

    def _callers(frame, depth=4):
        pdir = _pybads_dir()
        names = []
        f = frame
        while f is not None and len(names) < depth:
            if pdir is not None and f.f_code.co_filename.startswith(pdir):
                names.append(f.f_code.co_name)
            f = f.f_back
        return "<".join(names) if names else "-"

    def _level():
        b = _BADS[0]
        if b is None:
            return None
        st = getattr(b, "optim_state", None)
        if not isinstance(st, dict):
            return None
        lv = st.get("uncertainty_handling_level")
        if lv is not None:
            _S["level"] = int(lv)
        return lv

    def _mesh():
        b = _BADS[0]
        return None if b is None else getattr(b, "mesh_size", None)

    def _ctx_label():
        if _CTX:
            return _CTX[0]
        return "direct:" + _callers(sys._getframe(2), depth=1)

    def _mults(gp):
        out = []
        posteriors = getattr(gp, "posteriors", None)
        # An array of several posteriors has no truth value
        for p in [] if posteriors is None else posteriors:
            m = getattr(p, "sn2_mult", None)
            out.append(1.0 if m is None else float(m))
        return out

    def _blocks(gp):
        D = gp.D
        cov_N = gp.covariance.hyperparameter_count(D)
        noise_N = gp.noise.hyperparameter_count()
        mean_N = gp.mean.hyperparameter_count(D)
        return cov_N, noise_N, mean_N

    def _block_of(i, cov_N, noise_N):
        if i < cov_N:
            return "cov"
        if i < cov_N + noise_N:
            return "noise"
        return "mean"

    def _prior_state(gp):
        """Hyperparameters whose Gaussian (or Student's t) prior centre
        lies outside the bounds, and those with a zero normalization
        constant."""
        with np.errstate(all="ignore"):
            return _prior_state_items(gp)

    def _prior_state_items(gp):
        hp = gp.hyper_priors
        mu = np.asarray(hp["mu"], dtype=float)
        sigma = np.abs(np.asarray(hp["sigma"], dtype=float))
        lb = np.asarray(gp.lower_bounds, dtype=float)
        ub = np.asarray(gp.upper_bounds, dtype=float)
        has = np.isfinite(mu) & np.isfinite(sigma)
        outside = has & ((mu < lb) | (mu > ub))
        nc = getattr(gp, "normalization_constants", None)
        zero = (
            np.zeros(mu.shape, bool)
            if nc is None
            else ~(np.asarray(nc, dtype=float) > 0)
        )
        items = []
        cov_N, noise_N, _ = _blocks(gp)
        for i in np.flatnonzero(outside | zero):
            gap = (
                (lb[i] - mu[i]) / sigma[i]
                if mu[i] < lb[i]
                else (mu[i] - ub[i]) / sigma[i]
            )
            items.append(
                {
                    "i": int(i),
                    "block": _block_of(i, cov_N, noise_N),
                    "mu": float(mu[i]),
                    "sigma": float(sigma[i]),
                    "lb": float(lb[i]),
                    "ub": float(ub[i]),
                    "gap_sigmas": float(gap) if outside[i] else None,
                    "zero_norm": bool(zero[i]),
                }
            )
        return items

    def _train_n(gp):
        X = getattr(gp, "X", None)
        if X is None:
            return
        n = int(X.shape[0])
        nd = int(np.unique(X, axis=0).shape[0]) if n else 0
        t = _S["train_n"]
        if n <= 3:
            t["small_all"][str(n)] += 1
        if nd <= 3:
            t["small_distinct"][str(nd)] += 1
        t["min_all"] = n if t["min_all"] is None else min(t["min_all"], n)
        t["min_distinct"] = (
            nd if t["min_distinct"] is None else min(t["min_distinct"], nd)
        )

    def _after_training(name, gp, caller):
        with np.errstate(all="ignore"):
            _after_training_counts(name, gp, caller)

    def _after_training_counts(name, gp, caller):
        m = _mults(gp)
        rec = _S["posteriors"][f"{name}|{caller}"]
        rec["returns"] += 1
        if m and max(m) > 1:
            rec["inflated"] += 1
            rec["max_mult"] = max(rec["max_mult"], max(m))
            if len(_S["posterior_examples"]) < _MAX_EXAMPLES:
                p = gp.posteriors[int(np.argmax(m))]
                cov_N, noise_N, _ = _blocks(gp)
                hyp = np.asarray(p.hyp, dtype=float)
                sn2 = gp.noise.compute(
                    hyp[cov_N : cov_N + noise_N], gp.X, gp.y, gp.s2
                )
                sf2 = gp.covariance.compute(
                    hyp[:cov_N], gp.X[:1], compute_diag=True
                )[0, 0]
                _S["posterior_examples"].append(
                    {
                        "method": name,
                        "caller": caller,
                        "n": int(gp.X.shape[0]),
                        "mult": max(m),
                        "sn2_min": float(np.min(sn2)),
                        "sf2": float(sf2),
                        "L_chol": bool(getattr(p, "L_chol", True)),
                        "level": _level(),
                        "mesh": _mesh(),
                    }
                )
        _train_n(gp)
        if name == "fit":
            items = _prior_state(gp)
            blocks = collections.Counter(it["block"] for it in items)
            for blk in ("cov", "noise", "mean"):
                r = _S["priors_outside"][blk]
                r["fits"] += 1
                if any(
                    it["block"] == blk and it["gap_sigmas"] is not None
                    for it in items
                ):
                    r["outside"] += 1
                if any(it["block"] == blk and it["zero_norm"] for it in items):
                    r["zero_norm"] += 1
            if items and len(_S["priors_outside_examples"]) < _MAX_EXAMPLES:
                _S["priors_outside_examples"].append(
                    {"caller": caller, "blocks": dict(blocks), "items": items}
                )

    _PREDICT_SIG = inspect.signature(GP.predict)

    def _zero_sd(gp, x_star, idx):
        z = _S["zero_sd"]
        p = gp.posteriors[0]
        cov_N, noise_N, _ = _blocks(gp)
        hyp = np.asarray(p.hyp, dtype=float)
        xz = np.atleast_2d(np.asarray(x_star, dtype=float))[idx]
        with np.errstate(all="ignore"):
            kss = gp.covariance.compute(hyp[:cov_N], xz, compute_diag=True)[
                :, 0
            ]
            Ks = gp.covariance.compute(hyp[:cov_N], gp.X, xz)
            if p.L_chol:
                V = _gpmod._solve_triangular(p.L, p.sW * Ks, trans=1)
            else:
                z["low_noise_repr"] += len(idx)
                _SUPPRESS[0] = True
                try:
                    F = gp._GP__low_noise_factor(0)
                finally:
                    _SUPPRESS[0] = False
                V = _gpmod._solve_triangular(F, Ks, trans=1)
            raw = kss - np.sum(V * V, 0)
            sn2 = gp.noise.compute(
                hyp[cov_N : cov_N + noise_N], gp.X, gp.y, gp.s2
            )
            mult = 1.0 if p.sn2_mult is None else float(p.sn2_mult)
            sn2_eff = float(np.min(sn2)) * mult
            ell = np.exp(hyp[: gp.D])
            d = np.sqrt(
                np.min(
                    np.sum(
                        ((xz[:, None, :] - gp.X[None, :, :]) / ell) ** 2,
                        axis=2,
                    ),
                    axis=1,
                )
            )
        for k in range(len(idx)):
            z["points"] += 1
            r = float(raw[k])
            if r < 0:
                z["raw_negative"] += 1
            elif r == 0:
                z["raw_zero"] += 1
            else:
                z["raw_positive"] += 1
            rel = abs(r) / float(kss[k]) if kss[k] > 0 else float("inf")
            b = "0" if rel == 0 else str(int(math.floor(math.log10(rel))))
            z["raw_rel_log10_hist"][b] += 1
            dk = float(d[k])
            db = (
                "0"
                if dk == 0
                else "<1e-6"
                if dk < 1e-6
                else "<1e-3"
                if dk < 1e-3
                else "<1e-1"
                if dk < 1e-1
                else ">=1e-1"
            )
            z["dist_hist"][db] += 1
            snr = float(kss[k]) / sn2_eff if sn2_eff > 0 else float("inf")
            z["snr_log10_hist"][
                str(int(math.floor(math.log10(snr)))) if snr > 0 else "-inf"
            ] += 1
            if len(z["examples"]) < _MAX_EXAMPLES:
                z["examples"].append(
                    {
                        "raw": r,
                        "kss": float(kss[k]),
                        "rel": rel,
                        "dist_ell": dk,
                        "sn2_eff": sn2_eff,
                        "n_train": int(gp.X.shape[0]),
                        "log_ell": [float(v) for v in hyp[: gp.D]],
                        "L_chol": bool(p.L_chol),
                        "mesh": _mesh(),
                        "level": _level(),
                    }
                )

    def _after_predict(gp, args, kwargs, res, caller):
        bound = _PREDICT_SIG.bind(gp, *args, **kwargs)
        bound.apply_defaults()
        add_noise = bool(bound.arguments["add_noise"])
        s2 = np.asarray(res[1])
        lv = _level()
        rec = _S["predict"][f"{caller}|level={lv}|add_noise={add_noise}"]
        rec["calls"] += 1
        rec["points"] += int(s2.shape[0])
        m = _mults(gp)
        if m and max(m) > 1:
            rec["calls_inflated_post"] += 1
        if bound.arguments["separate_samples"]:
            zero = np.all(s2 == 0, axis=1)
        else:
            zero = s2.ravel() == 0
        nz = int(np.sum(zero))
        if nz:
            rec["zero_points"] += nz
            rec["calls_with_zero"] += 1
            if not add_noise and gp.y is not None:
                _zero_sd(gp, bound.arguments["x_star"], np.flatnonzero(zero))

    def _wrap_public(name):
        orig = GP.__dict__[name]

        def wrapper(self, *args, **kwargs):
            outer = not _CTX
            caller = None
            if outer:
                _S["calls"][name] += 1
                try:
                    fr = sys._getframe(1)
                    _find_run_and_bads(fr)
                    caller = _callers(fr)
                except Exception as e:  # noqa: BLE001
                    _hook_error("callers", e)
            i_try = None
            if outer and name == "fit":
                try:
                    fr = sys._getframe(1)
                    if fr.f_code.co_name == "_robust_gp_fit_":
                        i_try = fr.f_locals.get("i_try")
                except Exception as e:  # noqa: BLE001
                    _hook_error("i_try", e)
            t0 = time.perf_counter()
            _CTX.append(name if outer else _CTX[0])
            try:
                res = orig(self, *args, **kwargs)
            except BaseException as e:
                if outer:
                    _S["exceptions"][f"{name}:{type(e).__name__}"] += 1
                    if name == "fit":
                        _S["fit_seconds"]["raised"] += time.perf_counter() - t0
                        if i_try is not None:
                            _S["robust_fit"][
                                f"try{i_try}:{type(e).__name__}"
                            ] += 1
                raise
            finally:
                _CTX.pop()
            if outer and name == "fit":
                _S["fit_seconds"]["ok"] += time.perf_counter() - t0
                if i_try is not None:
                    _S["robust_fit"][f"try{i_try}:ok"] += 1
            if outer:
                try:
                    if name == "predict":
                        _after_predict(self, args, kwargs, res, caller)
                    else:
                        _after_training(name, self, caller)
                except Exception as e:  # noqa: BLE001
                    _hook_error(name, e)
            return res

        wrapper.__name__ = orig.__name__
        wrapper.__qualname__ = orig.__qualname__
        wrapper.__doc__ = orig.__doc__
        wrapper.__wrapped__ = orig
        setattr(GP, name, wrapper)

    for _name in ("fit", "update", "set_hyperparameters", "predict"):
        _wrap_public(_name)

    _orig_tc = GP.__dict__["_GP__training_cholesky"].__func__

    def _training_cholesky(K, sn2, L_chol, sn2_mult=1, *args, **kwargs):
        if _SUPPRESS[0]:
            return _orig_tc(K, sn2, L_chol, sn2_mult, *args, **kwargs)
        try:
            key = (
                f"{_ctx_label()}|{_CORE[-1] if _CORE else 'other'}"
                f"|L_chol={bool(L_chol)}"
            )
        except Exception as e:  # noqa: BLE001
            _hook_error("chol key", e)
            key = "?"
        rec = _S["chol"][key]
        rec["calls"] += 1
        try:
            L, sl, m = _orig_tc(K, sn2, L_chol, sn2_mult, *args, **kwargs)
        except scipy.linalg.LinAlgError:
            rec["raised"] += 1
            rec["fails_hist"][10] += 1
            raise
        fails = int(round(math.log10(m / sn2_mult))) if m != sn2_mult else 0
        if fails:
            rec["inflated"] += 1
        rec["fails_hist"][min(fails, 10)] += 1
        return L, sl, m

    GP._GP__training_cholesky = staticmethod(_training_cholesky)

    _orig_core = GP.__dict__["_GP__core_computation"]

    def _core_computation(self, hyp, compute_nlZ, *args, **kwargs):
        _CORE.append("nlZ" if compute_nlZ else "post")
        try:
            return _orig_core(self, hyp, compute_nlZ, *args, **kwargs)
        finally:
            _CORE.pop()

    GP._GP__core_computation = _core_computation

    _orig_lnf = GP.__dict__["_GP__low_noise_factor"]

    def _low_noise_factor(self, s):
        _CORE.append("lowfactor")
        try:
            return _orig_lnf(self, s)
        finally:
            _CORE.pop()

    GP._GP__low_noise_factor = _low_noise_factor

    _orig_lp = GP.__dict__["_GP__compute_log_priors"]

    def _compute_log_priors(self, hyp, compute_grad):
        res = _orig_lp(self, hyp, compute_grad)
        try:
            lp = res[0] if isinstance(res, tuple) else res
            key = _ctx_label()
            rec = _S["log_prior"][key]
            rec["calls"] += 1
            lp = float(np.asarray(lp).ravel()[0])
            if math.isnan(lp):
                rec["nan"] += 1
                if len(_S["log_prior_nan_examples"]) < _MAX_EXAMPLES:
                    _S["log_prior_nan_examples"].append(
                        {"ctx": key, "items": _prior_state(self)}
                    )
            elif lp == -math.inf:
                rec["neginf"] += 1
        except Exception as e:  # noqa: BLE001
            _hook_error("log_prior", e)
        return res

    GP._GP__compute_log_priors = _compute_log_priors

    _orig_obj = GP.__dict__["_GP__gp_obj_fun"]

    def _gp_obj_fun(self, hyp, compute_grad, swap_sign, *args, **kwargs):
        res = _orig_obj(self, hyp, compute_grad, swap_sign, *args, **kwargs)
        try:
            v = res[0] if isinstance(res, tuple) else res
            v = float(np.asarray(v).ravel()[0])
            rec = _S["obj_fun"][_ctx_label()]
            rec["calls"] += 1
            if math.isnan(v):
                rec["nan"] += 1
            elif math.isinf(v):
                rec["neginf_or_inf"] += 1
        except Exception as e:  # noqa: BLE001
            _hook_error("obj_fun", e)
        return res

    GP._GP__gp_obj_fun = _gp_obj_fun

    def _plain(x):
        if isinstance(x, dict):
            return {str(k): _plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_plain(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    @atexit.register
    def _write():
        if not _RUN:
            return
        os.makedirs(_OUT, exist_ok=True)
        path = os.path.join(_OUT, f"{_RUN['label']}_seed{_RUN['seed']}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"label": _RUN["label"], "seed": _RUN["seed"], **_plain(_S)},
                fh,
                indent=1,
            )
