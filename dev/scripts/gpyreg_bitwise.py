"""Compare two versions of gpyreg bit for bit, on its kernels and on whole
Gaussian processes: the gate of a change to gpyreg that must move nothing,
beside PyBADS's own (``fingerprint.py``, ``replay.py`` and the oracles'
``--against``), whose runs reach only the paths that PyBADS takes.

``dump OUT`` computes, with the gpyreg that the process imports (from
``PYTHONPATH``, or the installed one) and one BLAS thread:

* the kernels (squared exponential, Matern of degrees 1, 3 and 5, rational
  quadratic ARD), with and without periods, at 1 to 10 dimensions and 1 to
  150 inputs: the kernel, its gradient, its cross-covariance and its
  diagonal, for random hyperparameters, a rational-quadratic shape of 1,
  repeated points, Fortran-ordered, strided, float32 and long-double inputs
  and an infinite coordinate;
* Gaussian processes of each kernel (and the isotropic ones), with a
  constant or a negative quadratic mean and with or without a noise per
  point: priors, a fit, the priors' normalization constants, predictions
  (with the noise, the log predictive density, separate samples and
  cross-covariances), ``predict_full``, the objective and its gradient
  (the low-noise representation included), a posterior in the low-noise
  representation and its predictions, a single-point and a full update,
  ``random_function``, a GP whose factorization needs a noise multiplier,
  and a GP that refuses a failed factorization;

and writes to the pickle ``OUT`` a SHA-256 digest of each output, of the
type, shape and bytes of every array in it (or of the exception it
raised, with its message), with the platform key
(``harness.platform_key``) and the gpyreg that computed them. ``compare A
B`` compares two dumps output by output and names those that differ. It
refuses two dumps whose platform keys differ, since a platform can change
the last bits, and exits 1 unless every output is identical.

Run each version in a process of its own, from the repository root::

    PYTHONPATH=<gpyreg at the parent> python dev/scripts/gpyreg_bitwise.py \\
        dump dev/scripts/runs/gpyreg_parent.pkl
    PYTHONPATH=<gpyreg at the change> python dev/scripts/gpyreg_bitwise.py \\
        dump dev/scripts/runs/gpyreg_change.pkl
    python dev/scripts/gpyreg_bitwise.py compare \\
        dev/scripts/runs/gpyreg_parent.pkl dev/scripts/runs/gpyreg_change.pkl

A dump takes about two minutes. The inputs are drawn from fixed seeds, so
that two dumps hold the same computations in the same order.
"""

import argparse
import hashlib
import itertools
import pickle
import sys
import warnings

import harness

harness.single_thread_env()

import numpy as np  # noqa: E402


def _record(out, key, fn):
    """Append ``(key, digest)`` for the output of ``fn()``, or for the
    exception it raises, and return the output."""
    try:
        value = fn()
    except Exception as e:  # noqa: BLE001 (an exception is an output)
        value = ("raised", type(e).__name__, str(e))
    digest = hashlib.sha256(pickle.dumps(_canonical(value), protocol=4))
    out.append((key, digest.hexdigest(), _raised(value)))
    return value


def _raised(value):
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], str)
        and value[0] == "raised"
    )


def _kernels(cf, D, periodic):
    periods = None
    if periodic:
        periods = np.full(D, np.inf)
        periods[0] = 2.5
    return {
        "SE": cf.SquaredExponential(periods=periods),
        "Matern1": cf.Matern(1, periods=periods),
        "Matern3": cf.Matern(3, periods=periods),
        "Matern5": cf.Matern(5, periods=periods),
        "RQ": cf.RationalQuadraticARD(periods=periods),
    }


def dump_kernels(out):
    import gpyreg.covariance_functions as cf

    rng = np.random.default_rng(0)
    for D, N in itertools.product([1, 2, 3, 6, 10], [1, 2, 7, 54, 150]):
        for variant in (
            "plain",
            "repeated",
            "fortran",
            "strided",
            "float32",
            "longdouble",
            "infinite",
        ):
            X = rng.normal(size=(N, D)) * rng.uniform(0.1, 5)
            if variant == "repeated" and N > 3:
                X[1] = X[0]
                X[2, 0] = X[0, 0]
            elif variant == "fortran":
                X = np.asfortranarray(X)
            elif variant == "strided":
                X = np.repeat(X, 2, axis=1)[:, ::2]
            elif variant in ("float32", "longdouble"):
                X = X.astype(variant)
            elif variant == "infinite":
                X[-1, -1] = np.inf
            X_star = rng.normal(size=(int(rng.integers(1, 40)), D))
            for periodic in (False, True):
                for name, kernel in _kernels(cf, D, periodic).items():
                    cov_N = kernel.hyperparameter_count(D)
                    for h in range(3):
                        hyp = rng.normal(size=cov_N)
                        if name == "RQ" and h == 0:
                            hyp[-1] = 0.0
                        for args in (
                            {},
                            {"compute_grad": True},
                            {"X_star": X_star},
                            {"compute_diag": True},
                            {"X_star": X_star, "compute_diag": True},
                        ):
                            key = (
                                "kernel",
                                D,
                                N,
                                variant,
                                periodic,
                                name,
                                h,
                                tuple(sorted(args)),
                            )
                            _record(
                                out,
                                key,
                                lambda: kernel.compute(hyp.copy(), X, **args),
                            )


def dump_gps(out):
    import gpyreg
    import gpyreg.covariance_functions as cf
    import gpyreg.isotropic_covariance_functions as icf
    import gpyreg.mean_functions as mf
    import gpyreg.noise_functions as nf

    case = 0
    for D, N in itertools.product((1, 3, 6), (6, 40, 120)):
        rng = np.random.default_rng(1000 * D + N)
        X = rng.normal(size=(N, D))
        if N > 10:
            X[3] = X[2]
        y = np.sin(X.sum(1, keepdims=True)) + 0.1 * rng.normal(size=(N, 1))
        s2 = rng.uniform(0.01, 0.1, size=(N, 1))
        X_star = rng.normal(size=(50, D))
        kernels = dict(_kernels(cf, D, False))
        kernels.update(
            {
                f"{k}p": v
                for k, v in _kernels(cf, D, True).items()
                if k in ("SE", "Matern5", "RQ")
            }
        )
        kernels["SEiso"] = icf.SquaredExponentialIsotropic()
        kernels["Materniso"] = icf.MaternIsotropic(3)
        for kname, cov in kernels.items():
            for variant in ("const", "user_s2", "negquad"):
                case += 1
                key = ("gp", D, N, kname, variant)
                gp = gpyreg.GP(
                    D=D,
                    covariance=cov,
                    mean=(
                        mf.NegativeQuadratic()
                        if variant == "negquad"
                        else mf.ConstantMean()
                    ),
                    noise=nf.GaussianNoise(
                        constant_add=True,
                        user_provided_add=variant == "user_s2",
                    ),
                )
                priors = {}
                for name, n in (
                    gp.covariance.hyperparameter_info(D)
                    + gp.noise.hyperparameter_info()
                    + gp.mean.hyperparameter_info(D)
                ):
                    priors[name] = None
                    if rng.random() < 0.8:
                        priors[name] = (
                            "gaussian",
                            (
                                rng.normal(size=n) * 2,
                                rng.uniform(0.3, 3, size=n),
                            ),
                        )
                _record(out, key + ("priors",), lambda: gp.set_priors(priors))
                n_samples = 3 if (kname == "RQ" and N == 40) else 0
                fitted = _record(
                    out,
                    key + ("fit",),
                    lambda: gp.fit(
                        X,
                        y,
                        s2 if variant == "user_s2" else None,
                        options={
                            "n_samples": n_samples,
                            "opts_N": 2,
                            "init_N": 64,
                        },
                        rng=np.random.default_rng(case),
                    )[0],
                )
                if _raised(fitted):
                    continue
                _record(
                    out,
                    key + ("masses",),
                    lambda: gp.normalization_constants.copy(),
                )
                _record(out, key + ("predict",), lambda: gp.predict(X_star))
                _record(
                    out,
                    key + ("predict_noise_lpd",),
                    lambda: gp.predict(
                        X_star,
                        y_star=np.zeros((50, 1)),
                        add_noise=True,
                        return_lpd=True,
                    ),
                )
                _record(
                    out,
                    key + ("predict_separate",),
                    lambda: gp.predict(X_star, separate_samples=True),
                )
                _record(
                    out,
                    key + ("predict_cross",),
                    lambda: gp.predict(X_star, return_cross_covariance=True),
                )
                _record(
                    out,
                    key + ("predict_full",),
                    lambda: gp.predict_full(X_star[:10]),
                )
                hyp = gp.get_hyperparameters(as_array=True)[0]
                cov_N = gp.covariance.hyperparameter_count(D)
                for j in range(3):
                    h = hyp + rng.normal(size=hyp.size) * 0.5
                    if j == 1:
                        h[cov_N] = -9.0  # the low-noise representation
                    _record(
                        out,
                        key + ("log_likelihood", j),
                        lambda: gp.log_likelihood(h, compute_grad=True),
                    )
                    _record(
                        out,
                        key + ("log_posterior", j),
                        lambda: gp.log_posterior(h, compute_grad=True),
                    )
                low = hyp.copy()
                low[cov_N] = -9.0
                _record(
                    out,
                    key + ("low_noise",),
                    lambda: gp.update(hyp=low[None, :]),
                )
                _record(
                    out,
                    key + ("predict_low_noise",),
                    lambda: gp.predict(X_star, add_noise=True),
                )
                _record(
                    out,
                    key + ("update_one",),
                    lambda: gp.update(
                        X_new=rng.normal(size=(1, D)),
                        y_new=np.array([[0.3]]),
                        s2_new=(
                            np.array([[0.05]])
                            if variant == "user_s2"
                            else None
                        ),
                    ),
                )
                _record(
                    out,
                    key + ("predict_after_one",),
                    lambda: gp.predict(X_star),
                )
                _record(
                    out,
                    key + ("update_hyp",),
                    lambda: gp.update(hyp=hyp[None, :]),
                )
                _record(
                    out,
                    key + ("predict_after_hyp",),
                    lambda: gp.predict(X_star),
                )
                _record(
                    out,
                    key + ("random_function",),
                    lambda: gp.random_function(
                        X_star[:8], rng=np.random.default_rng(7)
                    ),
                )
        # Repeated inputs, long length scales and a tiny noise: a
        # factorization that needs the noise multiplier, and one that a GP
        # created with raise_on_cholesky_failure refuses.
        for kname in ("SE", "RQ"):
            cov = _kernels(cf, D, False)[kname]
            cov_N = cov.hyperparameter_count(D)
            h = np.zeros(cov_N + 2)
            h[:D] = 8.0
            h[D] = 5.0
            h[cov_N] = -15.0
            X_rep = np.repeat(X[:5], 3, axis=0)
            y_rep = np.repeat(y[:5], 3, axis=0)
            refusing = gpyreg.GP(
                D=D,
                covariance=cov,
                mean=mf.ConstantMean(),
                noise=nf.GaussianNoise(constant_add=True),
                raise_on_cholesky_failure=True,
            )
            key = ("gp", D, N, kname, "ill-conditioned")
            _record(
                out,
                key + ("refusing_fit",),
                lambda: refusing.fit(
                    X_rep,
                    y_rep,
                    options={"n_samples": 0, "opts_N": 1, "init_N": 16},
                    rng=np.random.default_rng(3),
                )[0],
            )
            _record(
                out,
                key + ("refusing_update",),
                lambda: refusing.update(
                    X_new=X_rep, y_new=y_rep, hyp=h[None, :]
                ),
            )
            gp = gpyreg.GP(
                D=D,
                covariance=cov,
                mean=mf.ConstantMean(),
                noise=nf.GaussianNoise(constant_add=True),
            )
            _record(
                out,
                key + ("multiplier",),
                lambda: (
                    gp.update(X_new=X_rep, y_new=y_rep, hyp=h[None, :]),
                    gp.posteriors[0].sn2_mult,
                    gp.predict(X_star),
                    gp.log_likelihood(h, compute_grad=True),
                )[1:],
            )


def dump(path):
    import gpyreg

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    out = []
    dump_kernels(out)
    dump_gps(out)
    meta = {
        "platform_key": harness.platform_key(),
        "gpyreg": harness.module_source("gpyreg"),
    }
    with open(path, "wb") as f:
        pickle.dump({"meta": meta, "records": out}, f)
    raised = sum(r for _, _, r in out)
    print(
        f"[dump] gpyreg {gpyreg.__file__}: {len(out)} outputs "
        f"({raised} exceptions) -> {path}"
    )


# The bytes that hold a long double's value: 10 of its 16 in x87's 80-bit
# format (a 64-bit mantissa, NumPy's long double on x86), all of them
# where it is IEEE's quadruple or double precision.
_LONG_DOUBLE_BYTES = (
    10
    if np.finfo(np.longdouble).nmant == 63
    else np.dtype(np.longdouble).itemsize
)


def _canonical(v):
    """A form of a value that is equal for two values exactly when their
    types, shapes and bytes are."""
    if isinstance(v, (tuple, list)):
        return (type(v).__name__, tuple(_canonical(x) for x in v))
    if isinstance(v, dict):
        return ("dict", tuple((k, _canonical(v[k])) for k in sorted(v)))
    if isinstance(v, np.ndarray) and v.dtype == object:
        return ("objects", v.shape, tuple(_canonical(x) for x in v.ravel()))
    if isinstance(v, (np.ndarray, np.generic, float, int)):
        a = np.ascontiguousarray(v)
        data = a.view(np.uint8)
        if a.dtype == np.longdouble:
            # Only the bytes of the value: the others are padding, which
            # holds whatever the memory held before.
            data = data.reshape(-1, a.dtype.itemsize)[:, :_LONG_DOUBLE_BYTES]
        return ("array", a.dtype.str, a.shape, data.tobytes())
    if v is None or isinstance(v, (str, bool)):
        return v
    if hasattr(v, "__dict__"):  # scipy's OptimizeResult, for instance
        return ("object", type(v).__name__, _canonical(dict(v.__dict__)))
    return repr(v)


def compare(path_a, path_b):
    with open(path_a, "rb") as f:
        a = pickle.load(f)
    with open(path_b, "rb") as f:
        b = pickle.load(f)
    print(f"[compare] {path_a}: gpyreg {a['meta']['gpyreg']}")
    print(f"[compare] {path_b}: gpyreg {b['meta']['gpyreg']}")
    ka, kb = a["meta"]["platform_key"], b["meta"]["platform_key"]
    differ = sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))
    if differ:
        sys.exit(
            f"the dumps were made under different platform keys "
            f"({', '.join(differ)}): dump both on one machine, with "
            f"one setting"
        )
    keys_a = [r[0] for r in a["records"]]
    if keys_a != [r[0] for r in b["records"]]:
        sys.exit(
            "the dumps hold different computations: dump both with "
            "this script"
        )
    different = [
        ra[0] for ra, rb in zip(a["records"], b["records"]) if ra[1] != rb[1]
    ]
    for k in different[:30]:
        print(f"  differs: {k}")
    print(f"[compare] {len(keys_a)} outputs, {len(different)} differ")
    return 1 if different else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dump", help="compute and store every output")
    p.add_argument("out")
    p = sub.add_parser("compare", help="compare two dumps bit for bit")
    p.add_argument("a")
    p.add_argument("b")
    args = parser.parse_args(argv)
    if args.command == "dump":
        dump(args.out)
        return 0
    return compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
