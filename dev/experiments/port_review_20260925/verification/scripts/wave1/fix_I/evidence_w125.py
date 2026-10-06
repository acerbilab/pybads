"""W1-25 evidence: with the switch off (the default), every result on
problems whose factorization needs the noise inflation is bit-identical to
main. Run once with PYTHONPATH at main's code and once at the branch, and
compare the printed hashes."""

import hashlib
import sys

import gpyreg as gpr
import numpy as np
from gpyreg import gaussian_process as gp_module

print("gpyreg.__file__ =", gpr.__file__)
explicit = len(sys.argv) > 1 and sys.argv[1] == "explicit"
kwargs = {"raise_on_cholesky_failure": False} if explicit else {}
print("constructor keywords:", kwargs)

# Count the factorizations and the inflated ones.
counts = {"calls": 0, "inflated": 0, "failed": 0}
real_chol = gp_module.GP._GP__training_cholesky


def counting_chol(*args, **kw):
    counts["calls"] += 1
    try:
        out = real_chol(*args, **kw)
    except Exception:
        counts["failed"] += 1
        raise
    if out[2] > (args[3] if len(args) > 3 else kw.get("sn2_mult", 1)):
        counts["inflated"] += 1
    return out


gp_module.GP._GP__training_cholesky = staticmethod(counting_chol)

designs = []
real_fill = gp_module.f_min_fill


def recording_fill(*a, **k):
    X0, y0 = real_fill(*a, **k)
    designs.append((X0.copy(), y0.copy()))
    return X0, y0


gp_module.f_min_fill = recording_fill

digest = hashlib.sha256()
case_digests = {}


def add(name, *values):
    h = hashlib.sha256()
    for v in values:
        a = np.asarray(v, dtype=float) if not isinstance(v, np.ndarray) else v
        h.update(np.ascontiguousarray(a).tobytes())
        h.update(str(a.shape).encode())
    case_digests[name] = h.hexdigest()[:16]
    digest.update(h.digest())


def posterior_values(gp):
    out = []
    for p in gp.posteriors:
        for f in ("hyp", "alpha", "sW", "L", "sn2_mult", "L_chol", "sl"):
            out.append(np.asarray(getattr(p, f), dtype=float))
        Lf = getattr(p, "L_factor", None)
        out.append(np.zeros(0) if Lf is None else Lf)
    return out


def gp_1d():
    return gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        **kwargs,
    )


x_star = np.reshape(np.linspace(-1.5, 1.5, 9), (-1, 1))
rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, size=(20, 1))
y = np.sin(3 * X)
for noise, hyp in (
    ("high", np.array([[0.0, 12.0, np.log(2e-3), 0.0]])),
    ("low", np.array([[0.0, 4.0, np.log(1e-8), 0.0]])),
    (
        "high2",
        np.array(
            [[0.0, 12.0, np.log(2e-3), 0.0], [0.3, 14.0, np.log(3e-3), 0.1]]
        ),
    ),
):
    gp = gp_1d()
    gp.update(X_new=X, y_new=y, hyp=hyp)
    mults = [p.sn2_mult for p in gp.posteriors]
    print(
        f"update {noise}: sn2_mult {mults}, L_chol {[p.L_chol for p in gp.posteriors]}"
    )
    pred = gp.predict(x_star)
    pred_n = gp.predict(x_star, add_noise=True, separate_samples=True)
    full = gp.predict_full(x_star, add_noise=True)
    ll = gp.log_likelihood(hyp[0], compute_grad=True)
    lp = gp.log_posterior(hyp[0], compute_grad=True)
    rf = gp.random_function(
        x_star, add_noise=True, rng=np.random.default_rng(3)
    )
    q = gp.quad(np.zeros((1, 1)), np.ones((1, 1)), compute_var=True)
    add(
        f"update_{noise}",
        *posterior_values(gp),
        *pred,
        *pred_n,
        *full,
        ll[0],
        ll[1],
        lp[0],
        lp[1],
        rf,
        *q,
    )
    # A single-point update of the inflated posterior (sn2_eff).
    gp.update(X_new=np.array([[0.123]]), y_new=np.array([[np.sin(0.369)]]))
    add(f"rank_one_{noise}", *posterior_values(gp), *gp.predict(x_star))

# A fit whose design holds starting points that need the inflation.
rng = np.random.default_rng(1)
Xf = rng.uniform(-1, 1, size=(20, 1))
yf = np.sin(3 * Xf) + 0.1 * rng.standard_normal((20, 1))
hyp0 = np.array(
    [
        [0.0, 12.0, np.log(2e-3), 0.0],
        [0.0, 0.0, np.log(0.1), 0.0],
        [0.5, 14.0, np.log(2e-3), 0.5],
    ]
)
bounds = {
    "lower_bounds": np.array([-2.0, -2.0, np.log(2e-3), -5.0]),
    "upper_bounds": np.array([2.0, 14.0, 0.0, 5.0]),
}
for init_N, opts_N, n_samples in ((16, 2, 0), (0, 2, 0), (64, 2, 4)):
    gp = gp_1d()
    designs.clear()
    before = dict(counts)
    hyp, res, samp = gp.fit(
        Xf,
        yf,
        hyp0=hyp0,
        options=dict(
            bounds, n_samples=n_samples, init_N=init_N, opts_N=opts_N
        ),
        rng=np.random.default_rng(0),
    )
    extra = []
    if designs:
        extra = list(designs[0])
    if samp is not None:
        extra += (
            [samp["samples"], samp["log_likelihoods"]]
            if "log_likelihoods" in samp
            else [samp["samples"]]
        )
    add(
        f"fit_{init_N}_{opts_N}_{n_samples}",
        hyp,
        res.x,
        res.fun,
        res.nit,
        *posterior_values(gp),
        *gp.predict(x_star),
        *extra,
    )
    print(
        f"fit init_N={init_N} opts_N={opts_N} n_samples={n_samples}: inflated factorizations {counts['inflated'] - before['inflated']} of {counts['calls'] - before['calls']}; design inflated values: {None if not designs else int(np.sum(designs[0][1] > 1e3))}"
    )

# A Rosenbrock-like dense fit in 2-D, as PyBADS builds it (RQ ARD).
rng = np.random.default_rng(5)
X2 = 1.0 + 0.05 * rng.standard_normal((45, 2))
y2 = (100 * (X2[:, 1] - X2[:, 0] ** 2) ** 2 + (1 - X2[:, 0]) ** 2)[
    :, None
] + 1e3
gp = gpr.GP(
    D=2,
    covariance=gpr.covariance_functions.RationalQuadraticARD(),
    mean=gpr.mean_functions.ConstantMean(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    **kwargs,
)
gp.set_bounds(
    {
        "covariance_log_lengthscale": (
            np.full(2, np.log(1e-6)),
            np.full(2, np.log(20.0)),
        ),
        "covariance_log_outputscale": (np.log(1e-3), 20.0),
        "covariance_log_shape": (np.array([-5.0]), np.array([5.0])),
        "noise_log_scale": (np.log(1e-3) - 1, 5.0),
        "mean_const": (np.array([-1e4]), np.array([1e4])),
    }
)
before = dict(counts)
hyp, res, _ = gp.fit(
    X2,
    y2,
    options={"n_samples": 0, "init_N": 256, "opts_N": 2},
    rng=np.random.default_rng(2),
)
add(
    "fit_rosenbrock",
    hyp,
    res.x,
    res.fun,
    res.nit,
    *posterior_values(gp),
    *gp.predict(X2[:10]),
)
print(
    f"fit rosenbrock: inflated factorizations {counts['inflated'] - before['inflated']} of {counts['calls'] - before['calls']}, failed {counts['failed'] - before['failed']}; sn2_mult {[p.sn2_mult for p in gp.posteriors]}"
)

for k, v in case_digests.items():
    print(f"  {k:24s} {v}")
print("TOTAL", digest.hexdigest())
print("counts", counts)
