"""W1-24 evidence: where the linear mass does not underflow to zero, every
result is bit-identical to main (normalization constants, log prior and
its gradient, the space-filling design, fits); where it does (z >= ~37.7),
the log prior and the design become finite. Run with PYTHONPATH at main's
code and at the branch, and compare."""

import hashlib
import warnings

import gpyreg as gpr
import numpy as np
from gpyreg import gaussian_process as gp_module
from gpyreg.f_min_fill import f_min_fill

print("gpyreg.__file__ =", gpr.__file__)
warnings.simplefilter("ignore")
digest = hashlib.sha256()
n_items = 0


def add(*values):
    global n_items
    for v in values:
        a = np.asarray(v, dtype=float)
        digest.update(np.ascontiguousarray(a).tobytes())
        digest.update(str(a.shape).encode())
        n_items += 1


NAMES = [
    "covariance_log_lengthscale",
    "covariance_log_outputscale",
    "noise_log_scale",
    "mean_const",
]
X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))


def one_d(prior, lower, upper, m0):
    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    hyp = np.array([0.0, 0.0, np.log(0.1), m0])
    gp.update(X_new=X, y_new=np.sin(X) + m0, hyp=hyp[None, :])
    b = {n: (-np.inf, np.inf) for n in NAMES}
    b["mean_const"] = (lower, upper)
    gp.set_bounds(b)
    p = {n: None for n in NAMES}
    p["mean_const"] = prior
    gp.set_priors(p)
    lp = gp.log_posterior(hyp)
    lpg = gp.log_posterior(hyp, compute_grad=True)
    return gp, lp, lpg


def design(hp, lower, upper, n=64):
    LB, UB = np.array([lower]), np.array([upper])
    Xd, yd = f_min_fill(
        lambda x: x[0],
        np.array([[0.5 * (lower + upper)]]),
        LB,
        UB,
        LB,
        UB,
        hp,
        n,
        rng=np.random.default_rng(0),
    )
    return Xd, yd


def hprior(kind, params):
    nan = np.nan
    if kind == "gaussian":
        mu, s = params
        return dict(mu=[mu], sigma=[s], df=[0.0], a=[nan], b=[nan])
    if kind == "student_t":
        mu, s, df = params
        return dict(mu=[mu], sigma=[s], df=[df], a=[nan], b=[nan])
    if kind == "smoothbox":
        a, b, s = params
        return dict(mu=[nan], sigma=[s], df=[0.0], a=[a], b=[b])
    a, b, s, df = params
    return dict(mu=[nan], sigma=[s], df=[df], a=[a], b=[b])


# 1. A Gaussian prior with bounds from its centre to 37.6 of its scales
# away (the subnormal masses included), above and below.
for z in (
    0.0,
    0.5,
    1.0,
    3.0,
    5.0,
    8.3,
    10.0,
    20.0,
    30.0,
    35.0,
    37.0,
    37.5,
    37.6,
):
    for side in (1, -1):
        lower, upper = (z, z + 10.0) if side > 0 else (-z - 10.0, -z)
        m0 = lower + 0.5 if side > 0 else upper - 0.5
        gp, lp, lpg = one_d(("gaussian", (0.0, 1.0)), lower, upper, m0)
        Xd, yd = design(
            {
                k: np.array(v)
                for k, v in hprior("gaussian", (0.0, 1.0)).items()
            },
            lower,
            upper,
        )
        add(gp.normalization_constants, lp, lpg[0], lpg[1], Xd, yd)
# The other families and a few more bounds, as in the existing tests.
for kind, params, bounds in (
    ("student_t", (0.0, 1.0, 3.0), (1e6, 2e6)),
    ("student_t", (0.3, 1.2, 3.0), (-5.0, -1.0)),
    ("smoothbox", (-0.5, 0.5, 1.0), (9.5, 10.5)),
    ("smoothbox", (-1.0, 1.6, 1.2), (-5.0, -1.0)),
    ("smoothbox_student_t", (-0.5, 0.5, 1.0, 3.0), (1e6, 2e6)),
    ("gaussian", (0.3, 1.2), (0.3, np.inf)),
    ("gaussian", (0.3, 1.2), (-np.inf, 0.5)),
    ("gaussian", (0.0, 0.2), (15.0, 115.0)),  # z = 75 (changes)
):
    m0 = bounds[0] + 0.5 if np.isfinite(bounds[0]) else bounds[1] - 0.5
    gp, lp, lpg = one_d((kind, params), bounds[0], bounds[1], m0)
    if kind == "gaussian" and bounds[0] == 15.0:
        print(
            f"changed case gaussian z=75: mass {gp.normalization_constants[3]:.3e}, log posterior {lp:.12g}"
        )
        continue
    Xd, yd = design(
        {k: np.array(v) for k, v in hprior(kind, params).items()}, *bounds
    )
    add(gp.normalization_constants, lp, lpg[0], lpg[1], Xd, yd)

# 2. Fits with priors on every hyperparameter, as PyBADS and PyVBMC set
# them, one with the mean prior 25 of its scales below its bounds.
rng = np.random.default_rng(0)
D = 2
X2 = rng.uniform(-1, 1, (30, D))
y2 = np.sum(X2**2, axis=1, keepdims=True) * 5 + 0.05 * rng.standard_normal(
    (30, 1)
)
for mean_lb, prior_mu, prior_sd, n_samples in (
    (0.0, 0.5, 1.0, 0),
    (5.0, 0.0, 0.2, 0),
    (0.0, 0.5, 1.0, 4),
):
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.RationalQuadraticARD(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    b = gp.get_bounds()
    b["covariance_log_lengthscale"] = (
        np.full(D, np.log(1e-6)),
        np.full(D, np.log(20.0)),
    )
    b["covariance_log_outputscale"] = (np.log(1e-3), np.log(1e6 * 1e-3 / 1e-6))
    b["covariance_log_shape"] = (np.array([-5.0]), np.array([5.0]))
    b["noise_log_scale"] = (np.log(1e-3) - 1, 5)
    b["mean_const"] = (np.array([mean_lb]), np.array([mean_lb + 100.0]))
    gp.set_bounds(b)
    p = gp.get_priors()
    p["covariance_log_lengthscale"] = ("gaussian", (-1.0, 2.0))
    p["covariance_log_outputscale"] = ("gaussian", (np.log(np.std(y2)), 2.0))
    p["covariance_log_shape"] = ("gaussian", (1.0, 1.0))
    p["noise_log_scale"] = ("student_t", (np.log(np.sqrt(1e-3)), 1.0, 3.0))
    p["mean_const"] = ("gaussian", (prior_mu, prior_sd))
    gp.set_priors(p)
    hyp, res, samp = gp.fit(
        X2,
        y2,
        options={"n_samples": n_samples, "init_N": 128, "opts_N": 2},
        rng=np.random.default_rng(1),
    )
    add(
        gp.normalization_constants,
        hyp,
        res.x,
        res.fun,
        res.nit,
        gp.log_posterior(hyp[0]),
        *gp.predict(X2[:5]),
    )
    if samp is not None:
        add(samp["samples"])
print("identical-group items", n_items)
print("TOTAL", digest.hexdigest())

# 3. The cases that change: z = 38 and 60, and the verifier's fit (z = 75).
for z in (38.0, 60.0):
    gp, lp, lpg = one_d(("gaussian", (0.0, 1.0)), z, z + 10.0, z + 0.5)
    Xd, __ = design(
        {k: np.array(v) for k, v in hprior("gaussian", (0.0, 1.0)).items()},
        z,
        z + 10.0,
    )
    print(
        f"changed case z={z}: mass {gp.normalization_constants[3]:.3e}, log posterior {lp:.12g}, "
        f"gradient finite {np.all(np.isfinite(lpg[1]))}, design finite {np.all(np.isfinite(Xd))}, "
        f"design in bounds {np.all((Xd >= z) & (Xd <= z + 10))}"
    )
