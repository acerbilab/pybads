"""Benchmark targets for PyBADS developer tooling.

One place that defines the benchmark problems (objective, minimum, bounds,
noise, BADS options) and the named suites built from them, so that
``population.py`` and any other tool run the same problems.

Synthetic targets (``make_problem(name, D)``); all but ``sphere_nonbox``
are shifted so that their minimum lies at a point ``c`` drawn uniformly in
the central half of the plausible box, with ``z = x - c``:

``sphere``         ``sum(z**2)``
``ellipsoid``      ``sum(a_i * z_i**2)``, axis-aligned, with ``a_i`` spaced
                   geometrically from 1 to ``ELLIPSOID_CONDITION``
``rosenbrock``     Rosenbrock's function of ``Q z + 1``, with ``Q`` a random
                   rotation
``ackley``         Ackley's function of ``z`` (a = 20, b = 0.2, c = 2 pi)
``rastrigin``      Rastrigin's function of ``z`` (A = 10)
``sphere_nonbox``  ``sum(x**2)`` with the non-box constraint of MATLAB
                   BADS's ``private/runtest.m``: points with
                   ``x_1 + x_2 < sqrt(2)`` are infeasible, so the minimum is
                   1, at ``(sqrt(2)/2, sqrt(2)/2, 0, ...)``; with the
                   bounds of ``get_test_opt_conf`` in the PyBADS tests, the
                   problem of their ``test_sphere_opt``

Every synthetic minimum is 0 except that of ``sphere_nonbox``. The shifted
targets share the hard bounds ``[-20, 20]`` and the plausible box
``[-5, 5]`` in each variable; a configuration with ``unbounded=True``
replaces the hard bounds by infinities and keeps the plausible box (BADS
accepts either all bounds finite or all infinite).

Real-data targets, each defined at one dimension: negative
log-likelihoods of two models of the 2020 noisy-VBMC paper, fitted by
maximum likelihood (no prior), within the paper's hard and plausible
bounds:

``timing``           Bayesian time-interval reproduction (Acerbi, Wolpert &
                     Vijayakumar 2012), D = 5, 1512 trials of one subject;
                     the observer's response distribution is integrated
                     numerically, about 40 ms per evaluation
``multisensory_s1``  visuo-vestibular causal inference (Acerbi, Dokka,
                     Angelaki & Ma 2018), D = 6, the 1069 trials of
                     subject 1; analytic, under 1 ms per evaluation

Their data are the archives under ``data/`` (layout and provenance in
``data/README.md``). The likelihoods, their constants and their pinned
values are PyVBMC's (``dev/scripts/benchmark_targets.py`` there), ports of
the lab's benchflow implementations. Their minimum is not analytic:
``f_min`` and ``x_min`` are a reference minimum, computed once by
``make_reference_optima.py`` and read from ``data/reference_optima.json``,
without which ``make_problem`` raises. The reference minimum of ``timing``
lies outside the plausible box, with the lapse rate on its lower hard
bound.

Noise (``Config.noise``): ``"none"``, left to BADS's own test of the start
point (``uncertainty_handling`` stays at its default); ``"homo"``, Gaussian
noise of standard deviation ``HOMO_SD``, with ``uncertainty_handling=True``;
``"hetero"``, Gaussian noise whose standard deviation grows with the
distance to the minimum in function value, ``hetero_sd(f - f_min)``, which
the target returns as its second output, with ``uncertainty_handling=True``
and ``specify_target_noise=True``.

Random streams: ``STRUCTURE_SEED`` fixes each target's shift and rotation
per ``(name, D)``, the same in every run. Per run, ``SeedSequence(seed)``
spawns two streams: the first draws the start point uniformly in the
plausible box (again until it satisfies a non-box constraint), the second
the target's noise; BADS gets ``random_seed=seed``. Two populations run with
the same seeds therefore share each seed's start point and noise stream.

A configuration's ``budget`` is its ``max_fun_evals`` as a multiple of
``D``. The ``default`` suite uses BADS's own default, 500 D: every run ends
on BADS's termination criteria, long before the budget, so that the runs
cover the whole algorithm, from the initial design to the fine mesh and the
stopping rules. At 30 seeds the suite runs in about 80 minutes as one
process with a fresh process per run (``population.py run``).

Command line (from the repository root)::

    python dev/scripts/benchmark_targets.py --list
    python dev/scripts/benchmark_targets.py --check [--suite default]
    python dev/scripts/benchmark_targets.py --smoke [--suite smoke]

``--check`` verifies each target: ``f_true(x_min)`` equals ``f_min`` (to
rounding), ``x_min`` lies inside the hard bounds (an analytic one inside the
plausible box too) and satisfies the non-box constraint, a real-data target
reproduces its pinned values, the target is finite and no point does better
than ``f_min`` on random samples of the plausible box, of the neighbourhood
of ``x_min`` and of the hard box, the start points are reproducible, inside
the plausible box and feasible, and the target returns what its noise kind
promises. ``--smoke`` runs each configuration of a suite for one seed,
each in a fresh spawned process, and prints its wall time including the
process start-up, with the projected time of the suite at 30 seeds. Both
take ``--only``, a comma-separated list of target names or configuration
labels.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package of this checkout, whichever checkout is installed.
sys.path.insert(0, str(REPO_ROOT))

# Fixes the shift and the rotation of every target per (name, D),
# independently of the run seed: every run of every version sees the same
# targets.
STRUCTURE_SEED = 20260924

HOMO_SD = 1.0
ELLIPSOID_CONDITION = 1e6

# Error f_true(x) - f_min below which a run counts as solved: BADS's default
# tol_fun for the unimodal deterministic targets; for the multimodal ones,
# a value below every non-global local minimum (0.995 for Rastrigin, 1.15
# for Ackley at D = 10 and more at lower D), so that a solved run has
# reached the global basin; a tenth of the noise standard deviation at the
# minimum for the noisy synthetic ones; for the real-data targets, with or
# without noise, half a unit of log-likelihood (a difference of 1 in AIC),
# below which two fits of a model are equally good for model comparison.
TOL_UNIMODAL = 1e-3
TOL_MULTIMODAL = 0.5
TOL_NOISY = 0.1
TOL_REAL = 0.5

# The box of the shifted targets.
SHIFTED_BOUNDS = (-20.0, 20.0, -5.0, 5.0)  # lb, ub, plb, pub


def hetero_sd(delta):
    """Noise standard deviation of a ``"hetero"`` target, ``delta`` being
    the noiseless distance ``f_true(x) - f_min``: ``HOMO_SD`` at the minimum,
    growing as the square root of ``delta`` (the rule of the PyBADS test
    ``he_noisy_sphere``, ``2 + sqrt(f)``, with its base set to ``HOMO_SD``).
    """
    return HOMO_SD + float(np.sqrt(max(delta, 0.0)))


# --------------------------------------------------------------------------
# Problem and Config
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Problem:
    """A benchmark problem with its minimum and its BADS setup.

    ``f_vec`` maps an ``(N, D)`` array to the ``(N,)`` noiseless values;
    ``f_true`` evaluates it on one ``(D,)`` point. ``fun`` is what BADS
    receives: ``f_true`` plus noise drawn from the problem's own generator,
    returned as ``(y, sd)`` for ``"hetero"`` noise. ``non_box_cons`` maps an
    ``(N, D)`` array to ``N`` booleans, True where a point is infeasible.
    Bounds, ``x_min`` and ``x0`` are ``(D,)`` arrays. ``tolerance`` is the
    error ``f_true(x) - f_min`` below which a run counts as solved.

    The minimum of a synthetic target is analytic. That of a real-data
    target is a reference: ``reference`` holds its entry of
    ``data/reference_optima.json``, and ``pins`` hold values of an
    independent implementation of its likelihood, ``(x, expected, kind,
    tol)``, which ``--check`` compares with ``-f_vec`` at ``x``: ``kind``
    ``"loglik"`` for a log-likelihood, ``"logp"`` for a log joint under
    PyVBMC's prior (``_spline_trapezoid_logpdf``). ``check_n`` is the size
    of each random sample of ``--check``, smaller for an expensive target.
    """

    name: str
    D: int
    f_vec: Callable[[np.ndarray], np.ndarray]
    f_min: float
    x_min: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    plb: np.ndarray
    pub: np.ndarray
    tolerance: float
    x0: Optional[np.ndarray] = None
    noise: str = "none"
    non_box_cons: Optional[Callable[[np.ndarray], np.ndarray]] = None
    options: dict = dataclasses.field(default_factory=dict)
    notes: str = ""
    reference: Optional[dict] = None
    pins: tuple = ()
    check_n: int = 20_000
    _noise_rng: Optional[np.random.Generator] = dataclasses.field(
        default=None, repr=False
    )

    def f_true(self, x):
        """Noiseless value at one point."""
        x = np.asarray(x, dtype=float).reshape(1, self.D)
        return float(self.f_vec(x)[0])

    def noise_sd(self, x):
        """Noise standard deviation at one point (0 when noiseless)."""
        if self.noise == "none":
            return 0.0
        if self.noise == "homo":
            return HOMO_SD
        return hetero_sd(self.f_true(x) - self.f_min)

    def fun(self, x):
        """The target BADS calls: a scalar, or ``(y, sd)`` for "hetero"."""
        y = self.f_true(x)
        if self.noise == "none":
            return y
        sd = self.noise_sd(x)
        y_obs = y + sd * float(self._noise_rng.standard_normal())
        if self.noise == "homo":
            return y_obs
        return y_obs, sd

    def bads_args(self):
        """``(positional_args, options)`` for ``BADS(*args, options=...)``."""
        args = (
            self.fun,
            self.x0.copy(),
            self.lb.copy(),
            self.ub.copy(),
            self.plb.copy(),
            self.pub.copy(),
            self.non_box_cons,
        )
        return args, dict(self.options)

    def feasible(self, X):
        """Boolean array: which rows of ``X`` satisfy the non-box
        constraint (all of them when there is none)."""
        X = np.atleast_2d(X)
        if self.non_box_cons is None:
            return np.ones(X.shape[0], dtype=bool)
        return ~np.asarray(self.non_box_cons(X), dtype=bool).reshape(-1)


NOISE_KINDS = ("none", "homo", "hetero")


@dataclasses.dataclass(frozen=True)
class Config:
    """One entry of a suite: a target at a dimension, its noise, bounds,
    extra BADS options and evaluation budget (``max_fun_evals`` as a
    multiple of ``D``)."""

    name: str
    D: int
    noise: str = "none"
    budget: int = 500
    unbounded: bool = False
    options: tuple = ()  # (key, value) pairs, so that the Config hashes
    tag: str = ""

    @property
    def label(self):
        s = f"{self.name}_D{self.D}"
        if self.noise != "none":
            s += f"_{self.noise}"
        if self.unbounded:
            s += "_unbounded"
        if self.tag:
            s += f"_{self.tag}"
        return s

    def options_dict(self):
        return dict(self.options)

    def max_fun_evals(self, budget_scale=1.0):
        return max(1, int(round(self.budget * budget_scale * self.D)))

    def make(self, seed=None, budget_scale=1.0):
        """The problem of one run: start point and noise stream from
        ``seed``; options with ``display="off"``, the budget and
        ``random_seed=seed``, then the configuration's own options."""
        prob = make_problem(
            self.name,
            self.D,
            noise=self.noise,
            seed=seed,
            unbounded=self.unbounded,
        )
        prob.options.update(
            display="off",
            max_fun_evals=self.max_fun_evals(budget_scale),
            random_seed=seed,
        )
        prob.options.update(self.options_dict())
        return prob


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------


def _structure_rng(name, D):
    # crc32, not hash(): the same integer in every process
    return np.random.default_rng(
        [STRUCTURE_SEED, zlib.crc32(name.encode()), int(D)]
    )


def _shifted_box(D):
    lb, ub, plb, pub = SHIFTED_BOUNDS
    return (
        np.full(D, lb),
        np.full(D, ub),
        np.full(D, plb),
        np.full(D, pub),
    )


def _shift(rng, plb, pub):
    """A point drawn uniformly in the central half of the plausible box."""
    width = pub - plb
    return plb + width / 4 + rng.random(plb.shape) * width / 2


def _rotation(rng, D):
    """A random rotation (Haar-distributed orthogonal matrix)."""
    Q, R = np.linalg.qr(rng.standard_normal((D, D)))
    return Q * np.sign(np.diag(R))


def _shifted_problem(name, D, rng, f_of_z, tolerance, notes):
    lb, ub, plb, pub = _shifted_box(D)
    c = _shift(rng, plb, pub)

    def f_vec(X):
        return f_of_z(np.atleast_2d(X) - c)

    return Problem(
        name=name,
        D=D,
        f_vec=f_vec,
        f_min=0.0,
        x_min=c.copy(),
        lb=lb,
        ub=ub,
        plb=plb,
        pub=pub,
        tolerance=tolerance,
        notes=notes,
    )


def _sphere(D, rng):
    return _shifted_problem(
        "sphere",
        D,
        rng,
        lambda Z: np.sum(Z**2, axis=1),
        TOL_UNIMODAL,
        "sum(z^2)",
    )


def _ellipsoid(D, rng):
    a = ELLIPSOID_CONDITION ** (np.arange(D) / max(D - 1, 1))
    return _shifted_problem(
        "ellipsoid",
        D,
        rng,
        lambda Z: np.sum(a * Z**2, axis=1),
        TOL_UNIMODAL,
        f"sum(a_i z_i^2), a_i from 1 to {ELLIPSOID_CONDITION:g}, axis-aligned",
    )


def _rosenbrock(D, rng):
    if D < 2:
        raise ValueError("rosenbrock needs D >= 2")
    lb, ub, plb, pub = _shifted_box(D)
    c = _shift(rng, plb, pub)
    Q = _rotation(rng, D)

    def f_vec(X):
        # z = Q (x - c) + 1, row-wise; exactly 1 at x = c
        Z = (np.atleast_2d(X) - c) @ Q.T + 1.0
        return np.sum(
            100.0 * (Z[:, 1:] - Z[:, :-1] ** 2) ** 2 + (1.0 - Z[:, :-1]) ** 2,
            axis=1,
        )

    return Problem(
        name="rosenbrock",
        D=D,
        f_vec=f_vec,
        f_min=0.0,
        x_min=c.copy(),
        lb=lb,
        ub=ub,
        plb=plb,
        pub=pub,
        tolerance=TOL_UNIMODAL,
        notes="Rosenbrock of Q (x - c) + 1, Q a seeded random rotation",
    )


def _ackley_of_z(Z):
    # 20 (1 - exp(-0.2 r)) + (e - exp(mean cos 2 pi z)), written with expm1
    # so that it is exactly 0 at z = 0
    r = np.sqrt(np.mean(Z**2, axis=1))
    m = np.mean(np.cos(2.0 * np.pi * Z), axis=1)
    return -20.0 * np.expm1(-0.2 * r) - np.e * np.expm1(m - 1.0)


def _ackley(D, rng):
    return _shifted_problem(
        "ackley",
        D,
        rng,
        _ackley_of_z,
        TOL_MULTIMODAL,
        "Ackley (a = 20, b = 0.2, c = 2 pi)",
    )


def _rastrigin(D, rng):
    return _shifted_problem(
        "rastrigin",
        D,
        rng,
        lambda Z: 10.0 * Z.shape[1]
        + np.sum(Z**2 - 10.0 * np.cos(2.0 * np.pi * Z), axis=1),
        TOL_MULTIMODAL,
        "Rastrigin (A = 10)",
    )


def _sphere_nonbox_cons(X):
    X = np.atleast_2d(X)
    return X[:, 0] + X[:, 1] < np.sqrt(2.0)


def _sphere_nonbox(D, rng):
    if D < 2:
        raise ValueError("sphere_nonbox needs D >= 2")
    x_min = np.zeros(D)
    x_min[:2] = np.sqrt(2.0) / 2.0
    return Problem(
        name="sphere_nonbox",
        D=D,
        f_vec=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1),
        f_min=1.0,
        x_min=x_min,
        lb=np.full(D, -100.0),
        ub=np.full(D, 100.0),
        plb=np.full(D, -8.0),
        pub=np.full(D, 12.0),
        tolerance=TOL_UNIMODAL,
        non_box_cons=_sphere_nonbox_cons,
        notes="sum(x^2), infeasible where x_1 + x_2 < sqrt(2)",
    )


# --------------------------------------------------------------------------
# Real-data targets (see the module docstring). The likelihoods, their
# constants and their pins are those of PyVBMC's
# dev/scripts/benchmark_targets.py at commit 4822ae1f, with the sign
# flipped.
# --------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
REFERENCE_OPTIMA = DATA_DIR / "reference_optima.json"
REFERENCE_GENERATOR = "dev/scripts/make_reference_optima.py"
PIN_TOL = 1e-6  # tolerance of the pinned reference values in --check
# name: the one dimension at which the target is defined
REAL_TARGETS = {"timing": 5, "multisensory_s1": 6}

_DATA_CACHE = {}
_REFERENCE_CACHE = {}


def _load_data(name):
    """The arrays of ``data/{name}.npz``, read once per process."""
    if name not in _DATA_CACHE:
        path = DATA_DIR / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; it is a copy of PyVBMC's"
                f" dev/scripts/data/{name}.npz (see data/README.md)"
            )
        with np.load(path, allow_pickle=False) as z:
            _DATA_CACHE[name] = {k: z[k] for k in z.files}
    return _DATA_CACHE[name]


def reference_optimum(name, D):
    """The entry of a real-data target in ``data/reference_optima.json``
    (read once per process), with ``x_min`` and ``f_min``."""
    if "file" not in _REFERENCE_CACHE:
        if not REFERENCE_OPTIMA.is_file():
            raise FileNotFoundError(
                f"{REFERENCE_OPTIMA} is missing; write it with"
                f" python {REFERENCE_GENERATOR}"
            )
        _REFERENCE_CACHE["file"] = json.loads(
            REFERENCE_OPTIMA.read_text(encoding="utf-8")
        )
    entry = _REFERENCE_CACHE["file"]["targets"].get(name)
    if entry is None or entry["D"] != D:
        raise ValueError(
            f"{REFERENCE_OPTIMA} has no reference for {name} at D = {D};"
            f" write it with python {REFERENCE_GENERATOR} --only {name}"
        )
    return entry


def _spline_trapezoid_logpdf(X, a, u, v, b):
    """Log density of PyVBMC's spline-trapezoidal prior at the rows of
    ``X``, for the ``"logp"`` pins of ``--check``: the targets themselves
    have no prior.

    Per dimension the density is uniform between the pivots ``u`` and ``v``
    and tapers to zero at the hard bounds ``a`` and ``b`` as the cubic
    ``3 z^2 - 2 z^3`` of the rescaled distance ``z`` from the bound, so that
    both the density and its derivative are continuous; the marginals are
    independent. ``X`` is ``(n, D)``, the four bound arrays are ``(D,)``, the
    result ``(n,)``.
    """
    X = np.atleast_2d(X)
    left = (X >= a) & (X < u)
    plateau = (X >= u) & (X < v)
    right = (X >= v) & (X <= b)
    z = np.zeros(X.shape)
    # the quotients are formed on the whole array and masked afterwards, so
    # a degenerate box (a == u or v == b) only produces discarded values
    with np.errstate(divide="ignore", invalid="ignore"):
        z[left] = ((X - a) / (u - a))[left]
        z[right] = (1.0 - (X - v) / (b - v))[right]
        taper = np.log(3.0 * z**2 - 2.0 * z**3)  # z = 0 at a and b
    # each taper integrates to half its width, so before normalization the
    # marginal integrates to (v - u) + (u - a) / 2 + (b - v) / 2
    log_norm = np.log(0.5 * (v - u + b - a))
    log_pdf = np.where(
        left | plateau | right,
        np.where(plateau, 0.0, taper) - log_norm,
        -np.inf,
    )
    return np.sum(log_pdf, axis=1)


# Bayesian time-interval reproduction (Acerbi, Wolpert & Vijayakumar 2012),
# one subject of Experiment 3. Parameters: the sensory and motor Weber
# fractions w_s and w_m, the observer's prior mean mu_p and SD sigma_p (in
# seconds), and the lapse rate. The likelihood is PyVBMC's port of
# benchflow's ``BayesianTiming.log_likelihood``, on the same grids and with
# the same scipy calls.
TIMING_N_S = 101  # points of the interval grid, over [0, 2] s
TIMING_N_X = 401  # points of the measurement grid, one per interval
TIMING_MAX_SD = 5  # half-width of the measurement grid, in sensory SDs
# benchflow's test value, computed there with the original MATLAB code
TIMING_PIN_X = (0.15, 0.15, 0.7875, 0.225, 0.035)
TIMING_PIN_LOGLIK = -4586.122592352263


def _timing(D, rng):
    if D != 5:
        raise ValueError("timing is defined for D = 5 only")
    data = _load_data("timing")
    stim_index = data["stim_index"]
    response = data["response"]
    stimuli = data["stimuli"]
    dr = float(data["bin_size"])  # responses are binned, so dr > 0
    n_trials = response.size
    # The paper's box, Table S2 of the 2020 paper, as exported, kept as
    # published although at the maximum-likelihood point the motor Weber
    # fraction w_m (about 0.031) lies below its lower plausible bound of
    # 0.05 and the lapse rate on its lower hard bound of 0.01: the box an
    # imperfect modeller would set.
    lb, ub = data["lb"], data["ub"]
    plb, pub = data["plb"], data["pub"]

    def loglik_one(x):
        ws, wm, mu_prior, sigma_prior, lambd = x
        srange = np.linspace(0.0, 2.0, TIMING_N_S)[:, None]
        ds = srange[1, 0] - srange[0, 0]
        ll = np.zeros((n_trials, 1))
        for i_stim, mu_s in enumerate(stimuli):
            sigma_s = ws * mu_s
            xrange = np.linspace(
                max(0.0, mu_s - TIMING_MAX_SD * sigma_s),
                mu_s + TIMING_MAX_SD * sigma_s,
                TIMING_N_X,
            )[None, :]
            dx = xrange[0, 1] - xrange[0, 0]
            xpdf = stats.norm.pdf(xrange, mu_s, sigma_s)
            xpdf = xpdf / np.trapezoid(xpdf, dx=dx)
            # the observer's posterior over the interval given each
            # measurement, on the (interval, measurement) grid, and the
            # estimate it produces: the posterior mean, shrunk by the motor
            # noise (the model's optimal reproduction target)
            like = stats.norm.pdf(
                xrange, srange, ws * srange + np.finfo(float).eps
            )
            prior = stats.norm.pdf(srange, mu_prior, sigma_prior)
            post = like * prior
            post = post / np.trapezoid(post, axis=0, dx=ds)
            s_hat = np.trapezoid(post * srange, axis=0, dx=ds) / (1 + wm**2)
            s_hat = s_hat[None, :]
            # probability of each observed response bin under motor noise,
            # marginalized over the measurement
            idx = stim_index == i_stim
            sigma_m = wm * s_hat
            r = response[idx][:, None]
            pr = stats.norm.cdf(r + 0.5 * dr, s_hat, sigma_m) - stats.norm.cdf(
                r - 0.5 * dr, s_hat, sigma_m
            )
            ll[idx] = np.trapezoid(xpdf * pr, axis=1, dx=dx)[:, None]
        # a lapse responds uniformly over the bins of the interval grid
        n_bins = (srange[-1, 0] - srange[0, 0]) / dr
        return float(np.sum(np.log(ll * (1 - lambd) + lambd / n_bins)))

    def f_vec(X):
        # about 40 ms per row, evaluated one by one (BADS asks for one point
        # per call)
        return np.array([-loglik_one(row) for row in np.atleast_2d(X)])

    return Problem(
        name="timing",
        D=D,
        f_vec=f_vec,
        f_min=np.nan,
        x_min=np.full(D, np.nan),
        lb=np.array(lb, dtype=float),
        ub=np.array(ub, dtype=float),
        plb=np.array(plb, dtype=float),
        pub=np.array(pub, dtype=float),
        tolerance=TOL_REAL,
        pins=((TIMING_PIN_X, TIMING_PIN_LOGLIK, "loglik", PIN_TOL),),
        check_n=100,
        notes=(
            "negative log-likelihood, Bayesian time-interval reproduction"
            f" (Acerbi et al. 2012), {n_trials} trials, the 2020 paper's box"
        ),
    )


# Visuo-vestibular unity judgments (Acerbi, Dokka, Angelaki & Ma 2018): the
# observer reports one source when the two noisy measurements differ by less
# than kappa, with a lapse, at three visual coherence levels. Parameters in
# the paper's order; the likelihood is PyVBMC's port of benchflow's
# ``Multisensory_6D``, whose order it remaps (sigma_vis x 3, sigma_vest,
# lambda, kappa). The bounds are the 2020 paper's.
MULTISENSORY_PARAMS = (
    "sigma_vest",
    "sigma_vis_low",
    "sigma_vis_med",
    "sigma_vis_high",
    "kappa",
    "lambda",
)
MULTISENSORY_LB = np.array([0.5, 0.5, 0.5, 0.5, 0.25, 0.005])
MULTISENSORY_UB = np.array([80.0, 80.0, 80.0, 80.0, 180.0, 0.5])
MULTISENSORY_PLB = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.01])
MULTISENSORY_PUB = np.array([40.0, 40.0, 40.0, 40.0, 45.0, 0.2])
# benchflow's stored posterior mode of subject 1 under PyVBMC's prior, in
# the paper's order, and the log joint there: it pins the likelihood and
# the prior together.
MULTISENSORY_S1_PIN_X = (
    7.00615081,
    2.3969857,
    1.37271846,
    8.43661978,
    10.13859551,
    0.02525963,
)
MULTISENSORY_S1_PIN_LOGP = -503.4863062430452


def _multisensory_s1(D, rng):
    if D != 6:
        raise ValueError("multisensory_s1 is defined for D = 6 only")
    data = _load_data("multisensory")
    # benchflow's three per-subject cells are taken in their stored order as
    # the low, medium and high coherence levels; the source file does not
    # label them, and the four noise parameters share one set of bounds, so
    # the order only matters for naming the parameters
    trials = [
        (data[f"s1_c{c}_stim"], data[f"s1_c{c}_resp"] == 2) for c in (1, 2, 3)
    ]
    n_trials = sum(len(report_two) for _, report_two in trials)
    # The likelihood depends on sigma_vest and the three sigma_vis only
    # through the three combined SDs sqrt(sigma_vest^2 + sigma_vis^2), so
    # its minimum is a curve, of which the reference x_min is one point.

    def loglik_vec(X):
        X = np.atleast_2d(X)
        sigma_vest = X[:, 0:1]
        sigma_vis = X[:, 1:4]
        kappa = X[:, 4:5]
        lambd = X[:, 5:6]
        out = np.zeros(X.shape[0])
        for level, (stim, report_two) in enumerate(trials):
            s_vest, s_vis = stim[:, 0], stim[:, 1]
            sigma_v = sigma_vis[:, level : level + 1]
            # the difference of the two measurements is Gaussian around the
            # difference of the directions with SD sqrt(sigma_vest^2 +
            # sigma_vis^2), written here in units of sigma_vis; p_one is the
            # probability that it falls within -+ kappa, mixed with the lapse
            scale = np.sqrt(1.0 + (sigma_vest / sigma_v) ** 2)
            a_plus = (s_vest - s_vis + kappa) / sigma_v
            a_minus = (s_vest - s_vis - kappa) / sigma_v
            p_one = 0.5 * lambd + (1 - lambd) * (
                stats.norm.cdf(a_plus / scale)
                - stats.norm.cdf(a_minus / scale)
            )
            # the response is coded 2 when the subject reported two sources
            out += stats.bernoulli.logpmf(report_two, p=1 - p_one).sum(1)
        return out

    return Problem(
        name="multisensory_s1",
        D=D,
        f_vec=lambda X: -loglik_vec(X),
        f_min=np.nan,
        x_min=np.full(D, np.nan),
        lb=MULTISENSORY_LB.copy(),
        ub=MULTISENSORY_UB.copy(),
        plb=MULTISENSORY_PLB.copy(),
        pub=MULTISENSORY_PUB.copy(),
        tolerance=TOL_REAL,
        pins=(
            (MULTISENSORY_S1_PIN_X, MULTISENSORY_S1_PIN_LOGP, "logp", PIN_TOL),
        ),
        check_n=2000,
        notes=(
            "negative log-likelihood, visuo-vestibular causal inference"
            f" (Acerbi et al. 2018), subject 1, {n_trials} trials"
        ),
    )


_REGISTRY = {
    "sphere": _sphere,
    "ellipsoid": _ellipsoid,
    "rosenbrock": _rosenbrock,
    "ackley": _ackley,
    "rastrigin": _rastrigin,
    "sphere_nonbox": _sphere_nonbox,
    "timing": _timing,
    "multisensory_s1": _multisensory_s1,
}

TARGET_NAMES = tuple(_REGISTRY)

# The noise kinds that set BADS options; "none" leaves uncertainty_handling
# at its default, so BADS tests the start point for noise.
_NOISE_OPTIONS = {
    "none": {},
    "homo": {"uncertainty_handling": True},
    "hetero": {"uncertainty_handling": True, "specify_target_noise": True},
}


def _draw_x0(prob, rng, max_tries=10_000):
    """Start point uniform in the plausible box, redrawn from the same
    stream until it satisfies the non-box constraint."""
    for _ in range(max_tries):
        x0 = prob.plb + rng.random(prob.D) * (prob.pub - prob.plb)
        if prob.feasible(x0)[0]:
            return x0
    raise RuntimeError(f"no feasible start point for {prob.name}")


def make_problem(
    name, D, noise="none", seed=None, unbounded=False, reference=True
):
    """Build the ``Problem`` of one run.

    ``seed`` is the run seed: ``SeedSequence(seed).spawn(2)`` gives the start
    point stream and the noise stream (``None``: fresh entropy). The target's
    structure depends only on ``(name, D)``. ``reference=False`` leaves a
    real-data target's stored reference minimum unread, with ``f_min`` and
    ``x_min`` NaN: ``make_reference_optima.py`` builds the target whose
    reference it writes.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown target {name!r}; known: {TARGET_NAMES}")
    if noise not in NOISE_KINDS:
        raise ValueError(f"unknown noise {noise!r}; known: {NOISE_KINDS}")
    if unbounded and name in REAL_TARGETS:
        # a likelihood is defined only inside its hard bounds
        raise ValueError(f"{name} cannot be unbounded")
    D = int(D)
    prob = _REGISTRY[name](D, _structure_rng(name, D))
    if reference and name in REAL_TARGETS:
        prob.reference = reference_optimum(name, D)
        prob.x_min = np.array(prob.reference["x_min"], dtype=float)
        prob.f_min = float(prob.reference["f_min"])
    if unbounded:
        prob.lb = np.full(D, -np.inf)
        prob.ub = np.full(D, np.inf)
    x0_ss, noise_ss = np.random.SeedSequence(seed).spawn(2)
    prob.x0 = _draw_x0(prob, np.random.default_rng(x0_ss))
    prob.noise = noise
    if noise != "none":
        if name not in REAL_TARGETS:
            prob.tolerance = TOL_NOISY
        prob._noise_rng = np.random.default_rng(noise_ss)
    prob.options.update(_NOISE_OPTIONS[noise])
    return prob


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------

# The default suite. Its 18 configurations cover every target, dimension (2,
# 3, 6, and 10 for sphere and ellipsoid; 5 for timing), noise kind and
# constraint type, not every combination; the ellipsoid at D = 3 appears
# with finite bounds, infinite bounds and both noise kinds, on the same
# shifted target, and multisensory_s1 with and without noise. Every budget
# is BADS's default, 500 D. A calibration at that budget (4 seeds per
# configuration, 2026-09-24) found every run ending on BADS's own
# termination, after 55 to 863 evaluations: 60 at sphere D2, about 800 at
# ellipsoid D10, 200 to 500 for the noisy synthetic targets, 200 to 330 for
# timing, about 300 for multisensory_s1 and 600 to 830 for it with noise.
# At about 40 ms per evaluation, a timing run takes 10 to 17 s. Starting a
# fresh process and importing PyBADS adds about 2 s per run.
_DEFAULT = [
    Config("sphere", 2, budget=500),
    Config("sphere", 10, budget=500),
    Config("ellipsoid", 3, budget=500),
    Config("ellipsoid", 6, budget=500),
    Config("ellipsoid", 10, budget=500),
    Config("rosenbrock", 2, budget=500),
    Config("rosenbrock", 6, budget=500),
    Config("ackley", 6, budget=500),
    Config("rastrigin", 3, budget=500),
    Config("sphere", 3, noise="homo", budget=500),
    Config("ellipsoid", 3, noise="homo", budget=500),
    Config("sphere", 3, noise="hetero", budget=500),
    Config("ellipsoid", 3, noise="hetero", budget=500),
    Config("sphere_nonbox", 3, budget=500),
    Config("ellipsoid", 3, budget=500, unbounded=True),
    Config("timing", 5, budget=500),
    Config("multisensory_s1", 6, budget=500),
    Config("multisensory_s1", 6, noise="homo", budget=500),
]

# One configuration per code path: deterministic, inferred noise, specified
# noise, non-box constraint, infinite bounds.
_SMOKE = (
    "sphere_D2",
    "sphere_D3_homo",
    "sphere_D3_hetero",
    "sphere_nonbox_D3",
    "ellipsoid_D3_unbounded",
)

SUITES = {
    "smoke": [c for c in _DEFAULT if c.label in _SMOKE],
    "default": _DEFAULT,
}


def suite_configs(suite):
    if suite == "all":
        seen = {}
        for s in SUITES.values():
            for c in s:
                if seen.setdefault(c.label, c) != c:
                    # a record names its configuration by label alone
                    raise ValueError(f"two configurations labelled {c.label}")
        return list(seen.values())
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; known: {list(SUITES)}")
    return list(SUITES[suite])


def find_config(label):
    for c in suite_configs("all"):
        if c.label == label:
            return c
    raise ValueError(f"unknown config label {label!r}")


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------

CHECK_SEED = 7
CHECK_RTOL = 1e-12  # f_true(x_min) against f_min: equal up to rounding


def _close(a, b):
    return abs(a - b) <= CHECK_RTOL * max(1.0, abs(a), abs(b))


def _pin_value(prob, x, kind):
    """What a pin of ``prob`` compares with its expected value at ``x``."""
    X = np.reshape(np.asarray(x, dtype=float), (1, prob.D))
    value = -float(prob.f_vec(X)[0])
    if kind == "logp":
        box = (prob.lb, prob.plb, prob.pub, prob.ub)
        value += float(_spline_trapezoid_logpdf(X, *box)[0])
    elif kind != "loglik":
        raise ValueError(f"unknown pin kind {kind!r}")
    return value


def check_problem(cfg, n=None):
    """Checks of one configuration's target; returns ``(ok, info,
    messages)``. ``n`` is the size of each random sample (default: the
    problem's ``check_n``)."""
    msgs = []

    def make(seed):
        return make_problem(cfg.name, cfg.D, cfg.noise, seed, cfg.unbounded)

    prob = make(0)
    D, x_min = prob.D, prob.x_min
    n = prob.check_n if n is None else n
    # the minimum
    f_at_min = prob.f_true(x_min)
    if not _close(f_at_min, prob.f_min):
        msgs.append(f"f_true(x_min) = {f_at_min!r} != f_min = {prob.f_min!r}")
    if not (np.all(prob.lb <= x_min) and np.all(x_min <= prob.ub)):
        msgs.append("x_min outside the hard bounds")
    # an analytic minimum lies inside the plausible box by construction; a
    # reference minimum need not (that of timing does not)
    if prob.reference is None and not (
        np.all(prob.plb < x_min) and np.all(x_min < prob.pub)
    ):
        msgs.append("x_min outside the plausible box")
    if not prob.feasible(x_min)[0]:
        msgs.append("x_min violates the non-box constraint")
    # BADS's requirements on the bounds
    if not np.all(
        (prob.lb <= prob.plb) & (prob.plb < prob.pub) & (prob.pub <= prob.ub)
    ):
        msgs.append("bounds not ordered lb <= plb < pub <= ub")
    finite = np.isfinite(np.concatenate([prob.lb, prob.ub]))
    if not (np.all(finite) or not np.any(finite)):
        msgs.append("hard bounds mix finite and infinite values")
    # values of an independent implementation of the likelihood
    pin_diffs = []
    for x, expected, kind, tol in prob.pins:
        d = abs(_pin_value(prob, x, kind) - expected)
        pin_diffs.append(d)
        if d > tol:
            msgs.append(
                f"pinned {kind} differs by {d:.2e} (tolerance {tol:.0e})"
            )
    # the target is finite, and no point does better than f_min, on random
    # samples of the plausible box, of the neighbourhood of x_min (within the
    # hard bounds) and of the hard box when it is finite
    rng = np.random.default_rng(CHECK_SEED)
    X_box = prob.plb + rng.random((n, D)) * (prob.pub - prob.plb)
    X_near = x_min + 1e-3 * (prob.pub - prob.plb) * rng.standard_normal((n, D))
    X_near = np.clip(X_near, prob.lb, prob.ub)
    X_hard = np.empty((0, D))
    if np.all(finite):
        X_hard = prob.lb + rng.random((n, D)) * (prob.ub - prob.lb)
    X = np.vstack([X_box, X_near, X_hard])
    X = X[prob.feasible(X)]
    f_sample = prob.f_vec(X)
    n_bad = int(np.sum(~np.isfinite(f_sample)))
    if n_bad:
        msgs.append(f"{n_bad} sampled points with a non-finite value")
        X, f_sample = X[np.isfinite(f_sample)], f_sample[np.isfinite(f_sample)]
    if np.min(f_sample) < prob.f_min - CHECK_RTOL * max(1.0, abs(prob.f_min)):
        i = int(np.argmin(f_sample))
        msgs.append(
            f"sampled point beats f_min: f = {f_sample[i]!r} at {X[i]}"
        )
    # start points: reproducible, seed-dependent, inside the box, feasible
    x0s = [make(s).x0 for s in range(5)]
    again = make(0).x0
    if not np.array_equal(x0s[0], again):
        msgs.append("x0 not reproducible from its seed")
    if len({tuple(x) for x in x0s}) < len(x0s):
        msgs.append("different seeds gave the same x0")
    for x0 in x0s:
        if not (np.all(prob.plb <= x0) and np.all(x0 <= prob.pub)):
            msgs.append("x0 outside the plausible box")
        if not prob.feasible(x0)[0]:
            msgs.append("x0 violates the non-box constraint")
    # the target returns what its noise kind promises
    out1, out2 = prob.fun(x_min), prob.fun(x_min)
    if prob.noise == "none":
        if not (isinstance(out1, float) and out1 == f_at_min == out2):
            msgs.append(f"noiseless target returned {out1!r}, {out2!r}")
    elif prob.noise == "homo":
        if not (isinstance(out1, float) and out1 != out2):
            msgs.append(f"homo target returned {out1!r}, {out2!r}")
    else:
        ok = (
            isinstance(out1, tuple)
            and len(out1) == 2
            and out1[1] == hetero_sd(0.0)
            and out1[0] != out2[0]
        )
        if not ok:
            msgs.append(f"hetero target returned {out1!r}, {out2!r}")
    info = (
        f"f_min={prob.f_min:g} |f_true(x_min)-f_min|="
        f"{abs(f_at_min - prob.f_min):.1e} min sampled f - f_min="
        f"{np.min(f_sample) - prob.f_min:.2e} tol={prob.tolerance:g}"
    )
    if pin_diffs:
        info += f" max |pin diff|={max(pin_diffs):.1e}"
    return not msgs, info, msgs


def _selected(cfg, only):
    """Whether a config passes an ``--only`` filter: ``None`` for all,
    otherwise a set of target names or config labels."""
    return only is None or cfg.name in only or cfg.label in only


def run_check(configs, only=None):
    all_ok = True
    for cfg in configs:
        if not _selected(cfg, only):
            continue
        ok, info, msgs = check_problem(cfg)
        all_ok &= ok
        print(
            f"[check] {'ok  ' if ok else 'FAIL'} {cfg.label:28s} {info}",
            flush=True,
        )
        for m in msgs:
            print(f"        ! {m}", flush=True)
    return all_ok


# --------------------------------------------------------------------------
# --smoke
# --------------------------------------------------------------------------


def single_thread_env():
    """One BLAS thread per process and a headless matplotlib, for the
    spawned processes of ``--smoke`` and ``population.py run``."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["MPLBACKEND"] = "Agg"


def _smoke_task(label, seed, budget_scale):
    """One BADS run of a configuration (executed in a spawned process)."""
    from pybads import BADS

    cfg = find_config(label)
    prob = cfg.make(seed=seed, budget_scale=budget_scale)
    args, options = prob.bads_args()
    t0 = time.perf_counter()
    res = BADS(*args, options=options).optimize()
    wall = time.perf_counter() - t0
    return {
        "bads_s": wall,
        "func_count": int(res["func_count"]),
        "max_fun_evals": options["max_fun_evals"],
        "true_error": prob.f_true(np.ravel(res["x"])) - prob.f_min,
        "message": str(res["message"]),
    }


def run_smoke(configs, seed=0, only=None, budget_scale=1.0, n_seeds=30):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    single_thread_env()
    configs = [c for c in configs if _selected(c, only)]
    all_ok = True
    # The pool of population.py run: one worker, a fresh spawned process per
    # run. The runs execute in submission order, and the pool starts each
    # next process as the previous one exits, so the time between two
    # completions is the cost of one run in a population, start-up included.
    t_start = t_prev = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        max_tasks_per_child=1,
    ) as ex:
        futs = [
            ex.submit(_smoke_task, c.label, seed, budget_scale)
            for c in configs
        ]
        for cfg, fut in zip(configs, futs):
            try:
                r = fut.result()
                ok = bool(np.isfinite(r["true_error"]))
                msg = (
                    f"BADS {r['bads_s']:5.1f} s  evals {r['func_count']:4d}/"
                    f"{r['max_fun_evals']:<4d} err {r['true_error']:.3g}"
                    f"  {r['message'][:48]}"
                )
            except Exception as e:  # noqa: BLE001
                ok = False
                msg = f"EXCEPTION {type(e).__name__}: {e}"
            now = time.perf_counter()
            wall, t_prev = now - t_prev, now
            all_ok &= ok
            print(
                f"[smoke] {'ok  ' if ok else 'FAIL'} {cfg.label:28s}"
                f" {wall:5.1f} s  {msg}",
                flush=True,
            )
    total = t_prev - t_start
    print(
        f"[smoke] total {total:.1f} s for one seed; {n_seeds} seeds:"
        f" about {total * n_seeds / 60:.0f} min",
        flush=True,
    )
    return all_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="list the suites")
    ap.add_argument("--check", action="store_true", help="verify targets")
    ap.add_argument("--smoke", action="store_true", help="one run each")
    ap.add_argument(
        "--suite",
        default=None,
        help=f"{'|'.join(SUITES)}|all (default: all for --check, smoke"
        " for --smoke)",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated target names or configuration labels",
    )
    ap.add_argument("--seed", type=int, default=0, help="--smoke run seed")
    ap.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="--smoke: multiply every configuration's budget",
    )
    args = ap.parse_args(argv)
    only = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        known = {c.label for c in suite_configs("all")} | set(TARGET_NAMES)
        unknown = sorted(only - known)
        if unknown:
            ap.error(f"unknown --only entries: {', '.join(unknown)}")
    if args.list or not (args.check or args.smoke):
        for s, cfgs in SUITES.items():
            print(f"{s}:")
            for c in cfgs:
                notes = make_problem(c.name, c.D, seed=0).notes
                evals = c.max_fun_evals()
                print(
                    f"    {c.label:24s} {c.budget:2d}*D = {evals:3d}"
                    f" evaluations  {notes}"
                    + (f"  options={c.options_dict()}" if c.options else "")
                )
        return 0
    ok = True
    if args.check:
        ok &= run_check(suite_configs(args.suite or "all"), only=only)
    if args.smoke:
        ok &= run_smoke(
            suite_configs(args.suite or "smoke"),
            seed=args.seed,
            only=only,
            budget_scale=args.budget_scale,
        )
    print("[benchmark_targets]", "all ok" if ok else "FAILURES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
