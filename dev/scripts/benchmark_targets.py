"""Benchmark targets for PyBADS developer tooling.

One place that defines the benchmark problems (objective, analytic minimum,
bounds, noise, BADS options) and the named suites built from them, so that
``population.py`` and any other tool run the same problems.

Targets (``make_problem(name, D)``); all but ``sphere_nonbox`` are shifted
so that their minimum lies at a point ``c`` drawn uniformly in the central
half of the plausible box, with ``z = x - c``:

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
                   bounds of ``get_test_opt_conf`` in the PyBADS
                   tests (whose ``test_sphere_opt`` marks the other side of
                   the line infeasible, which leaves the minimum 0 at the
                   origin)

Every minimum is 0 except that of ``sphere_nonbox``. The shifted targets
share the hard bounds ``[-20, 20]`` and the plausible box ``[-5, 5]`` in each
variable; a configuration with ``unbounded=True`` replaces the hard bounds by
infinities and keeps the plausible box (BADS accepts either all bounds
finite or all infinite).

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
``D``. The budgets of the ``default`` suite are set from the ``--smoke``
timings, so that the suite at 30 seeds runs in about 30 minutes as one
process with a fresh process per run (``population.py run``).

Command line (from the repository root)::

    python dev/scripts/benchmark_targets.py --list
    python dev/scripts/benchmark_targets.py --check [--suite default]
    python dev/scripts/benchmark_targets.py --smoke [--suite smoke]

``--check`` verifies each target: ``f_true(x_min)`` equals ``f_min`` (to
rounding), ``x_min`` lies inside the hard and plausible bounds and satisfies
the non-box constraint, no point of a random sample of the box and of the
neighbourhood of ``x_min`` does better, the start points are reproducible,
inside the plausible box and feasible, and the target returns what its noise
kind promises. ``--smoke`` runs each configuration of a suite for one seed,
each in a fresh spawned process, and prints its wall time including the
process start-up, with the projected time of the suite at 30 seeds. Both
take ``--only``, a comma-separated list of target names or configuration
labels.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

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
# minimum for the noisy ones.
TOL_UNIMODAL = 1e-3
TOL_MULTIMODAL = 0.5
TOL_NOISY = 0.1

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
    """A benchmark problem with its analytic minimum and its BADS setup.

    ``f_vec`` maps an ``(N, D)`` array to the ``(N,)`` noiseless values;
    ``f_true`` evaluates it on one ``(D,)`` point. ``fun`` is what BADS
    receives: ``f_true`` plus noise drawn from the problem's own generator,
    returned as ``(y, sd)`` for ``"hetero"`` noise. ``non_box_cons`` maps an
    ``(N, D)`` array to ``N`` booleans, True where a point is infeasible.
    Bounds, ``x_min`` and ``x0`` are ``(D,)`` arrays. ``tolerance`` is the
    error ``f_true(x) - f_min`` below which a run counts as solved.
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
    budget: int = 50
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


_REGISTRY = {
    "sphere": _sphere,
    "ellipsoid": _ellipsoid,
    "rosenbrock": _rosenbrock,
    "ackley": _ackley,
    "rastrigin": _rastrigin,
    "sphere_nonbox": _sphere_nonbox,
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


def make_problem(name, D, noise="none", seed=None, unbounded=False):
    """Build the ``Problem`` of one run.

    ``seed`` is the run seed: ``SeedSequence(seed).spawn(2)`` gives the start
    point stream and the noise stream (``None``: fresh entropy). The target's
    structure depends only on ``(name, D)``.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown target {name!r}; known: {TARGET_NAMES}")
    if noise not in NOISE_KINDS:
        raise ValueError(f"unknown noise {noise!r}; known: {NOISE_KINDS}")
    D = int(D)
    prob = _REGISTRY[name](D, _structure_rng(name, D))
    if unbounded:
        prob.lb = np.full(D, -np.inf)
        prob.ub = np.full(D, np.inf)
    x0_ss, noise_ss = np.random.SeedSequence(seed).spawn(2)
    prob.x0 = _draw_x0(prob, np.random.default_rng(x0_ss))
    prob.noise = noise
    if noise != "none":
        prob.tolerance = TOL_NOISY
        prob._noise_rng = np.random.default_rng(noise_ss)
    prob.options.update(_NOISE_OPTIONS[noise])
    return prob


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------

# The default suite. With a fresh process per run, starting the process and
# importing PyBADS costs 1.5 to 2 s per run, as much as a run itself (1 to 5
# s at these budgets, in the --smoke timings the budgets were set from), so
# 30 seeds in about 30 minutes allow about 15 configurations. They cover
# every target, dimension (2, 3, 6, and 10 for sphere and ellipsoid), noise
# kind and constraint type, not every combination; the ellipsoid at D = 3
# appears with finite bounds, infinite bounds and both noise kinds, on the
# same shifted target. Budgets: 50 D evaluations at D <= 3, 150 at D = 6 and
# D = 10; most runs end on the budget, so a change in speed shows in
# true_error.
_DEFAULT = [
    Config("sphere", 2, budget=50),
    Config("sphere", 10, budget=15),
    Config("ellipsoid", 3, budget=50),
    Config("ellipsoid", 6, budget=25),
    Config("ellipsoid", 10, budget=15),
    Config("rosenbrock", 2, budget=50),
    Config("rosenbrock", 6, budget=25),
    Config("ackley", 6, budget=25),
    Config("rastrigin", 3, budget=50),
    Config("sphere", 3, noise="homo", budget=50),
    Config("ellipsoid", 3, noise="homo", budget=50),
    Config("sphere", 3, noise="hetero", budget=50),
    Config("ellipsoid", 3, noise="hetero", budget=50),
    Config("sphere_nonbox", 3, budget=50),
    Config("ellipsoid", 3, budget=50, unbounded=True),
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


def check_problem(cfg, n_box=20_000, n_near=20_000):
    """Checks of one configuration's target; returns ``(ok, messages)``."""
    msgs = []

    def make(seed):
        return make_problem(cfg.name, cfg.D, cfg.noise, seed, cfg.unbounded)

    prob = make(0)
    D, x_min = prob.D, prob.x_min
    # the analytic minimum
    f_at_min = prob.f_true(x_min)
    if not _close(f_at_min, prob.f_min):
        msgs.append(f"f_true(x_min) = {f_at_min!r} != f_min = {prob.f_min!r}")
    if not (np.all(prob.lb <= x_min) and np.all(x_min <= prob.ub)):
        msgs.append("x_min outside the hard bounds")
    if not (np.all(prob.plb < x_min) and np.all(x_min < prob.pub)):
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
    # no sampled point does better than f_min
    rng = np.random.default_rng(CHECK_SEED)
    X_box = prob.plb + rng.random((n_box, D)) * (prob.pub - prob.plb)
    X_near = x_min + 1e-3 * (prob.pub - prob.plb) * rng.standard_normal(
        (n_near, D)
    )
    X = np.vstack([X_box, X_near])
    X = X[prob.feasible(X)]
    f_sample = prob.f_vec(X)
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
