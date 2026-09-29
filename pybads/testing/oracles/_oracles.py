"""The oracles: named computations of PyBADS on a rebuilt snapshot state.

Each oracle is a function ``(state, seed) -> dict[str, ndarray]`` over the
dict that :func:`._state.build_state` returns, registered with a tolerance
class for each of its outputs. The fixture generator
(``dev/scripts/make_oracle_fixtures.py``) stores what the oracles return as
the references; the tests recompute them and compare with :func:`compare`.

Draws. The oracles of the pieces that draw (the ES search's candidates, the
hedge's choice, ``poll_mads_2n``) take a :class:`ScriptedGenerator`, a
``numpy.random.Generator`` whose draws are computed from the raw 64-bit
stream of ``PCG64(seed)`` by exact arithmetic, the same on every platform
and every NumPy version, and which records each call. A change in the
number, the kind or the order of the draws moves these oracles on purpose.
The platform-bound oracles draw from ``numpy.random.default_rng(seed)``.

Tolerance classes. Per element, ``|out - ref| <= rtol * max(|ref|, q25) +
atol``, with ``q25`` the lower quartile of ``|ref|`` over the finite
entries (:func:`compare`); NaN and infinite entries must match exactly.

- ``exact`` (0, 0): integers, booleans, selections of rows, points on a
  grid (``force_to_grid`` divides and rounds, which IEEE arithmetic gives
  alike everywhere), and what prescribed draws give by such arithmetic.
- ``gp_free`` (1e-10, 1e-13): elementwise arithmetic, ``log``, ``exp``,
  ``erfc``, percentiles and sums, without BLAS.
- ``linalg`` (1e-9, 1e-13): small matrix products and the eigenvalues of
  ES-wcm's covariance (BLAS and LAPACK, no solve).
- ``gp_mean`` (1e-4, 1e-10): the GP's predictive mean, through the solve
  with its training covariance.
- ``gp_var`` (1e-3, 1e-8): the GP's predictive variance and SD, the LCB
  (the mean less a multiple of the SD) and the hedge's gains and
  probabilities that the GP's predictions set; a variance near a training
  point is a difference of nearly equal terms.

Portable and platform-bound outputs. The fixtures store the portable
outputs alone, which the tests compare on every platform under their
tolerances. An output is platform-bound (:func:`platform_bound`) when
rounding on another platform moves it beyond any tolerance that would
still catch a change: every output of the oracles of ``PLATFORM_BOUND``, a
GP fit (L-BFGS-B from several starts) and a whole ES search step with the
LCB (a ranking of thousands of candidates), which turn rounding into
different decisions; and, where a GP is ill-conditioned
(:func:`gp_condition_bound` above ``GP_CONDITION_MAX``), every output that
goes through its solve. Platform-bound outputs reproduce only on one
machine, with one set of libraries and one BLAS setting, and are compared
only between two commits there (``make_oracle_fixtures.py --dump`` and
``--against``). ES-wcm takes the eigenvectors of its covariance with the
signs that LAPACK gives them, so its candidates are left to
``es_search_step``; ``es_setup`` pins its covariance, in which the signs
cancel.

Views. A deterministic run's GP, after a refit, has its noise at its lower
bound and an ill-conditioned training covariance. On such a snapshot the
oracles with outputs through the GP's solve (:func:`view_oracles`) are
computed twice: on the state as stored, where those outputs are
platform-bound, and in the view ``"noise_floor"``, with the GP's noise
raised so that the bound on the condition number is
``NOISE_FLOOR_CONDITION`` (:func:`noise_floor_hyperparameters`; the
fixture stores the raised hyperparameters), where every output is
portable. An oracle's outputs in a view are stored under the case
:func:`case_name`, and :func:`oracle_cases` lists the cases of a snapshot.

Measured floors (2026-09-29; Linux x86_64 with 4 cores, NumPy 2.4.6 and
SciPy 1.17.1 with their OpenBLAS 0.3.31, gpyreg 1.3.3). The references were
computed with one BLAS thread and OpenBLAS's kernel for the CPU (SkylakeX),
and recomputed with 2 and 4 threads (4 is OpenBLAS's default there), with
one thread under ``OPENBLAS_CORETYPE`` Haswell and Sandybridge, with 4
threads under Sandybridge, and with NumPy's AVX-512 dispatch disabled
(``NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"``), alone and
under Sandybridge. The thread count moved nothing: the matrices are too
small for OpenBLAS to split. The kernels moved the GP classes and
``linalg``, and NumPy's dispatch ``gp_free`` by an ulp. The largest
deviation of the stored outputs, scaled as in the rule above, and the
margin that each tolerance leaves over it (the GP classes' largest are
those of the noise-floor views; of the stored states, 3.6e-10 and
1.8e-10):

=========  ==============  =========  ======
class      largest scaled  rtol       margin
=========  ==============  =========  ======
exact      0               0
gp_free    2.5e-16         1e-10      4e5
linalg     4.9e-15         1e-9       2e5
gp_mean    6.2e-10         1e-4       2e5
gp_var     1.6e-9          1e-3       6e5
=========  ==============  =========  ======

``gp_free`` involves no BLAS; its tolerance leaves room for another
platform's ``log``, ``exp``, ``erfc`` and SIMD summation, which can differ by
an ulp or two. The GP classes take PyVBMC's tolerances for its GP's
predictions (``pyvbmc/testing/oracles/_oracles.py``: 1e-4 for the mean at
candidate points, 1e-3 for the variance), set after its first CI runs on
Ubuntu and macOS, whose BLAS builds moved GP outputs far beyond the floors
measured on one machine (the predictive mean at candidate points by
1.2e-6). The snapshots whose GP is well-conditioned have bounds of 5e4 to
4e5 on the condition number, and the noise-floor views 1e6. The three
deterministic snapshots taken after refits have the GP's noise at its lower
bound, and bounds of 8e14 to 3e16: there the kernels moved the predictive
means by up to 4e-3 and the variances by up to 0.6 (scaled; measured
2026-09-28). The kernels moved the GP fit of every snapshot, and the ES
step's LCB of all but one. Windows and macOS have not been measured: where
a platform exceeds a tolerance, measure there
(``make_oracle_fixtures.py --check --verbose``) rather than loosen it.
"""

import contextlib
import copy
import importlib
import types
from fnmatch import fnmatchcase

import numpy as np

from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb
from pybads.bads.bads import BADS
from pybads.bads.gaussian_process_train import _gp_hyp, local_gp_fitting
from pybads.function_logger.constraints_check import contraints_check
from pybads.poll.poll_mads_2n import poll_mads_2n
from pybads.search.es_search import ESSearchELL, ESSearchWM, ucov
from pybads.search.grid_functions import force_to_grid, grid_units, udist
from pybads.search.search_hedge import ESSearchHedge
from pybads.utils import IterationHistory

from ._state import new_gp, snapshot_views

DEFAULT_SEED = 20260928

TOLERANCES = {
    "exact": (0.0, 0.0),
    "gp_free": (1e-10, 1e-13),
    "linalg": (1e-9, 1e-13),
    "gp_mean": (1e-4, 1e-10),
    "gp_var": (1e-3, 1e-8),
}
# The classes of the outputs that go through the solve with the GP's
# training covariance
GP_CLASSES = ("gp_mean", "gp_var")

PLATFORM_BOUND = frozenset({"es_search_step", "gp_refit"})
# The largest bound on the condition number of the GP's training covariance
# (`gp_condition_bound`) under which the outputs that go through its solve
# take their tolerances on every platform: the well-conditioned snapshots
# have bounds of at most 4e5, the others of at least 8e14
GP_CONDITION_MAX = 1e8
# The bound on the condition number that the noise-floor view gives
NOISE_FLOOR_CONDITION = 1e6


# --------------------------------------------------------------------------
# Registry and comparison
# --------------------------------------------------------------------------


def cast_outputs(out):
    """The outputs of an oracle as float64 arrays, the form in which the
    references are stored and compared (booleans become 0 and 1)."""
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


class Oracle:
    """An oracle and the tolerance classes of its outputs: ``tol`` is a
    class name, or a dict from output names, or ``fnmatch`` patterns of
    them, to class names, with a ``"default"`` entry. An exact name wins
    over a pattern, and the first matching pattern over the default.

    The outputs that go through the solve with the GP's training covariance
    are those of the classes ``gp_mean`` and ``gp_var`` and those matching
    a pattern of ``gp_keys`` (an output decided by such a value)."""

    def __init__(self, name, fn, tol, gp_keys=()):
        self.name = name
        self.fn = fn
        self.tol = tol if isinstance(tol, dict) else {"default": tol}
        self.gp_keys = tuple(gp_keys)
        for cls in self.tol.values():
            if cls not in TOLERANCES:
                raise ValueError(f"{name}: unknown tolerance class {cls!r}")

    def depends_on_gp(self, key):
        """Whether the output ``key`` goes through the GP's solve."""
        return self.tolerance_class(key) in GP_CLASSES or any(
            fnmatchcase(key, p) for p in self.gp_keys
        )

    def has_gp_outputs(self):
        """Whether some output goes through the GP's solve, by its class or
        a pattern of ``gp_keys``."""
        return bool(self.gp_keys) or any(
            cls in GP_CLASSES for cls in self.tol.values()
        )

    def tolerance_class(self, key):
        if key in self.tol:
            return self.tol[key]
        for pattern, cls in self.tol.items():
            if pattern != "default" and fnmatchcase(key, pattern):
                return cls
        return self.tol["default"]

    def tolerance(self, key):
        """``(rtol, atol)`` of the output ``key``."""
        return TOLERANCES[self.tolerance_class(key)]

    def __call__(self, state, seed=DEFAULT_SEED):
        return cast_outputs(self.fn(state, seed))


ORACLES = {}


def oracle(name, tol, gp_keys=()):
    """Register the decorated function as the oracle ``name``."""

    def wrap(fn):
        if name in ORACLES:
            raise ValueError(f"oracle {name!r} registered twice")
        ORACLES[name] = Oracle(name, fn, tol, gp_keys)
        return fn

    return wrap


def gp_condition_bound(gp, logger):
    """A bound on the condition number of the training covariance of any
    GP that the oracles build on a snapshot, from the snapshot's GP and
    log: ``(N * sf2 + noise_max) / noise_min``, with ``N`` the logged
    points, ``sf2`` the kernel's largest value and the noise variances that
    the GP assigns to the logged points, over the hyperparameter samples.
    Every training set of the oracles is a subset of the log."""
    n = logger.X_max_idx + 1
    X, y = logger.X[:n], logger.Y[:n]
    s2 = logger.S[:n] ** 2 if logger.noise_flag else None
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    bound = 0.0
    for hyp in gp.get_hyperparameters(as_array=True):
        sf2 = np.max(np.diag(gp.covariance.compute(hyp[:cov_N], X)))
        noise = gp.noise.compute(hyp[cov_N : cov_N + noise_N], X, y, s2)
        noise = np.broadcast_to(np.ravel(noise), (n,))
        bound = max(bound, (n * sf2 + np.max(noise)) / np.min(noise))
    return float(bound)


def noise_floor_hyperparameters(gp, logger, condition=NOISE_FLOOR_CONDITION):
    """The GP's hyperparameters with the log SD of its constant noise raised,
    where it is lower, to ``0.5 * log(N * sf2 / (condition - 1))``, which
    makes :func:`gp_condition_bound` ``condition`` for a GP whose noise is
    that constant alone (``N`` and ``sf2`` as there)."""
    if gp.noise.parameters[0] != 1:
        raise ValueError("the GP's noise has no constant term to raise")
    n = logger.X_max_idx + 1
    X = logger.X[:n]
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    hyp = np.array(gp.get_hyperparameters(as_array=True), dtype=float)
    for row in hyp:
        sf2 = np.max(np.diag(gp.covariance.compute(row[:cov_N], X)))
        floor = 0.5 * np.log(n * sf2 / (condition - 1.0))
        row[cov_N] = max(row[cov_N], floor)
    return hyp


def case_name(name, view):
    """The name under which the outputs of the oracle ``name`` in the view
    ``view`` are stored: the oracle's name in the view ``"stored"``, else
    ``"<name>@<view>"``."""
    return name if view == "stored" else f"{name}@{view}"


def view_oracles(view):
    """The names of the oracles computed in the view ``view``: every oracle
    in ``"stored"``; elsewhere, those with outputs through the GP's solve,
    but the platform-bound oracles."""
    return [
        name
        for name, orc in ORACLES.items()
        if view == "stored"
        or (name not in PLATFORM_BOUND and orc.has_gp_outputs())
    ]


def oracle_cases(snap, stored_only=True):
    """The cases of the decoded snapshot ``snap``, ``(case, name, view)``:
    each oracle in each view of the snapshot that computes it. With
    ``stored_only``, the cases whose outputs the fixture stores, those of
    the oracles that are not platform-bound."""
    return [
        (case_name(name, view), name, view)
        for view in snapshot_views(snap)
        for name in view_oracles(view)
        if not (stored_only and name in PLATFORM_BOUND)
    ]


def platform_bound(snap, view, name, key):
    """Whether the output ``key`` of the oracle ``name``, in the view
    ``view`` of the decoded snapshot ``snap``, is platform-bound: every
    output of an oracle of ``PLATFORM_BOUND``, and the outputs through the
    GP's solve where the view's ``meta["gp_condition_bound"]`` exceeds
    ``GP_CONDITION_MAX``. The fixtures store the other outputs."""
    if name in PLATFORM_BOUND:
        return True
    bound = snap["meta"]["gp_condition_bound"][view]
    return bound > GP_CONDITION_MAX and ORACLES[name].depends_on_gp(key)


def portable_outputs(snap, view, name, out):
    """The outputs of ``out`` (the oracle ``name`` in the view ``view``)
    that are not platform-bound."""
    return {
        k: v for k, v in out.items() if not platform_bound(snap, view, name, k)
    }


def compare(reference, output, tolerance):
    """Compare two dicts of output arrays; returns one row ``(key,
    max_abs_err, max_scaled_err, ok)`` per key.

    ``tolerance`` is ``(rtol, atol)`` or a callable ``key -> (rtol,
    atol)``. Per element, ``|out - ref| <= rtol * denom + atol`` with
    ``denom = max(|ref|, floor)`` and ``floor`` the lower quartile of
    ``|ref|`` over the finite entries; NaN and infinite entries must match
    exactly, and so must the shapes and the keys. The floor keeps the rule
    meaningful for quantities that are differences of nearly equal terms
    (a predictive variance near a training point, an LCB near zero), whose
    rounding is set by the scale of the terms rather than by the small
    result, while the per-element denominator keeps every entry
    load-bearing: a global scale would let the entries far below the
    largest change freely. ``max_scaled_err`` is the largest
    ``|out - ref| / denom``.
    """
    rows = []
    for key, ref in reference.items():
        rtol, atol = tolerance(key) if callable(tolerance) else tolerance
        ref = np.asarray(ref, dtype=float)
        if key not in output:
            rows.append((key, np.inf, np.inf, False))
            continue
        out = np.asarray(output[key], dtype=float)
        if out.shape != ref.shape:
            rows.append((key, np.inf, np.inf, False))
            continue
        finite = np.isfinite(ref) & np.isfinite(out)
        pattern_ok = bool(
            np.array_equal(np.isnan(ref), np.isnan(out))
            and np.array_equal(np.isposinf(ref), np.isposinf(out))
            and np.array_equal(np.isneginf(ref), np.isneginf(out))
        )
        if finite.any():
            a = np.abs(ref[finite])
            diff = np.abs(out[finite] - ref[finite])
            floor = max(float(np.quantile(a, 0.25)), np.finfo(float).tiny)
            denom = np.maximum(a, floor)
            abs_err = float(np.max(diff))
            scaled_err = float(np.max(diff / denom))
            within = bool(np.all(diff <= rtol * denom + atol))
        else:
            abs_err, scaled_err, within = 0.0, 0.0, True
        rows.append((key, abs_err, scaled_err, pattern_ok and within))
    for key in output:
        if key not in reference:
            rows.append((key, np.inf, np.inf, False))
    return rows


def format_rows(rows):
    return "\n".join(
        f"  {'ok ' if ok else 'BAD'} {k:28s} max|d| {a:.2e}  scaled {r:.2e}"
        for k, a, r, ok in rows
    )


# --------------------------------------------------------------------------
# Prescribed draws
# --------------------------------------------------------------------------

_SCRIPTED = frozenset(
    {"random", "standard_normal", "normal", "integers", "permutation"}
)
_REFUSED = frozenset(
    name
    for name in dir(np.random.Generator)
    if not name.startswith("_")
    and name not in _SCRIPTED
    and name != "bit_generator"
)
# The kinds of draw in the log of a ScriptedGenerator
DRAW_UNIFORM, DRAW_NORMAL, DRAW_INTEGERS, DRAW_PERMUTATION = 0, 1, 2, 3


def _count(size):
    return 1 if size is None else int(np.prod(size))


def _shaped(values, size):
    return values[0] if size is None else values.reshape(size)


class ScriptedGenerator(np.random.Generator):
    """A ``numpy.random.Generator`` whose draws are prescribed.

    The draws come from the raw 64-bit stream of ``numpy.random.PCG64(seed)``
    by exact arithmetic, so that they are the same on every platform and
    every NumPy version (NumPy keeps a bit generator's stream, not the
    algorithms of a ``Generator``'s distributions): a uniform draw is the
    top 53 bits of a raw word times ``2**-53``; a normal draw is the sum of
    12 uniform draws less 6, summed in a fixed order (Irwin and Hall's
    approximation of a normal draw: the code under test takes any real
    values); ``integers(low, high)`` is ``low + floor(u * (high - low))``;
    and ``permutation`` orders the indices by uniform draws, with a stable
    sort. Each call is logged as a row ``(kind, count, a, b)``
    (:meth:`log_array`). Only ``random``, ``standard_normal``, ``normal``,
    ``integers`` and ``permutation`` are scripted: any other draw raises
    ``NotImplementedError``. ``pybads.rng.get_rng`` returns such a generator
    unchanged, as it returns any ``Generator``.
    """

    def __init__(self, seed):
        super().__init__(np.random.PCG64(seed))
        self._raw = np.random.PCG64(seed)
        self._log = []

    def __getattribute__(self, name):
        if name in _REFUSED:
            raise NotImplementedError(
                f"ScriptedGenerator does not script {name!r}"
            )
        return super().__getattribute__(name)

    def _uniform(self, n):
        raw = np.asarray(self._raw.random_raw(n), dtype=np.uint64)
        return (raw >> np.uint64(11)).astype(np.float64) * 2.0**-53

    def _normal(self, n):
        u = self._uniform(12 * n).reshape(12, n)
        z = u[0].copy()
        for k in range(1, 12):
            z = z + u[k]
        return z - 6.0

    def random(self, size=None, dtype=np.float64, out=None):
        if out is not None or np.dtype(dtype) != np.float64:
            raise NotImplementedError("random: only float64, without out")
        n = _count(size)
        self._log.append((DRAW_UNIFORM, n, 0.0, 1.0))
        return _shaped(self._uniform(n), size)

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        if out is not None or np.dtype(dtype) != np.float64:
            raise NotImplementedError(
                "standard_normal: only float64, without out"
            )
        n = _count(size)
        self._log.append((DRAW_NORMAL, n, 0.0, 1.0))
        return _shaped(self._normal(n), size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if np.ndim(loc) or np.ndim(scale):
            raise NotImplementedError("normal: only scalar loc and scale")
        n = _count(size)
        self._log.append((DRAW_NORMAL, n, float(loc), float(scale)))
        return _shaped(loc + scale * self._normal(n), size)

    def integers(
        self, low, high=None, size=None, dtype=np.int64, endpoint=False
    ):
        if high is None:
            low, high = 0, low
        if float(low) != int(low) or float(high) != int(high):
            raise NotImplementedError("integers: only integral bounds")
        lo, hi = int(low), int(high) + (1 if endpoint else 0)
        if hi <= lo:
            raise ValueError("integers: high <= low")
        n = _count(size)
        self._log.append((DRAW_INTEGERS, n, float(lo), float(hi)))
        values = np.floor(self._uniform(n) * (hi - lo)).astype(np.int64) + lo
        return _shaped(values.astype(dtype), size)

    def permutation(self, x, axis=0):
        if axis != 0:
            raise NotImplementedError("permutation: only axis 0")
        if isinstance(x, (int, np.integer)):
            n, array = int(x), None
        else:
            array = np.asarray(x)
            n = array.shape[0]
        self._log.append((DRAW_PERMUTATION, n, 0.0, 0.0))
        order = np.argsort(self._uniform(n), kind="stable")
        return order if array is None else array[order]

    def log_array(self):
        """The calls so far, one row ``(kind, count, a, b)`` each: kind 0 a
        uniform draw in ``[a, b)``, 1 a normal draw of mean ``a`` and SD
        ``b``, 2 an integer draw in ``[a, b)``, 3 a permutation."""
        return np.array(self._log, dtype=float).reshape(-1, 4)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _patched(module_name, **attributes):
    """Replace attributes of a module for the duration of the block."""
    module = importlib.import_module(module_name)
    saved = {name: getattr(module, name) for name in attributes}
    for name, value in attributes.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _quadratic_score(center, seen):
    """A stand-in for ``acq_fcn_lcb`` in the ES search: a weighted squared
    distance from ``center``, by elementwise arithmetic in a fixed order,
    so that the search's ranking involves no GP and repeats everywhere.
    Records each candidate set in ``seen``."""
    center = np.ravel(center)

    def score(u_new, func_count, gp, sqrt_beta=None):
        u_new = np.asarray(u_new, dtype=float)
        seen.append(u_new.copy())
        z = np.zeros(u_new.shape[0])
        for j in range(u_new.shape[1]):
            d = u_new[:, j] - center[j]
            z = z + (1.0 + j) * (d * d)
        z = z.reshape(-1, 1)
        return z, z.copy(), np.zeros_like(z)

    return score


def _search_stub(kind, record):
    """A stand-in for an ES search class in the hedge, which records its
    construction and returns the incumbent."""

    class Stub:
        def __init__(self, mu, lamb, options_dict, rng=None):
            self.mu, self.lamb = mu, lamb

        def __call__(
            self,
            u,
            lb,
            ub,
            func_logger,
            gp,
            optim_state,
            sum_rule=True,
            non_box_cons=None,
        ):
            record.append((kind, self.mu, self.lamb, float(sum_rule)))
            return np.atleast_2d(u)[0].copy(), 0.0

    return Stub


def logger_rows(X, y, logger):
    """The indices of the function logger's rows that hold the points
    ``X`` with the values ``y``, sorted."""
    n = logger.X_max_idx + 1
    LX, LY = logger.X[:n], logger.Y[:n].ravel()
    rows = []
    for x, v in zip(np.atleast_2d(X), np.ravel(y)):
        match = np.flatnonzero(np.all(LX == x, axis=1) & (LY == v))
        if match.size != 1:
            raise ValueError(f"{match.size} rows of the log hold {x}, {v}")
        rows.append(match[0])
    return np.sort(np.array(rows, dtype=int))


def _context(state):
    """The quantities most oracles read, from a rebuilt state."""
    os_ = state["optim_state"]
    inc = state["incumbent"]
    return types.SimpleNamespace(
        os=os_,
        inc=inc,
        u=np.array(inc["u"], dtype=float),
        fval=float(np.ravel(inc["fval"])[0]),
        lower=inc["lower_bounds"],
        upper=inc["upper_bounds"],
        ms=float(os_["mesh_size"]),
        sms=float(os_["search_mesh_size"]),
        gp=state["gp"],
        fl=state["logger"],
        options=state["options"],
        nbc=state["non_box_cons"],
        X=np.array(state["cand"]["X"], dtype=float),
    )


# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------


@oracle("gp_predict", {"default": "gp_mean", "fs2*": "gp_var"})
def gp_predict(state, seed):
    """The GP's predictive mean and variance at the candidates, without and
    with the observation noise."""
    c = _context(state)
    fmu, fs2 = c.gp.predict(c.X)
    _, fs2_y = c.gp.predict(c.X, add_noise=True)
    return {"fmu": fmu, "fs2": fs2, "fs2_y": fs2_y}


@oracle("acq_lcb", {"default": "gp_var", "f_mu": "gp_mean"})
def acq_lcb(state, seed):
    """``acq_fcn_lcb`` at the candidates, with the default schedule of
    ``sqrt_beta``, a number and a callable."""
    c = _context(state)
    n = c.fl.func_count
    z, f_mu, f_s = acq_fcn_lcb(c.X, n, c.gp)
    z_fixed, _, _ = acq_fcn_lcb(c.X, n, c.gp, 1.5)
    z_callable, _, _ = acq_fcn_lcb(
        c.X, n, c.gp, lambda t, d: np.sqrt(np.log(t + d))
    )
    return {
        "z": z,
        "f_mu": f_mu,
        "f_s": f_s,
        "z_fixed": z_fixed,
        "z_callable": z_callable,
    }


def _transform_probes(os_):
    """Points in the original space beside the logged ones: the plausible
    box's corners and centre, and points beyond each finite bound, which
    the transform clips."""
    lb, ub = os_["lb_orig"], os_["ub_orig"]
    plb, pub = os_["plb_orig"], os_["pub_orig"]
    width = pub - plb
    below = np.where(np.isfinite(lb), lb - 0.1 * width, plb - 10 * width)
    above = np.where(np.isfinite(ub), ub + 0.1 * width, pub + 10 * width)
    return np.vstack([plb, pub, 0.5 * (plb + pub), below, above])


@oracle("transform", {"default": "gp_free", "apply_log_t": "exact"})
def transform(state, seed):
    """The ``VariableTransformer`` of the run, forward and inverse, on the
    logged points and on probes, and its transformed bounds."""
    c = _context(state)
    vt = state["var_transf"]
    Xo = np.vstack([c.fl.X_orig[c.fl.X_flag], _transform_probes(c.os)])
    U = vt(Xo)
    return {
        "U": U,
        "X_back": vt.inverse_transf(U),
        "lb": vt.lb,
        "ub": vt.ub,
        "plb": vt.plb,
        "pub": vt.pub,
        "apply_log_t": vt.apply_log_t,
    }


@oracle(
    "grid_functions",
    {"default": "gp_free", "force_to_grid*": "exact", "ucov*": "linalg"},
)
def grid_functions(state, seed):
    """``force_to_grid`` (at the candidates, and at halves of the grid
    step, which it takes away from zero), ``udist`` (from the incumbent,
    and from the first training inputs to all), ``grid_units`` (both ways)
    and ``ucov`` (weighted and not)."""
    c = _context(state)
    vt = state["var_transf"]
    D = c.X.shape[1]
    halves = np.outer(np.arange(-3, 3) + 0.5, np.ones(D)) * c.ms
    periodic = c.os["periodic_vars"]
    len_scale = c.gp.temporary_data["len_scale"]
    Xo = c.fl.X_orig[c.fl.X_flag]
    k = min(8, c.gp.X.shape[0])
    weights = np.linspace(1.0, 0.1, k)
    weights = weights / np.sum(weights)
    return {
        "force_to_grid": force_to_grid(c.X, c.sms),
        "force_to_grid_tol": force_to_grid(c.X, c.sms, tol=c.ms),
        "force_to_grid_halves": force_to_grid(halves, c.sms, tol=c.ms),
        "udist": udist(
            c.X,
            c.u,
            len_scale,
            c.os["lb"],
            c.os["ub"],
            c.os["scale"],
            periodic,
        ),
        "udist_train": udist(
            c.gp.X[:k],
            c.gp.X,
            1,
            c.os["lb"],
            c.os["ub"],
            c.os["scale"],
            periodic,
        ),
        "grid_units": grid_units(c.X, x0=c.u, scale=c.os["scale"]),
        "grid_units_transf": grid_units(Xo, var_trans=vt),
        "grid_units_transf_one": grid_units(Xo[:1], var_trans=vt),
        "ucov": ucov(
            c.gp.X[:k],
            c.u,
            weights,
            c.os["ub"],
            c.os["lb"],
            c.os["scale"],
            periodic,
        ),
        "ucov_unweighted": ucov(
            c.gp.X[:k],
            c.u,
            np.empty(0),
            c.os["ub"],
            c.os["lb"],
            c.os["scale"],
            periodic,
        ),
    }


@oracle("contraints_check", "exact")
def constraints_check(state, seed):
    """``contraints_check`` on the candidates put on the search grid, with
    points already evaluated, duplicates and points beyond the bounds:
    projected onto the search bounds or removed beyond the hard bounds,
    with and without the run's non-box constraint, on the run's grid of
    ``tol_mesh`` and on a coarser one."""
    c = _context(state)
    n = c.fl.X_max_idx + 1
    X = force_to_grid(c.X, c.sms)
    lbs, ubs = c.os["lb_search"], c.os["ub_search"]
    beyond = np.vstack(
        [
            c.upper + 0.1,
            c.lower - 0.1,
            np.where(np.arange(X.shape[1]) % 2 == 0, c.upper + 0.05, c.u),
        ]
    )
    U = np.vstack([X, c.fl.X[:n][:: max(1, n // 4)], X[:3], beyond])
    tol = c.os["tol_mesh"]
    return {
        "proj": contraints_check(U, lbs, ubs, tol, c.fl, True, c.nbc),
        "no_proj": contraints_check(
            U, c.lower, c.upper, tol, c.fl, False, c.nbc
        ),
        "proj_no_cons": contraints_check(U, lbs, ubs, tol, c.fl, True, None),
        "coarse": contraints_check(U, lbs, ubs, c.ms, c.fl, True, c.nbc),
    }


@oracle("gp_hyp", {"default": "gp_free", "*gp_s_N": "exact"})
def gp_hyp(state, seed):
    """``_gp_hyp``'s starting hyperparameters, bounds and priors for a new
    GP on the snapshot's training set, on one point, on one point repeated
    with two values (one distinct point: MATLAB BADS's definition values)
    and on two points."""
    c = _context(state)
    X, y = c.gp.X, c.gp.y
    sets = {
        "train": (X, y),
        "one": (X[:1], y[:1]),
        "one_repeated": (X[[0, 0]], y[[0, 0]] + np.array([[0.0], [0.5]])),
        "two": (X[:2], y[:2]),
    }
    out = {}
    for name, (Xs, ys) in sets.items():
        gp = new_gp(
            X.shape[1],
            c.os["gp_cov_fun"],
            c.os["gp_mean_fun"],
            c.os["gp_noisefun"],
        )
        gp, hyp0, gp_s_N = _gp_hyp(
            c.os, c.options, c.os["plb"], c.os["pub"], gp, Xs, ys, c.fl
        )
        out[f"{name}_hyp0"] = hyp0
        out[f"{name}_gp_s_N"] = gp_s_N
        out[f"{name}_lower_bounds"] = gp.lower_bounds
        out[f"{name}_upper_bounds"] = gp.upper_bounds
        for key in ("mu", "sigma", "df"):
            out[f"{name}_prior_{key}"] = gp.hyper_priors[key]
    return out


def training_set_cases(state):
    """The cases of the ``gp_training_set`` oracle, ``(name, center,
    options)``: around the incumbent and around the worst logged point, with
    the run's options and with a reduced ``n_train_max`` (the snapshot's
    ``inputs["small_n_train_max"]``, about 24) and ``n_train_min`` half of
    it, which make the choice of the nearest points bind on the snapshots'
    logs (at the defaults, it takes 50 points or more)."""
    c = _context(state)
    n = c.fl.X_max_idx + 1
    worst = c.fl.X[:n][np.argmax(c.fl.Y[:n].ravel())]
    small = dict(c.options)
    small["n_train_max"] = int(state["inputs"]["small_n_train_max"])
    small["n_train_min"] = small["n_train_max"] // 2
    return [
        (f"{size}{name}", center, options)
        for size, options in (("", c.options), ("small_", small))
        for name, center in (("incumbent", c.u), ("worst", worst))
    ]


@oracle(
    "gp_training_set",
    {
        "default": "gp_free",
        "*_rows": "exact",
        "*_ntrain": "exact",
        "*_exit_flag": "exact",
        "*_fmu": "gp_mean",
        "*_fs2": "gp_var",
    },
)
def gp_training_set(state, seed):
    """``local_gp_fitting`` without a refit, in the cases of
    :func:`training_set_cases`: the rows of the log that it takes as the
    training set, their number, the priors it sets from them, and the GP's
    predictions at the candidates under the snapshot's
    hyperparameters."""
    c = _context(state)
    out = {}
    for name, center, options in training_set_cases(state):
        gp = copy.deepcopy(state["gp"])
        optim_state = copy.deepcopy(c.os)
        gp, exit_flag = local_gp_fitting(
            gp, center, c.fl, options, optim_state, None, False
        )
        fmu, fs2 = gp.predict(c.X)
        out[f"{name}_rows"] = logger_rows(gp.X, gp.y, c.fl)
        out[f"{name}_ntrain"] = optim_state["ntrain"]
        out[f"{name}_exit_flag"] = exit_flag
        out[f"{name}_prior_mu"] = gp.hyper_priors["mu"]
        out[f"{name}_prior_sigma"] = gp.hyper_priors["sigma"]
        out[f"{name}_fmu"] = fmu
        out[f"{name}_fs2"] = fs2
    return out


# The selection masks of the ES search, for these (mu, lambda); the default
# options' pair is added from the snapshot's options
_SELECTION_PAIRS = ((32, 32), (7, 32), (50, 20), (1, 5))


@oracle(
    "es_setup",
    {"default": "exact", "wcm_*": "linalg", "ell_sqrt_sigma": "gp_free"},
)
def es_setup(state, seed):
    """The ES search's set-up: its generation weights and the selection
    mask of ``_get_selection_idx_mask_``, ES-wcm's covariance
    (``S.T @ S`` for its square root ``S``, in which the signs of the
    eigenvectors cancel, and its eigenvalues, with either normalization)
    and ES-ell's square root of its covariance."""
    c = _context(state)
    opts = c.options
    mu = int(opts["n_search"] / opts["n_search_iter"])
    out = {}
    for m, lamb in _SELECTION_PAIRS + ((mu, mu),):
        search = ESSearchELL(m, lamb, opts, rng=ScriptedGenerator(seed))
        out[f"mask_{m}_{lamb}"] = search._get_selection_idx_mask_(m, lamb)
        out[f"vec_{m}"] = search.vec.ravel()
        out[f"ns_{m}"] = search.ns
    wcm = ESSearchWM(mu, mu, opts, rng=ScriptedGenerator(seed))
    for sum_rule in (True, False):
        S = wcm._initialize_(c.u, c.gp, c.os, sum_rule)
        out[f"wcm_cov_{int(sum_rule)}"] = S.T @ S
        out[f"wcm_eig_{int(sum_rule)}"] = np.sum(S * S, axis=1)
    ell = ESSearchELL(mu, mu, opts, rng=ScriptedGenerator(seed))
    out["ell_sqrt_sigma"] = ell._initialize_(c.u, c.gp, c.os, True)
    return out


@oracle("es_generations", {"default": "exact", "*_scale": "gp_free"})
def es_generations(state, seed):
    """ES-ell's search with prescribed draws and a GP-free score in place
    of the LCB, with the default two generations and with three (where the
    scale adapts), on populations of 32: the candidate sets of each
    generation, after the grid and the constraints, the point returned,
    the final scale and the draws."""
    c = _context(state)
    center = c.u.ravel() + 0.37 * c.ms
    out = {}
    for name, n_iter in (("iter2", 2), ("iter3", 3)):
        opts = dict(c.options)
        opts["n_search_iter"] = n_iter
        rng = ScriptedGenerator(seed)
        search = ESSearchELL(32, 32, opts, rng=rng)
        seen = []
        with _patched(
            "pybads.search.es_search",
            acq_fcn_lcb=_quadratic_score(center, seen),
        ):
            us, z = search(c.u, c.lower, c.upper, c.fl, c.gp, c.os, 1, c.nbc)
        D = c.u.size
        out[f"{name}_candidates"] = (
            np.vstack(seen) if seen else np.empty((0, D))
        )
        out[f"{name}_sizes"] = [s.shape[0] for s in seen]
        out[f"{name}_u"] = us
        out[f"{name}_z"] = z
        out[f"{name}_scale"] = search.scale
        out[f"{name}_draws"] = rng.log_array()
    return out


@oracle(
    "hedge",
    {
        "default": "gp_free",
        "*chosen": "exact",
        "*searches": "exact",
        "*draws": "exact",
        "gamma0_*": "gp_var",
    },
    gp_keys=("gamma0_*",),
)
def hedge(state, seed):
    """``ESSearchHedge``: its choice of a search (the probabilities, the
    choice with a prescribed uniform draw, the search it runs, replaced by
    a stub) from several gains, then choices each followed by
    ``update_hedge`` with a search point (a success, failures with and
    without an SD, no point, no estimate); with the default ``hedge_gamma``
    and with 0, where the searches not chosen are scored by the GP."""
    c = _context(state)
    g0 = (
        np.array(state["hedge"]["g"], dtype=float)
        if state["hedge"] is not None
        else np.array([10.0, 0.0])
    )
    gains = [g0, [0.0, 0.0], [2.5, -1.0], [-3.0, 4.0], [1e3, 0.0]]
    step = np.atleast_2d(c.u).copy()
    step[0, 0] += c.ms
    step = force_to_grid(step, c.sms)
    s = max(abs(c.fval), 1.0)
    cases = [
        (step, c.fval - 0.5 * s, 0.3 * s),
        (step, c.fval + 0.2 * s, 0.0),
        (step, c.fval - 0.1 * s, 0.0),
        (None, c.fval, 0.0),
        (step, np.nan, np.nan),
        (step, c.fval - s, 2.0 * s),
    ]
    out = {}
    for label, gamma in (("", None), ("gamma0_", 0.0)):
        opts = dict(c.options)
        if gamma is not None:
            opts["hedge_gamma"] = gamma
        rng = ScriptedGenerator(seed)
        h = ESSearchHedge(opts["search_method"], opts, c.nbc, rng=rng)
        record, probs, chosen, phats, gs = [], [], [], [], []

        def choose():
            h(c.u, c.lower, c.upper, c.fl, c.gp, c.os)
            probs.append(h.prob.copy())
            chosen.append(h.chosen_hedge.item())
            phats.append(h.phat.copy())

        with _patched(
            "pybads.search.search_hedge",
            ESSearchWM=_search_stub(0, record),
            ESSearchELL=_search_stub(1, record),
        ):
            for g in gains:
                h.g = np.array(g, dtype=float)
                choose()
            h.g = g0.copy()
            for u_search, f, fs in cases:
                choose()
                h.update_hedge(u_search, c.fval, f, fs, c.gp, c.ms)
                gs.append(h.g.copy())
        out[f"{label}prob"] = np.array(probs)
        out[f"{label}chosen"] = np.array(chosen)
        out[f"{label}phat"] = np.array(phats)
        out[f"{label}g"] = np.array(gs)
        out[f"{label}searches"] = np.array(record, dtype=float)
        out[f"{label}draws"] = rng.log_array()
    return out


# Improvements (in units of max(|fval|, 1)) and the SDs of the incumbent and
# of the new point, for the improvement oracle
_IMPROVEMENT_D = np.array([-2.0, -0.7, -0.1, -1e-3, 0.0, 1e-3, 0.05, 0.3, 4.0])
_IMPROVEMENT_S_BASE = np.array([0.0, 0.1, 0.2, 0.05, 1.0, 0.3, 0.0, 0.7, 2.0])
_IMPROVEMENT_S_NEW = np.array([0.3, 0.0, 0.05, 0.02, 0.5, 0.0, 0.0, 1.2, 0.1])


def improvement_inputs(state):
    """The inputs of the improvement oracle: ``(f_base, f_new, s_base,
    s_new)`` arrays around the incumbent's value, and the frame sizes."""
    c = _context(state)
    s = max(abs(c.fval), 1.0)
    f_base = np.full(_IMPROVEMENT_D.size, c.fval)
    f_new = f_base - s * _IMPROVEMENT_D
    return (
        (f_base, f_new, s * _IMPROVEMENT_S_BASE, s * _IMPROVEMENT_S_NEW),
        (c.ms, 0.5, 1.0),
    )


@oracle("improvement", {"default": "gp_free", "sto_flags": "exact"})
def improvement(state, seed):
    """``BADS._eval_improvement_`` at several quantiles and without SDs,
    and the outcome of ``BADS._sto_success_improvement_`` (1, 0 or -1)
    for each pair of values at several frame sizes, with the default
    ``gamma_uncertain_interval`` and 1.5, and for a NaN value and an
    infinite SD."""
    (f_base, f_new, s_base, s_new), frames = improvement_inputs(state)
    out = {
        "z_nosd": BADS._eval_improvement_(None, f_base, f_new, None, None, 0.5)
    }
    for q in (50, 20, 90):
        out[f"z_q{q}"] = BADS._eval_improvement_(
            None, f_base, f_new, s_base, s_new, q / 100
        )
    flags = []
    for gamma in (None, 1.5):
        stand_in = types.SimpleNamespace(
            gamma_uncertain_interval=gamma, options=state["options"]
        )
        sto = BADS._sto_success_improvement_
        for frame in frames:
            for i in range(f_base.size):
                flags.append(
                    sto(
                        stand_in,
                        f_base[i],
                        f_new[i],
                        s_base[i],
                        s_new[i],
                        frame,
                    )
                )
            flags.append(sto(stand_in, f_base[0], np.nan, 0.1, 0.1, frame))
            flags.append(
                sto(stand_in, f_base[0], f_new[0], np.inf, 0.1, frame)
            )
    out["sto_flags"] = np.array(flags)
    return out


@oracle("poll_mads_2n", "exact")
def poll(state, seed):
    """``poll_mads_2n`` with prescribed draws, at the snapshot's mesh sizes
    and with the search mesh 4 and 16 times the poll mesh, where the basis
    is LTMADS's lower-triangular one: the directions and the draws."""
    c = _context(state)
    poll_scale = c.gp.temporary_data["poll_scale"]
    D = c.u.size
    out = {}
    variants = (("state", c.sms), ("nmax4", 4 * c.ms), ("nmax16", 16 * c.ms))
    for i, (name, search_mesh_size) in enumerate(variants):
        rng = ScriptedGenerator(seed + i)
        out[f"{name}_B"] = poll_mads_2n(
            D, poll_scale, search_mesh_size, c.ms, rng=rng
        )
        out[f"{name}_draws"] = rng.log_array()
    return out


@oracle("es_search_step", "exact")
def es_search_step(state, seed):
    """Platform-bound. The hedge's search step with the LCB, the default
    options and draws from ``default_rng(seed)``, from the snapshot's
    gains, and each ES search alone: the point returned and its LCB."""
    c = _context(state)
    opts = c.options
    h = ESSearchHedge(
        opts["search_method"], opts, c.nbc, rng=np.random.default_rng(seed)
    )
    if state["hedge"] is not None:
        h.g = np.array(state["hedge"]["g"], dtype=float)
        h.count = int(state["hedge"]["count"])
    us, z = h(c.u, c.lower, c.upper, c.fl, c.gp, c.os)
    out = {"hedge_u": us, "hedge_z": z, "hedge_chosen": h.chosen_hedge}
    for name, cls in (("wcm", ESSearchWM), ("ell", ESSearchELL)):
        search = cls(h.mu, h.lamb, opts, rng=np.random.default_rng(seed))
        us, z = search(c.u, c.lower, c.upper, c.fl, c.gp, c.os, 1, c.nbc)
        out[f"{name}_u"] = us
        out[f"{name}_z"] = z
    return out


@oracle("gp_refit", "exact")
def gp_refit(state, seed):
    """Platform-bound. ``local_gp_fitting`` with a refit around the
    incumbent, drawing from ``default_rng(seed)``: the hyperparameters and
    the geometry it sets."""
    c = _context(state)
    gp = copy.deepcopy(state["gp"])
    optim_state = copy.deepcopy(c.os)
    history = IterationHistory(["init_N", "ntrain"])
    gp, exit_flag = local_gp_fitting(
        gp,
        c.u,
        c.fl,
        c.options,
        optim_state,
        history,
        True,
        rng=np.random.default_rng(seed),
    )
    td = gp.temporary_data
    return {
        "hyp": gp.get_hyperparameters(as_array=True),
        "exit_flag": exit_flag,
        "len_scale": td["len_scale"],
        "poll_scale": td["poll_scale"],
        "effective_radius": td["effective_radius"],
    }
