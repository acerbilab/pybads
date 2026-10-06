import copy
import logging
import math
import os
import sys

import numpy as np
from gpyreg.gaussian_process import GP
from scipy.special import erfc, erfcinv
from scipy.stats import chi2, shapiro

from pybads.acquisition_functions import acq_fcn_lcb, check_sqrt_beta
from pybads.function_logger import FunctionLogger, contraints_check
from pybads.init_functions import init_sobol
from pybads.poll import poll_mads_2n
from pybads.rng import get_rng
from pybads.search import ESSearchHedge
from pybads.search.grid_functions import (
    force_to_grid,
    force_to_grid_periodic,
    grid_units,
    udist,
)
from pybads.utils import period_check
from pybads.utils.iteration_history import IterationHistory
from pybads.utils.timer import Timer
from pybads.utils.timer.stage_timer import NULL_STAGE_TIMER, StageTimer
from pybads.variable_transformer import VariableTransformer

from ._release_reminder import consider_release_reminder
from ._runtime_tips import consider_runtime_tip
from .gaussian_process_train import (
    add_and_update_gp,
    init_and_train_gp,
    local_gp_fitting,
)
from .optimize_result import OptimizeResult
from .options import Options


def _is_real(value):
    """Whether ``value`` is a Python or NumPy integer or float that is not a
    boolean."""
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, float, np.integer, np.floating)
    )


def _as_real_number(value):
    """Return ``value`` as a float if it is a real number (``_is_real``),
    and None otherwise: for a string, a complex number, an array (of one
    element too), a ``Decimal`` or a ``Fraction``, or an integer too large
    for a float."""
    if not _is_real(value):
        return None
    try:
        return float(value)
    except OverflowError:
        return None


def _is_whole_number(value):
    """Whether ``value``, a real number (``_is_real``), is a whole number: an
    integer of any size, or a finite float without a fraction. It does not
    call ``np.isfinite``, which refuses a Python integer beyond 64 bits."""
    if isinstance(value, (int, np.integer)):
        return True
    return math.isfinite(value) and float(value).is_integer()


def _as_limit(value):
    """Return ``value``, a limit of the run such as ``max_fun_evals``, as an
    int if it is a positive whole number (``_is_whole_number``), as inf if it
    is inf or a whole number beyond NumPy's 64-bit integers, which the run's
    NumPy arithmetic does not take, and None otherwise."""
    if not (
        _is_real(value)
        and value > 0
        and (_is_whole_number(value) or value == np.inf)
    ):
        return None
    if value == np.inf:
        return np.inf
    value = int(value)
    return np.inf if value > np.iinfo(np.int64).max else value


def _name_among(value, names):
    """The name among ``names`` that ``value`` is, compared as the searches
    compare it (``value == name``): a string, a NumPy string or an array of
    one; None for anything else, an array of several elements included."""
    for name in names:
        try:
            if np.size(value) == 1 and bool(np.all(value == name)):
                return name
        except (TypeError, ValueError):
            pass
    return None


def _is_named_pair(value, names):
    """Whether ``value`` has two elements, the first of them a name among
    ``names`` (``_name_among``), as a list or a tuple, or a NumPy array,
    has them."""
    try:
        return len(value) == 2 and _name_among(value[0], names) is not None
    except (TypeError, KeyError, IndexError):
        return False


def _precomputed_array(value, shape, name):
    """A float64 copy of ``value``, one of the arrays of
    ``precomputed_evaluations``, refused unless it has ``shape``, where
    ``None`` takes any length, and finite real values: of a boolean,
    integer or floating type, as the function logger takes a value."""
    expected = "({})".format(
        ", ".join("N" if n is None else str(n) for n in shape)
        + ("," if len(shape) == 1 else "")
    )
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as err:  # A ragged sequence
        raise ValueError(
            f"The {name} of precomputed_evaluations must be an array of "
            f"shape {expected}."
        ) from err
    if array.ndim != len(shape) or any(
        n is not None and n != m for n, m in zip(shape, array.shape)
    ):
        raise ValueError(
            f"The {name} of precomputed_evaluations must have shape "
            f"{expected}, not {array.shape}."
        )
    if array.dtype.kind not in "biuf":
        raise ValueError(
            f"The {name} of precomputed_evaluations must be real numbers, "
            f"not of type {array.dtype}."
        )
    array = array.astype(np.float64)
    finite = np.isfinite(array)
    if array.ndim > 1:
        finite = np.all(finite, axis=1)
    if not np.all(finite):
        raise ValueError(
            f"The {name} of precomputed_evaluations must be finite; rows "
            f"{np.flatnonzero(~finite).tolist()} are not."
        )
    return array


# Two values of a point given twice in precomputed_evaluations, without
# uncertainty handling, agree within this many spacings of float64 at their
# scale, as in PyVBMC
_PRECOMPUTED_DUPLICATE_ULPS = 4


def _precomputed_values_agree(first, second):
    """Whether two values of one point of ``precomputed_evaluations`` agree
    within ``_PRECOMPUTED_DUPLICATE_ULPS`` spacings of float64 at the scale
    of the larger, and of 1 below it."""
    scale = max(1.0, abs(first), abs(second))
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = np.spacing(scale)
        if not np.isfinite(spacing):  # At the largest float64
            spacing = scale - np.nextafter(scale, 0.0)
        return abs(first - second) <= _PRECOMPUTED_DUPLICATE_ULPS * spacing


def _bounds_as_rows(x0, lb, ub, plb, pub, x0_source=None):
    """
    ``x0`` and the bounds as float arrays of shape ``(1, D)``, ``D`` the
    number of elements of ``x0``: a scalar bound, or an array of one element,
    is replicated in each dimension, as MATLAB BADS does
    (``boundscheck.m``); floats, as MATLAB's doubles. ``x0_source`` names
    the bounds that sized ``x0`` when the user gave none, for the message
    of a bound of another size.

    Raises
    ------
    ValueError
        When ``x0`` has no element, when a bound is neither a scalar nor
        an array of ``D`` elements, when ``x0`` has more than one row, or
        when an input is not real.
    """
    N0, D = x0.shape
    if x0.size == 0:
        raise ValueError(
            "The starting point x0 (or, without it, the bounds that give "
            "its size) needs at least one element."
        )
    names = (
        "lower_bounds",
        "upper_bounds",
        "plausible_lower_bounds",
        "plausible_upper_bounds",
    )
    shapes = tuple(np.shape(bound) for bound in (lb, ub, plb, pub))
    # A single starting point, as in MATLAB BADS (boundscheck.m); without
    # x0, its shape is that of the bounds named by x0_source
    if N0 > 1:
        if x0_source is not None:
            raise ValueError(
                f"{x0_source} needs to be a scalar or a one-dimensional "
                "array, one element per variable; its shape is "
                f"{dict(zip(names, shapes))[x0_source]}."
            )
        raise ValueError(
            "The starting point x0 needs to be a single point, a "
            f"one-dimensional array with one element per variable; x0 has "
            f"{N0} rows."
        )
    lb, ub, plb, pub = (
        np.full((1, D), bound) if bound.size == 1 else bound
        for bound in map(np.atleast_2d, (lb, ub, plb, pub))
    )
    # check that all bounds are row vectors with D elements; D is the size
    # of x0, or, without it, of the bounds named by x0_source
    for name, bound, shape in zip(names, (lb, ub, plb, pub), shapes):
        if bound.shape != (1, D):
            source = "x0" if x0_source is None else x0_source
            raise ValueError(
                f"{name} needs to be a scalar or a one-dimensional array of "
                f"D = {D} elements, one per variable, where D is the size of "
                f"{source}; its shape is {shape}."
            )

    # Test that all vectors are real-valued
    if not all(np.all(np.isreal(a)) for a in (x0, lb, ub, plb, pub)):
        raise ValueError(
            "x0, lower_bounds, upper_bounds, plausible_lower_bounds and "
            "plausible_upper_bounds need to be real-valued."
        )
    return tuple(np.asarray(a, dtype=float) for a in (x0, lb, ub, plb, pub))


def _find_fixed_values(x0, lb, ub, plb, pub):
    """
    The values of the fixed variables, those whose four bounds are equal
    (and finite), as the ``fixed_values`` of ``VariableTransformer``: an
    array of shape ``(1, D)``, NaN at the other variables, all NaN when no
    variable is fixed. The inputs are rows of ``D`` floats
    (``_bounds_as_rows``). ``x0`` at a fixed variable is its value, or not
    finite, which stands for it; MATLAB BADS fixes a variable only where
    ``x0`` equals its bounds (``boundscheck.m:39-40``).

    Raises
    ------
    ValueError
        When ``x0`` is finite at a fixed variable and differs from its
        value, or when every variable is fixed.
    """
    fixed = (lb == ub) & (ub == plb) & (plb == pub) & np.isfinite(lb)
    moved = fixed & np.isfinite(x0) & (x0 != lb)
    if np.any(moved):
        raise ValueError(
            "The bounds of the variables (index) "
            f"{np.flatnonzero(moved).tolist()} are all equal, which fixes "
            "each of them at its bound, but x0 differs from it there; set "
            "x0 to the bound (or to NaN) at a fixed variable."
        )
    if np.all(fixed):
        raise ValueError(
            "The four bounds of every variable are "
            "equal, which fixes them all: there is nothing to optimize."
        )
    return np.where(fixed, lb, np.nan)


def _run_indices(indices, fixed_values):
    """The indices among the run's variables, all but the fixed ones, of the
    variables at ``indices`` among all of them; a fixed variable has none,
    and is left out. ``fixed_values`` is ``_find_fixed_values``'s row."""
    free = np.isnan(fixed_values[0])
    run_index = np.cumsum(free) - 1
    return [int(run_index[i]) for i in indices if free[i]]


def _user_indices(indices, fixed_values):
    """The indices among all the variables of the run's variables at
    ``indices``, the inverse of ``_run_indices``."""
    free = np.flatnonzero(np.isnan(fixed_values[0]))
    return free[np.asarray(indices, dtype=int)].tolist()


# The levels of the BADS logger's messages above the iteration lines (INFO),
# for MATLAB BADS's display levels: the opening message, the reports of the
# setup and the message of a random starting point from "notify" on, the
# final message from "final" on
_LOG_NOTIFY = 25
_LOG_FINAL = 22

_PB_UNSPECIFIED = (
    "Plausible lower/upper bounds not specified. Using "
    "hard upper/lower bounds instead."
)

# The search's acquisition functions (the first element of search_acq_fcn)
# that read the optimization target, for which the search computes it.
# MATLAB BADS computes it at every search (bads.m:539), where the functions
# that read it are acqNegEI, acqNegPI, acqNegEQI and acqNegSqEI, and
# acqNegEIMin and acqNegPIMin, whose target searchES.m also updates; none of
# them is ported. The LCB, the only one that search_acq_fcn takes, does not
# read it, and the poll computes its own
_SEARCH_ACQ_FCNS_READING_TARGET = frozenset()


class BADS:
    r"""
    BADS Constrained optimization using Bayesian Adaptive Direct Search.

    BADS attempts to solve problems of the form:
       :math:`\mathtt{argmin}_x  f(x)`  subject to:  lower_bounds :math:`<= x <=` upper_bounds, and optionally :math:`C(x) <= 0`


    Initialize a ``BADS`` object to set up the optimization problem, then run
    ``optimize()``. See the examples for more details under the `examples` directory.

    Parameters
    ----------
    fun : callable
        The target function to minimize. ``fun`` takes a point ``x``, a
        one-dimensional array of all the variables (the fixed ones included),
        and returns its value as a finite real number; with
        ``options['specify_target_noise']``, it returns a tuple ``(f, sd)``,
        the value and the SD of its noise at ``x``.
        In case the target function ``fun`` requires additional data/parameters,
        they can be handled using an anonymous function.
        For example: ``fun_for_pybads = lambda x: fun(x, data, extra_params)``,
        where ``fun`` is the function to optimize, and ``data`` and ``extra_params``
        are given in the outer scope.
    x0 : np.ndarray, optional
        Starting point for the optimization, a single point of ``D``
        elements, of shape ``(D,)`` or ``(1, D)``. If not specified or ``None``,
        or if an element is not finite (``nan``, ``inf`` or ``-inf``) at a
        variable that is not fixed (see below), the starting point ``x0`` is
        drawn at random, over the variables that are not fixed, inside the
        plausible box
        between ``plausible_lower_bounds`` and ``plausible_upper_bounds`` (see
        below). With ``non_box_cons``, a point that violates the constraints
        is drawn again, up to 1000 draws in all.
    lower_bounds, upper_bounds : np.ndarray, optional
        ``lower_bounds`` (``lb``) and ``upper_bounds`` (``ub``) define a set
        of strict lower and upper bounds for the coordinate vector, ``x``, so
        that the unknown function has support on ``lb`` <= ``x`` <= ``ub``.
        If scalars, the bound is replicated in each dimension. Use
        ``None`` for ``lb`` and ``ub`` if no bounds exist. Set
        ``lb[i] = -inf`` if the `i`-th coordinate is unbounded below, and
        ``ub[i] = inf`` if it is unbounded above (while other coordinates
        may be bounded). Note that if ``lb`` and/or ``ub`` contain infinite
        bounds, the respective values of ``plb`` and/or ``pub`` need to be
        specified (see below). By default ``None``.
    plausible_lower_bounds, plausible_upper_bounds : np.ndarray, optional
        Specifies a set of ``plausible_lower_bounds`` (``plb``) and
        ``plausible_upper_bounds`` (``pub``) such that ``lb`` <= ``plb`` < ``pub`` <= ``ub``
        at each variable that is not fixed (see below).
        Both ``plb`` and ``pub`` need to be finite, and are replicated in
        each dimension if scalars. If not specified, ``plb`` is ``lb`` and
        ``pub`` is ``ub``: ``plb`` needs to be specified when ``lb`` has an
        infinite bound, and ``pub`` when ``ub`` has one. ``plb`` and ``pub``
        represent a `plausible` range, which should denote a region where the
        global minimum is expected to be found. As a rule of thumb, set ``plausible_lower_bounds``
        and ``plausible_upper_bounds`` such that there is > 90% probability that
        the minimum is found within the box (where in doubt, just set
        ``plb = lb`` and ``pub = ub``).

        A variable whose four bounds are equal is fixed at their value;
        ``x0`` holds that value there, or a non-finite value. BADS optimizes
        the other variables, and the defaults of the options that depend on
        the number of variables count only those. The points you pass to
        BADS or receive from it hold all the variables, the fixed ones at
        their values: those that ``fun``, ``non_box_cons`` and
        ``options['output_fcn']`` receive, the result's ``x`` and ``x0``,
        the points ``"x"`` of ``iteration_history`` and the points of
        ``precomputed_evaluations``; the indices of
        ``options['periodic_vars']`` count all the variables.

    non_box_cons : callable, optional
        A given non-box constraints function that specifies constraint
        `violations`. It takes an array of shape ``(N, D)``, one point per
        row in the original space, and returns an array of shape ``(N,)``
        or ``(N, 1)``, one value per point, true or positive where the point
        violates the constraints. For example,
        ``lambda x: np.sum(x**2, axis=1) > 1`` keeps the search inside the
        unit ball. A feasible region thinner than the mesh can resolve, such
        as a band narrower than the poll steps, can leave every point of the
        initial design and of the polls infeasible, and the run can then end
        early, near ``x0``, on the stall criterion (an improvement below
        ``tol_fun`` over ``tol_stall_iters`` iterations). Reparametrize such a
        problem so that its feasible region is wide: for the band
        ``abs(x[0] - x[1]) <= w``, for example, optimize over
        ``(x[0] + x[1]) / 2`` and ``(x[0] - x[1]) / w``, the latter bounded
        by -1 and 1 in place of the constraint.

    options : dict, optional
        Additional options can be passed as a dict. Please refer to the
        BADS options page for the default options. If no `options` are
        passed, the default options are used.
        To run BADS on a noisy (stochastic) objective function, set
        ``options['uncertainty_handling']`` = ``True``. You can help BADS by
        providing an estimate of the noise. ``options['noise_size'] = sigma`` provides a global estimate of the
        SD of the noise in your problem in a good region of the parameter
        space. (If not specified, default ``sigma = 1.0``).
        Alternatively, you can specify the target noise `at each location`
        with ``options['specify_target_noise']`` = ``True``. In this case,
        ``fun`` is expected to return `two` values, the estimate of the
        target at ``x`` and an estimate of the SD of the noise at ``x``
        (see the examples).
        If ``options['uncertainty_handling']`` is not specified, BADS will
        determine at runtime if the objective function is noisy, or turn
        uncertainty handling on with ``options['specify_target_noise']``
        = ``True``.
        A variable that is periodic, such as an angle, is named in
        ``options['periodic_vars']``, a list of the indices of the
        variables, from 0, the fixed ones included:
        its hard bounds, which need to be finite, are its period, and BADS
        wraps it around them, so that ``lb`` and ``ub`` are the same point
        and the variable is optimized across them.
        To obtain reproducible results of the optimization, set
        ``options['random_seed']`` to a fixed integer (see ``rng`` below).

    gamma_uncertain_interval : float, optional, keyword-only
        With ``options['stobads']``, the multiplier of the half-width of the
        uncertainty interval of the Sto-BADS success rule. By default
        ``None``, which is 1.96. Deprecated along with
        ``options['stobads']``, which is experimental and may be removed in
        a future release.

    precomputed_evaluations : tuple, optional, keyword-only
        Evaluations of ``fun`` made before the run, for instance by an
        earlier run, as a tuple (or a list) ``(X, y)``, or ``(X, y, y_sd)``
        with ``options['specify_target_noise']``, which requires ``y_sd``.
        ``X`` holds one point per row in the original space, of shape ``(N,
        D)``, each within the hard bounds and satisfying ``non_box_cons``;
        ``y`` holds the values of ``fun`` at them and ``y_sd`` the SDs of
        their noise, both of shape ``(N,)``. The values are finite and the
        SDs positive. BADS uses them as training data of its Gaussian
        process, and does not evaluate again the points of its initial
        design that they hold; they do not count toward ``max_fun_evals``
        (``func_count``). The run still starts from ``x0``: its first
        incumbent is the best of the points it evaluates itself (pass the
        best of the evaluations as ``x0`` to start there). Unless
        ``options['uncertainty_handling']`` is ``True`` (or
        ``options['specify_target_noise']`` is), a point given twice must
        have the same value, and is kept once; otherwise each repeat is an
        observation of its own. By default ``None``, no evaluations.

    Attributes
    ----------
    rng : numpy.random.Generator
        The generator of every random draw of the run, including the random
        ``x0``. It is created with the ``BADS`` object from
        ``options['random_seed']``, which takes what
        ``numpy.random.default_rng`` takes, such as a non-negative integer or
        a ``SeedSequence``, or a ``Generator``, used as given. If the option
        is ``None`` (default), the generator is derived from NumPy's global
        random state, so that ``np.random.seed`` before creating the ``BADS``
        object fixes the run. The run does not otherwise draw from NumPy's
        global random state: a target that draws from it is not fixed by
        ``random_seed``. On Apple Silicon Macs, two runs with the same seed
        can end at slightly different points.

    Raises
    ------
    ValueError
        When ``x0`` is not specified and neither ``plausible_lower_bounds``
        nor ``lower_bounds`` is, or neither ``plausible_upper_bounds`` nor
        ``upper_bounds`` is: a missing plausible bound defaults to the hard
        bound, and the random ``x0`` is drawn between the plausible bounds.
    ValueError
        When ``x0`` or the bounds fail the checks of their shapes, values and
        order: for instance an ``x0`` of more than one row, plausible bounds
        that are not finite, bounds out of the order ``lb <= plb < pub <=
        ub`` at a variable that is not fixed, a finite ``x0`` that differs
        from the value of a fixed variable, or bounds that fix every
        variable.
    ValueError
        When ``non_box_cons``, given an ``(N, D)`` array, does not return an
        array of shape ``(N,)`` or ``(N, 1)``, or when ``x0`` violates the
        constraints, or each of the 1000 random draws of it does.
    ValueError
        When an option has an unknown name or a value that BADS does not
        take: for instance a ``max_fun_evals``, ``max_iter`` or
        ``tol_stall_iters`` that is neither a positive integer nor ``inf``,
        a ``tol_mesh`` that is not a positive finite number, a
        ``noise_size`` that is not one or two numbers, a value other than
        ``True`` or ``False`` for ``uncertainty_handling`` or for an option
        whose default is one of them (``plot`` excepted), a
        ``periodic_vars`` that is not a list of distinct indices of
        variables with finite bounds, or an ``f_vals`` that holds a finite
        value, a non-empty ``fun_values``, or ``acq_hedge=True``, options
        that are not supported.
    ValueError
        When ``precomputed_evaluations`` is not a tuple (or a list) of two
        or three arrays of the shapes above, of finite values and positive
        SDs, when it has ``y_sd`` without
        ``options['specify_target_noise']`` or lacks them with it, when one
        of its points lies outside the hard bounds or violates
        ``non_box_cons``, or when a point given twice has two different
        values where a point is kept once.
    ValueError
        When ``options['random_seed']`` is a negative integer.
    TypeError
        When ``options['random_seed']`` is a float that is not a whole
        number, a string, or another value that ``numpy.random.default_rng``
        does not take.


    References
    ----------
    .. [1]  Singh, S. G. & Acerbi, L. (2024).
            "PyBADS: Fast and robust black-box optimization in Python".
            Journal of Open Source Software, 9(94), 5694, https://doi.org/10.21105/joss.05694.

    .. [2]  Acerbi, L. & Ma, W. J. (2017). "Practical Bayesian
            Optimization for Model Fitting with Bayesian Adaptive Direct Search".
            In `Advances in Neural Information Processing Systems` 30, pages 1834-1844.
            (arXiv preprint: https://arxiv.org/abs/1705.04405).

    Examples
    --------
    For `BADS` usage examples, please look up the Jupyter notebook tutorials
    in the PyBADS documentation:
    https://acerbilab.github.io/pybads/examples.html
    """

    def __init__(
        self,
        fun: callable,
        x0: np.ndarray = None,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        non_box_cons: callable = None,
        options: dict = None,
        *,
        gamma_uncertain_interval: float = None,
        precomputed_evaluations: tuple = None,
    ):
        # set up root logger (only changes stuff if not initialized yet)
        logging.basicConfig(stream=sys.stdout, format="%(message)s")

        self.non_box_cons = non_box_cons

        # variable to keep track of logging actions
        self.logging_action = []

        # Initialize variables and algorithm structures. A missing x0 is a
        # random start of the size of the plausible bounds, or else of the
        # hard ones, which the missing plausible bounds are, as in MATLAB
        # BADS: a list or a Python scalar sizes it as an array does
        if x0 is None:
            lb_size = (
                lower_bounds
                if plausible_lower_bounds is None
                else plausible_lower_bounds
            )
            ub_size = (
                upper_bounds
                if plausible_upper_bounds is None
                else plausible_upper_bounds
            )
            if lb_size is None or ub_size is None:
                raise ValueError(
                    "Without x0, give plausible_lower_bounds and "
                    "plausible_upper_bounds, or lower_bounds and "
                    "upper_bounds, which set the number of variables."
                )
            x0 = np.full(np.shape(np.atleast_2d(lb_size)), np.nan)
            x0_source = (
                "lower_bounds"
                if plausible_lower_bounds is None
                else "plausible_lower_bounds"
            )
        else:
            x0_source = None
        x0 = np.atleast_2d(x0)

        # Empty lb and ub are Infs. Missing plausible bounds are the hard
        # bounds, with the warning _PB_UNSPECIFIED once the logger is set
        # up, as MATLAB BADS warns whenever it fills them
        # (boundscheck.m:12-16); a plausible bound that is then infinite is
        # refused by _bounds_check_
        if lower_bounds is None:
            lower_bounds = np.full((1, x0.shape[1]), -np.inf)
        if upper_bounds is None:
            upper_bounds = np.full((1, x0.shape[1]), np.inf)
        pb_filled = plausible_lower_bounds is None or (
            plausible_upper_bounds is None
        )
        if plausible_lower_bounds is None:
            plausible_lower_bounds = np.atleast_2d(lower_bounds).copy()
        if plausible_upper_bounds is None:
            plausible_upper_bounds = np.atleast_2d(upper_bounds).copy()
        x0, lb, ub, plb, pub = _bounds_as_rows(
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
            x0_source=x0_source,
        )

        # Fixed variables, whose four bounds are equal, are left out of the
        # run: D, with which the options are evaluated, counts the others, as
        # in MATLAB BADS. The variable transformer puts the fixed variables
        # back into every point that leaves the run, so that the target,
        # non_box_cons, output_fcn, the function log, the iteration history
        # and the result see points of all the variables; x0 holds them all
        # too, the fixed ones at their values
        self._fixed_values = _find_fixed_values(x0, lb, ub, plb, pub)
        free = np.isnan(self._fixed_values[0])
        x0 = np.where(free, x0, self._fixed_values)
        self.D = int(np.sum(free))

        # load basic and advanced options and validate the names
        pybads_path = os.path.dirname(os.path.realpath(__file__))
        basic_path = pybads_path + "/option_configs/basic_bads_options.ini"
        self.options = Options(
            basic_path,
            evaluation_parameters={"D": self.D},
            user_options=options,
        )
        self._check_tol_fun_()
        advanced_path = (
            pybads_path + "/option_configs/advanced_bads_options.ini"
        )
        self.options.load_options_file(
            advanced_path,
            evaluation_parameters={"D": self.D},
        )
        self.options.validate_option_names([basic_path, advanced_path])
        # uncertainty_handling is None (the default), True or False; plot
        # also takes the names of MATLAB BADS's plots
        self.options.validate_boolean_options(
            [basic_path, advanced_path],
            extra_names=("uncertainty_handling",),
            excluded_names=("plot",),
        )

        # set up the random generator of the run
        self._init_rng_()
        # The stage timer of a run, created by optimize(); the steps called
        # outside it time nothing
        self._stage_timer = NULL_STAGE_TIMER
        # Set by the first call of optimize(): a run rewrites the options and
        # fills the function log, so a BADS object runs once
        self._optimize_called = False

        # set up BADS logger, from the first three letters of the display
        # option, lower case, as in MATLAB BADS (bads.m): "off" and "none"
        # show the warnings only, "notify" (and any other value) also the
        # opening message and the reports of the setup, "final" also the
        # final message, "iter" and "all" also the iteration lines, and
        # "full", PyBADS's own, the debug messages too
        self.logger = logging.getLogger("BADS")
        display = str(self.options.get("display"))[:3].lower()
        if display in ("off", "non"):
            self.logger.setLevel(logging.WARNING)
        elif display == "fin":
            self.logger.setLevel(_LOG_FINAL)
        elif display in ("ite", "all"):
            self.logger.setLevel(logging.INFO)
        elif display == "ful":
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(_LOG_NOTIFY)
        if pb_filled:
            self.logger.warning(_PB_UNSPECIFIED)

        # Check boundaries and starting points
        self._bounds_check_(x0, lb, ub, plb, pub, non_box_cons)
        self.x0 = x0

        self.gamma_uncertain_interval = gamma_uncertain_interval
        # A BADS object considers one runtime tip, when its run starts
        self._runtime_tip_considered = False

        # Checked before _init_optim_state_ transforms the variables, which
        # never takes a periodic variable to log coordinates, and draws a
        # random x0
        self._check_periodic_vars_(lb, ub)

        # The run's bounds are those of the variables it optimizes, and the
        # transform that _init_optim_state_ builds from them takes the fixed
        # variables' values
        self.lower_bounds = lb[:, free]
        self.upper_bounds = ub[:, free]
        self.plausible_lower_bounds = plb[:, free]
        self.plausible_upper_bounds = pub[:, free]
        if not np.all(free):
            self.logger.log(
                _LOG_NOTIFY,
                "Variables (index) fixed at their bounds and left out of the "
                f"optimization: {np.flatnonzero(~free).tolist()}.",
            )

        # evaluate  starting point non-bound constraint (a missing or
        # non-finite start is drawn in _init_optim_state_, where it is put on
        # the mesh)
        if non_box_cons is not None and np.all(np.isfinite(self.x0)):
            if non_box_cons(self.x0) > 0:
                raise ValueError(
                    "x0 violates non_box_cons: pass a starting point that "
                    "satisfies the constraints, or x0=None to draw one at "
                    "random."
                )

        self.optim_state = self._init_optim_state_()

        # create and init the function logger; it holds the noise SDs only
        # when the target returns them, as MATLAB's funlogger does
        self.function_logger = FunctionLogger(
            fun=fun,
            D=self.D,
            noise_flag=self.optim_state.get("uncertainty_handling_level") > 1,
            uncertainty_handling_level=self.optim_state.get(
                "uncertainty_handling_level"
            ),
            cache_size=self.options.get("cache_size"),
            variable_transformer=self.var_transf,
        )
        self._import_precomputed_evaluations_(precomputed_evaluations)

        self.iteration_history = IterationHistory(
            [
                "iter",
                "func_count",
                "u",
                "x",
                "fval",
                "fsd",
                "yval",
                "ys",
                "mesh_size",
                "search_mesh_size",
                "lcbmax",
                "gp",
                "gp_hyp_full",
                "Ns_gp",
                "timer",
                "optim_state",
                "n_eff",
                "ntrain",
                "init_N",
                "logging_action",
            ]
        )

    def _bounds_check_(
        self,
        x0: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        plausible_lower_bounds: np.ndarray,
        plausible_upper_bounds: np.ndarray,
        non_box_cons: callable = None,
    ):
        """
        Check ``x0`` and the bounds, rows of ``D`` floats
        (``_bounds_as_rows``), for all the variables, fixed ones included.
        """
        # check that plausible bounds are finite
        not_finite = ~(
            np.isfinite(plausible_lower_bounds)
            & np.isfinite(plausible_upper_bounds)
        )
        if np.any(not_finite):
            raise ValueError(
                "plausible_lower_bounds and plausible_upper_bounds need to "
                "be finite, and are not at the variables (index) "
                f"{np.flatnonzero(not_finite).tolist()}; where they are not "
                "given, they are lower_bounds and upper_bounds."
            )

        # The fixed variables (_find_fixed_values), whose four bounds are equal,
        # pass the tests of distinct and ordered plausible bounds; any other
        # variable with equal plausible bounds is refused
        fixed = ~np.isnan(self._fixed_values)

        # Test that plausible bounds are different
        matching = (plausible_lower_bounds == plausible_upper_bounds) & ~fixed
        if np.any(matching):
            raise ValueError(
                "plausible_lower_bounds and plausible_upper_bounds are equal "
                "at the variables (index) "
                f"{np.flatnonzero(matching).tolist()}: they need to differ, "
                "except at a fixed variable, whose four bounds are all equal."
            )

        # Check that all X0 are inside the bounds. As in MATLAB BADS
        # (boundscheck.m, setupvars.m), neither x0 nor the plausible bounds
        # are moved: a start on a hard bound or outside the plausible box
        # stays where it is. A start that is not finite passes:
        # _init_optim_state_ draws a random point in its place, as MATLAB
        # BADS does (setupvars.m)
        outside = (x0 < lower_bounds) | (x0 > upper_bounds)
        if np.all(np.isfinite(x0)) and np.any(outside):
            raise ValueError(
                "x0 lies outside lower_bounds and upper_bounds at the "
                f"variables (index) {np.flatnonzero(outside).tolist()}."
            )

        # Test order of bounds, as in MATLAB BADS (setupvars.m)
        ordidx = (
            (lower_bounds <= plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds <= upper_bounds)
        )
        if np.any(np.invert(ordidx | fixed)):
            raise ValueError(
                "The bounds of the variables (index) "
                f"{np.flatnonzero(~(ordidx | fixed)).tolist()} are not in "
                "order: each variable needs lower_bounds <= "
                "plausible_lower_bounds < plausible_upper_bounds <= "
                "upper_bounds."
            )

        # Check non bound constraints: one violation per row of its input,
        # as in MATLAB BADS (setupvars.m)
        if non_box_cons is not None:
            message = (
                "non_box_cons should be a function that takes "
                + "an N x D array X, one point per row, and returns an array "
                + "of N constraint violations, of shape (N,) or (N, 1), true "
                + "or positive where a point violates the constraints."
            )
            try:
                y = non_box_cons(
                    np.vstack([plausible_lower_bounds, plausible_upper_bounds])
                )
            except Exception as err:
                raise ValueError(message) from err
            if not isinstance(y, np.ndarray) or y.shape not in [(2,), (2, 1)]:
                raise ValueError(message)

        # Gentle caution for infinite bounds, as in MATLAB BADS
        # (setupvars.m:28-39), which accepts a variable bounded on one side
        # only and prints it from "notify" on, as the other reports of the
        # setup
        is_inf = np.isinf(np.concatenate([lower_bounds, upper_bounds]))
        ninfs = np.sum(is_inf)
        if ninfs > 0:
            # Every bound of the variables that are not fixed
            if ninfs == 2 * np.sum(~fixed):
                self.logger.log(
                    _LOG_NOTIFY, "Detected fully unconstrained optimization."
                )
            else:
                self.logger.log(
                    _LOG_NOTIFY,
                    f"Detected {ninfs} infinite bound(s), in variables"
                    f" (index) {np.flatnonzero(np.any(is_inf, 0)).tolist()}.",
                )

    def _init_optim_state_(self):
        """
        A private function to initialize the optim_state dict that contains information about BADS variables.
        """
        # f_vals, the function values of the starting points, is not
        # supported: a value without a finite element stands for None
        f_vals = self.options["f_vals"]
        if f_vals is not None:
            try:
                has_values = np.any(np.isfinite(np.asarray(f_vals, float)))
            except (TypeError, ValueError):
                has_values = True
            if has_values:
                raise ValueError(
                    "options['f_vals'] is not supported: leave it None (its "
                    "default)."
                )

        optim_state = dict()
        optim_state["random_seed"] = self._random_seed
        optim_state["last_re_eval"] = -np.inf

        # Grid parameters
        self.mesh_size_integer = self.options[
            "init_mesh_size_integer"
        ]  # Mesh size in log base units
        optim_state["search_size_integer"] = np.minimum(
            0,
            self.mesh_size_integer * self.options.get("search_grid_multiplier")
            - self.options.get("search_grid_number"),
        )
        optim_state["mesh_size"] = (
            float(self.options.get("poll_mesh_multiplier"))
            ** self.mesh_size_integer
        )
        self.mesh_size = optim_state["mesh_size"]
        optim_state["search_mesh_size"] = (
            float(self.options.get("poll_mesh_multiplier"))
            ** optim_state["search_size_integer"]
        )
        self.search_mesh_size = optim_state["search_mesh_size"]
        optim_state["scale"] = 1.0

        # Compute transformation of variables
        self.var_transf = self._variable_transformer_()
        # optim_state["variables_trans"] = var_transf

        # Update the bounds with the new transformed bounds
        self.lower_bounds = self.var_transf.lb.copy()
        self.upper_bounds = self.var_transf.ub.copy()
        optim_state["lb"] = self.lower_bounds.copy()
        optim_state["ub"] = self.upper_bounds.copy()

        # Periodic variables, a (1, D) mask over the run's variables, which
        # leaves out a fixed one; their bounds were checked finite by
        # _check_periodic_vars_
        periodic_vars = np.zeros((1, self.D), dtype=bool)
        if self.options["periodic_vars"] is not None:
            periodic_vars[
                :,
                _run_indices(
                    self.options["periodic_vars"], self._fixed_values
                ),
            ] = True
        optim_state["periodic_vars"] = periodic_vars
        self.plausible_lower_bounds = self.var_transf.plb.copy()
        self.plausible_upper_bounds = self.var_transf.pub.copy()
        optim_state["plb"] = self.plausible_lower_bounds.copy()
        optim_state["pub"] = self.plausible_upper_bounds.copy()

        optim_state["lb_orig"] = self.var_transf.orig_lb.copy()
        optim_state["ub_orig"] = self.var_transf.orig_ub.copy()
        optim_state["plb_orig"] = self.var_transf.orig_plb.copy()
        optim_state["pub_orig"] = self.var_transf.orig_pub.copy()

        # Bounds for search mesh
        lb_search = force_to_grid(
            self.lower_bounds, optim_state["search_mesh_size"]
        )
        lb_search[lb_search < self.lower_bounds] = (
            lb_search[lb_search < self.lower_bounds]
            + optim_state["search_mesh_size"]
        )
        optim_state["lb_search"] = lb_search
        ub_search = force_to_grid(
            self.upper_bounds, optim_state["search_mesh_size"]
        )
        ub_search[ub_search > self.upper_bounds] = (
            ub_search[ub_search > self.upper_bounds]
            - optim_state["search_mesh_size"]
        )
        optim_state["ub_search"] = ub_search

        def start_on_mesh():
            # Starting point in grid coordinates, gridization
            u0 = force_to_grid(
                grid_units(self.x0, self.var_transf, optim_state["scale"]),
                optim_state["search_mesh_size"],
            )
            # Adjust points that fall outside bounds due to gridization
            u0[u0 < self.lower_bounds] = (
                u0[u0 < self.lower_bounds] + optim_state["search_mesh_size"]
            )
            u0[u0 > self.upper_bounds] = (
                u0[u0 > self.upper_bounds] - optim_state["search_mesh_size"]
            )
            # A periodic coordinate on its upper bound is the same point as
            # on its lower bound, where the candidates are wrapped
            return period_check(
                u0,
                self.lower_bounds,
                self.upper_bounds,
                optim_state["periodic_vars"],
            )

        def violates_non_box_cons(u0):
            return self.non_box_cons is not None and np.any(
                self.non_box_cons(self.var_transf.inverse_transf(u0)) > 0
            )

        if not np.all(np.isfinite(self.x0)):
            # A missing or non-finite start is drawn uniformly in the
            # transformed plausible box and put on the mesh, as in MATLAB
            # BADS (setupvars.m:83-85): log-uniform for a log-transformed
            # variable. MATLAB BADS refuses a point on the mesh that violates
            # non_box_cons (evalinitmesh.m:22-26); PyBADS draws again, up to
            # 1000 draws in all, while the draw, which the result reports as
            # x0, or its point on the mesh, the start evaluated, violates it
            for _ in range(1000):
                u_draw = self.rng.uniform(
                    low=self.var_transf.plb,
                    high=self.var_transf.pub,
                    size=(1, self.D),
                )
                self.x0 = self.var_transf.inverse_transf(u_draw)
                u0 = start_on_mesh()
                if not violates_non_box_cons(
                    u_draw
                ) and not violates_non_box_cons(u0):
                    break
            self.logger.log(
                _LOG_NOTIFY,
                "x0 is missing or has a non-finite element: the starting "
                "point is drawn at random in the plausible box.\n",
            )
            if violates_non_box_cons(u_draw) or violates_non_box_cons(u0):
                raise ValueError(
                    "None of 1000 starting points drawn at random in the "
                    "plausible box satisfies non_box_cons: pass an x0 that "
                    "satisfies it, or set plausible_lower_bounds and "
                    "plausible_upper_bounds around the feasible region."
                )
        else:
            u0 = start_on_mesh()

        # Check that the gridized points satisfies the non-bound constraints
        if violates_non_box_cons(u0):
            raise ValueError(
                "x0 is too close to the boundary of the region that "
                "non_box_cons allows: PyBADS starts from the point of its "
                "initial grid nearest to x0, which violates the constraints. "
                "Move x0 further inside the region."
            )

        optim_state["u"] = u0
        self.u = u0.flatten().copy()

        # Test starting point u0 is within bounds
        if np.any(u0 > self.upper_bounds) or np.any(u0 < self.lower_bounds):
            raise ValueError(
                "The start on the initial grid lies outside lower_bounds "
                "and upper_bounds."
            )

        # Report variable transformation, from "notify" on, as MATLAB BADS
        # does (setupvars.m:118-120), with the indices of the variables
        if np.any(self.var_transf.apply_log_t):
            log_vars = np.flatnonzero(self.var_transf.apply_log_t)
            self.logger.log(
                _LOG_NOTIFY,
                "Variables (index) internally transformed to log "
                f"coordinates: {_user_indices(log_vars, self._fixed_values)}.",
            )

        # Report the periodic variables
        if np.any(optim_state["periodic_vars"]):
            periodic_vars = np.flatnonzero(optim_state["periodic_vars"])
            self.logger.log(
                _LOG_NOTIFY,
                "Variables (index) defined with periodic boundaries: "
                f"{_user_indices(periodic_vars, self._fixed_values)}",
            )

        # Setup covariance information (unused)

        # MATLAB BADS's option of the evaluations made before the run
        # (setupvars.m:126-167) is refused: they are the argument
        # precomputed_evaluations, imported by
        # _import_precomputed_evaluations_
        fun_values = self.options["fun_values"]
        if fun_values is not None and len(fun_values) != 0:
            raise ValueError(
                "options['fun_values'] is not supported: leave it empty (its "
                "default, {}), and pass evaluations made before the run as "
                "BADS(..., precomputed_evaluations=(X, y))."
            )

        # Other variables initializations
        optim_state["search_factor"] = 1
        optim_state["sd_level"] = self.options["incumbent_sigma_multiplier"]
        optim_state["lastfitgp"] = -np.inf
        # Last fcn evaluation for which the gp was trained
        self.mesh_overflows = 0
        # Number of attempted mesh expansions when already at maximum size

        # List of points at the end of each iteration
        optim_state["iterlist"] = dict()
        optim_state["iterlist"]["u"] = []
        optim_state["iterlist"]["fval"] = []
        optim_state["iterlist"]["fsd"] = []
        optim_state["iterlist"]["fhyp"] = []

        # optim_state['es'] = es_update(es_mu, es_lambda)

        # Hedge struct
        optim_state["search_hedge"] = dict()

        # Before first iteration
        # Iterations are from 0 onwards in optimize so we should have -1
        optim_state["iter"] = -1

        # The run's limits are positive integers or inf (_as_limit), stored as
        # an int or inf, inf turning the limit off: max_fun_evals, as MATLAB
        # BADS's setupoptions.m checks it, and max_iter and tol_stall_iters,
        # which MATLAB BADS does not check. The loop compares the number of
        # evaluations or of the iteration with them, and reads the iteration
        # tol_stall_iters back, which a string, or a tol_stall_iters of 0 or
        # not whole, stops with TypeError or IndexError at the end of an
        # iteration
        for name in ("max_fun_evals", "max_iter", "tol_stall_iters"):
            value = _as_limit(self.options[name])
            if value is None:
                raise ValueError(
                    f"options['{name}'] needs to be a positive integer or "
                    f"inf, not {self.options[name]!r}."
                )
            self.options[name] = value
        # search_n_try, the number of searches of an iteration, is an integer
        # at least 0, 0 a run without searches, whose every pass of the loop
        # polls, as in MATLAB BADS (bads.m:516, 744), which does not check
        # it; a whole-number float is converted. A value that is not whole
        # ends no round of searches, after which the loop turns without
        # evaluating. The first iteration starts at the poll: its round of
        # searches counts as done
        search_n_try = self.options["search_n_try"]
        if not (
            _is_real(search_n_try)
            and _is_whole_number(search_n_try)
            and search_n_try >= 0
        ):
            raise ValueError(
                "options['search_n_try'] needs to be an integer at least 0, "
                f"not {search_n_try!r}."
            )
        self.options["search_n_try"] = int(search_n_try)
        optim_state["search_count"] = self.options["search_n_try"]
        # tol_mesh is a positive finite number (_as_real_number), stored as a
        # float and put on the mesh, as in MATLAB BADS (setupvars.m:105),
        # which does not check it: at 0 or below, the mesh criterion never
        # ends the run
        tol_mesh = self.options["tol_mesh"]
        value = _as_real_number(tol_mesh)
        if value is None or not 0 < value < np.inf:
            raise ValueError(
                "options['tol_mesh'] needs to be a positive finite number, "
                f"not {tol_mesh!r}."
            )
        self.options["tol_mesh"] = value
        optim_state["tol_mesh"] = self.options[
            "poll_mesh_multiplier"
        ] ** np.ceil(
            np.log(self.options["tol_mesh"])
            / np.log(self.options["poll_mesh_multiplier"])
        )
        # noise_final_samples, the evaluations of the returned point at the
        # end of a noisy run, is an integer at least 0; a whole-number float
        # is converted. MATLAB BADS does not check it; a value that is not
        # whole stops the run with TypeError after its last iteration
        noise_final_samples = self.options["noise_final_samples"]
        if not (
            _is_real(noise_final_samples)
            and _is_whole_number(noise_final_samples)
            and noise_final_samples >= 0
        ):
            raise ValueError(
                "options['noise_final_samples'] needs to be an integer at "
                f"least 0, not {noise_final_samples!r}."
            )
        self.options["noise_final_samples"] = int(noise_final_samples)
        # improvement_quantile lies in (0, 1), which MATLAB BADS checks when
        # it evaluates an improvement (bads.m:1269-1271). It and the hedge's
        # three options below take a real number (_as_real_number), which
        # is stored as a float
        improvement_quantile = self.options["improvement_quantile"]
        value = _as_real_number(improvement_quantile)
        if value is None or not 0 < value < 1:
            raise ValueError(
                "options['improvement_quantile'] needs to be greater than 0 "
                f"and less than 1, not {improvement_quantile!r}."
            )
        self.options["improvement_quantile"] = value
        # accelerate_mesh_steps is a positive integer: the accelerated mesh
        # reduction reads the iterate accelerate_mesh_steps iterations
        # back, which below 1 is not recorded yet (MATLAB BADS fails there
        # too, bads.m:976-979); a whole-number float is converted
        accelerate_mesh_steps = self.options["accelerate_mesh_steps"]
        if not (
            _is_real(accelerate_mesh_steps)
            and _is_whole_number(accelerate_mesh_steps)
            and accelerate_mesh_steps >= 1
        ):
            raise ValueError(
                "options['accelerate_mesh_steps'] needs to be a positive "
                f"integer, not {accelerate_mesh_steps!r}; "
                "options['accelerate_mesh'] = False turns the accelerated "
                "reduction of the mesh off."
            )
        self.options["accelerate_mesh_steps"] = int(accelerate_mesh_steps)
        # n_search, the number of candidates of the ES search, and
        # n_search_iter, the number of its generations, are positive
        # integers, and n_search_iter is at most n_search: each generation
        # draws n_search / n_search_iter candidates, rounded down, so that
        # above n_search the search has none (MATLAB BADS checks neither; it
        # loops over 1:Nsearchiter, searchES.m:125, and draws a population of
        # Nsearch / Nsearchiter points); a whole-number float is converted
        n_search = self.options["n_search"]
        if not (
            _is_real(n_search) and _is_whole_number(n_search) and n_search >= 1
        ):
            raise ValueError(
                "options['n_search'] needs to be a positive integer, "
                f"not {n_search!r}."
            )
        self.options["n_search"] = int(n_search)
        n_search_iter = self.options["n_search_iter"]
        if not (
            _is_real(n_search_iter)
            and _is_whole_number(n_search_iter)
            and 1 <= n_search_iter <= self.options["n_search"]
        ):
            raise ValueError(
                "options['n_search_iter'] needs to be a positive integer, at "
                "most options['n_search'] "
                f"({self.options['n_search']}), not {n_search_iter!r}."
            )
        self.options["n_search_iter"] = int(n_search_iter)
        # search_method is a non-empty list of pairs (name, sum-rule flag),
        # each name a search that ESSearchHedge runs, "ES-wcm" or "ES-ell",
        # as the hedge compares it (_is_named_pair: a NumPy array of pairs,
        # or of names, runs too); an entry with more elements is refused, as
        # a sign of another form, such as MATLAB's {@searchES, 1, 1}. MATLAB
        # BADS does not check it
        search_method = self.options["search_method"]
        try:
            methods = list(search_method)
        except TypeError:
            methods = []
        if not (
            len(methods) > 0
            and all(
                _is_named_pair(method, ("ES-wcm", "ES-ell"))
                for method in methods
            )
        ):
            raise ValueError(
                "options['search_method'] needs to be a non-empty list of "
                "pairs (name, sum-rule flag), each name 'ES-wcm' or "
                f"'ES-ell', not {search_method!r}."
            )
        # hedge_gamma, the smallest probability of each search method, lies
        # in [0, 1 / n], n the number of search methods: the hedge chooses a
        # method with the probabilities (1 - n * hedge_gamma) * softmax +
        # hedge_gamma, which invert above 1 / n and turn negative above
        # 1 / (n - 1); MATLAB BADS does not check it (searchHedge.m:46)
        hedge_gamma = self.options["hedge_gamma"]
        n_search_methods = len(search_method)
        value = _as_real_number(hedge_gamma)
        if value is None or not (0 <= value and n_search_methods * value <= 1):
            raise ValueError(
                "options['hedge_gamma'] needs to lie between 0 and 1 / n, n "
                "the number of search methods in options['search_method'] "
                f"({n_search_methods}), not {hedge_gamma!r}."
            )
        self.options["hedge_gamma"] = value
        # hedge_beta, the inverse temperature of the hedge's softmax, is a
        # finite number at least 0 (0 is a uniform choice): below 0 the hedge
        # favors the search of lower gain, and at inf or NaN (or far below 0,
        # where the softmax overflows) its probabilities and gains are NaN,
        # so that every choice is at random; MATLAB BADS does not check it
        # (searchHedge.m:45)
        hedge_beta = self.options["hedge_beta"]
        value = _as_real_number(hedge_beta)
        if value is None or not 0 <= value < np.inf:
            raise ValueError(
                "options['hedge_beta'] needs to be a finite number greater "
                f"than or equal to 0, not {hedge_beta!r}; its default is "
                "1e-3 / options['tol_fun']."
            )
        self.options["hedge_beta"] = value
        # hedge_decay, the decay of the hedge's gains at each update, lies in
        # [0, 1] (1 is no decay): above 1 the gains grow until they overflow
        # and every later choice is at random, and below 0 they alternate in
        # sign; MATLAB BADS does not check it (acqPortfolio.m:69)
        hedge_decay = self.options["hedge_decay"]
        value = _as_real_number(hedge_decay)
        if value is None or not 0 <= value <= 1:
            raise ValueError(
                "options['hedge_decay'] needs to lie between 0 and 1, not "
                f"{hedge_decay!r}."
            )
        self.options["hedge_decay"] = value
        # search_acq_fcn is the pair ("acq_LCB", sqrt_beta), its name as the
        # ES search compares it (_is_named_pair): the LCB is the search's
        # only acquisition function (MATLAB BADS's others, which read the
        # optimization target, are not ported), and more elements are
        # refused. Its sqrt_beta, which acq_fcn_lcb checks at each call, is
        # checked here too, before any evaluation
        search_acq_fcn = self.options["search_acq_fcn"]
        if not _is_named_pair(search_acq_fcn, ("acq_LCB",)):
            raise ValueError(
                "options['search_acq_fcn'] needs to be a pair ('acq_LCB', "
                f"sqrt_beta), not {search_acq_fcn!r}."
            )
        check_sqrt_beta(
            search_acq_fcn[1], "options['search_acq_fcn'][1] (sqrt_beta)"
        )
        if self.options["improvement_quantile"] > 0.5:
            self.logger.warning(
                "options['improvement_quantile'] is greater than 0.5. This "
                "might produce unpredictable behavior. Set "
                "options['improvement_quantile'] < 0.5 for conservative "
                "improvement."
            )

        # Copy maximum number of fcn. evaluations,
        # used by some acquisition fcns.
        optim_state["max_fun_evals"] = self.options.get("max_fun_evals")

        # Deal with user specified target noise
        if (
            self.options["specify_target_noise"]
            and self.options["uncertainty_handling"] is None
        ):
            self.options["uncertainty_handling"] = True

        if (
            self.options["specify_target_noise"]
            and self.options["uncertainty_handling"] is not None
            and self.options["uncertainty_handling"] == False
        ):
            raise ValueError(
                "If options['specify_target_noise'] is True, "
                "options['uncertainty_handling'] needs to be True too, or "
                "unset (None)."
            )

        # noise_size is a base noise SD, or MATLAB's pair of that base and
        # the SD of the prior over its logarithm: one or two real numbers
        # (_as_real_number), as a number, a sequence or an array, stored as a
        # float or an array of two floats
        noise_size = self.options["noise_size"]
        if noise_size is not None:
            elements = np.ravel(np.asarray(noise_size, dtype=object))
            values = [_as_real_number(element) for element in elements]
            if len(values) not in (1, 2) or None in values:
                raise ValueError(
                    "options['noise_size'] needs to be a number, or a pair "
                    "of the base noise SD and the SD of the prior over its "
                    f"logarithm, not {noise_size!r}."
                )
            self.options["noise_size"] = (
                values[0] if len(values) == 1 else np.array(values)
            )
        # Without specify_target_noise, which ignores it, the base is
        # positive, as MATLAB BADS's setupoptions.m checks, and finite, and a
        # finite SD of the prior is positive; one that is not finite stands
        # for its default, 1, as in MATLAB BADS (gpdef/gpdefBads.m:147-151,
        # private/gpupdate.m:379-381). The GP's noise prior, centred at the
        # log of the base with that SD, refuses the others at its first fit
        if (
            not self.options["specify_target_noise"]
            and self.options["noise_size"] is not None
        ):
            values = np.ravel(self.options["noise_size"])
            if not 0 < values[0] < np.inf:
                raise ValueError(
                    "options['noise_size'], if specified, needs a positive "
                    f"finite base noise SD, not {noise_size!r}."
                )
            if values.size == 2 and np.isfinite(values[1]) and values[1] <= 0:
                raise ValueError(
                    "options['noise_size'] needs a positive SD of the prior "
                    "over the log noise SD, or inf for its default of 1, not "
                    f"{noise_size!r}."
                )
        # The GP's noise is bounded above at a log SD of 5 (_gp_hyp), as in
        # MATLAB's gpdefBads.m
        if (
            not self.options["specify_target_noise"]
            and self.options["noise_size"] is not None
            and np.ravel(self.options["noise_size"])[0] > np.exp(5)
        ):
            self.logger.warning(
                "options['noise_size'] exceeds exp(5), about 148: the GP "
                "cannot represent a noise SD that large, and its noise will "
                "sit at that bound. Rescale the target to reduce its noise."
            )
        if (
            self.options["specify_target_noise"]
            and self.options["noise_size"] is not None
            and np.ravel(self.options["noise_size"])[0] > 0
        ):
            self.logger.warning(
                "If options['specify_target_noise'] is True, "
                "options['noise_size'] is ignored: leave it unset (None) "
                "or set it to 0 to silence this warning."
            )
        # Sto-BADS, PyBADS's own (KD-S-1), is deprecated: users are to know
        # that it is not the setting for a noisy target
        if self.options["stobads"]:
            self.logger.warning(
                "options['stobads'] is deprecated and may be removed in a "
                "future release: Sto-BADS is experimental and has not been "
                "found to improve noisy runs. For a noisy target, set "
                "options['uncertainty_handling'] = True and leave "
                "options['stobads'] off."
            )

        # Set uncertainty handling level
        # (0: none; 1: unknown noise level; 2: user-provided noise)
        if self.options.get("specify_target_noise"):
            optim_state["uncertainty_handling_level"] = 2
        elif (
            self.options["uncertainty_handling"] is not None
            and self.options["uncertainty_handling"]
        ):
            optim_state["uncertainty_handling_level"] = 1
        else:
            optim_state["uncertainty_handling_level"] = 0

        # Empty hedge struct for acquisition functions
        if self.options.get("acq_hedge"):
            optim_state["acq_hedge"] = dict()

        # List of points at the end of each iteration
        optim_state["iterlist"] = dict()
        optim_state["iterlist"]["u"] = []
        optim_state["iterlist"]["fval"] = []
        optim_state["iterlist"]["fsd"] = []
        optim_state["iterlist"]["fhyp"] = []

        # Initialize Gaussian process settings
        # Rational-quadratic kernel with separate length scales (MATLAB's
        # default, 'rq')
        optim_state["gp_cov_fun"] = 1

        if optim_state.get("uncertainty_handling_level") == 0:
            # Observation noise for stability
            optim_state["gp_noisefun"] = [1, 0, 0]
        elif optim_state.get("uncertainty_handling_level") == 1:
            # Infer noise
            optim_state["gp_noisefun"] = [1, 2, 0]
        elif optim_state.get("uncertainty_handling_level") == 2:
            # Provided heteroskedastic noise
            optim_state["gp_noisefun"] = [1, 1, 0]

        if (
            self.options.get("noise_shaping")
            and optim_state["gp_noisefun"][1] == 0
        ):
            optim_state["gp_noisefun"][1] = 1

        optim_state["gp_mean_fun"] = self.options.get("gp_mean_fun")
        # The constant mean is MATLAB's (gpdefBads.m); the negative
        # quadratic, PyVBMC's mean for log densities, has the wrong shape
        # for a minimizer and no priors in _gp_hyp
        valid_gp_mean_funs = ["zero", "const"]

        if not optim_state["gp_mean_fun"] in valid_gp_mean_funs:
            raise ValueError(
                "options['gp_mean_fun'] should be 'const' (a constant mean) "
                "or 'zero'; other GP mean functions are not supported."
            )
        optim_state["int_meanfun"] = self.options.get("gpintmeanfun")

        # MATLAB's per-dimension empirical prior over the length scales,
        # 'ard' (gpdefBads.m), is not supported: the GP's prior is 'iso'
        if self.options.get("gp_cov_prior") != "iso":
            raise ValueError(
                "options['gp_cov_prior'] should be 'iso' (an empirical prior "
                "shared by the GP length scales); 'ard' is not supported."
            )

        # MATLAB's acquisition hedge (AcqHedge), which MATLAB BADS labels
        # unsupported, is not ported
        if self.options.get("acq_hedge"):
            raise ValueError(
                "options['acq_hedge'] should be False: the acquisition hedge "
                "is not supported."
            )

        # A known noise level, which MATLAB refuses too (gpdefBads.m)
        if not self.options.get("fit_lik"):
            raise ValueError(
                "Fixed noise not supported: options['fit_lik'] should be "
                "True."
            )

        return optim_state

    def _import_precomputed_evaluations_(self, evaluations):
        """
        Check the evaluations made before the run (the argument
        ``precomputed_evaluations``, a tuple ``(X, y)`` or ``(X, y, y_sd)``)
        and add them to the function log, before any evaluation of the run.

        MATLAB BADS imports them from its option ``FunValues`` into its log
        (``setupvars.m:126-167``, ``funlogger.m:53-83``) with checks of the
        shapes and values only; PyBADS takes PyVBMC's interface and checks,
        and refuses the points outside the hard bounds or that violate
        ``non_box_cons``, which the run never evaluates. The log's count of
        evaluations (``func_count``) leaves them out, as MATLAB's
        ``funccount`` does. Without uncertainty handling, a point given twice
        is kept once, its values agreeing; with it, the logger keeps each
        repeat, merged into the point's row when ``fun`` returns the SDs.
        ``optim_state`` records the number of evaluations given
        (``"precomputed_observations"``), of distinct points
        (``"precomputed_locations"``) and of evaluations that the log's
        ``n_evals`` counts (``"precomputed_n_evals"``), all 0 without them.
        """
        self.optim_state["precomputed_observations"] = 0
        self.optim_state["precomputed_locations"] = 0
        self.optim_state["precomputed_n_evals"] = 0
        if evaluations is None:
            return
        if not isinstance(evaluations, (tuple, list)) or len(
            evaluations
        ) not in (2, 3):
            raise ValueError(
                "precomputed_evaluations must be a tuple (X, y), or (X, y, "
                "y_sd) with options['specify_target_noise']."
            )
        # The points hold all the variables, fixed ones included, as the
        # target takes them
        X = _precomputed_array(
            evaluations[0], (None, self.var_transf.D_orig), "points X"
        )
        n_rows = X.shape[0]
        y = _precomputed_array(evaluations[1], (n_rows,), "values y")
        y_sd = None
        if len(evaluations) == 3:
            y_sd = _precomputed_array(
                evaluations[2], (n_rows,), "noise SDs y_sd"
            )
            if np.any(y_sd <= 0):
                raise ValueError(
                    "The noise SDs y_sd of precomputed_evaluations must be "
                    f"positive; rows {np.flatnonzero(y_sd <= 0).tolist()} "
                    "are not."
                )

        level = self.optim_state["uncertainty_handling_level"]
        if level == 2 and y_sd is None:
            raise ValueError(
                "With options['specify_target_noise'], "
                "precomputed_evaluations must hold the noise SDs of the "
                "values: (X, y, y_sd)."
            )
        if level < 2 and y_sd is not None:
            raise ValueError(
                "precomputed_evaluations holds noise SDs y_sd, which require "
                'options["specify_target_noise"] = True.'
            )

        # The hard bounds of all the variables: a point whose coordinate at a
        # fixed variable is not its value lies outside them
        free = np.isnan(self._fixed_values[0])
        lb_orig, ub_orig = self._fixed_values.copy(), self._fixed_values.copy()
        lb_orig[:, free] = self.optim_state["lb_orig"]
        ub_orig[:, free] = self.optim_state["ub_orig"]
        outside = np.any((X < lb_orig) | (X > ub_orig), axis=1)
        if np.any(outside):
            raise ValueError(
                "The points X of precomputed_evaluations must lie within the "
                "hard bounds; rows "
                f"{np.flatnonzero(outside).tolist()} do not."
            )
        if self.non_box_cons is not None and n_rows > 0:
            violating = np.ravel(self.non_box_cons(X.copy())) > 0
            if np.any(violating):
                raise ValueError(
                    "The points X of precomputed_evaluations must satisfy "
                    "non_box_cons; rows "
                    f"{np.flatnonzero(violating).tolist()} violate it."
                )

        # The first row of each point, whose value the later ones repeat
        # without uncertainty handling (a key of -0.0 is that of 0.0)
        first_rows = {}
        retained = []
        for row, point in enumerate(map(tuple, X.tolist())):
            first = first_rows.setdefault(point, row)
            if first == row or level > 0:
                retained.append(row)
            elif not _precomputed_values_agree(y[first], y[row]):
                raise ValueError(
                    "Rows {} and {} of precomputed_evaluations give two "
                    "values at one point. For a noisy target, set "
                    "options['uncertainty_handling'] = True.".format(
                        first, row
                    )
                )

        # A point far outside the plausible box of an unbounded variable can
        # map beyond float64 in the transformed space
        with np.errstate(over="ignore", invalid="ignore"):
            U = self.var_transf(X[retained])
        infinite = ~np.all(np.isfinite(U), axis=1)
        if np.any(infinite):
            raise ValueError(
                "The points X of precomputed_evaluations must map to finite "
                "coordinates of the optimization; rows "
                f"{np.asarray(retained)[infinite].tolist()} lie too far "
                "outside the plausible bounds."
            )
        for u, row in zip(U, retained):
            self.function_logger.add(
                u, y[row], None if y_sd is None else y_sd[row]
            )

        self.optim_state["precomputed_observations"] = n_rows
        self.optim_state["precomputed_locations"] = len(first_rows)
        self.optim_state["precomputed_n_evals"] = int(
            np.sum(self.function_logger.n_evals[self.function_logger.X_flag])
        )

    def _check_tol_fun_(self):
        """
        Check the user's ``tol_fun``, before the advanced options are
        evaluated, the default of ``hedge_beta``, ``1e-3 / tol_fun``, among
        them: a real number (``_is_real``: a boolean, a string, an array or
        a complex number is refused), positive and at most e^6, as
        ``improvement_quantile`` and the hedge's options are real numbers.
        The GP's log noise SD is bounded below by
        ``log(tol_fun) - 1`` and above by 5 (``_gp_hyp``, as MATLAB's
        ``gpdefBads.m``), bounds that cross above e^6, so that a larger
        ``tol_fun``, inf included, stopped the run at its first fit of the
        GP. 0 and False stopped with a bare ``ZeroDivisionError`` at the
        default of ``hedge_beta``, which refused a negative value or NaN but
        not -inf, and not beside a user's ``hedge_beta``. MATLAB BADS does
        not check it.
        """
        tol_fun = self.options.get("tol_fun")
        if tol_fun is None:
            # Not set by the user: the default of advanced_bads_options.ini
            return
        if not (_is_real(tol_fun) and 0 < tol_fun <= math.exp(6)):
            raise ValueError(
                "options['tol_fun'] needs to be a positive number at most "
                f"e^6 (about 403), not {tol_fun!r}."
            )

    def _check_periodic_vars_(self, lower_bounds, upper_bounds):
        """
        Check ``periodic_vars`` and store it as a sorted list of indices, or
        ``None`` when it names no variable. An empty value names none, as in
        MATLAB BADS (``setupvars.m``). The indices are integers from 0 to
        one less than the number of variables, each given once; a boolean
        mask is refused rather than read as the indices 0 and 1. A periodic
        variable wraps around its hard bounds, ``[lb, ub)``, which must be
        finite, as MATLAB BADS requires.

        The indices count all the variables, fixed ones included, and are
        checked against their hard bounds, ``lower_bounds`` and
        ``upper_bounds``. A fixed variable is left out of the run, periodic
        or not: the run's mask of periodic variables and the transform map
        the indices to its variables (``_run_indices``).
        """
        value = self.options["periodic_vars"]
        if value is None or (
            not isinstance(value, str) and np.size(value) == 0
        ):
            self.options["periodic_vars"] = None
            return
        D_orig = lower_bounds.shape[1]
        indices = np.atleast_1d(np.asarray(value))
        # A boolean among integers, which NumPy casts to 0 or 1, is refused
        # as a mask is
        has_bool = any(
            isinstance(v, (bool, np.bool_))
            for v in np.atleast_1d(np.asarray(value, dtype=object)).ravel()
        )
        if indices.dtype.kind not in "iu" or indices.ndim != 1 or has_bool:
            raise ValueError(
                "options['periodic_vars'] should be a list of the indices of "
                "the periodic variables, integers from 0 to one less than the "
                "number of variables (a boolean mask m gives them as "
                f"np.flatnonzero(m)), not {value!r}."
            )
        if np.any(indices < 0) or np.any(indices >= D_orig):
            raise ValueError(
                "options['periodic_vars'] holds indices outside 0 to "
                f"{D_orig - 1}, those of the {D_orig} variables: {value!r}."
            )
        if np.unique(indices).size != indices.size:
            raise ValueError(
                "options['periodic_vars'] names a variable more than once: "
                f"{value!r}."
            )
        indices = sorted(int(i) for i in indices)
        infinite = [
            i
            for i in indices
            if not (
                np.isfinite(lower_bounds[0, i])
                and np.isfinite(upper_bounds[0, i])
            )
        ]
        if infinite:
            raise ValueError(
                "Periodic variables need to have finite lower and upper "
                "bounds, which set their period: the bounds of the "
                f"variables {infinite} of options['periodic_vars'] are not "
                "finite."
            )
        self.options["periodic_vars"] = indices

    def _variable_transformer_(self):
        """The transformation of the variables, from the bounds in the
        original space and the ``nonlinear_scaling`` option, with the values
        of the fixed variables, which it puts back into the points it
        returns to the original space."""
        if self.options["nonlinear_scaling"]:
            logflag = np.full((1, self.D), np.nan)
            periodic_vars = self.options["periodic_vars"]
            if periodic_vars is not None and len(periodic_vars) != 0:
                logflag[
                    :, _run_indices(periodic_vars, self._fixed_values)
                ] = 0  # Never transform periodic variables
        else:
            logflag = np.zeros((1, self.D))

        return VariableTransformer(
            self.D,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
            logflag,
            fixed_values=self._fixed_values,
        )

    def _init_rng_(self):
        """
        Create ``self.rng`` from the ``random_seed`` option, and store in
        ``self._random_seed`` the seed that the result reports: the option
        when it is an integer (a whole-number float converted to one) or
        ``None``, and ``None`` otherwise.
        """
        seed = self.options["random_seed"]
        if isinstance(seed, (float, np.floating)) and float(seed).is_integer():
            seed = int(seed)
        try:
            self.rng = get_rng(seed)
        except (TypeError, ValueError) as err:
            raise type(err)(
                "options['random_seed'] needs to be None or a value that "
                "numpy.random.default_rng takes, such as a non-negative "
                "integer, a numpy.random.SeedSequence or a "
                "numpy.random.Generator, not "
                f"{self.options['random_seed']!r}."
            ) from err
        if isinstance(seed, (int, np.integer)):
            self._random_seed = int(seed)
        else:
            self._random_seed = None

    def _init_mesh_(self):
        """
        A private function to initialize the mesh frame and the optimization problem.
        It evaluates the initial points, which includes the starting point and the generated point retrieved from a sobol sequence generating method.
        The init_mesh also assess if the target function is stochastic and set the parameter of BADS for handling stochastic targets.

        Returns
        ----------
        is_finished : bool
            True when the run ends here: with ``max_fun_evals = 1``, after
            the evaluation of the starting point.
        """
        # Evaluate starting point and initial mesh, determine if function is noisy
        self.yval, self.fsd, idx_start = self.function_logger(self.u)
        if self.fsd is None:
            self.fsd = np.nan
        self.fval = self.yval
        self.optim_state["fval"] = self.fval
        self.optim_state["yval"] = self.yval
        # The rows of the log of the start and the initial design, among
        # which the first incumbent is chosen, as in MATLAB BADS
        # (evalinitmesh.m:120-123), and on which the first GP is trained:
        # the log may also hold evaluations made before the run
        # (precomputed_evaluations)
        self._init_rows = np.array([idx_start])
        self._init_incumbent_row = idx_start

        if self.options["uncertainty_handling"] is None:
            # Test whether the function is noisy, only when the option is
            # left empty, as in MATLAB BADS: False declares it deterministic
            self.logging_action.append("Uncertainty test")
            # Its time stays out of the target's time, and it is not
            # recorded, as MATLAB BADS calls the target directly for it
            # (evalinitmesh.m:41)
            function_logger = self.function_logger
            total_fun_eval_time = function_logger.total_fun_eval_time
            yval_bis, _, _ = function_logger(
                self.u, record_duplicate_data=False
            )
            function_logger.total_fun_eval_time = total_fun_eval_time
            # The test counts in max_fun_evals and adds no point to the log,
            # so the GP's fit schedule leaves it out of its budget
            # (_get_gp_training_options)
            self.optim_state["n_noise_test"] = 1
            if np.abs(self.yval - yval_bis) > self.options["tol_noise"]:
                self.optim_state["uncertainty_handling_level"] = 1
                function_logger.uncertainty_handling_level = 1
                self.logging_action.append("Uncertainty test")
        else:
            self.optim_state["n_noise_test"] = 0
            self.logging_action.append("")

        if self.optim_state["uncertainty_handling_level"] > 0:
            if self.options["specify_target_noise"]:
                self.logger.log(
                    _LOG_NOTIFY,
                    "Beginning optimization of a STOCHASTIC objective function (specified noise)\n",
                )
            else:
                self.logger.log(
                    _LOG_NOTIFY,
                    "Beginning optimization of a STOCHASTIC objective function\n",
                )
        else:
            self.logger.log(
                _LOG_NOTIFY,
                "Beginning optimization of a DETERMINISTIC objective function\n",
            )

        # At most one note before the column headers, and none of them
        # touches a draw of the run: the reminder that the installed release
        # is more than a year old (pybads/bads/_release_reminder.py), or else
        # an occasional tip (pybads/bads/_runtime_tips.py)
        if not self._runtime_tip_considered:
            self._runtime_tip_considered = True
            release_reminder_shown = consider_release_reminder(
                logger=self.logger, enabled=self.options["show_tips"]
            )
            consider_runtime_tip(
                logger=self.logger,
                enabled=self.options["show_tips"],
                release_reminder_shown=release_reminder_shown,
            )

        # set up strings for logging of the iteration
        self.display_format = self._setup_logging_display_format()
        self._log_column_headers()
        self._display_function_log_(0, "")

        # Only one function evaluation
        if self.options["max_fun_evals"] == 1:
            return True

        # If dealing with a noisy function, use a large initial mesh
        if self.optim_state["uncertainty_handling_level"] > 0:
            self.options["fun_eval_start"] = np.minimum(
                np.maximum(20, self.options["fun_eval_start"]),
                self.options["max_fun_evals"],
            )

        if self.options["fun_eval_start"] > 0:
            # Evaluate initial points but not more than options['max_fun_evals']
            fun_eval_start = np.minimum(
                self.options["fun_eval_start"],
                self.options["max_fun_evals"] - 1,
            )
            if self.options["init_fun"] == "init_sobol":
                u1, _ = init_sobol(
                    self.u,
                    self.lower_bounds,
                    self.upper_bounds,
                    self.plausible_lower_bounds,
                    self.plausible_upper_bounds,
                    fun_eval_start,
                    rng=self.rng,
                )
                # The design, rounded up to a power of two, keeps its first
                # points within the evaluations left (the noise test counted)
                n_left = (
                    self.options["max_fun_evals"]
                    - self.function_logger.func_count
                )
                if np.isfinite(n_left):
                    u1 = u1[: int(n_left)]
                # Enforce periodicity and force the points on the search grid
                u1 = force_to_grid_periodic(
                    u1,
                    self.optim_state["search_mesh_size"],
                    self.lower_bounds,
                    self.upper_bounds,
                    self.optim_state["periodic_vars"],
                )

                # Remove already evaluated or unfeasible points from search set
                u1 = contraints_check(
                    u1,
                    self.optim_state["lb_search"],
                    self.optim_state["ub_search"],
                    self.optim_state["tol_mesh"],
                    self.function_logger,
                    True,
                    self.non_box_cons,
                )

                init_rows = [idx_start]
                for u_idx in range(len(u1)):
                    _, _, idx = self.function_logger(u1[u_idx])
                    init_rows.append(idx)

                # The first of the lowest values, in the order of the log
                self._init_rows = np.unique(init_rows)
                idx_yval = self._init_rows[
                    np.argmin(self.function_logger.Y[self._init_rows])
                ]
                self._init_incumbent_row = idx_yval
                self.u = self.function_logger.X[idx_yval].copy()
                self.yval = self.function_logger.Y[idx_yval].item()
                self.fval = self.yval
                self.logging_action.append("Initial points")
                self._display_function_log_(0, "Initial mesh")
            else:
                raise ValueError(
                    "options['init_fun'] needs to be 'init_sobol', the only "
                    "initial design available."
                )

        if not np.isfinite(self.yval):
            raise ValueError("Cannot find a valid starting point.")

        self.optim_state["fval"] = self.fval
        self.optim_state["yval"] = self.yval

        # The number of points evaluated, the start and the initial design,
        # which can differ from options['fun_eval_start'], for instance with
        # a noisy target or with non_box_cons: the evaluations but the noise
        # test, whose point is not recorded
        self.optim_state["eff_starting_points"] = (
            self.function_logger.func_count - self.optim_state["n_noise_test"]
        )

        return False

    def _init_optimization_(self):
        """
        A private function initialize the optimization problem.
        It calls the init_mesh, sets the option configurations required by BADS, and initializes the Guassian Process (GP)

        A run that ends in the initialization (``max_fun_evals = 1``) trains
        no GP: the returned ``gp`` is None.
        """
        gp = None
        self.reset_gp = False
        self.poll_moved = False
        hyp_dict = {}

        # Evaluate starting point and initial mesh,
        with self._stage_timer.stage("init"):
            is_finished = self._init_mesh_()

        # Change options for uncertainty handling
        if self.optim_state["uncertainty_handling_level"] > 0:
            self.options["tol_stall_iters"] = (
                2 * self.options["tol_stall_iters"]
            )
            self.options["n_train_max"] = max(200, self.options["n_train_max"])
            self.options["n_train_min"] = 2 * self.options["n_train_min"]
            self.options["mesh_overflow_warning"] = (
                2 * self.options["mesh_overflow_warning"]
            )
            self.options["min_failed_poll_steps"] = np.inf
            self.options["mesh_noise_multiplier"] = 0
            if (
                self.options["noise_size"] is None
                or self.optim_state["uncertainty_handling_level"] > 1
            ):
                # With specify_target_noise, noise_size is ignored (the
                # warning of _init_optim_state_ says so): the high-noise
                # check of the local GP takes the default base
                self.options["noise_size"] = 1.0

            # Keep some function evaluations for the final resampling, none
            # when none are left (so that max_fun_evals never grows)
            self.options["noise_final_samples"] = max(
                0,
                min(
                    self.options["noise_final_samples"],
                    self.options["max_fun_evals"]
                    - self.function_logger.func_count,
                ),
            )
            self.options["max_fun_evals"] = (
                self.options["max_fun_evals"]
                - self.options["noise_final_samples"]
            )

            # Specify the standard deviation of the function values
            # It corresponds to specify target noise of Matlab
            if self.optim_state["uncertainty_handling_level"] > 1:
                self.fsd = self.function_logger.S[
                    self._init_incumbent_row
                ].item()
            else:
                self.fsd = float(np.ravel(self.options["noise_size"])[0])

        else:
            if self.options["noise_size"] is None:
                self.options["noise_size"] = np.sqrt(self.options["tol_fun"])
            self.fsd = 0.0
            # Since the function is fully-deterministic no need of stobads
            if self.options["stobads"]:
                self.options["stobads"] = False

        self.optim_state["fsd"] = self.fsd
        self.u_best = self.u.copy()
        self.optim_state["usuccess"] = self.u_best.copy()
        self.optim_state["ysuccess"] = self.yval
        self.optim_state["fsuccess"] = self.fval
        self.optim_state["u"] = self.u.copy()
        self.optim_state["u_success"] = []
        self.optim_state["y_success"] = []
        self.optim_state["f_success"] = []

        if is_finished:
            return gp, None, None, hyp_dict

        # Initialize Gaussian Process (GP) structure
        with self._stage_timer.stage("gp_init"):
            gp, Ns_gp, sn2hpd, hyp_dict = init_and_train_gp(
                hyp_dict,
                self.optim_state,
                self.function_logger,
                self.iteration_history,
                self.options,
                self.plausible_lower_bounds,
                self.plausible_upper_bounds,
                rng=self.rng,
                rows=self._init_rows,
                timer=self._stage_timer,
            )

        self.gp_stats = IterationHistory(
            [
                "iter_gp",
                "fval",
                "ymu",
                "ys",
                "gp",
            ]
        )
        self.best_gp_hyp = gp.get_hyperparameters(as_array=True)

        return (
            gp,
            Ns_gp,
            sn2hpd,
            hyp_dict,
        )

    def optimize(self):
        """
        Run the optimization on an initialized ``BADS`` object.

        BADS starts at ``x0`` and finds a local minimum ``x`` of the target
        function ``fun``.

        A history of the optimization problem can be found in the
        ``iteration_history`` attribute of the ``BADS`` object.

        A ``BADS`` object runs a single optimization: create a new one for
        each run.

        Returns
        -------
        optimize_result : OptimizeResult
            Dictionary containing the result of the optimization. See the
            documentation of the ``OptimizeResult`` class for more details.
            For example, retrieve the final solution and its value with the
            attributes ``optimize_result.x`` and ``optimize_result.fval``.

        Raises
        ------
        RuntimeError
            If ``optimize`` has already been called on this object, whether
            or not that run completed.
        ValueError
            If ``fun`` returns a value that is not a finite real number, or,
            with ``options['specify_target_noise']``, does not return a
            tuple ``(f, sd)`` with a finite positive ``sd``. An error that
            ``fun`` raises reaches the caller.
        """
        if self._optimize_called:
            raise RuntimeError(
                "optimize() has already been called on this BADS object, "
                "which runs a single optimization: create a new BADS object "
                "for another run."
            )
        self._optimize_called = True

        # The stage timer of the run times each stage, exclusive of the
        # stages nested in it and of the target's evaluations, which form
        # the pseudo-stage "target": together they make total_time. It stays
        # on the BADS object, out of the states that a run deep-copies
        # (optim_state, the GP and its temporary_data); a plain snapshot of
        # its times goes to optim_state["stage_times"] at the end of the
        # run, and to iteration_history["timer"] at the end of each
        # iteration. It reads the target's time through the function logger
        # alone: a reference to the BADS object here (a bound method, or
        # self in a closure) would make a cycle, BADS -> timer -> BADS,
        # that keeps a finished run, every GP of its history included,
        # alive until the garbage collector goes through its oldest
        # generation.
        function_logger = self.function_logger
        self._stage_timer = StageTimer(
            lambda: function_logger.total_fun_eval_time
        )
        try:
            return self._optimize_()
        finally:
            # A run that raises leaves no stage open
            self._stage_timer.stop()

    def _optimize_(self):
        """The run of ``optimize``, under the stage timer that it
        creates."""
        is_finished = False
        poll_iteration = -1
        self.logging_action = []
        timer = Timer()
        timer.start_timer("BADS")
        stage_timer = self._stage_timer
        stage_timer.start()
        hyp_dict = {}
        self.search_success = 0
        self.last_skipped = -1
        # Last skipped iteration
        self.search_spree = 0
        self.restarts = self.options["restarts"]

        # Initialize gp; a run with max_fun_evals=1 ends there, without a GP
        gp, Ns_gp, sn2hpd, hyp_dict = self._init_optimization_()
        is_finished = gp is None
        # MATLAB BADS's exit flag, the result's status: 0 on max_fun_evals,
        # max_iter or a stop by the output function, 1 on tol_mesh and 2 on
        # the stall criterion
        exit_flag = 0
        msg = (
            "Optimization terminated: reached maximum number of function "
            "evaluations after initialization."
        )

        self.search_es_hedge = None  # init search hedge to None

        # The output function is called as in MATLAB BADS, at the start, at
        # the end of each poll and at the end, with a copy of optim_state;
        # a true return value stops the run
        output_fcn = self.options["output_fcn"]
        if output_fcn is not None:
            with stage_timer.stage("output_fcn"):
                stop = output_fcn(
                    self.var_transf.inverse_transf(self.u),
                    copy.deepcopy(self.optim_state),
                    "init",
                )
            if stop and not is_finished:
                is_finished = True
                msg = "Optimization terminated by options['output_fcn']."
        self.optim_state["termination_msg"] = msg

        poll_iteration += 1
        loop_iter = 0
        while not is_finished:
            self.optim_state["iter"] = poll_iteration
            self.gp_refitted_flag = False
            self.gp_exit_flag = np.inf
            action_txt = (
                ""  # Action performed this iteration (for printing purposes)
            )

            # Compute mesh size and search mesh size
            self.mesh_size = self.options["poll_mesh_multiplier"] ** (
                self.mesh_size_integer
            )
            self.optim_state["mesh_size"] = self.mesh_size

            if self.options["search_size_locked"]:
                self.optim_state["search_size_integer"] = np.minimum(
                    0,
                    self.mesh_size_integer
                    * self.options["search_grid_multiplier"]
                    - self.options["search_grid_number"],
                )

            self.optim_state["search_mesh_size"] = (
                self.options["poll_mesh_multiplier"]
                ** self.optim_state["search_size_integer"]
            )
            self.search_mesh_size = self.optim_state["search_mesh_size"]

            # Update bounds to grid search mesh
            (
                self.optim_state["lb_search"],
                self.optim_state["ub_search"],
            ) = self._update_search_bounds_()

            # Minimum improvement for a poll/search to be considered successful
            self.sufficient_improvement = self.options["tol_improvement"] * (
                self.mesh_size ** (self.options["forcing_exponent"])
            )
            if self.options["sloppy_improvement"]:
                self.sufficient_improvement = np.maximum(
                    self.sufficient_improvement, self.options["tol_fun"]
                )

            self.optim_state[
                "search_sufficient_improvement"
            ] = self.sufficient_improvement

            do_search_step_flag = (
                self.optim_state["search_count"] < self.options["search_n_try"]
                and len(self.function_logger.Y[self.function_logger.X_flag])
                > self.D
            )

            if do_search_step_flag:
                # Search stage
                with stage_timer.stage("search"):
                    (
                        u_search,
                        search_dist,
                        f_mu_search,
                        f_sd_search,
                        gp,
                    ) = self._search_step_(gp)
            # End Search step

            # Check whether to perform the poll stage, it can be run consecutively after the search.
            if (
                self.optim_state["search_count"] == 0
                or self.optim_state["search_count"]
                == self.options["search_n_try"]
            ):
                self.optim_state["search_count"] = 0
                if (
                    self.search_success > 0
                    and self.options["skip_poll_after_search"]
                ):
                    do_poll_step = False
                    self.search_spree += 1
                    if (
                        self.options["search_mesh_expand"] > 0
                        and np.mod(
                            self.search_spree,
                            self.options["search_mesh_expand"],
                        )
                        == 0
                        and self.options["search_mesh_increment"] > 0
                    ):
                        # Check if mesh size is already maximal
                        self._check_mesh_overflow_()

                        self.mesh_size_integer = np.minimum(
                            self.mesh_size_integer
                            + self.options["search_mesh_increment"],
                            self.options["max_poll_grid_number"],
                        )
                else:
                    do_poll_step = True
                    self.search_spree = 0

                self.search_success = 0
            else:  # In-between searches, no poll
                do_poll_step = False

            self.u = self.u_best

            # check and do poll step; the poll's GP goes on, as the search's
            if do_poll_step:
                with stage_timer.stage("poll"):
                    (_, _, _, _, gp) = self._poll_step_(gp)
                if output_fcn is not None:
                    with stage_timer.stage("output_fcn"):
                        stop = output_fcn(
                            self.var_transf.inverse_transf(self.u),
                            copy.deepcopy(self.optim_state),
                            "iter",
                        )
                    if stop:
                        is_finished = True

            # A poll that moved the incumbent asks for a rebuild of the local
            # GP at the end of every pass, until a poll that does not move,
            # as MATLAB BADS empties the posterior (bads.m:1049)
            if self.poll_moved:
                self.reset_gp = True

            # Finalize the iteration

            # TODO: Iteration plot
            if self.options["plot"] == "scatter":
                pass

            # GP hyperparameters at end of iteration
            self.best_gp_hyp = gp.get_hyperparameters(as_array=True)

            msg = ""
            if is_finished:  # stopped by the output function
                msg = "Optimization terminated by options['output_fcn']."
            # Check termination conditions
            if (
                self.function_logger.func_count
                >= self.options["max_fun_evals"]
            ):
                is_finished = True
                exit_flag = 0
                msg = "Optimization terminated: reached maximum number of function evaluations options['max_fun_evals']."

            if poll_iteration >= self.options["max_iter"] - 1:
                is_finished = True
                exit_flag = 0
                msg = "Optimization terminated: reached maximum number of iterations options['max_iter']."

            if self.optim_state["mesh_size"] < self.optim_state["tol_mesh"]:
                is_finished = True
                exit_flag = 1
                msg = "Optimization terminated: mesh size less than options['tol_mesh']."

            # Historic improvement
            if poll_iteration > self.options["tol_stall_iters"] - 1:
                idx = poll_iteration - self.options["tol_stall_iters"]
                f_base = self.iteration_history.get("fval")[idx]
                f_sd_base = self.iteration_history.get("fsd")[idx]
                self.f_q_historic_improvement = self._eval_improvement_(
                    f_base,
                    self.fval,
                    f_sd_base,
                    self.fsd,
                    self.options["improvement_quantile"],
                )

                if self.f_q_historic_improvement < self.options["tol_fun"]:
                    is_finished = True
                    exit_flag = 2
                    msg = "Optimization terminated: change in the function value less than options['tol_fun']."

            self.optim_state["termination_msg"] = msg

            # Store best points at the end of each iteration, or upon termination
            if do_poll_step or is_finished:
                with stage_timer.stage("history"):
                    self.iteration_history.record(
                        "u", self.u.flatten(), poll_iteration
                    )
                    self.iteration_history.record(
                        "x",
                        self.var_transf.inverse_transf(self.u.flatten()),
                        poll_iteration,
                    )
                    self.iteration_history.record(
                        "yval", float(self.yval), poll_iteration
                    )
                    self.iteration_history.record(
                        "fval", self.fval, poll_iteration
                    )
                    self.iteration_history.record(
                        "fsd", self.fsd, poll_iteration
                    )
                    self.iteration_history.record(
                        "mesh_size", self.mesh_size, poll_iteration
                    )
                    self.iteration_history.record(
                        "search_mesh_size",
                        self.search_mesh_size,
                        poll_iteration,
                    )
                    self.iteration_history.record(
                        "gp_hyp_full",
                        gp.get_hyperparameters(True),
                        poll_iteration,
                    )  # corresponds to self.best_gp_hyp
                    self.iteration_history.record("gp", gp, poll_iteration)
                    self.iteration_history.record(
                        "func_count",
                        self.function_logger.func_count,
                        poll_iteration,
                    )

            # Re-evaluate all noisy estimates at the end of the iteration
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and do_poll_step
                and poll_iteration > 0
            ):
                with stage_timer.stage("reestimate"):
                    self._re_evaluate_history_(gp)
                self.yval = self.iteration_history.get("yval")[poll_iteration]
                self.fval = self.iteration_history.get("fval")[poll_iteration]
                self.fsd = self.iteration_history.get("fsd")[poll_iteration]
                # optim_state keeps the incumbent's values in step
                self.optim_state["yval"] = self.yval
                self.optim_state["fval"] = self.fval
                self.optim_state["fsd"] = self.fsd
                self.best_gp_hyp = self.iteration_history.get("gp_hyp_full")[
                    poll_iteration
                ]

                f_q_re_impr = self._eval_improvement_(
                    self.fval,
                    self.iteration_history.get("fval").astype("float"),
                    self.fsd,
                    self.iteration_history.get("fsd").astype("float"),
                    self.options["improvement_quantile"],
                )
                f_q_re_impr = f_q_re_impr[1:]  # Skip the first iteration
                # An iterate without an estimate (NaN) is skipped, as by
                # MATLAB's max
                idx_impr = np.nanargmax(f_q_re_impr)
                improvement = f_q_re_impr[idx_impr]
                idx_impr = idx_impr + 1  # offset original index without skip

                # Check if any point got better
                if improvement > self.options["tol_fun"]:
                    # The incumbent moves to the iterate, its location with
                    # its value. MATLAB BADS moves u but not ubest
                    # (bads.m:1111-1118), and its next poll can run around
                    # the old incumbent with the iterate's value.
                    self._update_incumbent_(
                        self.iteration_history.get("u")[idx_impr],
                        self.iteration_history.get("yval")[idx_impr],
                        self.iteration_history.get("fval")[idx_impr],
                        self.iteration_history.get("fsd")[idx_impr],
                    )
                    # As MATLAB BADS does, only the target's hyperparameters
                    # move to the iterate; the working GP stays
                    self.best_gp_hyp = self.iteration_history.get(
                        "gp_hyp_full"
                    )[idx_impr]

            # The stage times up to the end of the iteration, its
            # re-estimation included
            if do_poll_step or is_finished:
                self.iteration_history.record(
                    "timer", stage_timer.snapshot(), poll_iteration
                )

            # if isFinished_flag
            if is_finished:
                # Multiple starts (deprecated)
                if self.restarts > 0:
                    pass
            else:
                if do_poll_step:
                    # Iteration corresponds to the number of polling iterations
                    poll_iteration += 1
                    self.optim_state["iter"] = poll_iteration

            loop_iter += 1

        # End while
        self.optim_state["exit_flag"] = exit_flag

        # Re-evaluate all best points for noisy evaluations
        yval_vec = self.yval if np.isscalar(self.yval) else self.yval.copy()
        # A run that ends in its initialization takes no final samples: the
        # result reports the incumbent's observation
        self.optim_state["yval_vec"] = np.atleast_1d(yval_vec).copy()
        self.optim_state["ysd_vec"] = None
        # The iterate whose point takes the final samples
        final_idx = None
        if (
            self.optim_state["uncertainty_handling_level"] > 0
            and poll_iteration > 0
        ):
            with stage_timer.stage("reestimate"):
                self._re_evaluate_history_(gp)

            # Order by lowest probabilistic upper bound and choose
            # the point with the lowest quantile values of the history of the optimization run: inf{x: F(x)>p}.
            sigma_multiplier = np.sqrt(2) * erfcinv(
                2 * self.options["final_quantile"]
            )  # Using inverted convention
            q_beta = self.iteration_history.get(
                "fval"
            ) + sigma_multiplier * self.iteration_history.get("fsd")
            # Skip the first iteration; an iterate without an estimate (NaN)
            # is skipped too, as by MATLAB's min
            min_q_beta_idx = np.nanargmin(q_beta[1:])
            min_q_beta_idx += 1  # offset original index with no skip
            self.yval = self.iteration_history.get("yval")[min_q_beta_idx]
            self.fval = self.iteration_history.get("fval")[min_q_beta_idx]
            self.fsd = self.iteration_history.get("fsd")[min_q_beta_idx]
            self.u = self.iteration_history.get("u")[min_q_beta_idx]
            self.u_best = self.u.copy()
            self.best_gp_hyp = self.iteration_history.get("gp_hyp_full")[
                min_q_beta_idx
            ]
            final_idx = min_q_beta_idx
        elif (
            self.optim_state["uncertainty_handling_level"] > 0
            and self.optim_state["iter"] == 0
        ):
            # A run that ends within its first iteration has one iterate,
            # the incumbent, which takes the final samples that the run
            # reserved; MATLAB BADS takes none then (bads.m:1138)
            final_idx = 0

        # Re-evaluate estimated function value and SD at final point
        final_samples = (
            final_idx is not None and self.options["noise_final_samples"] > 0
        )
        if final_samples:
            # Estimate function value and standard deviation at final point.
            # Note that by default we do *not* use YVAL because it is biased
            # (since it was an incumbent at some iteration, it is more likely to be a
            # random fluctuation lower than the mean)
            yval_vec = np.empty(self.options["noise_final_samples"])
            ysd_vec = np.empty(self.options["noise_final_samples"])
            with stage_timer.stage("final_samples"):
                for i_sample in range(self.options["noise_final_samples"]):
                    y, y_sd, _ = self.function_logger(
                        self.u, record_duplicate_data=False
                    )
                    yval_vec[i_sample] = y
                    ysd_vec[i_sample] = y_sd

            # With one sample and no noise estimate from the target, YVAL
            # is used as well (biased, but better than no uncertainty)
            if yval_vec.size == 1 and not self.options["specify_target_noise"]:
                yval_vec = np.append(yval_vec, self.yval)

            self.optim_state["yval_vec"] = np.copy(yval_vec)
            self.optim_state["ysd_vec"] = np.copy(ysd_vec)

            if self.options["specify_target_noise"]:
                # Weight the samples by the precisions the target returns
                precision = 1 / ysd_vec**2
                tot_precision = np.sum(precision)
                self.fval = (
                    np.sum(yval_vec * precision) / tot_precision
                ).item()
                self.fsd = (1 / np.sqrt(tot_precision)).item()
            else:
                # Mean of the samples and its standard error, from
                # their SD normalized by n - 1 (MATLAB's std)
                self.fval = np.mean(yval_vec).item()
                self.fsd = (
                    np.std(yval_vec, ddof=1) / np.sqrt(yval_vec.size)
                ).item()
            # The estimate describes that iterate
            self.iteration_history.record("fval", self.fval, final_idx)
            self.iteration_history.record("fsd", self.fsd, final_idx)

        if final_idx is not None:
            # optim_state keeps the returned point and its values in step,
            # for the output function's last call
            self.optim_state["u"] = self.u.copy()
            self.optim_state["yval"] = self.yval
            self.optim_state["fval"] = self.fval
            self.optim_state["fsd"] = self.fsd

        # Convert back to original space
        self.x = self.var_transf.inverse_transf(self.u)

        if output_fcn is not None:
            with stage_timer.stage("output_fcn"):
                output_fcn(
                    self.var_transf.inverse_transf(self.u),
                    copy.deepcopy(self.optim_state),
                    "done",
                )

        # Compute total running time and fractional overhead
        stage_timer.stop()
        timer.stop_timer("BADS")
        total_time = timer.get_duration("BADS")
        if self.function_logger.total_fun_eval_time > 0.0:
            overhead = (
                total_time / self.function_logger.total_fun_eval_time - 1
            )
        else:
            overhead = np.nan
        self.optim_state["total_time"] = total_time
        self.optim_state["overhead"] = overhead
        self.optim_state["stage_times"] = stage_timer.snapshot()

        self.logger.log(_LOG_FINAL, msg)
        if self.optim_state["uncertainty_handling_level"] > 0:
            if (np.isscalar(yval_vec) or yval_vec.size == 1) and final_samples:
                # one final sample, with the target's noise SD (MATLAB's
                # message calls them a GP mean and SEM)
                self.logger.log(
                    _LOG_FINAL,
                    f"Observed function value at minimum: {self.fval} ± {self.fsd} (1 sample ± the target's noise SD).",
                )
            elif np.isscalar(yval_vec) or yval_vec.size == 1:
                self.logger.log(
                    _LOG_FINAL,
                    f"Observed function value at minimum: {np.ravel(yval_vec)[0]} (1 sample). Estimated: {self.fval} ± {self.fsd} (no final samples were taken).",
                )
            else:
                # with the target's noise SDs, the samples are weighted by
                # their precisions (MATLAB's message says "mean" alone)
                mean = (
                    "precision-weighted mean"
                    if self.optim_state["uncertainty_handling_level"] == 2
                    else "mean"
                )
                self.logger.log(
                    _LOG_FINAL,
                    f"Estimated function value at minimum: {self.fval} ± {self.fsd} ({mean} ± SEM from {yval_vec.size} samples)",
                )
        else:
            self.logger.log(
                _LOG_FINAL, f"Function value at minimum: {self.fval}\n"
            )

        # BADS's output
        optimize_result = OptimizeResult(self)

        return optimize_result

    def _search_step_(self, gp: GP):
        """
        A private method that performs the search method using hedging search of Evolution Strategy (ES) searches.
        It also evaluates the performance of the search (success, unsuccess)

        Parameters
        ----------
        gp : gpyreg.gaussian_process.GP

        Returns
        ----------
        u_search : np.ndarray or None
            Candidate search point; None when the search set is empty.
        search_dist : np.ndarray or float
            Distance of the search point from the incumbent, each variable
            measured in the GP's length scale (``udist``), as an array of
            shape ``(1, 1)``; 0.0 when the search set is empty.
        f_mu_search : float
            Estimated mean function at the candidate search point.
        f_sd_search : float
            Estimated noise at the candidate search point.
        gp : gpyreg.gaussian_process.GP
        """
        # Check whether it is time to refit the GP
        refit_flag, do_gp_calibration = self._is_gp_refit_time_(
            self.options["normalpha_level"]
        )
        # A failed rebuild of the local GP asks for a refit at the next one.
        if not refit_flag and gp.temporary_data.get("needs_refit", False):
            refit_flag = True
            self._record_gp_refit_()
            do_gp_calibration = False

        stage_timer = self._stage_timer
        if (
            refit_flag
            or self.optim_state["search_count"] == 0
            or self.reset_gp
            or gp.temporary_data.get("needs_rebuild", False)
        ):
            # Local GP approximation on current incumbent
            with stage_timer.stage("gp_rebuild"):
                gp, gp_exit_flag = local_gp_fitting(
                    gp,
                    self.u,
                    self.function_logger,
                    self.options,
                    self.optim_state,
                    self.iteration_history,
                    refit_flag,
                    rng=self.rng,
                    timer=stage_timer,
                )
            # The rebuild answers a move of the incumbent, as in MATLAB BADS
            # it fills the posterior that the move emptied: once after a
            # search's move, and at every pass after a poll's move, until a
            # poll that does not move (a failed rebuild is marked for the
            # next step)
            self.reset_gp = False

            if refit_flag:
                self.gp_refitted_flag = True
            self.gp_exit_flag = np.minimum(self.gp_exit_flag, gp_exit_flag)
        # End fitting

        # The optimization target, for a search acquisition function that
        # reads it (_SEARCH_ACQ_FCNS_READING_TARGET, empty at present)
        if (
            _name_among(
                self.options["search_acq_fcn"][0],
                _SEARCH_ACQ_FCNS_READING_TARGET,
            )
            is not None
        ):
            with stage_timer.stage("target_from_gp"):
                self._update_target_(self.u_best, gp, self.best_gp_hyp)

        # Generate search set (normalized coordinate)
        self.optim_state["search_count"] += 1

        if self.search_es_hedge is None:
            self.search_es_hedge = ESSearchHedge(
                self.options["search_method"],
                self.options,
                self.non_box_cons,
                rng=self.rng,
            )
        with stage_timer.stage("search_es"):
            u_search_set, z = self.search_es_hedge(
                self.u,
                self.lower_bounds,
                self.upper_bounds,
                self.function_logger,
                gp,
                self.optim_state,
            )

        # Enforce periodicity and force the candidate points on search grid
        u_search_set = force_to_grid_periodic(
            u_search_set,
            self.optim_state["search_mesh_size"],
            self.lower_bounds,
            self.upper_bounds,
            self.optim_state["periodic_vars"],
        )

        # Remove already evaluated or unfeasible points from search set
        u_search_set = contraints_check(
            u_search_set,
            self.optim_state["lb_search"],
            self.optim_state["ub_search"],
            self.optim_state["tol_mesh"],
            self.function_logger,
            True,
            self.non_box_cons,
        )

        # The Acquisition Hedge policy is not yet supported (even in Matlab)
        index_acq = None
        if u_search_set.size > 0:
            # Batch evaluation of the search's acquisition function on the
            # search set, as MATLAB BADS does (bads.m:578)
            z, f_mu, _ = acq_fcn_lcb(
                u_search_set,
                self.function_logger.func_count,
                gp,
                self.options["search_acq_fcn"][1],
            )
            # Evaluate best candidate point in original coordinates (a NaN
            # value is skipped, as by MATLAB's min)
            index_acq = None if np.all(np.isnan(z)) else np.nanargmin(z)

            # TODO: In future handle acquisition portfolio (Acquisition Hedge), it's not even unsupported in Matlab

            # Randomly choose index if something went wrong (every value NaN)
            if index_acq is None:
                self.logger.debug(
                    "Acquisition function failed: a candidate is chosen at "
                    "random"
                )
                index_acq = self.rng.integers(0, len(u_search_set))

            # u_search at the candidate acquisition point
            u_search = u_search_set[index_acq]

            # TODO: Local optimization of the acquisition function (generally it does not improve results)
            if self.options["search_optimize"]:
                pass

            y_search, f_sd_search, idx = self.function_logger(u_search)

            if z.size > 0:
                # Save statistics of gp prediction, with the SD of the
                # observation, the GP's noise included (MATLAB's ys)
                _, y_s2 = gp.predict(np.atleast_2d(u_search), add_noise=True)
                self._save_gp_stats_(
                    y_search, f_mu[index_acq].item(), np.sqrt(y_s2).item()
                )

            # Add search point to training set, except at the last search
            # of a round, as MATLAB BADS does
            if (
                u_search.size > 0
                and self.optim_state["search_count"]
                < self.options["search_n_try"]
            ):
                # TODO: Handle fitness_shaping and rotate gp axes (latter one is unsupported)
                with stage_timer.stage("gp_update"):
                    gp = add_and_update_gp(
                        self.function_logger,
                        gp,
                        u_search,
                        y_search,
                        f_sd_search,
                        self.options,
                    )

            # If the function is non-deterministic we update the posterior of the GP with the new point
            if self.optim_state["uncertainty_handling_level"] > 0:
                with stage_timer.stage("gp_rebuild"):
                    new_gp = copy.deepcopy(gp)
                    # Update priors and posteriors
                    new_gp, _ = local_gp_fitting(
                        new_gp,
                        u_search,
                        self.function_logger,
                        self.options,
                        self.optim_state,
                        self.iteration_history,
                        False,
                        rng=self.rng,
                        timer=stage_timer,
                    )
                if new_gp.temporary_data.get("needs_rebuild", False):
                    # The rebuild failed and `new_gp` is the GP of its
                    # entry, not built around the point: no estimate there,
                    # as in MATLAB BADS, so the search counts as failed.
                    f_mu_search = np.nan
                    f_sd_search = np.nan
                else:
                    f_mu_search, f_sd_search = new_gp.predict(
                        np.atleast_2d(u_search)
                    )
                    f_mu_search = f_mu_search.item()
                    f_sd_search = np.sqrt(f_sd_search).item()
            else:
                f_mu_search = y_search
                f_sd_search = 0.0

            # Compute distance of search point from current point
            search_dist = np.sqrt(
                udist(
                    self.u_best,
                    u_search,
                    gp.temporary_data["len_scale"],
                    self.optim_state["lb"],
                    self.optim_state["ub"],
                    self.optim_state["scale"],
                    self.optim_state["periodic_vars"],
                )
            )

        else:
            # Search set is empty
            u_search = None
            y_search = self.yval
            f_mu_search = self.fval
            f_sd_search = 0.0
            search_dist = 0.0

        # TODO: CMA-ES like estimation of local covariance structure (unused)
        if (
            self.options["hessian_update"]
            and self.options["hessian_method"] == "cmaes"
        ):
            pass

        fval_old = self.fval

        # Evaluate search
        if not self.options["stobads"]:
            search_improvement = self._eval_improvement_(
                self.fval,
                f_mu_search,
                self.fsd,
                f_sd_search,
                self.options["improvement_quantile"],
            )

            # Declare if search was success or not
            is_search_success = (
                search_improvement
                > self.optim_state["search_sufficient_improvement"]
            )
            is_search_improved = (
                search_improvement > 0
                and self.options["sloppy_improvement"]
                or is_search_success
            )

        else:
            # For StoBads an improvement corresponds to a success
            sto_success = self._sto_success_improvement_(
                self.fval,
                f_mu_search,
                self.fsd,
                f_sd_search,
                self.mesh_size,
            )
            is_search_success = sto_success == 1
            if self.options["opp_stobads"] and sto_success == 0:
                # An uncertain outcome moves the incumbent only to a point
                # that improves on it, as an uncertain poll does
                search_improvement = self._eval_improvement_(
                    self.fval,
                    f_mu_search,
                    self.fsd,
                    f_sd_search,
                    self.options["improvement_quantile"],
                )
                is_search_improved = bool(search_improvement > 0)
            else:
                is_search_improved = is_search_success

        # An empty search set is a failed search. MATLAB BADS gives the same
        # status at improvement_quantile <= 0.5 (the default) or without
        # noise, and decays the hedge's gains as below; but at a larger
        # quantile in a noisy run it moves the incumbent to the previous
        # search's point, and it stops with an error when the run's first
        # search set is empty. PyBADS does neither.
        if u_search is None:
            is_search_improved = is_search_success = False

        # A search improvement implies an update of the incumbent
        if is_search_improved:
            if self.options["acq_hedge"]:
                # Acquisition hedge (acquisition portfolio) not supported yet
                pass
            else:
                method = self.search_es_hedge.chosen_search_fun[0]

            # StoBads or sufficient improvement
            if is_search_success:
                self.search_success += 1
                search_string = f"Successful search ({method})"
                self.optim_state["u_success"].append(u_search)
                self.optim_state["y_success"].append(y_search)
                self.optim_state["f_success"].append(f_mu_search)
                search_status = "success"
            else:
                search_string = f"Incremental search ({method})"
                search_status = "incremental"

            # Update incumbent point (self.yval, self.fval, self.fsd) and optim_state
            self._update_incumbent_(
                u_search, y_search, f_mu_search, f_sd_search
            )
            if self.optim_state["uncertainty_handling_level"] > 0:
                gp = new_gp

            self.reset_gp = True

        else:
            search_status = "failure"
            search_string = ""

        # Update portfolio acquisition function (not supported)

        # Update search portfolio (needs improvement); after an empty search
        # set every gain decays, with no reward, as in MATLAB BADS
        if self.search_es_hedge is not None:
            self.search_es_hedge.update_hedge(
                u_search,
                fval_old,
                f_mu_search,
                f_sd_search,
                gp,
                self.optim_state["mesh_size"],
            )

        # Update search statistics and search scale factor
        self._update_search_stats_(search_status, search_dist)

        if len(search_string) > 0:
            self.logging_action.append("")
            # The display counts iterations from 1, as MATLAB BADS does
            self._display_function_log_(
                self.optim_state["iter"] + 1, search_string
            )

        return u_search, search_dist, f_mu_search, f_sd_search, gp

    def _eval_improvement_(self, f_base, f_new, s_base, s_new, q):
        """
        A private method that compute the optimization improvement.

        Returns
        ----------
        z : np.array
            It is the improvement of f_new over f_base for a minimization problem (larger improvements are better).
        """
        if s_base is None or s_new is None:
            z = f_base - f_new
        else:
            # This needs to be corrected -- but for q=0.5 it does not matter
            mu = f_base - f_new
            sigma = np.sqrt(s_base**2 + s_new**2)
            x0 = -np.sqrt(2) * erfcinv(2 * q)
            z = sigma * x0 + mu
            z = z.flatten()

        return z

    def _sto_success_improvement_(
        self,
        f_base,
        f_new,
        s_base,
        s_new,
        frame_size,
    ):
        """
            A private method that evaluates if the improvement in the candidate incumbent using the uncertain interval method proposed in Sto-MADS [1].
        Returns
        ----------
            int : Return a flag integer value
                1   : sucessuful improvement
                0   : uncertain unsuccessful incumbent
                -1  : certain unsuccessful incumbent, or no estimate (a NaN
                      mean or SD)

        References
        ----------
        [1] Audet, Charles, Kwassi Joseph Dzahini, Michael Kokkolaras, and Sébastien Le Digabel. ‘Stochastic Mesh Adaptive Direct Search for Blackbox Optimization Using Probabilistic Estimates’. Computational Optimization and Applications 79, no. 1 (May 2021): 1–34. https://doi.org/10.1007/s10589-020-00249-0.
        """
        epsilon = np.sqrt(s_base**2 + s_new**2)
        mu = f_base - f_new
        # No estimate (a GP that could not take the point): a failure, as on
        # the path without Sto-BADS
        if not (np.isfinite(mu) and np.isfinite(epsilon)):
            return -1
        if self.gamma_uncertain_interval is None:
            gamma = 1.96  # gamma = norminv(0.975)
        else:
            gamma = self.gamma_uncertain_interval

        ub_uncertain_interval = (
            gamma
            * epsilon
            * frame_size ** (self.options["stobads_frame_size_scaling_power"])
        )

        if mu >= ub_uncertain_interval:
            # Successful
            return 1
        elif mu <= -ub_uncertain_interval:
            # Certain unsuccessful
            return -1

        # Uncertain unsuccessful
        return 0

    def _poll_step_(self, gp: GP):
        """
        A private method that performs the poll step, along the directions of
        ``poll_mads_2n`` (the signed coordinate directions at default).
        It also evaluates and update the incumbent and the poll parameters (like the ``mesh_size_integer``) according to the found improvement.

        Returns
        ---------
        u_poll_best : np.array
            Best poll point.
        f_poll_best : float
            GP prediction of the best poll point.
        y_poll_best : float
            Function value at the best poll point.
        f_sd_poll_best : float
            Estimated GP standard deviation at the best poll point.
        gp : gpyreg.gaussian_process.GP
        """

        poll_best_improvement = 0
        u_poll_best = self.u.copy()
        y_poll_best = self.yval
        f_poll_best = self.fval
        f_sd_poll_best = self.fsd
        gp_poll_hyp_best = self.best_gp_hyp.copy()
        poll_count = 0
        certain_good_poll = False
        # Sto-BADS: the best outcome over the poll's points (1 if some point
        # succeeds, 0 if none does and some is uncertain, -1 if every point
        # fails for certain, as in Sto-MADS), and the successful point of the
        # largest improvement
        sto_poll = -1
        sto_best = None
        sto_best_improvement = -np.inf
        B = None
        u_poll = None
        u_new = []
        stage_timer = self._stage_timer

        # Poll loop
        while (
            (
                (u_poll is not None and len(u_poll) > 0)
                or (B is None or len(B) == 0)
            )
            and self.function_logger.func_count < self.options["max_fun_evals"]
            and poll_count < self.D * 2
        ):
            # Fill in basis vectors (when poll_count == 0)
            if B is None or B.size == 0:
                # Create new poll vectors
                B_new = poll_mads_2n(
                    self.D,
                    gp.temporary_data["poll_scale"],
                    self.optim_state["search_mesh_size"],
                    self.optim_state["mesh_size"],
                    rng=self.rng,
                )

                # GP- based vector scaling (poll_scale broadcast)
                vv = (
                    B_new * self.optim_state["mesh_size"]
                ) * gp.temporary_data[
                    "poll_scale"
                ]  # scaling again using broadcast

                # Add vector to current point, enforce periodicity, and fix
                # to grid if asked
                if self.options["force_poll_mesh"]:
                    u_poll_new = force_to_grid_periodic(
                        self.u + vv,
                        self.optim_state["search_mesh_size"],
                        self.lower_bounds,
                        self.upper_bounds,
                        self.optim_state["periodic_vars"],
                    )
                else:
                    u_poll_new = period_check(
                        self.u + vv,
                        self.lower_bounds,
                        self.upper_bounds,
                        self.optim_state["periodic_vars"],
                    )

                u_poll_new = contraints_check(
                    u_poll_new,
                    self.lower_bounds,
                    self.upper_bounds,
                    self.optim_state["tol_mesh"],
                    self.function_logger,
                    False,
                    self.non_box_cons,
                )

                # The polling set and its basis, filled once: B is never
                # emptied, so the basis is not refilled
                u_poll = u_poll_new.copy()
                B = B_new.copy()

            # Cannot refill poll vector set, stop polling
            if u_poll is None or u_poll.size == 0:
                break

            # Check whether it is time to refit the GP. Without poll
            # training, the poll refits only in the first iteration, and
            # records no refit that it does not make (MATLAB BADS records
            # it, and delays the next refit of the search).
            poll_refit = (
                self.options["poll_training"] or self.optim_state["iter"] == 0
            )
            refit_flag, do_gp_calibration = self._is_gp_refit_time_(
                self.options["normalpha_level"], poll_refit
            )

            if (
                poll_refit
                and not refit_flag
                and gp.temporary_data.get("needs_refit", False)
            ):
                # A failed rebuild of the local GP asks for a refit at the
                # next one.
                refit_flag = True
                self._record_gp_refit_()
                do_gp_calibration = False

            # Local GP approximation around polled points
            if (
                refit_flag
                or poll_count == 0
                or self.reset_gp
                or gp.temporary_data.get("needs_rebuild", False)
            ):
                with stage_timer.stage("gp_rebuild"):
                    gp, gp_exit_flag = local_gp_fitting(
                        gp,
                        self.u,
                        self.function_logger,
                        self.options,
                        self.optim_state,
                        self.iteration_history,
                        refit_flag,
                        rng=self.rng,
                        timer=stage_timer,
                    )
                # The rebuild answers a move of the incumbent (see the
                # search step)
                self.reset_gp = False
                if refit_flag:
                    self.gp_refitted_flag = True
                self.gp_exit_flag = np.minimum(self.gp_exit_flag, gp_exit_flag)
                if gp.temporary_data.get("needs_rebuild", False):
                    # The rebuild failed and the GP is the previous one: it
                    # is unreliable, as MATLAB's GP with no posterior is.
                    do_gp_calibration = True

            # Update Target from GP prediction
            with stage_timer.stage("target_from_gp"):
                self._update_target_(u_poll_best, gp, gp_poll_hyp_best)

            # Evaluate acquisition function on poll vectors
            # Batch evaluation of acquisition function on search set (The Acquisition Hedge policy is not yet supported (even in Matlab))
            z, f_mu, fs = acq_fcn_lcb(
                u_poll, self.function_logger.func_count, gp
            )
            # Evaluate best candidate point in original coordinates (a NaN
            # value is skipped, as by MATLAB's min)
            index_acq = None if np.all(np.isnan(z)) else np.nanargmin(z)

            # In future handle acquisition portfolio (Acquisition Hedge), it's even unsupported in Matlab

            # Randomly choose index if something went wrong (every value NaN)
            if index_acq is None:
                self.logger.debug(
                    "Acquisition function failed: a candidate is chosen at "
                    "random"
                )
                index_acq = self.rng.integers(0, len(u_poll))
            # A zero predictive SD makes gamma_z infinite or NaN, which marks
            # the GP as unreliable below: NumPy's warnings are silenced for
            # this division only
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma_z = (
                    self.optim_state["f_target"]
                    - self.sufficient_improvement
                    - f_mu
                ) / fs
            if np.all(np.isfinite(gamma_z)) and np.all(np.isreal(gamma_z)):
                f_pi = 0.5 * erfc(-gamma_z / np.sqrt(2))
                # sort descend, over the points (f_pi is a column), and take
                # the D largest, as MATLAB BADS
                f_pi = np.sort(f_pi, axis=None)[::-1]
                p_less = np.prod(1 - f_pi[: self.D])
            else:
                p_less = 0
                do_gp_calibration = True

            # Consider whether to stop polling
            if not self.options["complete_poll"] and self._is_poll_stop_(
                certain_good_poll, do_gp_calibration, p_less, poll_count
            ):
                break

            # Evaluate function and store the value
            u_new = u_poll[index_acq]
            y_poll, y_sd_poll, f_idx_new = self.function_logger(u_new)

            # Remove polled vector from set.
            u_poll = np.delete(u_poll, index_acq, axis=0)

            # Save statistics of gp prediction, with the SD of the
            # observation, the GP's noise included (MATLAB's ys)
            _, y_s2 = gp.predict(np.atleast_2d(u_new), add_noise=True)
            self._save_gp_stats_(
                y_poll, f_mu[index_acq].item(), np.sqrt(y_s2).item()
            )

            if self.optim_state["uncertainty_handling_level"] > 0:
                # Update posterior with the new polled point
                n_train = gp.X.shape[0]
                with stage_timer.stage("gp_update"):
                    gp = add_and_update_gp(
                        self.function_logger,
                        gp,
                        u_new,
                        y_poll,
                        y_sd_poll,
                        self.options,
                    )  # u_new is already added from the function logger
                if gp.X.shape[0] > n_train:
                    f_poll, f_sd_poll = gp.predict(np.atleast_2d(u_new))
                    f_sd_poll = np.sqrt(f_sd_poll).item()
                    f_poll = f_poll.item()
                else:
                    # The update failed and the GP lacks the point: no
                    # estimate there, as in MATLAB BADS, so the point counts
                    # as no improvement.
                    f_poll = np.nan
                    f_sd_poll = np.nan
            else:
                f_poll = y_poll
                f_sd_poll = 0.0

            poll_improvement = self._eval_improvement_(
                self.fval,
                f_poll,
                self.fsd,
                f_sd_poll,
                self.options["improvement_quantile"],
            )

            # Check if current point improves over best polled point so far
            if poll_improvement > poll_best_improvement:
                u_poll_best = u_new.copy()
                y_poll_best = y_poll
                f_poll_best = f_poll
                f_sd_poll_best = f_sd_poll
                gp_poll_hyp_best = gp.get_hyperparameters(as_array=True)
                poll_best_improvement = poll_improvement

                if not self.options["stobads"]:
                    certain_good_poll = (
                        poll_best_improvement > self.sufficient_improvement
                    )

            # StoBads
            if self.options["stobads"]:
                sto_success = self._sto_success_improvement_(
                    self.fval,
                    f_poll,
                    self.fsd,
                    f_sd_poll,
                    self.mesh_size,
                )
                sto_poll = max(sto_poll, sto_success)
                if (
                    sto_success == 1
                    and poll_improvement > sto_best_improvement
                ):
                    sto_best = (u_new.copy(), y_poll, f_poll, f_sd_poll)
                    sto_best_improvement = poll_improvement
                certain_good_poll = sto_poll == 1

            # Increase poll counter
            poll_count += 1
        # End poll loop

        # Evaluate poll
        if not self.options["stobads"]:
            if (
                poll_best_improvement > 0
                and self.options["sloppy_improvement"]
            ) or poll_best_improvement > self.sufficient_improvement:
                # Update incumbent point (self.yval, self.fval, self.fsd) and optim_state
                self._update_incumbent_(
                    u_poll_best, y_poll_best, f_poll_best, f_sd_poll_best
                )
                is_poll_moved = True
            else:
                is_poll_moved = False
        else:
            # StoBads: a success moves to the successful point, and with
            # opp_stobads an uncertain poll moves to the best polled point
            # if that point improves on the incumbent
            if sto_poll == 1:
                self._update_incumbent_(*sto_best)
                is_poll_moved = True
            elif (
                self.options["opp_stobads"]
                and sto_poll == 0
                and poll_best_improvement > 0
            ):
                self._update_incumbent_(
                    u_poll_best, y_poll_best, f_poll_best, f_sd_poll_best
                )
                is_poll_moved = True
            else:
                is_poll_moved = False

        if certain_good_poll:
            is_sucess_poll_flag = True

            # Check if mesh size is already maximal
            self._check_mesh_overflow_()
            # Successful poll, increase mesh size
            self.mesh_size_integer = np.minimum(
                self.mesh_size_integer + 1,
                self.options["max_poll_grid_number"],
            )

            self.optim_state["u_success"].append(self.u_best.copy())
            self.optim_state["y_success"].append(self.yval)
            self.optim_state["f_success"].append(self.fval)
        else:
            is_sucess_poll_flag = False
            # Failed poll, decrease mesh size
            self.mesh_size_integer -= 1

            # Accelerated mesh reduction if certain unsucessfull or  stalling
            # if self.options['stobads'] and sto_success < 0:
            # certain unsucessfull poll
            #        self.mesh_size_integer -= 1
            # else:
            # Check stalling (MATLAB's iter > AccelerateMeshSteps, with its
            # iter counted from 1)
            iter = self.optim_state["iter"]
            if (
                self.options["accelerate_mesh"]
                and iter >= self.options["accelerate_mesh_steps"]
            ):
                f_base = self.iteration_history.get("fval")[
                    iter - self.options["accelerate_mesh_steps"]
                ]
                f_sd_base = self.iteration_history.get("fsd")[
                    iter - self.options["accelerate_mesh_steps"]
                ]
                self.f_q_historic_improvement = self._eval_improvement_(
                    f_base,
                    self.fval,
                    f_sd_base,
                    self.fsd,
                    self.options["improvement_quantile"],
                )
                if self.f_q_historic_improvement < self.options["tol_fun"]:
                    self.mesh_size_integer -= 1
                    self.logger.debug(
                        "The optimization is stalling, further decrease of the mesh size"
                    )

            self.optim_state["search_size_integer"] = np.minimum(
                self.optim_state["search_size_integer"],
                self.mesh_size_integer * self.options["search_grid_multiplier"]
                - self.options["search_grid_number"],
            )

            # TODO: Profile plot iteration

        # End POLL evaluation

        # Update mesh size
        self.mesh_size = (
            self.options["poll_mesh_multiplier"] ** self.mesh_size_integer
        )
        self.optim_state["mesh_size"] = self.mesh_size

        # Print iteration
        if is_sucess_poll_flag:
            poll_string = "Successful poll"
        else:
            poll_string = "Refine grid"

        # The actions of this pass, built anew at each poll as MATLAB BADS
        # does
        action_str = ""
        if self.gp_refitted_flag:
            action_str = "Train"
            if self.gp_exit_flag < 0:
                action_str += " (failed)"
                # self.gp_exit_flag = np.inf # Reset the flag

        if self.last_skipped == self.optim_state["iter"]:
            action_str = "Skip" if action_str == "" else action_str + ", skip"
        self.logging_action.append(action_str)

        # The display counts iterations from 1, as MATLAB BADS does
        self._display_function_log_(self.optim_state["iter"] + 1, poll_string)

        self.poll_moved = is_poll_moved

        return u_poll_best, f_poll_best, y_poll_best, f_sd_poll_best, gp

    def _save_gp_stats_(self, fval, ymu, ys):
        if (
            self.gp_stats.get("iter_gp") is None
            or len(self.gp_stats.get("iter_gp")) == 0
        ):
            iter = 0
        else:
            iter = self.gp_stats.get("iter_gp")[-1] + 1

        self.gp_stats.record("iter_gp", iter, iter)
        self.gp_stats.record("fval", fval, iter)
        self.gp_stats.record("ymu", ymu, iter)
        self.gp_stats.record("ys", ys, iter)

    def _is_gp_refit_time_(self, alpha, refit_allowed=True):
        """A private method that checks the calibration of the GP prediction and if a fitting is required.
        With ``refit_allowed`` false, no refit is due and none is recorded,
        and the calibration is checked all the same."""
        if self.function_logger.func_count < 200:
            refit_period = np.maximum(10, self.D * 2)
        else:
            refit_period = self.D * 5

        # Number of statistics of the GP prediction since the last refit
        # (MATLAB's gpstats.last)
        gp_iter_idx = self.gp_stats.get("iter_gp")
        n_stats = 0 if gp_iter_idx is None else len(gp_iter_idx)

        do_gp_calibration = False
        # empty stats
        if n_stats == 0:
            do_gp_calibration = True

        # if stats data is available check z_score
        if not do_gp_calibration:
            f_vals = (
                self.gp_stats.get("fval")[:n_stats].flatten().astype("float")
            )
            yvals = (
                self.gp_stats.get("ymu")[:n_stats].flatten().astype("float")
            )

            zscore = f_vals - yvals
            gp_ys = self.gp_stats.get("ys")[:n_stats].flatten().astype("float")

            zscore = zscore / gp_ys

            if np.any(np.isnan(zscore)):
                do_gp_calibration = True
            else:
                n = np.size(zscore)
                if n < 3:
                    # Quantiles of the chi-square distribution with n
                    # degrees of freedom (gppredcheck.m)
                    plo = chi2.ppf(alpha / 2, n)
                    phi = chi2.ppf(1 - alpha / 2, n)
                    total = np.sum(zscore**2)
                    if (
                        total < plo
                        or total > phi
                        or np.any(np.isnan(plo))
                        or np.any(np.isnan(phi))
                    ):
                        do_gp_calibration = True
                    else:
                        do_gp_calibration = False
                elif np.ptp(zscore) == 0:
                    # z-scores without spread (the GP predicted its last
                    # evaluations exactly) give MATLAB's swtest.m W = 0/0,
                    # a p-value of NaN and no calibration; SciPy's shapiro
                    # warns on them and returns NaN or 1, by its version
                    do_gp_calibration = False
                else:
                    shapiro_test = shapiro(zscore)
                    do_gp_calibration = shapiro_test.pvalue < alpha

        func_count = self.function_logger.func_count

        refit_flag = (
            refit_allowed
            and self.optim_state["lastfitgp"]
            < (func_count - self.options["min_refit_time"])
            and (n_stats >= refit_period or do_gp_calibration)
            and func_count > self.D
        )

        if refit_flag:
            self._record_gp_refit_()
            do_gp_calibration = False

        return refit_flag, do_gp_calibration

    def _is_poll_stop_(
        self, certain_good_poll, do_gp_calibration, p_less, poll_count
    ):
        """A private method that decides whether to stop polling before the
        next poll vector, from whether a poll was good so far, whether the
        GP is unreliable (``do_gp_calibration``) and the probability
        ``p_less`` that no poll vector improves. A stop without a good poll
        is recorded in ``self.last_skipped``."""
        # Stop polling if last poll was good
        if certain_good_poll:
            if do_gp_calibration:
                return True  # GP is unreliable, just stop polling
            # Use GP prediction whether to stop polling
            return p_less > 1 - self.options["tol_poi"]
        # No good polling so far -- if GP is reliable, stop polling
        # If probability of improvement at any location is to low
        if (
            not do_gp_calibration
            and (
                self.options["consecutive_skipping"]
                or self.last_skipped < self.optim_state["iter"] - 1
            )
            and poll_count >= self.options["min_failed_poll_steps"]
            and p_less > (1 - self.options["tol_poi"])
        ):
            self.last_skipped = self.optim_state["iter"]
            return True
        return False

    def _record_gp_refit_(self):
        """A private method that records a refit of the GP hyperparameters:
        the evaluation count at the refit, and a reset of the GP
        statistics."""
        self.optim_state["lastfitgp"] = self.function_logger.func_count

        # Reset the GP statistics
        self.gp_stats = IterationHistory(
            [
                "iter_gp",
                "fval",
                "ymu",
                "ys",
                "gp",
            ]
        )

    def _update_target_(self, u, gp: GP, hyp_best):
        """A private method that stores in ``optim_state`` the optimization
        target of ``_get_target_from_gp_`` at ``u``: ``f_target_mu``,
        ``f_target_s`` and ``f_target``, as MATLAB's ``UpdateTarget``
        does."""
        f_target_mu, f_target_s, f_target = self._get_target_from_gp_(
            u, gp, hyp_best
        )
        self.optim_state["f_target_mu"] = f_target_mu.item()
        self.optim_state["f_target_s"] = (
            f_target_s if np.isscalar(f_target_s) else f_target_s.copy()
        )
        self.optim_state["f_target"] = f_target.item()

    def _get_target_from_gp_(self, u, gp: GP, hyp_best):
        """A private method that retrieves the prediction of the GP at the
        input ``u`` and sets the optimization target ``f_target`` slightly
        below the mean prediction, in a noisy run and whenever
        ``uncertain_incumbent`` is on (the default); a prediction that is
        not finite is replaced by the incumbent's ``fval`` and ``fsd``.
        Otherwise the target is the incumbent's ``fval`` less ``tol_fun``.

        Parameters
        ----------
        u : np.ndarray
            The input point, the incumbent.
        gp : GP
            The GP.
        hyp_best : np.ndarray
            The hyperparameters under which the GP predicts: from a copy of
            the GP whose posterior is recomputed under them, or from the GP
            itself when they are its own.

        Returns
        -------
        f_target_mu : np.ndarray
            The GP's mean prediction at ``u``, of shape ``(1, 1)`` (the
            incumbent's ``fval`` when the prediction is not finite, and when
            no prediction is made).
        f_target_s : np.ndarray or float
            The GP's predictive standard deviation at ``u`` (the incumbent's
            ``fsd`` when the prediction is not finite, and ``np.zeros(1)``
            when no prediction is made).
        f_target : np.ndarray
            The optimization target, of shape ``(1, 1)``.
        """
        # Corresponds to Matlab: updateTarget
        if (
            self.optim_state["uncertainty_handling_level"] > 0
            or self.options["uncertain_incumbent"]
        ):
            if np.array_equal(hyp_best, gp.get_hyperparameters(as_array=True)):
                # The GP's posterior is the one under `hyp_best` on its data,
                # which a copy recomputed under them would give again, bit
                # for bit, as long as every update recomputes it in full
                # (`add_and_update_gp`): predict from it
                f_target_mu, fs2 = gp.predict(np.atleast_2d(u))
            else:
                tmp_gp = copy.deepcopy(gp)
                try:
                    tmp_gp.set_hyperparameters(hyp_best)
                    f_target_mu, fs2 = tmp_gp.predict(np.atleast_2d(u))
                except np.linalg.LinAlgError:
                    # The posterior under `hyp_best` cannot be computed:
                    # predict from the GP as it stands, whose posterior
                    # matches its data (MATLAB's UpdateTarget reuses the
                    # current posterior).
                    self.logger.debug(
                        "GP posterior under the best "
                        "hyperparameters failed; target predicted from the "
                        "current GP"
                    )
                    f_target_mu, fs2 = gp.predict(np.atleast_2d(u))

            f_target_s = np.sqrt(np.max(fs2, axis=0))
            if (
                ~np.isfinite(f_target_mu)
                | ~np.isreal(f_target_s)
                | ~np.isfinite(f_target_s)
            ):
                # An array, as the prediction is: the callers call `.item()`.
                f_target_mu = np.atleast_2d(
                    np.asarray(self.optim_state["fval"], dtype=float)
                )
                f_target_s = self.optim_state["fsd"]
                # The target's formula takes the incumbent's variance too,
                # where MATLAB BADS keeps the prediction's (bads.m:1321)
                fs2 = f_target_s**2

            # f_target: Set optimization target slightly below the current incumbent
            if self.options["alternative_incumbent"]:
                f_target = (
                    f_target_mu
                    - np.sqrt(self.D)
                    / np.sqrt(self.function_logger.func_count)
                    * f_target_s
                )
            else:
                f_target = f_target_mu - self.optim_state[
                    "sd_level"
                ] * np.sqrt(fs2 + self.options["tol_fun"] ** 2)
        else:
            # Arrays of the prediction's shapes: the callers call `.item()`
            f_target_mu = np.atleast_2d(
                np.asarray(self.optim_state["fval"], dtype=float)
            )
            f_target_s = np.zeros(1)
            f_target = f_target_mu - self.options["tol_fun"]

        return f_target_mu, f_target_s, f_target

    def _update_search_bounds_(self):
        lb = self.optim_state["lb"]
        lb_search = force_to_grid(lb, self.optim_state["search_mesh_size"])
        lb_search[lb_search < lb] = (
            lb_search[lb_search < lb] + self.optim_state["search_mesh_size"]
        )

        ub = self.optim_state["ub"]
        ub_search = force_to_grid(ub, self.optim_state["search_mesh_size"])
        ub_search[ub_search > ub] = (
            ub_search[ub_search > ub] - self.optim_state["search_mesh_size"]
        )
        return lb_search, ub_search

    def _update_incumbent_(self, u_new, yval_new, fval_new, fsd_new):
        """
        Move the incumbent (current point) to a new point.
        """
        self.optim_state["u"] = u_new.copy()
        self.optim_state["yval"] = yval_new
        self.optim_state["fval"] = fval_new
        self.optim_state["fsd"] = fsd_new
        self.u = u_new.copy()
        self.u_best = u_new.copy()
        self.yval = yval_new
        self.fval = fval_new
        self.fsd = fsd_new
        return yval_new, fval_new, fsd_new
        # Update estimate of curvature (Hessian) - not supported (GP usage)

    def _update_search_stats_(self, search_status, search_dist):
        if (
            not "search_stats" in self.optim_state
            or len(self.optim_state["search_stats"]) == 0
        ):
            search_stats = {}
            search_stats["log_search_factor"] = []
            search_stats["success"] = []
            search_stats["udist"] = []
            self.optim_state["search_stats"] = search_stats
        else:
            search_stats = self.optim_state["search_stats"]

        search_stats["log_search_factor"].append(
            np.log(self.optim_state["search_factor"])
        )
        search_stats["udist"].append(search_dist)

        if search_status == "success":
            search_stats["success"].append(1.0)
            self.optim_state["search_factor"] = (
                self.optim_state["search_factor"]
                * self.options["search_scale_success"]
            )
            if self.options["adaptive_incumbent_shift"]:
                self.optim_state["sd_level"] = self.optim_state["sd_level"] * 2

        elif search_status == "incremental":
            search_stats["success"].append(0.5)
            self.optim_state["search_factor"] = (
                self.optim_state["search_factor"]
                * self.options["search_scale_incremental"]
            )
            if self.options["adaptive_incumbent_shift"]:
                self.optim_state["sd_level"] = (
                    self.optim_state["sd_level"] * 2**2
                )

        elif search_status == "failure":
            search_stats["success"].append(0.0)
            self.optim_state["search_factor"] = np.maximum(
                self.options["search_factor_min"],
                self.optim_state["search_factor"]
                * self.options["search_scale_failure"],
            )
            if self.options["adaptive_incumbent_shift"]:
                self.optim_state["sd_level"] = np.maximum(
                    self.options["incumbent_sigma_multiplier"],
                    self.optim_state["sd_level"] / 2,
                )

        # Reset search factor at the end of each search
        if self.optim_state["search_count"] == self.options["search_n_try"]:
            self.optim_state["search_factor"] = 1

        return search_stats

    def _re_evaluate_history_(self, gp: GP):
        """A private method used in the case of a stochastic target function.
        It re-estimates the value and the SD of the target at the incumbent of
        each iteration, from the current data, as MATLAB BADS's
        ``reevaluateIterList`` does: a copy of the working GP ``gp``, with the
        hyperparameters recorded at the end of that iteration, is rebuilt
        around the incumbent without a refit. The GPs stored in the iteration
        history are left as they were recorded.

        An iterate whose rebuild fails has no estimate: its value and SD are
        NaN, as in MATLAB BADS, except for the current iterate, the last,
        which keeps the estimate it was recorded with.
        """
        if self.optim_state["last_re_eval"] != self.function_logger.func_count:
            # Re-evaluate gp outputs
            u_history = self.iteration_history.get("u")
            hyp_history = self.iteration_history.get("gp_hyp_full")
            n_iter = u_history.shape[0]
            tmp_gp = copy.deepcopy(gp)
            stage_timer = self._stage_timer
            for i in range(n_iter):
                u = u_history[i]
                tmp_gp.set_hyperparameters(
                    hyp_history[i], compute_posterior=False
                )
                with stage_timer.stage("gp_rebuild"):
                    tmp_gp, _ = local_gp_fitting(
                        tmp_gp,
                        u,
                        self.function_logger,
                        self.options,
                        self.optim_state,
                        self.iteration_history,
                        False,
                        rng=self.rng,
                        timer=stage_timer,
                    )
                if tmp_gp.temporary_data.get("needs_refit", False):
                    # The rebuild failed, and the GP has no posterior
                    if i == n_iter - 1:
                        continue
                    fval, fsd = np.nan, np.nan
                else:
                    fval, fsd = tmp_gp.predict(np.atleast_2d(u))
                    fval = fval.item()
                    fsd = np.sqrt(fsd).item()

                self.iteration_history.record("fval", fval, i)
                self.iteration_history.record("fsd", fsd, i)

            self.optim_state["last_re_eval"] = self.function_logger.func_count

    def _check_mesh_overflow_(self):
        if self.mesh_size_integer == self.options["max_poll_grid_number"]:
            self.mesh_overflows += 1
            if self.mesh_overflows == np.ceil(
                self.options["mesh_overflow_warning"]
            ):
                self.logger.warning(
                    "The mesh attempted to expand above maximum size too many times. Try widening plausible_lower_bounds and plausible_upper_bounds."
                )

    def _log_column_headers(self):
        """
        Private method to log the column headers for the iteration log.
        """
        if self.optim_state["uncertainty_handling_level"] > 0:
            self.logger.info(
                " Iteration    f-count      E[f(x)]        SD[f(x)]           MeshScale          Method              Actions"
            )
        else:
            self.logger.info(
                " Iteration    f-count         f(x)           MeshScale          Method             Actions"
            )

    def _setup_logging_display_format(self):
        """
        Private method to set up the display format for logging the iterations.
        """
        if self.optim_state["uncertainty_handling_level"] > 0:
            display_format = " {:5.0f}       {:5.0f}    {:12.6g}    "
            display_format += "{:12.6g}    {:12.6g}      {:^20s}        {}"
        else:
            display_format = " {:5.0f}       {:5.0f}    {:12.6g}    "
            display_format += "{:12.6g}     {:^20s}        {}"

        return display_format

    def _display_function_log_(self, iteration, method):
        if self.optim_state["uncertainty_handling_level"] > 0:
            self.logger.info(
                self.display_format.format(
                    iteration,
                    self.function_logger.func_count,
                    self.fval,
                    self.fsd,
                    self.optim_state["mesh_size"],
                    method,
                    "".join(self.logging_action[-1]),
                )
            )
        else:
            self.logger.info(
                self.display_format.format(
                    iteration,
                    self.function_logger.func_count,
                    float(self.fval),
                    self.optim_state["mesh_size"],
                    method,
                    "".join(self.logging_action[-1]),
                )
            )
