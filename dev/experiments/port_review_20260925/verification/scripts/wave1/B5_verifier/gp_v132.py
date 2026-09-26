"""Module for Gaussian Processes."""

import math
import numbers
import time
import warnings
from textwrap import indent
from typing import Union

import gpyreg.covariance_functions
import gpyreg.isotropic_covariance_functions as isotropic_covariance
import gpyreg.mean_functions
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from gpyreg.f_min_fill import (
    f_min_fill,
    smoothbox_cdf,
    smoothbox_sf,
    smoothbox_student_t_cdf,
    smoothbox_student_t_sf,
)
from gpyreg.formatting import full_repr
from gpyreg.rng import random_integer, resolve_rng
from gpyreg.slice_sample import SliceSampler, _whole_number

# Reuse the Cholesky factor across consecutive log-posterior evaluations of
# one fit when only mean-function hyperparameters moved (see
# GP.__core_computation). Module-level so a test can switch it off.
_REUSE_CHOLESKY = True


# These bundled covariance implementations return a fresh cross-kernel matrix
# on every call. Keep the original method objects so later class or instance
# overrides take the defensive-copy path.
_ZERO_COPY_CROSS_COVARIANCE_COMPUTES = {
    gpyreg.covariance_functions.SquaredExponential: (
        gpyreg.covariance_functions.SquaredExponential.compute
    ),
    gpyreg.covariance_functions.Matern: (
        gpyreg.covariance_functions.Matern.compute
    ),
    gpyreg.covariance_functions.RationalQuadraticARD: (
        gpyreg.covariance_functions.RationalQuadraticARD.compute
    ),
    isotropic_covariance.MaternIsotropic: (
        isotropic_covariance.MaternIsotropic.compute
    ),
    isotropic_covariance.SquaredExponentialIsotropic: (
        isotropic_covariance.SquaredExponentialIsotropic.compute
    ),
}


def _can_retain_cross_covariance(covariance):
    """Return whether a covariance method produces fresh bundled matrices."""
    expected = _ZERO_COPY_CROSS_COVARIANCE_COMPUTES.get(type(covariance))
    compute = getattr(covariance, "compute", None)
    return (
        expected is not None
        and type(covariance).compute is expected
        and getattr(compute, "__self__", None) is covariance
        and getattr(compute, "__func__", None) is expected
    )


def _solve_triangular(a, b, trans=0, lower=False):
    """``scipy.linalg.solve_triangular(a, b, trans, lower,
    check_finite=False)`` without scipy's Python layers.

    Calls the same LAPACK routine (``?trtrs``) with scipy's layout rule (a
    Fortran-contiguous ``a`` is passed as is; otherwise ``a.T`` with
    ``lower`` and ``trans`` flipped, which avoids a copy), so the result is
    bit-identical to scipy's; the per-call cost drops from about 30 us to
    5 us, which matters where the solve is one of thousands of small ones
    (``predict`` per hyperparameter sample, the log-posterior evaluations
    of the slice sampler). ``a`` and ``b`` are float arrays, ``b`` 2-D.
    """
    (trtrs,) = sp.linalg.get_lapack_funcs(("trtrs",), (a, b))
    if a.flags.f_contiguous or trans == 2:
        x, info = trtrs(a, b, lower=lower, trans=trans, unitdiag=False)
    else:
        x, info = trtrs(
            a.T, b, lower=not lower, trans=1 - trans, unitdiag=False
        )
    if info == 0:
        return x
    if info > 0:
        raise sp.linalg.LinAlgError(
            f"singular matrix: resolution failed at diagonal {info - 1}"
        )
    raise ValueError(f"illegal value in {-info}-th argument of internal trtrs")


def _check_hyperparameter_names(given, hyper_info, argument):
    """Refuse a dictionary that names a hyperparameter the model has not."""
    if given is None:
        return
    known = {info[0] for info in hyper_info}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ValueError(
            f"Unknown hyperparameter(s) in `{argument}`: "
            + ", ".join(unknown)
            + ". The hyperparameters of this GP are "
            + ", ".join(info[0] for info in hyper_info)
            + "."
        )


def _write_prior_block(hyper_priors, name, i, prior_type, prior_params):
    """Write the prior of the hyperparameter block ``name``, at the indices
    ``i`` of the arrays of ``hyper_priors``, as :py:meth:`GP.set_priors`
    takes it, and refuse one that it does not take. Return whether a
    coordinate of the block has a prior."""
    if prior_type == "gaussian":
        mu, sigma = prior_params
        hyper_priors["mu"][i] = mu
        hyper_priors["sigma"][i] = sigma
        # Zero degrees of freedom flag the Gaussian families; an infinite
        # number does too.
        hyper_priors["df"][i] = 0
    elif prior_type == "student_t":
        mu, sigma, df = prior_params
        hyper_priors["mu"][i] = mu
        hyper_priors["sigma"][i] = sigma
        hyper_priors["df"][i] = df
    elif prior_type == "smoothbox":
        a, b, sigma = prior_params
        hyper_priors["a"][i] = a
        hyper_priors["b"][i] = b
        hyper_priors["sigma"][i] = sigma
        # Zero degrees of freedom flag the Gaussian families; an infinite
        # number does too.
        hyper_priors["df"][i] = 0
    elif prior_type == "smoothbox_student_t":
        a, b, sigma, df = prior_params
        hyper_priors["a"][i] = a
        hyper_priors["b"][i] = b
        hyper_priors["sigma"][i] = sigma
        hyper_priors["df"][i] = df
    else:
        raise ValueError("Unknown hyperprior type " + prior_type)

    # The location of a coordinate is `mu` for the Gaussian and Student's t
    # families and the box `[a, b]` for the smooth-box ones, whose `mu`
    # stays NaN. A coordinate whose location and `sigma` are both NaN has
    # no prior, and every other coordinate needs a finite location, with
    # `a <= b` for a box, and a finite, positive `sigma`. A box of zero
    # width, `a == b`, has no plateau and a normalizer of one: it is the
    # Gaussian or the Student's t centred at `a`. `gplite_hypprior.m` reads
    # a coordinate as having no prior where its `mu` or its `sigma` is not
    # finite, a reading that the smooth-box families, gpyreg's own, cannot
    # share.
    if prior_type in ("smoothbox", "smoothbox_student_t"):
        location = np.vstack((hyper_priors["a"][i], hyper_priors["b"][i]))
        location_name = "end of its box (a or b)"
    else:
        location = hyper_priors["mu"][i][None, :]
        location_name = "mu"
    scale = hyper_priors["sigma"][i]
    has_prior = ~(np.all(np.isnan(location), axis=0) & np.isnan(scale))
    scale = scale[has_prior]
    location = location[:, has_prior]
    problem = None
    if np.any(np.isnan(scale)):
        problem = "a NaN sigma where its location is not NaN"
    elif np.any(np.isinf(scale)):
        problem = "an infinite sigma"
    elif np.any(scale <= 0.0):
        problem = "a sigma that is zero or negative"
    elif np.any(np.isnan(location)):
        problem = f"a NaN {location_name} where its sigma is not NaN"
    elif np.any(np.isinf(location)):
        problem = f"an infinite {location_name}"
    elif prior_type in ("smoothbox", "smoothbox_student_t") and (
        np.any(location[0] > location[1])
    ):
        problem = "a lower end a of its box above its upper end b"
    if problem is not None:
        raise ValueError(
            f"The prior of {name} has {problem}. A prior needs a finite "
            "location, with a <= b for a smooth box, and a finite, positive "
            "sigma; a hyperparameter without a prior is set to `None`, and a "
            "coordinate of a block without a prior has NaN for both its "
            "location and its sigma."
        )
    return bool(np.any(has_prior))


class GP:
    """
    A single Gaussian Process (GP).

    Parameters
    ==========
    D : int
        The dimension of the Gaussian Process.
    covariance : object
        The covariance function to use. This can be one of the objects
        from the following module: :py:mod:`gpyreg.covariance_functions`.
    mean : object
        The mean function to use. This can be one of the objects from the
        following module: :py:mod:`gpyreg.mean_functions`.
    noise : object
        The noise function to use. This can be one of the objects from the
        following module: :py:mod:`gpyreg.noise_functions`.
    """

    def __init__(
        self, D: int, covariance: object, mean: object, noise: object
    ):
        self.D = D
        self.covariance = covariance
        self.mean = mean
        self.noise = noise
        self.s2 = None
        self.X = None
        self.y = None
        self.posteriors = None
        # This is necessary as a flag for set_bounds to not do anything
        # before set_priors has been called.
        self.no_prior = None
        self.normalization_constants = None
        self.set_bounds()
        self.set_priors()

        # dict to store temporary data e.g. for pyvbmc
        self.temporary_data = {}

    def __repr__(self):
        return full_repr(
            self,
            "GP",
            order=[
                "D",
                "covariance",
                "mean",
                "noise",
                "X",
                "y",
                "s2",
                "lower_bounds",
                "upper_bounds",
                "posteriors",
            ],
            exclude=["_prior_cache"],
        )

    def __str__(self):
        dimension = "Dimension: " + str(self.D) + "\n"

        cov_N = self.covariance.hyperparameter_count(self.D)
        cov = "Covariance function: " + self.covariance.__class__.__name__
        if self.covariance.__class__.__name__ == "Matern":
            cov += "(degree=" + str(self.covariance.degree) + ")\n"
        if cov_N == 1:
            cov += ", " + str(cov_N) + " parameter\n"
        else:
            cov += ", " + str(cov_N) + " parameters\n"

        mean_N = self.mean.hyperparameter_count(self.D)
        mean = "Mean function: " + self.mean.__class__.__name__
        if mean_N == 1:
            mean += ", " + str(mean_N) + " parameter\n"
        else:
            mean += ", " + str(mean_N) + " parameters\n"

        noise_N = self.noise.hyperparameter_count()
        noise = "Noise function: " + self.noise.__class__.__name__
        if np.any(self.noise.parameters):
            noise += "("
            add_flag = False
            if self.noise.parameters[0] == 1:
                noise += "constant_add=True"
                add_flag = True

            if self.noise.parameters[1] == 1:
                if add_flag:
                    noise += ", "
                noise += "user_provided_add=True"

            if self.noise.parameters[1] == 2:
                if add_flag:
                    noise += ", "
                noise += "scale_user_provided=True"

            if self.noise.parameters[2] == 1:
                if add_flag:
                    noise += ", "
                noise += "rectified_linear_output_dependent_add=True"

            noise += ")"

        if noise_N == 1:
            noise += ", " + str(noise_N) + " parameter\n"
        else:
            noise += ", " + str(noise_N) + " parameters\n"

        priors = "Hyperparameter priors: "
        if self.no_prior:
            priors += "none\n"
        else:
            priors += "present\n"
        samples = "Hyperparameter samples: "
        if self.posteriors is None:
            samples += "0"
        else:
            samples += str(np.size(self.posteriors))

        title = "GP:\n"
        body = dimension + cov + mean + noise + priors + samples
        return title + indent(body, "    ")

    def set_bounds(self, bounds: dict = None):
        """
        Set the hyperparameter lower and upper bounds.

        Parameters
        ==========
        bounds : dict, optional
            A dictionary of GP hyperparameter names and tuples of their lower
            and upper bounds. All hyperparameters need to appear in the
            dictionary. Use the value ``None`` to set the bounds of a specific
            hyperparameter to the default recommended values. If
            ``bounds=None``, all hyperparameter bounds will be set to their
            default recommended values.

        Raises
        ------
        ValueError
            Raised when `bounds` is missing the entry of an expected
            hyperparameter.
        ValueError
            Raised when `bounds` is given, but a specified hyperparameter
            is unknown.
        ValueError
            Raised when a lower bound is above the upper bound of the same
            hyperparameter. Equal bounds fix a hyperparameter.
        """

        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()
        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        hyp_N = cov_N + mean_N + noise_N
        _check_hyperparameter_names(bounds, hyper_info, "bounds")
        lower_bounds = np.full((hyp_N,), np.nan)
        upper_bounds = np.full((hyp_N,), np.nan)

        lower = 0

        for info in hyper_info:
            if bounds is None:
                vals = None
            else:
                try:
                    vals = bounds[info[0]]
                except KeyError as _:
                    e_str = "Missing hyperparameter " + info[0]
                    raise ValueError(e_str) from None

            # None indicates no bounds.
            if vals is not None:
                upper = lower + info[1]
                lb, ub = vals
                i = range(lower, upper)
                lower_bounds[i] = lb
                upper_bounds[i] = ub

            lower += info[1]

        # An inverted pair is a mistake, as `get_recommended_bounds` and
        # `fit` hold it to be.
        inverted = lower_bounds > upper_bounds
        if np.any(inverted):
            raise ValueError(
                "Lower bound above upper bound for the hyperparameter(s) "
                + ", ".join(self.__hyperparameter_names(inverted))
                + "."
            )

        # Only set the bounds here due to exceptions
        # so that we don't only update say half of the bounds.
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self._prior_cache = None  # see __prior_masks

        # Make sure set_priors has been called so we can
        # recompute these.
        if self.no_prior is not None:
            self.__recompute_normalization_constants()

    def get_bounds(self):
        """
        Gets a dictionary with the current lower and upper bounds of the
        hyperparameters.

        Returns
        =======
        bounds_dict : dict
            A dictionary of the current hyperparameter names and their bounds.
        """
        return self.bounds_to_dict(self.lower_bounds, self.upper_bounds)

    def bounds_to_dict(
        self, lower_bounds: np.ndarray, upper_bounds: np.ndarray
    ):
        """
        Convert the given hyperparameter lower and upper bounds to a dict.

        Parameters
        ==========
        lower_bounds : ndarray, shape (hyp_N,)
            The lower bounds.
        upper_bounds : ndarray, shape (hyp_N,)
            The upper bounds.

        Returns
        =======
        bounds_dict : dict
            A dictionary of the current hyperparameter names and tuples
            of their lower and upper bounds.
        """

        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_hyper_info = self.noise.hyperparameter_info()
        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        bounds_dict = {}
        lower = 0

        for info in hyper_info:
            upper = lower + info[1]
            i = range(lower, upper)
            bounds_dict[info[0]] = (lower_bounds[i], upper_bounds[i])
            lower += info[1]

        return bounds_dict

    def __hyperparameter_names(self, mask):
        """The names of the hyperparameter blocks that a boolean mask over
        the hyperparameter vector touches."""
        return [
            name
            for name, pair in self.bounds_to_dict(mask, mask).items()
            if np.any(pair[0])
        ]

    def get_recommended_bounds(self, lower_bounds=None, upper_bounds=None):
        """
        Return the recommended hyperparameter lower and upper bounds as a dict.

        Parameters
        ----------
        lower_bounds : ndarray, optional
            If present, override the recommended lower bounds with these
            values. Any `nan` values will be replaced by the corresponding
            recommended bounds. Defaults to all `nan` values.
        upper_bounds : ndarray, optional
            If present, override the recommended upper bounds with these
            values. Any `nan` values will be replaced by the corresponding
            recommended bounds. Defaults to all `nan` values.

        Returns
        -------
        bounds_dict : dict
            A dictionary of the hyperparameter names and tuples of lower
            upper bounds.

        Raises
        ------
        ValueError
            Raise when GP does not have `X` or `y` set yet, or when provided
            bounds are not one of `"recommended"`/`None`, `"current"`, or
            array_like.
        ValueError
            Raised when a lower bound given is above the upper bound given
            for the same hyperparameter. A pair that comes out inverted
            once the recommendations fill its NaN entries is collapsed
            onto its lower bound instead.
        ValueError
            Raised when a column of the training inputs has no spread, as
            every column of a single training point has none, and a
            hyperparameter whose recommended bounds take their scale from
            the spread of that column (a length scale of the kernel, or
            the scale of :class:`gpyreg.mean_functions.NegativeQuadratic`)
            is not given a finite lower bound: its recommended bounds are
            both ``-inf``, and an upper bound left unset collapses onto
            the lower bound given.
        """
        if self.X is None or self.y is None:
            raise ValueError("GP does not have X or y set!")

        if not isinstance(lower_bounds, (list, tuple, np.ndarray)):
            if lower_bounds == "current":
                # Use existing bounds; fill any nan values with recommended
                # bounds
                lower_bounds = self.lower_bounds.copy()
            elif lower_bounds is None or lower_bounds == "recommended":
                # Use all recommended bounds
                lower_bounds = np.full_like(self.lower_bounds, np.nan)
            else:
                raise ValueError(
                    "`lower_bounds` should be 'recommended'/`None`, 'current',"
                    " or an array."
                )
        if not isinstance(upper_bounds, (list, tuple, np.ndarray)):
            if upper_bounds == "current":
                # Use existing bounds; fill any nan values with recommended
                # bounds
                upper_bounds = self.upper_bounds.copy()
            elif upper_bounds is None or upper_bounds == "recommended":
                # Use all recommended bounds
                upper_bounds = np.full_like(self.upper_bounds, np.nan)
            else:
                raise ValueError(
                    "`upper_bounds` should be 'recommended'/`None`, 'current',"
                    " or an array."
                )
        # Otherwise, use provided arrays as bounds, replacing nan values with
        # recommended bounds. `np.array` takes any array_like and copies,
        # so the caller's own arrays are not written into below.
        lower_bounds = np.array(lower_bounds, dtype=float)
        upper_bounds = np.array(upper_bounds, dtype=float)

        # A pair the caller gave inverted is a mistake. A recommended pair
        # that comes out inverted on a degenerate training set is
        # collapsed below, as `gplite_train.m:142` collapses it.
        inverted = lower_bounds > upper_bounds
        if np.any(inverted):
            raise ValueError(
                "Lower bound above upper bound for the hyperparameter(s) "
                + ", ".join(self.__hyperparameter_names(inverted))
                + "."
            )

        cov_N = self.covariance.hyperparameter_count(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        cov_bounds_info = self.covariance.get_bounds_info(self.X, self.y)
        mean_bounds_info = self.mean.get_bounds_info(self.X, self.y)
        noise_bounds_info = self.noise.get_bounds_info(self.X, self.y)

        lb = lower_bounds
        ub = upper_bounds

        lb_cov = lb[0:cov_N]
        lb_noise = lb[cov_N : cov_N + noise_N]
        lb_mean = lb[cov_N + noise_N : cov_N + noise_N + mean_N]

        lb_cov[np.isnan(lb_cov)] = cov_bounds_info["LB"][np.isnan(lb_cov)]
        lb_noise[np.isnan(lb_noise)] = noise_bounds_info["LB"][
            np.isnan(lb_noise)
        ]
        lb_mean[np.isnan(lb_mean)] = mean_bounds_info["LB"][np.isnan(lb_mean)]

        ub_cov = ub[0:cov_N]
        ub_noise = ub[cov_N : cov_N + noise_N]
        ub_mean = ub[cov_N + noise_N : cov_N + noise_N + mean_N]

        ub_cov[np.isnan(ub_cov)] = cov_bounds_info["UB"][np.isnan(ub_cov)]
        ub_noise[np.isnan(ub_noise)] = noise_bounds_info["UB"][
            np.isnan(ub_noise)
        ]
        ub_mean[np.isnan(ub_mean)] = mean_bounds_info["UB"][np.isnan(ub_mean)]

        lb = np.concatenate([lb_cov, lb_noise, lb_mean])
        ub = np.concatenate([ub_cov, ub_noise, ub_mean])

        # The recommendations take the scale of a length scale from the
        # width of its column of the training inputs (of every column,
        # for an isotropic kernel), through its logarithm, and so do they
        # for the scale of the negative quadratic mean. A column without
        # spread has width zero, which puts both bounds at -inf: a box
        # that holds no value, which the optimizer of `fit` cannot take,
        # and from a lower bound of -inf the fit can reach a scale of
        # zero, where the predictions are NaN. A single training point
        # has no spread in any column. Such a scale is fitted from a
        # finite lower bound that the caller gives it, onto which an
        # upper bound left unset collapses below.
        recommended_ub = np.concatenate(
            [
                cov_bounds_info["UB"],
                noise_bounds_info["UB"],
                mean_bounds_info["UB"],
            ]
        )
        empty = (recommended_ub == -np.inf) & ~np.isfinite(lb)
        if np.any(empty):
            names = ", ".join(self.__hyperparameter_names(empty))
            width = np.max(self.X, axis=0) - np.min(self.X, axis=0)
            columns = ", ".join(
                f"X[:, {j}]" for j in np.flatnonzero(width == 0)
            )
            if columns:
                reason = (
                    f"The training inputs have no spread in {columns}: "
                    "each of these columns holds a single value. The "
                    f"recommended bounds of {names}, which take their "
                    "scale from that spread, are empty (-inf to -inf)"
                )
            else:
                reason = (
                    f"The recommended upper bound of {names} is -inf, "
                    "which leaves no value"
                )
            raise ValueError(
                reason + ", so these hyperparameters need a finite lower "
                "bound from the caller (`set_bounds`, or the option "
                "`lower_bounds` of `fit`), which is their upper bound as "
                "well where that is left unset."
            )

        ub = np.maximum(lb, ub)

        return self.bounds_to_dict(lb, ub)

    def get_priors(self):
        """
        Return the current hyperparameter priors as a dict.

        Returns
        =======
        hyper_priors : dict
            A dictionary of the current hyperparameter names and their
            priors, in the form :py:meth:`set_priors` takes, which
            ``set_priors`` writes back as the GP holds them, so that
            ``set_priors(get_priors())`` changes nothing. A hyperparameter
            has ``None``, as ``set_priors`` takes it for no prior, where
            all of its prior's entries are NaN. Any other prior holds the
            arrays of its entries, NaN included. A block with a prior in
            some coordinate comes back under the name of the family it was
            set with, or of the matching Gaussian family where a Student's
            t family was set with zero degrees of freedom throughout. A
            block with no prior in any coordinate, whose location and
            ``sigma`` are NaN throughout, comes back under the Gaussian or
            the Student's t family that its degrees of freedom name:
            ``"gaussian"`` where it was set with ``"gaussian"`` or
            ``"smoothbox"``, or with ``"student_t"`` or
            ``"smoothbox_student_t"`` and zero degrees of freedom
            throughout; ``None`` where it was set with one of the latter
            two and NaN degrees of freedom throughout; and ``"student_t"``
            where it was set with one of them and other degrees of
            freedom.

        Raises
        ------
        ValueError
            Raised when the priors that the GP holds are not in a form
            ``set_priors`` takes, as priors written into ``hyper_priors``
            directly, or by a version of gpyreg that took them, may not
            be: a coordinate that has a location (``mu``, or the ends of a
            smooth box) and a ``sigma`` that is not finite and positive,
            or the reverse; an inverted smooth box; or a block that holds
            both a ``mu`` and the ends of a smooth box.
        """

        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_hyper_info = self.noise.hyperparameter_info()
        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        hyper_priors = {}
        lower = 0

        for info in hyper_info:
            upper = lower + info[1]
            i = range(lower, upper)
            mu, sigma, df, a, b = (
                self.hyper_priors[key][i]
                for key in ("mu", "sigma", "df", "a", "b")
            )

            # `set_priors` writes all five entries of a block given `None`
            # as NaN, the zero degrees of freedom of a Gaussian family
            # across the whole block, and the given ones, NaN, zero and
            # infinite included, for a Student's t family; it leaves `a`
            # and `b` NaN for the Gaussian and Student's t families, and
            # `mu` NaN for the smooth-box ones. Read in this order, the
            # entries give back the prior that writes them.
            gaussian = np.all(df == 0)
            if all(np.all(np.isnan(v)) for v in (mu, sigma, df, a, b)):
                vals = None
            elif np.all(np.isnan(a)) and np.all(np.isnan(b)):
                if gaussian:
                    vals = ("gaussian", (mu, sigma))
                else:
                    vals = ("student_t", (mu, sigma, df))
            elif np.all(np.isnan(mu)):
                if gaussian:
                    vals = ("smoothbox", (a, b, sigma))
                else:
                    vals = ("smoothbox_student_t", (a, b, sigma, df))
            else:
                raise ValueError(
                    f"`get_priors` cannot return the prior of {info[0]}: "
                    "the GP holds both a `mu` and the ends of a smooth box "
                    "for it, which no prior that `set_priors` takes has."
                )

            if vals is not None:
                # The checks of `set_priors`, on a scratch copy.
                scratch = {
                    key: np.full((info[1],), np.nan)
                    for key in ("mu", "sigma", "df", "a", "b")
                }
                try:
                    _write_prior_block(scratch, info[0], range(info[1]), *vals)
                except ValueError as err:
                    raise ValueError(
                        f"`get_priors` cannot return the prior of {info[0]} "
                        "that the GP holds, because `set_priors` refuses "
                        f"it. {err}"
                    ) from None

            hyper_priors[info[0]] = vals
            lower += info[1]

        return hyper_priors

    def set_priors(self, priors: dict = None):
        """
        Set the hyperparameter priors.

        Parameters
        ==========
        priors : dict, optional
            A dictionary of GP hyperparameter names and tuples of their priors.
            All hyperparameters need to appear in the dictionary.
            Use the value ``None`` to set no priors for a hyperparameter.
            If ``priors=None``, all hyperparameter priors are removed.
            Within a block of several hyperparameters, a coordinate whose
            location (``mu``, or ``a`` and ``b`` for the smooth-box
            families) and ``sigma`` are both NaN has no prior; every other
            coordinate needs a finite location, with ``a <= b``, and a
            finite, positive ``sigma``. A block with no prior in any
            coordinate holds none, and a GP whose blocks all hold none has
            no priors. A smooth box with ``a == b`` is the Gaussian (or
            Student's t) centred at ``a``.
            Degrees of freedom ``df`` that are zero, infinite or NaN make
            a ``"student_t"`` prior ``"gaussian"``, as
            ``gplite_hypprior.m`` reads them, and a
            ``"smoothbox_student_t"`` prior ``"smoothbox"``, gpyreg's own
            reading (gplite has no smooth-box priors). For the duration
            of :py:meth:`fit`, a NaN ``df`` takes the value of its option
            ``df_base`` instead.

        Raises
        ------
        ValueError
            Raised when ``priors`` is given, but missing the entry of an
            expected hyperparameter.
        ValueError
            Raised when ``priors`` is given, but a specified
            hyperparameter is unknown.
        ValueError
            Raised when a coordinate that has a prior is given a ``sigma``
            that is not finite and positive, or a location that is not
            finite, or a smooth box whose ``a`` is above its ``b``.
        """
        # The GP's own state changes only once every check has passed, so
        # a refused call leaves the priors, and the flag that says whether
        # there are any, as they were.
        remove_all = priors is None

        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()
        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        hyp_N = cov_N + mean_N + noise_N
        _check_hyperparameter_names(priors, hyper_info, "priors")
        # Set up a hyperprior dictionary with default values which can
        # be updated individually later.
        hyper_priors = {
            "mu": np.full((hyp_N,), np.nan),
            "sigma": np.full((hyp_N,), np.nan),
            "df": np.full((hyp_N,), np.nan),
            "a": np.full((hyp_N,), np.nan),
            "b": np.full((hyp_N,), np.nan),
        }

        non_trivial_flag = False
        lower = 0

        for info in hyper_info:
            if remove_all:
                vals = None
            else:
                try:
                    vals = priors[info[0]]
                except KeyError as _:
                    e_str = "Missing hyperparameter " + info[0]
                    raise ValueError(e_str) from None

            # None indicates no prior, and so does a block none of whose
            # coordinates has one, whatever its family.
            if vals is not None:
                upper = lower + info[1]
                prior_type, prior_params = vals
                if _write_prior_block(
                    hyper_priors,
                    info[0],
                    range(lower, upper),
                    prior_type,
                    prior_params,
                ):
                    non_trivial_flag = True

            lower += info[1]

        self.hyper_priors = hyper_priors
        self.no_prior = non_trivial_flag is not True
        self._prior_cache = None  # see __prior_masks
        self.__recompute_normalization_constants()

    def get_hyperparameters(self, as_array: bool = False):
        """
        Get the current hyperparameters of the GP.

        If hyperparameters have not been set yet, the result will
        be filled with ``NaN``.

        Parameters
        ==========
        as_array : bool, defaults to False
            Whether to return the hyperparameters as an array of shape
            ``(hyp_samples, hyp_N)``, or a list of dictionaries for each
            sample.

        Returns
        =======
        hyp : object
            The hyperparameters in the form specified by ``as_array``.
        """
        # If no hyperparameters have been set return an array/dict with NaN.
        if self.posteriors is None:
            cov_N = self.covariance.hyperparameter_count(self.D)
            mean_N = self.mean.hyperparameter_count(self.D)
            noise_N = self.noise.hyperparameter_count()
            hyp = np.full((1, cov_N + mean_N + noise_N), np.nan)
        else:
            hyp = np.zeros(
                (np.size(self.posteriors), np.size(self.posteriors[0].hyp))
            )
            for i in range(0, np.size(self.posteriors)):
                # Copy for avoiding reference issues.
                hyp[i, :] = self.posteriors[i].hyp.copy()

        if as_array:
            return hyp

        return self.hyperparameters_to_dict(hyp)

    def set_hyperparameters(
        self, hyp_new: object, compute_posterior: bool = True
    ):
        """
        Set new hyperparameters for the Gaussian Process.

        Parameters
        ==========
        hyp_new : object
            The new hyperparameters. This can be an array of shape
            ``(hyp_samples, hyp_N)`` where ``hyp_N`` is the number of
            hyperparameters, and ``hyp_samples`` is the amount of
            hyperparameter samples, a single dictionary with
            hyperparameter names and values, or a list of dictionaries.
            Passing a single dictionary or a list with one dictionary
            is equivalent.
        compute_posterior : bool, defaults to True
            Whether to compute the posterior for the new hyperparameters.

        Raises
        ------
        ValueError
            Raised when `hyp_new` is an array of the wrong shape.
        """
        if isinstance(hyp_new, np.ndarray):
            cov_N = self.covariance.hyperparameter_count(self.D)
            mean_N = self.mean.hyperparameter_count(self.D)
            noise_N = self.noise.hyperparameter_count()

            if hyp_new.ndim == 1:
                hyp_new = np.reshape(hyp_new, (1, -1))

            if hyp_new.shape[1] != cov_N + mean_N + noise_N:
                raise ValueError(
                    "Input hyperparameter array is the wrong shape!"
                )
            self.update(hyp=hyp_new, compute_posterior=compute_posterior)
        else:
            hyp_new_arr = self.hyperparameters_from_dict(hyp_new)
            self.update(hyp=hyp_new_arr, compute_posterior=compute_posterior)

    def hyperparameters_to_dict(self, hyp_arr: np.ndarray):
        """
        Convert a hyperparameter array to a list which contains a
        dictionary with hyperparameter names and values for each
        hyperparameter sample.

        Parameters
        ==========
        hyp_arr : ndarray
            An array of shape ``(hyp_samples, hyp_N)`` or shape
            ``(hyp_N,)``, which is interpreted as shape ``(1, hyp_N)``,
            containing hyperparameters.

        Returns
        =======
        hyp_dict : object
            A list which contains a dictonary with hyperparameter names and
            values for each sample.

        Raises
        ------
        ValueError
            Raised when the input hyperparameter array has the wrong shape.
        """
        hyp = []
        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()

        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        if hyp_arr.ndim == 1:
            hyp_arr = np.reshape(hyp_arr, (1, -1))

        if hyp_arr.shape[1] != cov_N + mean_N + noise_N:
            raise ValueError("Input hyperparameter array is the wrong shape!")

        for i in range(0, hyp_arr.shape[0]):
            # Make sure there are no accidents with references etc.
            hyp_tmp = hyp_arr[i, :].copy()
            hyp_dict = {}
            i = 0

            for info in hyper_info:
                hyp_dict[info[0]] = hyp_tmp[i : i + info[1]]
                i += info[1]

            hyp.append(hyp_dict)

        return hyp

    def hyperparameters_from_dict(self, hyp_dict_list):
        """
        Convert a list of hyperparameter dictionaries to a hyperparameter
        array.

        Parameters
        ==========
        hyp_dict_list : object
            A list of hyperparameter dictionaries with hyperparameter names
            and values. One can also pass just one dictionary instead of a
            list with one element.

        Returns
        =======
        hyp_arr : ndarray, shape (hyp_samples, hyp_N)
            The hyperparameter array where ``hyp_samples`` is the length of
            the list ``hyp_dict_list``.
        """
        if isinstance(hyp_dict_list, dict):
            hyp_dict_list = [hyp_dict_list]

        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()

        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        hyp_N = cov_N + mean_N + noise_N
        hyp_new_arr = np.zeros((len(hyp_dict_list), hyp_N))

        for i, hyp_tmp in enumerate(hyp_dict_list):
            j = 0

            for info in hyper_info:
                hyp_new_arr[i, j : j + info[1]] = hyp_tmp[info[0]]
                j += info[1]

        return hyp_new_arr

    def update(
        self,
        X_new: np.ndarray = None,
        y_new: np.ndarray = None,
        s2_new: np.ndarray = None,
        hyp: np.ndarray = None,
        compute_posterior: bool = True,
    ):
        """
        Add new data to the Gaussian Process.

        Parameters
        ==========
        X_new : ndarray, shape (N, D), optional
            New training inputs that will be added to the old training
            inputs.
        y_new : narray, shape (N, 1) optional
            New training targets that will be added to the old training
            targets.
        s2_new : ndarray, shape (N, 1), optional
            New input-dependent noise that will be added to the old training
            inputs. When only some training points come with a
            user-provided variance, the others are assigned zero.
        hyp : ndarray, shape (hyp_samples, hyp_N), optional
            New hyperparameters that will replace the old ones, one row per
            hyperparameter sample.
        compute_posterior : bool, defaults to True
            Whether to compute the new posterior or not.

        Raises
        =======
        ValueError
            Raised, before the update changes anything, when the update
            would make the number of targets, or of noise variances, that
            the GP holds differ from its number of inputs; a GP may hold
            inputs without targets, and data without noise variances. So
            ``y_new`` given without ``X_new``, one value per input held, is
            taken where the GP holds no targets, as ``s2_new`` is where it
            holds no variances, and either is refused where the GP holds
            what it gives or holds no inputs; ``X_new`` without ``y_new``
            is refused where the GP holds targets, and ``X_new`` with
            ``y_new`` where it holds inputs without targets.
        ValueError
            Raised by the check of the shapes of the training data given,
            before the update changes anything: when ``X_new`` is neither
            a 1-D nor a 2-D array or does not have ``D`` columns, when
            ``y_new`` does not hold one value per row of ``X_new``, or when
            an array ``s2_new`` does not hold one variance per row of
            ``X_new``; without ``X_new``, the rows are the inputs that the
            GP holds.
        TypeError
            Raised when ``s2_new`` is neither an array, a number nor
            ``None``, such as a list, before the update changes anything.
        LinAlgError
            Raised when the Cholesky decomposition failed multiple times even
            by adding numerical stability values to the matrix.
        ValueError
            Raised when ``hyp`` is not a 2D array with one column per
            hyperparameter of the GP.
        ValueError
            Raised when ``compute_posterior`` is ``True``, the GP has
            training data and a new posterior is computed in full, and a
            hyperparameter is NaN (not set), as it is on a GP whose
            hyperparameters were never given.
        """
        # Targets or variances without inputs, on a GP that holds none,
        # which the check of the shapes below cannot size.
        if X_new is None and self.X is None:
            for name, given in (
                ("targets", y_new),
                ("noise variances", s2_new),
            ):
                if given is not None:
                    raise ValueError(
                        f"update would leave the GP holding {name} without "
                        f"inputs: a GP holds as many {name} as inputs, or "
                        "none."
                    )

        X_new, y_new, s2_new = self._convert_shapes(X_new, y_new, s2_new)
        # Create local copies so we won't get trouble
        # with references later.
        if X_new is not None:
            X_new = X_new.copy()
        if y_new is not None:
            y_new = y_new.copy()
        if s2_new is not None:
            s2_new = s2_new.copy()

        if hyp is not None:
            hyp_N = (
                self.covariance.hyperparameter_count(self.D)
                + self.noise.hyperparameter_count()
                + self.mean.hyperparameter_count(self.D)
            )
            if np.ndim(hyp) != 2 or np.shape(hyp)[1] != hyp_N:
                raise ValueError(
                    f"The hyperparameters have shape {np.shape(hyp)}, but "
                    f"this GP has {hyp_N} hyperparameters and expects one "
                    "row per hyperparameter sample."
                )
            hyp = hyp.copy()

        # The training data after the update, stored below, after the
        # single-point extension of the posteriors, which reads the data
        # held before it. New data are appended; where the GP holds noise
        # variances or is given some, points without a supplied variance
        # get zero, the value the noise function uses when none is given.
        N_old = 0 if self.X is None else self.X.shape[0]
        X_all = self.X
        if X_new is not None:
            if self.X is None:
                X_all = X_new
            else:
                X_all = np.concatenate((self.X, X_new))

        y_all = self.y
        if y_new is not None:
            if self.y is None:
                y_all = y_new
            else:
                y_all = np.concatenate((self.y, y_new))

        s2_all = self.s2
        s2_added = s2_new
        if X_new is not None and s2_new is None and self.s2 is not None:
            s2_added = np.zeros((X_new.shape[0], 1))
        if s2_added is not None:
            if self.s2 is None:
                if X_new is not None and N_old > 0:
                    s2_all = np.concatenate((np.zeros((N_old, 1)), s2_added))
                else:
                    s2_all = s2_added
            else:
                s2_all = np.concatenate((self.s2, s2_added))

        # A GP holds as many targets, and as many noise variances, as
        # inputs, or none of them. An update that would make the numbers
        # differ is refused here, before it changes anything; numbers that
        # differ already, as after a `fit` given inputs of another number
        # and no variances, which keeps the variances held, are not
        # checked.
        N_all = 0 if X_all is None else X_all.shape[0]
        for name, held, stored in (
            ("targets", self.y, y_all),
            ("noise variances", self.s2, s2_all),
        ):
            agreed = held is None or held.shape[0] == N_old
            if agreed and stored is not None and stored.shape[0] != N_all:
                raise ValueError(
                    f"update would leave the GP holding {N_all} inputs and "
                    f"{stored.shape[0]} {name}: a GP holds as many {name} "
                    "as inputs, or none."
                )

        # Check whether to do a rank-1 update. The shortcut extends the
        # existing posteriors, so it applies only while their
        # hyperparameters stay in place (replacement hyperparameters need
        # a full recomputation) and only while they carry their factors:
        # after `clean` or an update with `compute_posterior=False` there
        # is nothing to extend.
        rank_one_update = False
        if X_new is not None and y_new is not None and compute_posterior:
            if (
                self.X is not None
                and self.y is not None
                and X_new.shape[0] == 1
                and y_new.shape[0] == 1
                and hyp is None
                and self.posteriors is not None
                and self.posteriors[0].alpha is not None
            ):
                rank_one_update = True
        full_updates = []  # Keep track of unstable rank-1 updates

        if rank_one_update:
            cov_N = self.covariance.hyperparameter_count(self.D)
            # mean_N = self.mean.hyperparameter_count(self.D)
            noise_N = self.noise.hyperparameter_count()

            # Compute prediction for all samples.
            m_star, v_star = self.predict(
                X_new,
                y_new,
                s2_star=s2_new,
                add_noise=True,
                separate_samples=True,
            )
            s_N = np.size(self.posteriors)

            # Loop over hyperparameter samples.
            for s in range(0, s_N):
                hyp_s = self.posteriors[s].hyp

                # Total noise variance of the new point (a scalar; the
                # noise function returns an array for output-dependent
                # noise).
                hyp_noise = hyp_s[cov_N : cov_N + noise_N]
                sn2 = np.ravel(
                    self.noise.compute(
                        hyp_noise,
                        X_new,
                        y_new,
                        0 if s2_new is None else s2_new,
                    )
                )[0]
                sn2_eff = sn2 * self.posteriors[s].sn2_mult

                # Noise scale of the existing factorization, kept for the
                # extended factor.
                sl = self.posteriors[s]._noise_scale()

                # Compute covariance and cross-covariance.
                hyp_cov = hyp_s[0:cov_N]
                K = self.covariance.compute(hyp_cov, X_new)
                Ks = self.covariance.compute(hyp_cov, self.X, X_new)

                L = self.posteriors[s].L
                L_chol = self.posteriors[s].L_chol

                full_update_s = False
                if L_chol:  # High-noise parametrization
                    # L^T L = (K + sn2_mult * sn2) / sl. With u = L^-T k*,
                    # the new column is u / sl and the new diagonal entry
                    # is sqrt(sl * sn2_eff + sl * k** - u^T u) / sl.
                    new_L_column = sp.linalg.solve_triangular(
                        L, Ks, trans=1, check_finite=False
                    )
                    # If rank-1 update is not numerically stable, perform a
                    # full update for this posterior instead:
                    sqrt_arg = (
                        sl * sn2_eff
                        + K * sl
                        - np.dot(new_L_column.T, new_L_column)
                    )
                    if sqrt_arg <= 0.0:
                        full_update_s = (
                            True  #  Mark this posterior for full update
                        )
                        full_updates.append(s)
                        warnings.warn(
                            "Rank-one update of Cholesky factor unstable "
                            + f"for posterior {s}. Reverting to full update.",
                            stacklevel=2,
                        )
                    else:  # Otherwise continue with rank-1 update:
                        alpha_update = (
                            sp.linalg.solve_triangular(
                                L,
                                new_L_column,
                                trans=0,
                                check_finite=False,
                            )
                            / sl
                        )
                        self.posteriors[s].L = np.block(
                            [
                                [L, new_L_column / sl],
                                [
                                    np.zeros((1, L.shape[0])),
                                    np.sqrt(sqrt_arg) / sl,
                                ],
                            ]
                        )

                else:  # Low-noise parametrization
                    # The extension divides by the predictive variance of
                    # the new point, which `predict` clamps at the noise
                    # level: a v_star that low carries no information
                    # about the latent variance, so fall through to a full
                    # recomputation as the branch above does. A posterior
                    # pickled without the Cholesky factor that the
                    # extension works on is recomputed as well.
                    L_factor = getattr(self.posteriors[s], "L_factor", None)
                    if v_star[0, s] <= sn2_eff or L_factor is None:
                        full_update_s = True
                        full_updates.append(s)
                        if L_factor is not None:
                            warnings.warn(
                                "Rank-one update of the posterior factor "
                                f"unstable for posterior {s}. Reverting to "
                                "full update.",
                                stacklevel=2,
                            )
                    else:
                        # inv(K + sn2_mult * sn2) k* from the Cholesky
                        # factor of the matrix: taken from the explicit
                        # inverse L, whose rounding grows as the noise
                        # shrinks, it would carry that rounding, divided
                        # by v_star, into alpha and L at every update. The
                        # factor is extended by the column L_factor^-T k*
                        # and the square root of v_star, the Schur
                        # complement that the inverse divides by.
                        new_column = sp.linalg.solve_triangular(
                            L_factor, Ks, trans=1, check_finite=False
                        )
                        alpha_update = sp.linalg.solve_triangular(
                            L_factor, new_column, trans=0, check_finite=False
                        )
                        v = -alpha_update / v_star[:, s]
                        self.posteriors[s].L = np.block(
                            [
                                [L + np.dot(v, alpha_update.T), -v],
                                [-v.T, -1 / v_star[:, s]],
                            ]
                        )
                        self.posteriors[s].L_factor = np.block(
                            [
                                [L_factor, new_column],
                                [
                                    np.zeros((1, L_factor.shape[0])),
                                    np.sqrt(v_star[:, s : s + 1]),
                                ],
                            ]
                        )

                # Finish rank-1 update if computation was stable for posterior
                # s
                if not full_update_s:
                    self.posteriors[s].sW = np.concatenate(
                        (
                            self.posteriors[s].sW,
                            np.array([[1 / np.sqrt(sl)]]),
                        )
                    )

                    # alpha_update now contains (K + \sigma^2 I) \ k*
                    self.posteriors[s].alpha = np.concatenate(
                        (self.posteriors[s].alpha, np.array([[0]]))
                    ) + (m_star[:, s] - y_new) / v_star[:, s] * np.concatenate(
                        (alpha_update, np.array([[-1]]))
                    )

        self.X, self.y, self.s2 = X_all, y_all, s2_all

        if rank_one_update:
            for s in full_updates:  # Compute full update where rank-1 failed
                hyp_s = self.posteriors[s].hyp
                self.posteriors[s] = self.__core_computation(hyp_s, 0, 0)

        else:
            if hyp is None:
                hyp = self.get_hyperparameters(as_array=True)
            s_N, _ = hyp.shape
            self.posteriors = np.empty((s_N,), dtype=Posterior)

            if compute_posterior and self.X is not None and self.y is not None:
                unset = np.any(np.isnan(hyp), axis=0)
                if np.any(unset):
                    raise ValueError(
                        "Cannot compute the posterior: the hyperparameters "
                        + ", ".join(self.__hyperparameter_names(unset))
                        + " are NaN (not set)."
                    )
                for i in range(0, s_N):
                    self.posteriors[i] = self.__core_computation(
                        hyp[i, :], 0, 0
                    )
            else:
                for i in range(0, s_N):
                    self.posteriors[i] = Posterior(
                        hyp[i, :], None, None, None, None, None
                    )

    def clean(self):
        """
        Clean auxiliary computational structures from the Gaussian Process,
        thus reducing memory usage. These can be reconstructed with a call to
        :py:func:`update` with ``compute_posterior=True``.

        Furthermore, the `temporary_data` attribute is being cleared.
        """

        # dict to store temporary data e.g. for pyvbmc
        self.temporary_data = {}

        # Check if there are posteriors to clean.
        if self.posteriors is not None:
            for posterior in self.posteriors:
                posterior.alpha = None
                posterior.sW = None
                posterior.L = None
                posterior.sn2_mult = None
                posterior.L_chol = None
                posterior.sl = None
                posterior.L_factor = None
        # Maybe add a call to garbage collection here? This would
        # make sure that the things set to None are actually no longer
        # using memory.

    def fit(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None,
        s2: np.ndarray = None,
        hyp0=None,
        options: dict = None,
        rng=None,
    ):
        """
        Train the hyperparameters of the Gaussian Process.

        Parameters
        ==========
        X : ndarray, shape (N, D), optional
            Training points that will replace the current training points
            of the GP. If not given the current training points are used.
        y : ndarray, shape (N, 1), optional
            Training targets that will replace the current training targets
            of the GP. If not given the current training targets are used.
        s2 : ndarray, shape (N, 1), optional
            Noise variance at training points that will replace the
            current noise variances of the GP. If not given the current
            noise variances are used.
        options : dict, optional
            A dictionary of options for training. The counts among them,
            ``opts_N``, ``init_N``, ``n_samples``, ``thin`` and ``burn``,
            are whole numbers of an integer or a float type (2.0 is taken
            as 2), or 0-d arrays that hold one. The possible options are:

                **opts_N** : int, defaults to 3
                    Number of hyperparameter optimization runs.
                **init_N** : int, defaults to 1024
                    Initial design size for hyperparameter optimization.
                **df_base** : int, defaults to 7
                    The degrees of freedom of a ``"student_t"`` or
                    ``"smoothbox_student_t"`` prior whose ``df`` is NaN,
                    filled into a copy of the priors for the duration of
                    the fit, as ``gplite_train.m`` fills its local copy.
                    The GP keeps the priors as they were set, and outside
                    the fit a NaN ``df`` reads as ``"gaussian"`` or
                    ``"smoothbox"`` (see :py:meth:`set_priors`).
                **n_samples** : int, defaults to 10
                    Number of hyperparameters to sample.
                **thin** : int, defaults to 5
                    Thinning parameter for slice sampling: one sample in
                    ``thin`` is kept. A whole number greater than zero.
                **burn** : int or None, defaults to ``thin * n_samples``
                    Burn parameter for slice sampling: the number of
                    samples drawn and dropped before the first one kept.
                    A whole number of at least zero, or ``None``, which
                    leaves it to :py:meth:`SliceSampler.sample`.
                **lower_bounds** : str or ndarray, defaults to "current"
                    User-provided lower bounds. Any values which are `nan` will
                    be filled with the recommended bounds. "recommended" means
                    use all recommended bounds. "current" means use current
                    bounds.
                **upper_bounds** : str or ndarray, defaults to "current"
                    User-provided upper bounds. Any values which are `nan` will
                    be filled with the recommended bounds. "recommended" means
                    use all recommended bounds. "current" means use current
                    bounds.
                **init_method** : {'sobol', 'rand'}, defaults to 'sobol'
                    Specify whether to use Sobol or random sequences for
                    the initial space-filling design.
                **sampler_name** : {'slicesample'}, defaults to 'slicesample'
                    The name of the sampler to use. Currently only slice
                    sampling is supported.
                **tol_opt** : float, defaults to 1e-5
                    Optimization tolerance for stopping.
                **tol_opt_mcmc** : float, defaults to 1e-3
                    Preliminary optimization tolerance when doing MCMC.
                **widths** : ndarray, shape (hyp_n,), optional
                    Default widths to use for sampling. If not provided
                    appropriate ones will be computed.
        rng : None, numpy.random.Generator or seed, optional
            Where the fit's random draws come from (the space-filling
            initial design and the slice sampler). ``None`` (default) keeps
            NumPy's global legacy stream, as before generators were
            supported, so ``np.random.seed`` still fixes a fit; a
            ``numpy.random.Generator`` is used as is (and shared with the
            caller); an integer or ``SeedSequence`` seeds a new generator.
            See :func:`gpyreg.rng.resolve_rng`.

        Returns
        =======
        hyp : ndarray, shape (hyp_samples, hyp_N)
            The fitted hyperparameters.
        optimize_result : OptimizeResult
            The optimization result represented as a ``OptimizeResult``
            object. For more details see :py:func:`scipy.optimize.minimize`.
        sampling_result : dict
            If sampling was performed this is a dictionary with info on the
            sampling run, and None otherwise.

        Raises
        ------
        ValueError
            Raised when the GP has no training data, neither given to the
            fit nor held from an earlier ``fit`` or ``update``, before the
            fit changes anything.
        ValueError
            Raised by the check of the shapes of the training data given,
            before the fit changes anything: when ``X`` is neither a 1-D
            nor a 2-D array or does not have ``D`` columns, when ``y`` does
            not hold one value per row of ``X``, or when an array ``s2``
            does not hold one variance per row of ``X``.
        TypeError
            Raised by the same check, before the fit changes anything,
            when ``s2`` is neither an array, a number nor ``None``, such as
            a list.
        ValueError
            Raised, before the fit changes anything, when the data it
            would hold, those given with those the GP holds for what is
            not given, would make the number of targets, or of the noise
            variances that the noise function reads (as
            :class:`gpyreg.noise_functions.GaussianNoise` with
            ``user_provided_add`` does), differ from the number of inputs:
            when ``X`` is given without ``y``, or without ``s2``, and the
            targets, or those variances, that the GP holds, one per input
            it holds, are not one per row of ``X``.
        ValueError
            Raised by :py:meth:`get_recommended_bounds`, through which the
            fit fills the bounds that are not set: when the option
            ``lower_bounds`` or ``upper_bounds`` is neither
            ``"recommended"``, ``None``, ``"current"`` nor an array, when
            a lower bound given is above its upper bound, or when a column
            of the training inputs has no spread and a hyperparameter
            whose recommended bounds take their scale from it is not given
            a finite lower bound.
        ValueError
            Raised when the option ``opts_N``, ``init_N`` or ``n_samples``
            is not a whole number of at least zero, the option ``thin``
            not a whole number of at least one, or the option ``burn``
            neither ``None`` nor a whole number of at least zero: a
            fraction, a number below that, an infinity, NaN, a bool or a
            value that is not a number, before the fit changes anything.
        ValueError
            Raised when ``n_samples`` is positive and ``sampler_name`` is
            not ``'slicesample'``, after the optimization.
        ValueError
            Raised by :py:class:`gpyreg.slice_sample.SliceSampler` when
            ``n_samples`` is positive, after the optimization: when the
            widths of the sampler, from the option ``widths`` or computed
            by the fit, are not all positive and finite, or when the log
            posterior is not finite at the optimized hyperparameters, where
            the chain starts.
        ValueError
            Raised by :py:meth:`update`, which computes the posterior of
            the fitted hyperparameters, when one of them is NaN.
        LinAlgError
            Raised when the Cholesky decomposition of the training
            covariance fails even after its noise is multiplied tenfold, up
            to ten times: by :py:meth:`update` at the fitted
            hyperparameters, or by the objective at a starting point or
            during the optimization, which lets it propagate. Whether a
            covariance that holds NaN, as from a starting point that does,
            fails there or gives NaN depends on the LAPACK build.
        """
        # Share one stream between the initial design and the sampler,
        # including when the caller supplies a seed rather than a generator.
        rng = resolve_rng(rng)

        ## Default options
        if options is None:
            options = {}
        # Counts, which the fit slices and loops with: a whole number of
        # either type is converted, and any other value is refused before
        # the fit changes anything.
        opts_N = _whole_number(
            options.get("opts_N", 3), "The option opts_N", 0
        )
        init_N = _whole_number(
            options.get("init_N", 2**10), "The option init_N", 0
        )
        init_method = options.get("init_method", "sobol")
        thin = _whole_number(options.get("thin", 5), "The option thin", 1)
        df_base = options.get("df_base", 7)
        widths = options.get("widths", None)
        log_p = options.get("log_P", None)  # Not used since no slicelite
        outwarp_fun = options.get("outwarp_fun", None)  # Not used
        step_size = options.get("step_size", None)  # Not used since no MALA
        tol_opt = options.get("tol_opt", 1e-5)
        tol_opt_mcmc = options.get("tol_opt_mcmc", 1e-3)
        # The documented name first, then the undocumented spelling
        # that PyVBMC writes.
        sampler_name = options.get(
            "sampler_name", options.get("sampler", "slicesample")
        )
        s_N = _whole_number(
            options.get("n_samples", 10), "The option n_samples", 0
        )
        # The burn-in of the sampler, checked with the counts whether or
        # not the fit draws samples; None leaves it to the sampler.
        burn_in = options.get("burn", thin * s_N)
        if burn_in is not None:
            burn_in = _whole_number(burn_in, "The option burn", 0)
        lower_bounds = options.get("lower_bounds", "current")
        upper_bounds = options.get("upper_bounds", "current")

        # The training data, given here or held by the GP, checked before
        # the fit changes anything.
        missing = [
            name
            for name, given, held in (("X", X, self.X), ("y", y, self.y))
            if given is None and held is None
        ]
        if missing:
            raise ValueError(
                "The GP has no training data: `fit` needs the inputs X and "
                "the targets y, given to it or held by the GP from an "
                "earlier `fit` or `update`; missing "
                + " and ".join(missing)
                + "."
            )

        X, y, s2 = self._convert_shapes(X, y, s2)

        # A GP holds as many targets, and as many of the noise variances
        # that its noise function reads, as inputs, or none of them, as in
        # `update`. Only inputs given without the targets, or the
        # variances, that the GP holds can make the numbers differ, and
        # such data are refused here, before the fit changes anything.
        # Numbers that differ already are not checked, and neither are
        # variances that the noise function does not read.
        N_old = None if self.X is None else self.X.shape[0]
        N_all = N_old if X is None else X.shape[0]
        counted = [("targets", "y", self.y, y, "")]
        noise_parameters = getattr(self.noise, "parameters", None)
        if noise_parameters is not None and noise_parameters[1] != 0:
            counted.append(
                (
                    "noise variances",
                    "s2",
                    self.s2,
                    s2,
                    ", which its noise function reads",
                )
            )
        for name, argument, held, given, note in counted:
            if given is None and held is not None:
                N_held = held.shape[0]
                if N_held == N_old and N_held != N_all:
                    raise ValueError(
                        f"fit would leave the GP holding {N_all} inputs and "
                        f"{N_held} {name}: X is given without {argument}, "
                        f"and the GP holds {N_held} {name}{note}; give "
                        f"{argument} with X, one per input."
                    )

        # Initialize GP if requested.
        if X is not None:
            self.X = X

        if y is not None:
            self.y = y

        if s2 is not None:
            self.s2 = s2

        cov_N = self.covariance.hyperparameter_count(self.D)
        # mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        ## Initialize inference of GP hyperparameters (bounds, priors, etc.)

        cov_bounds_info = self.covariance.get_bounds_info(self.X, self.y)
        mean_bounds_info = self.mean.get_bounds_info(self.X, self.y)
        noise_bounds_info = self.noise.get_bounds_info(self.X, self.y)

        # The default degrees of freedom fill what a prior leaves
        # unset for the duration of the fit alone: the objectives read
        # the GP's own priors, and the GP keeps the priors the caller
        # set, so a second fit with another `df_base` uses it
        # (`gplite_train.m:113-117` builds its own copy the same way).
        df_given = self.hyper_priors["df"]
        df_filled = df_given.copy()
        df_filled[np.isnan(df_filled)] = df_base
        self.hyper_priors["df"] = df_filled
        self._prior_cache = None  # the prior's type masks depend on df
        try:
            # Set any unset bounds:
            use_current_bounds = (
                isinstance(lower_bounds, str)
                and lower_bounds == "current"
                and isinstance(upper_bounds, str)
                and upper_bounds == "current"
            )
            if use_current_bounds and (
                np.any(np.isnan(self.lower_bounds))
                or np.any(np.isnan(self.upper_bounds))
            ):  # If we're using the existing bounds, fill any nan's:
                self.set_bounds(
                    self.get_recommended_bounds(
                        self.lower_bounds, self.upper_bounds
                    )
                )
            else:  # Otherwise set the bounds according to the provided options:
                self.set_bounds(
                    self.get_recommended_bounds(lower_bounds, upper_bounds)
                )

            LB = self.lower_bounds
            UB = self.upper_bounds

            # Plausible bounds for generation of starting points
            PLB = np.concatenate(
                [
                    cov_bounds_info["PLB"],
                    noise_bounds_info["PLB"],
                    mean_bounds_info["PLB"],
                ]
            )
            PUB = np.concatenate(
                [
                    cov_bounds_info["PUB"],
                    noise_bounds_info["PUB"],
                    mean_bounds_info["PUB"],
                ]
            )
            PLB = np.minimum(np.maximum(PLB, LB), UB)
            PUB = np.maximum(np.minimum(PUB, UB), LB)
            # With LB <= UB, which holds here, the two clips are one
            # monotone map into the hard box, so an ordered plausible pair
            # stays ordered. The noise recommends an inverted pair,
            # [0.5 * log(tol), log(std(y))], for targets whose standard
            # deviation is below 1e-3, as `gplite_noisefun.m:105-106`
            # does; the clips keep it inverted unless the range of the
            # targets is below 1e-6, where the hard pair collapses and
            # takes both bounds with it. gplite clips the same way
            # (`gplite_train.m:157-158`) and draws its design from the
            # inverted pair; the space-filling design here needs the pair
            # ordered, so an inverted one collapses onto its upper bound,
            # which the clip left inside the hard box.
            inverted = PLB > PUB
            PLB[inverted] = PUB[inverted]

            # If we are not provided with an initial hyperparameter guess then
            # either use the current hyperparameters if they exist, or use
            # plausible lower and upper bounds to guess.
            if hyp0 is None:
                if self.posteriors is not None:
                    hyp0 = self.get_hyperparameters(as_array=True)
                else:
                    hyp0 = np.reshape(
                        np.minimum(np.maximum((PLB + PUB) / 2, LB), UB),
                        (1, -1),
                    )
            elif isinstance(hyp0, dict):
                hyp0 = self.hyperparameters_from_dict(hyp0)

            ## Hyperparameter optimization
            # Each no-gradient objective owns one factorization cache for the
            # whole fit (consecutive evaluations that move only mean-function
            # hyperparameters reuse the Cholesky factor, see
            # __core_computation); the gradient objective of the optimizer gets
            # none, it needs the kernel derivatives.
            design_cache = {}
            objective_f_1 = lambda hyp_: self.__gp_obj_fun(
                hyp_, False, False, cache=design_cache
            )
            if s_N > 0:
                tol = tol_opt_mcmc
            else:
                tol = tol_opt

            # First evaluate GP log posterior on an informed space-filling design.
            t1_s = time.time()

            if init_N > 0:
                X0, y0 = f_min_fill(
                    objective_f_1,
                    hyp0,
                    LB,
                    UB,
                    PLB,
                    PUB,
                    self.hyper_priors,
                    init_N,
                    init_method,
                    rng=rng,
                )
                # Make sure we have at least one hyperparameter to use
                # later. A copy: the low-noise starting point below is
                # written into these rows, and the sampler widths are the
                # standard deviation of the design as it was returned
                # (`gplite_train.m:206-207`).
                hyp = X0[0 : np.maximum(opts_N, 1), :].copy()

                # Extract a good low-noise starting point for the 2nd optimization.
                if noise_N > 0 and 1 < opts_N < init_N:
                    xx = X0[opts_N:, :]
                    noise_y = y0[opts_N:]
                    noise_params = xx[:, cov_N]

                    # Order by noise parameter magnitude.
                    order = np.argsort(noise_params)
                    xx = xx[order, :]
                    noise_y = noise_y[order]
                    # Take the best amongst bottom 20% vectors.
                    idx_best = np.argmin(
                        noise_y[0 : math.ceil(0.2 * np.size(noise_y))]
                    )
                    hyp[1, :] = xx[idx_best, :]

                if init_N > 1:
                    widths_default = np.std(X0, axis=0, ddof=1)
                else:
                    widths_default = np.zeros(shape=PLB.shape)
            else:
                N = hyp0.shape[0]
                # The initial value of `gplite_train.m:250`; the loop
                # writes every entry.
                nll = np.full((N,), np.inf)
                for i in range(0, N):
                    nll[i] = objective_f_1(hyp0[i, :])
                order = np.argsort(nll)
                hyp = hyp0[order, :]
                widths_default = PUB - PLB

            # Fix zero widths.
            idx0 = widths_default == 0
            if np.any(idx0):
                if np.shape(hyp)[0] > 1:
                    std_hyp = np.std(hyp, axis=0, ddof=1)
                    widths_default[idx0] = std_hyp[idx0]
                    idx0 = widths_default == 0

                if np.any(idx0):
                    widths_default[idx0] = np.minimum(1, UB[idx0] - LB[idx0])

            t1 = time.time() - t1_s

            # Check that hyperparameters are within bounds.
            # Note that with infinite upper and lower bounds we have to be careful
            # with spacing since it returns NaN. Furthermore, if LB == UB then
            # we have to be careful about the lower bound not being larger than
            # the upper bounds. Also, copy is necessary to avoid LB or UB
            # getting modified.
            eps_LB = np.reshape(LB.copy(), (1, -1))
            eps_UB = np.reshape(UB.copy(), (1, -1))
            LB_idx = (eps_LB != eps_UB) & np.isfinite(eps_LB)
            UB_idx = (eps_LB != eps_UB) & np.isfinite(eps_UB)
            # np.spacing could return negative numbers so use nextafter
            eps_LB[LB_idx] = np.nextafter(eps_LB[LB_idx], np.inf)
            eps_UB[UB_idx] = np.nextafter(eps_UB[UB_idx], -np.inf)
            hyp = np.minimum(eps_UB, np.maximum(eps_LB, hyp))

            # Perform optimization from most promising opts_N hyperparameter
            # vectors.
            objective_f_2 = lambda hyp_: self.__gp_obj_fun(hyp_, True, False)
            nll = np.full((np.maximum(opts_N, 1),), np.inf)
            opt_results = []

            t2_s = time.time()
            # Make sure we don't overshoot.
            opts_N = np.minimum(opts_N, hyp.shape[0])
            for i in range(0, opts_N):
                res = sp.optimize.minimize(
                    fun=objective_f_2,
                    x0=hyp[i, :],
                    jac=True,
                    bounds=list(zip(LB, UB)),
                    tol=tol,
                )
                opt_results.append(res)
                hyp[i, :] = res.x
                nll[i] = res.fun

            # Take the best hyperparameter vector.
            if opts_N > 0:
                optimize_result = opt_results[np.argmin(nll)]
                hyp_start = hyp[np.argmin(nll), :].copy()
            else:
                optimize_result = None
                hyp_start = hyp[0, :].copy()
            t2 = time.time() - t2_s

            # In case n_samples is 0, just return the optimized hyperparameter
            # result.
            if s_N == 0:
                hyp_start = np.reshape(hyp_start, (1, -1))
                self.update(hyp=hyp_start)
                return hyp_start, optimize_result, None

            ## Sample from best hyperparameter vector using slice sampling

            t3_s = time.time()
            # Effective number of samples (thin after)
            eff_s_N = s_N * thin

            if sampler_name != "slicesample":
                raise ValueError("Unknown sampler!")

            sample_cache = {}
            sample_f = lambda hyp_: self.__gp_obj_fun(
                hyp_, False, True, cache=sample_cache
            )
            options = {"display": "off", "diagnostics": False}
            if widths is None:
                widths = widths_default
            else:
                widths = np.minimum(widths, widths_default)
            slicer = SliceSampler(
                sample_f, hyp_start, widths, LB, UB, options, rng=rng
            )
            sampling_result = slicer.sample(eff_s_N, burn=burn_in)

            # Thin samples
            hyp_pre_thin = sampling_result["samples"]
            hyp = hyp_pre_thin[thin - 1 :: thin, :]

            t3 = time.time() - t3_s
            # print(t1, t2, t3)

            # Recompute GP with finalized hyperparameters.
            self.update(hyp=hyp)
            return hyp, optimize_result, sampling_result
        finally:
            self.hyper_priors["df"] = df_given
            self._prior_cache = None
            self.__recompute_normalization_constants()

    def __recompute_normalization_constants(self):
        self.normalization_constants = np.full(self.lower_bounds.shape, 1.0)

        for i in range(0, np.size(self.lower_bounds)):
            mu = self.hyper_priors["mu"][i]
            sigma = np.abs(self.hyper_priors["sigma"])[i]
            df = self.hyper_priors["df"][i]
            a = self.hyper_priors["a"][i]
            b = self.hyper_priors["b"][i]
            lb = self.lower_bounds[i]
            ub = self.upper_bounds[i]

            # Fixed dimension
            if lb == ub:
                continue

            # No boundaries
            if not np.isfinite(lb) and not np.isfinite(ub):
                continue

            # Uniform
            if not np.isfinite(mu) and not np.isfinite(sigma):
                continue

            # The mass inside the bounds. Where the lower bound lies above
            # the centre of the prior (its median), the cumulative
            # distribution function is above one half at both bounds, where
            # a double resolves it only to a fixed absolute step, and the
            # difference of its two values loses the mass as the bounds
            # move up the tail, all of it with both beyond about 8.3 scales
            # of a Gaussian, where both values round to one; the survival
            # function, below one half there, keeps it. Everywhere else the
            # mass is the difference of the two values of the cumulative
            # distribution function.
            if np.isfinite(a) and np.isfinite(b):
                upper_half = lb > 0.5 * (a + b)
                if df == 0 or not np.isfinite(df):
                    p = smoothbox_sf if upper_half else smoothbox_cdf
                    p_lb = p(lb, sigma, a, b)
                    p_ub = p(ub, sigma, a, b)
                else:
                    p = (
                        smoothbox_student_t_sf
                        if upper_half
                        else smoothbox_student_t_cdf
                    )
                    p_lb = p(lb, df, sigma, a, b)
                    p_ub = p(ub, df, sigma, a, b)
            else:
                upper_half = lb > mu
                if df == 0 or not np.isfinite(df):
                    p = sp.stats.norm.sf if upper_half else sp.stats.norm.cdf
                    p_lb = p(lb, loc=mu, scale=sigma)
                    p_ub = p(ub, loc=mu, scale=sigma)
                else:
                    p = sp.stats.t.sf if upper_half else sp.stats.t.cdf
                    p_lb = p(lb, df, loc=mu, scale=sigma)
                    p_ub = p(ub, df, loc=mu, scale=sigma)

            if upper_half:
                self.normalization_constants[i] = p_lb - p_ub
            else:
                self.normalization_constants[i] = p_ub - p_lb

    def __prior_masks(self):
        """The hyperprior's type masks and normalization constants, which
        depend only on ``hyper_priors`` and the bounds.

        Built on first use and dropped by ``set_priors``, ``set_bounds`` and
        the ``df`` fill in ``fit`` (the only writers); read through
        ``getattr`` because GP objects pickled before this cache existed
        have no such attribute. ``__compute_log_priors`` is evaluated tens
        of thousands of times per fit by the slice sampler, and rebuilding
        the masks was a fifth of each evaluation.
        """
        cache = getattr(self, "_prior_cache", None)
        if cache is not None:
            return cache

        mu = self.hyper_priors["mu"]
        sigma = np.abs(self.hyper_priors["sigma"])
        df = self.hyper_priors["df"]
        a = self.hyper_priors["a"]
        b = self.hyper_priors["b"]
        lb = self.lower_bounds
        ub = self.upper_bounds

        f_idx = lb == ub
        sb_idx = (
            np.isfinite(a)
            & np.isfinite(b)
            & ((df == 0) | ~np.isfinite(df))
            & ~np.isfinite(mu)
            & np.isfinite(sigma)
        )
        sb_t_idx = (
            np.isfinite(a)
            & np.isfinite(b)
            & (df > 0)
            & ~np.isfinite(mu)
            & np.isfinite(sigma)
            & np.isfinite(df)
        )
        u_idx = ~np.isfinite(mu) & ~np.isfinite(sigma)
        g_idx = (
            ~u_idx
            & ~sb_idx
            & ((df == 0) | ~np.isfinite(df))
            & np.isfinite(sigma)
        )
        t_idx = ~u_idx & ~sb_t_idx & (df > 0) & np.isfinite(df)

        cache = {
            "sigma": sigma,
            "f_idx": f_idx,
            "sb_idx": sb_idx,
            "sb_t_idx": sb_t_idx,
            "g_idx": g_idx,
            "t_idx": t_idx,
            "gt_idx": g_idx | t_idx,
            "any_f": bool(np.any(f_idx)),
            "any_sb": bool(np.any(sb_idx)),
            "any_sb_t": bool(np.any(sb_t_idx)),
            "any_g": bool(np.any(g_idx)),
            "any_t": bool(np.any(t_idx)),
            "log_norm": np.sum(np.log(self.normalization_constants)),
        }
        # Normalization constants so that the integrals over the pdfs are 1.
        if cache["any_sb"]:
            cache["C_sb"] = 1.0 + (b[sb_idx] - a[sb_idx]) / (
                sigma[sb_idx] * np.sqrt(2 * np.pi)
            )
        if cache["any_sb_t"]:
            # The ratio of gamma functions through their logarithms: both
            # overflow from a few hundred degrees of freedom, where the
            # ratio itself is about sqrt(df / 2).
            log_ratio = sp.special.gammaln(
                0.5 * (df[sb_t_idx] + 1)
            ) - sp.special.gammaln(0.5 * df[sb_t_idx])
            cache["C_sb_t"] = 1.0 + (b[sb_t_idx] - a[sb_t_idx]) * np.exp(
                log_ratio
            ) / (sigma[sb_t_idx] * np.sqrt(df[sb_t_idx] * np.pi))
        self._prior_cache = cache
        return cache

    def __compute_log_priors(self, hyp: np.ndarray, compute_grad: bool):
        lp = 0
        dlp = None
        if compute_grad:
            dlp = np.zeros(hyp.shape)

        mu = self.hyper_priors["mu"]
        df = self.hyper_priors["df"]
        a = self.hyper_priors["a"]
        b = self.hyper_priors["b"]
        lb = self.lower_bounds

        masks = self.__prior_masks()
        sigma = masks["sigma"]
        f_idx = masks["f_idx"]
        sb_idx = masks["sb_idx"]
        sb_t_idx = masks["sb_t_idx"]
        g_idx = masks["g_idx"]
        t_idx = masks["t_idx"]
        gt_idx = masks["gt_idx"]

        # Quadratic form
        z2 = np.zeros(hyp.shape)
        z2[gt_idx] = ((hyp[gt_idx] - mu[gt_idx]) / sigma[gt_idx]) ** 2

        # A coordinate whose bounds are equal has no density off its
        # value. Its entry of the gradient is that of its own prior,
        # written below, and stays zero where the prior leaves it unset (no
        # prior, or a value inside a smooth box), as in
        # `gplite_hypprior.m`, whose gradient starts at zero and has no
        # branch for such coordinates.
        if masks["any_f"]:
            if np.any(hyp[f_idx] != lb[f_idx]):
                lp = -np.inf

        # Smooth box prior
        if masks["any_sb"]:
            C = masks["C_sb"]

            sb_idx_b = (hyp < a) & sb_idx
            sb_idx_a = (hyp > b) & sb_idx
            sb_idx_btw = (hyp >= a) & (hyp <= b) & sb_idx

            z2_tmp = np.zeros(hyp.shape)
            z2_tmp[sb_idx_b] = (
                (hyp[sb_idx_b] - a[sb_idx_b]) / sigma[sb_idx_b]
            ) ** 2
            z2_tmp[sb_idx_a] = (
                (hyp[sb_idx_a] - b[sb_idx_a]) / sigma[sb_idx_a]
            ) ** 2

            if np.any(sb_idx_b | sb_idx_a):
                tmp_idx = sb_idx_b | sb_idx_a
                lp -= 0.5 * np.sum(
                    np.log(
                        C[tmp_idx[sb_idx]] ** 2
                        * 2
                        * np.pi
                        * sigma[tmp_idx] ** 2
                    )
                    + z2_tmp[tmp_idx]
                )
            if np.any(sb_idx_btw):
                lp -= np.sum(
                    np.log(C[sb_idx_btw[sb_idx]] * sigma[sb_idx_btw])
                    + np.log(np.sqrt(2 * np.pi))
                )

            if compute_grad:
                if np.any(sb_idx_b):
                    dlp[sb_idx_b] = (
                        -(hyp[sb_idx_b] - a[sb_idx_b]) / sigma[sb_idx_b] ** 2
                    )
                if np.any(sb_idx_a):
                    dlp[sb_idx_a] = (
                        -(hyp[sb_idx_a] - b[sb_idx_a]) / sigma[sb_idx_a] ** 2
                    )

        # Smooth box Student's t prior
        if masks["any_sb_t"]:
            C = masks["C_sb_t"]

            sb_t_idx_b = (hyp < a) & sb_t_idx
            sb_t_idx_a = (hyp > b) & sb_t_idx
            sb_t_idx_btw = (hyp >= a) & (hyp <= b) & sb_t_idx

            z2_tmp = np.zeros(hyp.shape)
            z2_tmp[sb_t_idx_b] = (
                (hyp[sb_t_idx_b] - a[sb_t_idx_b]) / sigma[sb_t_idx_b]
            ) ** 2
            z2_tmp[sb_t_idx_a] = (
                (hyp[sb_t_idx_a] - b[sb_t_idx_a]) / sigma[sb_t_idx_a]
            ) ** 2

            if np.any(sb_t_idx_b | sb_t_idx_a):
                tmp_idx = sb_t_idx_b | sb_t_idx_a
                lp += np.sum(
                    sp.special.gammaln(0.5 * (df[tmp_idx] + 1))
                    - sp.special.gammaln(0.5 * df[tmp_idx])
                )
                lp += np.sum(
                    -0.5 * np.log(np.pi * df[tmp_idx])
                    - np.log(C[tmp_idx[sb_t_idx]] * sigma[tmp_idx])
                    - 0.5
                    * (df[tmp_idx] + 1)
                    * np.log1p(z2_tmp[tmp_idx] / df[tmp_idx])
                )
            if np.any(sb_t_idx_btw):
                tmp_idx = sb_t_idx_btw
                lp += np.sum(
                    sp.special.gammaln(0.5 * (df[tmp_idx] + 1))
                    - sp.special.gammaln(0.5 * df[tmp_idx])
                )
                lp += np.sum(
                    -0.5 * np.log(np.pi * df[tmp_idx])
                    - np.log(C[tmp_idx[sb_t_idx]] * sigma[tmp_idx])
                )

            if compute_grad:
                if np.any(sb_t_idx_b):
                    dlp[sb_t_idx_b] = (
                        -(df[sb_t_idx_b] + 1)
                        / df[sb_t_idx_b]
                        / (1 + z2_tmp[sb_t_idx_b] / df[sb_t_idx_b])
                        * (hyp[sb_t_idx_b] - a[sb_t_idx_b])
                        / sigma[sb_t_idx_b] ** 2
                    )
                if np.any(sb_t_idx_a):
                    dlp[sb_t_idx_a] = (
                        -(df[sb_t_idx_a] + 1)
                        / df[sb_t_idx_a]
                        / (1 + z2_tmp[sb_t_idx_a] / df[sb_t_idx_a])
                        * (hyp[sb_t_idx_a] - b[sb_t_idx_a])
                        / sigma[sb_t_idx_a] ** 2
                    )

        # Gaussian prior
        if masks["any_g"]:
            lp -= 0.5 * np.sum(
                np.log(2 * np.pi * sigma[g_idx] ** 2) + z2[g_idx]
            )
            if compute_grad:
                dlp[g_idx] = -(hyp[g_idx] - mu[g_idx]) / sigma[g_idx] ** 2

        # Student's t prior
        if masks["any_t"]:
            lp += np.sum(
                sp.special.gammaln(0.5 * (df[t_idx] + 1))
                - sp.special.gammaln(0.5 * df[t_idx])
            )
            lp += np.sum(
                -0.5 * np.log(np.pi * df[t_idx])
                - np.log(sigma[t_idx])
                - 0.5 * (df[t_idx] + 1) * np.log1p(z2[t_idx] / df[t_idx])
            )
            if compute_grad:
                dlp[t_idx] = (
                    -(df[t_idx] + 1)
                    / df[t_idx]
                    / (1 + z2[t_idx] / df[t_idx])
                    * (hyp[t_idx] - mu[t_idx])
                    / sigma[t_idx] ** 2
                )

        lp -= masks["log_norm"]

        if compute_grad:
            return lp, dlp

        return lp

    def log_likelihood(self, hyp: object, compute_grad: bool = False):
        """Compute the (positive) log marginal likelihood of the GP for given
        hyperparameters.

        Parameters
        ==========
        hyp : object
            Either an 1D array or a dictionary of hyperparameters.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        =======
        lZ : float
            The positive log marginal likelihood.
        dlZ : ndarray, shape (hyp_N,), optional
            The gradient with respect to hyperparameters.
        """
        if isinstance(hyp, dict):
            # One dictionary is one row of the array form.
            hyp = self.hyperparameters_from_dict(hyp)[0]
        if compute_grad:
            nlZ, dnlZ = self.__compute_nlZ(hyp, True, False)
            return -nlZ, -dnlZ
        return -self.__compute_nlZ(hyp, False, False)

    def log_posterior(self, hyp: object, compute_grad: bool = False):
        """Compute the (positive) log marginal likelihood of the GP with added
        log prior for given hyperparameters (that is, the unnormalized log
        posterior).

        Each hyperparameter's prior is renormalized over its bounds, so the
        value carries a constant that a hand-computed sum of the log
        marginal likelihood and the prior densities does not. The constant
        does not depend on the hyperparameters, and is zero for a
        hyperparameter with no bounds.

        Parameters
        ==========
        hyp : object
            Either an 1D array or a dictionary of hyperparameters.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        =======
        lZ_plus_posterior : float
            The positive log marginal likelihood with added log prior.
        dlZ_plus_d_posterior : ndarray, shape (hyp_N,), optional
            The gradient with respect to hyperparameters.

        Raises
        =======
        LinAlgError
            Raised when the Cholesky decomposition failed multiple times even
            by adding numerical stability values to the matrix.
        """
        if isinstance(hyp, dict):
            # One dictionary is one row of the array form.
            hyp = self.hyperparameters_from_dict(hyp)[0]
        if compute_grad:
            nlZ, dnlZ = self.__compute_nlZ(hyp, True, True)
            return -nlZ, -dnlZ
        return -self.__compute_nlZ(hyp, False, True)

    def __compute_nlZ(self, hyp, compute_grad, compute_prior, cache=None):
        if compute_grad:
            nlZ, dnlZ = self.__core_computation(hyp, 1, compute_grad, cache)
        else:
            nlZ = self.__core_computation(hyp, 1, compute_grad, cache)

        if compute_prior:
            if compute_grad:
                P, dP = self.__compute_log_priors(hyp, compute_grad)
                nlZ -= P
                dnlZ -= dP
            else:
                P = self.__compute_log_priors(hyp, compute_grad)
                nlZ -= P

        if compute_grad:
            return nlZ, dnlZ

        return nlZ

    def __gp_obj_fun(self, hyp, compute_grad, swap_sign, cache=None):
        if compute_grad:
            nlZ, dnlZ = self.__compute_nlZ(
                hyp, compute_grad, self.no_prior is not True, cache
            )
        else:
            nlZ = self.__compute_nlZ(
                hyp, compute_grad, self.no_prior is not True, cache
            )

        # Swap sign of negative log marginal likelihood (e.g. for sampling)
        if swap_sign:
            nlZ *= -1
            if compute_grad:
                dnlZ *= -1

        if compute_grad:
            return nlZ, dnlZ

        return nlZ

    def predict_full(
        self,
        x_star: np.ndarray,
        y_star: np.ndarray = None,
        s2_star: np.ndarray = None,
        add_noise: bool = False,
    ):
        """
        Compute the GP posterior mean and full covariance matrix for each
        hyperparameter sample.

        Parameters
        ==========
        x_star : ndarray, shape (M, D)
            The points we want to predict the values at.
        y_star : ndarray, shape (M, 1), optional
            True values at the points.
        s2_star : ndarray, shape (M, 1), optional
            Noise variance at the points.
        add_noise : bool, defaults to ``False``
            Whether to add the observation noise, which enters on the
            diagonal, to the returned covariance.

        Returns
        =======
        mu : ndarray, shape (M, sample_N)
            Posterior mean at the requested points for each hyperparameter
            sample.
        cov : ndarray, shape (M, M, sample_N)
            Covariance matrix for each hyperparameter sample. Its diagonal
            is not clamped at zero, unlike the variances
            :py:func:`predict` returns, so on a nearly singular posterior
            an entry can come out slightly negative.

        Raises
        ------
        ValueError
            Raised by the check of the shapes of the data given: when
            ``x_star`` is neither a 1-D nor a 2-D array or does not have
            ``D`` columns, when ``y_star`` does not hold one value per row
            of ``x_star``, or when an array ``s2_star`` does not hold one
            variance per row of ``x_star``.
        TypeError
            Raised when ``s2_star`` is neither an array, a number nor
            ``None``, such as a list.
        LinAlgError
            Raised when a posterior holds the negative inverse of the
            training covariance without its Cholesky factor, as one
            pickled by gpyreg 1.3.1 or earlier does where the smallest
            noise variance at the training inputs is below 1e-6, and the
            factorization that computes the factor again fails even after
            its noise is multiplied tenfold, up to ten times.
        """
        x_star, y_star, s2_star = self._convert_shapes(x_star, y_star, s2_star)
        s_N = self.posteriors.size
        N_star, _ = x_star.shape

        cov_N = self.covariance.hyperparameter_count(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        # Preallocate space
        mu = np.zeros((N_star, s_N))
        cov = np.zeros((s_N, N_star, N_star))

        for s in range(0, s_N):
            hyp = self.posteriors[s].hyp
            alpha = self.posteriors[s].alpha
            L = self.posteriors[s].L
            L_chol = self.posteriors[s].L_chol
            sW = self.posteriors[s].sW

            # Compute GP mean function at test points
            m_star = np.reshape(
                self.mean.compute(
                    hyp[cov_N + noise_N : cov_N + noise_N + mean_N], x_star
                ),
                (-1, 1),
            )

            # Compute kernel matrix
            K_star = self.covariance.compute(hyp[0:cov_N], x_star)

            if self.y is None:
                # No data, draw from prior
                tmp_mu = m_star
                C = K_star
            else:
                # Compute cross-kernel matrix Ks
                Ks = self.covariance.compute(
                    hyp[0:cov_N], self.X, X_star=x_star
                )

                # Conditional mean
                tmp_mu = m_star + np.dot(Ks.T, alpha)

                if L_chol:
                    V = sp.linalg.solve_triangular(
                        L,
                        np.tile(sW, (1, N_star)) * Ks,
                        trans=1,
                        check_finite=False,
                    )
                    C = K_star - np.dot(V.T, V)  # Predictive variances
                else:
                    # From the Cholesky factor of the matrix whose negative
                    # inverse L is: formed from L (`gplite_pred.m:95-96`),
                    # the covariance carries the rounding of the inverse,
                    # which grows as the noise shrinks.
                    V = sp.linalg.solve_triangular(
                        self.__low_noise_factor(s),
                        Ks,
                        trans=1,
                        check_finite=False,
                    )
                    C = K_star - np.dot(V.T, V)

            # Enforce symmetry if lost due to numerical errors.
            C = (C + C.T) / 2

            mu[:, s : s + 1] = tmp_mu
            cov[s, :, :] = C
            if add_noise:
                sn2_mult = self.posteriors[s].sn2_mult
                if sn2_mult is None:
                    sn2_mult = 1
                # Also the noise function. The observation noise is
                # independent between points, so it enters on the diagonal;
                # a constant noise function returns a scalar, which is
                # broadcast to one entry per point.
                sn2_star = self.noise.compute(
                    hyp[cov_N : cov_N + noise_N], x_star, y_star, s2_star
                )
                sn2_diag = np.broadcast_to(np.ravel(sn2_star), (N_star,))
                cov[s, :, :] += np.diag(sn2_diag) * sn2_mult

        return mu, cov.transpose(1, 2, 0)

    def predict(
        self,
        x_star: np.ndarray,
        y_star: np.ndarray = None,
        s2_star: np.ndarray = None,
        add_noise: bool = False,
        separate_samples: bool = False,
        return_lpd: bool = False,
        *,
        return_cross_covariance: bool = False,
    ):
        """
        Compute the GP posterior mean and variance at given points.

        Parameters
        ==========
        x_star : ndarray, shape (M, D)
            The points we want to predict the values at.
        y_star : ndarray, shape (M, 1), optional
            True values at the points.
        s2_star : ndarray, shape (M, 1), optional
            Noise variance at the points.
        add_noise : bool, defaults to ``False``
            Whether to add noise to the prediction results.
        separate_samples : bool, defaults to ``False``
            Whether to return the results separately for each hyperparameter
            sample or averaged.
        return_lpd : bool, defaults to ``False``
            Whether to return the log predictive density at the input
            points. The density always carries the observation noise,
            whichever variance ``add_noise`` selects for ``s2``. With
            ``separate_samples`` ``False`` it is the log density of a
            Gaussian whose mean is the mean of the per-sample means and
            whose variance is the mean of the per-sample predictive
            variances plus the sample variance (``ddof=1``) of the
            per-sample means, as ``gplite_pred.m`` pools them, and not the
            average of the per-sample log densities.
        return_cross_covariance : bool, defaults to ``False``
            Whether to append the latent training-to-prediction kernel
            matrices to the return values. The matrices are kept separate for
            each hyperparameter sample even when ``separate_samples`` is
            ``False``.

        Returns
        =======
        mu : ndarray
            Posterior mean at the requested points. If we requested
            separate samples the shape is ``(M, sample_N)`` while
            otherwise it is ``(M, 1)``.
        s2 : ndarray
            Variance at each point: the latent posterior variance, or, with
            ``add_noise``, that variance plus the observation noise. Pooled
            over several hyperparameter samples, it is the mean of the
            per-sample variances plus the sample variance (``ddof=1``) of
            the per-sample means. If we requested separate samples the
            shape is ``(M, sample_N)`` while otherwise it is ``(M, 1)``.
        lpd : ndarray, optional
            Log predictive density at each point. Returned when
            ``return_lpd`` is ``True`` and shaped like ``mu``.
        cross_covariance : tuple of ndarray or None, optional
            Returned when ``return_cross_covariance`` is ``True``. Entry
            ``s`` is the latent, unconditioned kernel matrix
            ``K(X, x_star)`` with shape ``(N, M)`` for hyperparameter sample
            ``s``. Prior-only GPs, which have no training targets, return
            ``None`` for every sample. The matrices are call-local and should
            be treated as read-only; custom covariance results may be copied
            to give each tuple entry stable values. Observation-noise inputs
            and ``add_noise`` do not affect these latent kernel matrices.

        Raises
        ------
        ValueError
            Raised by the check of the shapes of the data given: when
            ``x_star`` is neither a 1-D nor a 2-D array or does not have
            ``D`` columns, when ``y_star`` does not hold one value per row
            of ``x_star``, or when an array ``s2_star`` does not hold one
            variance per row of ``x_star``.
        TypeError
            Raised when ``s2_star`` is neither an array, a number nor
            ``None``, such as a list.
        ValueError
            Raised when ``return_lpd`` is ``True`` and ``y_star`` is
            ``None``.
        LinAlgError
            Raised when a posterior holds the negative inverse of the
            training covariance without its Cholesky factor, as one
            pickled by gpyreg 1.3.1 or earlier does where the smallest
            noise variance at the training inputs is below 1e-6, and the
            factorization that computes the factor again fails even after
            its noise is multiplied tenfold, up to ten times.
        """
        x_star, y_star, s2_star = self._convert_shapes(x_star, y_star, s2_star)

        s_N = self.posteriors.size
        N_star, D = x_star.shape

        # Preallocate space
        mu = np.zeros((N_star, s_N))
        s2 = np.zeros((N_star, s_N))
        if return_lpd:
            if y_star is None:
                raise ValueError(
                    "Cannot calculate log predictive density without y_star."
                )
            if separate_samples:
                lpd = np.zeros((N_star, s_N))
        if return_lpd or add_noise:
            y_s2 = np.zeros((N_star, s_N))
        if return_cross_covariance:
            cross_covariance = []
            retain_cross_covariance = _can_retain_cross_covariance(
                self.covariance
            )

        cov_N = self.covariance.hyperparameter_count(D)
        mean_N = self.mean.hyperparameter_count(D)
        noise_N = self.noise.hyperparameter_count()

        # Mean function at the test points for every hyperparameter sample
        # at once, (N_star, s_N); the per-sample values are those of
        # `mean.compute` (a loop over the samples when the mean function
        # has no batched form).
        mean_hyp = np.stack(
            [
                p.hyp[cov_N + noise_N : cov_N + noise_N + mean_N]
                for p in self.posteriors
            ]
        )
        compute_batched = getattr(self.mean, "compute_batched", None)
        if compute_batched is not None:
            # An inherited batched implementation must not bypass a more
            # specific compute override. Only its defining class or a
            # subclass can supply a compatible batched implementation.
            mro = type(self.mean).__mro__
            compute_owner = next(
                (cls for cls in mro if "compute" in cls.__dict__), None
            )
            batched_owner = next(
                (cls for cls in mro if "compute_batched" in cls.__dict__),
                None,
            )
            if (
                compute_owner is None
                or batched_owner is None
                or not issubclass(batched_owner, compute_owner)
            ):
                compute_batched = None
        if compute_batched is not None:
            m_star_all = compute_batched(mean_hyp, x_star)
        else:
            m_star_all = np.stack(
                [
                    np.reshape(self.mean.compute(h, x_star), (-1,))
                    for h in mean_hyp
                ],
                axis=1,
            )

        for s in range(0, s_N):
            hyp = self.posteriors[s].hyp
            alpha = self.posteriors[s].alpha
            L = self.posteriors[s].L
            L_chol = self.posteriors[s].L_chol
            sW = self.posteriors[s].sW

            m_star = m_star_all[:, s]

            kss = self.covariance.compute(
                hyp[0:cov_N], x_star, compute_diag=True
            )[:, 0]

            if self.y is not None:
                Ks = self.covariance.compute(hyp[0:cov_N], self.X, x_star)
                if return_cross_covariance:
                    if retain_cross_covariance:
                        cross_covariance.append(Ks)
                    else:
                        cross_covariance.append(
                            np.array(Ks, copy=True, order="K", subok=False)
                        )
                mu[:, s] = (
                    m_star + np.dot(Ks.T, alpha)[:, 0]
                )  # Conditional mean

                if L_chol:
                    # sW is (N, 1): broadcasting over the columns of Ks
                    # gives the products a tiled sW gives, without the copy.
                    V = _solve_triangular(L, sW * Ks, trans=1)
                    s2[:, s] = kss - np.sum(V * V, 0)  # predictive variance
                else:
                    # From the Cholesky factor of the matrix whose negative
                    # inverse L is: formed from L (`gplite_pred.m:95-96`),
                    # the variance carries the rounding of the inverse,
                    # which grows as the noise shrinks.
                    V = _solve_triangular(
                        self.__low_noise_factor(s), Ks, trans=1
                    )
                    s2[:, s] = kss - np.sum(V * V, 0)
            else:
                if return_cross_covariance:
                    cross_covariance.append(None)
                mu[:, s] = m_star
                s2[:, s] = kss

            # remove numerical noise, i.e. negative variances
            s2[:, s] = np.maximum(s2[:, s], 0)
            if return_lpd or add_noise:  # Both require predictive variance
                sn2_mult = self.posteriors[s].sn2_mult
                if sn2_mult is None:
                    sn2_mult = 1
                sn2_star = self.noise.compute(
                    hyp[cov_N : cov_N + noise_N], x_star, y_star, s2_star
                )
                # Predictive variance:
                y_s2[:, s : s + 1] = s2[:, s : s + 1] + sn2_star * sn2_mult

            # Compute log probability of test points (for separate samples)
            if return_lpd and separate_samples:
                lpd[:, s : s + 1] = -0.5 * (
                    y_star - mu[:, s : s + 1]
                ) ** 2 / y_s2[:, s : s + 1] - 0.5 * np.log(
                    2 * np.pi * y_s2[:, s : s + 1]
                )

        if add_noise:
            s2 = y_s2
        # Unless predictions for samples are requested separately
        # average over samples.
        if not separate_samples:
            if s_N > 1:
                mu_bar = np.reshape(np.sum(mu, 1), (-1, 1)) / s_N
                v = np.sum((mu - mu_bar) ** 2, 1) / (s_N - 1)
                s2 = np.reshape(np.sum(s2, 1) / s_N + v, (-1, 1))
                mu = mu_bar
            else:
                v = 0

            # Compute log probability of test points (for averaged samples)
            if return_lpd and add_noise:  # then s2 is already y_s2 average
                lpd = -0.5 * (y_star - mu) ** 2 / s2 - 0.5 * np.log(
                    2 * np.pi * s2
                )
            elif return_lpd:  # then we need to average y_s2
                y_s2 = np.reshape(np.sum(y_s2, 1) / s_N + v, (-1, 1))
                lpd = -0.5 * (y_star - mu) ** 2 / y_s2 - 0.5 * np.log(
                    2 * np.pi * y_s2
                )

        if return_lpd:
            if return_cross_covariance:
                return mu, s2, lpd, tuple(cross_covariance)
            return mu, s2, lpd
        if return_cross_covariance:
            return mu, s2, tuple(cross_covariance)
        return mu, s2

    def quad(
        self,
        mu,
        sigma,
        compute_var: bool = False,
        separate_samples: bool = False,
    ):
        """
        Bayesian quadrature for a Gaussian Process.

        Compute the integral of a function represented by a Gaussian
        Process with respect to a given Gaussian measure.

        Parameters
        ==========
        mu : array_like
            Either a array of shape ``(N, D)`` with each row containing the
            mean of a single Gaussian measure, or a single floating point
            number which is interpreted as an array of shape ``(1, D)``. A
            one-dimensional array of length ``D`` is one measure of ``D``
            dimensions, as ``gplite_quad.m``'s ``size(mu, 1)`` reads a row
            vector.
        sigma : array_like
            Either a array of shape ``(N, D)`` with each row containing the
            standard deviation of a single Gaussian measure, or a single
            floating point number which is interpreted as an array of shape
            ``(1, D)``, or an array of shape ``(N, 1)`` with one standard
            deviation per measure, the same in every dimension. A
            one-dimensional array of length ``D`` is one measure, as for
            ``mu``.
        compute_var : bool, defaults to False
            Whether to compute variance for each integral.
        separate_samples : bool, defaults to False
            Whether to return the results separately for each hyperparameter
            sample, or averaged.

        Returns
        =======
        F : ndarray
            The computed integrals in an array with shape ``(N, 1)`` if
            samples are averaged and shape ``(N, hyp_samples)`` if
            requested separately.
        F_var : ndarray, optional
            The computed variances of the integrals in an array with
            shape ``(N, 1)`` if samples are averaged and shape
            ``(N, hyp_samples)`` if requested separately.

        Raises
        ------
        ValueError
            Raised when the method is called and the covariance of the GP is
            not squared exponential, or the mean function is none of the
            zero, constant and negative quadratic means.
        ValueError
            Raised when the GP has no training data or no posterior
            factors, or when ``mu`` does not have one column per input
            dimension, or ``sigma`` neither one nor one per dimension.
        LinAlgError
            Raised when ``compute_var`` is True and a posterior holds the
            negative inverse of the training covariance without its
            Cholesky factor, as one pickled by gpyreg 1.3.1 or earlier does
            where the smallest noise variance at the training inputs is
            below 1e-6, and the factorization that computes the factor
            again fails even after its noise is multiplied tenfold, up to
            ten times.
        """

        if not isinstance(
            self.covariance, gpyreg.covariance_functions.SquaredExponential
        ):
            raise ValueError(
                "Bayesian quadrature only supports the squared exponential "
                "kernel."
            )
        if not isinstance(
            self.mean,
            (
                gpyreg.mean_functions.ZeroMean,
                gpyreg.mean_functions.ConstantMean,
                gpyreg.mean_functions.NegativeQuadratic,
            ),
        ):
            raise ValueError(
                "Bayesian quadrature only supports the zero, constant and "
                "negative quadratic mean functions."
            )
        if self.X is None or self.y is None:
            raise ValueError(
                "Bayesian quadrature needs the training data of the GP, "
                "which has none."
            )
        if self.posteriors is None or self.posteriors[0].alpha is None:
            raise ValueError(
                "Bayesian quadrature needs the posterior factors of the "
                "GP; call `update` with `compute_posterior=True` first."
            )

        N, D = self.X.shape
        # Number of hyperparameter samples.
        N_s = np.size(self.posteriors)

        # Number of GP hyperparameters.
        cov_N = self.covariance.hyperparameter_count(self.D)
        # mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        # A one-dimensional input is one measure of D dimensions, as
        # `gplite_quad.m:26`'s `size(mu, 1)` reads a row vector.
        mu = np.atleast_2d(np.asarray(mu, dtype=float))
        sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
        if np.size(mu) == 1:
            mu = np.tile(mu, (1, D))
        # A sigma of one column holds one standard deviation per measure,
        # the same in every dimension, which `gplite_quad.m` broadcasts.
        if sigma.shape[1] == 1:
            sigma = np.tile(sigma, (1, D))
        if mu.shape[1] != D or sigma.shape[1] != D:
            raise ValueError(
                "Each Gaussian measure needs one column per input "
                f"dimension, {D} of them, and sigma may also have one "
                f"column: mu has {mu.shape[1]} and sigma {sigma.shape[1]}."
            )

        N_star = mu.shape[0]

        quadratic_mean_fun = isinstance(
            self.mean, gpyreg.mean_functions.NegativeQuadratic
        )
        isotropic = isinstance(
            self.covariance, isotropic_covariance.SquaredExponentialIsotropic
        )

        F = np.zeros((N_star, N_s))
        if compute_var:
            F_var = np.zeros((N_star, N_s))

        # Loop over hyperparameter samples.
        for s in range(0, N_s):
            hyp = self.posteriors[s].hyp

            # Extract GP hyperparameters
            if isotropic:
                # A single shared log lengthscale, then the log output
                # scale.
                ell = np.full(D, np.exp(hyp[0]))
                ln_sf2 = 2 * hyp[1]
                sum_lnell = D * hyp[0]
            else:
                ell = np.exp(hyp[0:D])
                ln_sf2 = 2 * hyp[D]
                sum_lnell = np.sum(hyp[0:D])

            # GP mean function hyperparameters
            if isinstance(self.mean, gpyreg.mean_functions.ZeroMean):
                m0 = 0
            else:
                m0 = hyp[cov_N + noise_N]

            if quadratic_mean_fun:
                xm = hyp[cov_N + noise_N + 1 : cov_N + noise_N + D + 1]
                omega = np.exp(hyp[cov_N + noise_N + D + 1 :])

            # GP posterior parameters
            alpha = self.posteriors[s].alpha
            L = self.posteriors[s].L
            L_chol = self.posteriors[s].L_chol

            if compute_var and L_chol:
                # Normalization of the stored Cholesky factor,
                # L = chol((K + sn2_mult * sn2) / sl). The scale is the one
                # the factor was built with, which a rank-one update keeps
                # and the minimum of the current training noise need not
                # reproduce.
                sl = self.posteriors[s]._noise_scale()

            # Compute posterior mean of the integral
            tau = np.sqrt(sigma**2 + ell**2)
            lnnf = (
                ln_sf2 + sum_lnell - np.sum(np.log(tau), 1)
            )  # Covariance normalization factor
            sum_delta2 = np.zeros((N_star, N))

            for i in range(0, D):
                sum_delta2 += (
                    (mu[:, i] - np.reshape(self.X[:, i], (-1, 1))).T
                    / tau[:, i : i + 1]
                ) ** 2
            z = np.exp(np.reshape(lnnf, (-1, 1)) - 0.5 * sum_delta2)
            F[:, s : s + 1] = np.dot(z, alpha) + m0

            if quadratic_mean_fun:
                nu_k = -0.5 * np.sum(
                    1
                    / omega**2
                    * (mu**2 + sigma**2 - 2 * mu * xm + xm**2),
                    1,
                )
                F[:, s] += nu_k

            # Compute posterior variance of the integral
            if compute_var:
                tau_kk = np.sqrt(2 * sigma**2 + ell**2)
                nf_kk = np.exp(ln_sf2 + sum_lnell - np.sum(np.log(tau_kk), 1))
                if L_chol:
                    tmp_result = sp.linalg.solve_triangular(
                        L, z.T, trans=1, check_finite=False
                    )
                    invKzk = (
                        sp.linalg.solve_triangular(
                            L, tmp_result, trans=0, check_finite=False
                        )
                        / sl
                    )
                    J_kk = nf_kk - np.sum(z * invKzk.T, 1)
                else:
                    # From the Cholesky factor of the matrix whose negative
                    # inverse L is, as `predict` forms its variance: formed
                    # from L, the variance of the integral carries the
                    # rounding of the inverse, which grows as the noise
                    # shrinks.
                    W = sp.linalg.solve_triangular(
                        self.__low_noise_factor(s),
                        z.T,
                        trans=1,
                        check_finite=False,
                    )
                    J_kk = nf_kk - np.sum(W * W, 0)
                F_var[:, s] = np.maximum(
                    np.spacing(1), J_kk
                )  # Correct for numerical error

        # Unless predictions for samples are requested separately
        # average over samples
        if N_s > 1 and not separate_samples:
            F_bar = np.reshape(np.sum(F, 1), (-1, 1)) / N_s
            if compute_var:
                Fss_var = np.sum((F - F_bar) ** 2, 1) / (N_s - 1)
                F_var = np.reshape(np.sum(F_var, 1) / N_s + Fss_var, (-1, 1))
            F = F_bar

        if compute_var:
            return F, F_var

        return F

    # quantile doesn't work, requires gplite_qpred implementation
    def plot(
        self,
        x0: np.ndarray = None,
        lb: np.ndarray = None,
        ub: np.ndarray = None,
        delta_y: float = None,
        max_min_flag: bool = True,
    ):
        """
        Plot the Gaussian Process profile centered around a given point.

        The plot is a D-by-D panel matrix, in which panels on the diagonal
        show the profile of the Gaussian Process prediction (mean and +/- 1 SD)
        by varying one dimension at a time, whereas off-diagonal panels show
        2-D contour plots of the GP mean and standard deviation (respectively,
        above and below diagonal). In each panel, black lines indicate the
        location of the reference point.

        Parameters
        ==========
        x0 : ndarray, shape (D,), optional
            The reference point.
        lb : ndarray, shape (D,), optional
            Lower bounds for the plotting.
        ub : ndarray, shape (D,), optional
            Upper bounds for the plotting.
        delta_y : float, optional
            Range of the plot such that the plotted predictive GP mean
            approximately brackets ``[y0-delta_y, y0+delta_y]`` where
            ``y0`` is the predictive GP mean at ``x0``. If lower or upper
            bounds are given this will do nothing.
        max_min_flag : bool, defaults to True
            If set to ``False`` then the minimum, and if set to ``True``
            then the maximum of the GP training input is used as the reference
            point.
        """
        if lb is not None or ub is not None:
            delta_y = None

        s_N = self.posteriors.size  # Hyperparameter samples
        x_N = 100  # Grid points per visualization

        # Loop over hyperparameter samples.
        ell = np.zeros((self.D, s_N))
        for s in range(0, s_N):
            ell[:, s] = np.exp(
                self.posteriors[s].hyp[0 : self.D]
            )  # Extract length scale from HYP
        ellbar = np.sqrt(np.mean(ell**2, 1)).T

        if lb is None:
            if self.X is not None:
                lb = np.min(self.X, axis=0) - ellbar
            else:
                lb = -ellbar
        if ub is None:
            if self.X is not None:
                ub = np.max(self.X, axis=0) + ellbar
            else:
                ub = ellbar

        gutter = [0.05, 0.05]
        margins = [0.1, 0.01, 0.12, 0.01]
        linewidth = 1

        if x0 is None:
            if self.X is not None and self.y is not None:
                if max_min_flag:
                    i = np.argmax(self.y)
                else:
                    i = np.argmin(self.y)
                x0 = self.X[i, :]

        _, ax = plt.subplots(self.D, self.D, squeeze=False)

        flo = fhi = None
        for i in range(0, self.D):
            ax[i, i].set_position(
                self.__tight_subplot(self.D, self.D, i, i, gutter, margins)
            )

            xx_vec = np.reshape(
                np.linspace(lb[i], ub[i], np.ceil(x_N**1.5).astype(int)),
                (-1, 1),
            )
            if self.D > 1:
                if x0 is not None:
                    xx = np.tile(x0, (np.size(xx_vec), 1))
                else:
                    xx = np.tile(np.full((self.D,), 0.0), (np.size(xx_vec), 1))
                xx[:, i : i + 1] = xx_vec
            else:
                xx = xx_vec

            # do we need to add quantile prediction stuff etc here?
            fmu, fs2 = self.predict(xx, add_noise=False)
            flo = fmu - 1.96 * np.sqrt(fs2)
            fhi = fmu + 1.96 * np.sqrt(fs2)

            if delta_y is not None:
                fmu0, _ = self.predict(
                    np.reshape(x0, (1, -1)), add_noise=False
                )
                dx = xx_vec[1] - xx_vec[0]
                region = np.abs(fmu - fmu0) < delta_y
                if np.any(region):
                    idx1 = np.argmax(region)
                    idx2 = np.size(region) - np.argmax(region[::-1]) - 1
                    lb[i] = xx_vec[idx1] - 0.5 * dx
                    ub[i] = xx_vec[idx2] + 0.5 * dx
                else:
                    lb[i] = x0[i] - 0.5 * dx
                    ub[i] = x0[i] + 0.5 * dx

                xx_vec = np.reshape(
                    np.linspace(lb[i], ub[i], np.ceil(x_N**1.5).astype(int)),
                    (-1, 1),
                )
                if self.D > 1:
                    xx = np.tile(x0, (np.size(xx_vec), 1))
                    xx[:, i : i + 1] = xx_vec
                else:
                    xx = xx_vec

                # do we need to add quantile prediction stuff etc here?
                fmu, fs2 = self.predict(xx, add_noise=False)
                flo = fmu - 1.96 * np.sqrt(fs2)
                fhi = fmu + 1.96 * np.sqrt(fs2)

            ax[i, i].plot(xx_vec, fmu, "-k", linewidth=linewidth)
            ax[i, i].plot(
                xx_vec, fhi, "-", color=(0.8, 0.8, 0.8), linewidth=linewidth
            )
            ax[i, i].plot(
                xx_vec, flo, "-", color=(0.8, 0.8, 0.8), linewidth=linewidth
            )
            ax[i, i].set_xlim(lb[i], ub[i])
            ax[i, i].set_ylim(ax[i, i].get_ylim())

            # ax[i, i].tick_params(direction='out')
            ax[i, i].spines["top"].set_visible(False)
            ax[i, i].spines["right"].set_visible(False)

            if self.D == 1:
                ax[i, i].set_xlabel("x")
                ax[i, i].set_ylabel("y")
                if self.X is not None and self.y is not None:
                    ax[i, i].scatter(self.X, self.y, color="blue")
            else:
                if i == 0:
                    ax[i, i].set_ylabel(r"$x_" + str(i + 1) + r"$")
                if i == self.D - 1:
                    ax[i, i].set_xlabel(r"$x_" + str(i + 1) + r"$")
            if x0 is not None:
                ax[i, i].vlines(
                    x0[i],
                    ax[i, i].get_ylim()[0],
                    ax[i, i].get_ylim()[1],
                    colors="k",
                    linewidth=linewidth,
                )

        for i in range(0, self.D):
            for j in range(0, i):
                xx1_vec = np.reshape(np.linspace(lb[i], ub[i], x_N), (-1, 1)).T
                xx2_vec = np.reshape(np.linspace(lb[j], ub[j], x_N), (-1, 1)).T
                xx_vec = np.array(np.meshgrid(xx1_vec, xx2_vec)).T.reshape(
                    -1, 2
                )

                if x0 is not None:
                    xx = np.tile(x0, (x_N**2, 1))
                else:
                    xx = np.tile(np.full((self.D,), 0.0), (x_N**2, 1))
                xx[:, i] = xx_vec[:, 0]
                xx[:, j] = xx_vec[:, 1]

                fmu, fs2 = self.predict(xx, add_noise=False)

                for k in range(0, 2):
                    if k == 1:
                        i1 = j
                        i2 = i
                        mat = np.reshape(fmu, (x_N, x_N)).T
                    else:
                        i1 = 1
                        i2 = j
                        mat = np.reshape(np.sqrt(fs2), (x_N, x_N))
                    ax[i1, i2].set_position(
                        self.__tight_subplot(
                            self.D, self.D, i1, i2, gutter, margins
                        )
                    )
                    ax[i1, i2].spines["top"].set_visible(False)
                    ax[i1, i2].spines["right"].set_visible(False)

                    if k == 1:
                        Xt, Yt = np.meshgrid(xx1_vec, xx2_vec)
                        ax[i1, i2].contour(Xt, Yt, mat)
                    else:
                        Xt, Yt = np.meshgrid(xx2_vec, xx1_vec)
                        ax[i1, i2].contour(Xt, Yt, mat)
                    ax[i1, i2].set_xlim(lb[i2], ub[i2])
                    ax[i1, i2].set_ylim(lb[i1], ub[i1])
                    if self.X is not None:
                        ax[i1, i2].scatter(
                            self.X[:, i2], self.X[:, i1], color="blue", s=10
                        )

                    if x0 is not None:
                        ax[i1, i2].hlines(
                            x0[i1],
                            ax[i1, i2].get_xlim()[0],
                            ax[i1, i2].get_xlim()[1],
                            colors="k",
                            linewidth=linewidth,
                        )
                        ax[i1, i2].vlines(
                            x0[i2],
                            ax[i1, i2].get_ylim()[0],
                            ax[i1, i2].get_ylim()[1],
                            colors="k",
                            linewidth=linewidth,
                        )

                if j == 0:
                    ax[i, j].set_ylabel(r"$x_" + str(i + 1) + r"$")
                if i == self.D - 1:
                    ax[i, j].set_xlabel(r"$x_" + str(j + 1) + r"$")

        plt.show()

    @staticmethod
    def __tight_subplot(m, n, row, col, gutter=None, margins=None):
        if gutter is None:
            gutter = [0.002, 0.002]
        if margins is None:
            margins = [0.06, 0.01, 0.04, 0.04]
        Lmargin = margins[0]
        Rmargin = margins[1]
        Bmargin = margins[2]
        Tmargin = margins[3]

        unit_height = (1 - Bmargin - Tmargin - (m - 1) * gutter[1]) / m
        height = np.size(row) * unit_height + (np.size(row) - 1) * gutter[1]

        unit_width = (1 - Lmargin - Rmargin - (n - 1) * gutter[0]) / n
        width = np.size(col) * unit_width + (np.size(col) - 1) * gutter[0]

        bottom = (m - np.max(row) - 1) * (unit_height + gutter[1]) + Bmargin
        left = np.min(col) * (unit_width + gutter[0]) + Lmargin

        pos_vec = [left, bottom, width, height]

        return pos_vec

    def random_function(
        self, X_star: np.ndarray, add_noise: bool = False, rng=None
    ):
        """
        Draw a random function from the Gaussian Process.

        Parameters
        ==========
        X_star : ndarray, shape (M, D)
            The points at which to evaluate the drawn function.
        add_noise : bool, defaults to False
            Whether to add noise to the values of the drawn function.
        rng : None, numpy.random.Generator or seed, optional
            Where the draws come from (the hyperparameter sample and the
            function values). ``None`` (default) keeps NumPy's global legacy
            stream, as before generators were supported. See
            :func:`gpyreg.rng.resolve_rng`.

        Returns
        =======
        f_star : ndarray, shape (M, 1)
            The values of the drawn function at the requested points.

        Raises
        ------
        LinAlgError
            Raised when the covariance of the draw, as computed, has a
            negative eigenvalue beyond the rounding of the prior variance
            at ``X_star``.
        LinAlgError
            Raised when the posterior drawn from holds the negative inverse
            of the training covariance without its Cholesky factor, as one
            pickled by gpyreg 1.3.1 or earlier does where the smallest
            noise variance at the training inputs is below 1e-6, and the
            factorization that computes the factor again fails even after
            its noise is multiplied tenfold, up to ten times.
        """
        rng = resolve_rng(rng)
        N_star = X_star.shape[0]
        N_s = np.size(self.posteriors)

        cov_N = self.covariance.hyperparameter_count(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        # Draw from hyperparameter samples.
        s = random_integer(rng, N_s)

        hyp = self.posteriors[s].hyp
        alpha = self.posteriors[s].alpha
        L = self.posteriors[s].L
        L_chol = self.posteriors[s].L_chol
        sW = self.posteriors[s].sW

        # Compute GP mean function at test points
        m_star = np.reshape(
            self.mean.compute(
                hyp[cov_N + noise_N : cov_N + noise_N + mean_N], X_star
            ),
            (-1, 1),
        )

        # Compute kernel matrix
        K_star = self.covariance.compute(hyp[0:cov_N], X_star)

        if self.y is None:
            # No data, draw from prior
            f_mu = m_star
            C = K_star + np.spacing(1) * np.eye(N_star)
        else:
            # Compute cross-kernel matrix Ks
            Ks = self.covariance.compute(hyp[0:cov_N], self.X, X_star=X_star)

            # Conditional mean
            f_mu = m_star + np.dot(Ks.T, alpha)

            if L_chol:
                V = sp.linalg.solve_triangular(
                    L,
                    np.tile(sW, (1, N_star)) * Ks,
                    trans=1,
                    check_finite=False,
                )
            else:
                # The posterior holds the explicit inverse
                # -inv(K + sn2_mult * diag(sn2)), whose rounding grows as
                # the noise shrinks, and a covariance formed from it would
                # carry that rounding; the Cholesky factor of the matrix
                # itself does not.
                V = sp.linalg.solve_triangular(
                    self.__low_noise_factor(s),
                    Ks,
                    trans=1,
                    check_finite=False,
                )
            C = K_star - np.dot(V.T, V)  # Predictive variances

        # Enforce symmetry if lost due to numerical errors.
        C = (C + C.T) / 2

        # Draw random function. The predictive covariance is the prior
        # covariance minus what the data explain, so its rounding is
        # that of the prior variance at the test points, however small
        # the difference. The observation noise is drawn apart, below.
        T = self.__robust_cholesky(
            C, scale=np.max(np.diag(K_star), initial=0.0)
        )
        f_star = np.dot(T.T, rng.standard_normal((T.shape[0], 1))) + f_mu

        # Add observation noise.
        if add_noise:
            # Get observation noise hyperparameters and evaluate noise
            # at test points.
            sn2 = self.noise.compute(
                hyp[cov_N : cov_N + noise_N], X_star, None, None
            )
            sn2_mult = self.posteriors[s].sn2_mult
            if sn2_mult is None:
                sn2_mult = 1
            y_star = f_star + np.sqrt(sn2 * sn2_mult) * rng.standard_normal(
                size=f_mu.shape
            )
            return y_star

        return f_star

    def __low_noise_factor(self, s):
        """The upper triangular Cholesky factor of
        ``K + sn2_mult * diag(sn2)`` at the training inputs, for the
        posterior ``s`` in the low-noise representation, whose ``L`` is the
        negative inverse of that matrix. A posterior pickled without it
        has it computed from the kernel, the noise and the multiplier of
        the posterior, as the factorization of the posterior computed it.
        """
        posterior = self.posteriors[s]
        L_factor = getattr(posterior, "L_factor", None)
        if L_factor is not None:
            return L_factor
        cov_N = self.covariance.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()
        hyp = posterior.hyp
        K = self.covariance.compute(hyp[0:cov_N], self.X)
        sn2 = self.noise.compute(
            hyp[cov_N : cov_N + noise_N], self.X, self.y, self.s2
        )
        L_factor, __, __ = self.__training_cholesky(
            K, sn2, False, posterior.sn2_mult
        )
        return L_factor

    @staticmethod
    def __robust_cholesky(sigma, scale=None):
        """Cholesky-like decomposition for a covariance matrix.

        Returns a factor ``T`` with ``T.T @ T == sigma`` up to rounding,
        from the eigendecomposition where the direct Cholesky
        decomposition fails.

        Parameters
        ==========
        sigma : ndarray, shape (n, n)
            The covariance matrix.
        scale : float, optional
            The magnitude of the terms that formed ``sigma``, which sets
            the rounding tolerance where ``sigma`` is a difference of
            larger terms. Defaults to the largest absolute eigenvalue of
            ``sigma``, which also bounds the tolerance from below.

        Raises
        ------
        LinAlgError
            Raised when ``sigma`` has a negative eigenvalue beyond the
            rounding tolerance, so that no such factor exists.
        """
        try:
            T = sp.linalg.cholesky(sigma, check_finite=False)
        except sp.linalg.LinAlgError:
            # The symmetric solver: real eigenvectors, and an orthogonal
            # basis of a repeated eigenvalue, both of which the general
            # solver may fail to return.
            D, U = sp.linalg.eigh((sigma + sigma.T) / 2)
            # Sign convention: the largest entry of each eigenvector is
            # positive. Flipping a whole column leaves U D U^T unchanged.
            maxidx = np.argmax(np.abs(U), axis=0)
            negidx = U[maxidx, np.arange(U.shape[1])] < 0
            U[:, negidx] *= -1

            # Which eigenvalues carry the matrix rather than its
            # rounding (`gplite_rnd.m:102`). The abs is there to make sure
            # we don't have issues if np.spacing returns negative values.
            tol = np.abs(np.spacing(np.max(D))) * D.shape[0]
            t = np.abs(D) > tol

            # A surviving eigenvalue that is negative but of rounding size
            # is a zero of a semidefinite matrix: the symmetric solver is
            # backward stable, so an eigenvalue of a matrix of spectral
            # norm max|D| carries an error of order n * eps * max|D|, and
            # the factor of ten leaves room for the constant that bound
            # hides. A matrix formed as the difference of larger terms
            # carries the rounding of those terms, of order n * eps *
            # scale, however small the difference. A near-singular
            # predictive covariance, the case that brings a draw here,
            # has such eigenvalues of both signs.
            magnitude = np.max(np.abs(D))
            if scale is not None:
                magnitude = max(magnitude, scale)
            rounding = 10 * D.shape[0] * np.finfo(D.dtype).eps * magnitude
            negative = t & (D < 0)
            if np.any(D[negative] <= -rounding):
                # Not a covariance matrix: it has no factor.
                raise sp.linalg.LinAlgError(
                    "Matrix is not positive semidefinite: its smallest "
                    f"eigenvalue is {np.min(D):.6g}, beyond the rounding "
                    f"tolerance {rounding:.6g}."
                )
            t &= ~negative
            T = np.dot(np.diag(np.sqrt(D[t])), U[:, t].T)

        return T

    @staticmethod
    def __training_cholesky(K, sn2, L_chol, sn2_mult=1):
        """Cholesky factor of the training covariance with its noise.

        Factors ``(K + sn2_mult * diag(sn2)) / sl``, multiplying
        ``sn2_mult`` by ten after each failed attempt, up to ten attempts.

        Parameters
        ==========
        K : ndarray, shape (N, N)
            The kernel matrix at the training inputs.
        sn2 : float or ndarray, shape (N, 1)
            The noise variance at the training inputs.
        L_chol : bool
            Whether the matrix is scaled by ``sl = min(sn2) * sn2_mult``
            before it is factored (the Cholesky representation of the
            posterior) or not (``sl = 1``, the matrix whose negative
            inverse the low-noise representation holds).
        sn2_mult : int, defaults to 1
            The noise multiplier of the first attempt.

        Returns
        =======
        L : ndarray, shape (N, N)
            The upper triangular Cholesky factor.
        sl : float
            The scale the matrix was divided by before it was factored.
        sn2_mult : int
            The noise multiplier of the attempt that succeeded.

        Raises
        ======
        LinAlgError
            Raised when every attempt failed.
        """
        N = K.shape[0]
        L = None
        # The noise enters on the diagonal only: adding it in place to
        # a copy gives the entries of `K / sl + diag(...)` exactly
        # (adding 0.0 off the diagonal leaves an entry unchanged)
        # without forming and adding an N x N identity on every
        # evaluation. The copy is made C-contiguous, the layout the
        # old sum with a C-ordered identity produced (the factorization
        # scipy computes depends on the layout at rounding level).
        # Use float64 so custom float32 kernels do not lose small
        # diagonal noise that the old sum with an identity preserved.
        if L_chol:
            if np.isscalar(sn2):
                sn2_div = sn2
                sn2_diag = 1.0
            else:
                sn2_div = np.min(sn2)
                sn2_diag = sn2.ravel() / sn2_div
            for i in range(0, 10):
                try:  # Cholesky decomposition until it works
                    A = np.ascontiguousarray(
                        K / (sn2_div * sn2_mult), dtype=np.float64
                    )
                    A.flat[:: N + 1] += sn2_diag
                    L = sp.linalg.cholesky(A, check_finite=False)
                except sp.linalg.LinAlgError:
                    sn2_mult *= 10
                    continue
                break
            sl = sn2_div * sn2_mult
        else:
            sn2_diag = sn2 if np.isscalar(sn2) else sn2.ravel()

            for i in range(0, 10):
                try:
                    A = np.array(K, dtype=np.float64, order="C")
                    A.flat[:: N + 1] += sn2_mult * sn2_diag
                    L = sp.linalg.cholesky(A, check_finite=False)
                except sp.linalg.LinAlgError:
                    sn2_mult *= 10
                    continue
                break
            sl = 1

        if L is None:
            raise sp.linalg.LinAlgError(
                "Singular matrix for L Cholesky decomposition"
            )
        return L, sl, sn2_mult

    def __core_computation(
        self, hyp, compute_nlZ, compute_nlZ_grad, cache=None
    ):
        """Compute the Posterior.

        ``cache``, a dict owned by the caller, makes consecutive
        no-gradient log-likelihood evaluations reuse the Cholesky factor
        when only the mean-function hyperparameters changed: the kernel
        and the noise, and with them ``K + sn2 I``, its factor and its log
        determinant, depend on the covariance and noise blocks of ``hyp``
        alone. The slice sampler moves one coordinate per evaluation, so
        with a quadratic mean about two thirds of its evaluations qualify.
        A hit recomputes only the mean, ``alpha`` and the quadratic form,
        on the very factor a fresh computation would produce, so the result
        is bit-identical. The gradient path never uses the cache (it needs
        the kernel derivatives); neither does the ``Posterior`` path.

            Raises
            ------
        LinAlgError
            Raised when the Cholesky decomposition failed multiple times even
            by adding numerical stability values to the matrix.
        """
        N, d = self.X.shape
        cov_N = self.covariance.hyperparameter_count(d)
        mean_N = self.mean.hyperparameter_count(d)
        noise_N = self.noise.hyperparameter_count()

        use_cache = (
            _REUSE_CHOLESKY
            and cache is not None
            and compute_nlZ
            and not compute_nlZ_grad
        )
        key = hyp[: cov_N + noise_N]
        hit = (
            use_cache and "key" in cache and np.array_equal(cache["key"], key)
        )

        if compute_nlZ_grad:
            sn2, dsn2 = self.noise.compute(
                hyp[cov_N : cov_N + noise_N],
                self.X,
                self.y,
                self.s2,
                compute_grad=True,
            )
            m, dm = self.mean.compute(
                hyp[cov_N + noise_N : cov_N + noise_N + mean_N],
                self.X,
                compute_grad=True,
            )

            # This line is actually important due to behaviour of above
            # Maybe change that in the future.
            m = m.reshape((-1, 1))
            K, dK = self.covariance.compute(
                hyp[0:cov_N], self.X, compute_grad=True
            )
        else:
            m = np.reshape(
                self.mean.compute(
                    hyp[cov_N + noise_N : cov_N + noise_N + mean_N], self.X
                ),
                (-1, 1),
            )
            if not hit:
                sn2 = self.noise.compute(
                    hyp[cov_N : cov_N + noise_N], self.X, self.y, self.s2
                )
                K = self.covariance.compute(hyp[0:cov_N], self.X)

        if hit:
            L, sl, logdet = cache["L"], cache["sl"], cache["logdet"]
        else:
            L_chol = np.min(sn2) >= 1e-6
            L, sl, sn2_mult = self.__training_cholesky(K, sn2, L_chol)

            if L_chol:
                pL = L
            elif not compute_nlZ:
                pL = sp.linalg.solve_triangular(
                    -L,
                    sp.linalg.solve_triangular(
                        L, np.eye(N), trans=1.0, check_finite=False
                    ),
                    trans=0,
                    check_finite=False,
                )
            logdet = None

        # The same two triangular solves as scipy's, without its wrappers.
        alpha = (
            _solve_triangular(
                L, _solve_triangular(L, self.y - m, trans=1), trans=0
            )
            / sl
        )

        # Negative log marginal likelihood computation
        if compute_nlZ:
            if logdet is None:
                logdet = np.sum(np.log(np.diag(L)))
                if use_cache:
                    cache.update(key=key.copy(), L=L, sl=sl, logdet=logdet)
            nlZ = (
                np.dot((self.y - m).T, alpha / 2)
                + logdet
                + N * np.log(2 * np.pi * sl) / 2
            )

            if compute_nlZ_grad:
                dnlZ = np.zeros(hyp.shape)
                Q = sp.linalg.solve_triangular(
                    L,
                    sp.linalg.solve_triangular(
                        L, np.eye(N), trans=1, check_finite=False
                    ),
                    trans=0,
                    check_finite=False,
                ) / sl - np.dot(alpha, alpha.T)

                # Gradient of covariance hyperparameters.
                for i in range(0, cov_N):
                    dnlZ[i] = np.sum(np.sum(Q * dK[:, :, i])) / 2

                # Gradient of GP likelihood
                if np.isscalar(sn2):
                    tr_Q = np.trace(Q)
                    # The noise gradient is (1, noise_N) where the total
                    # noise does not vary by point, and (N, noise_N) where
                    # the noise function has a feature that could make it
                    # vary; a constant total noise has the same row
                    # everywhere, so the entry of hyperparameter i is that
                    # of the first row. (`gplite_core.m:244` indexes the
                    # array linearly and reads another entry.)
                    dsn2_row = np.atleast_2d(dsn2)[0, :]
                    for i in range(0, noise_N):
                        dnlZ[cov_N + i] = 0.5 * sn2_mult * dsn2_row[i] * tr_Q
                else:
                    dg_Q = np.diag(Q)
                    for i in range(0, noise_N):
                        dnlZ[cov_N + i] = (
                            0.5 * sn2_mult * np.sum(dsn2[:, i] * dg_Q)
                        )

                # Gradient of mean function.
                if mean_N > 0:
                    dnlZ[cov_N + noise_N :] = np.dot(-dm.T, alpha)[:, 0]

                return nlZ[0, 0], dnlZ

            return nlZ[0, 0]

        sl_post = np.min(sn2) * sn2_mult
        return Posterior(
            hyp,
            alpha,
            np.ones((N, 1)) / np.sqrt(sl_post),
            pL,
            sn2_mult,
            L_chol,
            sl_post,
            L_factor=None if L_chol else L,
        )

    def _convert_shapes(
        self,
        X: Union[np.ndarray, None],
        y: Union[np.ndarray, None],
        s2: Union[np.ndarray, float, int, None],
    ):
        """Convert input data to correct shapes."""
        if X is None and y is None and s2 is None:
            return X, y, s2

        if X is not None:
            if X.ndim == 1:
                X = X[None, :]
            if X.ndim != 2:
                raise ValueError("X need to be an array of shape (N, D)")
            N, D = X.shape
            if D != self.D:
                raise ValueError(
                    f"The dimension of input data {D} "
                    f"doesn't match GP's input dimension {self.D}."
                )
        else:
            try:
                N, D = self.X.shape
            except AttributeError as e:
                raise AttributeError(
                    "self.X is not a numpy array, " f"self.X = {self.X}"
                )

        if y is not None:
            y = y.reshape(N, 1)
        if isinstance(s2, np.ndarray) and s2.ndim > 0:
            # One variance per input, as `gplite_pred.m:16-23` requires:
            # a row of N is not a column of N.
            if s2.shape[0] != N:
                raise ValueError(
                    f"The noise variance has {s2.shape[0]} rows, but the "
                    f"input data has {N}."
                )
            s2 = s2.reshape(N, 1)
        elif isinstance(s2, numbers.Number) or isinstance(s2, np.ndarray):
            # A number, a NumPy scalar or a 0-d array: the same variance
            # at every input.
            s2 = float(s2) * np.ones((N, 1))
        elif s2 is None:
            s2 = None  # noiseless case
        else:
            raise TypeError(
                "s2 type need to be "
                "Union[np.ndarray, numbers.Number, None]."
            )
        return X, y, s2


class Posterior:
    """
    GP posterior for one hyperparameter vector.

    Stores the coefficients and matrix factor that prediction, sampling and
    quadrature need for a fixed set of hyperparameters, so that repeated
    calls reuse them instead of factorizing the training covariance again.
    ``GP.posteriors`` holds one instance per hyperparameter sample. Below,
    ``K`` is the training covariance, ``sn2`` the vector of training noise
    variances, ``m`` the mean function evaluated at the training inputs
    and ``sl = min(sn2) * sn2_mult`` at the time the posterior was
    computed; ``sl`` scales the Cholesky factor when ``L_chol`` is True.
    Every attribute except ``hyp`` is ``None`` when the posterior has not
    been computed, after :py:func:`GP.clean` or :py:func:`GP.update` with
    ``compute_posterior=False``.

    Attributes
    ==========
    hyp : ndarray, shape (hyp_N,)
        The hyperparameters this posterior was computed for.
    alpha : ndarray, shape (N, 1)
        ``inv(K + sn2_mult * diag(sn2)) @ (y - m)``, the weights of the
        training points in the posterior mean.
    sW : ndarray, shape (N, 1)
        ``1 / sqrt(sl)`` in every entry; the square root of the noise
        precision used to scale the factorization.
    L : ndarray, shape (N, N)
        If ``L_chol`` is True, the upper triangular Cholesky factor of
        ``(K + sn2_mult * diag(sn2)) / sl``. Otherwise
        ``-inv(K + sn2_mult * diag(sn2))``, the low-noise representation,
        as ``gplite_core.m`` has it where the noise is small.
    sn2_mult : int
        Multiplier applied to the noise variances, increased in powers of
        ten until the Cholesky decomposition succeeds.
    L_chol : bool
        Whether ``L`` is a Cholesky factor (``min(sn2) >= 1e-6``) or a
        negative inverse.
    sl : float
        ``min(sn2) * sn2_mult`` when the posterior was computed; the scale
        of ``L`` when ``L_chol`` is True. Rank-one updates extend the
        factorization with this scale, so it can differ from the minimum
        of the current training noise after observations have been
        appended.
    L_factor : ndarray, shape (N, N) or None
        If ``L_chol`` is False, the upper triangular Cholesky factor of
        ``K + sn2_mult * diag(sn2)``, the matrix whose negative inverse
        ``L`` is. :py:func:`GP.predict`, :py:func:`GP.predict_full`,
        :py:func:`GP.quad` and :py:func:`GP.random_function` form their
        covariances from it, and a single-point :py:func:`GP.update` takes
        from it the predictive variance of the new point and extends it:
        formed from ``L``, these would carry the rounding of the inverse,
        which grows as the noise shrinks. ``None`` if ``L_chol`` is True,
        where ``L`` is that factor, scaled. A posterior pickled by gpyreg
        1.3.1 or earlier has no such attribute: the predictions, the
        quadrature and the draws compute the factor again at each call,
        and a single-point update recomputes the posterior in full.
    """

    def __init__(
        self, hyp, alpha, sW, L, sn2_mult, Lchol, sl=None, L_factor=None
    ):
        self.hyp = hyp
        self.alpha = alpha
        self.sW = sW
        self.L = L
        self.sn2_mult = sn2_mult
        self.L_chol = Lchol
        self.sl = sl
        self.L_factor = L_factor

    def _noise_scale(self):
        """Return ``sl``, the scale of the factorization.

        Posteriors pickled before the scale was stored lack the attribute
        and recover it from ``sW``, whose entries are ``1 / sqrt(sl)``.
        """
        sl = getattr(self, "sl", None)
        if sl is None:
            sl = 1.0 / self.sW[0, 0] ** 2
        return sl
