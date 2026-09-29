import numpy as np

from pybads.decorators import handle_0D_1D_input


class VariableTransformer:
    """
    A class enabling linear or non-linear transformation of the bounds (plausible_lower_bounds and plausible_upper_bounds) and map them to an hypercube [-1, 1]^D

    Parameters
    ----------
    D : int
        The dimension of the space.
    lower_bounds : np.ndarray, optional
        The lower bounds of the space. ``lower_bounds`` and ``upper_bounds``
        define a set of strict lower and upper bounds for each variable,
        given in the original space. By default `None`, which is ``-inf``.
    upper_bounds : np.ndarray, optional
        The upper bounds of the space. ``lower_bounds`` and ``upper_bounds``
        define a set of strict lower and upper bounds for each variable,
        given in the original space. By default `None`, which is ``inf``.
    plausible_lower_bounds : np.ndarray, optional
        The plausible lower bounds such that ``lower_bounds <=
        plausible_lower_bounds < plausible_upper_bounds <= upper_bounds``.
        ``plausible_lower_bounds`` and ``plausible_upper_bounds`` represent a
        "plausible" range for each variable, given in the original space,
        and need to be finite. By default `None`, which is
        ``lower_bounds``.
    plausible_upper_bounds : np.ndarray, optional
        The plausible upper bounds such that ``lower_bounds <=
        plausible_lower_bounds < plausible_upper_bounds <= upper_bounds``.
        ``plausible_lower_bounds`` and ``plausible_upper_bounds`` represent a
        "plausible" range for each variable, given in the original space,
        and need to be finite. By default `None`, which is
        ``upper_bounds``.
    apply_log_t : np.ndarray, optional
        A boolean array of shape ``(1, D)`` that indicates the variables to
        which the non-linear log transformation applies; a scalar applies to
        every variable. By default `None`, in which case the log
        transformation is applied if the bounds are all positive and the
        plausible box spans at least one order of magnitude
        (``pub/plb >= 10``), as it is to a variable whose entry is NaN.
    fixed_values : np.ndarray, optional
        The values of the fixed variables of a problem, when the transform
        is of its other variables: one value per variable of the original
        space, of shape ``(1, D_orig)`` or ``(D_orig,)``, NaN at each of the
        ``D`` variables that the bounds describe, in their order, and the
        value elsewhere. ``inverse_transf`` then returns points of the
        ``D_orig`` variables, with the fixed ones at their values, and
        ``__call__`` takes such points and transforms their other
        coordinates. By default `None`, which, as a row of NaN, fixes no
        variable: the original space has the ``D`` variables of the bounds.

    Each bound is an array of ``D`` elements, of shape ``(1, D)`` or
    ``(D,)``, or a scalar or an array of one element, which stands for the
    same bound in each dimension; bounds of integers are taken as floats.

    Attributes
    ----------
    D_orig : int
        The number of variables of the original space: ``D`` and the fixed
        variables.
    fixed_values : np.ndarray or None
        ``fixed_values`` as a row of shape ``(1, D_orig)``, or None when it
        fixes no variable.

    Raises
    ------
    ValueError
        When a bound is not a number or an array of numbers, or is neither a
        scalar nor an array of one or ``D`` elements, when the plausible
        bounds are not finite or the bounds are out of the order above, or
        when the transform cannot be inverted at the bounds.
    ValueError
        When ``fixed_values`` is not a row of numbers, NaN at exactly ``D``
        of them and finite at the others.
    """

    def __init__(
        self,
        D,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        apply_log_t=None,
        fixed_values: np.ndarray = None,
    ):
        # Empty lb and ub are Infs
        if lower_bounds is None:
            lower_bounds = -np.inf
        if upper_bounds is None:
            upper_bounds = np.inf

        # Empty plausible bounds equal hard bounds
        if plausible_lower_bounds is None:
            plausible_lower_bounds = lower_bounds
        if plausible_upper_bounds is None:
            plausible_upper_bounds = upper_bounds

        # Float copies of shape (1, D), into which the log of a log-scaled
        # variable's bounds is written in place, so that integer bounds are
        # not truncated
        lb, ub, plb, pub = (
            _bound_as_row(bound, name, D)
            for bound, name in (
                (lower_bounds, "lower_bounds"),
                (upper_bounds, "upper_bounds"),
                (plausible_lower_bounds, "plausible_lower_bounds"),
                (plausible_upper_bounds, "plausible_upper_bounds"),
            )
        )

        # Save original vectors
        self.orig_ub = ub.copy()
        self.orig_lb = lb.copy()
        self.orig_plb = plb.copy()
        self.orig_pub = pub.copy()

        self.ub = ub
        self.lb = lb
        self.plb = plb
        self.pub = pub

        self.D = D
        # The fixed variables, which the transform leaves out of its points
        # and inverse_transf puts back at their values; self._free marks the
        # variables of the bounds among the D_orig of the original space. A
        # row of NaN, which fixes nothing, is None
        self.D_orig = D
        self.fixed_values = None
        self._free = None
        if fixed_values is not None:
            try:
                values = np.array(fixed_values, dtype=float, ndmin=2)
            except (TypeError, ValueError):
                values = None
            if (
                values is None
                or values.ndim != 2
                or values.shape[0] != 1
                or np.sum(np.isnan(values)) != D
                or np.any(np.isinf(values))
            ):
                raise ValueError(
                    "fixed_values needs to be a row of one number per "
                    f"variable of the original space, NaN at exactly D={D} "
                    "of them and finite at the others, not "
                    f"{fixed_values!r}."
                )
            if not np.all(np.isnan(values)):
                self.fixed_values = values
                self._free = np.isnan(values[0])
                self.D_orig = values.shape[1]
        # Nonlinear log transform: NaN marks a variable whose transform is
        # decided from its bounds, and a scalar applies to every variable
        if apply_log_t is None:
            apply_log_t = np.nan
        self.apply_log_t = np.array(apply_log_t, ndmin=2)
        if self.apply_log_t.size == 1:
            self.apply_log_t = np.full((1, self.D), self.apply_log_t.item())

        (
            self.lb,
            self.ub,
            self.plb,
            self.pub,
            self.g,
            self.ginv,
            self.z,
            self.zlog,
        ) = self.__create_hypercube_trans__()

    def __create_hypercube_trans__(self):
        """
        Standardize variables via linear or nonlinear transformation.
        The standardized transform maps ``plausible_lower_bounds`` (``plb``) and plausible_upper_bounds (``pub``) to the hypercube [-1,1]^D.
        If plb and/or pub are empty, ``lower_bound``(``lb``) and/or ``upper_bound``(``ub``) are used instead. Note that
        at least one among ``lb``, ``plb`` and one among ``ub``, ``pub`` needs to be nonempty.

        Parameters
        ----------

        D : scalar
            dimension
        lower_bound: np.ndarray
        upper_bound: np.ndarray
        plausible_lower_bounds: np.ndarray
        plausible_upper_bounds: np.ndarray


        """
        # Check finiteness of plausible range
        if not (np.all(np.isfinite(np.concatenate([self.plb, self.pub])))):
            raise ValueError(
                "Plausible interval ranges plausible_lower_bounds and plausible_upper_bounds need to be finite."
            )

        # Check that the order of bounds is respected
        if not (
            np.all(self.lb <= self.plb)
            and np.all(self.plb < self.pub)
            and np.all(self.pub <= self.ub)
        ):
            raise ValueError(
                "Interval bounds needs to respect the order lower_bound <= plausible_lower_bounds < plausible_upper_bounds <= upper_bound for all coordinates."
            )

        # A variable is converted to log scale if all bounds are positive and
        # the plausible range spans at least one order of magnitude
        check_idx_log_t = np.argwhere(np.isnan(self.apply_log_t.flatten()))
        for i in check_idx_log_t:
            self.apply_log_t[:, i] = (
                np.all(
                    np.concatenate(
                        [
                            self.lb[:, i],
                            self.ub[:, i],
                            self.plb[:, i],
                            self.pub[:, i],
                        ]
                    )
                    > 0
                )
                and (self.pub[:, i] / self.plb[:, i] >= 10).item()
            )
        self.apply_log_t = self.apply_log_t.astype(bool)

        self.lb[self.apply_log_t] = np.log(self.lb[self.apply_log_t])
        self.ub[self.apply_log_t] = np.log(self.ub[self.apply_log_t])
        self.plb[self.apply_log_t] = np.log(self.plb[self.apply_log_t])
        self.pub[self.apply_log_t] = np.log(self.pub[self.apply_log_t])

        mu = 0.5 * (self.plb + self.pub)
        gamma = 0.5 * (self.pub - self.plb)

        z = lambda x: maskindex((x - mu) / gamma, ~self.apply_log_t)
        zlog = lambda x: maskindex(
            (np.log(np.abs(x) + (x == 0)) - mu) / gamma, self.apply_log_t
        )

        apply_log_t_sum = np.sum(self.apply_log_t)
        if apply_log_t_sum == 0:
            g = lambda x: z(x)
            ginv = lambda y: gamma * y + mu

        elif apply_log_t_sum == self.D:
            g = lambda x: zlog(x)
            ginv = lambda y: np.minimum(
                np.finfo(np.float64).max, np.exp(gamma * y + mu)
            )
        else:
            g = lambda x: z(x) + zlog(x)

            def ginv(y):
                # The exponential of a linear variable, masked out, overflows
                # at a value above about 709, harmlessly
                with np.errstate(over="ignore"):
                    x_log = np.minimum(
                        np.finfo(np.float64).max, np.exp(gamma * y + mu)
                    )
                return maskindex(
                    gamma * y + mu, ~self.apply_log_t
                ) + maskindex(x_log, self.apply_log_t)

        # check that the transform works correctly in the range
        lbtest = self.orig_lb.copy()
        eps = np.spacing(1.0)
        lbtest[~np.isfinite(self.orig_lb)] = -1 / np.sqrt(eps)

        ubtest = self.orig_ub.copy()
        ubtest[~np.isfinite(self.orig_ub)] = 1 / np.sqrt(eps)
        ubtest[
            np.logical_and((~np.isfinite(self.orig_ub)), self.apply_log_t)
        ] = 1e6

        # accepted numerical error, relative to a bound larger than 1 in
        # magnitude (MATLAB BADS's transvars.m takes 1e-6 in absolute terms,
        # which rounding alone exceeds at bounds of large magnitude)
        numeps = 1e-6
        tests = np.zeros(4)
        for i, b in enumerate([lbtest, ubtest, self.orig_plb, self.orig_pub]):
            tol = numeps * np.maximum(1.0, np.abs(b))
            tests[i] = np.all(np.abs(ginv(g(b)) - b) < tol)
        if not np.all(tests):
            raise ValueError(
                "Cannot invert the transform to obtain the identity at the provided boundaries."
            )

        return (
            g(self.orig_lb),
            g(self.orig_ub),
            g(self.orig_plb),
            g(self.orig_pub),
            g,
            ginv,
            z,
            zlog,
        )

    def __call__(self, input: np.ndarray):
        """
        Performs direct transform of original variables ``input`` into
        the hypercube space.

        Parameters
        ----------
        input : np.ndarray
            A N x D array, where N is the number of input data
            and D is the number of dimensions; N x D_orig with
            ``fixed_values``, whose fixed coordinates are left out.

        Returns
        -------
        u : np.ndarray
            The variables transformed.
        """
        if self.fixed_values is not None:
            input = np.asarray(input)[..., self._free]
        y = self.g(input)
        y = np.minimum(
            np.maximum(y, self.lb), self.ub
        )  # Force to stay within bounds
        return y

    def inverse_transf(self, input: np.ndarray):
        """
        Performs inverse transform of the transformed variables  ``input`` in the hypercube into
        the original space.

        Parameters
        ----------
        input : np.ndarray
            The transformed variables that will be mapped in the original space.

        Returns
        -------
        x : np.ndarray
            The original variables retrieved by the inverse transform; with
            ``fixed_values``, all ``D_orig`` of them, the fixed ones at
            their values.
        """
        x = self.ginv(input)
        x = np.minimum(
            np.maximum(x, self.orig_lb), self.orig_ub
        )  # Force to stay within bounds
        x = x.reshape(input.shape)
        if self.fixed_values is not None:
            full = np.empty(x.shape[:-1] + (self._free.size,))
            full[...] = self.fixed_values[0]
            full[..., self._free] = x
            x = full

        return x


def _bound_as_row(bound, name, D):
    """A float copy of ``bound`` of shape ``(1, D)``: an array of ``D``
    elements, of shape ``(1, D)`` or ``(D,)``, or a scalar or an array of one
    element, replicated in each dimension. A bound that is not a number, or
    an array of numbers, is refused, a string included."""
    try:
        array = np.asarray(bound)
        if array.dtype.kind in "USV":
            raise TypeError("a string or bytes")
        row = np.array(array, dtype=float, ndmin=2)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"{name} needs to be a number or an array of numbers, not "
            f"{bound!r}."
        ) from err
    if row.size == 1:
        row = np.full((1, D), row.item())
    if row.shape != (1, D):
        raise ValueError(
            f"{name} needs to be a scalar or an array of D={D} elements, of "
            f"shape (1, D) or (D,), not an array of shape "
            f"{np.shape(bound)}."
        )
    return row


def maskindex(vector, bool_index):
    """
    Mask non-indexed elements in vector
    """
    result = vector.copy()
    result[:, ~bool_index.flatten()] = 0
    return result
