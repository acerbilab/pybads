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
        The lower bounds of the space. ``lower_bounds`` and ``upper_bounds`` define a set
        of strict lower and upper bounds for each variable, given in the
        original space. By default `None`.
    upper_bounds : np.ndarray, optional
        The upper bounds of the space. ``lower_bounds`` and ``upper_bounds`` define a set
        of strict lower and upper bounds for each variable, given in the
        original space. By default `None`.
    plausible_lower_bounds : np.ndarray, optional
        The plausible lower bounds such that ``lower_bounds < plausible_lower_bounds < plausible_upper_bounds <
        upper_bounds``. ``plausible_lower_bounds`` and ``plausible_upper_bounds`` represent a "plausible" range
        for each variable, given in the original space. By default `None`.
    plausible_upper_bounds : np.ndarray, optional
        The plausible upper bounds such that ``lower_bounds < plausible_lower_bounds < plausible_upper_bounds <
        upper_bounds``. ``plausible_lower_bounds`` and ``plausible_upper_bounds`` represent a "plausible" range
        for each variable, given in the original space. By default `None`.
    apply_log_t : np.ndarray, optional
        A boolean array of size (1, D) that indicates which variables to apply the non-linear log transformation. 
        By default `None`, in which case the log transformation is applied if the bounds are all positive and the variables span more than one order of magnitude.
    """
    def __init__(
        self,
        D,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        apply_log_t=None,
    ):
        # Empty lb and ub are Infs
        if lower_bounds is None:
            lower_bounds = np.ones((1, D)) * -np.inf
        if upper_bounds is None:
            upper_bounds = np.ones((1, D)) * np.inf

        # Empty plausible bounds equal hard bounds
        if plausible_lower_bounds is None:
            plausible_lower_bounds = np.copy(lower_bounds)
        if plausible_upper_bounds is None:
            plausible_upper_bounds = np.copy(upper_bounds)

        lb = (
            lower_bounds.copy()
            if lower_bounds is not None
            else np.ones((1, D)) * -np.inf
        )
        ub = (
            upper_bounds.copy()
            if upper_bounds is not None
            else np.ones((1, D)) * np.inf
        )

        plb = (
            lower_bounds.copy()
            if (plausible_lower_bounds is None)
            else plausible_lower_bounds.copy()
        )
        pub = (
            upper_bounds.copy()
            if (plausible_upper_bounds is None)
            else plausible_upper_bounds.copy()
        )

        if np.isscalar(lb):
            lb = lb * np.ones((1, D))
        if np.isscalar(ub):
            ub = ub * np.ones((1, D))
        if np.isscalar(plb):
            plb = plb * np.ones((1, D))
        if np.isscalar(pub):
            pub = pub * np.ones((1, D))

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
        # Nonlinear log transform
        if apply_log_t is None:
            self.apply_log_t = np.full((1, self.D), np.NaN)
        elif np.isscalar(apply_log_t):
            self.apply_log_t = (
                self.apply_log_t * np.ones((1, self.D))
            ).astype(bool)
        else:
            self.apply_log_t = apply_log_t.copy()

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
            raise ValueError("Plausible interval ranges plausible_lower_bounds and plausible_upper_bounds need to be finite.")

        # Check that the order of bounds is respected
        if  not (np.all(self.lb <= self.plb) \
            and np.all(self.plb < self.pub)\
            and np.all(self.pub <= self.ub)):
                raise ValueError("Interval bounds needs to respect the order lower_bound <= plausible_lower_bounds < plausible_upper_bounds <= upper_bound for all coordinates.")
         

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
            ginv = lambda y: maskindex(
                gamma * y + mu, ~self.apply_log_t
            ) + maskindex(
                np.minimum(np.finfo(np.float64).max, np.exp(gamma * y + mu)),
                self.apply_log_t,
            )

        # check that the transform works correctly in the range
        lbtest = self.orig_lb.copy()
        eps = np.spacing(1.0)
        lbtest[~np.isfinite(self.orig_lb)] = -1 / np.sqrt(eps)

        ubtest = self.orig_ub.copy()
        ubtest[~np.isfinite(self.orig_ub)] = 1 / np.sqrt(eps)
        ubtest[
            np.logical_and((~np.isfinite(self.orig_ub)), self.apply_log_t)
        ] = 1e6

        numeps = 1e-6  # accepted numerical error
        tests = np.zeros(4)
        tests[0] = np.all(np.abs(ginv(g(lbtest)) - lbtest) < numeps)
        tests[1] = np.all(np.abs(ginv(g(ubtest)) - ubtest) < numeps)
        tests[2] = np.all(np.abs(ginv(g(self.orig_plb)) - self.orig_plb) < numeps)
        tests[3] = np.all(np.abs(ginv(g(self.orig_pub)) - self.orig_pub) < numeps)
        if not np.all(tests):
            raise ValueError("Cannot invert the transform to obtain the identity at the provided boundaries.")

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
            and D is the number of dimensions

        Returns
        -------
        u : np.ndarray
            The variables transformed.
        """
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
            The original variables retrieved by the inverse transform.
        """
        x = self.ginv(input)
        x = np.minimum(
            np.maximum(x, self.orig_lb), self.orig_ub
        )  # Force to stay within bounds
        x = x.reshape(input.shape)

        return x


def maskindex(vector, bool_index):
    """
    Mask non-indexed elements in vector
    """
    result = vector.copy()
    result[:, ~bool_index.flatten()] = 0
    return result
