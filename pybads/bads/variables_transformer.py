import numpy as np
from pybads.decorators import handle_0D_1D_input

class VariableTransformer:
    """
    A class enabling linear or non-linear transformation of the bounds (PLB and PUB) and map them to an hypercube [-1, 1]^D
    """

    def __init__(self, D,
        lower_bounds : np.ndarray = None, upper_bounds : np.ndarray = None,
        plausible_lower_bounds : np.ndarray = None,
        plausible_upper_bounds : np.ndarray = None,
        apply_log_t = None):

        if (lower_bounds is None and plausible_lower_bounds is None) \
            or (upper_bounds is None and plausible_upper_bounds is None):
            raise ValueError("""hypercube_trans: At least one among LB, \
                         PLB and one among UB, PUB needs to be nonempty.""")

        lb =  lower_bounds.copy() if lower_bounds is not None else np.ones((1, D)) * -np.inf
        ub =  upper_bounds.copy() if upper_bounds is not None else np.ones((1, D)) * np.inf

        plb = lower_bounds.copy() if (plausible_lower_bounds is None) else plausible_lower_bounds.copy()
        pub = upper_bounds.copy() if (plausible_upper_bounds is None) else plausible_upper_bounds.copy()


        if np.isscalar(lb): lb = lb * np.ones((1, D))
        if np.isscalar(ub): ub = ub * np.ones((1, D))
        if np.isscalar(plb): plb = plb * np.ones((1, D))
        if np.isscalar(pub): pub = pub * np.ones((1, D))

        # Save original vectors
        self.orig_ub = ub.copy()
        self.orig_lb = lb.copy()
        self.orig_plb = plb.copy()
        self.orig_pub = pub.copy()

        self.ub = ub
        self.lb = lb
        self.plb = plb
        self.pub = pub

        # Nonlinear log transform
        if apply_log_t is None:
            self.apply_log_t = np.full((1, self.D), np.NaN).astype(bool)
        elif np.isscalar(apply_log_t):
            self.apply_log_t = (self.apply_log_t * np.ones((1, self.D))).astype(bool)
        else:
            self.apply_log_t = apply_log_t.copy()

        self.lb, self.ub, self.plb, self.pub, self.g, self.z, self.zlog = self.__create_hypercube_trans__()
        
    def __create_hypercube_trans__(self):
        """
        Standardize variables via linear or nonlinear transformation.
        The standardized transform maps PLB and PUB to the hypercube [-1,1]^D.
        If PLB and/or PUB are empty, LB and/or UB are used instead. Note that
        at least one among LB, PLB and one among UB, PUB needs to be nonempty.

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
        assert np.all(np.isfinite(np.concatenate([self.plb, self.pub]))), \
            'Plausible interval ranges PLB and PUB need to be finite.'

        # Check that the order of bounds is respected
        assert np.all(self.lb <= self.plb) and np.all(self.plb < self.pub) and np.all(self.pub <= self.ub), \
            'Interval bounds needs to respect the order LB <= PLB < PUB <= UB for all coordinates.'


        # A variable is converted to log scale if all bounds are positive and 
        # the plausible range spans at least one order of magnitude
        for i in np.argwhere(np.isnan(self.apply_log_t)):
            self.apply_log_t[:, i] = np.all(np.concatenate([self.ub[:, i], self.ub[:, i], self.plb[:, i], self.pub[:, i]]) > 0) \
                & (self.pub[:, i]/self.plb[:, i] >= 10)       
        self.apply_log_t = self.apply_log_t.astype(bool)

        self.lb[self.apply_log_t] = np.log(self.lb[self.apply_log_t])
        self.ub[self.apply_log_t] = np.log(self.ub[self.apply_log_t])
        self.plb[self.apply_log_t] = np.log(self.plb[self.apply_log_t])
        self.pub[self.apply_log_t] = np.log(self.pub[self.apply_log_t])

        mu = 0.5 * (self.plb + self.pub)
        gamma = 0.5 * (self.pub - self.plb)

        z = lambda x: maskindex((x - mu)/gamma, ~ self.apply_log_t)
        zlog = lambda x: maskindex((np.log(np.abs(x) + (x == 0)) - mu)/gamma, self.apply_log_t)

        apply_log_t_sum = np.sum(self.apply_log_t)
        if apply_log_t_sum == 0:
            g = lambda x: z(x)
            ginv = lambda y: gamma * y + mu

        elif apply_log_t_sum == self.D:
            g = lambda x: zlog(x)
            ginv = lambda y: min(np.finfo(np.float64).max, np.exp(gamma * y + mu) )
        else:
            g = lambda x: z(x) + zlog(x)
            ginv = lambda y: maskindex(gamma * y + mu, ~self.apply_log_t) \
                        + maskindex(min(np.finfo(np.float64).max, np.exp(gamma * y + mu)), self.apply_log_t)

        #check that the transform works correctly in the range
        lbtest = self.lb.copy()
        eps = np.spacing(1.0)
        lbtest[~np.isfinite(self.lb)] = -1/np.sqrt(eps)
        
        ubtest = self.ub.copy()
        ubtest[~np.isfinite(self.ub)] = 1/np.sqrt(eps)
        ubtest[np.logical_and((~np.isfinite(self.ub)), self.apply_log_t)] = 1e6

        numeps = 1e-6 #accepted numerical error
        tests = np.zeros(4)
        tests[0] = np.all(abs(ginv(g(lbtest)) - lbtest) < numeps)
        tests[1] = np.all(abs(ginv(g(ubtest)) - ubtest) < numeps)
        tests[2] = np.all(abs(ginv(g(self.plb)) - self.plb) < numeps)
        tests[3] = np.all(abs(ginv(g(self.pub)) - self.pub) < numeps)
        assert np.all(tests), 'Cannot invert the transform to obtain the identity at the provided boundaries.'

        return (g(self.lb), g(self.ub), g(self.plb), g(self.pub), g, z, zlog)

    def __call__(self, input: np.ndarray):
        y = self.g(input)
        y = np.minimum(np.maximum(y, self.lb), self.ub) #Force to stay within bounds
        return y

    def inverse_transf(self, input: np.ndarray) :
        x = self.ginv(input)
        x = np.min(np.max(x, self.oldlb), self.oldub) # Force to stay within bounds
        x = x.reshape(input.shape)

        return x

    @handle_0D_1D_input(patched_kwargs=["u"], patched_argpos=[0], return_scalar=True)
    def log_abs_det_jacobian(self, u: np.ndarray):
        r"""
        log_abs_det_jacobian returns the log absolute value of the determinant
        of the Jacobian of the parameter transformation evaluated at U, that is
        log \|D \du(g^-1(u))\|

        Parameters
        ----------
        u : np.ndarray
            The points where the log determinant of the Jacobian should be
            evaluated (in transformed space).

        Returns
        -------
        p : np.ndarray
            The log absolute determinant of the Jacobian.
        """
        u_c = np.copy(u)

        # # rotate input (copy array before)
        # if self.R_mat is not None:
        #     u_c = u_c * self.R_mat
        # # rescale input
        # if scale is not None:
        #     print(scale)

        p = np.zeros(u_c.shape)

        # Unbounded scalars
        mask = self.type == 0
        if np.any(mask):
            p[:, mask] = np.log(self.delta[mask])[np.newaxis]

        # Lower and upper bounded scalars
        mask = self.type == 3
        if np.any(mask):
            u_c[:, mask] = u_c[:, mask] * self.delta[mask] + self.mu[mask]
            z = -np.log1p(np.exp(-u_c[:, mask]))
            p[:, mask] = (
                np.log(self.ub_orig - self.lb_orig) - u_c[:, mask] + 2 * z
            )
            p[:, mask] = p[:, mask] + np.log(self.delta[mask])

        # Scale transform
        # if scale is not None:
        #     p + np.log(scale)
        p = np.sum(p, axis=1)
        return p
    
def maskindex (vector, bool_index):
    """
        Mask non-indexed elements in vector
    """
    result = vector.copy()
    result[~bool_index] = 0
    return result