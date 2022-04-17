import copy
import logging
import math
import os
import sys

import gpyreg as gpr
import matplotlib.pyplot as plt
import numpy as np
from requests import options
from scipy.special import gammaincinv
from scipy.special import erfcinv
from scipy.stats import shapiro
from sqlalchemy import true
from pybads import function_logger
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb

from pybads.search.search_hedge import SearchESHedge

from pybads.function_logger import FunctionLogger
from pybads.init_functions.init_sobol import init_sobol
from pybads.search.grid_functions import force_to_grid, grid_units, get_grid_search_neighbors, udist 
from pybads.utils.period_check import period_check
from pybads.utils.timer import Timer
from pybads.utils.iteration_history import IterationHistory
from pybads.bads.variables_transformer import VariableTransformer
from pybads.utils.constraints_check import contraints_check
from pybads.bads.gaussian_process_train import local_gp_fitting, reupdate_gp, train_gp
from gpyreg.gaussian_process import GP
from pybads.bads.options import Options


class BADS:
    """
    BADS Constrained optimization using Bayesian Adaptive Direct Search (v1.0.6)
    BADS attempts to solve problems of the form:
       min F(X)  subject to:  LB <= X <= UB
        X                        C(X) <= 0        (optional)

    Initialize a ``BADS`` object to set up the optimization problem, then run
    ``optimize()``. See the examples for more details.

    Parameters
    ----------
    fun : callable
        A given target log posterior `fun`. `fun` accepts input `x` and returns 
        the value of the target log-joint, that is the unnormalized 
        log-posterior density, at `x`.
    x0 : np.ndarray, optional
        Starting point.
    lower_bounds, upper_bounds : np.ndarray, optional
        `lower_bounds` (`LB`) and `upper_bounds` (`UB`) define a set
        of strict lower and upper bounds for the coordinate vector, `x`, so 
        that the posterior has support on `LB` < `x` < `UB`.
        If scalars, the bound is replicated in each dimension. Use
        ``None`` for `LB` and `UB` if no bounds exist. Set `LB` [`i`] = -``inf``
        and `UB` [`i`] = ``inf`` if the `i`-th coordinate is unbounded (while 
        other coordinates may be bounded). Note that if `LB` and `UB` contain
        unbounded variables, the respective values of `PLB` and `PUB` need to 
        be specified (see below), by default ``None``.
    plausible_lower_bounds, plausible_upper_bounds : np.ndarray, optional
        Specifies a set of `plausible_lower_bounds` (`PLB`) and
        `plausible_upper_bounds` (`PUB`) such that `LB` < `PLB` < `PUB` < `UB`.
        Both `PLB` and `PUB` need to be finite. `PLB` and `PUB` represent a
        "plausible" range, which should denote a region of the global minimum.

    nonbondcons: callable
        A given nNon-bound constraints function. e.g : lambda x: np.sum(x.^2, 1) > 1

    user_options : dict, optional
        Additional options can be passed as a dict. Please refer to the
        BDAS options page for the default options. If no `user_options` are 
        passed, the default options are used.

    Raises
    ------
    ValueError
        When neither `x0` or (`plausible_lower_bounds` and
        `plausible_upper_bounds`) are specified.
    ValueError
        When various checks for the bounds (LB, UB, PLB, PUB) of BADS fail.


    References
    ----------
    .. [1]  Acerbi, L. & Ma, W. J. (2017). "Practical Bayesian 
            Optimization for Model Fitting with Bayesian Adaptive Direct Search". 
            In Advances in Neural Information Processing Systems 30, pages 1834-1844.
            (arXiv preprint: https://arxiv.org/abs/1705.04405).

    Examples
    --------
    """

    def __init__(
        self,
        fun: callable,
        x0: np.ndarray = None,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        user_options: dict = None,
        nonbondcons: callable = None
    ):
        # set up root logger (only changes stuff if not initialized yet)
        logging.basicConfig(stream=sys.stdout, format="%(message)s")

        self.nonbondcons = nonbondcons

        # variable to keep track of logging actions
        self.logging_action = []
        
        # Initialize variables and algorithm structures
        if x0 is None:
            if (plausible_lower_bounds is None
                or plausible_upper_bounds is None):
                raise ValueError(
                    """bads:UnknownDims If no starting point is
                 provided, PLB and PUB need to be specified."""
                )
            else:
                x0 = np.full((plausible_lower_bounds.shape), np.NaN)

        self.D = x0.shape[1]

        # load basic and advanced options and validate the names
        pybads_path = os.path.dirname(os.path.realpath(__file__))
        basic_path = pybads_path + "/option_configs/basic_bads_options.ini"
        self.options = Options(
            basic_path,
            evaluation_parameters={"D": self.D},
            user_options=user_options,
        )
        advanced_path = (
            pybads_path + "/option_configs/advanced_bads_options.ini"
        )
        self.options.load_options_file(
            advanced_path,
            evaluation_parameters={"D": self.D},
        )
        self.options.validate_option_names([basic_path, advanced_path])

        # set up BADS logger
        self.logger = logging.getLogger("BADS")
        self.logger.setLevel(logging.INFO)
        if self.options.get("display") == "off":
            self.logger.setLevel(logging.WARN)
        elif self.options.get("display") == "iter":
            self.logger.setLevel(logging.INFO)
        elif self.options.get("display") == "full":
            self.logger.setLevel(logging.DEBUG)


        # Empty LB and UB are Infs
        if lower_bounds is None:
            lower_bounds = np.ones((1, self.D)) * -np.inf

        if upper_bounds is None:
            upper_bounds = np.ones((1, self.D)) * np.inf

        # Check/fix boundaries and starting points
        (
            self.x0,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,

        ) = self._boundscheck_(
            x0.copy(),
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
            nonbondcons
        )

        # starting point
        if not np.all(np.isfinite(self.x0)):
            self.logger.warn('Initial starting point is invalid or not provided.\
                 Starting from center of plausible region.\n')
            self.x0 = 0.5 * (self.plausible_lower_bounds + self.plausible_upper_bounds)
        
        # evaluate  starting point non-bound constraint
        if nonbondcons is not None:
            if nonbondcons(self.x0) > 0:
                raise ValueError('Initial starting point X0 does not satisfy non-bound constraints NONBCON.')
            

        self.optim_state = self._init_optim_state()

        # create and init the function logger
        self.function_logger = FunctionLogger(
            fun=fun,
            D=self.D,
            noise_flag=self.optim_state.get("uncertainty_handling_level") > 0,
            uncertainty_handling_level=self.optim_state.get("uncertainty_handling_level"),
            cache_size=self.options.get("cachesize"),
            variable_transformer=self.var_transf)

        self.iteration_history = IterationHistory(
            [
                "iter",
                "u",
                "fval",
                "yval",
                "ys",
                "lcbmax",
                "rindex", # Reliability index (not used in BADS)
                "gp",
                "gp_last_reset_idx",
                "gp_hyp_full",
                "Ns_gp",
                "timer",
                "optim_state",
                "func_count",
                "n_eff",
                "logging_action",
            ]
        )

    def _boundscheck_(
        self,
        x0: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        nonbondcons: callable = None
    ):
        """
        Private function to do the initial check of the BADS bounds.
        """

        N0, D = x0.shape

        #Estimation of the PLB and PUB if any of them is not specified
        if plausible_lower_bounds is None or plausible_upper_bounds is None:
            if N0 > 1:
                self.logger.warning(
                    "PLB and/or PUB not specified. Estimating"
                    + "plausible bounds from starting set X0..."
                )
                width = x0.max(0) - x0.min(0)
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = x0.min(0) - width / N0
                    plausible_lower_bounds = np.maximum(
                        plausible_lower_bounds, lower_bounds
                    )
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = x0.max(0) + width / N0
                    plausible_upper_bounds = np.minimum(
                        plausible_upper_bounds, upper_bounds
                    )

                idx = plausible_lower_bounds == plausible_upper_bounds
                if np.any(idx):
                    plausible_lower_bounds[idx] = lower_bounds[idx]
                    plausible_upper_bounds[idx] = upper_bounds[idx]
                    self.logger.warning(
                        "bads:pbInitFailed: Some plausible bounds could not be "
                        + "determined from starting set. Using hard upper/lower"
                        + " bounds for those instead."
                    )
            else:
                self.logger.warning(
                    "bads:pbUnspecified: Plausible lower/upper bounds PLB and"
                    "/or PUB not specified and X0 is not a valid starting set. "
                    + "Using hard upper/lower bounds instead."
                )
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = np.copy(lower_bounds)
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = np.copy(upper_bounds)

        # check that all bounds are row vectors with D elements
        if (
            np.ndim(lower_bounds) != 2
            or np.ndim(upper_bounds) != 2
            or np.ndim(plausible_lower_bounds) != 2
            or np.ndim(plausible_upper_bounds) != 2
            or lower_bounds.shape != (1, D)
            or upper_bounds.shape != (1, D)
            or plausible_lower_bounds.shape != (1, D)
            or plausible_upper_bounds.shape != (1, D)
        ):
            raise ValueError(
                """All input vectors (x0, lower_bounds, upper_bounds,
                 plausible_lower_bounds, plausible_upper_bounds), if specified,
                 need to be row vectors with D elements."""
            )

        # check that plausible bounds are finite
        if np.any(np.invert(np.isfinite(plausible_lower_bounds))) or np.any(
            np.invert(np.isfinite(plausible_upper_bounds))
        ):
            raise ValueError(
                "Plausible interval bounds PLB and PUB need to be finite."
            )

        # Test that all vectors are real-valued
        if (
            np.any(np.invert(np.isreal(x0)))
            or np.any(np.invert(np.isreal(lower_bounds)))
            or np.any(np.invert(np.isreal(upper_bounds)))
            or np.any(np.invert(np.isreal(plausible_lower_bounds)))
            or np.any(np.invert(np.isreal(plausible_upper_bounds)))
        ):
            raise ValueError(
                """All input vectors (x0, lower_bounds, upper_bounds,
                 plausible_lower_bounds, plausible_upper_bounds), if specified,
                 need to be real valued."""
            )

        # Fixed variables (all bounds equal) are not supported
        fixidx = (
            (lower_bounds == upper_bounds)
            & (upper_bounds == plausible_lower_bounds)
            & (plausible_lower_bounds == plausible_upper_bounds)
        )
        if np.any(fixidx):
            raise ValueError(
                """bads:FixedVariables BADS does not support fixed 
            variables. Lower and upper bounds should be different."""
            )

        # Test that plausible bounds are different
        if np.any(plausible_lower_bounds == plausible_upper_bounds):
            raise ValueError(
                """bads:MatchingPB:For all variables,
            plausible lower and upper bounds need to be distinct."""
            )

        # Check that all X0 are inside the bounds
        if np.any(x0 < lower_bounds) or np.any(x0 > upper_bounds):
            raise ValueError(
                """bads:InitialPointsNotInsideBounds: The starting
            nonbondconspoints X0 are not inside the provided hard bounds LB and UB."""
            )

        # # Compute "effective" bounds (slightly inside provided hard bounds)
        bounds_range = upper_bounds - lower_bounds
        bounds_range[np.isinf(bounds_range)] = 1e3
        scale_factor = 1e-3
        realmin = sys.float_info.min
        LB_eff = lower_bounds + scale_factor * bounds_range
        LB_eff[np.abs(lower_bounds) <= realmin] = (
            scale_factor * bounds_range[np.abs(lower_bounds) <= realmin]
        )
        UB_eff = upper_bounds - scale_factor * bounds_range
        UB_eff[np.abs(upper_bounds) <= realmin] = (
            -scale_factor * bounds_range[np.abs(upper_bounds) <= realmin]
        )
        # Infinities stay the same
        LB_eff[np.isinf(lower_bounds)] = lower_bounds[np.isinf(lower_bounds)]
        UB_eff[np.isinf(upper_bounds)] = upper_bounds[np.isinf(upper_bounds)]

        if np.any(LB_eff >= UB_eff):
            raise ValueError(
                """bads:StrictBoundsTooClose: Hard bounds LB and UB
                are numerically too close. Make them more separate."""
            )

        # Fix when provided X0 are almost on the bounds -- move them inside
        if np.any(x0 < LB_eff) or np.any(x0 > UB_eff):
            self.logger.warning(
                "bads:InitialPointsTooClosePB: The starting points X0 are on "
                + "or numerically too close to the hard bounds LB and UB. "
                + "Moving the initial points more inside..."
            )
            x0 = np.maximum((np.minimum(x0, UB_eff)), LB_eff)

        # Test order of bounds (permissive)
        ordidx = (
            (lower_bounds <= plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds <= upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """bads:StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB < PLB < PUB < UB."""
            )

        # Test that plausible bounds are reasonably separated from hard bounds
        if np.any(LB_eff > plausible_lower_bounds) or np.any(
            plausible_upper_bounds > UB_eff
        ):
            self.logger.warning(
                "bads:TooCloseBounds: For each variable, hard "
                + "and plausible bounds should not be too close. "
                + "Moving plausible bounds."
            )
            plausible_lower_bounds = np.maximum(plausible_lower_bounds, LB_eff)
            plausible_upper_bounds = np.minimum(plausible_upper_bounds, UB_eff)

        # Check that all X0 are inside the plausible bounds,
        # move bounds otherwise
        if np.any(x0 <= LB_eff) or np.any(x0 >= UB_eff):
            self.logger.warning(
                "bads:InitialPointsOutsidePB. The starting points X0"
                + " are not inside the provided plausible bounds PLB and "
                + "PUB. Expanding the plausible bounds..."
            )
            plausible_lower_bounds = np.minimum(
                plausible_lower_bounds, x0.min(0)
            )
            plausible_upper_bounds = np.maximum(
                plausible_upper_bounds, x0.max(0)
            )

        # Test order of bounds
        ordidx = (
            (lower_bounds <= plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds <= upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """bads:StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB <= PLB < PUB <= UB."""
            )

        # Check that variables are either bounded or unbounded
        # (not half-bounded)
        if (
            np.any(np.isfinite(lower_bounds))
            and np.any(np.invert(np.isfinite(upper_bounds)))
            or np.any(np.invert(np.isfinite(lower_bounds)))
            and np.any(np.isfinite(upper_bounds))
        ):
            raise ValueError(
                """bads:HalfBounds: Each variable needs to be unbounded or
            bounded. Variables bounded only below/above are not supported."""
            )

        # Check non bound constraints
        if nonbondcons is not None:
            y = nonbondcons(np.array([plausible_lower_bounds, plausible_upper_bounds]))
            if y.shape[0] != 2 or y.shape[1] != 1:
                raise ValueError("bads:NONBCON "
                    + "NONBCON should be a function that takes a matrix X as input"
                    + " and returns a column vector of bound violations.")

        # Gentle warning for infinite bounds
        ninfs = np.sum(np.isinf(np.concatenate([lower_bounds, upper_bounds])))
        if ninfs > 0:
            if ninfs == 2 *D:
                self.logger.warning("Detected fully unconstrainer optimization.")
            else:
                self.logger.warning(f"Detected {ninfs} infinite bound(s).")

        return (
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

    def _init_optim_state(self):
        """
        A private function to init the optim_state dict that contains
        information about BADS variables.
        """
        # Record starting points (original coordinates)
        y_orig = np.array(self.options.get("fvals")).flatten()
        if len(y_orig) == 0:
            y_orig = np.full([self.x0.shape[0]], np.nan)
        if len(self.x0) != len(y_orig):
            raise ValueError(
                """bads:MismatchedStartingInputs The number of
            points in X0 and of their function values as specified in
            self.options.fvals are not the same."""
            )

        optim_state = dict()
        optim_state["cache"] = dict()
        optim_state["cache"]["x_orig"] = self.x0
        optim_state["cache"]["y_orig"] = y_orig

        # Does the starting cache contain function values?
        optim_state["cache_active"] = np.any(
            np.isfinite(optim_state.get("cache").get("y_orig"))
        )

        # Grid parameters
        self.mesh_size_integer = 0 # Mesh size in log base units
        optim_state["search_size_integer"] = np.minimum(0, self.mesh_size_integer * self.options.get("searchgridmultiplier") - self.options.get("searchgridnumber"))
        optim_state["mesh_size"] =  float(self.options.get("pollmeshmultiplier"))**self.mesh_size_integer
        optim_state["search_mesh_size"] = float(self.options.get("pollmeshmultiplier"))**optim_state["search_size_integer"]
        optim_state["scale"] = 1.
        

        # Compute transformation of variables
        if self.options['nonlinearscaling']:
            logflag = np.full((1, self.D), np.NaN)
            periodicvars = self.options['periodicvars']
            if periodicvars is not None and len(periodicvars) != 0:
                logflag[periodicvars] = 0
        else:
            logflag = np.zeros((1, self.D))

        self.var_transf = VariableTransformer(self.D, self.lower_bounds, self.upper_bounds,
            self.plausible_lower_bounds, self.plausible_upper_bounds, logflag)
        #optim_state["variables_trans"] = var_transf

        # Update the bounds with the new transformed bounds
        self.lower_bounds = self.var_transf.lb.copy()
        self.upper_bounds = self.var_transf.ub.copy()
        optim_state["lb"] = self.lower_bounds.copy()
        optim_state["ub"] = self.upper_bounds.copy()
        self.plausible_lower_bounds = self.var_transf.plb.copy()
        self.plausible_upper_bounds = self.var_transf.pub.copy()
        optim_state["pub"] = self.plausible_lower_bounds.copy()
        optim_state["plb"] = self.plausible_upper_bounds.copy()

        optim_state["lb_orig"] = self.var_transf.orig_lb.copy()
        optim_state["ub_orig"] = self.var_transf.orig_ub.copy()
        optim_state["plb_orig"] = self.var_transf.orig_plb.copy()
        optim_state["pub_orig"] = self.var_transf.orig_pub.copy()
        

        # Bounds for search mesh
        lb_search = force_to_grid(self.lower_bounds, optim_state["search_mesh_size"])
        lb_search[ lb_search < self.lower_bounds] = lb_search[ lb_search < self.lower_bounds] + optim_state["search_mesh_size"]
        optim_state["lb_search"] = lb_search
        ub_search = force_to_grid(self.upper_bounds, optim_state["search_mesh_size"])
        ub_search[ ub_search > self.upper_bounds] = ub_search[ub_search > self.upper_bounds] - optim_state["search_mesh_size"]
        optim_state["ub_search"] = ub_search
        
        # Starting point in grid coordinates
        u0 =  force_to_grid(grid_units(self.x0, self.var_transf, optim_state["scale"]), optim_state['search_mesh_size'])
        
        # Adjust points that fall outside bounds due to gridization
        u0[u0 < self.lower_bounds] = u0[u0 < self.lower_bounds] + optim_state["search_mesh_size"]
        u0[u0 > self.upper_bounds] = u0[u0 > self.upper_bounds] - optim_state["search_mesh_size"]
        optim_state['u'] = u0
        self.u = u0.copy()

        # Test starting point u0 is within bounds
        if np.any(u0 > self.upper_bounds) or np.any(u0 < self.lower_bounds):
            self.logger.error("Initial starting point u0 is not within the hard bounds LB and UB")
            raise ValueError(
                """bads:Initpoint: Initial starting point u0 is not within the hard bounds LB and UB""") 

        # Report variable transformation
        if np.any(self.var_transf.apply_log_t):
            self.logger.info(f"Variables (index) internally transformed to log coordinates: {np.argwhere(self.var_transf.apply_log_t)}") 

        # Put TOLMESH on space
        optim_state['tol_mesh'] = self.options['pollmeshmultiplier']**np.ceil(np.log(self.options['tolmesh']) / np.log(self.options['pollmeshmultiplier']))
                

        #Periodic variables
        idx_periodic_vars = self.options['periodicvars']
        periodic_vars = np.zeros((1, self.D)).astype(bool)
        if idx_periodic_vars is not None or len(idx_periodic_vars) != 0:
            periodic_vars[idx_periodic_vars] = True
            finite_periodicvars = np.all(np.isfinite(self.lower_bounds[idx_periodic_vars])) and \
                np.all(np.isfinite(self.upper_bounds[idx_periodic_vars]))
            if not finite_periodicvars:
                raise ValueError('bads:InitOptimState:Periodic variables need to have finite lower and upper bounds.')
            self.logger.info(f"Variables (index) defined with periodic boundaries: {idx_periodic_vars}")
        optim_state['periodicvars'] = periodic_vars

        # Setup covariance information (unused)

        # Import prior function evaluations
        fun_values = self.options['funvalues']
        if fun_values is not None and len(fun_values) != 0:
            if  'X' not in fun_values or 'Y' not in fun_values:
                raise ValueError("""bads:funvalues: The 'FunValue' field in OPTIONS need to have X and Y fields (respectively, inputs and their function values)""")

            X = fun_values['X']
            Y = fun_values['Y']
            if len(X) != len(Y):
                raise ValueError("X and Y arrays in the OPTIONS.FunValues need to have the same number of rows (each row is a tested point).")
            if (not np.all(np.isfinite(X))) or (not np.all(np.isfinite(Y))) or (not np.isreal(X)) or (not np.isreal(Y)):
                raise ValueError('X and Y arrays need to be finite and real-valued')
            if len(X) != 0 and X.shape[1] != self.D:
                raise ValueError('X should be a matrix of tested points with the same dimensionality as X0 (one input point per row).')
            Y = np.atleast_2d(Y).T
            if len(Y) != 0 and Y.shape[1] != 1:
                raise ValueError('Y should be a vertical nd-array (, 1) of function values (one function value per row).')
            optim_state['X'] = X
            optim_state['Y'] = Y
        
        if 'S' in fun_values:
            S = fun_values['S']
            if len(S) != len(Y):
                raise ValueError('X, Y, and S arrays in the OPTIONS.FunValues need to have the same number of rows (each row is a tested point).')
            
            S = np.atleast_2d(S).T
            if len(S) != 0 and S.shape[1] != 1:
                raise ValueError('S should be a vertical nd-array (, 1) of estimated function SD values (one SD per row).')

            optim_state['S'] = S
            
        #Other variables initializations
        optim_state['search_factor']   =   1
        optim_state['sd_level']        = self.options['incumbentsigmamultiplier']
        optim_state['search_count']    = self.options['searchntry']       # Skip search at first iteration
        optim_state['lastreeval']     = -np.inf;                     # Last time function values were re-evaluated
        optim_state['lastfitgp']      = -np.inf;                     # Last fcn evaluation for which the gp was trained
        self.mesh_overflows  = 0;                       # Number of attempted mesh expansions when already at maximum size

        # List of points at the end of each iteration
        optim_state["iterlist"] = dict()
        optim_state["iterlist"]["u"] = []
        optim_state["iterlist"]["fval"] = []
        optim_state["iterlist"]["fsd"] = []
        optim_state["iterlist"]["fhyp"] = []

        
        #optim_state['es'] = es_update(es_mu, es_lambda)

        # Hedge struct
        optim_state['search_hedge'] = dict()


        # Before first iteration
        # Iterations are from 0 onwards in optimize so we should have -1
        optim_state["iter"] = -1

        # When GP hyperparameter sampling is switched with optimization
        if self.options.get("nsgpmax") > 0:
            optim_state["stop_sampling"] = 0
        else:
            optim_state["stop_sampling"] = np.Inf

        # Start with warm-up?
        optim_state["warmup"] = self.options.get("warmup")
        if self.options.get("warmup"):
            optim_state["last_warmup"] = np.inf
        else:
            optim_state["last_warmup"] = 0
        
        optim_state["vpK"] = self.options.get("kwarmup")

        # Tolerance threshold on GP variance (used by some acquisition fcns)
        optim_state["tol_gp_var"] = self.options.get("tolgpvar")

        # Copy maximum number of fcn. evaluations,
        # used by some acquisition fcns.
        optim_state["max_fun_evals"] = self.options.get("maxfunevals")

        # Set uncertainty handling level
        # (0: none; 1: unknown noise level; 2: user-provided noise)
        if self.options.get("specifytargetnoise"):
            optim_state["uncertainty_handling_level"] = 2
        elif len(self.options.get("uncertaintyhandling")) > 0:
            optim_state["uncertainty_handling_level"] = 1
        else:
            optim_state["uncertainty_handling_level"] = 0

        # Empty hedge struct for acquisition functions
        if self.options.get("acqhedge"):
            optim_state["acq_hedge"] = dict()

        # List of points at the end of each iteration
        optim_state["iterlist"] = dict()
        optim_state["iterlist"]["u"] = []
        optim_state["iterlist"]["fval"] = []
        optim_state["iterlist"]["fsd"] = []
        optim_state["iterlist"]["fhyp"] = []

        # Initialize Gaussian process settings
        # Squared exponential kernel with separate length scales
        optim_state["gp_covfun"] = 1

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
            self.options.get("noiseshaping")
            and optim_state["gp_noisefun"][1] == 0
        ):
            optim_state["gp_noisefun"][1] = 1

        optim_state["gp_meanfun"] = self.options.get("gpmeanfun")
        valid_gpmeanfuns = [
            "zero",
            "const",
            "negquad",
            "se",
            "negquadse",
            "negquadfixiso",
            "negquadfix",
            "negquadsefix",
            "negquadonly",
            "negquadfixonly",
            "negquadlinonly",
            "negquadmix",
        ]

        if not optim_state["gp_meanfun"] in valid_gpmeanfuns:
            raise ValueError(
                """bads:UnknownGPmean:Unknown/unsupported GP mean
            function. Supported mean functions are zero, const,
            egquad, and se"""
            )
        optim_state["int_meanfun"] = self.options.get("gpintmeanfun")
        # more logic here in matlab

        # Starting threshold on y for output warping
        if self.options.get("fitnessshaping"):
            optim_state["outwarp_delta"] = self.options.get(
                "outwarpthreshbase"
            )
        else:
            optim_state["outwarp_delta"] = []

        return optim_state

    def _init_mesh_(self):
        # Evaluate starting point and initial mesh, determine if function is noisy
        self.yval, self.fsd, _ = self.function_logger(self.u)
        
        # set up strings for logging of the iteration
        self.display_format = self._setup_logging_display_format()

        if self.optim_state["uncertainty_handling_level"] > 0:
            self.logger.info(
                "Beginning optimization assuming of a STOCHASTIC objective function")
        else:
            # test if the function is noisy 
            self.logging_action.append('Uncertainty test')
            yval_bis = self.function_logger.fun(self.x0)
            self.function_logger.func_count += 1
            if (np.abs(self.yval - yval_bis) > self.options['tolnoise'] ):
                self.optim_state['uncertainty_handling_level'] = 1
                self.logger.info(
                "Beginning optimization assuming of a STOCHASTIC objective function")
                self.logging_action.append('Uncertainty test')
            else:
                self.logger.info(
                    "Beginning optimization assuming of a DETERMINISTIC objective function")


        self._log_column_headers()

        self._display_function_log_(0, '')
        
        # Only one function evaluation
        if self.options['maxfunevals'] == 1:
            is_finished = True
            return

        # If dealing with a noisy function, use a large initial mesh
        if self.optim_state["uncertainty_handling_level"] > 0:
            self.options['ninit'] = np.minimum(np.maximum(20, self.options['ninit']), self.options['maxfunevals'])

        # TODO: Additional initial points
        if self.options['ninit'] > 0:
            # Evaluate initial points but not more than OPTIONS.MaxFunEvals
            ninit = np.minimum(self.options['ninit'], self.options['maxfunevals'] - 1)
            if self.options['initfcn'] == '@init_sobol':
                
                # TODO: call initialization function 
                u1 = init_sobol(self.u0, self.lower_bounds, self.upper_bounds,
                            self.plausible_lower_bounds, self.plausible_upper_bounds, ninit)
                # enforce periodicity TODO function
                u1 = period_check(u1, self.lower_bounds, self.upper_bounds, self.options['periodicvars'])

                # Force points to be on the search grid.
                u1 = force_to_grid(u1, self.optim_state['search_mesh_size'])

                # Remove already evaluated or unfeasible points from search set 
                u1 = contraints_check(u1, self.optim_state['lb_search'], self.optim_state['ub_search'], self.optim_state["tol_mesh"], self.function_logger, True, self.nonbondcons)
                
                yval_u1 = []
                for u_idx in range(len(u1)):
                    yval_u1.append(self.function_logger(u1[u_idx])[0])
                
                yval_u1 = np.array(yval_u1)
                idx_yval = np.argmin(yval_u1)
                self.u = u1[idx_yval]
                self.yval = yval_u1[idx_yval]
                self._display_function_log_(0, 'Initial mesh')
            else:
                raise ValueError('bads:initfcn:Initialization function not implemented yet')
        
        if not np.isfinite(self.yval):
            raise ValueError('init mesh: Cannot find valid starting point.')

        self.fval = self.yval
        self.optim_state['fval'] = self.fval
        self.optim_state['yval'] = self.yval

        return

    def _init_optimization_(self):
        gp = None
        hyp_dict = {}

        # Evaluate starting point and initial mesh,
        self._init_mesh_()
            
        # Change options for uncertainty handling
        if self.optim_state['uncertainty_handling_level'] > 0:
            self.options['tolstalliters'] = 2* self.options['tolstalliters']
            self.options['ndata'] = max(200,self.options['ndata'])
            self.options['minndata'] = 2 * self.options['minndata']
            self.options['meshoverflowswarning'] = 2 * self.options['meshoverflowswarning']
            self.options['minfailedpollsteps'] = np.inf
            self.options['meshnoisemultiplier'] = 0
            if self.options['noisesize']:
                self.options['noisesize'] = 1
            # Keep some function evaluations for the final resampling
            self.options['noisefinalsamples'] = min(self.options['noisefinalsamples'] , self.options['maxfunevals']  - self.function_logger.func_count)
            self.options['maxfunevals'] =self.options['maxfunevals']  - self.options['noisefinalsamples']
            
            if self.optim_state['uncertainty_handling_level'] > 1:
                self.fsd = self.optim_state['S'][np.argmin(self.optim_state['Y']).item()]
            else:
                self.fsd = self.options['noisesize']
        else:
            if self.options['noisesize'] is None:
                self.options['noisesize'] = np.sqrt(self.options['tolfun'])
            
            self.fsd = 0.0
            
        self.optim_state['fsd']= self.fsd
        self.ubest = self.u.copy()
        self.optim_state['usuccess'] = self.ubest.copy()
        self.optim_state['ysuccess'] = self.yval
        self.optim_state['fsuccess'] = self.fval
        self.optim_state['u'] = self.u.copy()
        self.optim_state['u_success'] = []
        self.optim_state['y_success'] = []
        self.optim_state['f_success'] = []
        
        # Initialize Gaussian Process (GP) structure
        gp, Ns_gp, sn2hpd, hyp_dict = train_gp(hyp_dict, self.optim_state, self.function_logger, self.iteration_history, self.options,
            self.plausible_lower_bounds, self.plausible_upper_bounds)
        self.iteration_history['gp_last_reset_idx'] = 0

        self.gp_stats = IterationHistory(["iter_gp","fval","ymu","ys","gp",])
        
        return gp, Ns_gp, sn2hpd, hyp_dict, 

    def optimize(self):
        """
        Run inference on an initialized ``BADS`` object. 

        Parameters
        ----------

        Returns
        ----------
        
        """
        is_finished = False
        poll_iteration = -1
        self.logging_action = []
        timer = Timer()
        hyp_dict = {}
        self.search_success = 0
        self.search_spree = 0
        self.restarts = self.options['restarts']

        # Initialize gp
        gp, Ns_gp, sn2hpd, hyp_dict = self._init_optimization_()

        if self.options['outputfcn'] is not None:
            output_fcn = self.options['outputfcn']
            is_finished = output_fcn(self.var_transf.inverse_transf(self.u), 'init')

        poll_iteration += 1
        loop_iter = 0
        while not is_finished:
            self.optim_state["iter"] = poll_iteration
            self.gp_refitted_flag = False 
            self.gp_exit_flag = np.inf
            action_txt = ''             # Action performed this iteration (for printing purposes)
            

            #Compute mesh size and search mesh size
            self.mesh_size = self.options['pollmeshmultiplier']**(self.mesh_size_integer)
            self.optim_state['mesh_size'] = self.mesh_size

            if self.options['searchsizelocked']:
                self.optim_state['search_size_integer'] = np.minimum(0,
                    self.mesh_size_integer * self.options['searchgridmultiplier'] - self.options['searchgridnumber'])

            self.optim_state['search_mesh_size'] = self.options['pollmeshmultiplier'] ** self.optim_state['search_size_integer']

            # Update bounds to grid search mesh
            self.optim_state['lb_search'], self.optim_state['ub_search'] = self._update_search_bounds_()

            # Minimum improvement for a poll/search to be considered successful
            sufficient_improvement = self.options["tolimprovement"] * (self.mesh_size ** (self.options['forcingexponent']))
            if self.options['sloppyimprovement']:
                sufficient_improvement = np.maximum(sufficient_improvement, self.options['tolfun'])
            
            self.optim_state['search_sufficient_improvement'] = sufficient_improvement.copy()

            do_search_step_flag = self.optim_state['search_count'] < self.options['searchntry'] \
                    and  len(self.function_logger.Y[self.function_logger.X_flag]) > self.D

            #TODO test search step
            do_search_step_flag = True
            if do_search_step_flag:
                # Search stage
                u_search, search_dist, f_mu_search, f_sd_search, gp = self._search_stage_(gp)
                

            # End Search step

            # Check wether to perform the poll stage, it can be run consecutively after the search.
            if self.optim_state['search_count'] == 0 \
                or self.optim_state['search_count'] == self.options['searchntry']:
                self.optim_state['search_count'] = 0
                
                if self.search_success > 0 and self.options['skippollaftersearch']:
                    do_poll_stage = False
                    self.search_spree += 1
                    if self.options['searchmeshexpand'] > 0 \
                        and np.mod(self.search_spree, self.options['searchmeshexpand']) == 0 \
                        and self.options['searchmeshincrement'] > 0:
                        # Check if mesh size is already maximal
                        self._check_mesh_overflow_()
                        self.mesh_size_integer = np.minimum(self.mesh_size_integer + self.options['searchmeshincrement'],
                                                    self.options['maxpollgridnumber'])
                else:
                    do_poll_stage = True
                    self.search_spree = 0
                
                self.search_success = 0
            else:
                do_poll_stage = True
            
            self.u = self.ubest
            #Check do_poll_stage
            if do_poll_stage:
                self._poll_stage()
                    
            # Finalize the iteration
            
            #TODO: Scatter plot of iteration
            if self.options['plot'] == 'scatter':
                pass
            
            #TODO: GP hyperparameters at end of iteration

            #TODO: Check termination conditions

            if self.function_logger.func_count >= self.options['maxfunevals']:
                is_finished = True
                #exit_flag = 0
                msg = 'Optimization terminated: reached maximum number of function evaluations OPTIONS.MaxFunEvals.'
            
            if poll_iteration >= self.options['maxiter']:
                is_finished = True
                #exit_flag = 0
                msg = 'Optimization terminated: reached maximum number of iterations OPTIONS.MaxIter.'
            
            if self.optim_state['mesh_size'] < self.optim_state['tol_mesh']:
                is_finished = True
                #exit_flag = 1
                msg = 'Optimization terminated: mesh size less than OPTIONS.TolMesh.'
            
            # TODO: Historic improvement
            if poll_iteration >  self.options['tolstalliters']:
                is_finished = True
                #exit_flag = 1
                msg = 'Optimization terminated: mesh size less than OPTIONS.TolMesh.'
            
            # TODO: Store best points at the end of each iteration, or upon termination
            if do_poll_stage or is_finished:
                self.iteration_history.record('u', self.u, poll_iteration)
                self.iteration_history.record('yval', self.yval, poll_iteration)
                self.iteration_history.record('fval', self.fval, poll_iteration)
                self.iteration_history.record('gp_hyp_full', gp.get_hyperparameters(as_array=True), poll_iteration)
                self.iteration_history.record('gp', gp, poll_iteration)

            #TODO: Re-evaluate all points (Uncertainty handling case)
            if self.optim_state['uncertainty_handling_level'] > 0 and do_poll_stage and poll_iteration > 0:
                self._re_evaluate_history_(gp)
                #TODO: Uncertainty handling case
            
            #TODO: if isFinished_flag
            if is_finished:
                # Multiple starts (deprecated)
                if self.restarts > 0:
                    pass
            else:
                if do_poll_stage:
                    # Iteration corresponds to the number of polling iterations
                    poll_iteration +=1
                    self.optim_state['iter'] = poll_iteration

            loop_iter += 1
        # End while

        #TODO Re-evaluate all best points (skip first iteration)
        yval_vec = self.yval if np.np.isscalar(self.yval) else self.yval.copy()
        if self.optim_state['uncertainty_handling_level'] > 0 and poll_iteration > 0:
            pass
        #TODO
        # if ~isempty(outputfun)
        #    isFinished_flag = outputfun(origunits(u,optimState),optimState,'done');

        # TODO: Convert back to original space

        #TODO:  Print final message
        return None 
    
    def _search_stage_(self, gp: GP):
        # Check where it is time to refit the GP
        refit_flag, do_gp_calibration = self.is_gp_refit_time(self.options['normalphalevel'])
        if refit_flag \
            or self.optim_state['search_count'] == 0:

            # Local GP approximation on current incumbent
            gp = local_gp_fitting(gp, self.u, self.function_logger, self.options, self.optim_state, self.iteration_history, refit_flag)

            if refit_flag:
                self.refitted_flag = True
            # TODO: self.gp_exit_flag = np.minimum(gp_temp_flag, gp_exit_flag) gp_temp_flag: in case of fitting errors
        # End fitting            

        # Update Target from GP prediction
        fmu, f_target_s, f_target = self._get_target_from_gp(self.ubest, gp)
        self.optim_state["fval"] = fmu.item()
        self.optim_state["f_target_s"] = f_target_s
        self.optim_state["f_target"] = f_target.item()


        # Generate search set (normalized coordinate)
        self.optim_state['search_count'] += 1
        
        self.search_es_hedge = SearchESHedge(self.options['searchmethod'], self.options, self.nonbondcons)
        u_search_set, z = self.search_es_hedge(self.u, self.lower_bounds, self.upper_bounds, self.function_logger, gp , self.optim_state)
        
        # Enforce periodicity
        u_search_set = period_check(u_search_set, self.lower_bounds, self.upper_bounds, self.optim_state['periodicvars'])

        # Force candidate points on search grid
        u_search_set = force_to_grid(u_search_set, self.optim_state["search_mesh_size"])

        # Remove already evaluated or unfeasible points from search set 
        u_search_set = contraints_check(u_search_set, self.optim_state['lb_search'], self.optim_state['ub_search'], self.optim_state['tol_mesh'],
                            self.function_logger, True, self.nonbondcons)

        # The Acquisition Hedge policy is not yet supported (even in Matlab)
        index_acq = None
        if u_search_set.size > 0:
            # Batch evaluation of acquisition function on search set 
            z, f_mu, fs = acq_fcn_lcb(u_search_set, self.function_logger, gp)
            # Evaluate best candidate point in original coordinates
            index_acq = np.argmin(z)

            # TODO: In future handle acquisition portfolio (Acquisition Hedge), it's even unsupported in Matlab
        
            # Randomly choose index if something went wrong
            if index_acq is None or index_acq.size < 1 or np.any(~np.isfinite(index_acq)):
                self.logger.warn("bads:optimze: Acquisition function failed")
                index_acq = np.random.randint(0, len(u_search_set) +1 )

            # u_search at the candidate acquisition point
            u_search = u_search_set[index_acq] 
            
            #TODO: Local optimization of the acquisition function (generally it does not improve results)
            if self.options["searchoptimize"]:
                pass

            y_search, f_sd_search, idx = self.function_logger(u_search)

            if z.size > 0:
                # TODO: in Matlab we distinguish stats of gp variable and iteration history
                # Save statistics of gp prediction, 
                self._save_gp_stats_(y_search, f_mu[index_acq], fs[index_acq])
            
            # Add search point to training setMeshSize
            if u_search.size > 0 & self.search_es_hedge.count < self.options["searchntry"]:
                # TODO: Handle FitnessShaping and rotate gp axes (latter one is unsupported)

                # update posterior, since we added the new point
                gp = reupdate_gp(self.function_logger, gp)

                if np.any(~np.isfinite(gp.y)):
                    self.logger.warn("bads:opt: GP prediction is non-finite")

            # If the function is non-deterministic we update the posterior of the GP
            if self.optim_state["uncertainty_handling_level"] > 0:
                gp = local_gp_fitting(gp, u_search, self.function_logger, self.options, self.optim_state, self.iteration_history, False)
                # TODO: prediction on u_search  gppred(usearch,gpstructnew); line 668
                f_mu_search, f_sd_search = gp.predict(u_search)
                f_sd_search = np.sqrt(f_sd_search)
            else:
                f_mu_search = y_search
                f_sd_search = 0
            
            # Compute distance of search point from current point
            search_dist = np.sqrt(udist(self.ubest, u_search, gp.temporary_data['len_scale'], self.optim_state['lb'], self.optim_state['ub'],
                            self.optim_state['scale'], self.optim_state['periodicvars']))

        else:
            # Search set is empty
            y_search = self.yval
            f_mu_search = self.fval
            f_sd_search = 0
            search_dist = 0
            
        # TODO: CMA-ES like estimation of local covariance structure (unused)
        if self.options['hessianupdate'] and self.options['hessianmethod'] == 'cmaes':
            pass
        
        # Evaluate search
        search_improvement = self.eval_improvement(self.fval, f_mu_search, self.fsd, f_sd_search, self.options['improvementquantile'])
        fval_old = self.fval if np.isscalar(self.fval) else self.fval.copy()

        # Declare if search was success or failure
        if (search_improvement > 0 and self.options['sloppyimprovoment']) \
            or search_improvement > self.optim_state['search_sufficient_improvement']:

            if self.options['acqhedge']:
                # TODO: Acquisition hedge (acquisition portfolio) not supported yet
                pass
            else:
                method = self.search_es_hedge.chosen_search_fun
            
            if search_improvement > self.optim_state['search_sufficient_improvement']:
                self.search_success += 1
                search_string = f'Successful search ({method})'
                self.optim_state['u_success'].append(u_search)
                self.optim_state['y_success'].append(y_search)
                self.optim_state['f_success'].append(f_mu_search)
                search_status = 'success'
            else:
                search_string = f'Incremental search ({method})'
                search_status = 'incremental'

            # Update incumbent point (self.yval, self.fval, self.fsd) and optim_state
            self._update_incumbent(u_search, y_search, f_mu_search, f_sd_search)
            
            #TODO reset posterior GP
            self.force_gp_fitting = True

        else:
            search_status = 'failure'
            search_string = ''
        
        # Update portfolio acquisition function (not supported)

        # Update search portfolio (needs improvement)
        if self.search_es_hedge is not None and u_search_set.size > 0:
            self.search_es_hedge.update_hedge(u_search, fval_old, f_mu_search, f_sd_search,
                                    gp, self.optim_state['mesh_size'])           

        # TODO: Update search statistics and search scale factor
        self._update_search_stats_(search_status, search_dist)

        # TODO: Print search results
        if len(search_string) > 0:
            self.logging_action.append('')
            self._display_function_log_(self.optim_state['iter'], search_string)   

        return u_search, search_dist, f_mu_search, f_sd_search, gp
    
    def eval_improvement(self, f_base, f_new, s_base, s_new, q):
        """
            Evaluate optimization improvement. Returns the improvement of FNEW over 
            FBASE for a minimization problem (larger improvements are better).
        """
        if s_base is None or s_new is None:
            z = f_base - f_new
        else:
            # This needs to be corrected -- but for q=0.5 it does not matter
            mu = f_base - f_new
            sigma = np.sqrt(s_base**2 + s_new**2)
            x0 = -np.sqrt(2) * erfcinv(2*q)
            z = sigma * x0 + mu

        return z
    
    def _poll_stage(self):
        pass

    def _save_gp_stats_(self, fval, ymu, ys):
        
        if self.gp_stats.get('iter_gp') is None or len(self.gp_stats.get('iter_gp')) == 0:
            iter = 0
        else:
            iter = self.gp_stats.get('iter_gp') + 1
        
        self.gp_stats.record('iter_gp', iter, iter)
        self.gp_stats.record('fval', fval, iter)
        self.gp_stats.record('ymu', ymu, iter)
        self.gp_stats.record('ys', ys, iter)



    def is_gp_refit_time(self, alpha):
        # Check calibration of Gaussian process prediction
        
        if self.function_logger.func_count < self.options['maxiter'] / self.D:
            refit_period = np.maximum(10, self.options['maxiter'])
        else:
            refit_period = self.options['maxiter'] * 5
        
        gp_iter_idx = self.gp_stats.get('gp_iter')

        do_gp_calibration = False
        # empty stats
        if gp_iter_idx is None or len(gp_iter_idx) == 0 or gp_iter_idx == 0:
            if gp_iter_idx is None:
                gp_iter_idx = 0
            do_gp_calibration = True
        
        # if stats data is available check z_score
        if not do_gp_calibration:
            zscore = self.gp_stats.get('fval')[:gp_iter_idx+1] - self.gp_stats.get('ymu')[:gp_iter_idx+1]
            zscore = zscore / (self.gp_stats.get('ys')[:gp_iter_idx+1])

            if np.any(np.isnan(zscore)):
                do_gp_calibration = True
            else: 
                n = np.size(zscore)
                if n < 3:
                    chi_to_inv = lambda y, v: gammaincinv(v/2, y)
                    plo = chi_to_inv(alpha/2, n)
                    phi = chi_to_inv(1-alpha/2, n)
                    total = np.sum(zscore**2)
                    if total < plo or total > phi or np.any(np.isnan(plo)) or np.any(np.isnan(phi)):
                        do_gp_calibration = True
                    else:
                        do_gp_calibration =  False
                else:
                    shapiro_test = shapiro(zscore)
                    do_gp_calibration = shapiro_test.pvalue < alpha
                    
        func_count = self.function_logger.func_count

        refit_flag = self.optim_state['lastfitgp'] < (func_count - self.options['minrefittime']) \
            and (gp_iter_idx >= refit_period or do_gp_calibration) and func_count > self.D
        
        if refit_flag:
            
            self.optim_state['lastfitgp'] = self.function_logger.func_count

            # Reset GP statistics GP
            self.gp_stats = IterationHistory(["iter_gp","fval","ymu","ys","gp",])
            do_gp_calibration = False

        return refit_flag, do_gp_calibration


    # Corresponds to Matlab: updateTarget
    def _get_target_from_gp(self, u, gp:GP):
        if self.optim_state['uncertainty_handling_level'] > 0 \
            or self.options['uncertainincumbent']:
            fmu, fs2 = gp.predict(u)
            # TODO sampling weight
            f_target_s = np.sqrt(np.max(fs2, axis=0))
            if ~np.isfinite(fmu) | ~np.isreal(f_target_s) | ~np.isfinite(f_target_s):
                fmu = self.optim_state['fval']
                f_target_s = self.optim_state['fsd']
            
            # f_target: Set optimization target slightly below the current incumbent
            if self.options['alternativeincumbent']:
                f_target = fmu - np.sqrt(self.D) / np.sqrt(self.function_logger.func_count) * f_target_s
            else:
                f_target = fmu - self.optim_state['sd_level'] * np.sqrt(fs2 + self.options['tolfun']**2)
        else:
            f_target = self.optim_state['fval'] - self.options['tolfun']
            fmu = self.optim_state['fval']
            f_target_s = 0
            
        return fmu.item, f_target_s, f_target        

    def _update_search_bounds_(self):
        lb = self.optim_state['lb']
        lb_search = force_to_grid(lb, self.optim_state['search_mesh_size'])
        lb_search[lb_search < lb] = lb_search[lb_search < lb] + self.optim_state['search_mesh_size']

        ub = self.optim_state['ub']
        ub_search = force_to_grid(ub, self.optim_state['search_mesh_size'])
        ub_search[ub_search > ub] = ub_search[ub_search > ub] - self.optim_state['search_mesh_size']
        return lb_search, ub_search

    def _update_incumbent(self, u_new, yval_new, fval_new, fsd_new):
        """
            Move incumbent (current point) to a new point.
        """
        self.optim_state['u'] = u_new
        self.optim_state['yval'] = yval_new
        self.optim_state['fval'] = fval_new
        self.u = u_new
        self.yval = yval_new
        self.fval = fval_new
        self.fsd = fsd_new
        return yval_new, fval_new, fsd_new
        #Update estimate of curvature (Hessian) - not supported (GP usage)

    def _update_search_stats_(self, search_status, search_dist):
        if not 'search_stats' in self.optim_state \
            or len(self.optim_state['search_stats']) == 0:
            search_stats = {}
            search_stats['log_search_factor'] = []
            search_stats['success'] = []
            search_stats['udist'] = []
            self.optim_state['search_stats'] = search_stats
        else:
            search_stats = self.optim_state['search_stats']
        
        search_stats['log_search_factor'].append(np.log(self.optim_state['search_factor']))
        search_stats['udist'].append(search_dist)

        if search_status == 'success':
            search_stats['success'].append(1.0)
            self.optim_state['search_factor'] = self.optim_state['search_factor']*self.options['searchscalesuccess']
            if self.options['adaptiveincumbentshift']:
                self.optim_state['sd_level'] = self.optim_state['sd_level'] * 2

        elif search_status == 'incremental':
            search_stats['success'].append(0.5)
            self.optim_state['search_factor'] = self.optim_state['search_factor']*self.options['searchscaleincremental']
            if self.options['adaptiveincumbentshift']:
                self.optim_state['sd_level'] = self.optim_state['sd_level'] * 2**2

        elif search_status == 'failure':
            search_stats['success'].append(0.)
            self.optim_state['search_factor'] = self.optim_state['search_factor']*self.options['searchscalefailure']
            if self.options['adaptiveincumbentshift']:
                self.optim_state['sd_level'] = np.maximum(self.options['incumbentsigmamultiplier'],
                                                    self.optim_state['sd_level']/2)
        
        # Reset search factor at the end of each search
        if self.optim_state['search_count'] == self.options['searchntry']:
            self.optim_state['search_factor'] = 1

        return

    def _re_evaluate_history_(self, iter):
        
        if self.optim_state['last_re_eval'] == self.function_logger.func_count:
            return

        # Why we need to re-evaluate if everything is already stored
        for i in range(iter):
            gp = self.iteration_history.get('gp')[i]
            u = self.iteration_history.get('u')[i]
            fval, fsd = gp.predict(u)
            fsd = np.sqrt(fsd)

            self.iteration_history.record('fval', fval, i)
            self.iteration_history.record('fsd', fsd, i)
        
        self.optim_state['last_re_eval'] = self.function_logger.func_count


    def _check_mesh_overflow_(self):
        if self.mesh_size_integer == self.options['maxpollgridnumber']:
            self.mesh_overflows += 1
            if self.mesh_overflows == np.ceil(self.options['meshoverflowswarning']):
                self.logger.warn('bads:meshOverflow \t The mesh attempted to expand above maximum size too many times. Try widening PLB and PUB.')
        

    
    def _create_result_dict(self, idx_best: int, termination_message: str):
        """
        Private method to create the result dict.
        """
        output = dict()
        output["function"] = str(self.function_logger.fun)
        if np.all(np.isinf(self.optim_state["lb"])) and np.all(
            np.isinf(self.optim_state["ub"])
        ):
            output["problemtype"] = "unconstrained"
        else:
            output["problemtype"] = "boundconstraints"

        output["iterations"] = self.optim_state["iter"]
        output["func_count"] = self.function_logger.func_count
        output["bestiter"] = idx_best
        output["trainsetsize"] = self.iteration_history["n_eff"][idx_best]
        output["components"] = self.vp.K
        output["rindex"] = self.iteration_history["rindex"][idx_best]
        if self.iteration_history["stable"][idx_best]:
            output["convergencestatus"] = "probable"
        else:
            output["convergencestatus"] = "no"

        output["overhead"] = np.NaN
        output["rngstate"] = "rng"
        output["algorithm"] = "Variational Bayesian Monte Carlo"
        output["version"] = "0.0.1"
        output["message"] = termination_message

        output["elbo"] = self.vp.stats["elbo"]
        output["elbo_sd"] = self.vp.stats["elbo_sd"]

        return output

    def _log_column_headers(self):
        """
        Private method to log column headers for the iteration log.
        """
        if self.optim_state["cache_active"]:
            self.logger.info(
                " Iteration f-count/f-cache     E[f(x)]     SD[f(x)]     MeshScale     Method     Actions")
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                self.logger.info(
                    " Iteration f-count     E[f(x)]     SD[f(x)]     MeshScale     Method     Actions")
            else:
                self.logger.info(
                    " Iteration f-count     f(x)     MeshScale     Method     Actions")

    def _setup_logging_display_format(self):
        """
        Private method to set up the display format for logging the iterations.
        """
        if self.optim_state["cache_active"]:
            display_format = " {:5.0f}     {:5.0f}/{:5.0f}   {:12.6f}  "
            display_format += ("{:12.6f}  {:12.6f}     {}       {}")
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                display_format = " {:5.0f}     {:5.0f}   {:12.6f}  "
                display_format += ("{:12.6f}  {:12.6f}     {}       {}")
            else:
                display_format = " {:5.0f}     {:5.0f}   {:12.6f}  "
                display_format += ("{:12.6f}     {}       {}")

        return display_format

    def _display_function_log_(self, iteration, method):
        if (self.optim_state["uncertainty_handling_level"] > 0 \
                and self.options.get("maxrepeatedobservations") > 0):

            self.logger.info(self.display_format.format(
                                iteration,
                                self.function_logger.func_count,
                                self.yval,
                                self.fsd,
                                self.optim_state["mesh_size"],
                                method,
                                "".join(self.logging_action[-1]),))
        else:
            self.logger.info(self.display_format.format(
                                iteration,
                                self.function_logger.func_count,
                                self.yval,
                                self.optim_state["mesh_size"],
                                method,
                                "".join(self.logging_action[-1]),))
        return
