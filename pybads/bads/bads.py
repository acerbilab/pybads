import copy
import logging
import math
import os
import sys

#import gpyreg as gpr
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaincinv
from scipy.stats import shapiro

from pybads.search.search_hedge import SearchESHedge

from pybads.function_logger import FunctionLogger
from pybads.init_functions.init_sobol import init_sobol
from pybads.search.grid_functions import force_to_grid, grid_units, get_grid_search_neighbors 
from pybads.search.es_search import es_update
from pybads.utils.period_check import period_check
from pybads.utils.timer import Timer
from pybads.utils.iteration_history import IterationHistory
from pybads.bads.variables_transformer import VariableTransformer
from pybads.utils.ucheck import ucheck
from .gaussian_process_train import reupdate_gp, train_gp
from gpyreg.gaussian_process import GP
from .options import Options


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
        nonbondcons: callable = None,
        user_options: dict = None
    ):
        # set up root logger (only changes stuff if not initialized yet)
        logging.basicConfig(stream=sys.stdout, format="%(message)s")

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

        ) = self._boundscheck(
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
                "fval",
                "ymu",
                "ys"
                "lcbmax",
                "gp",
                "gp_last_reset_idx"
                "gp_hyp_full",
                "Ns_gp",
                "timer",
                "optim_state",
                "func_count",
                "n_eff",
                "logging_action",
            ]
        )

    def _boundscheck(
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
            points X0 are not inside the provided hard bounds LB and UB."""
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
        self.mesh_size_int = 0 # Mesh size in log base units
        optim_state["mesh_size_integer"] = self.mesh_size_int
        optim_state["search_size_integer"] = np.minimum(0, self.mesh_size_int * self.options.get("searchgridmultiplier") - self.options.get("searchgridnumber"))
        optim_state["mesh_size"] =  float(self.options.get("pollmeshmultiplier"))**self.mesh_size_int
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

        optim_state["lb_orig"] = self.var_transf.orig_lb.copy()
        optim_state["ub_orig"] = self.var_transf.orig_ub.copy()
        optim_state["plb_orig"] = self.var_transf.orig_plb.copy()
        optim_state["pub_orig"] = self.var_transf.orig_pub.copy()

        # Bounds for search mesh
        lb_search = force_to_grid(self.lower_bounds, optim_state["search_mesh_size"])
        lb_search[ lb_search < self.lower_bounds] = lb_search[ lb_search < self.lower_bounds] + optim_state["search_mesh_size"]
        optim_state["lb_search"] = lb_search
        ub_search = force_to_grid(self.upper_bounds, optim_state["search_mesh_size"])
        ub_search[ ub_search > self.upper_bounds] = lb_search[ub_search > self.upper_bounds] - optim_state["search_mesh_size"]
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
        optim_state['tolmesh'] = self.options['pollmeshmultiplier']**np.ceil(np.log(self.options['tolmesh']) / np.log(self.options['pollmeshmultiplier']))
                

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
        optim_state['searchfactor']   =   1
        optim_state['sdlevel']        = self.options['incumbentsigmamultiplier']
        optim_state['search_count']    = self.options['searchntry']       # Skip search at first iteration
        optim_state['lastreeval']     = -np.inf;                     # Last time function values were re-evaluated
        optim_state['lastfitgp']      = -np.inf;                     # Last fcn evaluation for which the gp was trained
        optim_state['meshoverflows']  = 0;                       # Number of attempted mesh expansions when already at maximum size

        # List of points at the end of each iteration
        optim_state["iterlist"] = dict()
        optim_state["iterlist"]["u"] = []
        optim_state["iterlist"]["fval"] = []
        optim_state["iterlist"]["fsd"] = []
        optim_state["iterlist"]["fhyp"] = []

        
        optim_state['es'] = es_update(es_mu, es_lambda)

        # Hedge struct
        optim_state['search_hedge'] = dict()


        # Before first iteration
        # Iterations are from 0 onwards in optimize so we should have -1
        optim_state["iter"] = -1

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
        display_format = self._setup_logging_display_format()

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

        self.logger.info(display_format.format(
                            iteration,
                            self.function_logger.func_count,
                            self.yval,
                            self.fsd,
                            self.optim_state["mesh_size"],
                            '',
                            "".join(self.logging_action),)
                        )
        
        # Only one function evaluation
        if self.options['maxfuneval'] == 1:
            is_finished = True
            return

        # If dealing with a noisy function, use a large initial mesh
        if self.optim_state["uncertainty_handling_level"] > 0:
            self.options['ninit'] = np.minimum(np.maximum(20, self.options['ninit']), self.options['maxfuneval'])

        # TODO: Additional initial points
        if self.options['ninit'] > 0:
            # Evaluate initial points but not more than OPTIONS.MaxFunEvals
            ninit = np.minimum(self.options['ninit'], self.options['maxfuneval'] - 1)
            if self.options['initfcn'] == '@init_sobol':
                
                # call initialization function TODO function
                u1 = init_sobol(self.u0, self.lower_bounds, self.upper_bounds,
                            self.plausible_lower_bounds, self.plausible_upper_bounds, ninit)
                # enforce periodicity TODO function
                u1 = period_check(u1, self.lower_bounds, self.upper_bounds, self.options['periodicvars'])

                # Force points to be on the search grid.
                u1 = force_to_grid(u1, self.optim_state['search_mesh_size'])

                #TODO:  ucheck
                u1 = ucheck()
                
                yval_u1 = []
                for u_idx in range(len(u1)):
                    yval_u1.append(self.function_logger(u1[u_idx])[0])
                
                yval_u1 = np.array(yval_u1)
                idx_yval = np.argmin(yval_u1)
                self.u = u1[idx_yval]
                self.yval = yval_u1[idx_yval]
                self.logger.info(display_format.format(
                            0,
                            self.function_logger.func_count,
                            self.yval,
                            self.fsd,
                            self.optim_state["mesh_size"],
                            'Initial mesh', '',)
                        )
            else:
                raise ValueError('bads:initfcn:Initialization function not implemented yet')
        
        if not np.isfinite(self.yval):
            raise ValueError('init mesh: Cannot find valid starting point.')

        self.fval = self.yval
        self.optim_state['fval'] = self.fval.copy()
        self.optim_state['yval'] = self.yval.copy()

        return

    def optimize(self):
        """
        Run inference on an initialized ``BADS`` object. 

        Parameters
        ----------

        Returns
        ----------
        
        """
        is_finished = False
        iteration = -1
        self.logging_action = []
        timer = Timer()
        gp = None
        hyp_dict = {}
        search_sucess_flag = False
        search_spree = False
        restarts = self.options['restarts']

        # Evaluate starting point and initial mesh,
        self._init_mesh_()

        if self.options['outputfcn'] is not None:
            output_fcn = self.options['outputfcn']
            is_finished_flag = output_fcn(self.var_transf.inverse_transf(self.u), 'init')
        
        # Change options for uncertainty handling
        if self.optim_state['uncertaintyhandling'] > 0:
            self.options['tolstalliters'] = 2* self.options['tolstalliters']
            self.options['ndata'] = max(200,self.options['ndata'])
            self.options['minndata'] = 2 * self.options['minndata']
            self.options['meshoverflowswarning'] = 2 * self.options['meshoverflowswarning']
            self.options['minfailedpollsteps'] = np.inf
            self.options['meshnoisemultiplier'] = 0
            if (self.options['noisesize']):
                self.options['noisesize'] = 1
            # Keep some function evaluations for the final resampling
            self.options['noisefinalsamples'] = min(self.options['noisefinalsamples'] , self.options['maxfunevals']  - self.function_logger.func_count)
            self.options['maxfunevals'] =self.options['maxfunevals']  - self.options['noisefinalsamples']
            
            if self.optim_state['uncertaintyhandling'] > 1:
                self.fsd = self.optim_state['S'][np.argmin(self.optim_state['Y']).item()]
            else:
                self.fsd = self.options['noisesize']
        else:
            if self.options['noisesize'] is None:
                self.options['noisesize'] = np.sqrt(self.options['tolfun'])
            
            self.fsd = 0.0
            
        self.optim_state['fsd']= self.fsd
        self.ubest = self.u.copy()
        self.optim_state['usucess'] = self.ubest.copy()
        self.optim_state['ysucess'] = self.yval.copy()
        self.optim_state['fsuccess'] = self.fval.copy()
        self.optim_state['u'] = self.u.copy()
        
        gp, Ns_gp, sn2hpd, hyp_dict = train_gp(hyp_dict, self.optim_state, self.function_logger, self.iteration_history, self.options,
            self.plausible_lower_bounds, self.plausible_upper_bounds)
        gp.temporary_data['lenscale'] = 1
        gp.temporary_data['pollscale'] = np.ones((1,self.D))
        gp.temporary_data['effective_radius'] = 1.
        self.iteration_history['gp_last_reset_idx'] = 0
        
        iteration = 0

        while not is_finished:
            self.optim_state["iter"] = iteration
            gp_refitted_flag = False 
            gp_exit_flag = np.inf
            action_txt = ''             # Action performed this iteration (for printing purposes)

            #Compute mesh size and search mesh size
            mesh_size = self.options['pollmeshmultiplier']**(self.mesh_size_int)
            if self.options['searchsizelocked']:
                self.optim_state['search_size_integer'] = np.minimum(0,
                    self.mesh_size_int * self.options['searchgridmultiplier'] - self.options['searchgridnumber'])

            self.optim_state['mesh_size'] = mesh_size
            self.optim_state['search_mesh_size'] = self.options['pollmeshmultiplier'] ** self.optim_state['search_size_integer']

            # Update bounds to grid search mesh
            self.optim_state['lb_search'], self.optim_state['ub_search'] = self._update_search_bounds()

            # Minimum improvement for a poll/search to be considered successful
            sufficient_improvement = self.options["tolimprovement"] * (mesh_size ** (self.options['forcingexponent']))
            if self.options['sloppyimprovement']:
                sufficient_improvement = np.maximum(sufficient_improvement, self.options['tolfun'])
            self.optim_state['sufficient_improvement'] = sufficient_improvement.copy()

            do_search_step_flag = self.optimstate['search_count'] < self.options['searchntry'] \
                    and  len(self.function_logger.Y[self.function_logger.X_flag]) > self.D
            do_poll_stage = False

            if do_search_step_flag:
                # Search stage
                self._search_stage_(gp)
            else :
                # Check wether to perform the poll stage
                # TODO: do_poll_stage = ....
                pass
            
            #Check do_poll_stage
            if do_poll_stage:
                self._poll_stage()
                    

            iteration += 1  
            self.logging_action = []
            is_finished = True

        return None 
    
    def _search_stage_(self, gp):
        #check where it is time to refit the GP
        refit_flag, do_gp_calibration = self.is_gp_refit_time(self.options['normalphalevel'])

        if refit_flag \
            or self.optim_state['search_count'] == 0:
            # Local GP approximation on current point

            # Update the GP training set by setting the NEAREST neighbors (Matlab: gpTrainingSet)
            gp.X, gp.y, s2 = get_grid_search_neighbors(self.function_logger, self.u, gp, self.options, self.optim_state)
            if s2 is not None:
                gp.s2 = s2                    

            # TODO: Transformation of objective function
            if self.options['fitnessshaping']:
                self.logger.warn("bads:opt:Fitness shaping not implemented yet")
            
            idx_finite_y = np.isfinite(gp.y)
            if np.any(~idx_finite_y):
                y_idx_penalty = np.argmax(gp.y[idx_finite_y])
                gp.y[~idx_finite_y] = gp.y[y_idx_penalty].copy()
                if 'S' in self.optim_state["S"]:
                    gp.s2[~idx_finite_y] = gp.s2[y_idx_penalty]
                    
            gp.temporary_data['erry'] = ~idx_finite_y
            # TODO: No need of test points, don't thin is needed
            #self.function_logger.reset_fun_evaltime()

            #TODO: Rotate dataset (unsupported)

            # Update existing GP

            # TODO Update piors hyperparameters for GP  using empirical Bayes method.
            # TODO: Update empirical prior for GP mean
            # We assume that the mean of the GP is higher than what we see
            #ymean = prctile1(gpstruct.y,options.gpMeanPercentile);
            #yrange = feval(options.gpMeanRangeFun, ymean, gpstruct.y);

            
            #gpstruct.prior.mean{1}{2} = ymean;
            #if ~options.gpFixedMean
            #    gpstruct.prior.mean{1}{3} = yrange.^2/4;
            
            #TODO: Gpyreg has gp.set_priors()
            #TODO: Gpyreg has gp.set_bounds()
            
            #TODO: likelihood prior (scales with mesh size)
            # gpstruct.prior.lik{end}{2} = log(NoiseSize(1)) + options.MeshNoiseMultiplier*log(MeshSize);


            #TODO: Re-fit Guassian Process (optimize or sample -- only optimization supported)
            if refit_flag:
                self.logger.warn("bads:opt:Refit not implemented yet")

            # Recompute posterior
            gp.update(compute_posterior=True)   
            # End fitting

        #Update Target from GP prediction
        fmu, f_target_s, f_target = self._update_target_(self.ubest, gp)
        self.optim_state["fval"] = fmu
        self.optim_state["f_target_s"] = f_target_s
        self.optim_state["f_target"] = f_target

        # Generate search set (normalized coordinate)
        self.search_es_hedge = SearchESHedge(self.options['searchmethod'], self.options)
        u_search_set = self.search_es_hedge(self.u, self.lower_bounds, self.upper_bounds, self.function_logger, gp , self.optim_state)
        
        # Enforce periodicity
        u_search_set = period_check(u_search_set, self.lower_bounds, self.upper_bounds, self.optim_state["periodicvars"])

        # Force candidate points on search grid
        u_search_set = force_to_grid(u_search_set, self.optim_state["search_mesh_size"])

        # TODO: Remove already evaluated or unfeasible points from search set (already evaluated should be fine, look at unfeseabile points!)
        

        # if search set non empty -> evaluate acquisition function on search set, do the "Batch evaluation of acquisition function on search set."
        
        return None
    
    def _poll_stage(self):
        pass

    #TODO check indexes, it should be fine though, we store the last reset index, [gp_reset_idx : iter_idx]
    def is_gp_refit_time(self, alpha):
        
        if self.function_logger.func_count < self.options['maxiter'] / self.D:
            refit_period = np.maximum(10, self.options['maxiter'])
        else:
            refit_period = self.options['maxiter'] * 5
        
        gp_reset_idx = self.iteration_history["gp_last_reset_idx"]
        iter_idx = self.optim_state.get("iter")
        # Check calibration of Gaussian process prediction
        do_gp_calibration = False
        if gp_reset_idx == 0:
            do_gp_calibration = True
        
        zscore = self.iteration_history['fval'][gp_reset_idx : iter_idx+1] - self.iteration_history['fval'][gp_reset_idx : iter_idx+1]
        zscore = zscore / self.iteration_history['ys'][gp_reset_idx : iter_idx+1]

        if np.any(np.isnan(zscore)):
            do_gp_calibration = True
        
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
            and (gp_reset_idx >= refit_period or do_gp_calibration) and func_count > self.D
        
        if refit_flag:
            self.optim_state['lastfitgp'] = self.function_logger.func_count
            nsamples = np.maximum(1, self.options['gpsamples'])
            # Save statistics GP
            self.iteration_history['gp_last_reset_idx'] = self.iteration_history['iter'] + 1
            do_gp_calibration = False

        return refit_flag, do_gp_calibration

    def _update_target_(self, u, gp:GP):
        if self.optim_state['uncertainty_handling_level'] > 0 \
            or self.options['uncertainincumbent']:
            fmu, fs2 = gp.predict(u)
            # TODO sampling weight
            f_target_s = np.sqrt(np.max(fs2, axis=0))
            if ~np.isfinite(fmu) | ~np.isreal(f_target_s) | ~np.isfinite(f_target_s):
                fmu = self.optim_state['fval']
                f_target_s = self.optim_state['fsd']
            
            # f_target: set optimization target slightly below the current incumbent
            if self.options['alternativeincumbent']:
                f_target = fmu - np.sqrt(self.D) / np.sqrt(self.function_logger.func_count) * f_target_s
            else:
                f_target = fmu - self.optim_state['sdlevel'] * np.sqrt(fs2 + self.options['tolffun']**2)
        else:
            f_target = self.optim_state['fval'] - self.options['tolfun']
            fmu = self.optim_state['fval']
            f_target_s = 0
        return fmu, f_target_s, f_target        

    def _update_search_bounds(self):
        lb = self.optim_state['lb']
        lb_search = force_to_grid(self.optim_state['lb'], self.optim_state['search_mesh_size'])
        lb_search[lb_search < lb] = lb_search[lb_search < lb] + self.optim_state['search_mesh_size']

        ub = self.optim_state['ub']
        ub_search = force_to_grid(self.optim_state['lb'], self.optim_state['search_mesh_size'])
        ub_search[ub_search > ub] = ub_search[ub_search > ub] - self.optim_state['search_mesh_size']
        return lb_search, ub_search
        
        
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
