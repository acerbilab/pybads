import copy
import logging
import math
import os
import sys

import gpyreg as gpr
import matplotlib.pyplot as plt
import numpy as np

from pybads.function_logger import FunctionLogger
from pybads.utils.timer import Timer
from pybads.utils.iteration_history import IterationHistory

from .gaussian_process_train import reupdate_gp, train_gp
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
        nonbondcons = None,
        user_options: dict = None
    ):
        # set up root logger (only changes stuff if not initialized yet)
        logging.basicConfig(stream=sys.stdout, format="%(message)s")

        # set up BADS logger
        self.logger = logging.getLogger("BADS")
        self.logger.setLevel(logging.INFO)
        if self.options.get("display") == "off":
            self.logger.setLevel(logging.WARN)
        elif self.options.get("display") == "iter":
            self.logger.setLevel(logging.INFO)
        elif self.options.get("display") == "full":
            self.logger.setLevel(logging.DEBUG)

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
            x0,
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
            self.x0 = 0.5 * (
                self.plausible_lower_bounds + self.plausible_upper_bounds
            )

        self.optim_state = self._init_optim_state()

        self.function_logger = FunctionLogger(
            fun=fun,
            D=self.D,
            noise_flag=self.optim_state.get("uncertainty_handling_level") > 0,
            uncertainty_handling_level=self.optim_state.get(
                "uncertainty_handling_level"
            ),
            cache_size=self.options.get("cachesize"),
        )

        self.iteration_history = IterationHistory(
            [
                "rindex",
                "elcbo_impro",
                "stable",
                "elbo",
                "vp",
                "warmup",
                "iter",
                "elbo_sd",
                "lcbmax",
                "data_trim_list",
                "gp",
                "gp_hyp_full",
                "Ns_gp",
                "timer",
                "optim_state",
                "sKL",
                "sKL_true",
                "pruned",
                "varss",
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
        nonbondcons = None
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

        # % Compute "effective" bounds (slightly inside provided hard bounds)
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
            (lower_bounds < plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds < upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """bads:StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB < PLB < PUB < UB."""
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
        mesh_size_int = 0 # Mesh size in log base units
        optim_state["mesh_size_integer"] = mesh_size_int
        optim_state["search_size_integer"] = np.min(0, mesh_size_int * self.options.get("searchgridmultiplier") - self.options.get("searchgridnumber"))
        optim_state["mesh_size"] =  self.options.get("pollmeshmultiplier")^mesh_size_int
        optim_state["search_mesh_size"] = self.options.get("pollmeshmultiplier")^optim_state["search_size_integer"]
        optim_state["scale"] = 1.

        #TODO transvars

        # Compute transformation of variables
        if self.options.get("nonlinearcaling"):
            logflag = np.full((1, self.D), np.NaN)
            periodicvars = self.options.get("periodicvars")
            if len() != 0:
                logflag[periodicvars] = 0
        else:
            logflag = np.zeros((1, self.D))

        

        optim_state["lb_orig"] = self.lower_bounds
        optim_state["ub_orig"] = self.upper_bounds
        optim_state["plb_orig"] = self.plausible_lower_bounds
        optim_state["pub_orig"] = self.plausible_upper_bounds
        eps_orig = (self.upper_bounds - self.lower_bounds) * self.options.get(
            "tolboundx"
        )
        # inf - inf raises warning in numpy, but output is correct
        with np.errstate(invalid="ignore"):
            optim_state["lb_eps_orig"] = self.lower_bounds + eps_orig
            optim_state["ub_eps_orig"] = self.upper_bounds - eps_orig

        # Transform variables (Transform of lower_bounds and upper bounds can
        # create warning but we are aware of this and output is correct)
        with np.errstate(divide="ignore"):
            optim_state["lb"] = self.parameter_transformer(self.lower_bounds)
            optim_state["ub"] = self.parameter_transformer(self.upper_bounds)
        optim_state["plb"] = self.parameter_transformer(
            self.plausible_lower_bounds
        )
        optim_state["pub"] = self.parameter_transformer(
            self.plausible_upper_bounds
        )

        # Before first iteration
        # Iterations are from 0 onwards in optimize so we should have -1
        optim_state["iter"] = -1

        

        # Proposal function for search
        if self.options.get("proposalfcn") is None:
            optim_state["proposalfcn"] = "@(x)proposal_bads"
        else:
            optim_state["proposalfcn"] = self.options.get("proposalfcn")

        # Start with adaptive sampling
        optim_state["skip_active_sampling"] = False

        # Running mean and covariance of variational posterior
        # in transformed space
        optim_state["run_mean"] = []
        optim_state["run_cov"] = []
        # Last time running average was updated
        optim_state["last_run_avg"] = np.NaN

        # Number of variational components pruned in last iteration
        optim_state["pruned"] = 0

        # Need to switch from deterministic entropy to stochastic entropy
        optim_state["entropy_switch"] = self.options.get("entropyswitch")

        # Only use deterministic entropy if D larger than a fixed number
        if self.D < self.options.get("detentropymind"):
            optim_state["entropy_switch"] = False

        # Tolerance threshold on GP variance (used by some acquisition fcns)
        optim_state["tol_gp_var"] = self.options.get("tolgpvar")

        # Copy maximum number of fcn. evaluations,
        # used by some acquisition fcns.
        optim_state["max_fun_evals"] = self.options.get("maxfunevals")

        # By default, apply variance-based regularization
        # to acquisition functions
        optim_state["variance_regularized_acqfcn"] = True

        # Setup search cache
        optim_state["search_cache"] = []

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
            optim_state["hedge"] = []

        # List of points at the end of each iteration
        optim_state["iterlist"] = dict()
        optim_state["iterlist"]["u"] = []
        optim_state["iterlist"]["fval"] = []
        optim_state["iterlist"]["fsd"] = []
        optim_state["iterlist"]["fhyp"] = []

        optim_state["delta"] = self.options.get("bandwidth") * (
            optim_state.get("pub") - optim_state.get("plb")
        )

        # Deterministic entropy approximation lower/upper factor
        optim_state["entropy_alpha"] = self.options.get("detentropyalpha")

        # Repository of variational solutions (not used in Python)
        # optim_state["vp_repo"] = []

        # Repeated measurement streak
        optim_state["repeated_observations_streak"] = 0

        # List of data trimming events
        optim_state["data_trim_list"] = []

        # Expanding search bounds
        prange = optim_state.get("pub") - optim_state.get("plb")
        optim_state["lb_search"] = np.maximum(
            optim_state.get("plb")
            - prange * self.options.get("activesearchbound"),
            optim_state.get("lb"),
        )
        optim_state["ub_search"] = np.minimum(
            optim_state.get("pub")
            + prange * self.options.get("activesearchbound"),
            optim_state.get("ub"),
        )

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

    def optimize(self):
        """
        Run inference on an initialized ``BADS`` object. 

        Parameters
        ----------

        Returns
        ----------
        
        """
        is_finished = False
        # the iterations of pybads start at 0
        iteration = -1
        timer = Timer()
        gp = None
        hyp_dict = {}
        success_flag = True

        # set up strings for logging of the iteration
        display_format = self._setup_logging_display_format()

        if self.optim_state["uncertainty_handling_level"] > 0:
            self.logger.info(
                "Beginning  optimization assuming NOISY observations"
                + " of the log-joint"
            )
        else:
            self.logger.info(
                "Beginning variational optimization assuming EXACT observations"
                + " of the log-joint."
            )

        self._log_column_headers()

        while not is_finished:
            iteration += 1
            self.optim_state["iter"] = iteration
            self.optim_state["redo_roto_scaling"] = False
            vp_old = copy.deepcopy(self.vp)

            self.logging_action = []

            if iteration == 0 and self.optim_state["warmup"]:
                self.logging_action.append("start warm-up")

            # Switch to stochastic entropy towards the end if still
            # deterministic.
            if self.optim_state.get("entropy_switch") and (
                self.function_logger.func_count
                >= self.optim_state.get("entropy_force_switch")
                * self.optim_state.get("max_fun_evals")
            ):
                self.optim_state["entropy_switch"] = False
                self.logging_action.append("entropy switch")

            # Missing port: Input warping / reparameterization, line 530-625

            ## Actively sample new points into the training set
            timer.start_timer("activeSampling")

            if iteration == 0:
                new_funevals = self.options.get("funevalstart")
            else:
                new_funevals = self.options.get("funevalsperiter")

            # Careful with Xn, in MATLAB this condition is > 0
            # due to 1-based indexing.
            if self.function_logger.Xn >= 0:
                self.function_logger.ymax = np.max(
                    self.function_logger.y[self.function_logger.X_flag]
                )

            if self.optim_state.get("skipactivesampling"):
                self.optim_state["skipactivesampling"] = False
            else:
                if (
                    gp is not None
                    and self.options.get("separatesearchgp")
                    and not self.options.get("varactivesample")
                ):
                    # Train a distinct GP for active sampling
                    # Since we are doing iterations from 0 onwards
                    # instead of from 1 onwards, this should be checking
                    # oddness, not evenness.
                    if iteration % 2 == 1:
                        meantemp = self.optim_state.get("gp_meanfun")
                        self.optim_state["gp_meanfun"] = "const"
                        gp_search, Ns_gp, sn2hpd, hyp_dict = train_gp(
                            hyp_dict,
                            self.optim_state,
                            self.function_logger,
                            self.iteration_history,
                            self.options,
                            self.plausible_lower_bounds,
                            self.plausible_upper_bounds,
                        )
                        self.optim_state["sn2hpd"] = sn2hpd
                        self.optim_state["gp_meanfun"] = meantemp
                    else:
                        gp_search = gp
                else:
                    gp_search = gp

                # Perform active sampling
                if self.options.get("varactivesample"):
                    # FIX TIMER HERE IF USING THIS
                    # [optimState,vp,t_active,t_func] =
                    # variationalactivesample_bads(optimState,new_funevals,
                    # funwrapper,vp,vp_old,gp_search,options)
                    sys.exit("Function currently not supported")
                else:
                    self.optim_state["hyp_dict"] = hyp_dict
                    (
                        self.function_logger,
                        self.optim_state,
                        self.vp,
                    ) = active_sample(
                        gp_search,
                        new_funevals,
                        self.optim_state,
                        self.function_logger,
                        self.iteration_history,
                        self.vp,
                        self.options,
                    )
                    hyp_dict = self.optim_state["hyp_dict"]

            # Number of training inputs
            self.optim_state["N"] = self.function_logger.Xn
            self.optim_state["n_eff"] = np.sum(
                self.function_logger.nevals[self.function_logger.X_flag]
            )

            timer.stop_timer("activeSampling")

            ## Train gp

            timer.start_timer("gpTrain")

            gp, Ns_gp, sn2hpd, hyp_dict = train_gp(
                hyp_dict,
                self.optim_state,
                self.function_logger,
                self.iteration_history,
                self.options,
                self.plausible_lower_bounds,
                self.plausible_upper_bounds,
            )
            self.optim_state["sn2hpd"] = sn2hpd

            timer.stop_timer("gpTrain")

            # Check if reached stable sampling regime
            if (
                Ns_gp == self.options.get("stablegpsamples")
                and self.optim_state.get("stop_sampling") == 0
            ):
                self.optim_state["stop_sampling"] = self.optim_state.get("N")

            ## Optimize variational parameters
            timer.start_timer("variationalFit")

            if not self.vp.optimize_mu:
                # Variational components fixed to training inputs
                self.vp.mu = gp.X.T
                Knew = self.vp.mu.shape[1]
            else:
                # Update number of variational mixture components
                Knew = update_K(
                    self.optim_state, self.iteration_history, self.options
                )


            if self.optim_state.get("recompute_var_post") or (
                self.options.get("alwaysrefitvarpost")
            ):
                # Full optimizations
                N_slowopts = self.options.get("elbostarts")
                self.optim_state["recompute_var_post"] = False
            else:
                # Only incremental change from previous iteration
                N_fastopts = math.ceil(
                    N_fastopts * self.options.get("nselboincr")
                )
                N_slowopts = 1
            # Run optimization of variational parameters
            self.vp, varss, pruned = optimize_vp(
                self.options,
                self.optim_state,
                self.vp,
                gp,
                N_fastopts,
                N_slowopts,
                Knew,
            )

            self.optim_state["vpK"] = self.vp.K
            # Save current entropy
            self.optim_state["H"] = self.vp.stats["entropy"]

            # Get real variational posterior (might differ from training posterior)
            # vp_real = vp.vptrain2real(0, self.options)
            vp_real = self.vp
            elbo = vp_real.stats["elbo"]
            elbo_sd = vp_real.stats["elbo_sd"]

            timer.stop_timer("variationalFit")

            # Finalize iteration

            timer.start_timer("finalize")

            # Compute symmetrized KL-divergence between old and new posteriors
            Nkl = 1e5

            sKL = max(
                0,
                0.5
                * np.sum(
                    self.vp.kldiv(
                        vp2=vp_old,
                        N=Nkl,
                        gaussflag=self.options.get("klgauss"),
                    )
                ),
            )

            # Evaluate max LCB of GP prediction on all training inputs
            fmu, fs2 = gp.predict(gp.X, gp.y, gp.s2, add_noise=False)
            self.optim_state["lcbmax"] = np.max(
                fmu - self.options.get("elcboimproweight") * np.sqrt(fs2)
            )

            # Compare variational posterior's moments with ground truth
            if (
                self.options.get("truemean")
                and self.options.get("truecov")
                and np.all(np.isfinite(self.options.get("truemean")))
                and np.all(np.isfinite(self.options.get("truecov")))
            ):
                mubar_orig, sigma_orig = vp_real.moments(1e6, True, True)

                kl = kldiv_mvn(
                    mubar_orig,
                    sigma_orig,
                    self.options.get("truemean"),
                    self.options.get("truecov"),
                )
                sKL_true = 0.5 * np.sum(kl)
            else:
                sKL_true = None

            # Record moments in transformed space
            mubar, sigma = self.vp.moments(origflag=False, covflag=True)
            if len(self.optim_state.get("run_mean")) == 0 or len(
                self.optim_state.get("run_cov") == 0
            ):
                self.optim_state["run_mean"] = mubar.reshape(1, -1)
                self.optim_state["run_cov"] = sigma
                self.optim_state["last_run_avg"] = self.optim_state.get("N")
            else:
                Nnew = self.optim_state.get("N") - self.optim_state.get(
                    "last_run_avg"
                )
                wRun = self.options.get("momentsrunweight") ** Nnew
                self.optim_state["run_mean"] = wRun * self.optim_state.get(
                    "run_mean"
                ) + (1 - wRun) * mubar.reshape(1, -1)
                self.optim_state["run_cov"] = (
                    wRun * self.optim_state.get("run_cov") + (1 - wRun) * sigma
                )
                self.optim_state["last_run_avg"] = self.optim_state.get("N")

            timer.stop_timer("finalize")
            # timer.totalruntime = NaN;   # Update at the end of iteration
            # timer

            # store current gp in vp
            self.vp.gp = gp

            iteration_values = {
                "iter": iteration,
                "optim_state": self.optim_state,
                "vp": self.vp,
                "elbo": elbo,
                "elbo_sd": elbo_sd,
                "varss": varss,
                "sKL": sKL,
                "sKL_true": sKL_true,
                "gp": gp,
                "gp_hyp_full": gp.get_hyperparameters(as_array=True),
                "Ns_gp": Ns_gp,
                "pruned": pruned,
                "timer": timer,
                "func_count": self.function_logger.func_count,
                "lcbmax": self.optim_state["lcbmax"],
                "n_eff": self.optim_state["n_eff"],
            }

            # Record all useful stats
            self.iteration_history.record_iteration(
                iteration_values,
                iteration,
            )

            # Check warmup
            if (
                self.optim_state.get("iter") > 1
                and self.optim_state.get("stop_gp_sampling") == 0
                and not self.optim_state.get("warmup")
            ):
                if self._is_gp_sampling_finished():
                    self.optim_state[
                        "stop_gp_sampling"
                    ] = self.optim_state.get("N")

            # Check termination conditions
            (
                is_finished,
                termination_message,
                success_flag,
            ) = self._check_termination_conditions()

            # Save stability
            self.vp.stats["stable"] = self.iteration_history["stable"][
                iteration
            ]

            # Check if we are still warming-up
            if self.optim_state.get("warmup") and iteration > 0:
                if self.options.get("recomputelcbmax"):
                    self.optim_state["lcbmax_vec"] = self._recompute_lcbmax().T
                trim_flag = self._check_warmup_end_conditions()
                if trim_flag:
                    self._setup_bads_after_warmup()
                    # Re-update GP after trimming
                    gp = reupdate_gp(self.function_logger, gp)
                if not self.optim_state.get("warmup"):
                    self.vp.optimize_mu = self.options.get("variablemeans")
                    self.vp.optimize_weights = self.options.get(
                        "variableweights"
                    )

                    # Switch to main algorithm options
                    # options = options_main
                    # Reset GP hyperparameter covariance
                    # hypstruct.runcov = []
                    hyp_dict["runcov"] = None
                    # Reset VP repository (not used in python)
                    self.optim_state["vp_repo"] = []

                    # Re-get acq info
                    # self.optim_state['acqInfo'] = getAcqInfo(
                    #    options.SearchAcqFcn
                    # )
            # Needs to be below the above block since warmup value can change
            # in _check_warmup_end_conditions
            self.iteration_history.record(
                "warmup", self.optim_state.get("warmup"), iteration
            )

            # Check and update fitness shaping / output warping threshold
            if (
                self.optim_state.get("outwarp_delta") != []
                and self.optim_state.get("R") is not None
                and (
                    self.optim_state.get("R")
                    < self.options.get("warptolreliability")
                )
            ):
                Xrnd, _ = self.vp.sample(N=int(2e4), origflag=False)
                ymu, _ = gp.predict(Xrnd, add_noise=True)
                ydelta = max(
                    [0, self.function_logger.ymax - np.quantile(ymu, 1e-3)]
                )
                if (
                    ydelta
                    > self.optim_state.get("outwarp_delta")
                    * self.options.get("outwarpthreshtol")
                    and self.optim_state.get("R") is not None
                    and self.optim_state.get("R") < 1
                ):
                    self.optim_state["outwarp_delta"] = self.optim_state.get(
                        "outwarp_delta"
                    ) * self.options.get("outwarpthreshmult")

            # Write iteration output
            # Stopped GP sampling this iteration?
            if (
                Ns_gp == self.options["stablegpsamples"]
                and self.iteration_history["Ns_gp"][max(0, iteration - 1)]
                > self.options["stablegpsamples"]
            ):
                if Ns_gp == 0:
                    self.logging_action.append("switch to GP opt")
                else:
                    self.logging_action.append("stable GP sampling")

            if self.options.get("plot") and iteration > 0:
                self._log_column_headers()

            if self.optim_state["cache_active"]:
                self.logger.info(
                    display_format.format(
                        iteration,
                        self.function_logger.func_count,
                        self.function_logger.cache_count,
                        elbo,
                        elbo_sd,
                        sKL,
                        self.vp.K,
                        self.optim_state["R"],
                        "".join(self.logging_action),
                    )
                )

            else:
                if (
                    self.optim_state["uncertainty_handling_level"] > 0
                    and self.options.get("maxrepeatedobservations") > 0
                ):
                    self.logger.info(
                        display_format.format(
                            iteration,
                            self.function_logger.func_count,
                            self.optim_state["N"],
                            elbo,
                            elbo_sd,
                            sKL,
                            self.vp.K,
                            self.optim_state["R"],
                            "".join(self.logging_action),
                        )
                    )
                else:
                    self.logger.info(
                        display_format.format(
                            iteration,
                            self.function_logger.func_count,
                            elbo,
                            elbo_sd,
                            sKL,
                            self.vp.K,
                            self.optim_state["R"],
                            "".join(self.logging_action),
                        )
                    )
            self.iteration_history.record(
                "logging_action", self.logging_action, iteration
            )

            # Plot iteration
            if self.options.get("plot"):
                if iteration > 0:
                    previous_gp = self.iteration_history["vp"][
                        iteration - 1
                    ].gp
                    # find points that are new in this iteration
                    # (hacky cause numpy only has 1D set diff)
                    # future fix: active sampling should return the set of
                    # indices of the added points
                    highlight_data = np.array(
                        [
                            i
                            for i, x in enumerate(self.vp.gp.X)
                            if tuple(x) not in set(map(tuple, previous_gp.X))
                        ]
                    )
                else:
                    highlight_data = None

                if len(self.logging_action) > 0:
                    title = "BADS iteration {} ({})".format(
                        iteration, "".join(self.logging_action)
                    )
                else:
                    title = "BADS iteration {}".format(iteration)

                self.vp.plot(
                    plot_data=True,
                    highlight_data=highlight_data,
                    plot_vp_centres=True,
                    title=title,
                )
                plt.show()

        # Pick "best" variational solution to return
        self.vp, elbo, elbo_sd, idx_best = self.determine_best_vp()

        # Last variational optimization with large number of components
        self.vp, elbo, elbo_sd, changed_flag = self.finalboost(
            self.vp, self.iteration_history["gp"][idx_best]
        )

        if changed_flag:
            # Recompute symmetrized KL-divergence
            sKL = max(
                0,
                0.5
                * np.sum(
                    self.vp.kldiv(
                        vp2=vp_old,
                        N=Nkl,
                        gaussflag=self.options.get("klgauss"),
                    )
                ),
            )

            if self.options.get("plot"):
                self._log_column_headers()

            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                self.logger.info(
                    display_format.format(
                        np.Inf,
                        self.function_logger.func_count,
                        self.optim_state["N"],
                        elbo,
                        elbo_sd,
                        sKL,
                        self.vp.K,
                        self.iteration_history.get("rindex")[idx_best],
                        "finalize",
                    )
                )
            else:
                self.logger.info(
                    display_format.format(
                        np.Inf,
                        self.function_logger.func_count,
                        elbo,
                        elbo_sd,
                        sKL,
                        self.vp.K,
                        self.iteration_history.get("rindex")[idx_best],
                        "finalize",
                    )
                )

        # plot final vp:
        if self.options.get("plot"):
            self.vp.plot(
                plot_data=True,
                highlight_data=highlight_data,
                plot_vp_centres=True,
                title="BADS final ({} iterations)".format(iteration),
            )
            plt.show()

        # Set exit_flag based on stability (check other things in the future)
        if not success_flag:
            if self.vp.stats["stable"]:
                success_flag = True
        else:
            if not self.vp.stats["stable"]:
                success_flag = False

        # Print final message
        self.logger.warning(termination_message)
        self.logger.warning(
            "Estimated ELBO: {:.3f} +/-{:.3f}.".format(elbo, elbo_sd)
        )
        if not success_flag:
            self.logger.warning(
                "Caution: Returned variational solution may have"
                + " not converged."
            )

        result_dict = self._create_result_dict(idx_best, termination_message)

        return (
            copy.deepcopy(self.vp),
            self.vp.stats["elbo"],
            self.vp.stats["elbo_sd"],
            success_flag,
            result_dict,
        )
        
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
        output["funccount"] = self.function_logger.func_count
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
                " Iteration f-count/f-cache    Mean[ELBO]     Std[ELBO]     "
                + "sKL-iter[q]   K[q]  Convergence    Action"
            )
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                self.logger.info(
                    " Iteration   f-count (x-count)   Mean[ELBO]     Std[ELBO]"
                    + "     sKL-iter[q]   K[q]  Convergence  Action"
                )
            else:
                self.logger.info(
                    " Iteration  f-count    Mean[ELBO]    Std[ELBO]    "
                    + "sKL-iter[q]   K[q]  Convergence  Action"
                )

    def _setup_logging_display_format(self):
        """
        Private method to set up the display format for logging the iterations.
        """
        if self.optim_state["cache_active"]:
            display_format = " {:5.0f}     {:5.0f}  /{:5.0f}   {:12.2f}  "
            display_format += (
                "{:12.2f}  {:12.2f}     {:4.0f} {:10.3g}       {}"
            )
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                display_format = " {:5.0f}       {:5.0f} {:5.0f} {:12.2f}  "
                display_format += (
                    "{:12.2f}  {:12.2f}     {:4.0f} {:10.3g}     "
                )
                display_format += "{}"
            else:
                display_format = " {:5.0f}      {:5.0f}   {:12.2f} {:12.2f} "
                display_format += "{:12.2f}     {:4.0f} {:10.3g}     {}"

        return display_format
