import copy
import logging
import os
import sys
from asyncio.log import logger

import matplotlib.pyplot as plt
import numpy as np
from gpyreg.gaussian_process import GP
from scipy.special import erfc, erfcinv, gammaincinv
from scipy.stats import shapiro

from pybads.acquisition_functions import acq_fcn_lcb
from pybads.function_logger import FunctionLogger, contraints_check
from pybads.init_functions import init_sobol
from pybads.poll import poll_mads_2n
from pybads.search import ESSearchHedge
from pybads.search.grid_functions import force_to_grid, grid_units, udist
from pybads.utils import period_check
from pybads.utils.iteration_history import IterationHistory
from pybads.utils.timer import Timer
from pybads.variable_transformer import VariableTransformer

from .gaussian_process_train import (
    add_and_update_gp,
    init_and_train_gp,
    local_gp_fitting,
)
from .optimize_result import OptimizeResult
from .options import Options


class BADS:
    """
    BADS Constrained optimization using Bayesian Adaptive Direct Search.
    
    BADS attempts to solve problems of the form:
       :math:`\mathtt{argmin}_x  f(x)`  subject to:  lower_bounds :math:`<= x <=` upper_bounds, and optionally :math:`C(x) <= 0`
                               

    Initialize a ``PyBADS`` object to set up the optimization problem, then run
    ``optimize()``. See the examples for more details under the `examples` directory.

    Parameters
    ----------
    fun : callable
        A given target ``fun``. ``fun`` accepts input ``x`` and returns a scalar 
        function value of the target evaluated at ``x`` and the noise if provided.
        In case the target function ``fun`` requires additional data/parameters, 
        they can be handled using an anonymous function.
        For example: ``fun_for_pybads = lambda x: fun(x, data, extra_params)``, 
        where ``fun`` is the function to optimize, and ``data`` and ``extra_params`` 
        are given in the outer scope.
    x0 : np.ndarray, optional
        Starting point for the optimization. If not specified or ``None``, the 
        starting point ``x0`` is uniformly randomly drawn inside the plausible 
        box between ``plausible_lower_bounds`` and ``plausible_upper_bounds`` (see 
        below).
    lower_bounds, upper_bounds : np.ndarray, optional
        ``lower_bounds`` (``lb``) and ``upper_bounds`` (``ub``) define a set
        of strict lower and upper bounds for the coordinate vector, ``x``, so
        that the unknown function has support on ``lb`` <= ``x`` <= ``ub``.
        If scalars, the bound is replicated in each dimension. Use
        ``None`` for ``lb`` and ``ub`` if no bounds exist. Set ``lb[i] = -inf``
        and ``ub [i] = inf`` if the `i`-th coordinate is unbounded (while
        other coordinates may be bounded). Note that if ``lb`` and ``ub`` contain
        unbounded variables, the respective values of ``plb`` and ``pub`` need to
        be specified (see below). By default ``None``.
    plausible_lower_bounds, plausible_upper_bounds : np.ndarray, optional
        Specifies a set of ``plausible_lower_bounds`` (``plb``) and
        ``plausible_upper_bounds`` (``pub``) such that ``lb`` <= ``plb`` < ``pub`` <= ``ub``.
        Both ``plb`` and ``pub`` need to be finite. ``plb`` and ``pub`` represent a
        `plausible` range, which should denote a region where the global minimum
        is expected to be found. As a rule of thumb, set ``plausible_lower_bounds`` 
        and ``plausible_upper_bounds`` such that there is > 90% probability that 
        the minimum is found within the box (where in doubt, just set 
        ``plb = lb`` and ``pub = ub``).

    non_box_cons: callable, optional
        A given non-box constraints function that specifies constraint 
        `violations`, e.g : ``lambda x: np.sum(x.^2,1)>1``

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
        determine at runtime if the objective function is noisy.
        To obtain reproducible results of the optimization, set ``options['random_seed']`` 
        to a fixed integer value.

    Raises
    ------
    ValueError
        When neither ``x0`` or (``plausible_lower_bounds`` and
        ``plausible_upper_bounds``) are specified.
    ValueError
        When various checks for the bounds (``lower_bounds``, ``upper_bounds``, 
        ``plausible_lower_bounds``, ``plausible_upper_bounds``) of BADS fail.


    References
    ----------
    .. [1]  Acerbi, L. & Ma, W. J. (2017). "Practical Bayesian
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
        gamma_uncertain_interval=None,
        options: dict = None,
    ):
        # set up root logger (only changes stuff if not initialized yet)
        logging.basicConfig(stream=sys.stdout, format="%(message)s")

        self.non_box_cons = non_box_cons

        # variable to keep track of logging actions
        self.logging_action = []

        # Initialize variables and algorithm structures
        if plausible_lower_bounds is None and lower_bounds is not None:
            plausible_lower_bounds = lower_bounds.copy()
        if plausible_upper_bounds is None and upper_bounds is not None:
            plausible_upper_bounds = upper_bounds.copy()

        if x0 is None:
            if (
                plausible_lower_bounds is None
                or plausible_upper_bounds is None
            ):
                raise ValueError(
                    """bads:UnknownDims If no starting point is
                 provided, plausible_lower_bounds and plausible_upper_bounds need to be specified."""
                )
            else:
                x0 = np.full((plausible_lower_bounds.shape), np.NaN)

        x0 = np.atleast_2d(x0)
        self.D = x0.shape[1]

        # load basic and advanced options and validate the names
        pybads_path = os.path.dirname(os.path.realpath(__file__))
        basic_path = pybads_path + "/option_configs/basic_bads_options.ini"
        self.options = Options(
            basic_path,
            evaluation_parameters={"D": self.D},
            user_options=options,
        )
        advanced_path = (
            pybads_path + "/option_configs/advanced_bads_options.ini"
        )
        self.options.load_options_file(
            advanced_path,
            evaluation_parameters={"D": self.D},
        )
        self.options.validate_option_names([basic_path, advanced_path])

        if self.options["stobads"] is None or self.options["stobads"] == False:
            self.options["stobads"] = False

        # set up random seed
        self._init_random_seed_()

        # set up BADS logger
        self.logger = logging.getLogger("BADS")
        self.logger.setLevel(logging.INFO)
        if self.options.get("display") == "off":
            self.logger.setLevel(logging.WARN)
        elif self.options.get("display") == "iter":
            self.logger.setLevel(logging.INFO)
        elif self.options.get("display") == "full":
            self.logger.setLevel(logging.DEBUG)

        # Empty lb and ub are Infs
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
        ) = self._bounds_check_(
            x0.copy(),
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
            non_box_cons,
        )

        self.gamma_uncertain_interval = gamma_uncertain_interval

        # starting point
        if not np.all(np.isfinite(self.x0)):
            self.x0 = np.random.uniform(
                low=self.plausible_lower_bounds,
                high=self.plausible_upper_bounds,
                size=(1, self.D),
            )
            self.logger.log(
                25,
                "Initial starting point is invalid or not provided."
                + " Initial point randomly sampled uniformly from plausible box\n",
            )

        # evaluate  starting point non-bound constraint
        if non_box_cons is not None:
            if non_box_cons(self.x0) > 0:
                self.logger.error(
                    "Initial starting point X0 does not satisfy non-bound constraints (non_box_cons)."
                )
                raise ValueError(
                    "Initial starting point X0 does not satisfy non-bound constraints (non_box_cons)."
                )

        self.optim_state = self._init_optim_state_()

        # create and init the function logger
        self.function_logger = FunctionLogger(
            fun=fun,
            D=self.D,
            noise_flag=self.optim_state.get("uncertainty_handling_level") > 0,
            uncertainty_handling_level=self.optim_state.get(
                "uncertainty_handling_level"
            ),
            cache_size=self.options.get("cache_size"),
            variable_transformer=self.var_transf,
        )

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
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        non_box_cons: callable = None,
    ):
        """
        Private function for initial checks of the BADS bounds.
        """

        N0, D = x0.shape

        # Estimation of the plb and pub if any of them is not specified
        if plausible_lower_bounds is None or plausible_upper_bounds is None:
            if N0 > 1:
                self.logger.warning(
                    "plausible_lower_bounds and/or plausible_upper_bounds not specified. Estimating"
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
                    "bads:pbUnspecified: Plausible lower/upper bounds"
                    " not specified and X0 is not a valid starting set. "
                    + "Using hard upper/lower bounds instead."
                )
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = np.copy(lower_bounds)
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = np.copy(upper_bounds)

        # ensure at least 2d dimensions
        upper_bounds = np.atleast_2d(upper_bounds)
        lower_bounds = np.atleast_2d(lower_bounds)
        plausible_upper_bounds = np.atleast_2d(plausible_upper_bounds)
        plausible_lower_bounds = np.atleast_2d(plausible_lower_bounds)
        # check that all bounds are row vectors with D elements
        upper_bounds = np.atleast_2d(upper_bounds)
        lower_bounds = np.atleast_2d(lower_bounds)
        plausible_upper_bounds = np.atleast_2d(plausible_upper_bounds)
        plausible_lower_bounds = np.atleast_2d(plausible_lower_bounds)
        if (
            lower_bounds.shape != (1, D)
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
                "Plausible interval bounds plausible_lower_bounds and plausible_upper_bounds need to be finite."
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
        fix_idx = (
            (lower_bounds == upper_bounds)
            & (upper_bounds == plausible_lower_bounds)
            & (plausible_lower_bounds == plausible_upper_bounds)
        )
        if np.any(fix_idx):
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
                points X0 are not inside the provided hard bounds lower_bounds and upper_bounds."""
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
                """bads:StrictBoundsTooClose: Hard bounds lower_bounds and upper_bounds
                are numerically too close. Make them more separate."""
            )

        # Fix when provided X0 are almost on the bounds -- move them inside
        if np.any(x0 < LB_eff) or np.any(x0 > UB_eff):
            self.logger.warning(
                "bads:InitialPointsTooClosePB: The starting points X0 are on "
                + "or numerically too close to the hard bounds lower_bounds and upper_bounds. "
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
            plausible bounds should respect the ordering lower_bounds < plausible_lower_bounds < plausible_upper_bounds < upper_bounds."""
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
                + " are not inside the provided plausible bounds (plausible_lower_bounds and plausible_upper_bounds)."
                + " Expanding the plausible bounds..."
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
            plausible bounds should respect the ordering lower_bounds <= plausible_lower_bounds < plausible_upper_bounds <= upper_bounds."""
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
        if non_box_cons is not None:
            y = non_box_cons(
                np.vstack([plausible_lower_bounds, plausible_upper_bounds])
            )
            if y.shape[0] != 2 and y.ndim == 1:
                raise ValueError(
                    "bads:NONBCON "
                    + "NONBCON should be a function that takes a matrix X as input"
                    + " and returns a column vector of bound violations."
                )

        # Gentle warning for infinite bounds
        ninfs = np.sum(np.isinf(np.concatenate([lower_bounds, upper_bounds])))
        if ninfs > 0:
            if ninfs == 2 * D:
                self.logger.warning(
                    "Detected fully unconstrained optimization."
                )
            else:
                self.logger.warning(f"Detected {ninfs} infinite bound(s).")

        return (
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

    def _init_optim_state_(self):
        """
        A private function to initialize the optim_state dict that contains information about BADS variables.
        """
        # Record starting points (original coordinates)
        if self.options["f_vals"] is not None:
            y_orig = np.array(self.options.get("f_vals")).flatten()
            if len(y_orig) == 0:
                y_orig = np.full([self.x0.shape[0]], np.nan)
            if len(self.x0) != len(y_orig):
                raise ValueError(
                    """bads:MismatchedStartingInputs The number of
                points in X0 and of their function values as specified in
                self.options.['f_vals'] are not the same."""
                )
        else:
            y_orig = np.full([self.x0.shape[0]], np.nan)

        optim_state = dict()
        optim_state["random_seed"] = self._random_seed
        optim_state["cache"] = dict()
        optim_state["cache"]["x_orig"] = self.x0
        optim_state["cache"]["y_orig"] = y_orig
        optim_state["last_re_eval"] = -np.inf

        # Does the starting cache contain function values?
        optim_state["cache_active"] = np.any(
            np.isfinite(optim_state.get("cache").get("y_orig"))
        )

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

        # Check if periodic_vars is not None and raise error, since it is not yet supported
        if self.options["periodic_vars"] is not None:
            raise ValueError(
                "Periodic variables are not yet supported. Please set periodic_vars to None."
            )

        # Compute transformation of variables
        if self.options["nonlinear_scaling"]:
            logflag = np.full((1, self.D), np.NaN)
            periodic_vars = self.options["periodic_vars"]
            if periodic_vars is not None and len(periodic_vars) != 0:
                logflag[
                    :, periodic_vars
                ] = 0  # Never transform periodic variables
        else:
            logflag = np.zeros((1, self.D))

        self.var_transf = VariableTransformer(
            self.D,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
            logflag,
        )
        # optim_state["variables_trans"] = var_transf

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
        
        # Check that the gridized points satisfies the non-bound constraints
        if self.non_box_cons is not None and \
            np.any(self.non_box_cons(self.var_transf.inverse_transf(u0)) > 0):
            self.logger.error(
                """Initial starting point X0 does no longer satisfy non-bound constraint after being fit into the mesh grid."""
            )
            raise ValueError(
                """Initial starting point X0 does no longer satisfy non-bound constraint after being fit into the mesh grid."""
            )
        
        optim_state["u"] = u0
        self.u = u0.flatten().copy()

        # Test starting point u0 is within bounds
        if np.any(u0 > self.upper_bounds) or np.any(u0 < self.lower_bounds):
            self.logger.error(
                "Initial starting point u0 is not within the hard bounds lower_bounds and upper_bounds"
            )
            raise ValueError(
                """bads:Initpoint: Initial starting point u0 is not within the hard bounds lower_bounds and upper_bounds"""
            )

        # Report variable transformation
        if np.any(self.var_transf.apply_log_t):
            self.logger.info(
                f"Variables (index) internally transformed to log coordinates: {np.argwhere(self.var_transf.apply_log_t)}"
            )

        # Put tol_mesh on space
        optim_state["tol_mesh"] = self.options[
            "poll_mesh_multiplier"
        ] ** np.ceil(
            np.log(self.options["tol_mesh"])
            / np.log(self.options["poll_mesh_multiplier"])
        )

        # Periodic variables
        idx_periodic_vars = self.options["periodic_vars"]
        periodic_vars = np.zeros((1, self.D)).astype(bool)
        if idx_periodic_vars is not None:
            periodic_vars[:, idx_periodic_vars] = True
            finite_periodic_vars = np.all(
                np.isfinite(self.lower_bounds[:, idx_periodic_vars])
            ) and np.all(np.isfinite(self.upper_bounds[:, idx_periodic_vars]))
            if not finite_periodic_vars:
                raise ValueError(
                    "bads:InitOptimState:Periodic variables need to have finite lower and upper bounds."
                )
            self.logger.info(
                f"Variables (index) defined with periodic boundaries: {idx_periodic_vars}"
            )
        optim_state["periodic_vars"] = periodic_vars

        # Setup covariance information (unused)

        # Import prior function evaluations
        fun_values = self.options["fun_values"]
        if fun_values is not None and len(fun_values) != 0:
            if "X" not in fun_values or "Y" not in fun_values:
                raise ValueError(
                    """bads:fun_values: The 'fun_values' field in options need to have X and Y fields (respectively, inputs and their function values)"""
                )

            X = fun_values["X"]
            Y = fun_values["Y"]

            if len(X) != len(Y):
                raise ValueError(
                    "X and Y arrays in the options['fun_values'] need to have the same number of rows (each row is a tested point)."
                )
            if (
                (not np.all(np.isfinite(X)))
                or (not np.all(np.isfinite(Y)))
                or (not np.isreal(X))
                or (not np.isreal(Y))
            ):
                raise ValueError(
                    "X and Y arrays need to be finite and real-valued"
                )
            if len(X) != 0 and X.shape[1] != self.D:
                raise ValueError(
                    "X should be a matrix of tested points with the same dimensionality as X0 (one input point per row)."
                )

            if len(Y) != 0 and Y.shape[1] != 1:
                raise ValueError(
                    "Y should be a vertical nd-array (, 1) of function values (one function value per row)."
                )

            S = None
            if "S" in fun_values:
                S = fun_values["S"]
                if len(S) != len(Y):
                    raise ValueError(
                        "X, Y, and S arrays in the options['fun_values'] need to have the same number of rows (each row is a tested point)."
                    )
                S = np.atleast_2d(S).T
                if len(S) != 0 and S.shape[1] != 1:
                    raise ValueError(
                        "S should be a vertical nd-array (, 1) of estimated function SD values (one SD per row)."
                    )

            for i in range(len()):
                if S is None:
                    self.function_logger.add(X[i], Y[i])
                else:
                    self.function_logger.add(X[i], Y[i], S[i])

        # Other variables initializations
        optim_state["search_factor"] = 1
        optim_state["sd_level"] = self.options["incumbent_sigma_multiplier"]
        optim_state["search_count"] = self.options[
            "search_n_try"
        ]  # Skip search at first iteration
        optim_state["lastreeval"] = -np.inf
        # Last time function values were re-evaluated
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

        # Copy maximum number of fcn. evaluations,
        # used by some acquisition fcns.
        optim_state["max_fun_evals"] = self.options.get("max_fun_evals")

        # Deal with user specified target noise
        if self.options["specify_target_noise"] is None:
            self.options["specify_target_noise"] = False

        if (
            self.options["specify_target_noise"]
            and self.options["uncertainty_handling"] is None
        ):
            self.options["uncertainty_handling"] = False

        if (
            self.options["specify_target_noise"]
            and self.options["uncertainty_handling"] is not None
            and self.options["uncertainty_handling"] == False
        ):
            raise ValueError(
                "If options['specify_target_noise'] is True, options['uncertainty_handling'] should be True as well. \
                                Leave options['uncertainty_handling'] empty or set it to True to avoid this error."
            )
        if (
            self.options["specify_target_noise"]
            and self.options["noise_size"] is not None
            and np.array(self.options["noise_size"] > 0)[0]
        ):
            self.logger.warn(
                "If options['specify_target_noise'] is True, options['noise_size'] is ignored. \
                Leave options['noise_size'] empty or set it to 0 to silence this warning."
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
        # Squared exponential kernel with separate length scales
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
        valid_gp_mean_funs = [
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

        if not optim_state["gp_mean_fun"] in valid_gp_mean_funs:
            raise ValueError(
                """bads:UnknownGPmean:Unknown/unsupported GP mean
            function. Supported mean functions are zero, const,
            egquad, and se"""
            )
        optim_state["int_meanfun"] = self.options.get("gpintmeanfun")

        return optim_state

    def _init_random_seed_(self):
        # set random seed if provided
        if "random_seed" in self.options and \
            self.options["random_seed"] is not None:
            # set random seed to numpy and consequently to scipy (scipy uses the same number generator)
            random_seed = int(self.options["random_seed"])
            np.random.seed(random_seed)
        else:
            random_seed = None
        self._random_seed = random_seed
        return random_seed

    def _init_mesh_(self):
        """
        A private function to initialize the mesh frame and the optimization problem.
        It evaluates the initial points, which includes the starting point and the generated point retrieved from a sobol sequence generating method.
        The init_mesh also assess if the target function is stochastic and set the parameter of BADS for handling stochastic targets.
        """
        # Evaluate starting point and initial mesh, determine if function is noisy
        self.yval, self.fsd, _ = self.function_logger(self.u)
        if self.fsd is None:
            self.fsd = np.nan
        self.fval = self.yval
        self.optim_state["fval"] = self.fval
        self.optim_state["yval"] = self.yval

        if self.optim_state["uncertainty_handling_level"] < 1:
            # test if the function is noisy
            self.logging_action.append("Uncertainty test")
            yval_bis, _, _ = self.function_logger(
                self.u, record_duplicate_data=False
            )
            if np.abs(self.yval - yval_bis) > self.options["tol_noise"]:
                self.optim_state["uncertainty_handling_level"] = 1
                self.logging_action.append("Uncertainty test")
        else:
            self.logging_action.append("")

        if self.optim_state["uncertainty_handling_level"] > 0:
            if self.options["specify_target_noise"]:
                self.logger.info(
                    "Beginning optimization of a STOCHASTIC objective function (specified noise)\n"
                )
            else:
                self.logger.info(
                    "Beginning optimization of a STOCHASTIC objective function\n"
                )
        else:
            self.logger.info(
                "Beginning optimization of a DETERMINISTIC objective function\n"
            )

        # Only one function evaluation
        if self.options["max_fun_evals"] == 1:
            is_finished = True
            return

        # If dealing with a noisy function, use a large initial mesh
        if self.optim_state["uncertainty_handling_level"] > 0:
            self.options["fun_eval_start"] = np.minimum(
                np.maximum(20, self.options["fun_eval_start"]),
                self.options["max_fun_evals"],
            )

        # set up strings for logging of the iteration
        self.display_format = self._setup_logging_display_format()
        self._log_column_headers()
        self._display_function_log_(0, "")

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
                )
                # enforce periodicity TODO function
                u1 = period_check(
                    u1,
                    self.lower_bounds,
                    self.upper_bounds,
                    self.options["periodic_vars"],
                )

                # Force points to be in the search grid.
                u1 = force_to_grid(u1, self.optim_state["search_mesh_size"])

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

                for u_idx in range(len(u1)):
                    self.function_logger(u1[u_idx])

                idx_yval = np.argmin(
                    self.function_logger.Y[: self.function_logger.Xn + 1]
                )
                self.u = self.function_logger.X[idx_yval].copy()
                self.yval = self.function_logger.Y[idx_yval].item()
                self.fval = self.yval
                self.logging_action.append("Initial points")
                self._display_function_log_(0, "Initial mesh")
            else:
                raise ValueError(
                    "bads:init_fun:Initialization function not implemented yet"
                )

        if not np.isfinite(self.yval):
            raise ValueError("init mesh: Cannot find valid starting point.")

        self.optim_state["fval"] = self.fval
        self.optim_state["yval"] = self.yval

        # Save the efffective number of initial starting points, which can match with options['fun_eval_start']
        # but it might be different for example when considering a noisy target function OR when applying a non-box-contraint function.
        self.optim_state["eff_starting_points"] = self.function_logger.Xn + 1

        return

    def _init_optimization_(self):
        """
        A private function initialize the optimization problem.
        It calls the init_mesh, sets the option configurations required by BADS, and initializes the Guassian Process (GP)
        """
        gp = None
        self.reset_gp = False
        hyp_dict = {}
        # set random seed if provided
        self.optim_state["random_seed"] = self._init_random_seed_()
        
        # Evaluate starting point and initial mesh,
        self._init_mesh_()

        # Change options for uncertainty handling
        if self.optim_state["uncertainty_handling_level"] > 0:
            self.options["tol_stall_iters"] = 2 * self.options["tol_stall_iters"]
            self.options["n_train_max"] = max(200, self.options["n_train_max"])
            self.options["n_train_min"] = 2 * self.options["n_train_min"]
            self.options["mesh_overflow_warning"] = (
                2 * self.options["mesh_overflow_warning"]
            )
            self.options["min_failed_poll_steps"] = np.inf
            self.options["mesh_noise_multiplier"] = 0
            if self.options["noise_size"] is None:
                self.options["noise_size"] = 1.0
            if isinstance(
                self.options["noise_size"], np.ndarray
            ):  # ensure the noise_size is a scalar
                self.options["noise_size"] = self.options["noise_size"].item()

            # Keep some function evaluations for the final resampling
            self.options["noise_final_samples"] = min(
                self.options["noise_final_samples"],
                self.options["max_fun_evals"]
                - self.function_logger.func_count,
            )
            self.options["max_fun_evals"] = (
                self.options["max_fun_evals"]
                - self.options["noise_final_samples"]
            )

            # Specify the standard deviation of the function values
            # It corresponds to specify target noise of Matlab
            if self.optim_state["uncertainty_handling_level"] > 1:
                idx_min_y = np.argmin(
                    self.function_logger.Y[: self.function_logger.Xn + 1]
                ).item()
                self.fsd = self.function_logger.S[idx_min_y]
                self.fsd = self.fsd.item()
            else:
                self.fsd = self.options["noise_size"]

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

        # Initialize Gaussian Process (GP) structure
        gp, Ns_gp, sn2hpd, hyp_dict = init_and_train_gp(
            hyp_dict,
            self.optim_state,
            self.function_logger,
            self.iteration_history,
            self.options,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
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
        Run the optimization on an initialized ``PyBADS`` object.
        
        BADS starts at X0 and finds a local minimum X of the
        target function 'fun'.

        A history of the optimization problem can be found at the ``self.iteration_history`` variable of the ``PyBADS`` object.

        Returns
        ----------
            optimize_result: OptimizeResult
                Dictionary containing the result of the optimization. See the documentation of the ``OptimizeResult`` class for more details.
                For example, retrieve the final solution with the following attributes:
                    -  ``optimize_result.x``
                    -  ``optimize_result.fval`` 
        """
        is_finished = False
        poll_iteration = -1
        self.logging_action = []
        timer = Timer()
        timer.start_timer("BADS")
        hyp_dict = {}
        self.search_success = 0
        self.last_skipped = -1
        # Last skipped iteration
        self.search_spree = 0
        self.restarts = self.options["restarts"]

        # Initialize gp
        gp, Ns_gp, sn2hpd, hyp_dict = self._init_optimization_()
        
        self.search_es_hedge = None  # init search hedge to None

        if self.options["output_fcn"] is not None:
            output_fcn = self.options["output_fcn"]
            is_finished = output_fcn(
                self.var_transf.inverse_transf(self.u), "init"
            )

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
            ] = self.sufficient_improvement.copy()

            do_search_step_flag = (
                self.optim_state["search_count"] < self.options["search_n_try"]
                and len(self.function_logger.Y[self.function_logger.X_flag])
                > self.D
            )

            if do_search_step_flag:
                # Search stage
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
                            self.search_spree, self.options["search_mesh_expand"]
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

            # check and do poll step
            if do_poll_step:
                self._poll_step_(gp)

            # Finalize the iteration

            # TODO: Iteration plot
            if self.options["plot"] == "scatter":
                pass

            # GP hyperparameters at end of iteration
            self.best_gp_hyp = gp.get_hyperparameters(as_array=True)

            msg = ""
            # Check termination conditions
            if (
                self.function_logger.func_count
                >= self.options["max_fun_evals"]
            ):
                is_finished = True
                # exit_flag = 0
                msg = "Optimization terminated: reached maximum number of function evaluations options['max_fun_evals']."

            if poll_iteration >= self.options["max_iter"] - 1:
                is_finished = True
                # exit_flag = 0
                msg = "Optimization terminated: reached maximum number of iterations options['max_iter']."

            if self.optim_state["mesh_size"] < self.optim_state["tol_mesh"]:
                is_finished = True
                # exit_flag = 1
                msg = "Optimization terminated: change in the function value less than options['tol_mesh']"

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
                self.iteration_history.record("fsd", self.fsd, poll_iteration)
                self.iteration_history.record(
                    "mesh_size", self.mesh_size, poll_iteration
                )
                self.iteration_history.record(
                    "search_mesh_size", self.search_mesh_size, poll_iteration
                )
                self.iteration_history.record(
                    "gp_hyp_full", gp.get_hyperparameters(True), poll_iteration
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
                self._re_evaluate_history_(gp)
                self.yval = self.iteration_history.get("yval")[poll_iteration]
                self.fval = self.iteration_history.get("fval")[poll_iteration]
                self.fsd = self.iteration_history.get("fsd")[poll_iteration]
                self.best_gp_hyp = self.iteration_history.get("gp_hyp_full")[
                    poll_iteration
                ]
                gp = self.iteration_history.get("gp")[poll_iteration]

                f_q_re_impr = self._eval_improvement_(
                    self.fval,
                    self.iteration_history.get("fval").astype("float"),
                    self.fsd,
                    self.iteration_history.get("fsd").astype("float"),
                    self.options["improvement_quantile"],
                )
                f_q_re_impr = f_q_re_impr[1:]  # Skip the first iteration
                idx_impr = np.argmax(f_q_re_impr)
                improvement = f_q_re_impr[idx_impr]
                idx_impr = idx_impr + 1  # offset original index without skip

                # Check if any point got better
                if improvement > self.options["tol_fun"]:
                    self.yval = self.iteration_history.get("yval")[idx_impr]
                    self.fval = self.iteration_history.get("fval")[idx_impr]
                    self.fsd = self.iteration_history.get("fsd")[idx_impr]
                    self.u = self.iteration_history.get("u")[idx_impr]
                    self.best_u = self.u.copy()
                    self.best_gp_hyp = self.iteration_history.get(
                        "gp_hyp_full"
                    )[idx_impr]
                    gp = self.iteration_history.get("gp")[
                        idx_impr
                    ]  # overwrite best gp

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

        # Re-evaluate all best points for noisy evaluations
        yval_vec = self.yval if np.isscalar(self.yval) else self.yval.copy()
        if (
            self.optim_state["uncertainty_handling_level"] > 0
            and poll_iteration > 0
        ):
            self._re_evaluate_history_(gp)

            # Order by lowest probabilistic upper bound and choose
            # the point with the lowest quantile values of the history of the optimization run: inf{x: F(x)>p}.
            sigma_multiplier = np.sqrt(2) * erfcinv(
                2 * self.options["final_quantile"]
            )  # Using inverted convention
            q_beta = self.iteration_history.get(
                "fval"
            ) + sigma_multiplier * self.iteration_history.get("fsd")
            min_q_beta_idx = np.argmin(q_beta[1:])  # Skip first iteration
            min_q_beta_idx += 1  # offset original index with no skip
            self.yval = self.iteration_history.get("yval")[min_q_beta_idx]
            self.fval = self.iteration_history.get("fval")[min_q_beta_idx]
            self.fsd = self.iteration_history.get("fsd")[min_q_beta_idx]
            self.u = self.iteration_history.get("u")[min_q_beta_idx]
            self.u_best = self.u.copy()
            self.best_gp_hyp = self.iteration_history.get("gp_hyp_full")[
                min_q_beta_idx
            ]
            gp = self.iteration_history.get("gp")[min_q_beta_idx]

            # Re-evalate estimated function value and SD at final point
            if self.options["noise_final_samples"] > 0:
                # Estimate function value and standard deviation at final point.
                # Note that by default we do *not* use YVAL because it is biased
                # (since it was an incumbent at some iteration, it is more likely to be a
                # random fluctuation lower than the mean)
                yval_vec = np.empty(self.options["noise_final_samples"])
                ysd_vec = np.empty(self.options["noise_final_samples"])
                for i_sample in range(self.options["noise_final_samples"]):
                    y, y_sd, _ = self.function_logger(
                        self.u, record_duplicate_data=False
                    )
                    yval_vec[i_sample] = y
                    ysd_vec[i_sample] = y_sd

                if yval_vec.size == 1:
                    yval_vec = np.vstack((yval_vec, self.yval))
                    if self.options["specify_target_noise"]:
                        ysd_vec = np.vstack(
                            (
                                ysd_vec,
                                self.function_logger.S[
                                    self.function_logger.Xn
                                ],
                            )
                        )

                self.optim_state["yval_vec"] = np.copy(yval_vec)
                self.optim_state["ysd_vec"] = np.copy(ysd_vec)

                self.fval = np.mean(yval_vec).item()
                self.fsd = (np.std(yval_vec) / np.sqrt(yval_vec.size)).item()
                self.iteration_history.record(
                    "fval", self.fval, poll_iteration
                )
                self.iteration_history.record("fsd", self.fsd, poll_iteration)

        # Convert back to original space
        self.x = self.var_transf.inverse_transf(self.u)

        # Compute total running time and fractional overhead
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

        self.logger.info(msg)
        if self.optim_state["uncertainty_handling_level"] > 0:
            if np.isscalar(yval_vec) or yval_vec.size == 1:
                self.logger.info(
                    f"Observed function value at minimum: {yval_vec} (1 sample). Estimated: {self.fval} ± {self.fsd} (GP mean ± SEM)."
                )
            else:
                self.logger.info(
                    f"Estimated function value at minimum: {self.fval} ± {self.fsd} (mean ± SEM from {yval_vec.size} samples)"
                )
        else:
            self.logger.info(f"Function value at minimum: {self.fval}\n")

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
        u_search : np.ndarray
            Candidate search point.
        search_dist : np.ndarray
            Distance of the search point from thecurrent point.
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

        if (
            refit_flag
            or self.optim_state["search_count"] == 0
            or self.reset_gp
        ):
            # Local GP approximation on current incumbent
            gp, gp_exit_flag = local_gp_fitting(
                gp,
                self.u,
                self.function_logger,
                self.options,
                self.optim_state,
                self.iteration_history,
                refit_flag,
            )

            if refit_flag:
                self.gp_refitted_flag = True
            self.gp_exit_flag = np.minimum(self.gp_exit_flag, gp_exit_flag)
        # End fitting

        # Update Target from GP prediction
        f_target_mu, f_target_s, f_target = self._get_target_from_gp_(
            self.u_best, gp, self.best_gp_hyp
        )
        self.optim_state["f_target_mu"] = f_target_mu.item()
        self.optim_state["f_target_s"] = (
            f_target_s if np.isscalar(f_target_s) else f_target_s.copy()
        )
        self.optim_state["f_target"] = f_target.item()

        # Generate search set (normalized coordinate)
        self.optim_state["search_count"] += 1

        if self.search_es_hedge is None:
            self.search_es_hedge = ESSearchHedge(
                self.options["search_method"], self.options, self.non_box_cons
            )
        u_search_set, z = self.search_es_hedge(
            self.u,
            self.lower_bounds,
            self.upper_bounds,
            self.function_logger,
            gp,
            self.optim_state,
        )

        # Enforce periodicity
        u_search_set = period_check(
            u_search_set,
            self.lower_bounds,
            self.upper_bounds,
            self.optim_state["periodic_vars"],
        )

        # Force candidate points on search grid
        u_search_set = force_to_grid(
            u_search_set, self.optim_state["search_mesh_size"]
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
            # Batch evaluation of acquisition function on search set
            z, f_mu, fs = acq_fcn_lcb(u_search_set, self.function_logger.func_count, gp)
            # Evaluate best candidate point in original coordinates
            index_acq = np.argmin(z)

            # TODO: In future handle acquisition portfolio (Acquisition Hedge), it's not even unsupported in Matlab

            # Randomly choose index if something went wrong
            if (
                index_acq is None
                or index_acq.size < 1
                or np.any(~np.isfinite(index_acq))
            ):
                self.logger.warn("bads:optimize: Acquisition function failed")
                index_acq = np.random.randint(0, len(u_search_set) + 1)

            # u_search at the candidate acquisition point
            u_search = u_search_set[index_acq]

            # TODO: Local optimization of the acquisition function (generally it does not improve results)
            if self.options["search_optimize"]:
                pass

            y_search, f_sd_search, idx = self.function_logger(u_search)

            if z.size > 0:
                # Save statistics of gp prediction,
                self._save_gp_stats_(y_search, f_mu[index_acq].item(),
                                    fs[index_acq].item())

            # Add search point to training setMeshSize
            if (
                u_search.size
                > 0 & self.search_es_hedge.count
                < self.options["search_n_try"]
            ):
                # TODO: Handle fitness_shaping and rotate gp axes (latter one is unsupported)
                gp = add_and_update_gp(
                    self.function_logger,
                    gp,
                    u_search,
                    y_search,
                    f_sd_search,
                    self.options,
                )

                if np.any(~np.isfinite(gp.y)):
                    self.logger.warn("bads:opt: GP prediction is non-finite")

            # If the function is non-deterministic we update the posterior of the GP with the new point
            if self.optim_state["uncertainty_handling_level"] > 0:
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
                )
                f_mu_search, f_sd_search = new_gp.predict(
                    np.atleast_2d(u_search)
                )
                f_mu_search = f_mu_search.item()
                f_sd_search = np.sqrt(f_sd_search).item()
            else:
                f_mu_search = y_search
                f_sd_search = 0

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
            y_search = self.yval
            f_mu_search = self.fval
            f_sd_search = 0
            search_dist = 0

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
                self.gamma_uncertain_interval,
            )
            if self.options["opp_stobads"]:
                is_search_improved = sto_success > -1
                is_search_success = sto_success == 1
            else:
                is_search_improved = sto_success == 1
                is_search_success = is_search_improved

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

        # Update search portfolio (needs improvement)
        if self.search_es_hedge is not None and u_search_set.size > 0:
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
            self._display_function_log_(
                self.optim_state["iter"], search_string
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
        gamma_uncertain_interval=None,
    ):
        """
            A private method that evaluates if the improvement in the candidate incumbent using the uncertain interval method proposed in Sto-MADS [1].
        Returns
        ----------
            int : Return a flag integer value
                1   : sucessuful improvement
                0   : uncertain unsuccessful incumbent
                -1  : certain unsuccessful incumbent

        References
        ----------
        [1] Audet, Charles, Kwassi Joseph Dzahini, Michael Kokkolaras, and Sébastien Le Digabel. ‘Stochastic Mesh Adaptive Direct Search for Blackbox Optimization Using Probabilistic Estimates’. Computational Optimization and Applications 79, no. 1 (May 2021): 1–34. https://doi.org/10.1007/s10589-020-00249-0.
        """
        epsilon = np.sqrt(s_base**2 + s_new**2)
        mu = f_base - f_new
        if self.gamma_uncertain_interval is None:
            gamma = 1.96  # gamma = norminv(0.975)
        else:
            gamma = gamma_uncertain_interval  # gamma = norminv(0.975)

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
        A private method that performs poll step using the LTMADS poll direction method.
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
            Estimated GP variance at the best poll point.
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
        sto_success = 0
        B = None
        u_poll = None
        u_new = []

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
                )

                # GP- based vector scaling (poll_scale broadcast)
                vv = (
                    B_new * self.optim_state["mesh_size"]
                ) * gp.temporary_data[
                    "poll_scale"
                ]  # scaling again using broadcast

                # Add vector to current point, fix to grid
                u_poll_new = self.u + vv
                period_check(
                    u_poll_new,
                    self.lower_bounds,
                    self.upper_bounds,
                    self.optim_state["periodic_vars"],
                )

                if self.options["force_poll_mesh"]:
                    u_poll_new = force_to_grid(
                        u_poll_new, self.optim_state["search_mesh_size"]
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

                # Add new poll points to polling set
                if u_poll is None:
                    u_poll = u_poll_new.copy()
                else:
                    u_poll = np.vstack(u_poll, u_poll_new)

                if B is None:
                    B = B_new.copy()
                else:
                    B = np.vstack((B, B_new))

            # Cannot refill poll vector set, stop polling
            if u_poll is None or u_poll.size == 0:
                break

            # Check whether it is time to refit the GP
            refit_flag, do_gp_calibration = self._is_gp_refit_time_(
                self.options["normalpha_level"]
            )

            if (
                not self.options["poll_training"]
                and self.optim_state["iter"] > 0
            ):
                refit_flag = False

            # Local GP approximation around polled points
            if refit_flag or poll_count == 0 or self.reset_gp:
                gp, gp_exit_flag = local_gp_fitting(
                    gp,
                    self.u,
                    self.function_logger,
                    self.options,
                    self.optim_state,
                    self.iteration_history,
                    refit_flag,
                )
                if refit_flag:
                    self.gp_refitted_flag = True
                self.gp_exit_flag = np.minimum(self.gp_exit_flag, gp_exit_flag)

            # Update Target from GP prediction
            f_target_mu, f_target_s, f_target = self._get_target_from_gp_(
                u_poll_best, gp, gp_poll_hyp_best
            )
            self.optim_state["f_target_mu"] = f_target_mu.item()
            self.optim_state["f_target_s"] = (
                f_target_s if np.isscalar(f_target_s) else f_target_s.copy()
            )
            self.optim_state["f_target"] = f_target.item()

            # Evaluate acquisition function on poll vectors
            # Batch evaluation of acquisition function on search set (The Acquisition Hedge policy is not yet supported (even in Matlab))
            z, f_mu, fs = acq_fcn_lcb(u_poll, self.function_logger.func_count, gp)
            # Evaluate best candidate point in original coordinates
            index_acq = np.argmin(z)

            # In future handle acquisition portfolio (Acquisition Hedge), it's even unsupported in Matlab

            # Randomly choose index if something went wrong
            if (
                index_acq is None
                or index_acq.size < 1
                or np.any(~np.isfinite(index_acq))
            ):
                self.logger.warn("bads:optimize: Acquisition function failed")
                index_acq = np.random.randint(0, len(u_poll) + 1)
            if logging.getLogger().level > logging.DEBUG:
                np.seterr(divide="ignore")
            gamma_z = (
                self.optim_state["f_target"]
                - self.sufficient_improvement
                - f_mu
            ) / fs
            if np.all(np.isfinite(gamma_z)) and np.all(np.isreal(gamma_z)):
                f_pi = 0.5 * erfc(-gamma_z / np.sqrt(2))
                # sort descend
                f_pi = np.sort(f_pi)[::-1]
                p_less = np.prod(
                    1 - f_pi[0 : np.minimum(self.D + 1, len(f_pi))]
                )
            else:
                p_less = 0
                do_gp_calibration = True

            # Consider whether to stop polling
            if not self.options["complete_poll"]:
                # Stop polling if last poll was good
                if certain_good_poll:
                    if do_gp_calibration:
                        break  # GP is unreliable, just stop polling
                    elif p_less > 1 - self.options["tol_poi"]:
                        break  # Use GP prediction whether to stop polling
                else:
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
                        break

            # Evaluate function and store the value
            u_new = u_poll[index_acq]
            y_poll, y_sd_poll, f_idx_new = self.function_logger(u_new)

            # Remove polled vector from set.
            u_poll = np.delete(u_poll, index_acq, axis=0)

            # Save statistics of gp prediction
            self._save_gp_stats_(y_poll, f_mu[index_acq].item(),
                                fs[index_acq].item())

            if self.optim_state["uncertainty_handling_level"] > 0:
                # Update posterior with the new polled point
                gp = add_and_update_gp(
                    self.function_logger,
                    gp,
                    u_new,
                    y_poll,
                    y_sd_poll,
                    self.options,
                )  # u_new is already added from the function logger
                f_poll, f_sd_poll = gp.predict(np.atleast_2d(u_new))
                f_sd_poll = np.sqrt(f_sd_poll).item()
                f_poll = f_poll.item()
            else:
                f_poll = y_poll
                f_sd_poll = 0

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
                    self.gamma_uncertain_interval,
                )
                certain_good_poll = sto_success == 1

            # Increase poll counter
            poll_count += 1
        # End poll loop

        # Evaluate poll
        if not self.options["stobads"]:
            if (
                poll_best_improvement > 0 and self.options["sloppy_improvement"]
            ) or poll_best_improvement > self.sufficient_improvement:

                # Update incumbent point (self.yval, self.fval, self.fsd) and optim_state
                self._update_incumbent_(
                    u_poll_best, y_poll_best, f_poll_best, f_sd_poll_best
                )
                is_poll_moved = True
            else:
                is_poll_moved = False
        else:
            # StoBads
            if self.options["opp_stobads"] and sto_success > -1:
                self._update_incumbent_(
                    u_poll_best, y_poll_best, f_poll_best, f_sd_poll_best
                )
                is_poll_moved = True
            elif certain_good_poll:
                # Update incumbent point (self.yval, self.fval, self.fsd) and optim_state
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
                self.mesh_size_integer + 1, self.options["max_poll_grid_number"]
            )

            self.optim_state["u_success"].append(self.u_best.copy)
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
            # Check stalling
            iter = self.optim_state["iter"]
            if (
                self.options["accelerate_mesh"]
                and iter > self.options["accelerate_mesh_steps"]
            ):
                f_base = self.iteration_history.get("fval")[
                    iter - self.options["accelerate_mesh_steps"]
                ]
                f_sd_base = self.iteration_history.get("fsd")[
                    iter - self.options["accelerate_mesh_steps"]
                ]
                u_base = self.iteration_history.get("u")[
                    iter - self.options["accelerate_mesh_steps"]
                ]
                self.f_q_historic_improvement = self._eval_improvement_(
                    f_base,
                    self.fval,
                    f_sd_base,
                    self.fsd,
                    self.options["improvement_quantile"],
                )
                if (
                    self.f_q_historic_improvement < self.options["tol_fun"]
                ):  # or np.all(u_base.flatten() == self.u.flatten()):

                    self.mesh_size_integer -= 1
                    logger.debug(
                        "bads: The optimization is stalling, further decrease of the mesh size"
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

        if self.gp_refitted_flag:
            action_str = "Train"
            if self.gp_exit_flag < 0:
                action_str += " (failed)"
                # self.gp_exit_flag = np.inf # Reset the flag
            self.logging_action.append(action_str)

        if self.last_skipped == self.optim_state["iter"]:
            self.logging_action.append("Skip")

        self._display_function_log_(self.optim_state["iter"], poll_string)

        # TODO: if self.output_function is not None -> Implement output function for saving the result in a file.

        self.reset_gp = is_poll_moved

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

    def _is_gp_refit_time_(self, alpha):
        """A private method that checks the calibration of the GP prediction and if a fitting is required."""
        if self.function_logger.func_count < 200:
            refit_period = np.maximum(10, self.D * 2)
        else:
            refit_period = self.D * 5

        gp_iter_idx = self.gp_stats.get("iter_gp")

        do_gp_calibration = False
        # empty stats
        if (
            gp_iter_idx is None
            or len(gp_iter_idx) == 0
            or gp_iter_idx[-1] == 0
        ):
            if gp_iter_idx is None:
                gp_iter_idx = 0
            do_gp_calibration = True
        else:
            gp_iter_idx = gp_iter_idx[
                -1
            ]  # retrieve last recorded gp stat iteration

        # if stats data is available check z_score
        if not do_gp_calibration:
            f_vals = (
                self.gp_stats.get("fval")[: gp_iter_idx + 1]
                .flatten()
                .astype("float")
            )
            yvals = (
                self.gp_stats.get("ymu")[: gp_iter_idx + 1]
                .flatten()
                .astype("float")
            )
            
            zscore = f_vals - yvals
            gp_ys = (
                self.gp_stats.get("ys")[: gp_iter_idx + 1]
                .flatten()
                .astype("float")
            )
            
            # Avoid division by zero, sometimes the GP variance is zero (e.g at end of the optimization of a deterministic)
            idx_zero_gp_ys = np.where(np.isclose(0., gp_ys))[0]
            gp_ys[idx_zero_gp_ys] = 1e-6
            
            zscore = zscore / gp_ys

            if np.any(np.isnan(zscore)):
                do_gp_calibration = True
            else:
                n = np.size(zscore)
                if n < 3:
                    chi_to_inv = lambda y, v: gammaincinv(v / 2, y)
                    plo = chi_to_inv(alpha / 2, n)
                    phi = chi_to_inv(1 - alpha / 2, n)
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
                else:
                    shapiro_test = shapiro(zscore)
                    do_gp_calibration = shapiro_test.pvalue < alpha

        func_count = self.function_logger.func_count

        refit_flag = (
            self.optim_state["lastfitgp"]
            < (func_count - self.options["min_refit_time"])
            and (gp_iter_idx >= refit_period or do_gp_calibration)
            and func_count > self.D
        )

        if refit_flag:

            self.optim_state["lastfitgp"] = self.function_logger.func_count

            # Reset GP statistics GP
            self.gp_stats = IterationHistory(
                [
                    "iter_gp",
                    "fval",
                    "ymu",
                    "ys",
                    "gp",
                ]
            )
            do_gp_calibration = False

        return refit_flag, do_gp_calibration

    def _get_target_from_gp_(self, u, gp: GP, hyp_best):
        """A private method that retrieve the prediction of the gp at the input ``u``.
            If the target function is stochastic then set the optimization target ``f_target`` slightly below the mean prediction.

        Parameters
        ----------
            u : np.array
                input point u
            gp : GP
            hyp_best : np.ndarray
                Hyperparameter used by the GP in the prediction

        Returns:
            f_target_mu :
                GP prediction, it corresponds to the mean values.
            f_target_s :
                GP variance/noise at point u.
            f_target : optimization target, it is slighly below the GP prediction when the target function is stochastic.

        """
        # Corresponds to Matlab: updateTarget
        if (
            self.optim_state["uncertainty_handling_level"] > 0
            or self.options["uncertain_incumbent"]
        ):
            tmp_gp = copy.deepcopy(gp)
            tmp_gp.set_hyperparameters(hyp_best)
            f_target_mu, fs2 = tmp_gp.predict(np.atleast_2d(u))

            f_target_s = np.sqrt(np.max(fs2, axis=0))
            if (
                ~np.isfinite(f_target_mu)
                | ~np.isreal(f_target_s)
                | ~np.isfinite(f_target_s)
            ):
                f_target_mu = self.optim_state["fval"]
                f_target_s = self.optim_state["fsd"]

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
            f_target = self.optim_state["fval"] - self.options["tol_fun"]
            f_target_mu = self.optim_state["fval"]
            f_target_s = 0

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
            self.optim_state["search_factor"] = (
                self.optim_state["search_factor"]
                * self.options["search_scale_failure"]
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
        It updates the predicted values and the variance for each stored GP at each iteration, by computing the posterior on the current training set without refitting the parameters.
        """
        if self.optim_state["last_re_eval"] != self.function_logger.func_count:
            # Re-evaluate gp outputs
            u_history = self.iteration_history.get("u")
            gps = self.iteration_history.get("gp")
            for i in range(u_history.shape[0]):
                tmp_gp = gps[i]
                u = u_history[i]
                tmp_gp, _ = local_gp_fitting(
                    tmp_gp,
                    u,
                    self.function_logger,
                    self.options,
                    self.optim_state,
                    self.iteration_history,
                    False,
                )
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
                self.logger.warn(
                    "bads:meshOverflow \t The mesh attempted to expand above maximum size too many times. Try widening plausible_lower_bounds and plausible_upper_bounds."
                )

    def _log_column_headers(self):
        """
        Private method to log the column headers for the iteration log.
        """
        if self.optim_state["cache_active"]:
            self.logger.info(
                " Iteration f-count/f-cache     E[f(x)]     SD[f(x)]     MeshScale     Method       Actions"
            )
        else:
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
        if self.optim_state["cache_active"]:
            display_format = " {:5.0f}     {:5.0f}/{:5.0f}   {:12.6f}  "
            display_format += "{:12.6f}  {:12.6f}     {}       {}"
        else:
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
