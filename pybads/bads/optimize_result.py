import copy
import logging
from importlib.metadata import PackageNotFoundError, version

import numpy as np


class OptimizeResult(dict):
    """
    It represents the optimization result.
    The class is based on ``scipy.optimize.OptimizeResult``.
    See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html.

    Attributes:

        - fun: callable
            - The objective function to be minimized, the object passed to
              ``BADS``.
        - non_box_cons: callable
            - Non-box constraints function (if any), the object passed to
              ``BADS``.
        - x0: np.ndarray
            - Initial starting point, as given or drawn at random, with the
              fixed variables at their values.
        - x: np.ndarray
            - The solution of the optimization.
        - fval: float
            - Value of objective function at solution. In a noisy run, an
              estimate of the target's mean at ``x``: normally the average
              of ``yval_vec``, weighted by their precisions with
              ``specify_target_noise``.
        - fsd: float
            - Uncertainty (SD) of ``fval`` as an estimate of the target's
              value at ``x``, such as the standard error of the final
              samples in a noisy run; 0 for a deterministic target.
        - yval_vec: np.ndarray or None
            - In a noisy run, the observations of the target at ``x`` that
              ``fval`` averages; None otherwise.
        - ysd_vec: np.ndarray or None
            - Standard deviations of the final sampled observations
              (``"yval_vec"``) that the target returns with
              ``specify_target_noise``; None otherwise, and when no final
              sample was taken.
        - mesh_size: float
            - Final mesh size.
        - func_count: int
            - Number of evaluations of the objective functions.
              The evaluations made before the run
              (``precomputed_evaluations``) are not counted.
        - precomputed_observations: int
            - Number of evaluations made before the run that ``BADS`` was
              given (``precomputed_evaluations``); present only when it was
              given at least one.
        - precomputed_locations: int
            - Number of distinct points among them; present with
              ``precomputed_observations``.
        - iterations: int
            - Number of iterations performed by the optimizer.
        - success: bool
            - True when the run converged, on ``tol_mesh`` or the stall
              criterion (``status`` 1 or 2); False when it stopped on
              ``max_fun_evals``, ``max_iter`` or ``output_fcn``
              (``status`` 0).
        - status: int
            - The criterion that ended the run: 0 for ``max_fun_evals``,
              ``max_iter`` or ``output_fcn``; 1 when the mesh size fell
              below ``tol_mesh``; 2 when the improvement over the last
              ``tol_stall_iters`` iterations fell below ``tol_fun``.
        - message: str
            - Termination message.
        - problem_type: str
            - Type of problem (unconstrained, bound constraints, non-box constraints).
        - target_type: str
            - ``"deterministic"``, ``"stochastic"`` for a noisy target
              whose noise BADS infers, or ``"stochastic (specified
              noise)"`` with ``specify_target_noise``.
        - total_time: float
            - Time taken by ``optimize()``, in seconds; the setup made when
              ``BADS`` is created is not counted.
        - overhead: float
            - The optimizer's own time relative to the time spent
              evaluating the target: ``total_time`` divided by the
              evaluations' time, minus 1.
        - random_seed: int or None
            - The ``random_seed`` option if it is an integer, and ``None``
              otherwise.
        - algorithm: str
            - ``"Bayesian adaptive direct search"``.
        - version: str
            - Version of the optimizer.

    Parameters:
        bads: pybads.BADS
            - An Instance of the BADS class. It is used to set the attributes of the optimization result.
    """

    _keys = [
        "x",
        "x0",  # Initial starting point
        "success",
        "status",
        "message",
        "fun",
        "func_count",  # Number of evaluations of the objective functions
        "precomputed_observations",  # Evaluations made before the run
        "precomputed_locations",  # Their distinct points
        "iterations",  # Number of iterations performed by the optimizer.
        "target_type",
        "problem_type",
        "mesh_size",
        "non_box_cons",  # non_box_constraint function
        "yval_vec",
        "ysd_vec",
        "fval",
        "fsd",
        "total_time",
        "overhead",
        "random_seed",
        "algorithm",
        "version",
    ]

    def __init__(self, bads=None):
        super().__init__()
        if bads is not None:
            self.set_attributes(bads)

    def set_attributes(self, bads):
        """Set the attributes of the dictionary.

        Parameters:
            - bads: pybads.BADS
                An Instance of the BADS class. It is used to set the attributes of the optimization result.
        """
        self["fun"] = bads.function_logger.fun
        self["non_box_cons"] = bads.non_box_cons
        if bads.optim_state["uncertainty_handling_level"] > 0:
            if bads.options["specify_target_noise"]:
                self["target_type"] = "stochastic (specified noise)"
            else:
                self["target_type"] = "stochastic"
        else:
            self["target_type"] = "deterministic"

        if (
            np.all(np.isinf(bads.lower_bounds))
            and np.all(np.isinf(bads.upper_bounds))
            and bads.non_box_cons is None
        ):
            self["problem_type"] = "unconstrained"
        elif bads.non_box_cons is None:
            self["problem_type"] = "bound constraints"
        else:
            self["problem_type"] = "non-box constraints"

        # optim_state["iter"] counts from 0, and is -1 during initialization
        self["iterations"] = bads.optim_state["iter"] + 1
        self["func_count"] = bads.function_logger.func_count
        # As PyVBMC reports them, only for a run given evaluations
        if bads.optim_state["precomputed_observations"] > 0:
            self["precomputed_observations"] = bads.optim_state[
                "precomputed_observations"
            ]
            self["precomputed_locations"] = bads.optim_state[
                "precomputed_locations"
            ]
        self["mesh_size"] = bads.mesh_size
        self["overhead"] = bads.optim_state["overhead"]
        self["algorithm"] = "Bayesian adaptive direct search"
        if (
            bads.optim_state["uncertainty_handling_level"] > 0
            and bads.options["noise_final_samples"] > 0
        ):
            self["yval_vec"] = bads.optim_state["yval_vec"].copy()
        else:
            self["yval_vec"] = None

        if (
            bads.options["specify_target_noise"]
            and bads.options["noise_final_samples"] > 0
        ):
            self["ysd_vec"] = bads.optim_state["ysd_vec"]
        else:
            self["ysd_vec"] = None

        self["x0"] = bads.x0.copy()
        self["x"] = bads.x.copy()
        self["fval"] = bads.fval
        self["fsd"] = bads.fsd
        self["total_time"] = bads.optim_state["total_time"]

        self["random_seed"] = bads.optim_state["random_seed"]
        self["status"] = bads.optim_state["exit_flag"]

        try:
            __version__ = version("pybads")
        except PackageNotFoundError:
            # package is not installed
            __version__ = None
            logger = logging.getLogger("BADS")
            logger.warning("Cannot read version number from package metadata.")

        self["version"] = __version__

        # A positive exit flag, the convention of MATLAB and scipy: False
        # when a limit (max_fun_evals, max_iter) or output_fcn ends the run
        self["success"] = self["status"] > 0
        self["message"] = bads.optim_state["termination_msg"]

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __setitem__(self, key: str, val: object):
        if key not in OptimizeResult._keys:
            raise ValueError("""The key is not part of OptimizeResult._keys""")
        elif key in ("fun", "non_box_cons"):
            # The callables are kept by reference: a copy of a bound method
            # or a callable object copies its instance, which may hold what
            # cannot be copied (a lock, an open file)
            dict.__setitem__(self, key, val)
        else:
            dict.__setitem__(self, key, copy.deepcopy(val))
