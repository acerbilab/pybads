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
            - The objective function to be minimized.
        - non_box_cons: callable
            - Non-box constraints function (if any).
        - x0: np.ndarray
            - Initial starting point.
        - x: np.ndarray
            - The solution of the optimization.
        - fval: float
            - Value of objective function at solution.
        - fsd: float
            - Standard deviation of objective function at solution (0 if noiseless).
        - yval_vec: np.ndarray or None
            - Final sampled observations at the solution; the incumbent's
              observation alone if the run ends within its first iteration.
              None for a run without uncertainty handling or with
              ``noise_final_samples = 0``.
        - ysd_vec: np.ndarray or None
            - Standard deviations of the final sampled observations
              (``"yval_vec"``) that the target returns with
              ``specify_target_noise``; None otherwise, and when no final
              sample was taken.
        - mesh_size: float
            - Final mesh size.
        - func_count: int
            - Number of evaluations of the objective functions.
        - iterations: int
            - Number of iterations performed by the optimizer.
        - success: bool
            - True when the run ended on one of its convergence criteria,
              ``tol_mesh`` or the stall criterion (``status`` 1 or 2); False
              when ``max_fun_evals`` or ``max_iter`` ended it, the
              ``output_fcn`` stopped it or it ended in its initialization
              (``status`` 0): the convention of MATLAB's exit flags and of
              ``scipy.optimize``.
        - status: int
            - The exit flag of MATLAB BADS, the criterion that ended the
              run: 0 when it reached ``max_fun_evals`` or ``max_iter``, the
              ``output_fcn`` stopped it or it ended in its initialization; 1
              when the mesh size fell below ``tol_mesh``; 2 when the
              improvement over the last ``tol_stall_iters`` iterations fell
              below ``tol_fun``.
        - message: str
            - Termination message.
        - problem_type: str
            - Type of problem (unconstrained, bound constraints, non-box constraints).
        - total_time: float
            - Total time taken by the optimizer.
        - overhead: float
            - Fractional overhead taken by the optimizer, compared to function time.
        - random_seed: int or None
            - The ``random_seed`` option if it is an integer (a float that is a whole number is converted to one), and ``None`` otherwise.
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
        else:
            dict.__setitem__(self, key, copy.deepcopy(val))
