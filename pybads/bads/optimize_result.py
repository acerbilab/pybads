import copy

from importlib.metadata import version, PackageNotFoundError
import logging

import numpy as np

class OptimizeResult(dict):
    """
    It represents the optimization result.
    
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
        - yval_vec: np.ndarray
            - Final sampled observations at the solution.
        - ysd_vec: np.ndarray
            - Standard deviations of the final sampled observations (``"yval_vec"``).
        - mesh_size: float
            - Final mesh size.
        - func_count: int
            - Number of evaluations of the objective functions.
        - iterations: int
            - Number of iterations performed by the optimizer.
        - message: str
            - Termination message.
        - problem_type: str
            - Type of problem (unconstrained, bound constraints, non-box constraints).
        - total_time: float
            - Total time taken by the optimizer.
        - overhead: float
            - Fractional overhead taken by the optimizer, compared to function time.
        - random_seed: int
            - Random seed used by the optimizer (``None`` if not set).
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

        self["iterations"] = bads.optim_state["iter"]
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

        try:
            __version__ = version("pybads")
        except PackageNotFoundError:
            # package is not installed
            __version__ = None
            logger = logging.getLogger("BADS")
            logger.warning("Cannot read version number from package metadata.")
            
        self["version"] = __version__
        
        self[
            "success"
        ] = True  # TODO: In our case when an error occurs, the application just stops.
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
