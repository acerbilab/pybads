import logging
from abc import ABC, abstractclassmethod
from typing import Callable

import numpy as np
import scipy
from gpyreg.gaussian_process import GP

from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb
from pybads.function_logger import FunctionLogger
from pybads.function_logger.constraints_check import contraints_check
from pybads.rng import get_rng
from pybads.rounding import round_half_away

from .grid_functions import force_to_grid


class ESSearch(ABC):
    """An Abstract class describing an Evolutionary Strategy Search.

    Its random draws come from ``rng``, a ``numpy.random.Generator``; if
    ``None``, a generator is derived from NumPy's global random state
    (``pybads.rng.get_rng``).
    """

    def __init__(self, mu, lamb, options_dict, rng=None):
        self.rng = get_rng(rng)
        self.mu = mu
        self.lamb = lamb
        self.vec = np.array([-1, 0])
        self.w = (
            options_dict["poll_mesh_multiplier"] ** self.vec
        )  # helps with the stability
        # MATLAB's round (searchES.m), which takes halves away from zero
        self.ns = np.diff(
            round_half_away(
                np.linspace(0, self.mu, np.size(self.w) + 1)
            ).astype(int)
        )

        self.vec = np.empty((0, 1), dtype="float")
        for i in range(0, len(self.w)):
            self.vec = np.append(
                self.vec, self.w[i] * np.ones((self.ns[i], 1)), axis=0
            )

        self.scale = options_dict["es_start"]
        self.n_search_iter = options_dict["n_search_iter"]
        self.search_acq_fcn = options_dict["search_acq_fcn"]
        self.es_beta = options_dict["es_beta"]
        self.logger = logging.getLogger("BADS")

    def _get_selection_idx_mask_(self, mu, lamb):
        """
        Corresponds to esupdate of the  BADS Matlab version
        """

        tot = mu + lamb
        sqrt_tot = np.sqrt(np.arange(1, tot + 1))
        w = np.ceil((1.0 / sqrt_tot) / np.sum((1.0 / sqrt_tot)) * lamb).astype(
            int
        )
        nonzero = np.sum(w > 0)
        while (np.sum(w) - lamb) > nonzero:
            w = np.maximum(0, w - 1)
            nonzero = np.sum(w > 0)
        delta = np.sum(w) - lamb
        lastnonzero = (np.argwhere(w > 0)[-1]).item()
        strt_point = np.maximum(0, lastnonzero - int(delta.item()) + 1).item()
        w[strt_point : lastnonzero + 1] = w[strt_point : lastnonzero + 1] - 1

        # Create selection mask: parent k, 0-based, repeated w[k] times, which
        # is MATLAB's 1-based selectmask minus one
        select_mask = np.repeat(np.arange(len(w)), w)

        return select_mask

    @abstractclassmethod
    def _initialize_(self, u, gp: GP, optim_state, sum_rule):
        """
        Get the covariance matrix and initialize internal variables
        """

        return None

    def __call__(
        self,
        u,
        lb: np.ndarray,
        ub: np.ndarray,
        func_logger: FunctionLogger,
        gp: GP,
        optim_state,
        sum_rule=True,
        non_box_cons: Callable = None,
    ):
        """Run the evolution-strategy search from the incumbent.

        Parameters
        ----------
        u : np.ndarray
            The incumbent.
        lb : np.ndarray
            The lower bounds of the search.
        ub : np.ndarray
            The upper bounds of the search.
        func_logger : FunctionLogger
            The function logger, whose evaluated points are removed from the
            candidates.
        gp : GP
            The local Gaussian process, which ranks the candidates by their
            acquisition value.
        optim_state : dict
            The optimization state.
        sum_rule : bool, optional
            ES-wcm normalizes the eigenvalues of its covariance by their sum
            if ``True``, and by the largest otherwise; ES-ell ignores it.
        non_box_cons : callable, optional
            The non-box constraints, which remove the candidates that
            violate them.

        Raises
        ------
        ValueError
            If the search's acquisition function is not ``'acq_LCB'``.

        Returns
        -------
        u_search : np.ndarray
            The best candidate of every generation, or an empty array when no
            candidate is left.
        z : np.ndarray
            Its acquisition value, or an empty array.
        """

        self.mesh_size = optim_state["mesh_size"]
        self.search_factor = optim_state["search_factor"]
        self.search_mesh_size = optim_state["search_mesh_size"]
        self.tol_mesh = optim_state["tol_mesh"]

        U = gp.X
        nvars = U.shape[1]

        self.sqrt_sigma = self._initialize_(u, gp, optim_state, sum_rule)

        # Rescale by current scale
        self.sqrt_sigma = self.mesh_size * self.search_factor * self.sqrt_sigma

        N = int(self.mu)
        u_new = u + self.vec * (
            self.rng.normal(size=(N, nvars)) @ self.sqrt_sigma
        )

        # TODO add check rotate gp flag

        # The candidates kept and their acquisition values: each generation
        # sets both, and with no generation the search returns the empty set,
        # as MATLAB's searchES does
        us = np.empty((0, nvars))
        z = np.empty(0)
        # Loop over evolutionary strategies iterations
        for i in range(0, self.n_search_iter):
            # TODO: enforce periodicity

            # Force candidates points on search grid
            u_new = force_to_grid(u_new, self.search_mesh_size)

            # Remove already evaluated or unfeasible points from search set
            u_new = contraints_check(
                u_new,
                optim_state["lb_search"],
                optim_state["ub_search"],
                optim_state["tol_mesh"],
                func_logger,
                True,
                non_box_cons,
            )

            if self.search_acq_fcn[0] == "acq_LCB":
                z_new, fmu, fs = acq_fcn_lcb(
                    u_new, func_logger.func_count, gp, self.search_acq_fcn[1]
                )
                z_new = z_new.flatten()
            else:
                raise ValueError(
                    "es_search: No acquisition function found for the Search "
                    "phase"
                )

            # TODO: handle other acqs fcns: acqNegEIMin, acqNegPIMi

            # No candidate left in this generation: it adds none, and the
            # candidates of the earlier generations are kept, as in MATLAB
            if u_new.shape[0] == 0:
                self.logger.debug(
                    f"bads:es_search: No candidate left in generation {i + 1} "
                    "of the search, once the points already evaluated or "
                    "violating the constraints are removed"
                )

            # Candidates kept before this generation (none before the first)
            nold = us.shape[0] if i > 0 else 0
            if i == 0:
                us_candidates = u_new.copy()
                z_candidates = z_new.copy()
            else:
                us_candidates = np.append(
                    us_candidates, u_new, axis=0
                )  # nsearch_iter and self.lambd decides us size
                z_candidates = np.append(z_candidates, z_new, axis=0)

            N = np.minimum(us_candidates.shape[0], self.lamb)

            # Order candidates and select
            z_idx = np.argsort(z_candidates, kind="stable")
            # New candidates among the best ntest, as in MATLAB's searchES:
            # the pool is not trimmed, and this generation's are its last rows
            ntest = np.minimum(u_new.shape[0], nold)
            n_new = np.sum(
                z_idx[0:ntest] >= us_candidates.shape[0] - u_new.shape[0]
            )
            z = z_candidates[z_idx[0:N]]
            us = us_candidates[z_idx[0:N]]  # zlist in Matlab is not used

            if us.shape[0] == 0:
                break  # no candidate left to reproduce

            if i < self.n_search_iter - 1:
                # Update scale parameter, unless this generation added no
                # candidate (MATLAB's fraction is then 0/0)
                if i > 0 and ntest > 0:
                    frac = n_new / ntest
                    self.scale = self.scale * np.exp(
                        self.es_beta * (frac - 0.2)
                    )

                # Reproduce
                selection_mask = self._get_selection_idx_mask_(
                    us.shape[0], self.lamb
                )
                ll = np.minimum(self.lamb, us.shape[0])

                u_new = (
                    us[selection_mask[0:ll]]
                    + (self.rng.normal(size=(ll, nvars)) @ self.sqrt_sigma)
                    * self.scale
                )

        # No candidate left: an empty set, as MATLAB's searchES returns
        if us.shape[0] == 0:
            return us, z
        return us[0], z[0]


class ESSearchWM(ESSearch):
    def __init__(self, mu, lamb, options_dict, rng=None):
        super().__init__(mu, lamb, options_dict, rng)
        self.frac = 0.5

    # Ovveride abstract method
    def _initialize_(self, u, gp: GP, optim_state, sum_rule):
        # Small jitter added to each direction
        self.jit = self.get_jitter(optim_state)

        U = gp.X
        Y = gp.y.flatten()
        # Compute vector weights
        mu = self.frac * U.shape[0]

        weights = np.log(mu + 0.5) - np.log(np.arange(1, np.floor(mu + 1)))
        weights = weights / np.sum(weights)

        # Compute best vectors
        y_idx = np.argsort(Y, kind="stable")
        idx_sel = (y_idx[0 : np.floor(mu).astype(int)]).flatten()
        Ubest = U[idx_sel].copy()

        # Compute the covariance matrix wrt u0: the unweighted scatter of the
        # best vectors, since the weights, which sum to one, do not weight
        # it, as in MATLAB's ucov.m
        C = ucov(
            Ubest,
            u,
            weights,
            optim_state["ub"],
            optim_state["lb"],
            optim_state["scale"],
            optim_state["periodic_vars"],
        )

        # Rescale covariance matrix according to mean vector length
        eig_values, E = scipy.linalg.eigh(C)
        eig_values = np.maximum(0, eig_values) + self.jit**2
        if sum_rule:
            eig_values = eig_values / np.sum(eig_values)
        else:
            eig_values = eig_values / np.max(eig_values)

        # Square root of covariance matrix
        sqrt_sigma = np.diag(np.sqrt(eig_values)) @ np.transpose(E)
        return sqrt_sigma

    def get_jitter(self, optim_state):
        return optim_state["mesh_size"]


class ESSearchELL(ESSearch):
    def _initialize_(self, u, gp: GP, optim_state, sum_rule):
        rescaled_len_scale = gp.temporary_data["poll_scale"]
        rescaled_len_scale = rescaled_len_scale / np.sqrt(
            np.sum(rescaled_len_scale**2)
        )
        sqrt_sigma = np.diag(rescaled_len_scale)

        # TODO add check rotate gp flag -> sqrt_sigma =

        return sqrt_sigma


def ucov(U, u, w, ub, lb, scale, periodic_vars=None):
    width_scaled = (ub - lb) / scale
    U_tmp = U.copy()
    u_tmp = u.copy()
    if periodic_vars is not None and np.any(periodic_vars):
        U_tmp[:, periodic_vars] = (
            U[:, periodic_vars]
            - u[periodic_vars]
            + 0.5 * width_scaled[periodic_vars]
        )
        U_tmp[:, periodic_vars] = (
            np.mod(U[:, periodic_vars], width_scaled[periodic_vars])
            - 0.5 * width_scaled[periodic_vars]
        )
        u_tmp[periodic_vars] = 0.0

    u_shift = U_tmp - u_tmp

    if w.size != 0:
        # Each weight times the whole scatter, summed: the scatter times the
        # sum of the weights (one for ES-wcm's), not weighted, as in MATLAB's
        # ucov.m
        weights = w.reshape(-1, *([1] * u_shift.ndim))
        C = np.matmul(u_shift.transpose(), weights * u_shift)
        C = np.sum(C, axis=0)
    else:
        C = u_shift.T @ u_shift

    return C
