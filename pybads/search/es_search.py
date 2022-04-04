
from typing import final
import numpy as np

from gpyreg.gaussian_process import GP
from abc import ABC, abstractclassmethod
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb
from pybads.utils import ucheck
from pybads.function_logger.function_logger import FunctionLogger

from pybads.search.grid_functions import force_to_grid

class SearchES(ABC):

    def __init__(self, mu, lamb, options_dict):
        self.mu = mu
        self.lamb = lamb
        self.vec = np.array[-1, 0]
        self.w = options_dict["pollmeshmultiplier"]**self.vec #helps with the stability
        self.ns = np.diff(np.round(np.linspace(0,self.mu, np.size(self.w)+1)))

        self.vec = np.empty((0,1), dtype='float')
        for i in range(0, len(self.w)):
            self.vec = np.append(self.w[i] * np.ones((self.ns[i], 1)), axis = 0)

        self.scale = options_dict["esstart"]
        self.n_search_iter = options_dict["nsearchiter"]
        self.search_acq_fcn = options_dict["searchacqfcn"]
        self.es_beta = options_dict["esbeta"]


    def _get_selection_mask_(self, mu, lamb):
        """
            Corresponds to esupdate of the Matlab version.
        """

        tot = mu + lamb
        sqrt_tot = np.sqrt(np.arange(1, tot+1))
        w = np.ceil((1. / sqrt_tot) / np.sum((1. / sqrt_tot)) * lamb).astype(int)
        nonzero = np.sum(w > 0)
        while (np.sum(w) - lamb) > nonzero:
            w = np.maximum(0, w-1)
            nonzero = np.sum(w > 0)
        delta = np.sum(w) - lamb
        lastnonzero = (np.argwhere(w > 0)[-1]).item()
        strt_point = np.maximum(0, lastnonzero - int(delta.item()) +1 ).item()
        w[strt_point : lastnonzero] = w[strt_point: lastnonzero] - 1 

        # Create selection mask
        cw = np.cumsum(w) - w 
        idx = np.zeros(np.max(cw)+1, dtype=int)
        idx[cw] = 1
        select_max = np.cumsum(idx[0:-1])

        return select_max

    @abstractclassmethod
    def _initialize_(self, u, gp:GP, optim_state, sum_rule):
        """
            Get the covariance matrix and initialize internal variables
        """
        
        return None

    def __call__(self, u, lb:np.ndarray, ub:np.ndarray, func_logger:FunctionLogger, gp:GP, optim_state, sum_rule=True):
        
        self.mesh_size = optim_state['mesh_size']
        self.search_factor = optim_state['search_factor']
        self.search_mesh_size = optim_state["search_mesh_size"]
        self.tol_mesh = optim_state["tol_mesh"]

        U = gp.X
        Y = gp.y
        nvars = len(U)

        self.sqrt_sigma = self._initialize_(u, gp, optim_state, sum_rule)

        # Rescale by current scale
        self.sqrt_sigma = self.mesh_size * self.search_factor * self.sqrt_sigma

        N = self.mu
        u_new = u + self.vec * np.random.normal(size=(N, nvars)) * self.sqrt_sigma

        # TODO add check rotate gp flag

        us_rows = np.minimum(u_new.shape[0], self.lamb)
        us = np.empty((us_rows, u_new.shape[1]))
        z = np.empty((us_rows, u_new.shape[1])) #TODO: test shapes by passing different mu and lambda with Matlab!
        # Loop over evolutionary strategies iterations
        for i in range(0, self.n_search_iter):

            u_new = force_to_grid(u_new, self.search_mesh_size)

            # TODO: 
            u_new = ucheck()

            if self.search_acq_fcn == 'acqLCB':
                z_new, fmu, fs, _ = acq_fcn_lcb(u_new, func_logger, gp, optim_state, False)

            # TODO: handle other acqs fcns: acqNegEIMin, acqNegPIMi
            
            nold = us.shape[0]
            if i == 0:
                us_candidates = u_new    
                z_candidates = np.random.rand(u_new.shape[0], 1)
            else:
                us_candidates = np.concatenate(us_candidates, u_new, axis = 0) # nsearch_iter and self.lambd decides us size
                z_candidates = np.append(z_candidates, z_new, axis = 0)
            
            N = np.minimum(us_candidates.shape[0], self.lamb)

            # Order candidates and select
            z_idx = np.argsort(z_candidates)
            ntest = np.minimum(u_new.shape[0], nold)
            n_new = np.sum(z_idx[0:ntest+1] > nold)
            z = z_candidates[z_idx[0:N]]
            us = us_candidates[z_idx[0:N]] #zlist in Matlab is not used

            if i < self.n_search_iter-1:
                frac = n_new / ntest
                # Update scale parameter
                if i > 0:
                    self.scale = self.scale * np.exp(self.es_beta * (frac -0.2 ))
                
                # Reproduce
                es_iter = 0
                selection_mask = self._get_selection_mask_(us.shape[0], self.lambd)
                ll = np.minimum(self.lamb, us.shape[0])

                u_new = us[selection_mask[0:ll]] + np.random.normal(size=(ll, nvars)) * self.sqrt_sigma * self.scale

        return us, z


class SearchESWM(SearchES):

    def __init__(self, mu, lamb, options_dict):
        super().__init__(mu, lamb, options_dict)
        self.active_flag = False
        self.frac = 0.5

    # Ovveride abstract method
    def _initialize_(self, u, gp:GP, optim_state, sum_rule):
        # Small jitter added to each direction
        self.jit = self.get_jitter(optim_state)

        U = gp.X
        Y = gp.y
        # Compute vector weights
        nvars = len(U)
        mu = self.frac * U.shape[0]
        weights = np.zeros((1, 1, np.floor(mu)))
        weights[0, 0, ] = np.log(mu + 0.5) - np.log(np.arange(1, np.floor(mu+1)))
        weights = weights/np.sum(weights)

        # Compute best vectors
        y_idx = np.np.argsort(Y)
        Ubest = U[y_idx[0:np.floor(mu+1)]]

        # Compute weighted covariance matrix wrt u0
        C = ucov(Ubest, u, weights, optim_state["ub"], optim_state["lb"], optim_state["scale"], optim_state["periodicvars"])
        if self.active_flag:
            U_worst = U[y_idx[-1:-1: (len(y_idx) - np.floor(mu)+1)]] #TODO check indexes better, critique point
            negC = ucov(U_worst, u, weights, optim_state) #TODO heeeere
            negmueff = np.sum(1./weights**2)
            negcov = 0.25 * negmueff / ((nvars+2)**1.5 + 2*negmueff)
            C = C - negcov * negC

        # Rescale covariance matrix according to mean vector length
        eig_values, E = np.linalg.eig(C)
        eig_values = np.diag(np.maximum(0,eig_values)) + self.jit**2
        if sum_rule:
            eig_values = eig_values/np.sum(eig_values)
        else:
            eig_values = eig_values/np.max(eig_values)
        
        # Square root of covariance matrix
        sqrt_sigma = np.diag(np.sqrt(eig_values)) @ np.transpose(E)
        return sqrt_sigma

    def get_jitter(self, optim_state):
        return optim_state["mesh_size"]


class SearchESCMA(SearchESWM):
    def __init__(self, mu, lamb, options_dict):
        super().__init__(mu, lamb, options_dict)
        self.active_flag = True
        self.frac = 0.25

    def get_jitter(self, optim_state):
        return optim_state["search_mesh_size"]


class SearchESELL(SearchES):

    def _initialize_(self, u, gp:GP, optim_state, sum_rule):
        rescaled_len_scale = gp.temporary_data["pollscale"]
        rescaled_len_scale = rescaled_len_scale / np.sqrt(np.sum(rescaled_len_scale**2))
        sqrt_sigma = np.diag(rescaled_len_scale)
        # TODO add check rotate gp flag -> sqrt_sigma = 

        return sqrt_sigma

def ucov(U, u, w, ub, lb, scale, periodic_vars):
    if w.size==0:
        weights = 1
    else:
        weights = w.reshape((1, 1, w.size))
    
    width_scaled = (ub - lb) / scale

    U[:, periodic_vars] = U[:, periodic_vars] - u[periodic_vars] + 0.5 * width_scaled[periodic_vars]
    U[:, periodic_vars] = np.mod(U[:, periodic_vars], width_scaled[periodic_vars]) - 0.5 * width_scaled[periodic_vars]
    u[periodic_vars] = 0.0

    u_shift = U - u
    C = np.sum(weights * (u_shift.T @ u_shift), axis=2)

    return C

    
