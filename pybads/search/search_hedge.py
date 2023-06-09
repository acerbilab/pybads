import numpy as np
from gpyreg.gaussian_process import GP
from scipy.special import erfc

from .es_search import ESSearchELL, ESSearchWM


class ESSearchHedge:
    """
    It performs a hedging search, that chooses between different evolution strategies.
    It tracks the record of cumulative improvements of all the search strategies according to the Hedge algorithm [1] (default).
    It currently handles two different search strategies: ES-wcm and ES-ell.

    Parameters
    ----------
    search_fcns : array of tuples
        Array of search strategies to use in the hedge search.
    options_dict : dict
        Options for the hedge search
    non_box_cons: callable function
        A given non-bound constraints function. e.g : lambda x: np.sum(x.^2, 1) > 1

    ----------
    References
    [1]. Hoffman, M. D., Brochu, E., & de Freitas, N. (2011). Portfolio Allocation for Bayesian Optimization. In *UAI* (pp. 327-336). ([link](https://pdfs.semanticscholar.org/1a7f/d7b566697c9b69e64b27b68db4384314d925.pdf))
    [2]. Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003). Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES). *Evolutionary Computation*, **11**(1), 1-18. ([link](https://www.lri.fr/~hansen/evco_11_1_1_0.pdf))

    """

    def __init__(
        self,
        search_fcns=[("ES-wcm", 1), ("ES-ell", 1)],
        options_dict=None,
        non_box_cons=None,
    ):

        self.search_fcns = search_fcns
        self.n_funs = len(search_fcns)
        self.g = np.zeros(self.n_funs)
        self.g[0] = 10
        self.count = -1
        self.options_dict = options_dict
        self.non_box_cons = non_box_cons

        self.gamma = options_dict["hedge_gamma"]
        self.beta = options_dict["hedge_beta"]
        self.decay = options_dict["hedge_decay"]

        # Create vector of ES weights (for ESSearch)
        es_iter = self.options_dict["n_search_iter"]
        self.mu = int(self.options_dict["n_search"] / es_iter)
        self.lamb = self.mu

    def __call__(self, u, lb, ub, func_logger, gp: GP, optim_state):
        self.count += 1
        # gamma = np.minimum(1/n_funs, np.sqrt(np.log(n_funs)/(n_funs*count)));

        self.prob = np.exp(self.beta * (self.g - np.max(self.g)))
        self.prob = self.prob / np.sum(
            np.exp(self.beta * (self.g - np.max(self.g)))
        )
        self.prob = self.prob * (1 - self.n_funs * self.gamma) + self.gamma

        rand_uni = np.random.rand()
        self.chosen_hedge = np.argwhere(rand_uni < np.cumsum(self.prob))[0]
        if len(self.chosen_hedge) == 0:
            self.chosen_hedge = np.random.randint(0, self.n_funs)

        if self.gamma == 0:
            self.phat = np.ones(self.g.shape)
        else:
            self.phat = np.full(self.g.shape, np.inf)
            self.phat[self.chosen_hedge] = self.prob[self.chosen_hedge]

        self.chosen_search_fun = self.search_fcns[self.chosen_hedge.item()]
        if self.chosen_search_fun[0] == "ES-wcm":
            search = ESSearchWM(self.mu, self.lamb, self.options_dict)
            us, z = search(
                u,
                lb,
                ub,
                func_logger,
                gp,
                optim_state,
                self.chosen_search_fun[1],
                self.non_box_cons,
            )
            return us, z
        elif self.chosen_search_fun[0] == "ES-ell":
            search = ESSearchELL(self.mu, self.lamb, self.options_dict)
            us, z = search(
                u,
                lb,
                ub,
                func_logger,
                gp,
                optim_state,
                self.chosen_search_fun[1],
                self.non_box_cons,
            )
            return us, z
        else:
            raise ValueError(
                "search_hedge:Requested search method not implemented yet"
            )

    def update_hedge(self, u_search, fval_old, f, fs, gp: GP, mesh_size):
        """
        Update the probability of improvement which will be used for updating the weight of the hedge strategy
        """

        for i_hedge in range(self.n_funs):

            u_hedge = u_search[np.minimum(i_hedge, len(u_search) - 1) :].copy()

            if i_hedge == self.chosen_hedge:
                f_hedge = f
                fs_hedge = fs
            elif self.gamma == 0:
                f_hedge, fs_hedge = gp.predict(u_hedge)
                fs_hedge = np.sqrt(fs_hedge)
            else:
                f_hedge = 0
                fs_hedge = 1

            if fs_hedge == 0:
                er = np.maximum(0, fval_old - f_hedge)
            elif (
                np.isfinite(f_hedge)
                and np.isfinite(fs_hedge)
                and np.isreal(fs_hedge)
                and fs_hedge > 0
            ):
                # Probability of improvement
                gamma_z = (fval_old - f_hedge) / fs_hedge
                fpi = 0.5 * erfc(-gamma_z / np.sqrt(2))

                # Expected reward
                er = fs_hedge * (
                    gamma_z * fpi
                    + np.exp(-0.5 * (gamma_z**2) / np.sqrt(2 * np.pi))
                )
            else:
                er = 0

            self.g[i_hedge] = (
                self.decay * self.g[i_hedge]
                + er / self.phat[i_hedge] / mesh_size
            )
