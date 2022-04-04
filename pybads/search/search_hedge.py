
from encodings import search_function
import numpy as np
from gpyreg.gaussian_process import GP
from .es_search import SearchESWM, SearchESELL

class SearchESHedge():

    def __init__(self, search_fcns, options_dict):
        
        self.search_fcns = search_fcns
        self.n_funs = len(search_fcns)
        self.g = np.zeros((1, self.n_funs))
        self.g[0] = 10
        self.count = 0
        self.options_dict=options_dict
        
        self.gamma = options_dict["hedgegamma"]
        self.beta = options_dict["hedgebeta"]
        self.decay = options_dict["hedgedecay"]

        # Create vector of ES weights (for SearchES)
        es_iter = self.options_dict['nsearchiter']
        self.mu = self.options_dict['nsearch'] / es_iter
        self.lamb = self.mu


    def __call__(self, u, lb, ub, func_logger, gp: GP, optim_state):
        self.count +=1
        # gamma = np.minimum(1/n_funs, np.sqrt(np.log(n_funs)/(n_funs*count)));

        self.prob = np.exp(self.beta * (self.g - np.max(self.g)))
        self.prob = self.prob / np.sum(np.exp(self.beta * (self.g - np.max(self.g))))
        self.prob = self.prob * (1 - self.n_funs * self.gamma) + self.gamma

        self.chosen_hedge = np.argwhere(np.random.rand() < np.cumsum(self.prob), axis=0)
        if len(self.chosen_hedge) == 0:
            self.chosen_hedge = np.random.randint(0, self.n_funs)
        
        if self.gamma == 0:
            self.phat = np.ones(self.g.shape)
        else:
            self.phat = np.full(self.g.shape, np.inf)
            self.phat[self.chosen_hedge] = self.prob[self.chosen_hedge]
        
        chosen_search_fun = self.search_fcns[self.chosen_hedge]
        if chosen_search_fun[0] == 'ES-wcm':
            search = SearchESWM(optim_state, self.options_dict)
            us, _ = search(u, lb, ub, func_logger, gp, optim_state, chosen_search_fun[1])
            return us
        elif chosen_search_fun[0] == 'ES-ell':
            search = SearchESELL(optim_state, self.options_dict)
            us, _ = search.search_es(u, lb, ub, func_logger, gp, optim_state,  chosen_search_fun[1])
            return us
        else:
            raise ValueError("search_hedge:Requested search method not implemented yet")
            






    