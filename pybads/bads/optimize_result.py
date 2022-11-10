
import numpy as np
import copy
import pkg_resources


class OptimizeResult(dict):
    
    _keys = ['x',
             'x0',                  # Initial starting point
             'success',
             'status',
             'message',
             'fun',
             'func_count',          # Number of evaluations of the objective functions
             'iterations',          # Number of iterations performed by the optimizer.
             'target_type',
             'problem_type',
             'mesh_size',
             'non_box_cons',        # non_box_constraint function
             'yval_vec',
             'ysd_vec',
             'fval',
             'fsd',
             'total_time',
             'overhead',
             'random_seed',
             'algorithm',
             'version'
            ] 
    
    def __init__(self, bads=None):
        super().__init__()
        if bads is not None:
            self._set_attributes(bads)
        
    def _set_attributes(self, bads):
        self['fun'] = bads.function_logger.fun
        self['non_box_cons'] = bads.non_box_cons
        if bads.optim_state["uncertainty_handling_level"] > 0:
            if bads.options['specify_target_noise']:
                self['target_type'] = 'stochastic (specified noise)'
            else:
                self['target_type'] = 'stochastic'
        else:
            self['target_type'] = 'deterministic'
        
        
        if np.all(np.isinf(bads.lower_bounds)) and np.all(np.isinf(bads.upper_bounds)) \
                and bads.non_box_cons is None:
            self['problem_type'] = 'unconstrained'
        elif bads.non_box_cons is None:
            self['problem_type'] = 'bound constraints'
        else:
            self['problem_type'] = 'non-box constraints'
            
        self['iterations'] = bads.optim_state['iter']
        self['func_count'] = bads.function_logger.func_count
        self['mesh_size'] = bads.mesh_size
        self['overhead'] = bads.optim_state['overhead']
        self['algorithm'] = 'Bayesian adaptive direct search'
        if bads.optim_state['uncertainty_handling_level'] > 0 and bads.options['noise_final_samples'] > 0:
            self['yval_vec'] = bads.optim_state['yval_vec'].copy()
        else: 
            self['yval_vec'] = None
        
        if bads.options['specify_target_noise'] and bads.options['noise_final_samples'] > 0:
            self['ysd_vec'] = bads.optim_state['ysd_vec']
        else:
            self['ysd_vec'] = None
        
        self['x0'] = bads.x0.copy()
        self['x'] = bads.x.copy()
        self['fval'] = bads.fval
        self['fsd'] = bads.fsd
        self['total_time'] = bads.optim_state['total_time']
        
        #self['version'] = bads.version   # part of setuptools version = pkg_resources.require("MyProject")[0].version
        #'status',
        self['success'] = True          # TODO: In our case when an error occurs, the application just stops.
        self['random_seed'] = None      # TODO: PyBADS does not receive seed as input right now
        self['version'] = '0.0.1'       # TODO: Retrieve the version from setup.py, or define it somewhere in BADS class.
                                                #version = pkg_resources.require("pybads")[0].version
        self['message'] = bads.optim_state['termination_msg']
        
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
            raise ValueError(
                """The key is not part of OptimizeResult._keys"""
            )
        else:
            dict.__setitem__(self, key, copy.deepcopy(val))
        
        
                
        
        
        
        