import numpy as np


def rosenbrocks(x):
    '''
        ROSENBROCKS Rosenbrock's 'banana' function in any dimension.
    '''
    return np.sum(100 * (x[:,0:-1]**2 - x[:, 1:])**2 + (x[:, 0:-1]-1)**2, axis=1)

def rosenbrocks_single_sample(x):
    '''
        ROSENBROCKS Rosenbrock's 'banana' function in D-dimension for a single sample.
    '''
    
    return np.sum(100 * (x[0:-1]**2 - x[1:])**2 + (x[0:-1]-1)**2)