import numpy as np


def rosenbrocks(x):
    '''
        ROSENBROCKS Rosenbrock's 'banana' function in any dimension.
    '''
    
    return np.sum( 100 * (x[:,0:-1]**2 - x[:, 1:])**2 + (x[:, 0:-1]-1)**2)