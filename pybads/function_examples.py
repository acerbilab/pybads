import numpy as np


def rosenbrocks_fcn(x):
    '''
        rosenbrocks_fcn Rosenbrock's 'banana' function in any dimension.
    '''
    x_2d = np.atleast_2d(x)
    return np.sum(100 * (x_2d[:, 0:-1]**2 - x_2d[:, 1:])**2 + (x_2d[:, 0:-1]-1)**2, axis=1)

def quadratic_noisy_fcn(x):
    X = np.atleast_2d(x)
    return np.sum(X**2, axis=1) + np.random.randn(X.shape[0])

def extra_noisy_quadratic_fcn(x):
    X = np.atleast_2d(x)
    quad_sum = np.sum(X**2, axis=1)
    return quad_sum + (3 + 0.1 * np.sqrt(quad_sum)) * np.random.randn(X.shape[0])