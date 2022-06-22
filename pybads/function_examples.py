from typing import Callable
import numpy as np


def rosenbrocks_fcn(x):
    '''
        rosenbrocks_fcn Rosenbrock's 'banana' function in any dimension.
    '''
    x_2d = np.atleast_2d(x)
    return np.sum(100 * (x_2d[:, 0:-1]**2 - x_2d[:, 1:])**2 + (x_2d[:, 0:-1]-1)**2, axis=1)

def quadratic_unknown_noisy_fcn(x):
    X = np.atleast_2d(x)
    return np.sum(X**2, axis=1) + np.random.randn(X.shape[0])

def quadratic_noisy_fcn(x):
    X = np.atleast_2d(x)
    noise =  np.random.lognormal(size=X.shape[0])  + np.sqrt(np.abs(np.min(X, axis=1)))
    return (np.sum(X**2, axis=1) + noise, noise.item())

def rosebrocks_hetsk_noisy_fcn(x):
    X = np.atleast_2d(x)
    f_X = rosenbrocks_fcn(X)
    f_min = 0.
    noise = np.random.normal() + 1 + 0.1 * (f_X - f_min)
    return (f_X + noise, noise)
    

def extra_noisy_quadratic_fcn(x):
    X = np.atleast_2d(x)
    quad_sum = np.sum(X**2, axis=1)
    return quad_sum + (3 + 0.1 * np.sqrt(quad_sum)) * np.random.randn(X.shape[0])

def quadratic_non_bound_constr(x):
    X = np.atleast_2d(x)
    return np.sum(X**2, axis=1) > 1


def ackley_fcn(X):
    U = np.atleast_2d(X)
    f = -20 * np.exp(-0.2 * np.sqrt(np.sum(U*U, axis=1) / U.shape[1])) \
                - np.exp(np.sum(np.cos(2*np.pi*U), axis=1) / U.shape[1]) \
                + 20 + 2.7182818284590452353602874713526625
    return f

def rastrigin(X):
    U = np.atleast_2d(X)
    return 10 * U.shape[1] + np.sum( U**2 - 10 * np.cos(2*np.pi*U) + 10, axis=1)
    