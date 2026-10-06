"""Shared helpers for the B6 verifier's checks."""

import inspect
import textwrap

import gpyreg
import numpy as np

import pybads
import pybads.bads.bads as bads_mod
import pybads.bads.gaussian_process_train as gpt

_printed = False


def banner():
    global _printed
    if not _printed:
        print("pybads.__file__ =", pybads.__file__)
        print("gpyreg.__file__ =", gpyreg.__file__)
        _printed = True


banner()


def sphere(x):
    x = np.atleast_1d(x)
    return float(np.sum(x**2))


def rosenbrock(x):
    x = np.atleast_1d(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def ackley(x):
    x = np.atleast_1d(x)
    D = x.size
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / D))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / D)
        + 20
        + np.e
    )


def noisy(f, sd, seed):
    rng = np.random.default_rng(seed)

    def g(x):
        return f(x) + sd * rng.standard_normal()

    return g


def noisy_with_sd(f, sd, seed):
    rng = np.random.default_rng(seed)

    def g(x):
        return (f(x) + sd * rng.standard_normal(), sd)

    return g


def patched_local_gp_fitting(write_noise_prior=True):
    """A copy of local_gp_fitting whose noise prior is written back.

    Returns a function defined in the module namespace of
    gaussian_process_train, identical to the original except for one
    inserted line after `prior_noise = (...)`.
    """
    src = textwrap.dedent(inspect.getsource(gpt.local_gp_fitting))
    anchor = (
        "    prior_noise = (prior_noise[0], (mu_noise_prior, "
        "prior_noise[1][1]))\n"
    )
    assert anchor in src, "anchor not found"
    if write_noise_prior:
        src = src.replace(
            anchor, anchor + '    gp_priors["noise_log_scale"] = prior_noise\n'
        )
    ns = dict(vars(gpt))
    exec(compile(src, "<patched local_gp_fitting>", "exec"), ns)
    return ns["local_gp_fitting"]
