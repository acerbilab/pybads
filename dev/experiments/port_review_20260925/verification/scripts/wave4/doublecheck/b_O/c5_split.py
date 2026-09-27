"""W4-21: the ES search's first split against searchES.m:111 for every
integer mu from 1 to 4096 (MATLAB's linspace and round, exactly)."""
import os
from fractions import Fraction

import hdr  # noqa
import numpy as np

import pybads
from pybads.bads.options import Options
from pybads.search.es_search import ESSearchWM

d = os.path.join(os.path.dirname(pybads.__file__), "bads", "option_configs")
opts = Options(
    os.path.join(d, "basic_bads_options.ini"),
    evaluation_parameters={"D": 3},
    user_options={},
)
opts.load_options_file(
    os.path.join(d, "advanced_bads_options.ini"),
    evaluation_parameters={"D": 3},
)


def matlab_split(N):
    # linspace(0,N,3) = [0, N/2, N], exact for integer N; round half away
    pts = [Fraction(0), Fraction(N, 2), Fraction(N)]
    r = [int(p) + (1 if p - int(p) >= Fraction(1, 2) else 0) for p in pts]
    return [r[1] - r[0], r[2] - r[1]]


bad = []
for mu in range(1, 4097):
    s = ESSearchWM(mu, mu, opts, rng=np.random.default_rng(0))
    if list(s.ns) != matlab_split(mu):
        bad.append(mu)
print("mu 1..4096: port split != MATLAB's at", bad[:10], "count", len(bad))
for nsi in (3, 5, 7, 4095):
    mu = int(4096 / nsi)
    s = ESSearchWM(mu, mu, opts, rng=np.random.default_rng(0))
    print(
        f"n_search_iter={nsi}: mu={mu} ns={list(s.ns)} MATLAB(N=mu)={matlab_split(mu)}"
    )
