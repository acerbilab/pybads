"""ESSearch.__init__'s split of the initial population (es_search.py:33-35)
against searchES.m:111 (MATLAB's round), for the mu that n_search and
n_search_iter give."""
import hdr  # noqa: F401
import numpy as np
from ucheck_ref import mround

for n_search, n_iter in [
    (4096, 2),
    (4096, 4),
    (4096, 8),
    (1002, 2),
    (4096, 3),
    (4096, 4096),
]:
    mu = int(n_search / n_iter)
    port = np.diff(np.round(np.linspace(0, mu, 3)).astype(int))
    mat = np.diff(mround(np.linspace(0, mu, 3)).astype(int))
    print(
        f"n_search {n_search}, n_search_iter {n_iter}: mu {mu}, port ns {port.tolist()}, MATLAB's round {mat.tolist()}"
    )
