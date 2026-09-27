"""W3-15 at n_search_iter >= 3, generation by generation: with a quantized
LCB (many ties), the candidates each generation offers the acquisition, in
the port and in the searchES.m transcription of c5."""
import contextlib
import io

import hdr  # noqa: F401
import numpy as np

import pybads.search.es_search as es_module

with contextlib.redirect_stdout(io.StringIO()):
    import c5_es_transcription as t

from pybads.search.es_search import ESSearchWM

orig_lcb = es_module.acq_fcn_lcb
seen = []


def quantized(u, *args, **kwargs):
    z, f, s = orig_lcb(u, *args, **kwargs)
    seen.append(u.copy())
    return np.round(z * 2) / 2, f, s


es_module.acq_fcn_lcb = quantized
t.acq_fcn_lcb = quantized
for n_iter in (3, 4, 5):
    first_diff = []
    for seed in range(8):
        b, gp = t.state(seed=seed, n_search_iter=n_iter)
        mu = int(b.options["n_search"] / n_iter)
        seen.clear()
        s = ESSearchWM(mu, mu, b.options, rng=np.random.default_rng(seed))
        s(b.u, None, None, b.function_logger, gp, b.optim_state, True, None)
        port = [x for x in seen]
        seen.clear()
        t.searchES_matlab(
            1,
            b.u,
            gp,
            b.optim_state,
            b.options,
            np.random.default_rng(seed),
            None,
            b.function_logger,
        )
        mat = [x for x in seen]
        k = next(
            (
                g + 1
                for g, (p, m) in enumerate(zip(port, mat))
                if not np.array_equal(p, m)
            ),
            None,
        )
        first_diff.append(k)
    print(
        f"quantized LCB, n_search_iter {n_iter}: first generation whose candidates differ, per seed: {first_diff}",
        flush=True,
    )
