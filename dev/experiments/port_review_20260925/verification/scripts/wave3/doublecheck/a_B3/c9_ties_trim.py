"""W3-15 at n_search_iter >= 3: with tied acquisition values (quantized
LCB), the ES against the searchES.m transcription of c5 (whose pool is
trimmed to lambda at every generation, where the port's is not)."""
import c5_es_transcription as t  # noqa: E402  (runs its own checks first)
import hdr  # noqa: F401
import numpy as np

import pybads.search.es_search as es_module
from pybads.search.es_search import ESSearchWM

orig_lcb = es_module.acq_fcn_lcb


def quantized(u, *args, **kwargs):
    z, f, s = orig_lcb(u, *args, **kwargs)
    return np.round(z * 2) / 2, f, s  # few distinct values: many ties


es_module.acq_fcn_lcb = quantized
t.acq_fcn_lcb = quantized
for n_iter in (2, 3, 4):
    same = tot = 0
    for seed in range(8):
        b, gp = t.state(seed=seed, n_search_iter=n_iter)
        mu = int(b.options["n_search"] / n_iter)
        s = ESSearchWM(mu, mu, b.options, rng=np.random.default_rng(seed))
        up, zp = s(
            b.u, None, None, b.function_logger, gp, b.optim_state, True, None
        )
        um, zm, sm, _ = t.searchES_matlab(
            1,
            b.u,
            gp,
            b.optim_state,
            b.options,
            np.random.default_rng(seed),
            None,
            b.function_logger,
        )
        tot += 1
        same += np.array_equal(up, um) and s.scale == sm
    print(
        f"quantized LCB, n_search_iter {n_iter}: same point and scale as searchES.m in {same}/{tot}",
        flush=True,
    )
