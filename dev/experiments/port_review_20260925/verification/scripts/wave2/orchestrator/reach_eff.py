"""Which configurations of the benchmark's default suite reach the effective
bounds of _bounds_check_ (the plausible bounds or x0 within 1e-3 (ub-lb) of
a finite hard bound)? Construction only, at seeds 0-29 for x0."""
import sys

sys.path.insert(0, "/home/user/pybads/dev/scripts")
import benchmark_targets as bt
import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
for c in bt.suite_configs("default") + bt.suite_configs("oned"):
    hits = set()
    for seed in range(30):
        p = c.make(seed=seed)
        fun, x0, lb, ub, plb, pub = p.bads_args()[0][:6]
        lb, ub, plb, pub, x0 = (
            np.atleast_2d(np.asarray(v, float)) for v in (lb, ub, plb, pub, x0)
        )
        rng_ = ub - lb
        rng_[np.isinf(rng_)] = 1e3
        LBe = lb + 1e-3 * rng_
        UBe = ub - 1e-3 * rng_
        LBe[np.isinf(lb)] = lb[np.isinf(lb)]
        UBe[np.isinf(ub)] = ub[np.isinf(ub)]
        if np.any(plb < LBe) or np.any(pub > UBe):
            hits.add("plb/pub")
        if np.any(x0 < LBe) or np.any(x0 > UBe):
            hits.add("x0")
    print(f"{c.label:40s} {sorted(hits) if hits else '-'}")
