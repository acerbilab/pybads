"""Which configurations of the benchmark's suites pass BADS a bound or x0
that is not a float array (W2-45)? Construction arguments only, seeds 0-29."""
import sys

sys.path.insert(0, "/home/user/pybads/dev/scripts")
import benchmark_targets as bt
import numpy as np

suites = sys.argv[1:] or ["default", "oned", "bounds"]
for c in [c for su in suites for c in bt.suite_configs(su)]:
    kinds = set()
    for seed in range(30):
        p = c.make(seed=seed)
        for name, v in zip(
            ("x0", "lb", "ub", "plb", "pub"), p.bads_args()[0][1:6]
        ):
            if v is not None:
                kinds.add((name, np.asarray(v).dtype.kind))
    odd = sorted(k for k in kinds if k[1] != "f")
    print(f"{c.label:40s} {odd if odd else 'all float'}")
