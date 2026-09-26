"""W2-4 on the bounds suite: the plausible box BADS takes for each
configuration (seed 0), beside the one given, at the checkout named as the
first argument (benchmark_targets puts the checkout that holds it first on
sys.path, so PYTHONPATH does not select PyBADS here)."""
import sys

sys.path.insert(0, sys.argv[1] + "/dev/scripts")
import benchmark_targets as bt
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
for c in bt.suite_configs("bounds"):
    p = c.make(seed=0)
    (fun, x0, lb, ub, plb, pub), opts = p.bads_args()[0][:6], p.bads_args()[1]
    b = BADS(fun, x0, lb, ub, plb, pub, options=dict(opts, display="off"))
    s = b.optim_state
    print(
        f"{c.label:20s} given plb {None if plb is None else np.ravel(plb)[0]:} "
        f"pub {None if pub is None else np.ravel(pub)[0]}; taken "
        f"plb {s['plb_orig'].ravel()[0]:.4g} pub {s['pub_orig'].ravel()[0]:.4g}; "
        f"x0 given {np.ravel(x0)[0]:.4g} taken {np.ravel(b.x0)[0]:.4g}; "
        f"log {bool(np.ravel(b.var_transf.apply_log_t)[0])}"
    )
