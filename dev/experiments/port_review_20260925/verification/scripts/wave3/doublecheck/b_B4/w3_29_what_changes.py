"""W3-29: at each rebuild of the local GP that only a poll's move asks for
(a search after the first of its round, no refit, no search move before
it), what the rebuild changes: the training set, the hyperparameters, the
prediction at the incumbent, poll_scale, effective_radius. Benchmark
rastrigin_D3, seeds 1, 3, 5, at the revision of the extracted tree."""
import copy
import logging
import sys
from pathlib import Path

tree = sys.argv[1]
sys.path.insert(0, str(Path(tree) / "dev" / "scripts"))
import benchmark_targets as bt  # noqa: E402
import gpyreg  # noqa: E402
import numpy as np  # noqa: E402

import pybads  # noqa: E402
import pybads.bads.bads as bm  # noqa: E402
from pybads import BADS  # noqa: E402

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
orig = bm.local_gp_fitting
state = {}


def fitting(gp, u, *args, **kwargs):
    b = state["bads"]
    sc = b.optim_state["search_count"]
    refit = args[4]
    persist = (
        sc > 0
        and not refit
        and b.reset_gp
        and b.poll_moved
        and not state["search_moved"]
    )
    if persist:
        X0 = gp.X.copy()
        y0 = gp.y.copy()
        h0 = gp.get_hyperparameters(as_array=True).copy()
        m0, s0 = gp.predict(np.atleast_2d(u))
        ps0 = np.array(gp.temporary_data["poll_scale"]).copy()
        er0 = gp.temporary_data.get("effective_radius")
        pr0 = copy.deepcopy(gp.get_priors())
    out = orig(gp, u, *args, **kwargs)
    if persist:
        g = out[0]
        m1, s1 = g.predict(np.atleast_2d(u))
        same_set = X0.shape == g.X.shape and np.array_equal(
            np.sort(X0, axis=0), np.sort(g.X, axis=0)
        )
        pr1 = g.get_priors()
        diff_priors = [k for k in pr1 if repr(pr1[k]) != repr(pr0[k])]
        state["rows"].append(
            dict(
                n=(X0.shape[0], g.X.shape[0]),
                same_set=same_set,
                same_hyp=np.array_equal(
                    h0, g.get_hyperparameters(as_array=True)
                ),
                d_mu=float(abs(m1 - m0).max()),
                d_s2=float(abs(s1 - s0).max()),
                same_poll_scale=np.array_equal(
                    ps0, g.temporary_data["poll_scale"]
                ),
                er=(er0, g.temporary_data.get("effective_radius")),
                priors_changed=diff_priors,
            )
        )
    return out


bm.local_gp_fitting = fitting
orig_search = BADS._search_step_


def search(self, gp):
    u0 = self.u_best.copy()
    out = orig_search(self, gp)
    state["search_moved"] = not np.array_equal(u0, self.u_best)
    return out


BADS._search_step_ = search
for seed in (1, 3, 5):
    cfg = bt.find_config("rastrigin_D3")
    prob = cfg.make(seed=seed, budget_scale=1.0)
    args, options = prob.bads_args()
    options["max_fun_evals"] = 200
    b = BADS(*args, options=options)
    state.update(bads=b, rows=[], search_moved=False)
    b.optimize()
    print(
        f"seed {seed}: {len(state['rows'])} rebuilds asked only by a poll's move",
        flush=True,
    )
    for r in state["rows"]:
        print("   ", r, flush=True)
