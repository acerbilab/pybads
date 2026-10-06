import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)

events = []
orig_search = BADS._search_step_


def search(self, gp):
    if not np.array_equal(np.ravel(self.u), np.ravel(self.u_best)):
        events.append(
            dict(
                iter=self.optim_state["iter"],
                u=np.ravel(self.u).copy(),
                u_best=np.ravel(self.u_best).copy(),
                fval=float(self.fval),
                search_count=self.optim_state["search_count"],
            )
        )
    return orig_search(self, gp)


BADS._search_step_ = search
orig_poll = BADS._poll_step_
polls = []


def poll(self, gp):
    polls.append(
        (
            self.optim_state["iter"],
            np.ravel(self.u).copy(),
            np.ravel(self.u_best).copy(),
        )
    )
    return orig_poll(self, gp)


BADS._poll_step_ = poll

D = 2
for seed in range(6):
    nrng = np.random.default_rng(seed)
    f = lambda x: float(np.sum((x - 0.3) ** 2) + 1.0 * nrng.normal())
    events.clear()
    polls.clear()
    b = BADS(
        f,
        np.full(D, 1.0),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=dict(
            display="off",
            random_seed=seed,
            max_fun_evals=200,
            uncertainty_handling=True,
        ),
    )
    r = b.optimize()
    print(
        f"seed {seed}: iterations {r['iterations']}, moves detected at search entry: {len(events)}"
    )
    for e in events[:3]:
        it = e["iter"]
        pol = [p for p in polls if p[0] == it]
        hu = b.iteration_history.get("u")
        print(
            f"   round (0-based) {it}: search centred at u={np.round(e['u'],3)}, incumbent u_best={np.round(e['u_best'],3)};"
            f" poll of that round centred at {np.round(pol[0][1],3) if pol else None};"
            f" recorded u of that round {np.round(hu[it],3) if it < len(hu) else None}"
        )
