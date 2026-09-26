"""F1: the re-evaluation's move of the incumbent sets self.best_u, not
self.u_best; check whether the next pass reverts self.u to the old incumbent
while keeping the moved iterate's fval/fsd."""

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


class Probe(BADS):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.events = []
        self._pending = None

    def _search_step_(self, gp):
        self._check("search")
        return super()._search_step_(gp)

    def _poll_step_(self, gp):
        self._check("poll")
        return super()._poll_step_(gp)

    def _check(self, where):
        if not np.array_equal(np.ravel(self.u), np.ravel(self.u_best)):
            # a move by the re-evaluation happened at the end of the last pass
            self.events.append(
                dict(
                    kind="move_seen",
                    where=where,
                    iter=self.optim_state["iter"],
                    u=np.ravel(self.u).copy(),
                    u_best=np.ravel(self.u_best).copy(),
                    fval=float(self.fval),
                    best_u=np.ravel(getattr(self, "best_u", np.nan)).copy(),
                )
            )
            self._pending = self.events[-1]
        elif self._pending is not None and where == "poll":
            ev = self._pending
            # at the poll of that iteration: where is the incumbent?
            self.events.append(
                dict(
                    kind="at_poll",
                    iter=self.optim_state["iter"],
                    u=np.ravel(self.u).copy(),
                    fval=float(self.fval),
                    at_moved=np.array_equal(np.ravel(self.u), ev["u"]),
                    at_old=np.array_equal(np.ravel(self.u), ev["u_best"]),
                    fval_is_moved=float(self.fval) == ev["fval"],
                )
            )
            self._pending = None


def run(seed):
    noise_rng = np.random.default_rng(1000 + seed)

    def fun(x):
        x = np.ravel(x)
        return float(np.sum(x**2) + 0.5 * noise_rng.standard_normal())

    b = Probe(
        fun,
        np.array([[1.5, -1.0]]),
        np.array([[-5, -5]]),
        np.array([[5, 5]]),
        np.array([[-2, -2]]),
        np.array([[2, 2]]),
        options=dict(
            uncertainty_handling=True,
            max_fun_evals=200,
            random_seed=seed,
            display="off",
        ),
    )
    res = b.optimize()
    return b, res


tot_moves = 0
tot_reverted = 0
tot_kept = 0
for seed in range(6):
    b, res = run(seed)
    moves = [e for e in b.events if e["kind"] == "move_seen"]
    polls = [e for e in b.events if e["kind"] == "at_poll"]
    rev = sum(p["at_old"] and not p["at_moved"] for p in polls)
    kept = sum(p["at_moved"] for p in polls)
    tot_moves += len(moves)
    tot_reverted += rev
    tot_kept += kept
    print(
        f"seed {seed}: iters {res['iterations']}, fcount {res['func_count']}, moves {len(moves)}, "
        f"poll at old incumbent {rev}, poll at moved iterate {kept}, other {len(polls)-rev-kept}"
    )
    for p in polls[:3]:
        if p["at_old"]:
            print(
                "   iter",
                p["iter"],
                "poll around old u",
                p["u"],
                "with fval of moved iterate:",
                p["fval_is_moved"],
            )
print("total moves", tot_moves, "reverted", tot_reverted, "kept", tot_kept)
