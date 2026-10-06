"""K3 / K9 / C-F5: failures injected into the rebuilds of _re_evaluate_history_.
Checks, at fef6c14:
 (a) the GPs stored in iteration_history are not changed by later re-evaluations;
 (b) a past iterate whose rebuild fails gets NaN; the current one keeps its value;
 (c) the working GP (object, temporary_data markers, hyperparameters, data) is untouched;
 (d) both choices skip NaN; the incumbent and the result are never NaN;
 (e) NaN left in iteration_history after the run when the final re-evaluation fails."""

import common
import gpyreg as gpr
import numpy as np

from pybads import BADS

STATE = dict(active=False, fail_at=set(), i=0)
orig_update = gpr.GP.update


def update(self, *a, **k):
    if STATE["active"] and k.get("compute_posterior", True):
        i = STATE["i"]
        STATE["i"] += 1
        if i in STATE["fail_at"]:
            raise np.linalg.LinAlgError("injected")
    return orig_update(self, *a, **k)


gpr.GP.update = update


def gp_sig(gp):
    return (
        None if gp.X is None else float(np.sum(gp.X)),
        None if gp.y is None else float(np.sum(gp.y)),
        tuple(np.round(gp.get_hyperparameters(as_array=True).ravel(), 12)),
    )


class Probe(BADS):
    def __init__(self, *a, plan=None, **k):
        super().__init__(*a, **k)
        self.plan = plan or {}
        self.n_reeval = 0
        self.logs = []
        self.first_sig = {}

    def _re_evaluate_history_(self, gp):
        if self.optim_state["last_re_eval"] == self.function_logger.func_count:
            return super()._re_evaluate_history_(gp)  # skipped: nothing new
        gps = self.iteration_history.get("gp")
        for j, g in enumerate(gps):
            if j not in self.first_sig:
                self.first_sig[j] = gp_sig(g)
        n = len(self.iteration_history.get("u"))
        before = dict(
            fval=self.iteration_history.get("fval").astype(float).copy(),
            td=dict(gp.temporary_data),
            sig=gp_sig(gp),
            id=id(gp),
        )
        which = self.plan.get(self.n_reeval, None)
        if which == "final":  # fail every iterate but the current one
            which = list(range(n - 1))
        STATE.update(
            active=True,
            i=0,
            fail_at=set()
            if which is None
            else set(
                which
                if isinstance(which, list)
                else [n - 1 if which == "current" else which]
            ),
        )
        # one update call per iterate in local_gp_fitting (no refit): index = iterate
        super()._re_evaluate_history_(gp)
        STATE["active"] = False
        assert STATE["i"] == n, (STATE["i"], n)
        after = self.iteration_history.get("fval").astype(float)
        self.logs.append(
            dict(
                k=self.n_reeval,
                n=n,
                failed=sorted(STATE["fail_at"]),
                before=before["fval"],
                after=after.copy(),
                gp_same=(
                    id(gp) == before["id"] and gp_sig(gp) == before["sig"]
                ),
                td_same=(gp.temporary_data.keys() == before["td"].keys()),
                markers=(
                    gp.temporary_data.get("needs_rebuild"),
                    gp.temporary_data.get("needs_refit"),
                ),
            )
        )
        self.n_reeval += 1


nrng = np.random.default_rng(5)
f = lambda x: float(np.sum(np.ravel(x) ** 2) + 0.5 * nrng.standard_normal())
b = Probe(
    f,
    np.array([1.5, -1.0]),
    np.full(2, -5.0),
    np.full(2, 5.0),
    np.full(2, -2.0),
    np.full(2, 2.0),
    options=dict(
        uncertainty_handling=True,
        max_fun_evals=150,
        random_seed=3,
        display="off",
    ),
    plan={1: 0, 2: "current", 3: [1, 2]},
)
# the final re-evaluation is the last call; mark it by patching after the loop: use a large key set later
orig_opt = Probe._re_evaluate_history_
r = None


class Final(Probe):
    pass


# run once to learn the number of re-evaluations, then rerun with the final one failing
r0 = b.optimize()
nre = b.n_reeval
nrng = np.random.default_rng(5)
b = Probe(
    f,
    np.array([1.5, -1.0]),
    np.full(2, -5.0),
    np.full(2, 5.0),
    np.full(2, -2.0),
    np.full(2, 2.0),
    options=dict(
        uncertainty_handling=True,
        max_fun_evals=150,
        random_seed=3,
        display="off",
    ),
    plan={1: 0, 2: "current", 3: [1, 2], nre - 1: "final"},
)
r = b.optimize()
print(
    "re-evaluations:",
    nre,
    "| same trajectory up to the injected failures? func_count",
    r0["func_count"],
    r["func_count"],
)
for L in b.logs:
    if L["failed"]:
        n = L["n"]
        print(
            f"re-eval {L['k']} (n={n}) failed at {L['failed']}: before {np.round(L['before'], 4)}"
        )
        print(
            f"      after {np.round(L['after'], 4)} | working GP same object/data/hyp: {L['gp_same']}, markers {L['markers']}"
        )
changed = [
    j
    for j, s in b.first_sig.items()
    if gp_sig(b.iteration_history.get("gp")[j]) != s
]
print(
    "stored GPs changed after their record:", changed, "of", len(b.first_sig)
)
print(
    "result fval",
    r["fval"],
    "fsd",
    r["fsd"],
    "| iteration_history fval after run:",
    np.round(b.iteration_history.get("fval").astype(float), 4),
)
print(
    "NaN left in history after run at",
    list(
        np.flatnonzero(np.isnan(b.iteration_history.get("fval").astype(float)))
    ),
)
