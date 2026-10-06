"""K1 / I-F1 / C-F4 / I-F2: after the re-estimation moves the incumbent's value
to an earlier iterate idx, where do the next search and the next poll run,
with which value, and with which target hyperparameters?"""
import sys

import common
import numpy as np

from pybads import BADS


class Probe(BADS):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.snap = None
        self.moves = []
        self.pending = None

    def _re_evaluate_history_(self, gp):
        super()._re_evaluate_history_(gp)
        ih = self.iteration_history
        self.snap = dict(
            fval=ih.get("fval").astype(float).copy(),
            u=np.array(list(ih.get("u"))),
            hyp=[np.array(h) for h in ih.get("gp_hyp_full")],
            it=self.optim_state["iter"],
            ubest=np.ravel(self.u_best).copy(),
            gp=gp,
        )

    def _entry(self, where):
        if self.snap is not None:
            s = self.snap
            self.snap = None
            it = s["it"]
            if self.fval != s["fval"][it]:
                idx = int(np.flatnonzero(s["fval"] == self.fval)[0])
                self.pending = dict(
                    it=it,
                    idx=idx,
                    u_idx=s["u"][idx],
                    u_old=s["ubest"],
                    f_moved=self.fval,
                    hyp_idx=s["hyp"][idx],
                    diffloc=not np.allclose(s["u"][idx], s["ubest"]),
                    first=where,
                    search_center_moved=None,
                )
                self.moves.append(self.pending)
        ev = self.pending
        if (
            ev is not None
            and where == "search"
            and ev["search_center_moved"] is None
        ):
            ev["search_center_moved"] = bool(
                np.array_equal(np.ravel(self.u), np.ravel(ev["u_idx"]))
            )
            ev["target_at_old"] = bool(
                np.array_equal(np.ravel(self.u_best), ev["u_old"])
            )
            ev["search_target_hyp_moved"] = bool(
                np.array_equal(
                    np.ravel(self.best_gp_hyp), np.ravel(ev["hyp_idx"])
                )
            )
        if ev is not None and where == "poll":
            self.pending = None
            ev["poll_at_idx"] = bool(
                np.array_equal(np.ravel(self.u), np.ravel(ev["u_idx"]))
            )
            ev["poll_at_old"] = bool(
                np.array_equal(np.ravel(self.u), ev["u_old"])
            )
            ev["poll_fval_moved"] = self.fval == ev["f_moved"]
            ev["poll_hyp_moved"] = bool(
                np.array_equal(
                    np.ravel(self.best_gp_hyp), np.ravel(ev["hyp_idx"])
                )
            )

    def _search_step_(self, gp):
        self._entry("search")
        return super()._search_step_(gp)

    def _poll_step_(self, gp):
        self._entry("poll")
        return super()._poll_step_(gp)


extra = dict(search_n_try=0) if "nosearch" in sys.argv else {}
tot = {}
for seed in range(6 if not extra else 3):
    nrng = np.random.default_rng(100 + seed)
    f = lambda x: float(
        np.sum(np.ravel(x) ** 2) + 0.5 * nrng.standard_normal()
    )
    b = Probe(
        f,
        np.array([1.5, -1.0]),
        np.full(2, -5.0),
        np.full(2, 5.0),
        np.full(2, -2.0),
        np.full(2, 2.0),
        options=dict(
            uncertainty_handling=True,
            max_fun_evals=200,
            random_seed=seed,
            display="off",
            **extra,
        ),
    )
    r = b.optimize()
    mv = [m for m in b.moves if m["diffloc"]]
    c = dict(
        moves=len(b.moves),
        to_other_location=len(mv),
        search_centred_at_idx=sum(
            bool(m.get("search_center_moved")) for m in mv
        ),
        search_target_at_old_ubest=sum(
            bool(m.get("target_at_old")) for m in mv
        ),
        search_target_hyp_moved=sum(
            bool(m.get("search_target_hyp_moved")) for m in mv
        ),
        polled=sum("poll_at_old" in m for m in mv),
        poll_at_idx=sum(bool(m.get("poll_at_idx")) for m in mv),
        poll_at_old_with_moved_fval=sum(
            bool(m.get("poll_at_old")) and bool(m.get("poll_fval_moved"))
            for m in mv
        ),
        poll_hyp_moved=sum(bool(m.get("poll_hyp_moved")) for m in mv),
    )
    for k, v in c.items():
        tot[k] = tot.get(k, 0) + v
    print(
        f"seed {seed}: iterations {r['iterations']} func_count {r['func_count']}:",
        c,
    )
print("total", extra, tot)
