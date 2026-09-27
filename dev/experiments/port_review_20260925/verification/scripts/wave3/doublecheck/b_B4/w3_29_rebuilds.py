"""W3-29: the search's rebuilds of the local GP in PyBADS runs against
MATLAB's rule, simulated on the same events: MATLAB's post is emptied at a
search when refitgp_flag or searchcount == 0 (bads.m:523), by a search move
(707), at a poll's first step or a refit (826), and at the end of every pass
while pollmoved_flag is set (1049; pollmoved_flag set at 956/958 when the
poll stage runs, kept otherwise); a search or a poll step rebuilds when post
is empty (525, 829). Optional: an empty poll set forced at the k-th poll."""
import logging

import gpyreg
import numpy as np

import pybads
import pybads.bads.bads as bads_module
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)

events = []
fits = []
orig_fit = bads_module.local_gp_fitting


current = {}


def fitting(*args, **kwargs):
    # (refit_flag, search_count when called): the search's rebuild comes
    # before search_count is incremented, a noisy search's rebuild around
    # the search point (MATLAB's gpstructnew) after
    b = current.get("bads")
    fits.append((bool(args[6]), b.optim_state["search_count"] if b else None))
    return orig_fit(*args, **kwargs)


bads_module.local_gp_fitting = fitting
orig_check = bads_module.contraints_check
force = {"poll": None, "count": 0, "polls": 0}


def check(U, lb, ub, tol_mesh, function_logger, proj, non_box_cons):
    out = orig_check(U, lb, ub, tol_mesh, function_logger, proj, non_box_cons)
    if (
        not proj
        and force["poll"] is not None
        and force["polls"] == force["poll"]
    ):
        return out[:0]
    return out


bads_module.contraints_check = check


class Probe(BADS):
    @property
    def best_gp_hyp(self):
        return self._bgh

    @best_gp_hyp.setter
    def best_gp_hyp(self, v):
        # set at the end of every pass (bads.py:1480 at 0d866e8), after the
        # rebuild request of W3-29, and again by a noisy re-estimate
        if (
            getattr(self, "_probe_on", False)
            and events
            and events[-1][0] != "end"
        ):
            events.append(("end",))
        self._bgh = v

    def _search_step_(self, gp):
        self._probe_on = True
        sc = self.optim_state["search_count"]
        u0, n0 = self.u_best.copy(), len(fits)
        out = super()._search_step_(gp)
        rebuild = [r for r, c in fits[n0:] if c == sc]
        events.append(
            ("search", sc, rebuild, not np.array_equal(u0, self.u_best))
        )
        return out

    def _poll_step_(self, gp):
        u0, n0 = self.u_best.copy(), len(fits)
        self._probe_on = True
        out = super()._poll_step_(gp)
        force["polls"] += 1
        events.append(
            (
                "poll",
                [r for r, _ in fits[n0:]],
                not np.array_equal(u0, self.u_best),
            )
        )
        return out


def simulate(evts):
    """MATLAB's post state over the events; returns the mismatches of the
    searches' rebuilds and the counts."""
    post_empty, pollmoved = True, False
    bad, n_search, n_persist = [], 0, 0
    for i, e in enumerate(evts):
        if e[0] == "search":
            _, sc, f, moved = e
            refit = any(f)
            m_rebuild = refit or sc == 0 or post_empty
            p_rebuild = len(f) > 0
            n_search += 1
            if m_rebuild and not (refit or sc == 0):
                n_persist += 1 if pollmoved else 0
            if m_rebuild != p_rebuild:
                bad.append((i, e, post_empty, pollmoved))
            post_empty = False
            if moved:
                post_empty = True
        elif e[0] == "poll":
            _, f, moved = e
            if len(f) > 0:  # the poll ran a step: rebuild at pollcount 0
                post_empty = False
            pollmoved = moved
        elif e[0] == "end":
            if pollmoved:
                post_empty = True
    return bad, n_search, n_persist


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


class Noisy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        return float(np.sum(np.ravel(x) ** 2) + 0.3 * self.rng.normal())


cases = [
    ("rosenbrock D3", rosen, 3, {"max_fun_evals": 150}, None),
    ("rosenbrock D2", rosen, 2, {"max_fun_evals": 150}, None),
    ("rosenbrock D4", rosen, 4, {"max_fun_evals": 200}, None),
    (
        "noisy sphere D2",
        Noisy(5),
        2,
        {"max_fun_evals": 150, "uncertainty_handling": True},
        None,
    ),
    (
        "rosenbrock D3, empty poll set at polls 3-9",
        rosen,
        3,
        {"max_fun_evals": 150},
        "sweep",
    ),
]
for name, fun, D, opts, forced in cases:
    polls_to_force = [None] if forced is None else list(range(3, 10))
    for fp in polls_to_force:
        for seed in (0, 1):
            events.clear()
            fits.clear()
            force.update(poll=fp, polls=0)
            b = Probe(
                fun,
                np.full(D, -1.2),
                -5 * np.ones(D),
                5 * np.ones(D),
                -3 * np.ones(D),
                3 * np.ones(D),
                options={"display": "off", "random_seed": seed, **opts},
            )
            current["bads"] = b
            b.optimize()
            bad, n_search, n_persist = simulate(events)
            n_poll_moves = sum(1 for e in events if e[0] == "poll" and e[2])
            n_search_moves = sum(
                1 for e in events if e[0] == "search" and e[3]
            )
            n_norebuild_polls = sum(
                1 for e in events if e[0] == "poll" and not e[1]
            )
            print(
                f"{name} seed {seed} forced {fp}: searches {n_search}, "
                f"rebuilds that only a poll's move asks for {n_persist}, "
                f"poll moves {n_poll_moves}, search moves {n_search_moves}, "
                f"polls without a rebuild {n_norebuild_polls}; "
                f"mismatches with MATLAB's rule {len(bad)}",
                flush=True,
            )
            for m in bad[:3]:
                print("   ", m, flush=True)
