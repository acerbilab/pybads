"""Record one 2-D PyBADS run on the film's landscape (landscape.py) and write it as a trace for film.html.

The recorder subclasses BADS and wraps its search and poll steps, the target and the module names of the
acquisition function and the ES classes. It only reads: it draws nothing from ``bads.rng`` and writes none of the
run's state. ``--check`` reruns the same seed without the recorder, in the same process, and compares the
evaluations bit for bit. ``trace.js`` keeps only the GP states that a step used, renumbered in order, as
surface.py stores them; the candidates of the searches' generations are recorded but not written, since the film
does not draw them.

Run from the film's folder, dev/film, with the development environment's Python (PyBADS and gpyreg) and one BLAS
thread. ``trace.js`` is the kept trace: write a new export outside the repository and compare it before replacing it.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/export_trace.py --out OUT/trace_new.js --check
"""

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import gpyreg
import landscape as LS
import numpy as np
import scipy
import surface

import pybads
import pybads.bads.bads as bads_mod
import pybads.search.es_search as es_mod
import pybads.search.search_hedge as hedge_mod
from pybads import BADS

# The film's landscape (landscape.py): a diagonal valley down to A, and a narrow trench from A along +x1.
VARIANT = dict(a=0.08, b=4.0, A=(-0.5, -0.5), tstar=4.0)
LB, UB, PLB, PUB, X0 = LS.LB, LS.UB, LS.PLB, LS.PUB, LS.X0
_F, XMIN, _FMIN, INFO = LS.make(**VARIANT)
N_GRID = 128


def target_fun(x):
    return _F(x)


FMIN = target_fun(XMIN)


class Recorder:
    def __init__(self, grid):
        self.grid = grid
        self.stage = "init"
        self.evals = []
        self.steps = []  # searches and polls, in order
        self.states = []  # GP on the grid, deduplicated
        self._state_keys = {}
        self._es = None
        # (stage, state index, n points, picked mu, picked sd, candidates)
        self._acq = []


def source_revision(module):
    root = Path(module.__file__).resolve().parent.parent
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-uno"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def gp_state(rec, bads, gp):
    """The GP's mean and SD on the grid, stored once per distinct state."""
    hyp = gp.get_hyperparameters(as_array=True).ravel()
    key = (hyp.tobytes(), gp.X.tobytes(), gp.y.tobytes())
    if key in rec._state_keys:
        return rec._state_keys[key]
    mu, s2 = gp.predict(bads.var_transf(rec.grid))
    half = (PUB - PLB) / 2
    X = bads.var_transf.inverse_transf(gp.X.copy())
    idx = len(rec.states)
    rec.states.append(
        dict(
            mu=np.asarray(mu).ravel().copy(),
            sd=np.sqrt(np.maximum(np.asarray(s2).ravel(), 0.0)),
            hyp=hyp.tolist(),
            ell=(np.exp(hyp[:2]) * half).tolist(),
            ntrain=int(gp.X.shape[0]),
            X=X,
            n_evals=len(rec.evals),
        )
    )
    rec._state_keys[key] = idx
    return idx


class TracedBADS(BADS):
    def attach(self, rec):
        self._rec = rec
        return self

    def _x(self, u):
        return self.var_transf.inverse_transf(np.atleast_2d(u)).ravel()

    def _search_step_(self, gp):
        rec, os_ = self._rec, self.optim_state
        pre = dict(
            kind="search",
            iter=int(os_["iter"]),
            count=int(os_["search_count"]),
            f0=float(self.fval),
            x0=self._x(self.u).tolist(),
            msi=int(self.mesh_size_integer),
            mesh=float(self.mesh_size),
            search_factor=float(os_["search_factor"]),
            e0=len(rec.evals),
            a0=len(rec._acq),
        )
        rec.stage = "search"
        rec._es = None
        out = super()._search_step_(gp)
        rec.stage = "between"
        status = os_["search_stats"]["success"][-1]
        acq = rec._acq[pre.pop("a0") :]
        step = dict(
            pre,
            method=self.search_es_hedge.chosen_search_fun[0],
            status={1.0: "success", 0.5: "incremental", 0.0: "failure"}[
                status
            ],
            evals=list(range(pre["e0"], len(rec.evals))),
            f1=float(self.fval),
            x1=self._x(self.u).tolist(),
            state=acq[-1][1] if acq else None,
            mu_pick=acq[-1][3] if acq else None,
            sd_pick=acq[-1][4] if acq else None,
            es=rec._es,
        )
        rec.steps.append(step)
        return out

    def _poll_step_(self, gp):
        rec, os_ = self._rec, self.optim_state
        n_succ = len(os_["u_success"])
        pre = dict(
            kind="poll",
            iter=int(os_["iter"]),
            f0=float(self.fval),
            x0=self._x(self.u).tolist(),
            msi=int(self.mesh_size_integer),
            mesh=float(self.mesh_size),
            e0=len(rec.evals),
            a0=len(rec._acq),
        )
        rec.stage = "poll"
        out = super()._poll_step_(gp)
        rec.stage = "between"
        acq = rec._acq[pre.pop("a0") :]
        rec.steps.append(
            dict(
                pre,
                success=len(os_["u_success"]) > n_succ,
                moved=bool(self.poll_moved),
                evals=list(range(pre["e0"], len(rec.evals))),
                f1=float(self.fval),
                x1=self._x(self.u).tolist(),
                msi1=int(self.mesh_size_integer),
                mesh1=float(self.mesh_size),
                state=acq[0][1] if acq else None,
                states=[a[1] for a in acq],
                cands=acq[0][5] if acq else [],
                n_calls=len(acq),
            )
        )
        return out


class Patches:
    """Wrap the acquisition function and the ES classes; restore on exit."""

    def __init__(self, rec, ref):
        self.rec, self.ref = rec, ref

    def __enter__(self):
        rec, ref = self.rec, self.ref
        self.o_bads, self.o_es = bads_mod.acq_fcn_lcb, es_mod.acq_fcn_lcb
        self.o_wm, self.o_ell = hedge_mod.ESSearchWM, hedge_mod.ESSearchELL
        o_bads, o_es = self.o_bads, self.o_es

        def bads_acq(xi, func_count, gp, *a, **k):
            out = o_bads(xi, func_count, gp, *a, **k)
            z = np.asarray(out[0]).ravel()
            mu = np.asarray(out[1]).ravel()
            sd = np.asarray(out[2]).ravel()
            pick = None if np.all(np.isnan(z)) else int(np.nanargmin(z))
            cand = None
            if rec.stage == "poll":
                # the poll set: where each remaining step would land, and
                # what the GP predicts there
                X = ref[0].var_transf.inverse_transf(
                    np.array(np.atleast_2d(xi), copy=True)
                )
                cand = [
                    (X[i].copy(), float(mu[i]), float(sd[i]), float(z[i]))
                    for i in range(len(X))
                ]
            rec._acq.append(
                (
                    rec.stage,
                    gp_state(rec, ref[0], gp),
                    int(np.atleast_2d(xi).shape[0]),
                    None if pick is None else float(mu[pick]),
                    None if pick is None else float(sd[pick]),
                    cand,
                )
            )
            return out

        def es_acq(xi, func_count, gp, *a, **k):
            out = o_es(xi, func_count, gp, *a, **k)
            if rec._es is not None:
                rec._es["gens"].append(
                    (
                        ref[0].var_transf.inverse_transf(
                            np.array(xi, copy=True)
                        ),
                        np.asarray(out[0]).ravel().copy(),
                    )
                )
            return out

        def traced(cls):
            class Traced(cls):
                def __call__(self, u, *a, **k):
                    rec._es = {"cls": cls.__name__, "gens": []}
                    out = super().__call__(u, *a, **k)
                    rec._es["sqrt_sigma"] = np.array(self.sqrt_sigma)
                    return out

            Traced.__name__ = cls.__name__
            return Traced

        bads_mod.acq_fcn_lcb = bads_acq
        es_mod.acq_fcn_lcb = es_acq
        hedge_mod.ESSearchWM = traced(self.o_wm)
        hedge_mod.ESSearchELL = traced(self.o_ell)
        return self

    def __exit__(self, *exc):
        bads_mod.acq_fcn_lcb, es_mod.acq_fcn_lcb = self.o_bads, self.o_es
        hedge_mod.ESSearchWM, hedge_mod.ESSearchELL = self.o_wm, self.o_ell
        return False


def options(seed):
    return {"display": "off", "random_seed": seed, "show_tips": False}


def run_traced(seed):
    g = np.linspace(LB[0], UB[0], N_GRID)
    gx, gy = np.meshgrid(g, g)  # row = x2, column = x1
    rec = Recorder(np.column_stack([gx.ravel(), gy.ravel()]))

    def target(x):
        y = target_fun(x)
        rec.evals.append(
            dict(stage=rec.stage, x=np.array(x, dtype=float).copy(), y=y)
        )
        return y

    ref = [None]
    with Patches(rec, ref):
        bads = TracedBADS(
            target, X0, LB, UB, PLB, PUB, options=options(seed)
        ).attach(rec)
        ref[0] = bads
        rec.result = bads.optimize()
    rec.bads = bads
    return rec


def run_plain(seed):
    evals = []

    def target(x):
        y = target_fun(x)
        evals.append(np.r_[np.array(x, dtype=float), y])
        return y

    res = BADS(target, X0, LB, UB, PLB, PUB, options=options(seed)).optimize()
    return np.array(evals), res


def r(v, n=5):
    return [round(float(t), n) for t in np.ravel(v)]


def build(rec, first, last, seed):
    """The trace of the evaluations first..last (1-based, inclusive): every
    evaluation of the run, and the GP states and candidate clouds of the
    steps in that window."""
    ev = rec.evals
    for s in rec.steps:
        for i in s["evals"]:
            ev[i]["step"] = rec.steps.index(s)
    steps = [
        s
        for s in rec.steps
        if s["evals"] and first - 1 <= s["evals"][0] <= last - 1
    ]
    used = sorted({s["state"] for s in steps if s["state"] is not None})
    remap = {old: new for new, old in enumerate(used)}
    grid, surfaces = surface.encode(
        [rec.states[i]["mu"] for i in used],
        [rec.states[i]["sd"] for i in used],
        N_GRID,
    )
    states = []
    for i, (mu, sd) in zip(used, surfaces):
        st = rec.states[i]
        states.append(
            dict(
                ell=r(st["ell"], 4),
                hyp=r(st["hyp"], 5),
                ntrain=st["ntrain"],
                n_evals=st["n_evals"],
                mu=mu,
                sd=sd,
            )
        )
    out_steps = []
    for s in steps:
        d = {k: s[k] for k in ("kind", "iter", "msi", "mesh", "f0", "f1")}
        d["x0"], d["x1"] = r(s["x0"]), r(s["x1"])
        d["evals"] = s["evals"]
        d["state"] = remap[s["state"]]
        if s["kind"] == "search":
            d.update(
                method=s["method"],
                status=s["status"],
                count=s["count"],
                search_factor=s["search_factor"],
                mu_pick=round(s["mu_pick"], 4),
                sd_pick=round(s["sd_pick"], 4),
            )
            es = s["es"]
            half = (PUB - PLB) / 2
            d["sqrt_sigma"] = r(es["sqrt_sigma"] * half[None, :], 5)
        else:
            arm = s["mesh"] * (PUB - PLB) / 2
            arms = []
            c = np.array(s["x0"])
            done = {tuple(np.round(ev[i]["x"], 6)): i for i in s["evals"]}
            before = [tuple(np.round(e["x"], 6)) for e in ev[: s["evals"][0]]]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = c + np.array([dx * arm[0], dy * arm[1]])
                key = tuple(np.round(p, 6))
                if np.any(p < LB) or np.any(p > UB):
                    status, i = "outside", None
                elif key in done:
                    i = done[key]
                    status = "hit" if ev[i]["y"] < s["f0"] else "miss"
                elif key in before:
                    status, i = "known", before.index(key)
                else:
                    status, i = "skipped", None
                one = dict(x=r(p), status=status, eval=i)
                for cx, cmu, csd, cz in s["cands"]:
                    if np.allclose(cx, p, atol=1e-6):
                        one.update(
                            mu=round(cmu, 4),
                            sd=round(csd, 4),
                            lcb=round(cz, 4),
                        )
                arms.append(one)
            d.update(
                arms=arms,
                success=s["success"],
                moved=s["moved"],
                msi1=s["msi1"],
                mesh1=s["mesh1"],
                n_calls=s["n_calls"],
            )
        out_steps.append(d)
    inc = np.inf
    evals = []
    for i, e in enumerate(ev):
        inc = min(inc, e["y"])
        evals.append(
            [
                round(float(e["x"][0]), 5),
                round(float(e["x"][1]), 5),
                round(e["y"], 5),
                {"init": 0, "search": 1, "poll": 2}[e["stage"]],
            ]
        )
    res = rec.result
    return dict(
        meta=dict(
            seed=seed,
            window=[first, last],
            pybads=version("pybads"),
            pybads_revision=source_revision(pybads),
            gpyreg=version("gpyreg"),
            gpyreg_revision=source_revision(gpyreg),
            numpy=np.__version__,
            scipy=scipy.__version__,
            python=sys.version.split()[0],
            options="default",
        ),
        target=dict(
            kind="valley+trench",
            a=INFO["a"],
            b=INFO["b"],
            A=INFO["A"],
            tstar=INFO["tstar"],
            w=INFO["w"],
            S=INFO["S"],
            lb=LB.tolist(),
            ub=UB.tolist(),
            plb=PLB.tolist(),
            pub=PUB.tolist(),
            x0=X0.tolist(),
            xmin=XMIN.tolist(),
            fmin=FMIN,
        ),
        grid=grid,
        evals=evals,
        steps=out_steps,
        states=states,
        result=dict(
            x=r(res["x"]),
            fval=float(res["fval"]),
            n=len(ev),
            iterations=int(res["iterations"]),
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=87)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dump", default=None, help="npz of every GP state")
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="print the steps, write no trace",
    )
    args = ap.parse_args()

    rec = run_traced(args.seed)
    print(
        f"traced: {len(rec.evals)} evaluations, {len(rec.steps)} steps, "
        f"{len(rec.states)} GP states, f = {rec.result['fval']!r}",
        flush=True,
    )
    if args.check:
        plain, res = run_plain(args.seed)
        traced = np.array([np.r_[e["x"], e["y"]] for e in rec.evals])
        same = plain.shape == traced.shape and np.array_equal(plain, traced)
        print(
            f"check: plain {len(plain)} evaluations, identical = {same}, "
            f"f = {res['fval']!r}",
            flush=True,
        )
        if not same:
            raise SystemExit("the recorder changed the run")
    for k, s in enumerate(rec.steps):
        e = s["evals"]
        if s["kind"] == "search":
            x = rec.evals[e[0]]["x"] if e else [np.nan, np.nan]
            y = rec.evals[e[0]]["y"] if e else np.nan
            print(
                f"  step {k:2d} search ev {e[0] + 1 if e else 0:3d} "
                f"{s['method']:6s} {s['status']:11s} "
                f"x=({x[0]:+.3f},{x[1]:+.3f}) y={y:+9.4f} "
                f"pred={s['mu_pick']:+9.4f}+-{s['sd_pick']:.3f} "
                f"state {s['state']} "
                f"ell={rec.states[s['state']]['ell'][0]:.2f},"
                f"{rec.states[s['state']]['ell'][1]:.2f}",
                flush=True,
            )
        else:
            print(
                f"  step {k:2d} poll   ev {[i + 1 for i in e]} "
                f"success={s['success']} msi {s['msi']} -> {s['msi1']} "
                f"states {s['states']}",
                flush=True,
            )
    if args.print_only:
        return
    trace = build(rec, args.first, args.last, args.seed)
    for s in trace["steps"]:
        if s["kind"] == "poll":
            for a in s["arms"]:
                print(
                    "  arm",
                    a["x"],
                    a["status"],
                    f"predicted {a['mu']:+.3f} +- {a['sd']:.3f}, bound {a['lcb']:+.3f}"
                    if "mu" in a
                    else "(not in the poll set)",
                )
            print(
                "   acquisition calls:", s["n_calls"], " incumbent:", s["f0"]
            )
    text = "window.BADS_TRACE = " + json.dumps(trace, separators=(",", ":"))
    Path(args.out).write_text(text + ";\n", encoding="utf-8")
    print(
        f"wrote {args.out}: {len(text) / 1e6:.2f} MB, "
        f"{len(trace['states'])} states, {len(trace['steps'])} steps, "
        f"z in [{trace['grid']['z_lo']:.2f}, {trace['grid']['z_hi']:.2f}], "
        f"sd up to {trace['grid']['sd_hi']:.2f}",
        flush=True,
    )
    if args.dump:
        np.savez_compressed(
            args.dump,
            mu=np.array([s["mu"] for s in rec.states]),
            sd=np.array([s["sd"] for s in rec.states]),
            n_evals=np.array([s["n_evals"] for s in rec.states]),
            ell=np.array([s["ell"] for s in rec.states]),
            evals=np.array([np.r_[e["x"], e["y"]] for e in rec.evals]),
        )
        print(f"wrote {args.dump}", flush=True)


if __name__ == "__main__":
    main()
