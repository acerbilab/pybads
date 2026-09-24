"""Reference minima of the real-data benchmark targets.

The error of a benchmark run is ``f_true(x) - f_min``. For the real-data
targets of ``benchmark_targets.py`` (``REAL_TARGETS``, negative
log-likelihoods) ``f_min`` is not analytic: this script computes it once
and stores it in ``data/reference_optima.json``, which
``benchmark_targets.py`` reads.

For each target: BADS restarts from start points drawn uniformly in the
plausible box (``--restarts``, from a stream seeded by ``START_SEED`` and
the target's name), on the deterministic target, at a long budget
(``--budget``, a multiple of ``D``); then ``scipy.optimize.minimize``
polishes the best few distinct restart results (``--polish``) within the
hard bounds, with L-BFGS-B (finite-difference gradients) followed by
Nelder-Mead from its result. For ``timing`` the paper's maximum-likelihood
point (``paper_mle`` in ``timing.npz``) is a candidate too. The best
candidate is the reference: ``x_min``, and ``f_min = f_true(x_min)``
evaluated by this implementation.

Each target's entry records ``x_min``, ``f_min`` and ``source`` (the
candidate chosen), every restart (start point, result, value, evaluations,
message, wall time), the spread of the restart values above ``f_min``,
each polish, for ``timing`` the comparison with the paper's point, the
settings and the provenance (git state, versions, the source and commit of
the imported PyBADS and gpyreg, the SHA-256 of the data archives, start and
end times). ``--only`` recomputes some targets and keeps the others'
entries.

Run from the repository root, with gpyreg selected as for a population and
an unbuffered log; at the defaults it takes about 7 minutes, most of them
on ``timing`` (about 40 ms per evaluation)::

    python -u dev/scripts/make_reference_optima.py \
        > dev/scripts/runs/make_reference_optima_$(date +%s).log 2>&1
"""

import argparse
import hashlib
import json
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_targets as bt  # noqa: E402
import population as pp  # noqa: E402  (provenance helpers)

START_SEED = 20260925
NELDER_MEAD_MAXFEV = 200  # per dimension


def _starts(prob, n):
    rng = np.random.default_rng([START_SEED, zlib.crc32(prob.name.encode())])
    return prob.plb + rng.random((n, prob.D)) * (prob.pub - prob.plb)


def _restart(prob, x0, seed, budget):
    from pybads import BADS

    options = {
        "display": "off",
        "max_fun_evals": budget * prob.D,
        "uncertainty_handling": False,
        "random_seed": seed,
    }
    t0 = time.perf_counter()
    res = BADS(
        prob.f_true,
        x0.copy(),
        prob.lb.copy(),
        prob.ub.copy(),
        prob.plb.copy(),
        prob.pub.copy(),
        options=options,
    ).optimize()
    x = np.asarray(res["x"], dtype=float).ravel()
    return {
        "seed": seed,
        "x0": x0.tolist(),
        "x": x.tolist(),
        "f": prob.f_true(x),
        "func_count": int(res["func_count"]),
        "iterations": int(res["iterations"]),
        "message": str(res["message"]),
        "wall_s": time.perf_counter() - t0,
    }


def _polish(prob, x_start):
    """L-BFGS-B within the hard bounds, then Nelder-Mead from its result."""
    from scipy.optimize import minimize

    bounds = list(zip(prob.lb, prob.ub))
    t0 = time.perf_counter()
    r1 = minimize(prob.f_true, x_start, method="L-BFGS-B", bounds=bounds)
    x1 = np.clip(r1.x, prob.lb, prob.ub)
    f1 = prob.f_true(x1)
    r2 = minimize(
        prob.f_true,
        x1,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "xatol": 1e-7,
            "fatol": 1e-7,
            "maxfev": NELDER_MEAD_MAXFEV * prob.D,
        },
    )
    x2 = np.clip(r2.x, prob.lb, prob.ub)
    f2 = prob.f_true(x2)
    x, f = (x2, f2) if f2 < f1 else (x1, f1)
    return {
        "f_start": prob.f_true(x_start),
        "lbfgsb": {
            "f": f1,
            "nfev": int(r1.nfev),
            "nit": int(r1.nit),
            "message": str(r1.message),
        },
        "nelder_mead": {
            "f": f2,
            "nfev": int(r2.nfev),
            "message": str(r2.message),
        },
        "x": x.tolist(),
        "f": f,
        "wall_s": time.perf_counter() - t0,
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _provenance():
    return {
        "git": pp.git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": pp.pkg_version("scipy"),
        "pybads": pp.pkg_version("pybads"),
        "pybads_source": pp.module_source("pybads"),
        "gpyreg": pp.pkg_version("gpyreg"),
        "gpyreg_source": pp.module_source("gpyreg"),
        "data_sha256": {
            f.name: _sha256(f) for f in sorted(bt.DATA_DIR.glob("*.npz"))
        },
    }


def reference(name, D, n_restarts, budget, n_polish):
    """The entry of one target in ``reference_optima.json``."""
    started = _now()
    t_start = time.perf_counter()
    prob = bt.make_problem(name, D, seed=0, reference=False)
    tag = f"[{name}]"
    print(
        f"{tag} D = {D}: {n_restarts} BADS restarts at {budget} D ="
        f" {budget * D} evaluations, then {n_polish} polishes",
        flush=True,
    )
    restarts = []
    for i, x0 in enumerate(_starts(prob, n_restarts)):
        r = _restart(prob, x0, i, budget)
        restarts.append(r)
        print(
            f"{tag} restart {i:2d}: f = {r['f']:.6f}  evals"
            f" {r['func_count']:5d}  {r['wall_s']:6.1f} s"
            f"  {r['message'][:60]}",
            flush=True,
        )
    candidates = [
        (r["f"], np.array(r["x"]), f"BADS restart {r['seed']}")
        for r in restarts
    ]
    # the best restarts at distinct points
    order = np.argsort([r["f"] for r in restarts], kind="stable")
    picked = []
    for i in order:
        x = np.array(restarts[i]["x"])
        if all(not np.allclose(x, restarts[j]["x"]) for j in picked):
            picked.append(int(i))
        if len(picked) == n_polish:
            break
    polishes = []
    for i in picked:
        p = _polish(prob, np.array(restarts[i]["x"]))
        p["restart"] = i
        polishes.append(p)
        candidates.append((p["f"], np.array(p["x"]), f"polish of restart {i}"))
        print(
            f"{tag} polish of restart {i:2d}: {p['f_start']:.6f} ->"
            f" L-BFGS-B {p['lbfgsb']['f']:.6f} ({p['lbfgsb']['nfev']} evals)"
            f" -> Nelder-Mead {p['nelder_mead']['f']:.6f}"
            f" ({p['nelder_mead']['nfev']} evals), {p['wall_s']:.1f} s",
            flush=True,
        )
    best_found = min(c[0] for c in candidates)
    paper = None
    if name == "timing":
        data = bt._load_data("timing")
        x_paper = np.array(data["paper_mle"], dtype=float)
        f_paper = -float(data["paper_mle_val"])
        f_port = prob.f_true(x_paper)
        paper = {
            "x": x_paper.tolist(),
            "f_paper": f_paper,
            "f_here": f_port,
            "f_here_minus_f_paper": f_port - f_paper,
            "best_found_minus_f_here": best_found - f_port,
        }
        candidates.append((f_port, x_paper, "the paper's MLE (paper_mle)"))
        print(
            f"{tag} paper MLE: -loglik {f_paper:.6f} in the paper,"
            f" {f_port:.6f} here (difference {f_port - f_paper:.2e}); best"
            f" found minus the paper's point here: {best_found - f_port:+.6f}",
            flush=True,
        )
    _, x_min, source = min(candidates, key=lambda c: c[0])
    f_min = prob.f_true(x_min)  # the value --check compares with
    f_restarts = np.array([r["f"] for r in restarts])
    gaps = f_restarts - f_min
    spread = {
        "min": float(np.min(gaps)),
        "quartiles": np.percentile(gaps, [25, 50, 75]).tolist(),
        "max": float(np.max(gaps)),
        "within_tol": int(np.sum(gaps < bt.TOL_REAL)),
        "within_1e-3": int(np.sum(gaps < 1e-3)),
        "tol": bt.TOL_REAL,
    }
    print(
        f"{tag} f_min = {f_min!r} from {source}; x_min = {x_min.tolist()}",
        flush=True,
    )
    print(
        f"{tag} restarts above f_min: min {spread['min']:.2e}, median"
        f" {spread['quartiles'][1]:.2e}, max {spread['max']:.2e};"
        f" {spread['within_tol']}/{n_restarts} within {bt.TOL_REAL},"
        f" {spread['within_1e-3']}/{n_restarts} within 1e-3",
        flush=True,
    )
    entry = {
        "D": D,
        "x_min": x_min.tolist(),
        "f_min": f_min,
        "source": source,
        "restart_spread": spread,
        "restarts": restarts,
        "polishes": polishes,
    }
    if paper is not None:
        entry["paper_mle"] = paper
    entry["settings"] = {
        "restarts": n_restarts,
        "budget": budget,
        "max_fun_evals": budget * D,
        "polish": n_polish,
        "start_seed": START_SEED,
        "nelder_mead_maxfev": NELDER_MEAD_MAXFEV * D,
    }
    entry["provenance"] = _provenance()
    entry["provenance"].update(started=started, finished=_now())
    entry["elapsed_s"] = time.perf_counter() - t_start
    print(f"{tag} done in {entry['elapsed_s'] / 60:.1f} min", flush=True)
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated target names (default: all real-data targets)",
    )
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument(
        "--budget",
        type=int,
        default=1000,
        help="max_fun_evals of each restart, as a multiple of D",
    )
    ap.add_argument("--polish", type=int, default=3, help="restarts to polish")
    ap.add_argument("--out", type=Path, default=bt.REFERENCE_OPTIMA)
    args = ap.parse_args(argv)
    names = list(bt.REAL_TARGETS)
    if args.only:
        names = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = sorted(set(names) - set(bt.REAL_TARGETS))
        if unknown:
            ap.error(f"not real-data targets: {', '.join(unknown)}")
    doc = {"targets": {}}
    if args.out.is_file():
        doc = json.loads(args.out.read_text(encoding="utf-8"))
    doc["about"] = (
        "Reference minima of the real-data targets of"
        " dev/scripts/benchmark_targets.py, written by"
        " dev/scripts/make_reference_optima.py (see its docstring and"
        " data/README.md)."
    )
    for name in names:
        doc["targets"][name] = reference(
            name,
            bt.REAL_TARGETS[name],
            args.restarts,
            args.budget,
            args.polish,
        )
        # written after each target, so a finished target survives an
        # interruption of the next
        doc["targets"] = dict(sorted(doc["targets"].items()))
        tmp = args.out.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(pp.jsonable(doc), indent=1) + "\n", encoding="utf-8"
        )
        tmp.replace(args.out)
        print(f"[make_reference_optima] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
