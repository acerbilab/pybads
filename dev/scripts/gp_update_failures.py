"""How often the guarded GP updates of PyBADS fail over a benchmark suite,
and how each guard ends, under the gpyreg that ``PYTHONPATH`` selects
(``dev/plans/gp-update-guards.md``, Phase 3).

Each (configuration, seed) of ``benchmark_targets.py`` runs as in
``population.py``: the same problem, options and seed, in a fresh spawned
process with one BLAS thread. ``gpyreg.GP.update``,
``GP.set_hyperparameters`` and ``GP.predict`` are wrapped for counting, and
so are ``add_and_update_gp`` and ``local_gp_fitting`` where ``bads.py``
calls them. For each run, one entry of the output JSON holds:

- ``calls``: the calls of the three guarded sites (``add_and_update_gp``,
  ``local_gp_fitting``, ``_get_target_from_gp_``) that compute a posterior
  (the outermost gpyreg call only, whose innermost PyBADS frame is the site);
- ``failures``: those that raised ``LinAlgError``, by site and method, and
  how many of them were injected;
- ``outcomes``: ``add_dropped`` (a point left out until the next rebuild),
  ``local_recovered`` (the previous hyperparameters on the new training
  set), ``local_restored`` (the GP of the entry, marked for a rebuild with a
  refit: a GP that ``local_gp_fitting`` returns with ``needs_refit``),
  ``target_current_gp`` (the target predicted from the current GP)
  and ``target_nonfinite`` (a non-finite target prediction, which falls
  back to the incumbent);
- ``restores_by_caller``: the restores of ``local_gp_fitting`` by the
  function that called it: ``_search_step_`` (its GP, and in a noisy run
  the copy it rebuilds around the search point), ``_poll_step_``, and
  ``_re_evaluate_history_`` (the GPs of ``IterationHistory``);
- ``max_restore_streak``: the most consecutive restores by
  ``_search_step_`` and ``_poll_step_`` without a successful rebuild by
  them in between;
- ``stale_evals``: the evaluations made while the GP of the search or the
  poll carried ``needs_rebuild``;
- ``final``: ``x``, ``fval``, ``func_count``, and the exception of a run
  that raised.

``--check DIR`` compares each run's ``x``, ``fval`` and ``func_count`` with
the record of the same run in the population ``DIR``, exactly: this shows
that the wrappers change nothing and that the runs are the population's.

``--inject P`` makes a fraction ``P`` of the guarded computations raise
``LinAlgError`` before they start. Each decision is drawn once per distinct
computation (the site, and a hash of the training inputs and targets, the
new point and the hyperparameters) from the script's own generator, seeded
by ``--inject-seed``, the configuration and the seed, never from
``bads.rng`` or NumPy's global stream. A retry of the same computation
therefore fails again, as a real failure does, so the restore of
``local_gp_fitting`` is reached. Usage, from the repository root::

    PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 python -u dev/scripts/gp_update_failures.py OUT.json --seeds 0-29 --workers 4 --check DIR
    PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 python -u dev/scripts/gp_update_failures.py OUT.json --seeds 0-9 --workers 4 --inject 0.02
"""

import argparse
import hashlib
import inspect
import json
import sys
import time
import traceback
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_targets as bt  # noqa: E402
import population as pop  # noqa: E402

SITES = ("add_and_update_gp", "local_gp_fitting", "_get_target_from_gp_")
STEPS = ("_search_step_", "_poll_step_")


def _pybads_callers(frame):
    """Names of the innermost PyBADS function on the stack, by module, and
    of the PyBADS function that called it (``None`` where there is none)."""
    names = []
    while frame is not None and len(names) < 2:
        if frame.f_globals.get("__name__", "").startswith("pybads."):
            names.append(frame.f_code.co_name)
        frame = frame.f_back
    return (names + [None, None])[:2]


def _innermost_pybads(frame):
    """Name of the innermost PyBADS function on the stack, by module."""
    return _pybads_callers(frame)[0]


def _digest(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            h.update(b"-")
            continue
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Probe:
    """Counters of one run, and the wrappers that feed them."""

    def __init__(self, inject_p, inject_rng):
        self.calls = Counter()
        self.failures = Counter()
        self.injected = 0
        self.outcomes = Counter()
        self.restores_by_caller = Counter()
        self.depth = 0
        self.streak = 0
        self.max_streak = 0
        self.local_pending = False
        self.stale_since = None
        self.stale_evals = 0
        self.inject_p = inject_p
        self.inject_rng = inject_rng
        self.decisions = {}

    # Injection -------------------------------------------------------

    def _key(self, site, gp, method, args):
        if method == "update":
            hyp = args["hyp"]
            if hyp is None:
                hyp = gp.get_hyperparameters(as_array=True)
            return (
                site,
                _digest(gp.X, gp.y, args["X_new"], args["y_new"], hyp),
            )
        hyp = args["hyp_new"]
        if not isinstance(hyp, np.ndarray):
            hyp = gp.hyperparameters_from_dict(hyp)
        return (site, _digest(gp.X, gp.y, None, None, hyp))

    def _inject(self, site, gp, method, args):
        if not self.inject_p:
            return False
        key = self._key(site, gp, method, args)
        if key not in self.decisions:
            self.decisions[key] = self.inject_rng.random() < self.inject_p
        return self.decisions[key]

    # Outcomes ----------------------------------------------------------

    def _ended(self, site, method, failed, caller):
        if site == "add_and_update_gp" and failed:
            self.outcomes["add_dropped"] += 1
        elif site == "_get_target_from_gp_" and failed:
            self.outcomes["target_current_gp"] += 1
        elif site == "local_gp_fitting" and method == "update" and failed:
            self.local_pending = True

    def _local_ended(self, gp, caller):
        """Classifies a call of ``local_gp_fitting`` by the GP it returns:
        a restored GP carries ``needs_refit``, which a rebuild that leaves a
        posterior removes. A restore makes no further call after the failed
        update when there was no refit to retry from."""
        step = caller in STEPS
        if gp.temporary_data.get("needs_refit", False):
            self.outcomes["local_restored"] += 1
            self.restores_by_caller[caller] += 1
            if step:
                self.streak += 1
                self.max_streak = max(self.max_streak, self.streak)
        else:
            if self.local_pending:
                self.outcomes["local_recovered"] += 1
            if step:
                self.streak = 0
        self.local_pending = False

    # Wrappers ----------------------------------------------------------

    def wrap_gp_method(self, cls, method):
        original = getattr(cls, method)
        signature = inspect.signature(original)

        def wrapper(gp, *a, **k):
            if self.depth:
                return original(gp, *a, **k)
            bound = signature.bind(gp, *a, **k)
            bound.apply_defaults()
            site, caller = _pybads_callers(sys._getframe(1))
            if site not in SITES or not bound.arguments["compute_posterior"]:
                self.depth += 1
                try:
                    return original(gp, *a, **k)
                finally:
                    self.depth -= 1
            self.calls[site] += 1
            injected = self._inject(site, gp, method, bound.arguments)
            self.depth += 1
            try:
                if injected:
                    raise np.linalg.LinAlgError("injected failure")
                out = original(gp, *a, **k)
            except np.linalg.LinAlgError:
                self.failures[f"{site}.{method}"] += 1
                self.injected += injected
                self._ended(site, method, True, caller)
                raise
            finally:
                self.depth -= 1
            self._ended(site, method, False, caller)
            return out

        setattr(cls, method, wrapper)

    def wrap_predict(self, cls):
        original = cls.predict

        def predict(gp, *a, **k):
            out = original(gp, *a, **k)
            if not self.depth and (
                _innermost_pybads(sys._getframe(1)) == "_get_target_from_gp_"
            ):
                if not np.all(np.isfinite(out[0])):
                    self.outcomes["target_nonfinite"] += 1
            return out

        cls.predict = predict

    def wrap_step_function(self, module, name, logger_index):
        original = getattr(module, name)

        def watched(*a, **k):
            out = original(*a, **k)
            caller = sys._getframe(1).f_code.co_name
            gp = out[0] if isinstance(out, tuple) else out
            if name == "local_gp_fitting":
                self._local_ended(gp, caller)
            if caller in STEPS:
                count = a[logger_index].func_count
                marked = gp.temporary_data.get("needs_rebuild", False)
                if marked and self.stale_since is None:
                    self.stale_since = count
                elif not marked and self.stale_since is not None:
                    if name == "local_gp_fitting":
                        self.stale_evals += count - self.stale_since
                        self.stale_since = None
            return out

        setattr(module, name, watched)

    def summary(self, func_count):
        stale = self.stale_evals
        if self.stale_since is not None and func_count is not None:
            stale += func_count - self.stale_since
        return {
            "calls": dict(self.calls),
            "failures": dict(self.failures),
            "injected": self.injected,
            "outcomes": dict(self.outcomes),
            "restores_by_caller": dict(self.restores_by_caller),
            "max_restore_streak": self.max_streak,
            "stale_evals": stale,
        }


def run_one(label, seed, inject_p, inject_seed):
    """One run, in its own process; returns its entry."""
    import gpyreg

    import pybads.bads.bads as bads_module
    from pybads import BADS

    inject_rng = np.random.default_rng(
        [inject_seed, seed, zlib.crc32(label.encode())]
    )
    probe = Probe(inject_p, inject_rng)
    probe.wrap_gp_method(gpyreg.GP, "update")
    probe.wrap_gp_method(gpyreg.GP, "set_hyperparameters")
    probe.wrap_predict(gpyreg.GP)
    probe.wrap_step_function(bads_module, "add_and_update_gp", 0)
    probe.wrap_step_function(bads_module, "local_gp_fitting", 2)

    prob = bt.find_config(label).make(seed=seed)
    args, options = prob.bads_args()
    final = {"x": None, "fval": None, "func_count": None, "exception": None}
    t0 = time.perf_counter()
    bads = None
    try:
        bads = BADS(*args, options=options)
        res = bads.optimize()
        final.update(
            x=np.asarray(res["x"], dtype=float).ravel().tolist(),
            fval=float(np.asarray(res["fval"]).item()),
            func_count=int(res["func_count"]),
        )
    except Exception as e:  # noqa: BLE001
        final["exception"] = f"{type(e).__name__}: {e}"
        final["traceback"] = traceback.format_exc()
        if bads is not None:
            final["func_count"] = int(bads.function_logger.func_count)
    entry = {"label": label, "seed": seed, "final": final}
    entry.update(probe.summary(final["func_count"]))
    entry["wall_s"] = time.perf_counter() - t0
    return entry


def check(entries, population_dir):
    """Mismatches between the entries and the population's records."""
    mismatches = []
    for e in entries:
        path = pop.record_path(Path(population_dir), e["label"], e["seed"])
        if not path.exists():
            mismatches.append((e["label"], e["seed"], "no record"))
            continue
        ref = json.loads(path.read_text(encoding="utf-8"))["final"]
        for key in ("x", "fval", "func_count"):
            if e["final"][key] != ref[key]:
                mismatches.append((e["label"], e["seed"], key))
    return mismatches


def report(entries):
    by_label = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)
    lines = [
        "| config | runs | runs with a failure | failures (injected) |"
        " add_dropped | local_recovered | local_restored |"
        " target_current_gp | target_nonfinite | max streak |"
        " stale evals | crashed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, es in by_label.items():
        out = Counter()
        for e in es:
            out.update(e["outcomes"])
        lines.append(
            f"| {label} | {len(es)}"
            f" | {sum(1 for e in es if e['failures'])}"
            f" | {sum(sum(e['failures'].values()) for e in es)}"
            f" ({sum(e['injected'] for e in es)})"
            + "".join(
                f" | {out[k]}"
                for k in (
                    "add_dropped",
                    "local_recovered",
                    "local_restored",
                    "target_current_gp",
                    "target_nonfinite",
                )
            )
            + f" | {max(e['max_restore_streak'] for e in es)}"
            f" | {sum(e['stale_evals'] for e in es)}"
            f" | {sum(1 for e in es if e['final']['exception'])} |"
        )
    calls = Counter()
    for e in entries:
        calls.update(e["calls"])
    lines.append("")
    lines.append(
        "Guarded calls: "
        + ", ".join(f"{site} {calls[site]}" for site in SITES)
        + "."
    )
    return "\n".join(lines)


def main(argv=None):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--suite", default="default")
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--only", default=None, help="comma-separated labels")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inject", type=float, default=0.0)
    ap.add_argument("--inject-seed", type=int, default=0)
    ap.add_argument("--check", type=Path, default=None)
    args = ap.parse_args(argv)

    bt.single_thread_env()  # inherited by the spawned processes
    cfgs = bt.suite_configs(args.suite)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cfgs = [c for c in cfgs if c.label in wanted]
    tasks = [(c.label, s) for c in cfgs for s in pop.parse_seeds(args.seeds)]
    print(
        f"[gp_update_failures] {len(tasks)} runs, {args.workers} worker(s),"
        f" inject {args.inject}",
        flush=True,
    )
    entries = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        max_tasks_per_child=1,
    ) as ex:
        futs = [
            ex.submit(run_one, label, seed, args.inject, args.inject_seed)
            for label, seed in tasks
        ]
        for k, fut in enumerate(as_completed(futs), 1):
            e = fut.result()
            entries.append(e)
            print(
                f"[gp_update_failures] {k}/{len(tasks)} {e['label']}"
                f"_seed{e['seed']} failures={sum(e['failures'].values())}"
                f" outcomes={e['outcomes']}"
                f"{' CRASH ' + e['final']['exception'] if e['final']['exception'] else ''}"
                f" [{(time.time() - t0) / 60:.1f} min]",
                flush=True,
            )
    entries.sort(key=lambda e: (e["label"], e["seed"]))
    out = {
        "meta": {
            "git": pop.git_info(),
            "gpyreg_source": pop.module_source("gpyreg"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "suite": args.suite,
            "seeds": args.seeds,
            "inject": args.inject,
            "inject_seed": args.inject_seed,
        },
        "runs": entries,
    }
    status = 0
    if args.check is not None:
        mismatches = check(entries, args.check)
        out["check"] = {
            "population": str(args.check),
            "mismatches": [list(m) for m in mismatches],
        }
        print(
            f"[gp_update_failures] check against {args.check}:"
            f" {len(mismatches)} mismatches",
            flush=True,
        )
        status = 1 if mismatches else 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(report(entries), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
