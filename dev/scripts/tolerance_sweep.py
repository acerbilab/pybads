"""Errors of the seeded optimization tests over a range of seeds: the
evidence behind their tolerances. Run from the repository root.

Each test of ``pybads/testing/bads/test_bads_optimization.py`` runs through
its own test function, at ``SEED = s`` and ``NOISE_SEED = s + 1000`` for each
seed ``s``, with its tolerance disabled, and the error that the test compares
with its tolerance is recorded, with the tolerance itself.

``run`` prints one JSON record per run on stdout (BADS's own output and
warnings go to stderr); ``summary LOG`` prints, per test, the largest and
the median error, the range of evaluations, the tolerance, the ratio of the
tolerance to the largest error, and the runs that crashed or reached the
tolerance::

    python -u dev/scripts/tolerance_sweep.py run --seeds 0-99 > dev/scripts/runs/test_tolerances/sweep_$(date +%s).log 2>&1
    python dev/scripts/tolerance_sweep.py summary dev/scripts/runs/test_tolerances/sweep_<time>.log

Seeds 0-99 of every test take about 40 minutes.
"""

import argparse
import collections
import json
import logging
import sys
import time

import numpy as np


def parse_seeds(text):
    """``"0-99"`` or ``"0,5,7"`` or a mix, to a list of integers."""
    seeds = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def run(seeds, names=None):
    """Yield one record per test and seed."""
    import pybads.testing.bads.test_bads_optimization as t

    tests = {
        name: test
        for name, test in vars(t).items()
        if name.startswith("test_") and callable(test)
    }
    if names:
        unknown = set(names) - set(tests)
        if unknown:
            raise SystemExit(f"unknown tests: {sorted(unknown)}")
        tests = {name: tests[name] for name in names}

    original_run_bads = t.run_bads
    original_seeds = t.SEED, t.NOISE_SEED
    captured = {}

    def run_bads(*args, tol_err, **kwargs):
        optimize_result, err = original_run_bads(
            *args, tol_err=np.inf, **kwargs
        )
        captured.update(
            err=err,
            tol_err=tol_err,
            func_count=int(optimize_result["func_count"]),
        )
        return optimize_result, err

    t.run_bads = run_bads
    try:
        for name, test in tests.items():
            for seed in seeds:
                t.SEED, t.NOISE_SEED = seed, seed + 1000
                captured.clear()
                start = time.perf_counter()
                record = {"test": name, "seed": seed}
                try:
                    test()
                except Exception as e:  # a crash, or an assertion of the test
                    record["error"] = repr(e)
                if "err" not in captured and "error" not in record:
                    record["error"] = "the test did not call run_bads"
                record.update(captured)
                record["time"] = round(time.perf_counter() - start, 2)
                yield record
    finally:
        t.run_bads = original_run_bads
        t.SEED, t.NOISE_SEED = original_seeds


def summary(path):
    records = collections.defaultdict(list)
    with open(path, encoding="utf-8") as log:
        for line in log:
            if line.startswith("{"):
                record = json.loads(line)
                records[record["test"]].append(record)
    for name, rs in records.items():
        done = [r for r in rs if "err" in r]
        errs = np.array([r["err"] for r in done])
        print(f"{name}: {len(rs)} runs")
        if done:
            worst = done[int(np.argmax(errs))]
            tol = worst["tol_err"]
            evals = [r["func_count"] for r in done]
            print(
                f"  largest error {errs.max():.3g} (seed {worst['seed']}),"
                f" median {np.median(errs):.3g},"
                f" evaluations {min(evals)}-{max(evals)},"
                f" tolerance {tol:g} ({tol / errs.max():.3g} times the"
                " largest)"
            )
            over = [r["seed"] for r in done if r["err"] >= r["tol_err"]]
            if over:
                print(f"  at or above the tolerance: seeds {over}")
        for r in rs:
            if "error" in r:
                print(f"  seed {r['seed']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the tests over seeds")
    p_run.add_argument("--seeds", default="0-99", help="e.g. 0-99 or 0,3,7")
    p_run.add_argument("--tests", nargs="*", help="test names (default all)")
    p_sum = sub.add_parser("summary", help="summarize a log of `run`")
    p_sum.add_argument("log")
    args = parser.parse_args()

    if args.command == "summary":
        summary(args.log)
        return
    # BADS configures the root logger on stdout unless it has a handler:
    # keep stdout for the records, and only warnings on stderr.
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    logging.basicConfig(handlers=[handler], format="%(message)s")
    for record in run(parse_seeds(args.seeds), args.tests):
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
