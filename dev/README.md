# Developer notes

`dev/` is for human review: major findings, proposals, discussions and
consolidated decisions. Use dated names (`YYYY-MM-DD-short-slug.md`) for
these notes. Update or consolidate a related narrative instead of creating
a top-level file for each agent, working session or experiment phase.
Separate notes are appropriate for genuinely separate topics.

- `TODO.md` lists the open work.
- `plans/` holds implementation plans, checklists and execution worklogs.
  Keep them current while the work is open, updating the existing file in
  place, and retain them afterwards.
- `results/` holds detailed findings and experiment writeups, with dated
  names and no further date or campaign subdirectory. A top-level note
  summarizes the evidence and decisions and links to them.
- `experiments/` holds the machine-readable evidence that a result cites.
- `scripts/` holds developer tooling that is not part of the package or the
  test suite. Run it from the repository root with the project venv, as
  `python dev/scripts/<name>.py`. Its output goes under `scripts/runs/`,
  which is gitignored: a result that matters is summarized in a plan or a
  result, not committed raw. A machine that keeps raw artifacts there lists
  them in the gitignored `scripts/runs/LOCAL.md`; tracked documents point
  at that file and never say "this machine".

A directory is created with its first file. These are maintainer records,
not user documentation: `docs/` is gitignored Sphinx output published to
`gh-pages`, so it cannot hold source notes.

## Scripts

Keep to one heavy process at a time: a population run, the test suite and
the example scripts never run concurrently, and `population.py run` uses
one worker unless `--workers` says otherwise. A long run writes an
unbuffered log of its own under `scripts/runs/` and is read from the log:

```console
python -u dev/scripts/<name>.py ... > dev/scripts/runs/<name>_$(date +%s).log 2>&1
```

- `fingerprint.py` prints one hash of six seeded runs (three deterministic,
  three with inferred noise). A change that must not move results shows
  the same hash before and after, on one machine: BLAS and platform
  differences can change the value.
- `benchmark_targets.py` defines the benchmark problems (shifted sphere,
  ellipsoid, rotated Rosenbrock, Ackley and Rastrigin, with and without
  noise, one with a non-box constraint, one with infinite bounds, and two
  maximum-likelihood fits to real data, `timing` and `multisensory_s1`)
  and the suites `smoke` and `default`. `--list` prints the suites,
  `--check` verifies each target's minimum, bounds and noise, and the
  pinned likelihood values of the real-data targets, and `--smoke` runs
  each configuration of a suite once, in a fresh process as a population
  does, and prints its wall time with the projected time of 30 seeds. The
  `default` suite runs every configuration at BADS's default budget,
  500 D, so that each run ends on BADS's own termination criteria; a
  population of 30 seeds takes about 80 minutes.
- `data/` holds the data of the real-data targets, copied from PyVBMC,
  and their reference minima, `reference_optima.json`, against which the
  error of a run on those targets is measured; `data/README.md` describes
  the files and their provenance.
- `make_reference_optima.py` computes the reference minima (BADS restarts
  at a long budget, the best of them polished with SciPy) and writes
  `data/reference_optima.json`, in about 7 minutes. Rerun it when a
  real-data likelihood or its data changes.
- `population.py` runs, summarizes and compares populations of seeded
  runs. `run --suite default --seeds 0-29 --out DIR` writes one JSON record
  per run (result, error against the target's minimum, effective options,
  provenance) and skips the runs already recorded, so it resumes after an
  interruption. `summary DIR` writes `DIR/summary.md`. `compare REF NEW`
  tests each configuration for a change in the error and in the number of
  evaluations, with one Holm correction over all the tests, prints effect
  sizes, and exits 1 on a flag; `compare REF --split` compares the even
  and the odd seeds of one population, as a null check. The paired test
  of `compare` assumes that both populations share each seed's start point
  and noise, that is, the same `benchmark_targets.py`; `compare` warns when
  the recorded start points differ. To run against another gpyreg
  checkout, put it on `PYTHONPATH`: the records identify gpyreg by its
  source path and commit, since the version string is that of the
  installed gpyreg.
- `calibrate_budgets.py` runs each configuration at 500 D for a few seeds
  and records where the runs end: the evidence behind the suite's budgets.
- `gpyreg_issue_checks.py` runs the known-noise path (`fit_lik=False`) and
  counts the failed fits inside `_robust_gp_fit_` over the suite, under the
  gpyreg that `PYTHONPATH` selects.
- `tolerance_sweep.py` runs the seeded optimization tests of
  `pybads/testing/bads/test_bads_optimization.py` over a range of seeds,
  through the test functions with their tolerances disabled, and records
  the error of each run; `summary LOG` prints, per test, the largest and
  the median error, the evaluations and the ratio of the tolerance to the
  largest error. Seeds 0-99 take about 40 minutes.
- `test_population.py` checks the record schema, the reference minima of
  the real-data targets, resumability and the statistics of `compare`:
  `python -m pytest dev/scripts/test_population.py`.

A population's raw output goes to `scripts/runs/population/<name>/`. A
population that serves as a reference for later comparisons is copied
(the JSON records and `summary.md`) to `experiments/<name>/`, named
`population_<purpose>_<YYYYMMDD>` after the day of the run, with a
`README.md` that holds the command, the provenance, the null check, a
positive control, and the smallest effect the comparison detects at the
reference's number of seeds.

## Index

- [experiments/population_gpyreg133_20260924/](experiments/population_gpyreg133_20260924/README.md)
  — the reference population of the benchmark (default suite, 30 seeds,
  draws through a `numpy.random.Generator`, gpyreg 1.3.3), with its null
  check.
- [gpyreg 1.3.3 for PyBADS](results/2026-09-25-gpyreg-1.3.3.md) — the
  suite, the agreement of 1.3.2 and 1.3.3 at default options, and the
  benchmark comparison with 1.3.1 (five configurations flagged, each
  improved by 1.3.2's low-noise predictions; no run outside the low-noise
  regime changed) behind the move of the minimum and the CI pin to 1.3.3.
- [experiments/population_generator_20260924/](experiments/population_generator_20260924/README.md)
  — the previous reference (gpyreg 1.3.1), with its comparison with the
  baseline that validated the generator.
- [experiments/population_baseline_20260924/](experiments/population_baseline_20260924/README.md)
  — the first reference (global random stream), with the positive control
  and the detectable effect sizes that the later references cite.
- [plans/tooling-and-rng.md](plans/tooling-and-rng.md) — repository
  tooling after PyVBMC's (formatting, CI with gpyreg pinned, packaging,
  changelog, release), seed tests, the benchmark suite and population
  comparison, random draws through a `numpy.random.Generator`, and the
  assessment of gpyreg 1.3.3 for PyBADS.
- [Codebase survey](results/2026-09-23-codebase-survey.md) — failures
  observed in the test suite at `273a5b7`, the candidate defects found by a
  read of the code (not verified), and the tests that checked less than
  they appeared to, with their fixes, the seed sweep behind the tolerances
  of the optimization tests and three candidate defects found on the way.
  The starting point of the deferred bug hunt in `TODO.md`.
