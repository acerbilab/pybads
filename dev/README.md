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
  installed gpyreg. PyBADS, and `benchmark_targets.py` with its targets
  and seeds, come from the checkout that holds the script, which it puts
  first on `sys.path`: to run a commit, run the
  `dev/scripts/population.py` of a worktree at it, from the main
  checkout's root.
- `calibrate_budgets.py` runs each configuration at 500 D for a few seeds
  and records where the runs end: the evidence behind the suite's budgets.
- `gpyreg_issue_checks.py` runs the known-noise path (`fit_lik=False`) and
  counts the failed fits inside `_robust_gp_fit_` over the suite, under the
  gpyreg that `PYTHONPATH` selects.
- `gp_update_failures.py` runs a suite as `population.py` does, counts the
  failures of the three guarded GP updates (`add_and_update_gp`,
  `local_gp_fitting`, `_get_target_from_gp_`) and how each guard ended, and
  writes one JSON file. `--check DIR` compares the runs' results with a
  population's records. `--inject P` makes a fraction of the guarded
  computations fail, the same way each time a computation is repeated, as
  a stress test.
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

- [experiments/population_gpfixes_20260925/](experiments/population_gpfixes_20260925/README.md)
  — the reference population of the benchmark on Windows (default suite,
  30 seeds, gpyreg 1.3.3, at `ab4dded`: #67 and the three GP fixes of #66),
  with its null check, its comparison with the previous Windows reference,
  which flags the five configurations that the same fixes flag on Linux,
  all better, and the runs in which the prior of the GP mean falls outside
  the bounds of the mean and the log prior is NaN.
- [experiments/population_targetnoise_20260925/](experiments/population_targetnoise_20260925/README.md)
  — the previous reference on Windows (at `c044fea`, with the
  noise-variance fix of `020d6a8`), with its null check and its comparison
  with the one before, which flags nothing: the two configurations with
  target noise change, and the other 16 are identical run by run.
- [experiments/population_linux_gpfixes_20260925/](experiments/population_linux_gpfixes_20260925/README.md)
  — the reference population of the benchmark on Linux (default suite, 30
  seeds, gpyreg 1.3.3, at `97b2c66`: a repeated point merged into its own
  row, the GP mean prior re-centred at each rebuild, and the GP log length
  scales bounded by the log of the maximum), with its null check, the
  gates of each step, and its comparison with the previous Linux
  reference, which flags five configurations, all better: the
  deterministic ellipsoids and `rosenbrock_D6` end closer to their minima.
- [experiments/population_linux_targetnoise_20260925/](experiments/population_linux_targetnoise_20260925/README.md)
  — the previous reference on Linux (at `1c8c71d`, with the noise-variance
  fix of `020d6a8`), with its null check and its comparison with the one
  before, which flags nothing: the two configurations with target noise
  change, and the other 16 are identical run by run.
- [experiments/population_ellipsoid_hetero_20260925/](experiments/population_ellipsoid_hetero_20260925/README.md)
  — `ellipsoid_D3_hetero` over seeds 30-89 before and after `020d6a8`: with
  the reference seeds, the median error rises from 0.21 to 0.54 over 90
  paired seeds.
- [experiments/population_ellipsoid_hetero_linux_20260925/](experiments/population_ellipsoid_hetero_linux_20260925/README.md)
  — the same regression on Linux over seeds 0-89 (median error 0.18 to
  0.58), and the candidate causes: the three of `TODO.md`, the defect found
  beside them (`032dfcb`), the bound of the GP length scales (`97b2c66`),
  which brings the flat axis back, and the removal of evaluated points;
  with the three commits the median is 0.25.
- [experiments/population_linux_20260925/](experiments/population_linux_20260925/README.md)
  — the previous reference on Linux (default suite, 30 seeds, gpyreg
  1.3.3, the guards of `plans/gp-update-guards.md`, before `020d6a8`),
  identical run by run to the same code before the guards, with its null
  check and an information-only comparison with the previous Windows
  reference (`population_gpyreg133_20260924`).
- [experiments/population_gpyreg133_20260924/](experiments/population_gpyreg133_20260924/README.md)
  — the previous reference on Windows (gpyreg 1.3.3, before `020d6a8`),
  with its null check.
- [gpyreg 1.3.3 for PyBADS](results/2026-09-25-gpyreg-1.3.3.md) — the
  suite, the agreement of 1.3.2 and 1.3.3 at default options, and the
  benchmark comparison with 1.3.1 (five configurations flagged, each
  improved by 1.3.2's low-noise predictions; no run outside the low-noise
  regime changed) behind the move of the minimum and the CI pin to 1.3.3.
- [experiments/population_generator_20260924/](experiments/population_generator_20260924/README.md)
  — an earlier reference (gpyreg 1.3.1), with its comparison with the
  baseline that validated the generator.
- [experiments/population_baseline_20260924/](experiments/population_baseline_20260924/README.md)
  — the first reference (global random stream), with the positive control
  and the detectable effect sizes that the later references cite.
- [plans/port-correctness-review.md](plans/port-correctness-review.md) —
  the independent correctness review of the port against MATLAB BADS
  v1.1.3, after PyVBMC's: slices, waves, the reviewer brief, the gates and
  the worklog; its records (the known-differences sheet, the counterpart
  map, the reviewers' reports) under `experiments/port_review_20260925/`.
- [plans/gp-update-guards.md](plans/gp-update-guards.md) — guards on the
  three GP calls that stopped benchmark runs with `LinAlgError`, after
  MATLAB BADS: a consistent GP handed on, a rebuild (and, after a failed
  rebuild, a refit) at the next step, no change to runs without a failure;
  the failure counts and the stress run with injected failures.
- [plans/tooling-and-rng.md](plans/tooling-and-rng.md) — repository
  tooling after PyVBMC's (formatting, CI with gpyreg pinned, packaging,
  changelog, release), seed tests, the benchmark suite and population
  comparison, random draws through a `numpy.random.Generator`, and the
  assessment of gpyreg 1.3.3 for PyBADS.
- [Codebase survey](results/2026-09-23-codebase-survey.md) — failures
  observed in the test suite at `273a5b7`, the candidate defects found by a
  read of the code (not verified), and the tests that checked less than
  they appeared to, with their fixes, the seed sweep behind the tolerances
  of the optimization tests, three candidate defects found on the way, the
  checks behind running each test once in CI and requiring NumPy 2, the
  fixes of five small defects of `bads.py` with their gates, and the
  reruns of the four crashing benchmark runs, whose failing calls share
  degenerate GP hyperparameters. The starting point of the deferred bug
  hunt in `TODO.md`.
