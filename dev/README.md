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
  the same hash before and after, on one machine and with the same number
  of BLAS threads: BLAS, its thread count and platform differences can
  change the value.
- `replay.py` records short seeded runs step by step and compares two
  recordings exactly. `record` runs eight configurations of the benchmark
  (`sphere_D2`, `ellipsoid_D3`, `rosenbrock_D6`, `sphere_D3` with both
  noise kinds, `sphere_nonbox_D3`, `ellipsoid_D3_unbounded` and
  `logsphere_D3`) at seed 0 and 50 D evaluations, each in a fresh process
  with one BLAS thread and, on x86_64, OpenBLAS's Haswell kernels, in
  about 30 s, and writes one trace per run under `scripts/runs/replay/`:
  every call of the target with the state of the run's generator, every
  search and poll step, every GP computation with its hyperparameters,
  `iteration_history`, the result, and the platform key (CPU, libraries,
  BLAS kernel and threads). `check BASE NEW` refuses two recordings whose
  platform keys differ (unless `--force`), warns when they differ in the
  gpyreg that ran, the requested options or the budget scale, and
  reports, per run, identity or the first divergence: the evaluation, its
  iteration and stage, whether the generator's states agree there (a
  value moved) or not (a branch changed), and the first step and the
  earliest GP computation that differ; it exits 1 unless every run is
  identical. For a change that must move nothing, record at the parent
  commit, in a worktree at it, and at the change, on one machine, and
  check the two. Both sides record with the change's `replay.py`, copied
  alone into the parent's worktree when the parent has an older one or
  none (a commit from before 2026-09-28): traces written by two versions
  of the tool can differ in their layout, which `check` reports as runs
  that differ. The copy takes the parent's `benchmark_targets.py` and
  `population.py`, which have what it needs, the default configurations
  included, at every commit from `0d866e84` (2026-09-27) on; at an older
  commit the recording stops with `MissingName`, since its `BADS` does not
  set `poll_moved`, which the recorder reads. `--repeat 2` records each
  run twice in one process, and `check DIR` compares the repeats. `report
  DIR` tabulates a recording. The replay is exact only on one machine, one
  set of versions, one BLAS kernel and one thread count, and not on macOS
  arm64, where two runs of one seed need not match bit for bit
  ([results/2026-09-28-macos-arm64-repeatability.md](results/2026-09-28-macos-arm64-repeatability.md)).
  Measured at `948e0d96` on Linux (a container with 4 virtual CPUs, Intel
  Xeon at 2.10 GHz; NumPy 2.4.6, SciPy 1.17.1, OpenBLAS 0.3.31, gpyreg
  1.3.3), a seeded run repeats exactly, in one process and across processes.
  Under `OPENBLAS_CORETYPE=Sandybridge` (`--coretype Sandybridge`) the first
  GP fit of every run differs, by 1e-15 to 1e-10, and every run of the
  default set parts after 15 to 40 evaluations, into different decisions
  (the values of `rosenbrock_D6`, whose target rotates its input by a matrix
  product, differ from the first evaluation, and its points after 24). With
  four threads, seven of the eight runs part after 22 to 50 evaluations, and
  `sphere_D3_homo` differs only in its GP hyperparameters, by about 1e-10.
  So it is a developer tool, not a test: the part of a run that involves no
  BLAS work, the initial design, is pinned on every platform by
  `pybads/testing/bads/test_initial_design_pin.py`.
- `make_oracle_fixtures.py` writes and checks the oracles of
  `pybads/testing/oracles/`: the states of six short seeded runs
  (`_recipes.py`: deterministic at D = 2 and 3, with inferred noise, with
  the target's noise, with log-transformed variables and with a non-box
  constraint), taken at the start of a search or poll step and saved as
  plain arrays and JSON, with the outputs of PyBADS's components computed
  from them: the GP's predictions, the LCB, the variable transform, the
  grid functions, `contraints_check`, `_gp_hyp`, the choice of the local
  training set, the ES search's set-up and generations, the hedge, the
  improvement and Sto-BADS's outcome, and `poll_mads_2n`. Where a
  component draws, its draws are prescribed: exact arithmetic on PCG64's
  raw stream, the same everywhere. The fixtures store the portable outputs
  alone, those that rounding on another platform moves by less than their
  tolerances, measured across BLAS threads and kernels (the docstring of
  `_oracles.py`); the tests (`pytest pybads/testing/oracles`, about 4 s)
  compare them on every platform, and so does `--check`, which exits 1 on
  a failure. The platform-bound outputs are not stored: a GP refit, a whole
  ES search step, and the outputs through the solve of a GP whose
  condition number exceeds 1e8, as it does after a refit on three of the
  six states; for those three, a view with the GP's noise raised to bound
  the condition number by 1e6 keeps the arithmetic of the GP's
  predictions, the LCB, the training set and the hedge covered on every
  platform, in a smoother regime than the run's GP (a noise SD of 10 to 65
  against training values of median 0.25 to 5.7; predictions that
  correlate with the stored view's by 0.69 to 0.998), so that their values
  in the near-interpolating regime of the run's GP are covered only by
  `--dump` and `--against`, on one machine. Every mode reports the outputs
  it compared and those it left out, and why. `--check --exact` compares
  bit for bit, with one BLAS thread (the script's default), and refuses
  under another platform key than the fixtures'; on any machine, `--dump
  DIR` at the parent commit and `--check --exact --against DIR` at the
  change compare every output, the platform-bound ones included: the gate
  for a change that must move nothing. `--rebaseline ORACLE --reason TEXT`
  replaces one oracle's references, for a change that moves it on purpose,
  after the check of its decisions' margins that `--write` makes, and
  records the reason and the commit in the fixtures; `--write --reason TEXT`
  reruns the recipes, a new baseline, from a clean checkout. An option of a
  stored state that the code no longer has is dropped when the state is
  rebuilt, and listed; a key that the code reads from `optim_state` or a
  GP's `temporary_data` and that a stored state lacks gets a default in
  `STATE_DEFAULTS` of `_state.py`, in the commit that makes the code read
  it; `test_every_case_computes` computes every oracle, the platform-bound
  ones included, and fails on such a key. The oracles gate a component's
  numbers on fixed inputs, not a run: whole trajectories are `replay.py`'s,
  on one machine, and the distribution of results the population
  comparison's.
- `benchmark_targets.py` defines the benchmark problems (shifted sphere,
  ellipsoid, rotated Rosenbrock, Ackley and Rastrigin, with and without
  noise, one with a non-box constraint, one with infinite bounds, a sphere
  in log-scaled variables, `logsphere`, a sphere whose minimum lies on a
  hard bound, `edgesphere`, a nonsmooth valley along the diagonal,
  `ridge`, a sphere in a thin feasible band, `sphere_band`, and two
  maximum-likelihood fits to real data, `timing` and `multisensory_s1`)
  and the suites `smoke`, `default`, `oned` (the configurations at D = 1),
  `bounds` (plausible bounds omitted, a start on a hard bound, and
  `logsphere`: the setup's checks of the bounds and the start),
  `geometry` (`edgesphere`, `ridge` and `sphere_band`: the gates of W3-1
  and W3-24 of the port review), `thinband` (`sphere_band` at D = 2 and 3
  with inferred noise and with the target's noise, whose GP starts on one
  point: the gate of a change to that GP), `warmstart` (the sphere and the
  ellipsoid at D = 3 and Rosenbrock's function at D = 6 without noise, and
  the sphere at D = 3 with both kinds of noise, each run given as
  `precomputed_evaluations` the function log of an earlier BADS run: of 15 D
  evaluations at the run's seed and start, a rerun, whose log holds the
  run's initial design, or of 20 D at the seed plus 1000; the earlier run is
  made in the run's process, and the record names its log by a digest; the
  gate of a change to how a run uses evaluations made before it) and
  `profile` (the seven configurations whose time `profile_suite.py`
  measures). `--list` prints the suites, `--check` verifies each target's
  minimum, bounds and noise, and the pinned likelihood values of the
  real-data targets, and `--smoke` runs each configuration of a suite once,
  in a fresh process as a population does, and prints its wall time with the
  projected time of 30 seeds. The `default` suite runs every configuration
  at BADS's default budget, 500 D, so that each run ends on BADS's own
  termination criteria; a population of 30 seeds takes about 80 minutes. The
  processes that the tools start for their runs have one BLAS thread, with
  the variables of `THREAD_VARS` (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS`) set to 1, and their
  records hold those variables.
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
  per run (result, error against the target's minimum, the run's stage
  times, effective options, provenance) and skips the runs already
  recorded, so it resumes after an interruption. `summary DIR` writes
  `DIR/summary.md`. `compare REF NEW` tests each configuration for a
  change in the error and in the number of evaluations, with one Holm
  correction over all the tests, prints effect sizes, and exits 1 on a
  flag; `compare REF --split` compares the even and the odd seeds of one
  population, as a null check. The paired test of `compare` assumes that
  both populations share each seed's start point and noise, that is, the
  same `benchmark_targets.py`; `compare` warns when the recorded start
  points differ. To run against another gpyreg checkout, put it on
  `PYTHONPATH`: the records identify gpyreg by its source path and commit,
  since the version string is that of the installed gpyreg. PyBADS, and
  `benchmark_targets.py` with its targets and seeds, come from the checkout
  that holds the script, which it puts first on `sys.path`: to run a commit,
  run the `dev/scripts/population.py` of a worktree at it, from the main
  checkout's root.
- `calibrate_budgets.py` runs each configuration at 500 D for a few seeds
  and records where the runs end: the evidence behind the suite's budgets.
- `gpyreg_issue_checks.py` runs the known-noise path (`fit_lik=False`,
  which `BADS` refuses since W1-32 of the port review) and counts the failed
  fits inside `_robust_gp_fit_` over the suite, under the gpyreg that
  `PYTHONPATH` selects.
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
- `profile_run.py --config LABEL --seed N` runs one configuration as
  `population.py` does and records where its time goes: the result, the
  run's stage times (`optim_state["stage_times"]`: each second charged to
  the innermost open stage, the target's evaluations to `target`) by
  path, top-level stage and leaf, with their entries and the residual
  `total_time` less the stages and the target, one row per iteration from
  `iteration_history["timer"]`, and with `--cprofile` a cProfile of
  `optimize()` and the cumulative times of a curated list of functions
  (`BUCKETS`). The profiler slows the run, most where calls are many and
  short: stage times come from the runs without it.
- `profile_suite.py` runs `profile_run.py` over a suite (`profile` by
  default) and a set of seeds (`--seeds`, as `0-2`; seed 0 by default),
  plain, under cProfile or both, one run per process and one BLAS thread,
  resumable, and writes `aggregate.json` and `aggregate.md` (medians over
  the seeds) in the campaign directory (`--out`, by default a new one
  under `scripts/runs/profile/`); it exits 1 when a run fails or leaves no
  summary, or when the campaign holds no run. `--probe CONFIG` times one
  configuration before and after the campaign, to show a machine that
  slowed down.
- `profile_compare.py BASE NEW` pairs the runs of two campaigns by
  configuration, seed and mode, and prints the median ratios of the wall
  time, the own time and each stage, a large control stage that the
  change does not reach (`--control`, `search_es` by default: a ratio far
  from 1 is the machine's speed, not the code's), whether each pair ran
  the same trajectory, and the cProfile buckets with their times per
  call. A commit from before the stage timers is measured with these
  scripts, `population.py` and `benchmark_targets.py` copied into a
  worktree at it: its runs have no stages, and the comparison takes their
  wall times and buckets.
- `divergence_trace.py` repeats the optimization of an output-function
  test of `test_run_control.py` in one process and finds where two runs of
  one seed differ: `loop N` compares their evaluations, `trace N OUT_DIR`
  the digests of every call and return of PyBADS, gpyreg and NumPy's and
  SciPy's linear algebra, down to the first local variable that differs,
  and `align N` counts the results of the linear algebra of a GP fit at
  each alignment of its arrays. Its docstring gives the details.
- `test_population.py` checks the record schema, a record whose stage
  times cannot be read, the suites' configurations, the reference minima
  of the real-data targets, resumability and the statistics of `compare`:
  `python -m pytest dev/scripts/test_population.py`.
- `test_replay.py` checks the comparison of `replay.py` on synthetic
  traces, its warnings, the recorder's reading of the refit flag, its
  failure on a private name that the package no longer has and on an
  error of its own code inside a run, and one short recording repeated in
  one process: `python -m pytest dev/scripts/test_replay.py`.
- `gp_health_hooks/sitecustomize.py` counts, per run, what the GP layer
  does: the factorizations that fail and gpyreg's noise multiplier, the
  posteriors that keep it, the refits and their failed tries, the zero
  predictive SDs and their value before the clamp, the NaN log priors, the
  smallest training sets, and the outcomes of the Sto-BADS rule. With the
  directory first on `PYTHONPATH` and `GP_HEALTH_OUT` set, every run of
  `population.py` writes its counters there, and changes nothing (its
  docstring says how to check that); `GP_FORCE_RAISE_ON_CHOLESKY_FAILURE=1`
  turns on gpyreg's `raise_on_cholesky_failure` in every GP, an experiment
  that does change results. `gp_health.py summary DIR...` tabulates the
  counters, one row per configuration.

A population's raw output goes to `scripts/runs/population/<name>/`. A
population that serves as a reference for later comparisons is copied
(the JSON records and `summary.md`) to `experiments/<name>/`, named
`population_<purpose>_<YYYYMMDD>` after the day of the run, with a
`README.md` that holds the command, the provenance, the null check, a
positive control, and the smallest effect the comparison detects at the
reference's number of seeds.

## Index

- [The port correctness review](results/2026-09-28-port-correctness-review.md)
  — the consolidated ledger of the independent review of PyBADS against
  MATLAB BADS v1.1.3 (2026-09-25 to 09-28, five waves): its 173 rows, each
  with its classification, the PI's disposition and its fix or `TODO.md`
  item; the net change on the benchmark on Windows (100 seeds) and Linux
  (30 seeds); the open ends; and the defects found on the MATLAB side. The
  deliberate differences it settled are catalogued in
  `pybads/bads/README.md`.
- [Where PyBADS spends its time](results/2026-09-28-where-pybads-spends-its-time.md)
  — the stages of PyBADS's own time on six configurations (the ES search's
  candidates and the GP fits take nearly all of it, gpyreg's kernel 31 to
  45 %), the saving of computing the optimization target without a copy
  of the GP (1.6 to 3.7 %), and gpyreg's rank-1 update of the posterior
  measured beside every addition of a point: agreement, the noise
  multiplier it carries over, and a saving of at most 2.5 %, behind the
  decision to keep the full recomputation.
- [The stage times of PyBADS's runs](results/2026-09-28-stage-times.md) —
  the baseline campaign of the profiler, whose stage timers charge each
  second of a run to one stage: on the `profile` suite, the ES search's
  candidates take 17 to 65 % of the own time and the GP's fits 14 to 71 %,
  the failed fits alone 49 % of `ellipsoid_D3`'s; the stages and the target
  make `total_time` to 2e-5 s (6e-5 s under cProfile); the noise of the
  machine between two passes of the same runs, and the choice of a control
  stage.
- [The GP layer's numerical health](results/2026-09-28-gp-health.md) —
  the failed factorizations and gpyreg's noise multiplier, the zero
  predictive SDs, the NaN log priors and the smallest training sets over
  four suites at the close of the review, one cause behind the first two
  (an output variance far above the noise floor, in MATLAB's bounds), and
  gpyreg's switch to MATLAB's rule measured against it: kept off (PI,
  2026-09-28). Its evidence:
  [experiments/gp_health_linux_20260928/](experiments/gp_health_linux_20260928/README.md)
  and
  [experiments/gp_switch_linux_20260928/](experiments/gp_switch_linux_20260928/README.md).
- [Sto-BADS's success rule](results/2026-09-28-stobads-rule.md) — the
  current rule, the rule without the mesh factor and the limit of the
  uncertain moves, on the noisy configurations at 60 seeds, with every
  decision of the rule counted: no gain over BADS without Sto-BADS, and the
  rulings on W0-12 and W0-13. Its evidence:
  [experiments/stobads_linux_20260928/](experiments/stobads_linux_20260928/README.md).
- [Seeded runs on macOS arm64](results/2026-09-28-macos-arm64-repeatability.md)
  — two runs of one seed need not match bit for bit on macOS arm64, where
  Accelerate's results depend on the alignment of the arrays: measured in
  CI on three NumPy and SciPy stacks against Linux, the first difference
  located in gpyreg's triangular solve; the output-function test rewritten
  to check what the comparison stood for, and the seed tests comparing
  there only what the seed decides.
- [experiments/w225_linux_20260928/](experiments/w225_linux_20260928/README.md)
  — row W2-25 of the port review (a noisy run's move to an earlier iterate
  takes its location with its value) against its revert, the five noisy
  configurations at 90 seeds on Linux: no measurable effect on their
  errors, evaluations or fraction solved; the lower fraction solved of its
  30-seed gate belongs to those seeds.
- [experiments/w236_linux_20260928/](experiments/w236_linux_20260928/README.md)
  — row W2-36 of the port review (a noisy run's first incumbent is the raw
  minimum of its initial design, as in MATLAB BADS) against the first
  incumbent's value from the initial GP, the five noisy configurations at
  90 seeds on Linux: no flag; the variant raises `sphere_D3_hetero`'s
  fraction solved from 0.50 to 0.61 and changes each other configuration's
  by at most one run; MATLAB BADS's behaviour kept (PI, 2026-09-29).
- [experiments/one_point_gp_linux_20260928/](experiments/one_point_gp_linux_20260928/README.md)
  — the GP whose initial training set holds one point takes MATLAB BADS's
  definition values without a fit (`73d517a`), against its parent, the
  `geometry` and `thinband` suites at 30 seeds on Linux: no flag; the
  results of exactly the 144 runs that start on one point and go past it
  change, by no consistent amount; their initialization prints no warning,
  and refits print gpyreg's warnings on inputs without spread in a
  coordinate in every run of `sphere_band_D3` and 3 of 30 of
  `sphere_band_D2_hetero`.
- [experiments/warmstart_gp_linux_20260929/](experiments/warmstart_gp_linux_20260929/README.md)
  — a first GP fitted on the incumbent's neighbours in the whole log when
  the run is given evaluations made before it (`58d922a1`), against its parent, the `warmstart` suite at 90 seeds on
  Linux: the reruns' first GP holds the start alone in the base and 35 to
  90 rows of the log in the change; flagged for more evaluations on
  `rosenbrock_D6_rerun` (median 394 to 426) at an unchanged error, the
  pooled fraction solved 0.90 against 0.88, not significant; not adopted
  (PI, 2026-09-29), its diff kept with the record.
- [experiments/population_wave4_20260928/](experiments/population_wave4_20260928/README.md)
  — the reference population of the benchmark on Windows (default suite,
  100 seeds, gpyreg 1.3.3, at `a4dcd65`, `dev-next` after wave 4 of the
  port review and its doublecheck), with its null check and its comparison
  with the pre-review baseline, the net change of the whole review, which
  flags nine configurations: two better, two spheres with slightly larger
  errors far below their tolerance, `ellipsoid_D10` with more evaluations
  for a smaller error, and four configurations with noise whose runs stop
  earlier.
- [experiments/population_prereview_20260927/](experiments/population_prereview_20260927/README.md)
  — the pre-review baseline on Windows (default suite, 100 seeds, gpyreg
  1.3.3, at `ab4dded`, before the port review's fixes), a fixed population
  to compare any later version with on this platform, with its null check;
  it extends `population_gpfixes_20260925` from 30 seeds to 100.
- [experiments/population_gpfixes_20260925/](experiments/population_gpfixes_20260925/README.md)
  — the previous reference population of the benchmark on Windows (default
  suite, 30 seeds, gpyreg 1.3.3, at `ab4dded`: #67 and the three GP fixes
  of #66), with its null check, its comparison with the previous Windows
  reference, which flags the five configurations that the same fixes flag
  on Linux, all better, and the runs in which the prior of the GP mean
  falls outside the bounds of the mean and the log prior is NaN.
- [experiments/population_targetnoise_20260925/](experiments/population_targetnoise_20260925/README.md)
  — the previous reference on Windows (at `c044fea`, with the
  noise-variance fix of `020d6a8`), with its null check and its comparison
  with the one before, which flags nothing: the two configurations with
  target noise change, and the other 16 are identical run by run.
- [experiments/population_linux_wave4_20260927/](experiments/population_linux_wave4_20260927/README.md)
  — the reference population of the benchmark on Linux (default suite, 30
  seeds, gpyreg 1.3.3, at the package code of wave 4's fix pass of the
  port review, `46af65a`, where each seed has its own initial design), with
  its null check and its comparison with the previous Linux reference, the
  net change of the pass, which flags nothing.
- [experiments/population_linux_wave3_20260927/](experiments/population_linux_wave3_20260927/README.md)
  — the previous reference population of the benchmark on Linux (default
  suite, 30 seeds, gpyreg 1.3.3, at the package code of wave 3's fix pass
  of the port review, `a14524d`, without W3-24, which the PI reverted), the
  baseline of wave 4's fix pass, with its null check and its comparison
  with the previous Linux reference, the net change of that pass, which
  flags nothing.
- [experiments/population_linux_wave2_20260926/](experiments/population_linux_wave2_20260926/README.md)
  — the previous reference population of the benchmark on Linux (default
  suite, 30 seeds, gpyreg 1.3.3, at the package code of wave 2's fix pass
  of the port review, `8510ca8`), the baseline of wave 3's fix pass, with
  its null check and its comparison with the previous Linux reference, the
  net change of that pass, which flags one configuration, more evaluations
  on `ellipsoid_D10` with an unchanged error.
- [experiments/population_linux_wave1_20260926/](experiments/population_linux_wave1_20260926/README.md)
  — the previous reference population of the benchmark on Linux (default
  suite, 30 seeds, gpyreg 1.3.3, at the package code of wave 1's fix pass
  of the port review, `fef6c14`), the baseline of wave 2's fix pass, with
  its null check and its comparison with the previous Linux reference, the
  net change of that pass, which flags one configuration, a lower error on
  `ackley_D6`.
- [experiments/population_linux_wave0_20260926/](experiments/population_linux_wave0_20260926/README.md)
  — the previous reference population of the benchmark on Linux (default
  suite, 30 seeds, gpyreg 1.3.3, at the package code of `e004c79`: wave 0
  of the port review, W0-1 included), the baseline of wave 1's fix pass,
  with its null check and its comparison with the previous Linux
  reference, which flags the number of evaluations of three configurations
  with noise, fewer, as W0-1's gate on Windows does; the 13 configurations
  without noise are identical run by run, and the wave 0 fix pass without
  W0-1 reproduces the previous reference.
- [experiments/population_linux_gpfixes_20260925/](experiments/population_linux_gpfixes_20260925/README.md)
  — the previous reference on Linux (default suite, 30
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
  the worklog, closed on 2026-09-28; its records (the known-differences
  sheet, the counterpart map, the reviewers' reports, the per-wave ledgers)
  under `experiments/port_review_20260925/`.
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
  read of the code, and the tests that checked less than they appeared to,
  with their fixes, the seed sweep behind the tolerances
  of the optimization tests, three candidate defects found on the way, the
  checks behind running each test once in CI and requiring NumPy 2, the
  fixes of five small defects of `bads.py` with their gates, and the
  reruns of the four crashing benchmark runs, whose failing calls share
  degenerate GP hyperparameters. The starting point of the port
  correctness review, which closed its candidate table.
