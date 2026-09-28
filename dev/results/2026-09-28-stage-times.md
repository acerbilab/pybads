# The stage times of PyBADS's runs

Measured on 2026-09-28, the baseline campaign of the profiler
(`dev/scripts/profile_run.py`, `profile_suite.py` and `profile_compare.py`)
on the `profile` suite of `dev/scripts/benchmark_targets.py`. A run of
PyBADS charges each second of `optimize()` to one stage, the innermost one
open, and the target's evaluations to the pseudo-stage `target`
(`pybads/utils/timer/stage_timer.py`); the stages and the target make
`total_time`. The question: where PyBADS's own time goes, stage by stage,
as a reference for later changes.

## Summary

- **Two stages take nearly all of the own time.** The ES search's
  candidates (`search_es`: generating them, removing those already
  evaluated and scoring them by LCB) take 17 to 65 % of it, and the GP's
  hyperparameter fits, failed tries, retries and fallbacks included, 14 to
  71 %. The rebuilds of the local training set take 4.6 to 24 %, most in the
  noisy runs, whose search rebuilds a copy of the GP around each point and
  whose re-estimation of the history (6 to 14 %) rebuilds one around each
  iterate. The rest of the search, the poll's own code, the posterior
  updates, the optimization target, the initial design and the first GP,
  the history records and the loop take at most 4 % each.
- **The failed fits.** Fits that raise `LinAlgError` take 49 % of
  `ellipsoid_D3`'s own time, 32 % of `ellipsoid_D3_homo`'s and 13 % of
  `ellipsoid_D10`'s, and none of the other configurations'. The retries'
  preparations and the fallbacks take at most 2.4 %.
- **The own time** is 12 to 40 ms per evaluation. The target takes 0.02 s
  or less per run, except `multisensory_s1_D6_homo`'s, 0.6 to 0.8 s.
- **The accounting.** In every run, `total_time` less the stages and the
  target is 1.2e-5 to 2.1e-5 s, the two readings of the clock between the
  starts and between the stops of the run's two timers. A run has 890 to
  7,500 transitions between stages, at 1.2 µs each: under 0.05 % of its
  time.
- **The noise of the machine.** A second pass of the plain runs, five
  minutes after the first, took 0.98 to 1.03 times as long per
  configuration (median over the seeds), except on `ellipsoid_D3`, the
  first configuration of the first pass, which ran right after a
  population run had ended: 0.90 times as long in every stage, the
  control `search_es` included. Nine runs of the same trajectory
  (`ellipsoid_D3`, seed 0) took 3.2 to 3.9 s. A change below about 10 % in
  a configuration's time needs a control stage near 1 to be believed.
  `gp_init`, one fit, varied by up to 17 % between the passes: a control
  is a large stage that the change does not reach.
- **cProfile** slows the runs by 1.10 to 1.62 times, most on
  `ellipsoid_D3`, whose failed fits make many short calls; the plain runs
  give the stage times, the profiled ones the functions inside them. Every
  profiled run ran the plain run's trajectory. gpyreg's kernel,
  `RationalQuadraticARD.compute`, takes 28 to 44 % of a profiled run with
  what it calls, as it did on Windows ([where PyBADS spends its
  time](2026-09-28-where-pybads-spends-its-time.md): 31 to 45 % in its own
  code).

## Setup

- **Code.** PyBADS at `a4a423b3`, with the stage timers (`cf371d4e`) and the
  profiler; gpyreg 1.3.3 from a clone at its tag, on `PYTHONPATH`.
- **Environment.** A Linux container (Firecracker VM) with 4 virtual CPUs
  (Intel Xeon at 2.10 GHz) and 15 GB of memory; Python 3.11.14, NumPy
  2.4.6, SciPy 1.17.1, OpenBLAS 0.3.31 (scipy-openblas). One BLAS thread
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to
  1), one run per process, one process at a time, nothing else running.
- **Configurations.** The `profile` suite: `ellipsoid_D3`, `ellipsoid_D10`,
  `rosenbrock_D6` and `ackley_D6` (deterministic),
  `multisensory_s1_D6_homo` and `ellipsoid_D3_homo` (noise inferred), and
  `sphere_D3_hetero` (noise given by the target), each at its budget of
  500 D, set up as `population.py` sets it up. Seeds 0, 1 and 2, plain and
  under cProfile.
- **Command.** `profile_suite.py --suite profile --seeds 0-2 --mode both
  --probe ellipsoid_D3`. The plain runs were then run again (`--mode
  plain`), after the machine had settled; the tables hold the second pass,
  and the first pass is the null comparison above.
- **The stages.** At the top level: `init` (the start, the noise test and
  the initial design, `_init_mesh_`), `gp_init` (the first GP), `search`,
  `poll`, `history` (the records of an iteration), `reestimate` (the
  re-estimation of the history in a noisy run), `final_samples`,
  `output_fcn` and `loop`, the rest. Nested in them: `gp_rebuild`
  (`local_gp_fitting`, with the copy of the GP it is given in the noisy
  search), `gp_fit` (the hyperparameter fit, `_robust_gp_fit_` or the
  initial fit) with `gp_fit_failed`, `gp_fit_retry` and `gp_fit_fallback`
  inside it, `gp_update` (`add_and_update_gp`), `target_from_gp` and
  `search_es` (the ES search's hedge). The noise test's evaluation, which
  the target's time leaves out, counts in `init`.

## Where the own time goes

Medians over seeds 0 to 2 of the plain runs; the shares are percentages of
the own time (`total_time` less the target's evaluations), each top-level
stage with the stages nested in it. Each column's median is taken
separately, so a row need not add up to 100.

| configuration | own s | ms per eval. | evals | init | gp_init | search | poll | history | reestimate | loop |
|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3 | 3.29 | 20.2 | 147 | 0.4 | 0.8 | 60.9 | 37.6 | 0.1 | - | 0.2 |
| ellipsoid_D10 | 18.46 | 26.9 | 677 | 0.1 | 0.3 | 78.8 | 20.6 | 0.0 | - | 0.2 |
| rosenbrock_D6 | 5.89 | 14.9 | 413 | 0.2 | 0.6 | 83.4 | 15.3 | 0.1 | - | 0.3 |
| ackley_D6 | 4.47 | 11.7 | 382 | 0.3 | 0.7 | 77.9 | 20.6 | 0.1 | - | 0.4 |
| multisensory_s1_D6_homo | 20.09 | 25.1 | 801 | 0.1 | 0.3 | 75.1 | 14.8 | 0.0 | 10.9 | 0.2 |
| ellipsoid_D3_homo | 12.69 | 39.6 | 314 | 0.1 | 0.6 | 70.3 | 23.3 | 0.0 | 5.8 | 0.2 |
| sphere_D3_hetero | 7.47 | 21.4 | 345 | 0.2 | 0.5 | 68.2 | 14.3 | 0.1 | 13.8 | 0.3 |

`final_samples` and `output_fcn` (no output function is set) take 0.0 %.

By leaf, wherever the stage happens (% of the own time; entries per run
in parentheses):

| configuration | ES search | fits | failed fits | retries | rebuilds (rest) | posterior updates | target |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3 | 17.3 (69) | 19.6 (16) | 49.0 (52) | 2.4 (52) | 4.8 (66) | 0.7 (52) | 0.6 (72) |
| ellipsoid_D10 | 46.2 (340) | 29.9 (27) | 13.0 (8) | 0.1 (8) | 4.6 (235) | 1.6 (306) | 1.1 (320) |
| rosenbrock_D6 | 64.5 (205) | 19.6 (23) | 0 | 0 | 6.1 (120) | 1.6 (171) | 1.2 (194) |
| ackley_D6 | 59.1 (156) | 20.9 (22) | 0 | 0 | 6.2 (97) | 1.7 (130) | 1.9 (216) |
| multisensory_s1_D6_homo | 53.2 (348) | 16.3 (34) | 0 | 0 | 20.9 (1387) | 3.4 (702) | 0.9 (424) |
| ellipsoid_D3_homo | 27.5 (156) | 22.9 (25) | 32.2 (46) | 0.6 (46) | 11.8 (479) | 1.6 (232) | 0.5 (116) |
| sphere_D3_hetero | 51.2 (156) | 13.8 (27) | 0 | 0 | 24.0 (632) | 2.7 (261) | 1.0 (139) |

The fits' fallback, after a refit in which no try fits, happened in two
runs of `ellipsoid_D3`, once in each, and took under 1 ms. The rebuilds of the noisy runs are
many: those of the search's copy around each point and those of the
re-estimation around each iterate, which is the re-estimation's time.

Compared with the Windows measurement of six of these configurations
([where PyBADS spends its
time](2026-09-28-where-pybads-spends-its-time.md), whose "ES candidates"
is `search_es` and whose "hyperparameter fits" are the fits with their
failures), the fits' shares agree within 4.3 points, and the ES search's
are 9 points lower to 2 points higher here, which ran with the faster
removal of evaluated candidates that that note describes (the entry "Cost
of removing evaluated candidates" of `CHANGELOG.md`). The own time per
evaluation is 0.76 to 1.15 times Windows'.

## cProfile

The cumulative time of the curated functions, as a percentage of the
profiled `optimize()` (medians over the seeds): `ESSearchHedge.__call__`
12 to 55 %, `acq_fcn_lcb` 8.5 to 44 %, `contraints_check` 1.5 to 7 %, `GP.fit`
14 to 73 %, `GP.predict` 9 to 44 %, `GP.update` 2 to 10 %,
`RationalQuadraticARD.compute` 28 to 44 %, SciPy's `cholesky` 1 to 5 % and
`solve_triangular` 2 to 10 %, `IterationHistory.record` and `deepcopy`
under 1.2 % each. The slice sampler is never called at the default options.

## Raw data

The campaign's runs, logs, `aggregate.json` and `aggregate.md`, and the
first pass of the plain runs, are machine-local, under
`dev/scripts/runs/profile/stage_times_20260928/` and
`dev/scripts/runs/profile/stage_times_20260928_first_plain/`
(`dev/scripts/runs/LOCAL.md`).
