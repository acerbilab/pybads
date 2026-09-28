# Reference population on Windows after wave 4 of the port review: the default suite, 100 seeds, gpyreg 1.3.3

The reference on Windows for `dev/scripts/population.py compare` until a
later reference replaces it. The one on Linux is
[`population_linux_wave4_20260927`](../population_linux_wave4_20260927/README.md),
at `46af65a`. 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-99, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3.

Its package code is `a4dcd65`, `dev-next` after the doublecheck of wave 4
of the port correctness review (#81). Its default runs are those of
`46af65a`, the Linux reference's package code: #81's changes to the package
are checks of options that move no default run (the fingerprint of
`dev/scripts/fingerprint.py` on Linux is `4146a986863602cb` at both). It
replaces [`population_gpfixes_20260925`](../population_gpfixes_20260925/README.md)
(`ab4dded`, 30 seeds), from which the package differs by #71 (the noise
options, the final estimate and the iteration count, which change no field
that `compare` reads) and by waves 0 to 4 of the port review with their fix
passes and doublechecks (#72 to #81).

The reference holds for #84 (`79697835`) too, which moves no default run:
on Windows, the fingerprint of `dev/scripts/fingerprint.py` is the same at
`a4dcd65` and `79697835`, `dca2b20df2743512` with the default number of
BLAS threads and `093cb1d05a16d889` with one (gpyreg 1.3.3 clone), and on
Linux it is `4146a986863602cb` at both (#84's description).

## Command and provenance

From the main checkout's root, with a clean detached worktree at `a4dcd65`
under `dev/scripts/runs/worktrees/`:

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 .venv/Scripts/python.exe -u dev/scripts/runs/worktrees/winref_a4dcd65/dev/scripts/population.py run --suite default --seeds 0-99 --workers 4 --out dev/scripts/runs/population/population_reviewafter_20260928
```

- PyBADS at `a4dcd65`, run by the worktree's own `population.py`, which
  puts that checkout first on `sys.path`: every record's `meta.git` and
  `meta.pybads_source` name the worktree at `a4dcd651`, clean. gpyreg 1.3.3
  from a clone checked out at the tag `v1.3.3` (`98ab5a4`), selected with
  `PYTHONPATH`.
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1; one BLAS thread per
  run (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1), a
  fresh process per run.
- The runs went in five sessions on 2026-09-28, from 06:49 to 11:57, with
  8, 4 or 2 runs at a time. Four ended in power cuts of the machine, whose
  firmware cut the power under a sustained load of several busy cores; the
  fifth ran with the laptop's fan profile at "Standard", a lower power limit
  of the CPU, and finished. `population.py` writes a run's record when the
  run ends and skips existing records when restarted, so a run under way at
  a cut left no record and ran again. The number of runs at a time and the
  power limit change the wall time alone, as in
  [`population_prereview_20260927`](../population_prereview_20260927/README.md),
  where 1 and 8 runs at a time give equal records.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the pre-review
  baseline) and `null_check.md`.

## Outcome

All 1800 runs finished, none crashed. The fraction solved ranges from 0.04
(`rastrigin_D3`, whose runs end in local minima) to 1.00.

## Comparison with the pre-review baseline

The previous reference has 30 seeds; the comparison is with
`population_prereview_20260927`, the same code (`ab4dded`) at 100 seeds, so
it measures the net change of #71 and of the whole port review.
`compare dev/experiments/population_prereview_20260927 <this population>`
(`comparison.md`) flags nine configurations:

| Configuration | Median error, before → after | Median evaluations | Solved | Median paired log10 error ratio [95% CI] | Flagged tests |
|---|---|---|---|---|---|
| `ackley_D6` | 3.4e-4 → 2.0e-4 | 400 → 388 | 1.00 → 1.00 | -0.27 [-0.34, -0.24] | error, evaluations, signed-rank |
| `rosenbrock_D2` | 7.4e-6 → 2.5e-6 | 95 → 94 | 1.00 → 1.00 | -0.51 [-0.86, -0.25] | error, signed-rank |
| `sphere_D2` | 5.5e-7 → 1.5e-6 | 55 → 55 | 1.00 → 1.00 | +0.33 [+0.17, +0.58] | error, signed-rank |
| `sphere_D10` | 7.0e-8 → 9.7e-8 | 454 → 449 | 1.00 → 1.00 | +0.20 [+0.04, +0.32] | error, signed-rank |
| `ellipsoid_D10` | 7.3e-7 → 4.7e-7 | 629 → 668 | 1.00 → 1.00 | -0.16 [-0.29, -0.07] | evaluations |
| `ellipsoid_D3_homo` | 0.076 → 0.081 | 380 → 315 | 0.64 → 0.55 | +0.11 [-0.10, +0.23] | evaluations |
| `ellipsoid_D3_hetero` | 0.29 → 0.31 | 366 → 304 | 0.19 → 0.16 | -0.00 [-0.24, +0.14] | evaluations |
| `sphere_D3_homo` | 0.0078 → 0.0136 | 313 → 226 | 1.00 → 1.00 | +0.17 [-0.02, +0.34] | evaluations |
| `multisensory_s1_D6_homo` | 0.20 → 0.15 | 668 → 580 | 0.96 → 0.97 | -0.02 [-0.15, +0.07] | evaluations |

- The deterministic configurations end with an equal or smaller error, but
  for the two spheres, whose errors grow while staying at least three
  orders of magnitude below their tolerance (0.001). `ellipsoid_D10` takes 6% more
  evaluations for a smaller error.
- The five configurations with noise stop earlier, with 12 to 28% fewer
  evaluations (`sphere_D3_hetero`, unflagged, 376 → 330), the direction
  that the gate of W0-1 first flagged
  ([`population_linux_wave0_20260926`](../population_linux_wave0_20260926/README.md)).
  Their errors move in both directions, and none of their error tests is
  flagged.
- Unflagged: `rosenbrock_D6` solves 0.81 of its runs (0.72 before; which
  basin a run ends in changes with any perturbation of its trajectory),
  `rastrigin_D3` ends at a median error of 3.0 (4.5), and the intervals of
  `ellipsoid_D3_unbounded` (-0.43 [-0.53, -0.13]) and `sphere_nonbox_D3`
  (-0.17 [-0.35, -0.04]) exclude zero, better.
- On Linux, the same comparison at 30 seeds, between
  [`population_linux_gpfixes_20260925`](../population_linux_gpfixes_20260925/README.md)
  (`ab4dded`'s package code but for #67) and `population_linux_wave4_20260927`,
  flags four of these configurations, in the same directions: `ackley_D6`
  and `rosenbrock_D2` (error), `ellipsoid_D10` and `ellipsoid_D3_homo`
  (evaluations).

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 36 tests.
- **Positive control**: the one in the first reference
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)).
  This population has the same configurations and tests.

## What the comparison detects at 100 seeds

A full comparison holds 54 tests; after Holm's correction, the first step
needs p ≤ 9.3e-4. For 100 against 100 runs, that is a KS statistic of at
least 0.28. The paired signed-rank test detects, with 80% power, a shift of
the paired log10 error ratios of about 0.44 of their standard deviation
(simulated; the same simulation gives the 0.87 of 30 seeds that the first
reference states). Between two versions on this platform, a run that a
change does not reach is identical in both populations.
