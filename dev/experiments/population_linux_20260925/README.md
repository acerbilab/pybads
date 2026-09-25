# Reference population on Linux: the default suite, 30 seeds, gpyreg 1.3.3

The reference for `dev/scripts/population.py compare` on Linux, beside the
Windows reference
[`population_gpyreg133_20260924`](../population_gpyreg133_20260924/README.md),
until a later one replaces it. It covers the 18 configurations of the
`default` suite of `dev/scripts/benchmark_targets.py` × seeds 0-29. Each
run uses BADS's default budget (500 D) and ends on BADS's own termination
criteria, every random draw goes through the run's
`numpy.random.Generator`, and gpyreg is 1.3.3. Pairing by seed is only
meaningful against a population from the same platform and versions (see
below), so a change made on Linux compares with this reference, and one
made on Windows with the Windows reference.

## Command and provenance

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 .venv/bin/python -u dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out dev/scripts/runs/population/population_linux_post_20260925
```

- PyBADS at `676083d` (the guards of
  [`dev/plans/gp-update-guards.md`](../../plans/gp-update-guards.md)),
  clean tree. gpyreg 1.3.3 comes from a clone checked out at the tag
  `v1.3.3` (`98ab5a4`), selected with `PYTHONPATH`, and installed editable
  from a checkout at the same tag, so the version string also reads 1.3.3.
- Linux (a cloud container), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1.
  One BLAS thread per run (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` = 1), four runs at a time, a fresh process per run.
  24.6 minutes on 2026-09-25.
- The files are one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the Windows
  reference) and `null_check.md`.

## Outcome

All 540 runs finished. The fraction solved ranges from 0.03 (Rastrigin,
whose runs end in local minima) to 1.00.

## The same runs before the guards

The same command at `500ff1b`, whose package code is that of `09996b5`
(before the guards), gave `population_linux_pre_20260925`, not kept here.
Every one of its 540 records equals this population's in every `final`
field except `wall_s`. `dev/scripts/gp_update_failures.py`, run with the
same seeds, found no failed call among the 484,773 guarded GP updates of
these runs, and reproduced every run's `x`, `fval` and `func_count`. The
guards change nothing in a run without a failure, so the two populations
are one population, and this reference also stands for the code before the
guards on this platform.

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 36 tests.
- **Against the Windows reference** (`comparison.md`, information only):
  no flag in 54 tests. Every median paired log10 error ratio lies within
  [-0.28, +0.08], and every bootstrap interval contains zero. The two
  populations differ in platform and in the versions of Python, NumPy and
  SciPy (3.12.6, 2.5.3 and 1.18.1 there), so the runs of one seed follow
  different trajectories, and pairing by seed means nothing. The
  comparison says only that the distributions agree.
- **Positive control**: the one in the first reference
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)).
  This population has the same configurations, seeds and tests.

## What the comparison detects at 30 seeds

The same as for the first reference: 54 tests, the first Holm step at
p ≤ 9.3e-4, a KS statistic of at least 0.50, and, for the paired
signed-rank test, a shift of about 0.87 of the standard deviation of the
paired log10 error ratios at 80% power. Between two versions on this
platform, a run that a change does not reach is identical in both
populations, as the comparison with `population_linux_pre_20260925` shows.
