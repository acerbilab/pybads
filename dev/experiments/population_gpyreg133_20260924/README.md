# Reference population: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Windows for `dev/scripts/population.py compare` until a
later reference replaces it (the one on Linux is
[`population_linux_20260925`](../population_linux_20260925/README.md)): 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3, the
release PyBADS requires and pins in CI. It replaces
[`population_generator_20260924`](../population_generator_20260924/README.md),
the same code with gpyreg 1.3.1; the comparison with it and its reading
are in [the gpyreg 1.3.3 note](../../results/2026-09-25-gpyreg-1.3.3.md).

## Command and provenance

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 .venv/Scripts/python.exe -u dev/scripts/population.py run --suite default --seeds 0-29 --out dev/scripts/runs/population/population_gpyreg133_20260924
```

- PyBADS at `2059506`, clean tree; gpyreg 1.3.3 from a clone checked out
  at the tag `v1.3.3` (`98ab5a4`), selected with `PYTHONPATH`. Every
  record's `meta` names that source path and commit; its `gpyreg` version
  string reads 1.3.1, the version of the gpyreg installed in the venv, which
  `PYTHONPATH` overrides.
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1; one BLAS thread per
  run (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1),
  one run at a time, a fresh process per run; 72.2 minutes, from 22:50 on
  2026-09-24 to 00:03 on 2026-09-25.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md` (per configuration: median and interquartile range of the
  error and of the evaluations, fraction solved, crashes), `comparison.md`
  (the comparison with the generator population, gpyreg 1.3.1) and
  `null_check.md`.

## Outcome

All 540 runs finished. The fraction solved ranges from 0.07 (Rastrigin,
whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_generator_20260924 <this population>`
(`comparison.md`) flags five configurations, each with smaller errors in
this population: `sphere_D10`, `timing_D5`, `multisensory_s1_D6`,
`rosenbrock_D6` and `rastrigin_D3`. Every run that differs between the two
populations had GP noise variances below `1e-6`, the regime whose
predictions gpyreg 1.3.2 changed; the others are identical. The note
linked above has the effect sizes.

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 36 tests.
- **Positive control**: the first reference's
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)),
  where three configurations at 50 D flagged, `ellipsoid_D10` on the
  error. This population has the same configurations, seeds and tests, and
  its comparison with the previous reference flagged the changes of gpyreg
  1.3.2's predictions.

## What the comparison detects at 30 seeds

As for the first reference: 54 tests, the first Holm step at p ≤ 9.3e-4, a
KS statistic of at least 0.50, and, for the paired signed-rank test, a
shift of about 0.87 of the standard deviation of the paired log10 error
ratios at 80% power. The pairs share each seed's start point and target
noise (`benchmark_targets.py`); BADS's own draws also match between two
versions that draw the same numbers in the same order, and then the runs
that a change does not reach are identical.
