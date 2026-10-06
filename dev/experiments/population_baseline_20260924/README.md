# Reference population: the default suite, 30 seeds, gpyreg 1.3.1

Replaced as the reference by
[`population_generator_20260924`](../population_generator_20260924/README.md),
whose runs draw through a `numpy.random.Generator` and which passed the
comparison with this population.

The reference for `dev/scripts/population.py compare` until a later
reference replaces it: 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria. Randomness
still goes through NumPy's global stream, seeded by `random_seed`.

## Command and provenance

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.1 python -u dev/scripts/population.py run --suite default --seeds 0-29 --out dev/scripts/runs/population/population_baseline_20260924
```

- PyBADS at `2226883`, clean tree; gpyreg 1.3.1 from a clone checked out
  at the tag `v1.3.1` (`1dbbfc5`), selected with `PYTHONPATH` (every
  record's `meta` names the source path and commit).
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1; one BLAS thread per
  run (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1),
  one run at a time, a fresh process per run; 75.7 minutes on 2026-09-24.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md` (per configuration: median and interquartile range of the
  error and of the evaluations, fraction solved, crashes), `null_check.md`
  and `positive_control.md` (the outputs of the two checks below).

## Outcome

538 runs finished; 2 crashed with `LinAlgError: Singular matrix for L
Cholesky decomposition` on a GP update that PyBADS does not guard
(`ellipsoid_D3` seed 20, `ellipsoid_D10` seed 7;
`dev/results/2026-09-23-codebase-survey.md`; guarded from `676083d`). A crash is an outcome of the
comparison: a configuration is flagged when its crash count rises from
zero, so in these two configurations, which already hold one crash, a
further crash does not flag by itself. The fraction solved ranges from 0.03 (Rastrigin, whose runs
end in local minima) to 1.00, most configurations between 0.2 and 1.0.

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone): no flag in 36 tests.
- **Positive control**: three configurations whose runs all go beyond
  50 D evaluations here (`ellipsoid_D10`, `sphere_D3_homo`,
  `multisensory_s1_D6_homo`), rerun at `--budget-scale 0.1` (50 D, so that
  every run stops on the budget) and compared with this population: all
  three flagged; `ellipsoid_D10` on the error (median error 39 times
  larger, +1.59 in log10, solved 0.97 → 0.30) and on the evaluations, the
  two noisy configurations on the evaluations alone, their error being set
  by the noise rather than by the budget. Part of the control's runs was
  recorded with a pending edit of `dev/TODO.md` (documentation only).

## What the comparison detects at 30 seeds

A full comparison against this reference holds 54 tests (per
configuration, KS tests of the error and of the evaluations, and the paired
signed-rank test of the log10 error); after Holm's correction, the first
step needs p ≤ 0.05 / 54 ≈ 9.3e-4. For 30 against 30 runs, that is a KS
statistic of at least 0.50. The paired test detects, with 80% power, a
shift of the paired log10 error ratios of about 0.87 of their standard
deviation (simulated). "No flag" therefore means no change of that size:
smaller changes need more seeds, and the effect sizes that `compare`
prints (the median paired log10 error ratio with its 95% interval, the
change of the fraction solved) say how large the change could be.
