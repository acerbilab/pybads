# Reference population: the default suite, 30 seeds, draws through a generator

Replaced as the reference by
[`population_gpyreg133_20260924`](../population_gpyreg133_20260924/README.md),
the same code with gpyreg 1.3.3.

The reference for `dev/scripts/population.py compare` until a later
reference replaces it: 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria. Every random
draw of a run comes from its `numpy.random.Generator` (`bads.rng`), created
from `random_seed`. It replaces
[`population_baseline_20260924`](../population_baseline_20260924/README.md),
whose runs drew from NumPy's global stream, and passed the comparison with
it below.

## Command and provenance

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.1 .venv/Scripts/python.exe -u dev/scripts/population.py run --suite default --seeds 0-29 --out dev/scripts/runs/population/population_generator_20260924
```

- PyBADS at `c85cddb` (the generator change is `551aa47`), clean tree;
  gpyreg 1.3.1 from a clone checked out at the tag `v1.3.1` (`1dbbfc5`),
  selected with `PYTHONPATH` (every record's `meta` names the source path
  and commit).
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1; one BLAS thread per
  run (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1),
  one run at a time, a fresh process per run; 84.5 minutes on 2026-09-24.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md` (per configuration: median and interquartile range of the
  error and of the evaluations, fraction solved, crashes), `comparison.md`
  (the comparison with the baseline) and `null_check.md`.

## Outcome

538 runs finished; 2 crashed with `LinAlgError: Singular matrix for L
Cholesky decomposition`, both in `ellipsoid_D10` (seeds 13 and 26, at 751
and 341 evaluations). In both, `local_gp_fitting`
(`bads/gaussian_process_train.py`) catches the failure of
`gp.update(hyp=hyp_gp)` and restores the previous hyperparameters with
`gp.set_hyperparameters(old_hyp_gp)`, which fails in turn and is not
caught; seed 13 reaches it from the search step, seed 26 from the poll
step. The baseline's two crashes (`ellipsoid_D3` seed 20, `ellipsoid_D10`
seed 7) came from two other unguarded GP updates; under the generator
those seeds follow other trajectories and finish (`dev/TODO.md`).

## Comparison with the baseline

`compare dev/experiments/population_baseline_20260924 <this population>`
(`comparison.md`): no flag in 54 tests. The median paired log10 error
ratio (this population over the baseline) lies between -0.37 and +0.35 in
every configuration, and its 95% interval contains zero in all but two:
`ellipsoid_D3_homo`, +0.26 [+0.06, +0.41] (median error 0.055 → 0.100,
solved 0.63 → 0.50), and `ellipsoid_D3_hetero`, +0.21 [+0.005, +0.38]
(0.199 → 0.255, solved 0.23 → 0.20). Their signed-rank tests give p = 0.25
and 0.096 before correction; for `ellipsoid_D3_homo` the bootstrap
interval of the median excludes zero while the signed-rank test of all the
pairs does not come near significance. With 18 intervals about one is
expected to exclude zero by chance (two or more with probability about
0.23 if they were independent); that both are the noisy variants of the
same target is a pattern this population cannot separate from chance. The largest changes of the fraction solved are
+0.17 (`timing_D5`, `sphere_D3_hetero`) and -0.13 (`ellipsoid_D3_homo`).
The crash count rises in `ellipsoid_D10` (1 → 2) and falls in
`ellipsoid_D3` (1 → 0).

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 36 tests.
- **Positive control**: the baseline's (its `README.md`), where three
  configurations at 50 D flagged against it, `ellipsoid_D10` on the error.
  This population has the same configurations, seeds and tests.

## What the comparison detects at 30 seeds

As for the baseline: 54 tests, the first Holm step at p ≤ 9.3e-4, a KS
statistic of at least 0.50, and, for the paired signed-rank test, a shift
of about 0.87 of the standard deviation of the paired log10 error ratios
at 80% power. The pairs share each seed's start point and target noise
(`benchmark_targets.py`); BADS's own draws differ between two PyBADS
versions whenever the number or order of draws changes.
