# Reference population on Windows: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Windows for `dev/scripts/population.py compare` until a
later reference replaces it (the one on Linux is
[`population_linux_20260925`](../population_linux_20260925/README.md)): 18
configurations of the `default` suite of `dev/scripts/benchmark_targets.py`
× seeds 0-29, each run at BADS's default budget (500 D) and ending on
BADS's own termination criteria, every random draw through the run's
`numpy.random.Generator`, with gpyreg 1.3.3. It replaces
[`population_gpyreg133_20260924`](../population_gpyreg133_20260924/README.md):
`020d6a8` gives the GP the squares of the noise standard deviations that a
target returns under `specify_target_noise`, which changes the runs of
`sphere_D3_hetero` and `ellipsoid_D3_hetero` and no other.

## Command and provenance

```console
PYTHONPATH=<worktree>;dev/scripts/runs/gpyreg/v1.3.3 .venv/Scripts/python.exe -u dev/scripts/population.py run --suite default --seeds 0-29 --out dev/scripts/runs/population/population_targetnoise_20260925
```

- PyBADS at `c044fea`, run from a clean detached worktree at that commit,
  first on `PYTHONPATH` (the records' `meta.pybads_source` names its path;
  their `pybads` version string, `1.1.1.dev23+g10d74a7ab`, is the metadata
  of the venv's editable install). gpyreg 1.3.3 from a clone checked out at
  the tag `v1.3.3` (`98ab5a4`), selected with `PYTHONPATH`.
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1; one BLAS thread per
  run (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1),
  one run at a time, a fresh process per run; 81.6 minutes, from 15:11 to
  16:32 on 2026-09-25.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`.

## Outcome

All 540 runs finished. The fraction solved ranges from 0.07 (Rastrigin,
whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_gpyreg133_20260924 <this population>`
(`comparison.md`) flags no configuration in 54 tests. The records of 16
configurations equal the previous reference's in every `final` field
except `wall_s`. The two configurations with target noise differ in every
run:

| Configuration | Median error, previous → this | Largest error | Solved | Median evaluations | Signed-rank p (Holm) |
|---|---|---|---|---|---|
| `sphere_D3_hetero` | 0.20 → 0.10 | 2.29 → 0.45 | 0.40 → 0.50 | 401 → 375 | 0.0081 (0.44) |
| `ellipsoid_D3_hetero` | 0.26 → 0.37 | 1.31 → 4.13 | 0.20 → 0.17 | 329 → 380 | 0.10 (1) |

The same two configurations at `10d74a7`, the commit before the fix, equal
the previous reference in every run (`final` fields except `wall_s`), so
the differences come from `020d6a8`.

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 36 tests.
- **Positive control**: the one in the first reference
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)).
  This population has the same configurations, seeds and tests.

## What the comparison detects at 30 seeds

As for the first reference: 54 tests, the first Holm step at p ≤ 9.3e-4, a
KS statistic of at least 0.50, and, for the paired signed-rank test, a
shift of about 0.87 of the standard deviation of the paired log10 error
ratios at 80% power. Between two versions on this platform, a run that a
change does not reach is identical in both populations.
