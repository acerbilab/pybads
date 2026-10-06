# `ellipsoid_D3_hetero` over 60 more seeds, before and after `020d6a8`

Seeds 30-89 of the configuration `ellipsoid_D3_hetero` of the `default`
suite of `dev/scripts/benchmark_targets.py`, run at the commit before
`020d6a8` (`before/`) and after it (`after/`). With seeds 0-29 of
[`population_gpyreg133_20260924`](../population_gpyreg133_20260924/README.md)
(before) and
[`population_targetnoise_20260925`](../population_targetnoise_20260925/README.md)
(after), they give 90 paired seeds. `020d6a8` gives the GP the squares of
the noise standard deviations that the target returns.

The target is a 3-D ellipsoid `sum(a_i (x_i - c_i)**2)` with `a` = (1, 1e3,
1e6), shifted to a fixed point `c`, with noise of standard deviation
`1 + sqrt(f)` (`specify_target_noise=True`).

## Commands and provenance

```console
PYTHONPATH="<worktree at 10d74a7>;dev/scripts/runs/gpyreg/v1.3.3" .venv/Scripts/python.exe -u <worktree at 10d74a7>/dev/scripts/population.py run --suite default --only ellipsoid_D3_hetero --seeds 30-89 --out dev/scripts/runs/population/ellhet_before_20260925
PYTHONPATH="<worktree at c044fea>;dev/scripts/runs/gpyreg/v1.3.3" .venv/Scripts/python.exe -u <worktree at c044fea>/dev/scripts/population.py run --suite default --only ellipsoid_D3_hetero --seeds 30-89 --out dev/scripts/runs/population/ellhet_after_20260925
```

- PyBADS at `10d74a7` and at `c044fea`, each run by the `population.py` of
  a clean detached worktree at that commit, which imports the package of
  its own checkout (the raw records' `meta.pybads_source` names the
  worktree). The package code of the two differs only by `020d6a8`. gpyreg
  1.3.3 from a clone checked out at the tag `v1.3.3` (`98ab5a4`).
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1; one BLAS thread per
  run, one run at a time, a fresh process per run; from 16:33 to 16:55 on
  2026-09-25.

## Outcome

All 120 runs finished.

| Seeds 0-89 | Before | After |
|---|---|---|
| Median error | 0.21 | 0.54 |
| Interquartile range | 0.11–0.36 | 0.22–1.27 |
| 90th percentile | 0.61 | 2.36 |
| Largest error | 1.75 | 2,478 |
| Runs with error ≥ 1 | 4 | 28 |
| Solved (error < 0.1) | 0.21 | 0.13 |
| Median evaluations | 330 | 368 |

The error is smaller after the fix in 25 of the 90 pairs (signed-rank test
on the log10 errors, p = 8.5e-6; seeds 30-89 alone, 13 of 60, p = 2.1e-5).
The increase comes mostly from the flat coordinate (`a_1 = 1`): its median
contribution to the error rises from 0.057 to 0.145, and it carries more
than half of the error in 17 of the 28 runs at or above 1. Those runs end
between 0.95 and 4.6 from `c_1` along the flat axis, 14 of them with the
message that names `tol_mesh` and 3 with the one that names `tol_fun`. The
largest error, seed 78 (915 evaluations), is off by 0.05 along the
steepest axis.
