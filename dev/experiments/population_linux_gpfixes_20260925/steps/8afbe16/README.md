# Step: the default suite at `8afbe16` (row fix and GP mean prior)

The gate of `032dfcb` and `8afbe16`, the step before `97b2c66`: 18
configurations of the `default` suite × seeds 0-29 at `8afbe16`, compared
with the Linux reference of the base,
[`population_linux_targetnoise_20260925`](../../../population_linux_targetnoise_20260925/README.md).
`032dfcb` merges a repeated point into its own row of the function log
under `specify_target_noise`, and `8afbe16` re-centres the prior of the GP
mean at each rebuild of the local GP, as MATLAB BADS does, which changes
the runs of every configuration. The reference of the final code is the
population one level up.

## Command and provenance

```console
cd dev/scripts/runs/worktrees/wip-meanprior   # a clean worktree at 8afbe16
PYTHONPATH=<repository>/dev/scripts/runs/gpyreg/v1.3.3 <repository>/.venv/bin/python -u dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out <repository>/dev/scripts/runs/population/gate_meanprior_8afbe16
```

- PyBADS at `8afbe16`, run from a clean worktree at that commit (the
  records' `meta.git` and `meta.pybads_source`; their `pybads` version
  string, `1.1.1.dev23+g1c8c71d2b`, is the metadata of the venv's editable
  install, built at `1c8c71d`). gpyreg 1.3.3 from a clone checked out at
  the tag `v1.3.3` (`98ab5a4`), selected with `PYTHONPATH`.
- Linux (a cloud container), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1. One
  BLAS thread per run, four runs at a time, a fresh process per run; 25.3
  minutes, from 16:44 to 17:09 UTC on 2026-09-25.
- The files: one JSON record per run, `summary.md`, `comparison.md` and
  `null_check.md`, and in `rosenbrock_D6_seeds30-89/` the runs of the
  paragraph on `rosenbrock_D6` below.

## Comparison with the base reference

All 540 runs finished. `comparison.md` flags five configurations, all of
them better:

| Configuration | Median error, `1c8c71d` → `8afbe16` | Median evaluations | Solved | Median paired log10 error ratio [95% CI] |
|---|---|---|---|---|
| `ellipsoid_D3` | 4.5e-5 → 3.6e-6 | 152 → 148 | 0.90 → 1.00 | -1.28 [-1.94, -0.52] |
| `ellipsoid_D3_unbounded` | 4.1e-5 → 7.3e-6 | 175 → 158 | 0.90 → 1.00 | -0.86 [-1.72, -0.23] |
| `ellipsoid_D6` | 2.6e-5 → 2.3e-7 | 403 → 358 | 1.00 → 1.00 | -1.76 [-2.17, -1.40] |
| `ellipsoid_D10` | 7.5e-5 → 9.4e-7 | 725 → 631 | 0.97 → 1.00 | -1.99 [-2.31, -1.61] |
| `rosenbrock_D6` | 3.3e-5 → 3.8e-6 | 459 → 452 | 0.83 → 0.73 | -0.87 [-1.43, -0.14] |

The flag of `rosenbrock_D6` is its KS test on the error alone. Its
unsolved runs end in the function's local minimum (error 3.974), which 5
runs reach at `1c8c71d` and 8 at `8afbe16`, 1 of them in both. Over seeds
0-89 (seeds 30-89 run at both commits with `--only rosenbrock_D6 --seeds
30-89`, in `rosenbrock_D6_seeds30-89/before` and `after`), 27 runs reach it
before and 28 after, 9 of them in both, and in the 44 runs that both
versions solve the median error falls from 2.6e-5 to 1.5e-6 (signed-rank
p = 2.8e-12): the change makes the solved runs more precise and leaves the
basin a run falls into to chance.

No other test is flagged; the largest median paired log10 error ratio is
+0.26 (`sphere_D2`, [-0.42, +0.62]). The null check (`null_check.md`, even
against odd seeds) flags nothing in 36 tests.
