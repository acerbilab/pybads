# Reference population on Linux after three fixes to the GP: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Linux for `dev/scripts/population.py compare` until a
later reference replaces it (the one on Windows,
[`population_targetnoise_20260925`](../population_targetnoise_20260925/README.md),
predates the three fixes): 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3. It
replaces
[`population_linux_targetnoise_20260925`](../population_linux_targetnoise_20260925/README.md)
(`1c8c71d`), from which it differs by three commits:

- `032dfcb` merges a repeated point into its own row of the function log
  under `specify_target_noise` (the two configurations with target noise);
- `8afbe16` re-centres the prior of the GP mean at each rebuild of the
  local GP, as MATLAB BADS does (every configuration);
- `97b2c66` bounds the GP log length scales by the log of the maximum
  length scale, as MATLAB BADS does, where the port used the maximum
  itself (every configuration whose GP reaches the bound).

`steps/` holds the gates of the first two commits and of the third on top
of them.

## Command and provenance

```console
cd dev/scripts/runs/worktrees/part3   # a clean worktree at 97b2c66
PYTHONPATH=<repository>/dev/scripts/runs/gpyreg/v1.3.3 <repository>/.venv/bin/python -u dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out <repository>/dev/scripts/runs/population/gate_lenbound_97b2c66
```

- PyBADS at `97b2c66`, run from a clean worktree at that commit, whose own
  `dev/scripts/population.py` puts that checkout first on `sys.path` (the
  records' `meta.git` and `meta.pybads_source`; their `pybads` version
  string, `1.1.1.dev23+g1c8c71d2b`, is the metadata of the venv's editable
  install, built at `1c8c71d`). gpyreg 1.3.3 from a clone checked out at
  the tag `v1.3.3` (`98ab5a4`), selected with `PYTHONPATH`.
- Linux (a cloud container), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, as
  for the previous Linux reference. The container restarted between the
  step populations and this one (its kernel string changed from `fc-v37`
  to `fc-v42`); before this run, 16 runs of the `8afbe16` step (4
  configurations × seeds 0-3) were reproduced exactly. One BLAS thread per
  run, four runs at a time, a fresh process per run; 30.2 minutes, from
  18:46 to 19:16 UTC on 2026-09-25.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`; `steps/` below. The run went to
  `dev/scripts/runs/population/gate_lenbound_97b2c66`; the headers of the
  `.md` files name this directory instead.

## Outcome

All 540 runs finished. The fraction solved ranges from 0.00 (Rastrigin,
whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_linux_targetnoise_20260925 <this population>`
(`comparison.md`) flags five configurations, all of them better:

| Configuration | Median error, previous → this | Median evaluations | Solved | Median paired log10 error ratio [95% CI] |
|---|---|---|---|---|
| `ellipsoid_D3` | 4.5e-5 → 1.7e-6 | 152 → 142 | 0.90 → 1.00 | -1.29 [-1.87, -0.51] |
| `ellipsoid_D3_unbounded` | 4.1e-5 → 2.1e-6 | 175 → 153 | 0.90 → 1.00 | -1.12 [-1.82, -0.44] |
| `ellipsoid_D6` | 2.6e-5 → 1.1e-7 | 403 → 356 | 1.00 → 1.00 | -2.12 [-2.43, -1.80] |
| `ellipsoid_D10` | 7.5e-5 → 9.5e-7 | 725 → 624 | 0.97 → 1.00 | -2.04 [-2.30, -1.74] |
| `rosenbrock_D6` | 3.3e-5 → 3.8e-6 | 459 → 446 | 0.83 → 0.77 | -0.66 [-1.11, -0.14] |

The flag of `rosenbrock_D6` is its KS test on the error alone: the runs
that reach the global minimum end closer to it, and those that do not end
in the function's local minimum (error 3.974), 5 of 30 before and 7 after.
Which basin a run falls into changes with any perturbation of its
trajectory: over seeds 0-89 at `8afbe16`, 27 runs reach the local minimum
before and 28 after, 9 of them in both (`steps/8afbe16/README.md`).

No other test is flagged, and every other interval of the median paired
log10 error ratio contains zero; the largest ratio is +0.24
(`multisensory_s1_D6`, [-0.06, +0.51]). With target noise,
`ellipsoid_D3_hetero` has a median error of 0.30 (0.58 before) and
`sphere_D3_hetero` 0.10 (0.10); their evidence over 90 seeds is in
[`population_ellipsoid_hetero_linux_20260925`](../population_ellipsoid_hetero_linux_20260925/README.md).

## Steps

- **`032dfcb`** (`steps/032dfcb_comparison.md`, against the previous
  reference): no flag in 54 tests. The 16 configurations without target
  noise are identical run by run, since the changed branch runs only when
  the target returns a noise standard deviation. `ellipsoid_D3_hetero`
  changes in 18 runs and `sphere_D3_hetero` in 25
  (`steps/032dfcb_sphere_D3_hetero/`), with no change of its median error
  (0.10).
- **`8afbe16`** (`steps/8afbe16/`, a full population with its null check,
  against the previous reference): the same five configurations flagged,
  all better, and the `rosenbrock_D6` runs of seeds 30-89 at both commits.
- **`97b2c66` on top of `8afbe16`** (`steps/97b2c66_comparison.md`): no
  flag in 54 tests. Every interval of the median paired log10 error ratio
  contains zero but that of `sphere_D2` (-0.46, better). The largest
  ratio, +0.99 for `rosenbrock_D2` ([-0.19, +1.36], signed-rank p = 0.14),
  takes back the gain of `8afbe16` there: against the previous reference
  the ratio is -0.03; one of its runs ends at 1.4e-3, above the 1e-3 of
  "solved".

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
