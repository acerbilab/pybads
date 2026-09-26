# Reference population on Windows after three fixes to the GP: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Windows for `dev/scripts/population.py compare` until a
later reference replaces it (the one on Linux is
[`population_linux_gpfixes_20260925`](../population_linux_gpfixes_20260925/README.md),
at `97b2c66`, whose package code is that of `ab4dded` but for #67): 18
configurations of the `default` suite of `dev/scripts/benchmark_targets.py`
× seeds 0-29, each run at BADS's default budget (500 D) and ending on
BADS's own termination criteria, every random draw through the run's
`numpy.random.Generator`, with gpyreg 1.3.3. It replaces
[`population_targetnoise_20260925`](../population_targetnoise_20260925/README.md)
(`c044fea`), from which the package differs by two pull requests:

- #67 (`068e57f`): the noise options, the loggers and the final estimate.
  Of what these runs reach, it changes the `fval` and `fsd` of the two
  configurations with target noise, which `compare` does not read, and the
  termination message of a run that ends on `tol_mesh`
  ([survey](../../results/2026-09-23-codebase-survey.md), below its
  candidate table);
- #66 (`ab4dded`), three fixes of the GP:
  - `032dfcb` merges a repeated point into its own row of the function log
    under `specify_target_noise` (the two configurations with target
    noise);
  - `8afbe16` re-centres the prior of the GP mean at each rebuild of the
    local GP, as MATLAB BADS does (every configuration);
  - `97b2c66` bounds the GP log length scales by the log of the maximum
    length scale, as MATLAB BADS does (every configuration whose GP reaches
    the bound).

## Command and provenance

From the main checkout's root, with a clean detached worktree at `ab4dded`
under `dev/scripts/runs/worktrees/`:

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 .venv/Scripts/python.exe -u dev/scripts/runs/worktrees/winref_ab4dded/dev/scripts/population.py run --suite default --seeds 0-29 --out dev/scripts/runs/population/population_gpfixes_20260925
```

- PyBADS at `ab4dded`, the head of `dev-next`, run by the worktree's own
  `population.py`, which puts that checkout first on `sys.path`: every
  record's `meta.git` and `meta.pybads_source` name the worktree at
  `ab4dded`, clean. Their `pybads` version string,
  `1.0.7.dev11+g0b3a65bfd.d20260924`, is package metadata that `importlib`
  found on the path, not the code that ran. gpyreg 1.3.3 from a clone
  checked out at the tag `v1.3.3` (`98ab5a4`), selected with `PYTHONPATH`.
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1; one BLAS thread per
  run (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1),
  one run at a time, a fresh process per run; 84.7 minutes, from 22:54 on
  2026-09-25 to 00:19 on 2026-09-26.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`.

## Outcome

All 540 runs finished. The fraction solved ranges from 0.00 (Rastrigin,
whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_targetnoise_20260925 <this population>`
(`comparison.md`) flags five configurations, all of them better, the five
that the same commits flag on Linux:

| Configuration | Median error, previous → this | Median evaluations | Solved | Median paired log10 error ratio [95% CI] |
|---|---|---|---|---|
| `ellipsoid_D3` | 2.4e-5 → 4.1e-6 | 147 → 142 | 0.93 → 1.00 | -1.09 [-1.71, -0.31] |
| `ellipsoid_D3_unbounded` | 4.5e-5 → 2.6e-6 | 163 → 149 | 0.83 → 1.00 | -0.96 [-1.75, -0.54] |
| `ellipsoid_D6` | 3.4e-5 → 1.3e-7 | 400 → 348 | 1.00 → 1.00 | -2.32 [-2.86, -1.38] |
| `ellipsoid_D10` | 6.8e-5 → 8.7e-7 | 705 → 632 | 1.00 → 1.00 | -1.98 [-2.30, -1.66] |
| `rosenbrock_D6` | 4.6e-5 → 1.4e-6 | 468 → 431 | 0.77 → 0.70 | -1.13 [-2.10, -0.52] |

The flag of `rosenbrock_D6` is its KS test on the error alone (the
signed-rank test gives p = 0.16): the runs that reach the global minimum
end closer to it, and those that do not end in the function's local
minimum (error 3.974), 7 of 30 before and 9 after. Which basin a run falls
into changes with any perturbation of its trajectory, as on Linux.

No other test is flagged. Of the other intervals of the median paired
log10 error ratio, one excludes zero, `sphere_D2` (-0.20 [-0.64, -0.05],
better); the largest ratio is +0.19 (`multisensory_s1_D6`, [-0.25, +0.53]).
With target noise, `ellipsoid_D3_hetero` has a median error of 0.25 (0.37
before) and `sphere_D3_hetero` 0.091 (0.10). No run is identical to its
counterpart in the previous reference, since `8afbe16` changes the GP of
every configuration.

## The prior of the GP mean outside the bounds of the mean

In 67 runs the log of the GP hyperprior evaluates to NaN during the
hyperparameter fits: in all 30 of `ackley_D6`, 28 of `sphere_D10` and 9 of
`sphere_nonbox_D3`. Their stderr carries two `RuntimeWarning`s, `invalid
value encountered in scalar subtract` from gpyreg's log prior
(`gaussian_process.py:2354`, `lp -= masks["log_norm"]`) and `invalid value
encountered in subtract` from NumPy's variance; the log of the previous
reference has none. The cause, from seed 0 of `sphere_D10` and of
`ackley_D6` rerun with every `RuntimeWarning` recorded (a probe kept on the
machine that ran it, `dev/scripts/runs/LOCAL.md`): the bounds of the
constant GP mean are set once, from the initial design (`_gp_hyp`), while
`8afbe16` re-centres its prior at the targets of each local training set.
Once a run has descended far below its initial design, the prior lies many
standard deviations below the lower bound of the mean, for instance
N(0.42, 0.078²) against [8.87, 175.8] in `sphere_D10` and N(0.45, 0.076²)
against [6.77, 12.78] in `ackley_D6`; its mass inside the bounds, gpyreg's
normalization constant, underflows to 0, and its log to -inf. The fits of
the search and of the poll rebuilds (`_robust_gp_fit_`) then meet a NaN
log posterior, and the mean settles at its lower bound, above every
training target (at most 1.58 and 0.77 there): in `sphere_D10` seed 0,
the mean of 9 of the 15 GPs recorded at the ends of the iterations is
8.87, its lower bound, while the largest training target of the last one
is 2.8e-5. MATLAB BADS leaves the
bounds of the mean infinite (`gpdefBads.m`), a row of the survey's
candidate table; the port review takes it up (slice B6 of
`dev/plans/port-correctness-review.md`). None of these runs failed, and
none of the three configurations changed significantly: their median
errors are 4.7e-4, 7.2e-8 and 1.2e-5, against 3.9e-4, 8.4e-8 and 9.0e-6
before.

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
