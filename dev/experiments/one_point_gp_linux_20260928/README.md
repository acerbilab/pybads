# The GP on one point: MATLAB BADS's definition values, on Linux, gpyreg 1.3.3

The gate of `73d517a`, the PI's ruling of 2026-09-28 on the `dev/TODO.md`
item "The GP on a one-point training set.", which it closed (rows W2-37 and
W3-40 of the port review, wave 1's "Found while fixing" and wave 2's
"Found while verifying"): a GP whose initial training set holds one
distinct point is not fitted, and takes the values of MATLAB BADS's
definition (`gpdef/gpdefBads.m`): log length scales, log output scale and
log shape 0, the log noise SD at the log of the noise size, the mean at
the point's target, and the mean's prior centred there with the SD 1
(KD-B6-5). Before it, `init_and_train_gp` fitted the GP on that point,
under the priors alone: gpyreg's recommendations replace a single target
by `[0, 1]`, so that the mean's prior was centred at 0.5 whatever the
target, and printed six `RuntimeWarning`s, three times each. A non-box
constraint that leaves only `x0` feasible reaches it, as do
`max_fun_evals=2` with the noise test and `fun_eval_start=0`; no
configuration of the `default` suite does. This population measures the
change against its parent on the configurations that reach it: the
`geometry` suite (its thin bands, `sphere_band_D2` and `sphere_band_D3`,
without noise) and the `thinband` suite (the same bands with inferred
noise and with the target's noise), 30 seeds each, paired by seed.

**Outcome: no flag; the results change in exactly the runs that start on
one point and go past it.** The comparison flags nothing in 33 tests.
Every run of the thin bands starts on one point but seeds 21 and 22 at
D = 2 and seed 23 at D = 3 of each noisy configuration, whose initial
design of 32 points puts a second point in the band; those runs, and every
run of `edgesphere` and `ridge`, give identical results in both arms. So do
the runs of `sphere_band_D2` without noise, which end after 2 evaluations
at `x0` (W2-37) whatever the GP holds; their GP differs. The results of the
144 runs that change move by no consistent amount: the median paired log10
error ratio of the changed runs is between −0.11 and +0.17 per
configuration, none significant, and the fraction solved moves from 0.33
to 0.43 on `sphere_band_D3_hetero` and from 0.87 to 0.90 on
`sphere_band_D3_homo`. No run prints a warning from its initialization,
against every run that starts on one point before. The refits print
gpyreg's warnings on inputs without spread in a coordinate in every run of
`sphere_band_D3` (5 of 30 before) and in 3 of 30 of `sphere_band_D2_hetero`
(none before): on `sphere_band_D3` a GP with MATLAB's output scale of 1
fails the check of its predictions at once and is refitted at the first
evaluation that allows it, on `x0` and two poll points along the one axis
the band leaves free.

## Arms

| arm | the GP on one point at the initialization | code |
| --- | --- | --- |
| base | fitted under the priors, the mean's prior centred at 0.5 by gpyreg's recommendations | `b255effb`, with `d9772a04` (the `thinband` suite) cherry-picked as `07280f79` on a detached worktree, not pushed |
| change | MATLAB BADS's definition values, not fitted | `d9772a04`: `73d517a` with the `thinband` suite |

Both arms run the same `benchmark_targets.py`, so that each seed has the
same start point and noise stream in both; the records name each arm's
package in `meta.pybads_source`, clean. `07280f79`, which no pushed branch
holds, is `b255effb` with the two files of `d9772a04`, `dev/README.md` and
`dev/scripts/benchmark_targets.py`: its `pybads/` is `b255effb`'s.

## Command and provenance

```console
W=dev/scripts/runs/worktrees
R=dev/scripts/runs/population/p3_one_point
E=dev/experiments/one_point_gp_linux_20260928
git worktree add --detach $W/p3_base b255effb
git -C $W/p3_base cherry-pick d9772a04     # the records name 07280f79
git worktree add --detach $W/p3_change d9772a04
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=<gpyreg clone at v1.3.3>
for arm in base change; do
  for suite in geometry thinband; do
    .venv/bin/python -u $W/p3_$arm/dev/scripts/population.py run \
      --suite $suite --seeds 0-29 --workers 4 --out $R/$arm
  done
done
python dev/scripts/population.py compare $E/base $E/change
python dev/scripts/population.py compare $E/change --split
python dev/scripts/population.py compare $E/base --split
python $E/pairs.py $E/base $E/change
# gp_warnings.py copied into each worktree and run there:
C=sphere_band_D2,sphere_band_D3,sphere_band_D2_homo,sphere_band_D3_homo,sphere_band_D2_hetero,sphere_band_D3_hetero
.venv/bin/python -u $W/p3_$arm/$E/gp_warnings.py run --configs $C \
  --seeds 0-29 --out $E/warnings_$arm.jsonl
.venv/bin/python -u $W/p3_$arm/$E/gp_warnings.py run --band1 \
  --seeds 0-29 --out $E/warnings_band1_$arm.jsonl
python $E/gp_warnings.py summary $E/warnings_*.jsonl
```

- gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3` (`98ab5a4`),
  selected with `PYTHONPATH`, where `gpyreg.__file__` lies.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1. One BLAS thread per run, four runs at a time,
  a fresh process per run, from 21:30 to 21:41 UTC on 2026-09-28; the
  warnings, one process per arm, from 21:41 to 21:55.
- The files: `base/` and `change/`, one JSON record per run
  (`<label>_seed<seed>.json`) and `summary.md`; `comparison.md`, the
  comparison of the change (NEW) with the base (REF); `null_check_change.md`
  and `null_check_base.md`, each arm's even seeds against its odd seeds;
  `pairs.md`, the output of `pairs.py`, the runs that change; `warnings.md`,
  the output of `gp_warnings.py summary` over the four `warnings_*.jsonl`
  files, one record per run of where its warnings arose, its refits on one
  point, and its initial GP's training set, mean and prior of the mean.

## The other gates

- `dev/scripts/fingerprint.py`: `4146a986863602cb` at `d9772a04`, with one
  BLAS thread and with the default, as at the parent: its runs never start
  on one point.
- `dev/scripts/replay.py`: recorded at the base, with the base worktree's
  script, and at the change; `check` finds the eight runs identical.
- The test suite: 823 passed.

## Outcome

All 660 runs finished; none crashed. The comparison flags nothing in 33
tests (Holm; a flag needs a KS statistic of at least 0.5 at 30 against 30
runs, a large effect), the change's null check nothing in 22, the base's
nothing in 22.

| configuration | runs on one point | changed pairs | median error, base → change | median evaluations | solved | changed runs: median log10 error ratio (signed-rank p) |
| --- | --- | --- | --- | --- | --- | --- |
| `edgesphere_D2`, `_D4`, `_D3_homo`, `ridge_D2`, `_D4` | 0 | 0 | unchanged | unchanged | unchanged | — |
| `sphere_band_D2` | 30 | 0 | 15.1 → 15.1 | 2 → 2 | 0.00 → 0.00 | — |
| `sphere_band_D3` | 30 | 30 | 9.7e-6 → 9.9e-6 | 57 → 56.5 | 1.00 → 1.00 | −0.035 (0.42) |
| `sphere_band_D2_homo` | 28 | 28 | 7.99 → 6.45 | 116.5 → 124.5 | 0.03 → 0.03 | −0.001 (0.10) |
| `sphere_band_D3_homo` | 29 | 29 | 0.033 → 0.019 | 217 → 244.5 | 0.87 → 0.90 | −0.111 (0.20) |
| `sphere_band_D2_hetero` | 28 | 28 | 13.4 → 13.9 | 94.5 → 72 | 0.00 → 0.00 | +0.000 (0.77) |
| `sphere_band_D3_hetero` | 29 | 29 | 0.143 → 0.179 | 239.5 → 235 | 0.33 → 0.43 | +0.169 (0.38) |

"Solved" is an error below the configuration's tolerance (0.001 without
noise, 0.1 with it). "Runs on one point" counts the runs whose initial
training set holds one distinct point (`warnings.md`, the same in both
arms). The changed pairs differ in their results, the eight fields of
`final` that `pairs.py` compares (`x`, `fval`, `fsd`, `true_error`,
`func_count`, `iterations`, `message` and `crashed`); it leaves out the
other fields of `final`, the exception of a crashed run, the wall time, the
stage times and `min_noise_var`, the smallest noise variance of the run's
recorded GPs.

- **Which runs change.** The results of exactly the runs that start on
  one point, but those of `sphere_band_D2`, which end at `x0` after 2
  evaluations in both arms: 144 runs. Beyond the GP's values, the skipped
  fit's random draws no longer come from the run's generator, which shifts
  its later draws. The GP of `sphere_band_D2` differs in each of its 30
  runs: its `min_noise_var` is 0.001, the definition's, in the change, and
  0.00079 to 0.0010008 in the base, fitted. The stage times agree: no run
  that starts on one point has a `gp_init/gp_fit` stage in the change, and
  every run has one in the base.
- **The initial GP.** In the base, the fitted mean of the 201 runs that
  start on one point (the six configurations of `warnings.md` and
  `band1`) lies between 0.48 and 1.99 for targets at `x0` from 0.36 to
  118, under a prior centred at 0.5; in the change it is the target, the
  prior's centre too.
- **The noisy bands at D = 2** do not converge in either arm: the band is
  thinner than the mesh can resolve (W2-37), and their errors stay at
  several units.

## The warnings

`warnings.md` counts the runs whose `init_and_train_gp` or whose refits
(`_robust_gp_fit_`) print a warning, in both arms, over the six thin-band
configurations and `band1`, a problem outside the benchmark (D = 1, the
target `(x - 1)**2 + 10` with noise of SD 1, `x0 = 0` and the band
`|x| <= 0.005`; the module docstring of `gp_warnings.py` gives it).

| configuration | runs warned at the initialization, base → change | runs warned in a refit, base → change |
| --- | --- | --- |
| `sphere_band_D2` | 30 → 0 | 0 → 0 |
| `sphere_band_D3` | 30 → 0 | 5 → 30 |
| `sphere_band_D2_homo` | 28 → 0 | 0 → 0 |
| `sphere_band_D3_homo` | 29 → 0 | 0 → 0 |
| `sphere_band_D2_hetero` | 28 → 0 | 0 → 3 |
| `sphere_band_D3_hetero` | 29 → 0 | 0 → 0 |
| `band1` | 27 → 0 | 27 → 27 |

- **The initialization.** Each run that starts on one point printed the
  six warnings in the base (a log of zero at `covariance_functions.py:476`
  to `479`, and NumPy's "Degrees of freedom <= 0" and "invalid value
  encountered in divide" from the SD of one row at `480`), and prints none
  in the change.
- **The refits.** A refit on inputs with a coordinate without spread
  prints a log of zero at the same five lines of gpyreg's helpers, which
  `fit` calls. On `sphere_band_D3` every run of the change refits on
  three points at its fourth evaluation, the first that `_is_gp_refit_time_`
  allows (`func_count > D`): `x0` and the two points of a poll along the
  third axis, the one that the band leaves free, with no spread in the
  first two coordinates. With MATLAB's output scale of 1, the GP's
  predictions at the first poll points lie tens of SDs from their values
  (z-scores of 96 and −29 at seed 0), which fails the check of its
  predictions in `_is_gp_refit_time_` (MATLAB's `gppredcheck`) and calls
  for the refit; the base's fitted GP, with an output scale near the
  target's size, passed it (0.5 and 3.4), and its first refit came later,
  on inputs with spread, in 25 of the 30 runs. By a reading of its code,
  not run, MATLAB BADS, with the same values and the same check, refits at
  the same evaluation, and its GP code prints nothing there.
- **A refit on one point** occurs at D = 1: the noise test counts in
  `func_count`, on both sides, so that after it `func_count` is 2 > D. In
  27 of the 30 runs of `band1`, whose noise the test finds, the run's first
  refit is on `x0` alone, in both arms, with the six warnings; the other 3
  have a second feasible point in the initial design. Such a refit is fitted
  under the priors alone, in MATLAB BADS too by a reading of its code; what
  its fit does with its degenerate priors only MATLAB shows.

The warnings of the refits on inputs without spread, and the refit on one
point at D = 1, stay with gpyreg's helpers, whatever MATLAB computes (PI,
2026-09-29): `dev/TODO.md`, "For gpyreg's maintainers." [2026-09-29:
that item closed by the PI's ruling: the warnings go in gpyreg, every
value the same (acerbilab/gpyreg#66, in `dev/TODO.md`'s "gpyreg releases
after 1.3.3."), and the helpers' `[0, 1]` in place of a single target
stays (`dev/results/2026-09-28-port-correctness-review.md`, "Open
ends")].
