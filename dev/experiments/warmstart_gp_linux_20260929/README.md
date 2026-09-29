# The first GP with evaluations made before the run, on Linux, gpyreg 1.3.3

The gate of `58d922a1`, the PI's ruling of 2026-09-29: with evaluations
made before the run (`precomputed_evaluations`), the first GP's
hyperparameters are fitted on the incumbent's neighbours in the whole log,
chosen and capped as a rebuild chooses them (KD-B6-5). `_init_optimization_`
keeps the fit on the start and the initial design (MATLAB BADS's definition
values on one distinct point), then, within `gp_init`, rebuilds the GP with
`local_gp_fitting` and a refit, once. Before it, the first GP was fitted on
the start and the design alone, and the loop's first rebuild brought the
evaluations given into its training set under those hyperparameters until
the first refit, which waits until the run has made more than D
evaluations. In a rerun given the log of an earlier run with the same seed
and start, the log holds the whole initial design, which is not evaluated
again, so that the first GP held the start alone, with the definition's
values. No configuration of the other suites gives a run evaluations made
before it; the `warmstart` suite (`2822c561`) does: the sphere and the
ellipsoid at D = 3 and Rosenbrock's function at D = 6 without noise, and
the sphere at D = 3 with inferred noise (`homo`) and with the target's
noise (`hetero`), each run given the function log of an earlier BADS run on
the same target, made in the run's process from the seed: of 15 D
evaluations at the run's seed and start (`_rerun`), or of 20 D at the seed
plus 1000, with its own start and noise (`_other`). This population
measures the change against its parent on that suite, 90 seeds, paired by
seed: the gate's 30 (0-29), then 60 more (30-89), since the gate's pooled
fraction solved leaned towards the base.

**Outcome: flagged, and not adopted; the PI to rule.** At the gate's 30
seeds the comparison flags nothing in 30 tests, but the fraction solved,
pooled over the ten configurations, falls from 0.91 to 0.86 (14 runs solved
by the change alone, 27 by the base alone; McNemar p = 0.06, paired
difference −0.043 [−0.083, −0.003]). At 90 seeds the comparison flags
`rosenbrock_D6_rerun`, for more evaluations: the median rises from 394 to
426, its quantiles from the 10th to the 90th by 15 to 35, for an unchanged
error (the median paired log10 error ratio −0.05, signed-rank p = 0.97) and
a fraction solved of 0.83 against 0.79 (McNemar p = 0.57). The pooled
fraction solved at 90 seeds is 0.90 against 0.88 (53 against 69, p = 0.17,
−0.018 [−0.041, +0.007]), the hetero configurations' 0.64 → 0.59 and 0.67 →
0.60 none significant. The change does what it is for: in the base, every
rerun's first GP holds the start alone with the definition's values; in the
change none does, and it holds 35 to 90 rows of the log. Since the change's
fit draws from the run's generator, every run of the change parts from its
base (900 of 900 pairs differ), with the same start, noise and log. The
change is kept off the branch that holds this record: its commit,
`58d922a1`, is `change_58d922a1.patch` here.

## Arms

| arm | the first GP with evaluations made before the run | code |
| --- | --- | --- |
| base | fitted on the start and the initial design (the definition's values on one point); the evaluations given enter the GP at the loop's first rebuild | `f6c13872`, with `2822c561` (the `warmstart` suite) cherry-picked as `ee0d9c29` on a detached worktree, not pushed |
| change | then rebuilt on the incumbent's neighbours in the whole log and refitted, in the initialization | `58d922a1`: the change, on `2822c561`; not adopted, its diff in `change_58d922a1.patch` |

Both arms run the same `benchmark_targets.py`, so that each seed has the
same start point, noise stream and log of evaluations given in both; the
earlier run that makes the log gives no evaluations before it, and so runs
the same in both arms, which `first_gp.py logs` checks from the digests of
the records (`precomputed`). The records name each arm's package in
`meta.pybads_source`, clean. `ee0d9c29`, which no pushed branch holds, is
`f6c13872` with the four files of `2822c561`, `dev/README.md`,
`dev/scripts/benchmark_targets.py`, `population.py` and
`test_population.py`: its `pybads/` is `f6c13872`'s.

## Command and provenance

```console
W=dev/scripts/runs/worktrees
R=dev/scripts/runs/population/warmstart
E=dev/experiments/warmstart_gp_linux_20260929
git worktree add --detach $W/w_base f6c13872
git -C $W/w_base cherry-pick 2822c561      # the records name ee0d9c29
git worktree add --detach $W/w_change 58d922a1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=<gpyreg clone at v1.3.3>
for seeds in 0-29 30-89; do
  for arm in base change; do
    .venv/bin/python -u $W/w_$arm/dev/scripts/population.py run \
      --suite warmstart --seeds $seeds --workers 4 --out $R/$arm
  done
done
python dev/scripts/population.py summary $E/base
python dev/scripts/population.py summary $E/change
python dev/scripts/population.py compare $E/base $E/change
python dev/scripts/population.py compare $E/base --split
python dev/scripts/population.py compare $E/change --split
python $E/pairs.py $E/base $E/change
python $E/w236_pairs.py $E/base $E/change
python $E/first_gp.py logs $E/base $E/change
.venv/bin/python -u $E/first_gp.py run --root $W/w_$arm --seeds 0-29 \
  --out $E/first_gp_$arm.jsonl                  # for each arm
python $E/first_gp.py summary $E/first_gp_base.jsonl $E/first_gp_change.jsonl
```

- gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3` (`98ab5a4`),
  selected with `PYTHONPATH`, where `gpyreg.__file__` lies.
- Linux (a cloud container, kernel `6.18.44-fc-v37`, 4 virtual CPUs),
  Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1. One BLAS thread per run, four
  runs at a time, a fresh process per run: seeds 0-29 of the base from
  06:43 to 07:04 UTC on 2026-09-29 and of the change from 07:07 to 07:18,
  seeds 30-89 of the base from 07:19 to 07:40 and of the change from 07:40
  to 08:02. The base's seeds 0-29 ran at about half the pace of the rest
  (median wall time 11.5 s against 6.7 s), from the machine's load: its
  seeds 30-89 took 6.7 s, as the change's did; the comparison does not
  read the wall time.
- The files: `base/` and `change/`, one JSON record per run
  (`<label>_seed<seed>.json`) and `summary.md`; `comparison.md`, the
  comparison of the change (NEW) with the base (REF) at 90 seeds, and
  `comparison_seeds0-29.md`, the gate's at 30; `null_check_base.md` and
  `null_check_change.md`, each arm's even seeds against its odd seeds, at
  90 seeds, and the same at 30 (`_seeds0-29`); `pairs.md`, the output of
  `pairs.py` (the changed runs, the medians and the evaluations, from
  `one_point_gp_linux_20260928`) and of `w236_pairs.py` (the fraction
  solved with McNemar's test and a bootstrap interval, from
  `w236_linux_20260928`), at 90 seeds, and `pairs_seeds0-29.md` at 30;
  `logs.md`, the check that each pair was given the same log;
  `first_gp.py`, `first_gp_base.jsonl`, `first_gp_change.jsonl` and
  `first_gp.md`, the first GP of each run of seeds 0-29 as the
  initialization leaves it; `change_58d922a1.patch`, the change.

## The other gates

On the change's package code (`pybads/` at `58d922a1`), which moves
nothing without evaluations made before the run:

- `dev/scripts/fingerprint.py`: `4146a986863602cb` with one BLAS thread and
  with the default, as at `f6c13872`.
- `dev/scripts/replay.py`: recorded at `f6c13872`, in a worktree at it,
  and on the change's package code, each with its own `replay.py`; `check`
  finds the eight runs identical.
- `dev/scripts/make_oracle_fixtures.py --check --exact` passes, and so does
  `--check --exact --against` a `--dump` of `f6c13872`: 1056 outputs
  identical.
- The test suite, with the change's tests: 989 passed.

## Outcome

All 1800 runs finished; none crashed. The comparison at 90 seeds flags one
configuration in 30 tests (Holm; a flag needs a KS statistic of at least
0.289 at 90 against 90 runs), each arm's null check nothing in 20; at 30
seeds, nothing in 30, and nothing in 20 in each null check.

| configuration | median error, base → change | median evaluations | solved | solved by one arm alone, base, change (McNemar p) | changed runs: median log10 error ratio (signed-rank p) |
| --- | --- | --- | --- | --- | --- |
| `sphere_D3_rerun` | 1.58e-6 → 1.80e-6 | 75 → 68 | 1.00 → 1.00 | 0, 0 | −0.141 (0.82) |
| `sphere_D3_other` | 1.79e-7 → 1.89e-7 | 72 → 77 | 1.00 → 1.00 | 0, 0 | +0.009 (0.92) |
| `ellipsoid_D3_rerun` | 3.79e-6 → 3.57e-6 | 117 → 120 | 1.00 → 1.00 | 0, 0 | −0.036 (0.97) |
| `ellipsoid_D3_other` | 3.50e-6 → 3.09e-6 | 119.5 → 118 | 0.99 → 1.00 | 0, 1 (1.00) | +0.080 (0.76) |
| `rosenbrock_D6_rerun` | 2.03e-6 → 1.88e-6 | 394 → 426, flagged | 0.83 → 0.79 | 16, 12 (0.57) | −0.050 (0.97) |
| `rosenbrock_D6_other` | 1.47e-6 → 1.58e-6 | 406.5 → 416.5 | 0.87 → 0.84 | 12, 10 (0.83) | +0.018 (0.68) |
| `sphere_D3_homo_rerun` | 0.0106 → 0.0086 | 262 → 290 | 1.00 → 1.00 | 0, 0 | −0.087 (0.63) |
| `sphere_D3_homo_other` | 0.0066 → 0.0067 | 332 → 329 | 1.00 → 1.00 | 0, 0 | −0.057 (0.42) |
| `sphere_D3_hetero_rerun` | 0.070 → 0.087 | 335.5 → 326 | 0.67 → 0.60 | 22, 16 (0.42) | +0.014 (0.38) |
| `sphere_D3_hetero_other` | 0.068 → 0.081 | 359 → 357 | 0.64 → 0.59 | 19, 14 (0.49) | +0.021 (0.90) |

At 90 seeds. "Solved" is an error below the configuration's tolerance
(0.001 without noise, 0.1 with it). Every run changes: the change's refit
draws from the run's generator, which shifts its later draws, so that the
paired runs share their start, noise and log but not their trajectory.

- **The flag.** `rosenbrock_D6_rerun`'s evaluations: KS statistic 0.30,
  p = 0.0006 (0.017 after Holm). Their 10th, 25th, 50th, 75th and 90th
  percentiles move from 346, 371, 394, 420 and 454 to 362, 397, 426, 455
  and 477; the median paired change is +29, the
  iterations' median 16 in both arms. At the gate's 30 seeds the same test
  gave KS 0.37, p = 0.035, not flagged. `rosenbrock_D6_other` moves the same
  way by less (406.5 → 416.5, KS 0.13, p = 0.40). Why a fitted first GP costs
  a rerun on Rosenbrock's function more evaluations is not measured here.
- **The fraction solved.** The unsolved runs of `rosenbrock_D6` end at
  3.97, the local minimum of Rosenbrock's function at D = 6, in both arms;
  those of the hetero configurations end between 0.10 and 0.81. No
  configuration's change in the fraction solved is significant at 90
  seeds; the gate's pooled fall of 0.043 shrinks to 0.018, its interval
  containing 0.
- **The cost of the fit.** The stage `gp_init` takes a median 0.125 s in
  the change against 0.017 s in the base (seeds 30-89).

## The first GP

`first_gp.md` counts, per arm and configuration, the runs of seeds 0-29
whose start and initial design are one point, whose first GP holds one
point or the definition's values, the first GP's rows and, among them, the
rows at a point of the log given (in a rerun, the run's start is one), and
the runs with a stage `gp_init/gp_rebuild`.

| configurations | base: first GP | change: first GP |
| --- | --- | --- |
| the five reruns | the start alone, with the definition's values, in each of the 150 runs | 35 to 90 rows, every one at a point of the log given, fitted; none on one point |
| the five others | the start and the design: 5 rows at D = 3, 9 at D = 6, 33 with noise | 56 to 110 rows, 50 to 108 of them given, fitted |

The rerun's log holds its initial design: at level 2 (`hetero`) the run's
start merges into the given start's row (KD-B7-3), and the first GP holds
the 35 rows of the log; at level 1 (`homo`) the start is a row of its own
beside the given one, 36 rows.

## Conclusion

The change reaches what the ruling asked for, a first GP fitted on the
whole log's neighbours, and moves nothing without evaluations made before
the run. On the `warmstart` suite it improves no configuration, and at 90
seeds it is flagged for more evaluations on `rosenbrock_D6_rerun` at an
unchanged error, with a pooled fraction solved that leans towards the base
without significance. By the gate's rule a change flagged worse is not
adopted: the PI rules on it.
