# W2-36: a noisy run's first incumbent from the initial GP, on Linux, gpyreg 1.3.3

Row W2-36 of the port review: a noisy run's first incumbent is the raw
minimum of its initial design, the lowest of its noisy observations, a
biased order statistic, with `fsd` set to `noise_size` at uncertainty level
1 (1 by default) or to the SD that the target returned there at level 2
(`BADS._init_optimization_`); the incumbent's first re-estimate from the GP
comes at the end of the second iteration (`poll_iteration > 0` in
`BADS.optimize`). MATLAB BADS does the same (`bads.m:447-457`, `1097`). The
PI kept MATLAB's behaviour and allowed its measurement as a separate step,
since W2-25 moved the noisy runs
([`port_review_20260925/verification/wave2.md`](../port_review_20260925/verification/wave2.md),
"Fix pass"). This population measures one variant, the first incumbent's
value from the initial GP, against the head: the five noisy configurations
at 90 seeds, paired by seed. The other variant, a re-estimate from the end
of the first iteration, is not measured.

**Outcome: no flag; one configuration better on the fraction solved.** The
variant changes 39 of the 450 pairs, and the comparison flags nothing in 15
tests. On `sphere_D3_hetero`, where the raw minimum lies furthest below the
true value, it changes 27 runs and raises the fraction solved from 0.50 to
0.61 (11 runs solved by the variant alone, 1 by the head alone; McNemar
p = 0.006, 0.03 after a Holm correction over the five configurations), in
both seed ranges, with a median of 324 evaluations against 322. Elsewhere
it changes at most
10 runs of a configuration and the fraction solved by at most one run.
Pooled, the fraction solved is 0.65 at the head and 0.66 with the variant,
a paired difference of +0.018 [+0.002, +0.033]. PyBADS keeps MATLAB's
behaviour.

## Arms

The five noisy configurations of `benchmark_targets.py`'s `default` suite
(`sphere_D3_homo`, `ellipsoid_D3_homo` and `multisensory_s1_D6_homo` at
uncertainty level 1, `sphere_D3_hetero` and `ellipsoid_D3_hetero` at level
2, with `specify_target_noise`), seeds 0-89, 450 runs an arm, at default
options:

| arm | the first incumbent's `fval` and `fsd` | code |
| --- | --- | --- |
| head | the raw minimum of the initial design, with `noise_size` or the target's SD there, until the re-estimate at the end of the second iteration, as MATLAB BADS | `5fd61bc`; the records of [`w225_linux_20260928/head/`](../w225_linux_20260928/head/), made at `58e7dd5` |
| variant | the initial GP's mean and SD at the incumbent, set once `init_and_train_gp` has fitted it at the end of `_init_optimization_`; `yval` stays the observation | `5fd61bc` with `variant_w2-36.patch`, committed as `9a7c361e` on a detached worktree, not pushed |

The arms are what they claim:

- The head arm is W2-25's head, whose 450 records this comparison reuses.
  The package changes between `58e7dd5` and `5fd61bc` (#92, `7110bba`,
  `5ac25aa`, `28fef97`, `c0aeaf4`) leave the fingerprint of
  `dev/scripts/fingerprint.py` at `4146a986863602cb` (one BLAS thread), and
  seeds 0-2 of each configuration, rerun at `5fd61bc` (`head_rerun/`),
  equal those records in every field of `final` but the wall time, 15 of
  15, with the same start points.
- The variant's records name its worktree's package in
  `meta.pybads_source`, `9a7c361e`, clean. In seeds 0-4 of each
  configuration, its first incumbent's `fval` and `fsd` equal the initial
  GP's mean and SD at the incumbent (25 of 25,
  `first_incumbent_variant_0-4.jsonl`), and the head's `fval` equals
  `yval` in all 450 runs (`first_incumbent_head.jsonl`).
- The variant's fingerprint is `4146a986863602cb` too: none of its three
  noisy runs changes.

## Command and provenance

```console
R=dev/scripts/runs/w236_20260928
W=dev/scripts/runs/worktrees/w236_variant
E=dev/experiments/w236_linux_20260928
git worktree add --detach $W 5fd61bc
git -C $W apply <repository>/$E/variant_w2-36.patch
git -C $W commit -am "exp: W2-36's variant"   # the records name 9a7c361e
ONLY=sphere_D3_homo,ellipsoid_D3_homo,sphere_D3_hetero,ellipsoid_D3_hetero,multisensory_s1_D6_homo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=<gpyreg clone at v1.3.3>
.venv/bin/python -u dev/scripts/population.py run --suite default \
  --only $ONLY --seeds 0-2 --workers 4 --out $R/headcheck/pop
.venv/bin/python -u $W/dev/scripts/population.py run --suite default \
  --only $ONLY --seeds 0-89 --workers 4 --out $R/variant/pop
python dev/scripts/population.py compare \
  dev/experiments/w225_linux_20260928/head $E/variant
python dev/scripts/population.py compare $E/variant --split
python $E/w236_pairs.py dev/experiments/w225_linux_20260928/head \
  $E/variant $E/head_rerun
.venv/bin/python -u $E/w236_first_incumbent.py --seeds 0-89 \
  --out $E/first_incumbent_head.jsonl
# the same script copied into $W, run there with --seeds 0-4, for
# first_incumbent_variant_0-4.jsonl
python $E/w236_first_incumbent.py summary $E/first_incumbent_head.jsonl
```

- gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3` (`98ab5a4`),
  selected with `PYTHONPATH`, where `gpyreg.__file__` lies. The head's
  records name another clone of the same commit, and the gpyreg version
  string of the venv's editable install at the time,
  `1.3.4.dev11+g1893effc9`; the variant's name the clone and `1.3.3`.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, the environment of W2-25's measurement. One
  BLAS thread per run, four runs at a time, a fresh process per run; the
  variant from 19:48 to 20:45 UTC on 2026-09-28, while other processes
  shared the machine, so that its wall times are about twice the head's
  and not comparable with them.
- The files: `variant/`, one JSON record per run
  (`<label>_seed<seed>.json`) and `summary.md`; `head_rerun/`, the 15
  records of the rerun; `comparison.md`, the comparison of the variant
  (NEW) with the head (REF); `null_check_variant.md`, the variant's even
  seeds against its odd seeds (the head's is W2-25's `null_check_head.md`);
  `pairs.md`, the output of `w236_pairs.py`, the analysis of the pairs
  below; `first_incumbent.md` and `first_incumbent_head.jsonl`, the first
  incumbent of every run at the head, from `w236_first_incumbent.py`;
  `variant_w2-36.patch`.

## Outcome

All 450 runs of the variant finished; none crashed. The comparison flags
nothing in 15 tests (Holm; a flag needs a KS statistic of at least 0.267 at
90 against 90 runs), and the null check of the variant nothing in 10, nor
did the head's.

| configuration | changed pairs | median error, head → variant | median evaluations | solved | paired difference in solved [95% CI] | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| `sphere_D3_homo` | 10 | 0.0111 → 0.0114 | 206 → 206 | 1.00 → 1.00 | +0.000 | 1 |
| `ellipsoid_D3_homo` | 1 | 0.078 → 0.081 | 308 → 309 | 0.61 → 0.60 | −0.011 [−0.033, +0.000] | 1 |
| `sphere_D3_hetero` | 27 | 0.103 → 0.073 | 322 → 324 | 0.50 → 0.61 | +0.111 [+0.044, +0.189] | 0.006 |
| `ellipsoid_D3_hetero` | 1 | 0.311 → 0.316 | 302 → 302 | 0.16 → 0.14 | −0.011 [−0.033, +0.000] | 1 |
| `multisensory_s1_D6_homo` | 0 | 0.188 → 0.188 | 538 → 538 | 0.97 → 0.97 | +0.000 | 1 |
| all five | 39 | | | 0.65 → 0.66 | +0.018 [+0.002, +0.033] | 0.057 |

"Solved" is an error below the configuration's tolerance (0.1, and 0.5
for `multisensory_s1_D6_homo`). The paired difference is the variant's
fraction solved minus the head's, with the percentile interval of 10,000
bootstrap resamples of the pairs within each configuration; the McNemar
test is exact, on the pairs that one arm solves and the other does not (11
solved by the variant alone, 3 by the head alone, over the five). The
median error of a configuration moves with a single changed run when that
run crosses the median.

- **Which runs change.** The first incumbent's value and SD serve until
  a search or a poll moves the incumbent, at the latest until the
  re-estimate at the end of the second iteration: in the test of whether a
  search or a poll improves on the incumbent, and the value in the reward
  of the search hedge, whose probabilities choose the next search. A run
  changes only where they change one of these outcomes. The first
  incumbent lies close to the minimum, relative to the noise, only on the
  spheres (a median `f - f_min` of 2.9 and 3.9, against 1.7e4 on the
  ellipsoids and 49 on `multisensory_s1_D6_homo`), and the raw minimum
  lies far below the true value only on `sphere_D3_hetero`, where the
  target's noise SD at the first incumbent is about 3
  (`first_incumbent.md`):

  | configuration | median `yval - f` | median `abs(yval - f)` | median GP mean `- f` | median `abs(GP mean - f)` | GP closer |
  | --- | --- | --- | --- | --- | --- |
  | `sphere_D3_homo` | −0.34 | 0.78 | −0.39 | 0.45 | 0.73 |
  | `ellipsoid_D3_homo` | +0.09 | 0.63 | +0.06 | 0.76 | 0.36 |
  | `sphere_D3_hetero` | −2.88 | 2.88 | −1.30 | 1.55 | 0.84 |
  | `ellipsoid_D3_hetero` | +11.2 | 62.9 | +8.6 | 71.3 | 0.43 |
  | `multisensory_s1_D6_homo` | −0.16 | 0.75 | −0.15 | 0.75 | 0.52 |

  `f` is the target's noiseless value at the first incumbent after the
  33 evaluations of the initialization; "GP closer" is the fraction of runs
  whose GP mean is closer to `f` than the raw minimum is.
- **`sphere_D3_hetero`.** In the 27 changed runs the error falls by a
  median log10 ratio of −0.185 (signed-rank p = 0.011 over those runs);
  over the 90 pairs the signed-rank test of `compare` gives p = 0.0125,
  0.19 after its Holm correction, and the KS tests of the error and the
  evaluations p = 0.31 and 0.87. The gain in the fraction solved holds in
  both seed ranges: 0.60 → 0.73 on seeds 0-29 (4 runs solved by the
  variant alone, none by the head alone) and 0.45 → 0.55 on seeds 30-89
  (7 and 1). The tolerance, 0.1, lies near the median error of both arms,
  so the fraction solved is sensitive to small errors there.
- **The other configurations.** `multisensory_s1_D6_homo` is identical run
  by run; the ellipsoids change in one run each, which the head solves and
  the variant does not; `sphere_D3_homo` changes in 10 runs, all solved in
  both arms.
- **The initial GP's SD.** On `ellipsoid_D3_homo` the initial GP's SD at
  the first incumbent is exactly 0 in 49 of the 90 runs (the zero
  predictive SDs of `dev/TODO.md`), and its 2 SD interval holds `f` in 0.44
  of the runs, against 0.94 for the raw minimum with `noise_size`. The
  variant gives those 49 runs a first incumbent with an SD of 0, and none
  of them changes; the one run that changes (seed 55) has a GP mean 28
  below `f`, with an SD of 28, where the raw minimum lies 0.05 below it.
