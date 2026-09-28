# W2-25 against its revert, on Linux, gpyreg 1.3.3

Row W2-25 of the port review makes a noisy run's move to an earlier iterate,
after the re-estimate at the end of an iteration, take the iterate's
location with its value (KD-B2-7 of `pybads/bads/README.md`); MATLAB BADS
moves the value alone. Its gate, the `default` suite at 30 seeds, flagged
nothing, but the fraction solved fell on `ellipsoid_D3_homo` (0.80 → 0.67),
`ellipsoid_D3_hetero` (0.20 → 0.13) and `sphere_D3_hetero` (0.43 → 0.37)
([`port_review_20260925/verification/wave2.md`](../port_review_20260925/verification/wave2.md),
"Fix pass"). This population measures that at 90 seeds: the five noisy
configurations, with the head and with W2-25 reverted, paired by seed.

**Outcome: no measurable effect.** The move reaches nearly every noisy run
(439 of the 450 pairs differ), and no test is flagged. The fraction solved
is 0.65 with W2-25 and 0.66 without, a paired difference of −0.011 [−0.051,
+0.029] over the five configurations. The drops of wave 2's gate belong to
the seeds it ran: on seeds 0-29, W2-25 lowers `ellipsoid_D3_homo`'s
fraction solved from 0.60 to 0.43, and on seeds 30-89 it raises it from
0.63 to 0.70.

## Arms

The five noisy configurations of `benchmark_targets.py`'s `default` suite
(`sphere_D3_homo`, `ellipsoid_D3_homo`, `sphere_D3_hetero`,
`ellipsoid_D3_hetero`, `multisensory_s1_D6_homo`), seeds 0-89, 450 runs an
arm, at default options:

| arm | the move after the re-estimate | code |
| --- | --- | --- |
| head | the incumbent takes the iterate's location with its value (`_update_incumbent_`), as since W2-25 (`a9fbb97`) | `58e7dd5`, `dev-next` with #88 |
| reverted | `self.u`, `yval`, `fval` and `fsd` take the iterate's values; `u_best` and `optim_state["u"]` stay at the old incumbent, so that the next pass sets `self.u = self.u_best`, as MATLAB BADS does (`bads.m:1111-1118`, `769`) and PyBADS did before `a9fbb97` | `58e7dd5` with `revert_w2-25.diff`, the local branch `exp/w2-25-reverted` (`0cc795f`) |

The arms are what they claim:

- The reverted arm's records name its worktree's package in
  `meta.pybads_source`, `0cc795f`, clean; the head's name `58e7dd5`.
- `test_re_estimation_moves_the_incumbent_with_its_value`
  (`test_noisy_runs.py`) passes at the head and fails on the reverted arm.
- The fingerprint of `dev/scripts/fingerprint.py` is `4146a986863602cb`
  (one BLAS thread) in both, the hash of the Linux reference: none of its
  noisy runs moves before its last re-estimate, as W2-25's fix report
  found.
- The head's seeds 0-29 reproduce the records of the Linux reference
  [`population_linux_wave4_20260927`](../population_linux_wave4_20260927/README.md)
  for these configurations: 150 of 150 runs equal in every field of
  `final` but the wall time.

## Command and provenance

```console
R=dev/scripts/runs/w225_20260928
W=dev/scripts/runs/worktrees/w225_reverted   # worktree of 0cc795f
ONLY=sphere_D3_homo,ellipsoid_D3_homo,sphere_D3_hetero,ellipsoid_D3_hetero,multisensory_s1_D6_homo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=<gpyreg clone at v1.3.3>
.venv/bin/python -u dev/scripts/population.py run --suite default \
  --only $ONLY --seeds 0-89 --workers 4 --out $R/head/pop
.venv/bin/python -u $W/dev/scripts/population.py run --suite default \
  --only $ONLY --seeds 0-89 --workers 4 --out $R/reverted/pop
python dev/scripts/population.py compare $R/reverted/pop $R/head/pop
python dev/scripts/population.py compare $R/<arm>/pop --split
python dev/experiments/w225_linux_20260928/w225_pairs.py \
  dev/experiments/w225_linux_20260928/reverted \
  dev/experiments/w225_linux_20260928/head \
  dev/experiments/population_linux_wave4_20260927
```

- gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3` (`98ab5a4`),
  selected with `PYTHONPATH`; the records' gpyreg version string,
  `1.3.4.dev11+g1893effc9`, is the metadata of the venv's editable install
  of `../gpyreg`, and `meta.gpyreg_source` names the clone. Their `pybads`
  version string is the metadata of the venv's editable install.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, the environment of the Linux reference. One
  BLAS thread per run, four runs at a time, a fresh process per run; the
  head from 16:40 to 17:01 UTC on 2026-09-28, the reverted arm from 17:01
  to 17:21.
- The files: `head/` and `reverted/`, one JSON record per run
  (`<label>_seed<seed>.json`) and `summary.md`; `comparison.md`, the
  comparison of the head (NEW) with the reverted arm (REF);
  `null_check_head.md` and `null_check_reverted.md`, each arm's even seeds
  against its odd seeds; `pairs.md`, the output of `w225_pairs.py`, the
  analysis of the pairs below; `revert_w2-25.diff`.

## Outcome

All 900 runs finished; none crashed. The comparison flags nothing in 15
tests (Holm; a flag needs a KS statistic of at least 0.267 at 90 against
90 runs), and neither null check flags anything in 10.

| configuration | changed pairs | median error, reverted → head | median evaluations | solved | paired difference in solved [95% CI] | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| `sphere_D3_homo` | 84 | 0.0131 → 0.0111 | 212 → 206 | 1.00 → 1.00 | +0.000 | 1 |
| `ellipsoid_D3_homo` | 88 | 0.076 → 0.078 | 313 → 308 | 0.62 → 0.61 | −0.011 [−0.144, +0.122] | 1 |
| `sphere_D3_hetero` | 90 | 0.086 → 0.103 | 322 → 322 | 0.54 → 0.50 | −0.044 [−0.167, +0.078] | 0.60 |
| `ellipsoid_D3_hetero` | 89 | 0.336 → 0.311 | 298 → 302 | 0.16 → 0.16 | +0.000 [−0.089, +0.089] | 1 |
| `multisensory_s1_D6_homo` | 88 | 0.178 → 0.188 | 502 → 538 | 0.97 → 0.97 | +0.000 [−0.033, +0.033] | 1 |
| all five | 439 | | | 0.66 → 0.65 | −0.011 [−0.051, +0.029] | 0.67 |

"Solved" is an error below the configuration's tolerance (0.1, and 0.5
for `multisensory_s1_D6_homo`). The paired difference is the head's
fraction solved minus the reverted arm's, with the percentile interval of
10,000 bootstrap resamples of the pairs within each configuration; the
McNemar test is exact, on the pairs that one arm solves and the other does
not (41 solved by the head alone, 46 by the reverted arm alone, over the
five).

- **Error.** The median paired log10 error ratio, head over reverted, lies
  between −0.007 and +0.000 on each configuration (`comparison.md`, with
  its intervals), and no signed-rank test is flagged.
- **Evaluations.** Only `multisensory_s1_D6_homo`'s differ beyond chance
  before the correction: the head takes more (median 538 against 502; KS
  p = 0.036, 0.54 after Holm), unflagged.
- **Fraction solved.** Pooled over the five configurations, the interval
  excludes a drop larger than about 0.05. One configuration's interval at
  90 seeds is wider: for the three that wave 2's gate named, its lower end
  lies between −0.09 and −0.17, so a drop of that size on one of them is
  not excluded.
- **Seeds 0-29 against 30-89.** The fraction solved moves in both
  directions between the two sets of seeds (`pairs.md`):
  `ellipsoid_D3_homo` 0.60 → 0.43 on seeds 0-29 and 0.63 → 0.70 on 30-89,
  `ellipsoid_D3_hetero` 0.13 → 0.23 and 0.17 → 0.12, `sphere_D3_hetero`
  0.60 → 0.60 and 0.52 → 0.45. The null checks show the same spread
  within one arm: the head's even and odd seeds solve `sphere_D3_hetero`
  in 0.44 and 0.56 of their runs.

The head's seeds 0-29 are the Linux reference's runs, whose comparison with
the code before the port review ([the consolidated
ledger](../../results/2026-09-28-port-correctness-review.md), "Net effect
on the benchmark") shows `ellipsoid_D3_homo` solved in 0.67 of the runs
before and 0.43 after. On those seeds, W2-25 accounts for most of the drop
(0.60 with it reverted); on the other 60 seeds it does not lower the
fraction solved.
