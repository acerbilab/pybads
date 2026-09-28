# Sto-BADS's success rule at `a4dcd65`, on Linux, gpyreg 1.3.3

The population that the PI asked for on 2026-09-26 to decide rows W0-12
and W0-13 of the port review (`dev/TODO.md`, "The uncertainty interval of
Sto-BADS"): `stobads=True` on the noisy configurations of the `default`
suite, with the current rule, the rule without the mesh factor, and the
search's uncertain moves limited to a positive estimated improvement,
against Sto-BADS off. The findings are in
[`results/2026-09-28-stobads-rule.md`](../../results/2026-09-28-stobads-rule.md).

## Arms

The five noisy configurations of `benchmark_targets.py`'s `default` suite
(`sphere_D3_homo`, `ellipsoid_D3_homo`, `sphere_D3_hetero`,
`ellipsoid_D3_hetero`, `multisensory_s1_D6_homo`), seeds 0-59, 300 runs an
arm:

| arm | options | code |
| --- | --- | --- |
| base | Sto-BADS off (the default) | the head |
| A | `stobads=True`: the success rule `mu >= 1.96 * epsilon * mesh_size**2` | the head |
| B | `stobads=True`, `stobads_frame_size_scaling_power=0`: `mu >= 1.96 * epsilon`, a z-test | the head |
| C | as A, and the search's move on an uncertain outcome only where the estimated improvement (`_eval_improvement_`) is positive, as the poll's since W4-15 | `arm_C.diff`, behind `PYBADS_EXP_OPP_STOBADS_POSITIVE=1` |
| D | B and C together | as C |

`opp_stobads` is on (its default) in every Sto-BADS arm: an uncertain
outcome moves the search's incumbent, and an uncertain poll moves to its
best point where that point improves on the incumbent (W4-15).

## Command and provenance

```console
R=dev/scripts/runs/stobads_20260928
ONLY=sphere_D3_homo,ellipsoid_D3_homo,sphere_D3_hetero,ellipsoid_D3_hetero,multisensory_s1_D6_homo
GP_HEALTH_OUT=$R/A/health \
PYTHONPATH=dev/scripts/gp_health_hooks:<gpyreg clone at v1.3.3> \
  .venv/bin/python -u dev/scripts/population.py run --suite default \
  --only $ONLY --seeds 0-59 --workers 4 --options '{"stobads": true}' \
  --out $R/A/pop
# B: --options '{"stobads": true, "stobads_frame_size_scaling_power": 0}'
# C, D: PYBADS_EXP_OPP_STOBADS_POSITIVE=1 and the population.py of a
#   worktree with arm_C.diff (the local branch exp/stobads-opp-positive,
#   73d5c28, on 1eb86f5), which puts that checkout first on sys.path
# base: seeds 0-29 copied from population_linux_wave4_20260927, seeds
#   30-59 run without options and without the counters
python dev/scripts/population.py compare $R/<REF arm>/pop $R/<NEW arm>/pop
python dev/scripts/gp_health.py summary $R/<arm>/health --pop $R/<arm>/pop
```

- PyBADS's package code is `a4dcd65`'s in every arm but C and D, whose
  records name `73d5c28`. The records of A name `b276da0` (12 runs, and 2
  more marked dirty) and `886bbff` (286); those commits differ only in
  `dev/scripts/gp_health.py`. The base's seeds 0-29 are the Linux
  reference's runs (`46af65a`, the same package behaviour: the default
  suite's records of `a4dcd65` equal them, `gp_health_linux_20260928`).
- gpyreg 1.3.3 from a clone at the tag `v1.3.3` (`98ab5a4`), selected with
  `PYTHONPATH`.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, one BLAS thread per run, four runs at a time.
- The Sto-BADS arms ran with the GP-health counters
  (`dev/scripts/gp_health_hooks/sitecustomize.py` at `b276da0`), which
  count each outcome of `_sto_success_improvement_` and change nothing:
  the fix of W0-13, run without them on the 300 runs of arm C, gave arm
  C's records (`dev/scripts/runs/stobads_20260928/run_gate.log`), and the
  worktree's arm C with the knob off gave arm A's records on two seeds.
- The files: `compare_<NEW>_vs_<REF>.md`, the comparisons (60 pairs a
  configuration; their Holm family is the 15 tests of one comparison);
  `summary_<arm>.md`, the counters' tables of a Sto-BADS arm, whose last
  table is its decisions; `arm_C.diff`. The per-run records stay on the
  machine that ran them, under the gitignored
  `dev/scripts/runs/stobads_20260928/`.

## Outcome

All 1,500 runs finished; none crashed. Medians over the 60 seeds of the
number of evaluations and of the error, and the runs whose error exceeds
the configuration's tolerance (0.1 for the spheres and the ellipsoids,
0.5 for the multisensory target):

| configuration | base | A | B | C | D |
| --- | --- | --- | --- | --- | --- |
| sphere_D3_homo | 214, 0.011, 0 | 341, 0.009, 0 | 184, 0.008, 0 | 312, 0.012, 0 | 174, 0.014, 0 |
| sphere_D3_hetero | 322, 0.095, 29 | 362, 0.110, 31 | 196, 0.094, 29 | 357, 0.101, 30 | 220, 0.102, 30 |
| ellipsoid_D3_homo | 308, 0.088, 26 | 320, 0.067, 22 | 288, 0.088, 27 | 310, 0.090, 29 | 292, 0.100, 30 |
| ellipsoid_D3_hetero | 304, 0.303, 50 | 298, 0.412, 54 | 254, 0.600, 56 | 292, 0.390, 50 | 269, 0.495, 54 |
| multisensory_s1_D6_homo | 535, 0.191, 1 | 557, 0.134, 3 | 384, 0.205, 2 | 518, 0.153, 3 | 405, 0.192, 4 |

The flags (Holm, 15 tests a comparison):

- A against base: `sphere_D3_homo`'s evaluations. No error flag; the
  paired log10 error ratios lie between −0.17 (`ellipsoid_D3_homo`,
  [−0.28, +0.02]) and +0.09.
- B against base: the evaluations of all five, fewer; the error of
  `ellipsoid_D3_hetero`, higher (+0.35 [+0.15, +0.50]).
- B against A: the evaluations of all five, fewer; the error of
  `ellipsoid_D3_hetero`, higher (+0.29 [+0.08, +0.40]).
- C against A and against base: nothing.
- D against base: the evaluations of all five, fewer; no error flag
  (`ellipsoid_D3_hetero` +0.25 [−0.05, +0.47]).
- D against B: the evaluations of `sphere_D3_homo`, `sphere_D3_hetero`
  and `multisensory_s1_D6_homo`; no error flag.
- D against A: the evaluations of all five, fewer; no error flag.

The rule's decisions, summed over the five configurations (the counters;
per configuration in `summary_<arm>.md`). The shares of certain outcomes
are over all the certain outcomes with an estimate: "SD 0" are those whose
two SDs are both 0, which the rule decides by the sign of `mu` alone, all
of them on the two noisy ellipsoids (41.5% of the search's certain
outcomes of `ellipsoid_D3_homo` under A, 59.3% under B); "within 0.5 SD"
those with a positive SD whose `abs(mu)` is under half of it.

| arm | search: success, uncertain | uncertain with mu < 0 | certain, SD 0 | certain within 0.5 SD | poll: success, uncertain | uncertain with mu < 0 | certain, SD 0 | certain within 0.5 SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 19.6%, 3.7% | 58% | 10% | 61% | 5.2%, 1.2% | 64% | 6% | 38% |
| B | 9.1%, 59.7% | 52% | 44% | 0% | 0.8%, 62.4% | 62% | 19% | 0% |
| C | 19.0%, 3.8% | 60% | 11% | 60% | 5.2%, 1.1% | 68% | 6% | 39% |
| D | 8.0%, 62.6% | 74% | 48% | 0% | 0.7%, 60.3% | 83% | 19% | 0% |

The counters pool the certain successes with the certain failures: they do
not show how far from zero the successes alone are. The counters of
`b276da0` left the certain outcomes with an SD of 0 out of the histogram;
`summary_<arm>.md` counts them as the certain outcomes the histogram lacks,
none of which had a missing estimate in two runs checked outcome by
outcome (the doublecheck).

The termination messages differ little in total (base 161 on `tol_fun`,
139 on the mesh size; A 144 and 156; B 142 and 158; C as the base), but
per configuration they can: on `sphere_D3_homo`, A stops on the mesh size
in 33 of 60 runs, the base in 17.
