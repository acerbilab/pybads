# Sto-BADS's success rule at `a4dcd65`

The population that decides rows W0-12 and W0-13 of the port review
(`dev/TODO.md`, "The uncertainty interval of Sto-BADS"; PI, 2026-09-26):
`stobads=True` on the five noisy configurations of the `default` suite,
seeds 0-59, with the current rule, the rule without the mesh factor, the
search's uncertain moves limited to a positive estimated improvement, and
the last two together, against Sto-BADS off, with every decision of the
rule counted. The runs, the comparisons and the tables are in
[`experiments/stobads_linux_20260928/`](../experiments/stobads_linux_20260928/README.md).

**Verdict.** At default options Sto-BADS brings nothing measurable on
these configurations: the current rule brings no configuration closer to
its minimum than BADS without it, and spends more evaluations on
`sphere_D3_homo`. Most of its "certain" outcomes with a positive SD lie
within half a standard deviation of no improvement (W0-12), because the
mesh factor shrinks the interval at small meshes; on the noisy ellipsoids,
many more are decided on an SD of 0, by the sign of the estimate alone.
The rule without the factor, a z-test where the SD is positive, ends the
runs sooner at the same accuracy on four configurations, and is worse on
`ellipsoid_D3_hetero`. Most of the search's uncertain moves go to a point
estimated worse than the incumbent (W0-13), but they are few under the
current rule, and limiting them changes nothing measurable.

## Setup

Five arms of 300 runs, paired by seed: **base**, Sto-BADS off; **A**,
`stobads=True`, the rule `mu >= 1.96 * epsilon * mesh_size**2`, with `mu`
the estimated improvement and `epsilon` its SD from the GP; **B**, as A
with `stobads_frame_size_scaling_power = 0`, `mu >= 1.96 * epsilon`;
**C**, as A with the search's move on an uncertain outcome only where the
estimated improvement is positive, as the poll's since W4-15 (an
experimental patch behind an environment variable); **D**, B and C
together. `opp_stobads` is on in every Sto-BADS arm. PyBADS at `a4dcd65`'s
package code, gpyreg 1.3.3, Linux, the environment of the Linux reference.
The GP-health counters count each outcome of `_sto_success_improvement_`.

## Sto-BADS against BADS

With the current rule (A), no configuration ends closer to its minimum
than without Sto-BADS: no error flag in 15 tests, paired log10 error
ratios between −0.17 (`ellipsoid_D3_homo`, interval [−0.28, +0.02]) and
+0.09, and the runs above the tolerance within a few of the base's (158
of its 300 runs end closer than their paired run without it, 142 farther
or level). `sphere_D3_homo` takes more evaluations, 341 in median against
214 (flagged); `sphere_D3_hetero` 362 against 322, unflagged.

## W0-12: what "certain" means under the current rule

The rule calls an outcome certain when the estimated improvement `mu`
exceeds `1.96 * epsilon * mesh_size**2` in size, `epsilon` the SD of `mu`
from the GP. At the small meshes where the runs spend most of their
evaluations the interval is a small fraction of `epsilon`: 61% of the
search's certain outcomes and 38% of the poll's lie within half an SD of no
improvement (89% and 90% of the search's on the two spheres), where the
sign of `mu` is right with a probability of at most about 69%. On the two
noisy ellipsoids a further class appears: outcomes whose two SDs are both
0, the GP's zero predictive SDs, which the rule declares certain whatever
`mu` is, by its sign: 41.5% of the search's certain outcomes of
`ellipsoid_D3_homo`, 21% of `ellipsoid_D3_hetero`'s (10% of the search's
and 6% of the poll's over the five configurations).

A poll success enlarges the mesh; 5.2% of A's poll decisions are
successes, against 0.8% under B. The counters pool the certain successes
with the certain failures, so they do not show how far from zero the
successes alone are, nor whether the enlarged meshes explain the spheres'
extra evaluations. On `sphere_D3_homo` the runs of A end on the mesh size
in 33 of 60 runs, those of the base in 17.

The rule without the mesh factor (B, `stobads_frame_size_scaling_power =
0`) is a z-test on the GP's estimate where its SD is positive: none of
those certain outcomes lies within 1.96 SD, and most decisions become
uncertain (60% of the search's, 62% of the poll's). Where the SD is 0 it
is still a test of the sign of `mu`: on the noisy ellipsoids most of its
certain search outcomes are (59% on `ellipsoid_D3_homo`, 44% over the five
configurations). Its runs end sooner than those of A and of the base
(`sphere_D3_homo` 184 evaluations, `sphere_D3_hetero` 196,
`multisensory_s1_D6_homo` 384 against 535 without Sto-BADS), with the
base's accuracy on four configurations, and a worse one on
`ellipsoid_D3_hetero` (median error 0.60 against 0.30, flagged against the
base and against A).

## W0-13: the uncertain moves

Under `opp_stobads` every uncertain outcome of the search moves its
incumbent, and 58% of them (A) have a negative estimated improvement: most
uncertain moves go to a point that the GP estimates worse. Under the
current rule they are 3.7% of the search's decisions, and limiting them to
a positive estimated improvement, as the poll's since W4-15 (C), changes
no configuration measurably, against A or against the base. Under B they
are the majority of the decisions; limiting them there (D) keeps B's
savings of evaluations. Its error on `ellipsoid_D3_hetero` is no longer
flagged against the base (+0.25, interval [−0.05, +0.47]; median error
0.50 against the base's 0.30), and does not differ measurably from B's
(−0.09, interval [−0.29, +0.13]): the limit does not show an effect there.
D flags no error against B (its
evaluations differ from B's on three configurations, fewer on
`sphere_D3_homo`, more on `sphere_D3_hetero` and
`multisensory_s1_D6_homo`).

## Doublecheck

The same review (see the GP-health note, "Doublecheck") found the counts of
the rule's decisions right but for one class: the counters left out of
their histogram the certain outcomes whose SD is 0, decided by the sign of
`mu` alone, which are many on the two noisy ellipsoids. The shares of
"certain within half an SD" are now over all the certain outcomes (61% and
38% under A, where 68% and 41% were written), and the class has its own
column. Also corrected: the verdict's "no run" (no configuration), the
evaluations of `sphere_D3_hetero` (unflagged), the termination messages of
`sphere_D3_homo` (they differ from the base's), the causal reading of A's
poll successes (the counters cannot support it), and arm D's effect on
`ellipsoid_D3_hetero` (no longer flagged, not shown to be an effect). The
fix of W0-13 checked out: the same logic as arm C in every case, tests that
fail on the old code, and arm C's records at the merged head.

## For `dev/TODO.md` (after the Close step)

- **The item's population** has run (this note). Nothing it measured
  needs a fix for a default run: Sto-BADS is off by default.
- **W0-12.** Proposed: say in the description of
  `stobads_frame_size_scaling_power` that its default of 2 makes a
  "certain" outcome at small meshes one within a fraction of the estimate's
  SD, and that 0 makes the rule a z-test at `1.96` SD, which ends the runs
  sooner at the same accuracy on four of the five configurations and is
  worse on `ellipsoid_D3_hetero`. Changing the default to 0 (with D's
  limit, the arm that takes fewer evaluations than the base on every
  configuration with no error flagged against it) is the alternative;
  either way it is a change of a non-default mode, whose users see
  different results. **PI, 2026-09-28: document only**; the option's
  description in `advanced_bads_options.ini` says it.
- **W0-13.** Proposed: limit the search's uncertain move as the poll's
  (arm C's patch, as a fix with a test and a changelog line): it removes
  moves to estimated-worse points, costs nothing measurable under the
  current rule, and matters under power 0. **PI, 2026-09-28: fix it**;
  fixed on this branch, in `_search_step_`, with the tests of
  `test_stobads.py`, the changelog's "Sto-BADS poll and search" and
  `opp_stobads`'s description.
- **Scope.** Five noisy configurations at D ≤ 6 with the default noise
  handling; Sto-BADS may be meant for other regimes (heavier noise, larger
  budgets), which this population does not reach.
