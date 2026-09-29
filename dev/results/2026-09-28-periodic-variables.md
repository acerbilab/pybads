# Periodic variables (`periodic_vars`) for PyBADS 1.5

PyBADS supports periodic variables, MATLAB BADS's `PeriodicVars`, which it
refused before. This note records the decisions and the evidence; the
catalogue entry KD-B1-6 of `pybads/bads/README.md` states the behaviour
and its differences from MATLAB BADS, and `AGENTS.md` the call sites that
wrap points.

## Decisions (PI, 2026-09-28 and, where dated, 2026-09-29)

- The port goes into 1.5.0, and PyBADS's minimum gpyreg moves to the gpyreg
  release that carries the kernel's periods, 1.4.0 (2026-09-29;
  `dev/TODO.md`, "gpyreg releases after 1.3.3").
- The `default` suite holds two configurations of the `periodic` suite,
  `periodic_D4` and `periodic_D3_homo`, and its references, which lack
  them, are extended at gpyreg 1.4.0's release; the Linux reference stays
  at 30 seeds (2026-09-29).
- A periodic length scale is in the units of the other variables: gpyreg's
  kernel replaces a periodic squared difference `d**2` by the squared chord
  `(p/pi)**2 * sin(pi*d/p)**2`, which matches it at short range, so the
  prior and bounds of the length scales, `len_scale` and `poll_scale` treat
  a periodic variable as any other. MATLAB BADS's kernel works on the unit
  circle, and its prior and its use of the length scale in `udist` and
  `pollscale` convert between the two units only in part (KD-B1-6).
- gpyreg's three ARD kernels (`SquaredExponential`, `Matern`,
  `RationalQuadraticARD`) take fixed `periods`, not only the kernel that
  PyBADS uses; the isotropic ones refuse them, and so does `GP.quad`
  (gpyreg's branch `claude/todo-discussion-q8c9im`, `2c9cdfb`).
- A sixth example notebook adapts MATLAB BADS's `bads_examples.m`,
  Example 5.

## Evidence

**Runs without periodic variables do not move.** A run without periodic
variables builds its kernel without `periods` (`_gp_periods` gives
`None`), and `period_check` returns its input itself. On Linux (Python
3.11.15, NumPy 2.4.6, SciPy 1.17.1, one BLAS thread),
`dev/scripts/fingerprint.py` prints `4146a986863602cb` at `12cf2f29`
with gpyreg 1.3.3 and with gpyreg at `2c9cdfb`, the hash of the Linux
reference. On gpyreg's side, a hash of kernel matrices, gradients, fits,
predictions and one-point updates of every kernel, over several
dimensions and sizes, is the same on `main` and on the branch with
`periods` absent, `None` or all infinite, with one BLAS thread and with
the default.

**Tests.** The PyBADS suite passes with gpyreg at `2c9cdfb` (821 tests),
among them the checks of `periodic_vars`, `period_check`, `udist`, `ucov`
and the GP's periods, and two whole optimizations. The latter's
tolerances come from their errors over seeds 0-99 (the survey's section
"The seed sweep behind the tolerances"). gpyreg's suite passes on its
branch (911 tests, of which 140 are new).

**The `periodic` suite, `periodic_vars` on and off.**
[`experiments/population_periodic_linux_20260928/`](../experiments/population_periodic_linux_20260928/README.md)
holds the runs, 30 seeds of six configurations in each arm, and a control;
its "on" arm is the reference of the suite on Linux. With `periodic_vars`,
every run of the deterministic configurations is solved, where without it
a run whose start lies across the bounds from the minimum stops on the
bound (13 to 47% solved); the median error of the homoskedastic noisy
configuration falls from 0.091 to 0.028 (0.70 → 0.97 solved); and
MATLAB's Example 5, solved either way, ends with a fifth of the error. The
heteroskedastic noisy configuration's errors do not differ (p = 0.75), with
more evaluations. With the minima in the middle of the period, where
reaching them crosses no bound, the errors with and without `periodic_vars`
do not differ detectably under either kind of noise (paired signed-rank
p = 1.0 and 0.27 at 30 seeds): no worsening of noisy runs by the handling
of periodic variables is seen.

## Time per evaluation (2026-09-29)

Measured on Linux, with the evidence in
[`experiments/periodic_kernel_linux_20260929/`](../experiments/periodic_kernel_linux_20260929/README.md).
Two populations of the same code, run on 2026-09-28 and 2026-09-29,
differ by 0.96 to 1.09 in the median time per evaluation of a
configuration, which bounds what the timings below resolve.

**The cause was gpyreg's periodic kernel.** With gpyreg at `b44634f` (the
merge of acerbilab/gpyreg#61), the runs of the `periodic` suite with
`periodic_vars` took 1.1 to 1.35 times the median time per evaluation of
the same problems without it in the deterministic configurations
(MATLAB's Example 5 as long), and 2.3 (`homo`) and 1.8 (`hetero`) times in
the noisy ones, over 30 seeds. Profiles of seeds 0-2 of each
configuration, in which every call of gpyreg's kernel was repeated on the
same inputs without its periods, put the difference in gpyreg: the kernel
with periods took 2.5 to 3.9 times as long as the same calls without
them (1.6 times on `periodic_D2`), a difference of 27 to 38 % of each run
(10 to 11 % on `periodic_D2`). Most of it went to the predictions at the
ES search's 2048 candidates per generation, where the kernel at
`b44634f` computes an `fmod` and a sine of every pair of inputs, one
periodic dimension at a time, beside one `cdist` for the other
dimensions, and a fit's gradient computes each periodic dimension's term
a second time.

**The kernel on the circle.** gpyreg's commit `0f27db5` wraps each
periodic coordinate exactly into `(-p/2, p/2]` and maps it onto the circle
of circumference `p`, as two coordinates whose squared Euclidean distance
is the squared chord, and takes the distances with `cdist`: the
trigonometric functions take one evaluation per point and periodic
dimension instead of one per pair of points. MATLAB BADS's
`covPPERard_fast` maps a periodic input onto the unit circle in the same
way; the circle of circumference `p` keeps the length scale in the units
of the input (KD-B1-6). The commit `91ea28e` after it adds a test and
changes docstrings, not the kernels' code.
- *Numerics.* Over random inputs, the kernels of gpyreg's three ARD
  kernels (Matern at its three degrees) agree with those computed at
  `b44634f` to 2e-15 of the output variance, and their gradients to 2e-15
  of their largest entry; inputs a whole number of periods apart remain
  the same input to the last bit. At `b44634f` the chord of two close
  inputs kept the relative precision of their difference; at `0f27db5` it
  carries the rounding of their mapped coordinates, as a non-periodic
  dimension carries that of its scaled coordinates. The kernel's values
  differ between the two by the order of their rounding.
- *Runs without periodic variables* do not reach this code:
  `dev/scripts/fingerprint.py` prints `4146a986863602cb` with `0f27db5`.
- *The gate.* The `periodic` suite at 30 seeds against its Linux reference
  flags nothing. 34 of the 180 runs return the reference's point, error
  and number of evaluations; in the others a difference in the kernel's
  last bits leads the run along another path. gpyreg's suite passes
  (914 tests at `0f27db5`, 917 at `91ea28e`), and so does PyBADS's (871
  tests).
- *Speed.* The kernel with periods takes 1.4 to 1.6 times as long as the
  same calls without them (1.3 times on `periodic_D2`), a difference of 8
  to 13 % of each run (5 to 6 % on `periodic_D2`). The time per
  evaluation of each seed falls to 0.66 to 0.81 of its time at `b44634f`,
  in the median over the seeds of each configuration, and to 0.93 on
  `periodic_D2`. Relative to the same problems without `periodic_vars`, the
  ratios of the median times per evaluation are:

| Configuration | With `periodic_vars` / without, at `b44634f` | at `0f27db5` |
|---|---|---|
| `periodic_D2` | 1.09 | 0.99 |
| `periodic_D4` | 1.22 | 0.99 |
| `periodic_D6` | 1.35 | 0.93 |
| `periodic_rosenbrock_D4` | 0.98 | 0.77 |
| `periodic_D3_homo` | 2.26 | 1.48 |
| `periodic_D3_hetero` | 1.84 | 1.22 |

**What remains** is mostly the length of the noisy runs. With
`periodic_vars` they take more evaluations (medians 348 and 364 against
207 and 310), and the time per evaluation of a noisy run grows with its
length: a least-squares line of the time per evaluation on the number of
evaluations, fitted to each arm, gives at the median evaluations of the
runs without `periodic_vars` 1.19 (`homo`) and 1.09 (`hetero`) times the
value of the line fitted to those runs. Of PyBADS's own code, `udist`'s
periodic branch builds the `N x M x D` array of differences with
`np.mod`, where a run without periodic variables takes one `cdist`: 0.84
against 0.11 ms per call under cProfile, about 0.5 s of the 8.9 s that
the run of `periodic_D3_homo` at seed 0 takes under cProfile at
`b44634f` (`dev/TODO.md`, "`udist` on periodic variables").

`0f27db5` and `91ea28e` are on gpyreg's `main` since acerbilab/gpyreg#62
(merge commit `e10120c`), which gpyreg 1.4.0 is to carry
(`dev/TODO.md`).

## Not done

- No run on Windows, and no reference of the `periodic` suite there.
- No run of MATLAB BADS: the comparison with MATLAB is by reading its code
  (KD-B1-6).
- gpyreg 1.4.0, the release that carries `periods`, is not out yet. The
  `default` suite holds two of the periodic configurations (PI,
  2026-09-29), which its references lack; at gpyreg's release, the
  comparison run with its clone becomes the new references on both
  platforms, those two included (`dev/TODO.md`, "gpyreg releases after
  1.3.3").
