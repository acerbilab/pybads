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

## Not done

- No run on Windows, and no reference of the `periodic` suite there.
- The runs with `periodic_vars` take longer per evaluation than the same
  problems without it, 1.05 to 1.33 times in the deterministic
  configurations and about twice in the noisy ones; the cause is not
  investigated. gpyreg's periodic gradient computes each periodic
  dimension's term twice, which a change within gpyreg could avoid.
- No run of MATLAB BADS: the comparison with MATLAB is by reading its code
  (KD-B1-6).
- gpyreg 1.4.0, the release that carries `periods`, is not out yet. The
  `default` suite holds two of the periodic configurations (PI,
  2026-09-29), which its references lack; at gpyreg's release, the
  comparison run with its clone becomes the new references on both
  platforms, those two included (`dev/TODO.md`, "gpyreg releases after
  1.3.3").
