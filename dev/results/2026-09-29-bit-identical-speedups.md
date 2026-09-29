# Speed-ups that change no result

Measured on 2026-09-29. The question: with the port review closed, what
could still make PyBADS's runs faster without changing any number they
compute, and by how much. The answer is a change to gpyreg, for its release
1.4.0, and one line of PyBADS: together they cut PyBADS's own time (the
run's time less the target's evaluations) by 23 to 28 % on the `profile`
suite under SciPy 1.17 and by 19 to 27 % under SciPy 1.18, and every
result is the same to the last bit.

## Summary

- **Where the time is.** gpyreg's calls take 75 to 89 % of a profiled run
  of the `profile` suite, PyBADS's own lines 5 to 14 %, and the NumPy and
  SciPy calls that PyBADS makes itself, with the target, the rest. The
  gains are in gpyreg, where earlier rounds, for PyVBMC
  (acerbilab/gpyreg#43) and for PyBADS (#60 and #62), had already removed
  the repeated work: what remained was the cost of intermediate arrays and
  of SciPy's Python layers around small calls.
- **The changes**, on gpyreg's branch `perf/bit-identical-speedups`
  (its code at `421f1b0`, on `e10120c`; merged as acerbilab/gpyreg#63,
  merge commit `4126dbe`), less the factorization of the training
  covariance by a direct call of LAPACK, which acerbilab/gpyreg#64
  (`61cbfd3`, merge commit `d84bf39`) gives back to
  `scipy.linalg.cholesky` since the direct call's bits are not SciPy
  1.18's (see "Not adopted"), with the share of PyBADS's own time that
  each saves alone on the `profile` suite, under SciPy 1.17.1:
  - the kernel without its gradient, and `predict`, computed in place:
    about six fewer arrays of the size of the training set by the ES
    search's 2048 candidates at each prediction; 9 to 23 %;
  - the kernel's gradient computed in place, with the squared differences
    of every dimension taken in one broadcast and the distances among the
    training inputs by `cdist(a, a)` instead of `squareform(pdist(a))`,
    whose SciPy 1.17 wrappers cost twice the distance computation; 2 to
    16 %, most on the ellipsoids, whose failed fits make many gradient
    calls;
  - the masses of the Gaussian priors, which gpyreg recomputes whenever
    the priors or the bounds change (at every rebuild of PyBADS's local
    GP), and the Gaussian draws of the space-filling design of `fit`, by
    `scipy.special.ndtr` and `ndtri` (about 1 µs a call) instead of
    `scipy.stats.norm` (about 50 µs); 3 to 7 %;
  - the four triangular solves that still went through SciPy's wrapper
    (two for the gradient of the objective, two for a posterior in the
    low-noise representation), by `_solve_triangular`, gpyreg's direct
    call of LAPACK, which already made most of its solves; 1 to 6 %
    together with the factorization's direct call, most on
    `ellipsoid_D3`, whose failed fits retry each factorization with a
    larger noise (the solves' share alone was not measured);
  - the gradient of the objective summed in one reused array; up to 4 %.

  The gradient's changes reach the squared exponential and Matern
  kernels too, and the in-place values the squared exponential kernel,
  PyVBMC's. On the PyBADS side, `local_gp_fitting` copies the
  priors it has just taken instead of taking them again from gpyreg, whose
  `get_priors` re-checks them (0.4 to 3 %).
- **Every result is unchanged.**
  - `dev/scripts/fingerprint.py` gives, with one BLAS thread,
    `4146a986863602cb` under SciPy 1.17.1 and `917c279f2c777dbb` under
    SciPy 1.18.1, as PyBADS `618652d6` with gpyreg `e10120c` does.
  - `dev/scripts/replay.py check` finds the 8 recorded runs identical to
    those of PyBADS `618652d6` with gpyreg `e10120c`, step by step, under
    either SciPy. It takes `--force`, since the recorder of `618652d6`
    leaves the CPU count and NumPy's CPU features out of the platform key.
  - `make_oracle_fixtures.py --check --exact --against` a `--dump` of
    `618652d6` with gpyreg `e10120c` compares 1,056 outputs, every one
    identical, under SciPy 1.17.1. The script refuses a dump made under
    another gpyreg, since its gate is for a change to PyBADS; the dump's
    record of gpyreg was set to the changed gpyreg's, the only field of
    the platform key that differed, so that the comparison could cross the
    change to gpyreg.
  - `dev/scripts/gpyreg_bitwise.py` finds the 31,974 outputs of `61cbfd3`
    identical to those of `e10120c` under SciPy 1.17.1 and 1.18.1. Under
    SciPy 1.18.1 they are also those of gpyreg 1.3.3, in a variant of the
    script that leaves out what 1.3.3 lacks (the periods and the GP that
    refuses a failed factorization, which becomes an ordinary one).
  - At `421f1b0`, under SciPy 1.17.1, two earlier sweeps: gpyreg's
    kernels against those of `e10120c`, 15,000 random cases, bit for bit,
    strides included (the three ARD kernels, Matern of degrees 1, 3 and 5,
    with and without periods, with the gradient, against test points and
    on the diagonal, a shape of 1 for the rational-quadratic kernel,
    repeated points, Fortran-ordered and strided inputs); and whole GPs
    against `e10120c`, each version in a process of its own: 5,724
    outputs of fits, predictions (with noise, the log predictive density,
    separate samples and cross-covariances), `predict_full`, the objective
    with its gradient, posteriors in the low-noise representation,
    single-point and full updates, `random_function`, GPs whose
    factorization needed a noise multiplier of 100 or 1000, and a GP that
    refuses a failed factorization, all bit for bit.
  - gpyreg's suite, with new tests that compare the kernels with their
    direct formulas, the factor of the training covariance with
    `scipy.linalg.cholesky`'s, the standard normal's functions with
    `scipy.stats.norm`, and `predict`'s in-place path with the path that
    returns the cross-covariance, under SciPy 1.17.1 and 1.18.1; and
    PyBADS's suite with gpyreg `d84bf39`, under SciPy 1.17.1.

## Setup

- **Code.** The original: PyBADS `618652d6` (#102) with gpyreg `e10120c`,
  the parent of acerbilab/gpyreg#63. The changed code: PyBADS's one line
  (#103, `e6733592`) with gpyreg's branch `perf/bit-identical-speedups`
  during the search; PyBADS `a5d69243` with gpyreg `61cbfd3` for the
  checks and the runs below.
- **Environment.** A Linux container with 4 virtual CPUs, one BLAS thread;
  Python 3.11, NumPy 2.4.6, SciPy 1.17.1, OpenBLAS 0.3.31 in NumPy and
  0.3.30 in SciPy, and Python 3.12, NumPy 2.5.3, SciPy 1.18.1, OpenBLAS
  0.3.34 in NumPy and 0.3.31 in SciPy (SciPy 1.18 needs Python 3.12);
  gpyreg selected by `PYTHONPATH`.
- **Search.** Under SciPy 1.17.1, as are the Summary's profile and
  shares and the prototype's ratios under "The runs": a cProfile of seed 0
  of each configuration of the `profile` suite
  (`dev/scripts/profile_suite.py`), then four searches, each of one
  region of the code: the ES search and its predictions, the
  hyperparameter fits, the GP's rebuilds and updates and the paths of
  noisy runs, and the overhead common to them with the configurations that
  the `profile` suite lacks (periodic, non-box constraints, bounds, the
  60-dimensional test problem). Each prototype was checked by the
  fingerprint and the replay before it was timed.
- **Timing.** Plain runs of `dev/scripts/profile_run.py`, seed 0, the
  original and the changed code alternating, two rounds under each SciPy;
  the tables give both rounds and the ratio of the faster of each. Two
  runs of one trajectory differ by 2 % at the median and by up to 9 %
  (11 % on `periodic_D4`).

## The runs

Under SciPy 1.17.1:

| configuration | own time, original (s) | own time, changed (s) | ratio |
|---|---|---|---|
| rosenbrock_D6 | 6.79 / 6.89 | 5.05 / 4.91 | 0.72 |
| multisensory_s1_D6_homo | 23.82 / 23.76 | 17.90 / 17.42 | 0.73 |
| ackley_D6 | 4.92 / 5.04 | 3.66 / 3.82 | 0.74 |
| sphere_D3_hetero | 9.09 / 9.00 | 6.87 / 6.71 | 0.75 |
| ellipsoid_D3 | 4.34 / 4.05 | 3.28 / 3.02 | 0.75 |
| ellipsoid_D10 | 21.10 / 20.01 | 15.37 / 15.58 | 0.77 |
| ellipsoid_D3_homo | 12.47 / 12.39 | 9.56 / 9.66 | 0.77 |
| periodic_D4 | 1.35 / 1.45 | 1.08 / 1.11 | 0.80 |

Under SciPy 1.18.1:

| configuration | own time, original (s) | own time, changed (s) | ratio |
|---|---|---|---|
| sphere_D3_hetero | 7.55 / 7.43 | 5.44 / 5.57 | 0.73 |
| ackley_D6 | 6.00 / 6.04 | 4.42 / 4.46 | 0.74 |
| rosenbrock_D6 | 6.96 / 6.81 | 5.02 / 5.07 | 0.74 |
| ellipsoid_D3 | 4.38 / 4.67 | 3.60 / 3.34 | 0.76 |
| multisensory_s1_D6_homo | 15.13 / 15.24 | 11.57 / 11.77 | 0.76 |
| ellipsoid_D10 | 17.37 / 18.50 | 13.56 / 14.03 | 0.78 |
| ellipsoid_D3_homo | 13.06 / 12.83 | 10.40 / 10.37 | 0.81 |
| periodic_D4 | 1.49 / 1.34 | 1.14 / 1.22 | 0.85 |

Each run of the changed code gave the original's result: the same
returned point and value, evaluations and iterations. The two
environments differ in their last bits (their SciPy, NumPy, OpenBLAS and
Python all differ), so that a run can follow another trajectory in each,
and a configuration's times do not compare across the tables. The ratio
compares the faster run of each; `periodic_D4`, of the `periodic` suite,
whose runs last about a second, is the least certain. A first measurement
of the changes, on a prototype of them, had given ratios of 0.65 to 0.81
on these configurations.

## Not adopted

- **The training covariance factorized by a direct call of LAPACK.**
  `421f1b0` called `potrf` with `lower=False`, as `scipy.linalg.cholesky`
  does up to SciPy 1.17, which saved SciPy's Python layers (3 to 11 µs a
  call, one BLAS thread) and gave SciPy's bits under SciPy 1.17. From
  SciPy 1.18, which needs Python 3.12, SciPy's upper factor is the
  transpose of `potrf(lower=True)`, which can differ in the last bits,
  and SciPy's call takes less time than the direct call from 50 inputs
  on, at every size measured up to 4000 (7 to 37 % less at 50 to 150
  inputs over two measurements, 14 to 29 % at 150 to 4000; one BLAS
  thread). Under SciPy 1.18 the direct call moved 4,168 of the 31,974
  outputs of `gpyreg_bitwise.py`, and gave the fingerprint the value it
  has under SciPy 1.17, `4146a986863602cb`; `61cbfd3` factorizes by
  SciPy's call again. Under SciPy 1.17, PyBADS's own time on the
  `profile` suite was 0.71 to 0.77 of the original's with the direct call
  (two rounds at `a79f84b`, with a lighter process running beside them),
  and is 0.72 to 0.77 without it, a difference within the spread between
  two runs.
- **Changes that move the last bits.** Solving the triangular system of
  `predict` from the right with BLAS's `dtrsm` instead of LAPACK's `trtrs`
  on the transposed system, which needs no copy of the right-hand side,
  would save a further 6 to 9 % on the configurations that the ES search
  dominates; the variances change by about 1e-8 relative, and by up to 1e-1
  where they are near zero, so that runs would part and the change would
  need the population comparison. Forming the objective's inverse with
  LAPACK's `potri` would save 1 to 4 %, with differences of about 1e-14.
- **Gains too small for the code they need.** Skipping the checks of
  `get_priors` on priors unchanged since they were checked (gpyreg, 0.3 to
  2 % beyond PyBADS's one line); caching the priors' masses (0.1 % once
  they take `ndtr`); skipping the rebuilds of a noisy run that repeat one
  on the same training set and hyperparameters, 9 % of them on
  `multisensory_s1_D6_homo` (1.5 to 2.4 %, none in deterministic runs);
  sharing the neighbours and priors of the re-estimation's iterates at one
  point (1.4 to 2.2 %); packing `contraints_check`'s bins into 64-bit keys
  (about 1 %); the ES loop's own code (5.5 % in all, no line above
  1.5 %); `shapiro` in `_is_gp_refit_time_` (0.8 to 1.9 %, only through
  SciPy's private `swilk`); the poll, the function logger, the history's
  records and deep copies, the option lookups and the stage timer (under
  0.5 % each).
- **No gain, or not the same bits.** Predicting the 2048 candidates in
  blocks gains nothing over the in-place prediction in whole runs, and its
  mean depends on the block size through OpenBLAS's `dgemv`. The only
  faster form of `round_half_away` turns -0.0 into 0.0.
- **No redundant work in the fits.** One Cholesky factorization serves the
  value and the gradient of each evaluation of the objective; the retry
  of a failed fit starts from another point; the failed evaluation itself
  costs 0.1 to 0.5 ms of a fit of 5 to 90 ms.

## Raw data

The profiles and the timings were machine-local and were not kept. The
comparison of gpyreg's kernels and Gaussian processes between two versions
is `dev/scripts/gpyreg_bitwise.py`, whose dumps of `e10120c` and
`421f1b0` under SciPy 1.17.1 hold 31,974 outputs, every one identical,
where `a79f84b` differs in 922 (the kernels' outputs on long-double and
infinite inputs); under SciPy 1.18.1, `4126dbe`, which factorizes the
training covariance by the direct call, differs from `e10120c` in 4,168,
and `61cbfd3` in none. It supersedes the two sweeps above, whose scripts
were not kept.

The first timings and comparisons were taken at the branch's first
commit, `a79f84b`, and the tables above at `61cbfd3`. The independent
review of `a79f84b` found that the broadcast and `cdist` differed from
`pdist` and `squareform` on inputs that are not float64 (a float32 input
under NumPy 1.x, a long-double input) and on an infinite coordinate;
`d53af17` takes `pdist` for the former and sets the diagonal to zero for
the latter, which changes nothing for finite float64 inputs, and
`421f1b0` imports the SciPy modules that gpyreg uses. At `421f1b0`, the
fingerprint and the replay are unchanged under SciPy 1.17.1.
