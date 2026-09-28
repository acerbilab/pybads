# Where PyBADS spends its time, and what the rank-1 GP update would change

Measured on 2026-09-28 on six configurations of the default benchmark
suite. The questions: which stages of a run take PyBADS's own time (the
run's time less the target's evaluations); what computing the
optimization target without a copy of the GP saves; and whether gpyreg's
single-point ("rank-1") update of the GP posterior could replace the full
recomputation that PyBADS makes after each new point, how far its results
differ, and what it would save.

## Summary

- PyBADS's own time is 15 to 35 ms per evaluation on these
  configurations. Two stages take nearly all of it: the GP predictions at
  the candidates of the evolution-strategy (ES) search, 34 to 68 %, and
  the GP hyperparameter fits, 14 to 52 %. Every other stage takes at most
  9 %: the rebuilds of the local training set 3 to 8 %, the
  re-evaluation of the history in noisy runs 4 to 9 %, the posterior
  update after a new point 1 to 3 %, and the deep copies that
  `IterationHistory` records under 0.3 %.
- Inside those two stages, one gpyreg function dominates: the rational
  quadratic ARD kernel (`RationalQuadraticARD.compute`), 31 to 45 % of the
  whole run in its own code. Its gradient recomputes the same power of the
  distance matrix once per input dimension; computed once, the gradient
  takes half the time at D = 6 and D = 10, with bit-identical results.
- Computing the target only where it is read, and predicting it from the
  GP's own posterior when the best iteration's hyperparameters are the
  GP's, cuts the target's stage from 1.9 to 5.0 % of the own time to 0.3
  to 1.3 %: a saving of 1.6 to 3.7 %, with the same trajectories.
- gpyreg's rank-1 update agrees with the full recomputation to about
  1e-11 of the targets' spread on three of the six configurations, the
  one with the target's noise included, and to 8e-4 on `rosenbrock_D6`,
  whose GPs sit at the noise's lower bound. On the ellipsoids, 40 to 44 %
  of the additions happen on a GP whose factorization needed gpyreg's
  noise multiplier; there the two differ by up to 1e-2 of the spread, and
  by 0.21 once, when the multiplier that the rank-1 update carries over
  differs from the one the recomputation picks. The rank-1 update falls
  back to the full recomputation in 0 to 10 % of the additions. It would
  save 0.3 to 2.5 % of PyBADS's own time.

## Setup

- **Code.** The base is PyBADS at `81385ac`, whose default runs are those
  of `a4dcd65`. The head is the base with two changes, which #84
  (`79697835`) also makes: the timer measures with `time.perf_counter`,
  which changes no computation, and the target change (the search
  computes the target only for a search acquisition that reads it, which
  the LCB does not; the poll predicts it from the GP's own posterior when
  the best hyperparameters are the GP's). Where they are not, the
  measured head predicted from a shallow copy of the GP, and #84 from a
  deep copy, as the base does; #84's saving is smaller by the cost of
  those copies, under 0.3 % of the own time (the deep copies of all the
  base's target calls take 0.6 to 1.2 %, below).
- **Environment.** Windows 11 laptop, Python 3.12.6, NumPy 2.5.3, SciPy
  1.18.1, OpenBLAS 0.3.34. gpyreg 1.3.3 from a clone at its tag
  (`98ab5a4`), on `PYTHONPATH`. One BLAS thread (`OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1). One run per
  process, one process at a time.
- **Configurations.** `ellipsoid_D10`, `rosenbrock_D6` and `ackley_D6`
  (deterministic); `multisensory_s1_D6_homo` and `ellipsoid_D3_homo`
  (noise inferred); `sphere_D3_hetero` (noise given by the target). Each
  run at the suite's budget of 500 D, set up as `dev/scripts/population.py`
  sets it up from `dev/scripts/benchmark_targets.py`. Seeds 0, 1 and 2 of
  each configuration, at the base and at the head. The base and the head
  ran the same trajectory in all 18 pairs: the same returned point,
  value and number of evaluations.
- **Stage times.** Wrappers timed with `time.perf_counter`, in the same
  process, around the functions of each stage. A stage's time leaves out
  the stages nested in it. What runs inside the re-evaluation of the
  history counts there. The wrappers change no computation.
- **Hot spots.** One run of each configuration (seed 0) at each commit
  under cProfile.
- **Rank-1 update.** Beside every call of `add_and_update_gp` in the head's
  runs (seeds 0 to 2), a deep copy of the GP took the new point through
  gpyreg's rank-1 path, `gp.update(X_new, y_new, s2_new)` with no
  hyperparameters. The copy was compared with the GP after PyBADS's full
  recomputation: predictive means and variances at the training inputs
  and at 20 random convex combinations of them, and the parametrization
  and noise multiplier of the posterior. The run continued on PyBADS's
  own GP, and its trajectory is the profiled one. Synthetic GPs built as
  PyBADS builds them completed the check: the RQ-ARD kernel, a constant
  mean, a constant noise term with and without a noise variance per point,
  and the hyperparameter ranges of `_gp_hyp`.

Stage times vary by about 5 % between two runs of the same trajectory;
the base's and the head's own times differ by that much in either
direction. The target's stage is measured directly and its change is far
larger than that noise.

## Where the own time goes

Shares of PyBADS's own time at the head, summed over seeds 0 to 2:

| configuration | own time per run, s (base → head) | ms per evaluation | hyperparameter fits | training-set rebuilds (rest) | posterior update after a new point | ES candidates | search (rest) | poll (rest) | re-evaluation of the history | `IterationHistory` records | target (base → head) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid_D10 | 19.3 → 18.1 | 27.2 | 47 % | 3 % | 1 % | 44 % | 2 % | 1 % | 0 % | 0.1 % | 3.3 → 0.8 % |
| rosenbrock_D6 | 8.3 → 7.7 | 19.6 | 20 % | 4 % | 1 % | 68 % | 4 % | 2 % | 0 % | 0.2 % | 3.7 → 0.8 % |
| ackley_D6 | 5.4 → 5.6 | 14.5 | 19 % | 4 % | 1 % | 66 % | 4 % | 3 % | 0 % | 0.3 % | 5.0 → 1.3 % |
| multisensory_s1_D6_homo | 12.3 → 11.8 | 22.6 | 20 % | 7 % | 3 % | 58 % | 3 % | 2 % | 6 % | 0.1 % | 3.7 → 0.7 % |
| ellipsoid_D3_homo | 10.8 → 10.6 | 34.5 | 52 % | 5 % | 1 % | 34 % | 2 % | 1 % | 4 % | 0.1 % | 1.9 → 0.3 % |
| sphere_D3_hetero | 5.2 → 5.0 | 19.4 | 14 % | 8 % | 2 % | 60 % | 3 % | 1 % | 9 % | 0.2 % | 3.2 → 0.7 % |

The columns:
- **Hyperparameter fits:** `gpyreg.GP.fit`, wherever it is called.
- **Training-set rebuilds (rest):** `local_gp_fitting` without its fits.
- **ES candidates:** `ESSearchHedge.__call__`, which generates the
  candidates, removes those already evaluated and scores them by LCB.
- **Search and poll (rest):** the step's own code, and the LCB of the
  points it chooses among.
- **Rest:** the initial design, the initial GP, the function logger and
  the main loop. They take 0.4 to 0.9 %, the column left out of the table.

The target's evaluations take 0.03 s or less per run, except those of
`multisensory_s1_D6_homo`, which take about 0.3 s.

## Inside the two large stages

cProfile at the head, seed 0:
- **The kernel.** `RationalQuadraticARD.compute` of gpyreg takes 31 to 45 %
  of each run in its own code (1.8 to 10.2 s of 6 to 23 s). It is called by
  `predict` and by the objective of the fits.
  - `predict` computes the kernel between the training set and each
    generation of the ES candidates (2048 per generation, two per search
    at default): about 2 to 8 ms for 50 to 150 training points.
  - The fits compute it with its gradient. The elementwise power in
    `M ** (-alpha)` dominates both. The gradient computes
    `sf2 * M ** (-alpha - 1)` again for each of the D length scales.
    Computed once before the loop, the gradient's results are
    bit-identical. The time of kernel with gradient drops from 0.22 to
    0.16 ms at D = 3, N = 50, from 1.15 to 0.55 ms at D = 6, N = 110, and
    from 3.97 to 2.01 ms at D = 10, N = 150.
  - On `ellipsoid_D10` the fits' kernel takes 6.0 s of a 23-s run, so the
    fits would lose about a quarter of their time there. This is a change
    to gpyreg, not to PyBADS.
- **The triangular solves.** gpyreg's triangular solves, in the
  predictions and the fits, take 6 to 11 % with SciPy's.
- **Removing evaluated candidates.** `contraints_check` removes the
  candidates already evaluated or infeasible. It takes 4.5 to 13 % of the
  run, nearly all of it called by the ES search, and most of that in
  `np.unique` (its sort) over the candidates. A later change, the entry
  "Cost of removing evaluated candidates" of `CHANGELOG.md`, replaced its
  two `np.unique` calls by one stable sort of the candidates' bins, which
  returns the same candidates in the same order. Both versions were timed
  on every call of the same runs, the old one's output taken: seeds 0-2 of
  the `default`, `oned`, `bounds` and `geometry` suites, on Linux (Python
  3.11, NumPy 2.4.6), one BLAS thread per run and four runs at a time. The
  new check took 16 to 44 % of the old one's time, and a run's time fell
  by 4 to 24 % (median 12 %), over 35 of the 36 configurations; the runs
  of `sphere_band_D2` end within 0.05 s.
- **Deep copies.** They take 0.6 to 1.2 % of the base's run and 0.1 to
  0.5 % of the head's. Most of the target's saving comes from the
  posterior that it no longer recomputes, not from the copy.

## The target change

| configuration | base: target time (calls, ms per call) | head: target time (calls, ms per call) | saving, % of the base's own time |
|---|---|---|---|
| ellipsoid_D10 | 1.94 s (1947, 0.99) | 0.45 s (1007, 0.45) | 2.6 % |
| rosenbrock_D6 | 0.91 s (1150, 0.79) | 0.18 s (523, 0.34) | 2.9 % |
| ackley_D6 | 0.81 s (1122, 0.72) | 0.21 s (636, 0.33) | 3.7 % |
| multisensory_s1_D6_homo | 1.37 s (1450, 0.95) | 0.26 s (751, 0.34) | 3.0 % |
| ellipsoid_D3_homo | 0.61 s (795, 0.76) | 0.10 s (304, 0.34) | 1.6 % |
| sphere_D3_hetero | 0.49 s (647, 0.76) | 0.11 s (300, 0.35) | 2.5 % |

Times summed over seeds 0 to 2. The head's calls are the poll's alone.
The search made 43 to 62 % of the base's calls, and nothing read their
result.

## The rank-1 GP update

**What each side does.** MATLAB BADS adds a point by
`gpupdate(..., 'add', ...)` (`private/gpupdate.m`), which calls
`utils/update_posterior.m`:
- It uses one homoskedastic noise variance, `exp(2 * hyp.lik)`, and
  leaves its `y_sd` argument unused. `gpupdate` skips it under
  `SpecifyTargetNoise`.
- It takes the first hyperparameter sample only. At BADS's default
  (`gpSamples = 0`) there is one.
- It falls back to the full recomputation only when it raises.

gpyreg 1.3.3's single-point path (`GP.update` with one new point and no
`hyp`) works differently:
- It computes the new point's noise from the noise function, the point's
  own variance included, multiplied by the posterior's multiplier
  `sn2_mult`.
- It keeps the existing factor's scale and parametrization.
- It solves with the stored Cholesky factor in the low-noise
  parametrization, where MATLAB uses the explicit inverse.
- It falls back to the full recomputation, with a `UserWarning`, where
  the extension is numerically unsafe.

PyBADS passes the hyperparameters to `gp.update` in `add_and_update_gp`,
which makes gpyreg recompute every posterior in full.

**Synthetic GPs.** 20 GPs per case, D = 3 to 10, N = 20 to 100:
- **Agreement to rounding.** The rank-1 update and the full recomputation
  agree to rounding in both parametrizations:
  - the mean within 2e-14 of the targets' spread;
  - the variance within 5e-14 in absolute terms;
  - alpha within 4e-12 of its largest entry.
- **Where it agrees.** The agreement holds with a variance per point, for
  a new point noisier or less noisy than every existing point, and when
  the new point takes the smallest noise variance below the 1e-6 at which
  gpyreg switches parametrization. There the rank-1 update stays in the
  high-noise form and the recomputation switches.
- **Numerically singular GPs.** On 17 GPs whose K + sn2 I is numerically
  singular, the full factorization needed a multiplier of 10 to 1000:
  - The two differed by up to 2.3e-4 of the spread in the mean.
  - Their variances were rounding noise in both.
  - In one case the recomputation picked a multiplier ten times smaller
    than the one the rank-1 update carried over.

**Benchmark runs.** Every addition of the head's runs, seeds 0 to 2:

| configuration | additions | fallbacks | additions on a GP with a multiplier > 1 | multiplier differs after the addition | largest mean difference / spread (at the new point) | time, rank-1 against full (s, 3 runs) |
|---|---|---|---|---|---|---|
| ellipsoid_D10 | 846 | 81 (9.6 %) | 371 (44 %) | 6 | 9.7e-3 (5.3e-5) | 0.48 against 0.87 |
| rosenbrock_D6 | 523 | 3 (0.6 %) | 17 (3 %) | 1 | 8.1e-4 (1.6e-5) | 0.22 against 0.30 |
| ackley_D6 | 405 | 0 | 0 | 0 | 3.6e-12 (8.6e-13) | 0.16 against 0.21 |
| multisensory_s1_D6_homo | 1331 | 0 | 0 | 0 | 8.5e-12 (4.9e-12) | 0.58 against 1.51 |
| ellipsoid_D3_homo | 668 | 15 (2.2 %) | 266 (40 %) | 4 | 2.1e-1 (1.3e-5) | 0.27 against 0.58 |
| sphere_D3_hetero | 554 | 0 | 0 | 0 | 7.7e-13 (4.5e-13) | 0.18 against 0.37 |

- **Three configurations agree to rounding.** On `ackley_D6`,
  `multisensory_s1_D6_homo` and `sphere_D3_hetero` no factorization
  needed the multiplier, and the two agree to about 1e-11 of the spread.
  `sphere_D3_hetero`'s points each carry the noise variance that the
  target returns.
- **`rosenbrock_D6`.** Its GPs sit at the lower bound of the noise
  (variance 1.4e-7, the low-noise form). There the two differ by up to
  8e-5 of the spread in the two seeds without a multiplier, and by 8e-4
  in the seed with one.
- **The ellipsoids.** On them the multiplier is common, also at level 1,
  where the smallest noise variance is 2e-4 or more.
  - Differences reach 1e-2 of the spread where both posteriors have the
    same multiplier. They are small at the new point itself.
  - The 0.21 of `ellipsoid_D3_homo` is an addition after which the
    rank-1 update kept a multiplier of 10 while the recomputation needed
    none: the two posteriors model a noise ten times apart.
  - The predictive variances differ by up to five times their largest
    value in this regime.
- **The fallbacks.** They are gpyreg's warnings "Rank-one update of
  Cholesky factor unstable" (high-noise form) and "Rank-one update of the
  posterior factor unstable" (low-noise form). No rank-1 update raised,
  and no full recomputation failed.

**Time.** Per addition, the full recomputation takes 0.1 to 0.8 ms at
PyBADS's sizes (up to 200 training points). The rank-1 update, whose
single-point prediction has fixed costs, is slower below about 60 points.
Median of 30 additions:

| D | N | rank-1, ms | full recomputation, ms |
|---|---|---|---|
| 3 | 30 | 0.19 | 0.13 |
| 6 | 60 | 0.19 | 0.16 |
| 10 | 100 | 0.21 | 0.28 |
| 10 | 200 | 0.27 | 0.82 |

In the runs above, the rank-1 update would save 0.05 to 0.93 s per three
runs: 0.3 % (`ackley_D6`, `rosenbrock_D6`) to 2.5 %
(`multisensory_s1_D6_homo`, 200 training points) of PyBADS's own time.

**What adopting it would take.**
- `add_and_update_gp` passes no hyperparameters.
- gpyreg's fallback warnings go to the BADS logger at debug level.
- A `LinAlgError` of the fallback keeps the GP marked for a rebuild, as
  now.
- The target's reuse of the GP's own posterior assumes posteriors
  computed in full. Under the rank-1 update, the target would come from
  the extended posterior.
- In the multiplier regime the posterior after an addition would carry
  the multiplier of the last full factorization, which bears on the
  inflation of the GP noise (W1-25, KD-B6-6 of `pybads/bads/README.md`).
- Since the runs of the ellipsoids change, adopting it needs the
  population comparison.

**Decision.** Not adopted (PI, 2026-09-28): it saves at most 2.5 % of the
own time, and in the multiplier regime it would change the ellipsoids'
runs. `add_and_update_gp` keeps the full recomputation, with a comment
that points here, and `dev/TODO.md` says what would reopen the question.

**A candidate on MATLAB's side.** `gpupdate.m` applies the rank-1 update
with the new value before it replaces a non-finite value by the penalty,
the largest target of the training set. A NaN raises nothing in
`update_posterior.m`, so the posterior would take the NaN while
`gpstruct.y` records the penalty. It cannot happen at MATLAB's defaults:
`private/funlogger.m:103` stops the run on a non-finite value, and
`FitnessShaping`, which could make one, is off.

## Raw data

The scripts, logs and cProfile dumps are machine-local, under
`dev/scripts/runs/perf_target/` (`dev/scripts/runs/LOCAL.md`).
