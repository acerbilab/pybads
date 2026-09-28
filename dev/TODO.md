# PyBADS: open work

Updated 2026-09-28. The list describes scope, not priority or execution
order.

- [ ] **Rank-1 GP update when adding a point: not adopted, to revisit if
  its terms change.** MATLAB BADS adds a point to the GP by a rank-1
  update of the posterior (`private/gpupdate.m`, `utils/update_posterior.m`);
  PyBADS's `add_and_update_gp` recomputes every posterior in full, and says
  why at its call of `gp.update`. The measurement
  ([results/2026-09-28-where-pybads-spends-its-time.md](results/2026-09-28-where-pybads-spends-its-time.md))
  settled the questions: gpyreg's rank-1 path agrees with the full
  recomputation to rounding, target noise included (so there is no reason
  to skip it under `specify_target_noise`, as MATLAB does); it would save
  at most 2.5 % of PyBADS's own time, and is slower below about 60
  training points; but on the ellipsoids 40 to 44 % of the additions meet a
  GP whose factorization needed gpyreg's noise multiplier, where it
  differs from the recomputation by up to 1e-2 of the targets' spread, and
  by 0.21 when the multiplier it carries over differs from the one the
  recomputation picks. The PI did not adopt it (2026-09-28). Revisit if the
  training sets grow well beyond 200 points, if gpyreg's handling of the
  multiplier changes (KD-B6-6 of `pybads/bads/README.md`, kept as it is
  by the PI on 2026-09-28), or if a profile shows the update after
  a new point taking a larger share. Adopting it moves the ellipsoids'
  runs, so it needs the population comparison, and the target's reuse of
  the GP's own posterior (`_get_target_from_gp_`), which relies on
  posteriors computed in full, needs revisiting with it.
- [ ] **A faster kernel gradient in gpyreg.** gpyreg's rational-quadratic
  ARD kernel (`RationalQuadraticARD.compute`) takes 31 to 45 % of a
  PyBADS run in its own code, in the predictions at the ES search's
  candidates and in the hyperparameter fits
  ([results/2026-09-28-where-pybads-spends-its-time.md](results/2026-09-28-where-pybads-spends-its-time.md)).
  Its gradient recomputes `sf2 * M ** (-alpha - 1)` for each input
  dimension; computed once before the loop, it gives bit-identical results
  and halves the time of the kernel with its gradient at D = 6 and D = 10,
  about a quarter of the fits' time on `ellipsoid_D10`. A change for
  gpyreg; reaching PyBADS, it is a gpyreg release, which moves PyBADS's
  minimum and CI pin after its gate.
- [ ] **The old `LinAlgError` crashes and the bound of the GP length
  scales.** `_gp_hyp` bounded each log length scale by `cov_range = min(100,
  10 * (ub - lb) / scale)`, where MATLAB's `gpdefBads.m` bounds it by
  `log(covrange)`: 80 against 4.38 on the targets of the benchmark with its
  shifted box; `97b2c66` takes MATLAB's bound. At the failing calls of the
  four `LinAlgError` crashes of the survey's section "Crashes on unguarded
  GP updates", most log length scales exceed 4.38, up to 59.7, so that many
  distinct inputs coincide numerically, and the output scale, at its upper
  bound (MATLAB's too), sets an output variance 2e22 to 2e24 times the
  noise variance on them. Open:
  - whether MATLAB's bound would have kept those GPs factorizable: the
    GP of each failing call refitted under it. The inputs, targets and
    hyperparameters saved at those calls (machine-local,
    `dev/scripts/runs/LOCAL.md`) lack the priors and bounds that
    `_gp_hyp` sets; a capture that keeps a deep copy of the GP before
    each call of `gpyreg.GP.update` (a wrapper loaded through a
    `sitecustomize.py` first on `PYTHONPATH`) has them all. A rerun with
    the bound changed follows another trajectory and cannot show it. A
    rerun of the crashes needs Windows with the environment of their
    records (Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1, one BLAS thread),
    their commits `2226883` and `c85cddb` (reachable from
    `refs/pull/59/head`) and a clone of gpyreg at v1.3.1, since no run of
    the suite fails under gpyreg 1.3.3.
- [ ] **`ellipsoid_D3_hetero` after `020d6a8`.** Squaring the target's noise
  standard deviations, as MATLAB does, made the runs of this benchmark
  configuration worse: over 90 seeds the median error rose from 0.21 to
  0.54 on Windows and from 0.18 to 0.58 on Linux, mostly along the flat
  axis of the ellipsoid
  ([Windows](experiments/population_ellipsoid_hetero_20260925/README.md),
  [Linux](experiments/population_ellipsoid_hetero_linux_20260925/README.md)),
  while the spheres with target noise improved. The fix stays. Three
  differences from MATLAB, fixed on Linux, bring the median to 0.25 and
  the flat axis back to its error before `020d6a8`: the bound of the GP log
  length scales (`97b2c66`, the largest effect), a repeated point merged
  into another point's row of the function log (`032dfcb`), and the GP
  mean prior, re-centred at each rebuild (`8afbe16`). Returning the
  observation of a repeated point, as MATLAB's `funlogger` does, makes the
  runs worse, and the lower bound of the noise hyperparameter never moves,
  since no fit fails. The runs remain worse than before `020d6a8` (p =
  0.0008), now along the two steep axes. The port review fixed two more
  differences:
  - the evaluated points that `contraints_check` kept: W3-1 (`149d528`,
    wave 3) removes them, as MATLAB does; over seeds 0-29 of this
    configuration its 100 repeats (of 8800 evaluations, in 17 runs) are
    gone, and the median error moved from 0.43 to 0.36, unflagged
    (`experiments/port_review_20260925/verification/wave3_fixpass/`);
  - the bounds of the GP mean, which the port set from the initial design
    and MATLAB leaves infinite; once `8afbe16` re-centred the prior of the
    mean, it could fall outside them, which made the log prior NaN in the
    fits of 67 runs of the Windows population at `ab4dded`
    ([experiments/population_gpfixes_20260925/](experiments/population_gpfixes_20260925/README.md)).
    W1-23 (`172df00`, wave 1) leaves them infinite, as MATLAB does.

  Still open: a run of MATLAB BADS on this problem, which would show
  whether correct noise handling alone gives such runs. Measured on
  2026-09-28: its GP runs at an output variance near 1e15 times its noise
  variance, and most of its fits keep a noise that gpyreg multiplied
  (KD-B6-6); with gpyreg's switch to MATLAB's rule its errors are not
  measurably smaller, nor with any of the four Sto-BADS arms measured
  ([results/2026-09-28-gp-health.md](results/2026-09-28-gp-health.md),
  [results/2026-09-28-stobads-rule.md](results/2026-09-28-stobads-rule.md)).
- [ ] **A test whose outcome varies on macOS arm64.**
  `test_run_control.py::test_output_fcn_that_changes_nothing_leaves_the_run_unchanged`
  runs the same seeded optimization twice in one process, the second with
  an output function that alters only its copy of `optim_state`, and
  requires the same result. In the CI of #90 it failed on `macos-latest`
  with Python 3.10 (runner image `macos-26-arm64`, NumPy 2.2.6, SciPy
  1.15.3, gpyreg 1.3.3): the two runs ended at different points. It passed
  on the job's re-run, in the same job of #88 at the same package code and
  versions, and on Linux and Windows. So on that platform something in a
  run, or in a library it calls, does not repeat between two runs in one
  process; no decision of `bads.py` reads the wall clock. A hypothesis, not
  yet tested: results of Accelerate or NumPy that depend on the alignment
  of the arrays, which the output function's deep copies of `optim_state`
  shift. To settle on macOS arm64: run the test in a loop until it fails,
  find the first computation at which the two runs differ, and fix it
  there; if it lies in a library, the test is rewritten to compare only
  what the platform repeats. `dev/scripts/replay.py` is the instrument
  for the second step: `record --repeat N` records runs repeated in one
  process, and `check DIR` reports the first evaluation, step and GP
  computation at which a repeat parts from the first run.
- [ ] **conda-forge recipe.** The test command of `conda-forge/pybads-feedstock`
  (`recipe/meta.yaml`) passes `--reruns=5` and requires
  pytest-rerunfailures. The tests of 1.1.0, which it runs, are not all
  seeded, so both stay until the first release after 1.1.0, whose tests
  are: drop them in the version-update PR that the feedstock's bot opens
  for that release, before it is merged.
- [ ] **"What's new" at the next release.** `README.md` and
  `docsrc/source/index.rst` list under "What's new in PyBADS 1.1" that
  every random draw of a run comes from one generator created from
  `random_seed`, which 1.1.0's initial design did not follow (its
  scrambling was seeded from the start). The list of the next release
  replaces it, and says that `random_seed` now decides the initial design
  (the doublecheck of wave 4 of the port review,
  `experiments/port_review_20260925/verification/wave4.md`,
  "Doublecheck"). At the same release, `skills/pybads/SKILL.md`, which
  names no release, names it, as PyVBMC's names 1.5.
- [ ] **Zero predictive SDs: how often MATLAB gives them.** The predictive
  SD of the GP is exactly 0 at about a tenth of the poll's acquisitions
  over the four suites, up to 40% on some configurations, noisy ones
  included, and each makes the poll's GP unreliable (W3-28). Counted and
  traced on 2026-09-28
  ([results/2026-09-28-gp-health.md](results/2026-09-28-gp-health.md)):
  rounding, the latent variance `kss - v'v` cancelled below the rounding
  of `kss` near the training inputs and clamped at 0, where the output
  variance exceeds the noise by 1e12 or more, the same cause as KD-B6-6;
  MATLAB's `mygp.m:187` clamps the same way. Open only: how often MATLAB's
  own fits reach them, which needs MATLAB.
- [ ] **Numerical oracles**, after PyVBMC's (`pyvbmc/testing/oracles/`).
  PyVBMC's oracles are snapshots of its own numerics, not MATLAB's: each
  fixture is an algorithm state saved as arrays, with the outputs of each
  stage computed from it, compared under tolerances measured across BLAS
  settings, and regenerated one reference at a time on an intended
  change; its MATLAB-comparison helpers
  (`pyvbmc/testing/_compare_matlab.py`: `randn2` and the draws that
  reproduce MATLAB's random stream) have no caller. Snapshots of PyBADS's
  stages (the variable transform, the poll's basis, the LCB, the ES
  search's candidates, the GP's fit and predictions) would gate a change
  that must move nothing on every platform, where
  `dev/scripts/replay.py` compares runs step by step on one machine only
  and `pybads/testing/bads/test_initial_design_pin.py` pins the initial
  design. Oracles computed by MATLAB BADS, which would check the port's
  numbers rather than their stability, need MATLAB and the BADS toolbox
  to generate.
- [ ] **Porting gaps** listed in `pybads/bads/README.md` (periodic
  variables, benchmarking on neurobench). A port of periodic variables also
  assigns `period_check`'s result at every call site, as MATLAB BADS does,
  where the poll discards it, and gives the initial design `optim_state`'s
  boolean mask of the periodic variables, where it passes the option, a
  list of indices (rows W3-35 and W4-11 of the port review). The port
  rewrites the periodic branches of `udist`
  (`pybads/search/grid_functions.py`) and `ucov`
  (`pybads/search/es_search.py`) from MATLAB's `utils/udist.m` and
  `utils/ucov.m`: `udist`'s takes the row indices that `np.nonzero` gives
  of the `(1, D)` mask for the variables', indexes the rows of its matrix
  of summed squared distances by them and divides by the length scales
  after summing, and `ucov`'s wraps the points without the shift it
  computes (KD-B1-6).
- [ ] **`gp_cov_prior="ard"`.** MATLAB's per-dimension empirical prior of
  the GP length scales (`gpdef/gpdefBads.m:254-274`) is not ported; by the
  ruling on row W1-28 of the port review
  (`experiments/port_review_20260925/verification/wave1.md`), PyBADS
  refuses the value with a message instead. A port needs its own population
  comparison with the option set.
- [ ] **Prior evaluations (`fun_values`).** MATLAB BADS imports
  evaluations made before the run into its log and its GP
  (`private/setupvars.m:126-167`, `private/funlogger.m`) and takes its
  first incumbent from `x0` and the initial design only
  (`private/evalinitmesh.m:120-123`). PyBADS's `fun_values` never worked,
  and by the ruling on row W2-6 of the port review
  (`experiments/port_review_20260925/verification/wave2.md`) a non-empty
  value is refused with a message. A port imports them after the function
  logger exists, keeps them out of the choice of the first incumbent, and
  needs a test that its GP holds them. `FunctionLogger.add`, which such a
  port would call, keeps checks of its own on the value and its SD, records
  a missing SD as 1 when the logger holds SDs and drops a given one when it
  does not, and what it records for a repeated point is settled with it
  (row W4-10); so is the bookkeeping of the final samples, which still add
  to the incumbent's `n_evals` in the log and average their times into its
  row, after the run's last decision (PI, 2026-09-28).
- [ ] **The example notebooks' saved outputs.** Nothing runs the notebooks
  of `examples/`, and the saved outputs of all five predate the port
  review, whose fix passes change their numbers, and some of their
  messages: `pybads_example_2_nonbox_constraints.ipynb` shows the warning
  `bads:TooCloseBounds`, which W2-4 removed;
  `pybads_example_4_user_provided_noise.ipynb` a termination message on
  `tol_mesh` that speaks of the change in the function value, which wave 0
  corrected; `pybads_example_5_extended_usage.ipynb` a result with
  `'fsd': 0` and without `status` (W2-12, W2-13), from version
  `0.8.3.dev21`. The rerun of all five waited for the review's fix passes,
  so that the outputs would not be regenerated at every pass
  (`experiments/port_review_20260925/verification/wave2.md`, "Fix pass");
  the passes have all landed (the review closed on 2026-09-28), so the
  rerun goes with the headless run of the examples before the release.
- [ ] **gpyreg releases after 1.3.3.** PyBADS's minimum gpyreg
  (`pyproject.toml`) and its CI pin (`GPYREG_PIN`) name one release, 1.3.3
  as of 2026-09-25 ([assessment](results/2026-09-25-gpyreg-1.3.3.md)).
  Each new release moves both, after the population comparison
  (`dev/scripts/population.py compare`) against the current reference
  shows that it has no effect on PyBADS, or explains the one it has.
  gpyreg's `main` holds, unreleased, the fix of the port review's W1-24
  (the log prior of a prior far outside its bounds, acerbilab/gpyreg#57)
  and W1-25's switch (acerbilab/gpyreg#56), which stays off in PyBADS
  (KD-B6-6). At `1893eff`, with the switch off, `main` gives gpyreg 1.3.3's
  records in all 1,080 runs of the `default`, `geometry`, `oned` and
  `bounds` suites on Linux
  ([experiments/gp_switch_linux_20260928/](experiments/gp_switch_linux_20260928/README.md)):
  a release of it moves nothing there; its gate also takes the Windows
  comparison.
- [ ] **For gpyreg's maintainers.** gpyreg lists pytest and
  pytest-rerunfailures among its runtime dependencies (`pyproject.toml`,
  every release from 1.0.4 to 1.3.3), so installing PyBADS still installs
  them. Its hyperparameter helpers, the `get_bounds_info` of its kernels,
  means and noise, which `fit` calls even where the caller sets every
  bound and prior (`gaussian_process.py:1762-1764`, and `555-557` through
  the recommended bounds), are degenerate on inputs or targets without
  spread (1.3.3). The kernels' helper takes the log of each column's
  width and of its SD with `ddof=1` (`covariance_functions.py:476-480`),
  which prints `RuntimeWarning`s (a log of zero; on one point also NumPy's
  "Degrees of freedom <= 0" and an invalid division) on a column without
  spread and on one point; the three replace a single target by `[0, 1]`
  (`covariance_functions.py:472`, `mean_functions.py:491`,
  `noise_functions.py:129`), which centres the constant mean's
  recommendation at 0.5 whatever the target. PyBADS gives the GP on one
  point MATLAB BADS's values without a fit (KD-B6-5), but its refits reach
  the helpers: on inputs that a poll along one axis leaves without spread
  in a coordinate (every run of `sphere_band_D3` at `73d517a`, whose first
  refit takes `x0` and two poll points along the third axis, and 3 of 30
  of `sphere_band_D2_hetero`), and on one point at D = 1, where the noise
  test brings `func_count` to 2 > D (27 of 30 runs of a noisy band that
  leaves only `x0` feasible)
  ([experiments/one_point_gp_linux_20260928/](experiments/one_point_gp_linux_20260928/README.md)).
  The helpers could centre on the one target for N <= 1 and keep the upper
  bound of -inf of a column without spread, on which the recommended
  bounds' refusal of such a column relies (`gaussian_process.py:586-620`).
