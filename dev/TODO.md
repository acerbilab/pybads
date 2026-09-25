# PyBADS: open work

Updated 2026-09-25. The list describes scope, not priority or execution
order.

- [ ] **Rank-1 GP update when adding a point.** MATLAB BADS adds a point
  to the GP (`gpupdate(..., 'add', ...)`, `private/gpupdate.m`) by a rank-1
  update of the posterior (`utils/update_posterior.m`), falls back to the
  full recomputation when that fails, and skips the rank-1 update under
  `SpecifyTargetNoise`. The port's `add_and_update_gp` recomputes every
  posterior in full. gpyreg's `update` has a rank-1 path of its own (one
  new point, no new hyperparameters, posteriors that hold their factors),
  which PyBADS does not take, and which accepts a noise variance for the
  new point. The guards of
  [plans/gp-update-guards.md](plans/gp-update-guards.md) keep the full
  recomputation (its Open Question 5) so that runs without a failure do
  not move. Taking the rank-1 path would move results at default options,
  so it needs the population comparison. To settle: whether gpyreg's path
  follows MATLAB's, whether PyBADS should skip it with target noise as
  MATLAB does, and what it saves in time.
- [ ] **Previously evaluated points evaluated again.** `contraints_check`
  (`pybads/function_logger/constraints_check.py`, "Remove previously
  evaluated vectors") keeps the first occurrences of `np.unique` over the
  candidates stacked above the evaluated points. Those always fall among
  the candidates, so a candidate that repeats an evaluated point is kept.
  MATLAB's `utils/uCheck.m` removes such points with `setdiff`. Without a
  target noise SD, `FunctionLogger` records the repeat as a new row, and
  it becomes a duplicate training input of the GP. At low noise a
  duplicate can make the training covariance singular, but duplicates
  explain none of the four `LinAlgError` crashes behind
  `plans/gp-update-guards.md` (the survey's section "Crashes on unguarded
  GP updates"): one failing call held three exact duplicate pairs, but
  removing most sets of three of its rows lets it succeed, and the other
  three held none. On Linux at `8fc1dff`
  (gpyreg 1.3.3), one exact repeat was evaluated in `ellipsoid_D10` seed
  7, and one in `sphere_D3_homo` seed 0; none in `ellipsoid_D3` seed 20.
  To settle:
  - count the repeats over the default suite;
  - fix the removal, as MATLAB does it. That moves results, so it is gated
    by the population comparison against the current reference of the
    platform (`README.md`), and the seeded tests are re-checked over their
    seeds.

  The survey's subsection "Found while fixing the tests" also records the
  defect.
- [ ] **Upper bound of the GP length scales.** `_gp_hyp`
  (`gaussian_process_train.py`) bounds each log length scale by
  `cov_range = min(100, 10 * (ub - lb) / scale)`, where MATLAB's
  `gpdefBads.m` bounds it by `log(covrange)`: 80 against 4.38 on the
  targets of the benchmark with its shifted box. At the failing calls of
  the four `LinAlgError` crashes of the survey's section "Crashes on
  unguarded GP updates", most log length scales exceed 4.38, up to 59.7,
  so that many distinct inputs coincide numerically, and the output scale,
  at its upper bound (MATLAB's too), sets an output variance 2e22 to 2e24
  times the noise variance on them. To settle:
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
    the suite fails under gpyreg 1.3.3;
  - how often the fits of the default suite end with a log length scale
    above `log(cov_range)`;
  - the fix, which can move results at default options wherever a fit
    reaches `log(cov_range)`: gated by the population comparison against the current reference of
    the platform, with the seeded tests re-checked over their seeds.
- [ ] **Small defects of the noise options and the final estimate**, rows
  of the survey's candidate table:
  - without target noise, the final `fsd` divides by `n`, where MATLAB's
    `std` divides by `n - 1`;
  - the final `fval` and `fsd` are recorded in the iteration history at
    the last iteration, not at the iterate they describe;
  - a list `noise_size`, or an array of two elements (MATLAB's base value
    and prior SD), raises in `optimize()`;
  - the high-noise check of `local_gp_fitting` reads `noise_size` under
    `specify_target_noise`, where the warning says that it is ignored, and
    `noise_size=0` makes every refit a high-noise one; MATLAB does the
    same, so this one needs a decision more than a fix;
  - the reported `iterations` is one below MATLAB's count;
  - `output_fcn` is called only at the start, with two arguments where
    MATLAB passes three, and one that stops the run there raises
    `UnboundLocalError`;
  - `max_fun_evals=1` raises `KeyError: 'eff_starting_points'`.

  A fix that a user can notice gets a changelog entry; the fingerprint of
  `dev/scripts/fingerprint.py` shows whether a fix moves results at
  default options.
- [ ] **A Linux reference after `020d6a8`.** `020d6a8` changes the runs of
  `sphere_D3_hetero` and `ellipsoid_D3_hetero`, the two configurations with
  target noise, so `experiments/population_linux_20260925/` no longer
  stands for the current code in those two. The replacement is the default
  suite at 30 seeds on Linux, with its null check, as for the Windows
  reference.
- [ ] **`ellipsoid_D3_hetero` after `020d6a8`.** Squaring the target's noise
  standard deviations, as MATLAB does, makes the runs of this benchmark
  configuration worse: over 90 seeds the median error rises from 0.21 to
  0.54, mostly along the flat axis of the ellipsoid
  ([experiments/population_ellipsoid_hetero_20260925/](experiments/population_ellipsoid_hetero_20260925/README.md)),
  while the spheres with target noise improve. The fix stays; what is open
  is which other difference from MATLAB the correct noise exposes. The
  candidates, each a row of the survey's candidate table:
  - the GP mean prior, which MATLAB re-centres at every rebuild
    (`gpdefBads.m`) and the port never updates;
  - the merged value with the raw standard deviation that a repeated point
    adds to the GP;
  - the lower bound of the noise hyperparameter, which the port raises
    after a failed fit and MATLAB does not.

  A run of MATLAB BADS on this problem would show whether correct noise
  handling alone gives such runs.
- [ ] **conda-forge recipe.** The test command of `conda-forge/pybads-feedstock`
  (`recipe/meta.yaml`) passes `--reruns=5` and requires
  pytest-rerunfailures. The tests of 1.1.0, which it runs, are not all
  seeded, so both stay until the first release after 1.1.0, whose tests
  are: drop them in the version-update PR that the feedstock's bot opens
  for that release, before it is merged.
- [ ] **Follow-ups of the GP-update guards**
  ([plans/gp-update-guards.md](plans/gp-update-guards.md)). Each has a row
  in the survey's candidate table, marked "at `676083d`" or "at
  `a83bd51`":
  - the target's posterior, recomputed under the best iteration's
    hyperparameters, where MATLAB reuses the current posterior. That gives
    other targets at default options, and it is why that call can fail;
  - the refit forced after a failed rebuild, which ignores
    `min_refit_time`, where MATLAB refits through `gppredcheck`. It sends
    such runs into `_robust_gp_fit_`, whose fifth consecutive failed fit
    raises `ValueError`, a combination no test or stress run covers;
  - after a failed rebuild, the search still ranks its candidates by the
    previous GP, where MATLAB takes the first candidate;
  - `init_and_train_gp` retries a failing initial fit without bound;
  - `_re_evaluate_history_` rebuilds the GPs stored in `IterationHistory`
    in place;
  - under `stobads`, a NaN estimate counts as uncertain, not as a failure.

  No failure of the guarded calls occurs in the default suite under gpyreg
  1.3.3 (484,773 calls on Linux), so only the tests
  (`test_gp_update_failures.py`) and the stress run of
  `dev/scripts/gp_update_failures.py --inject` reach these paths.
- [ ] **Bug hunt and verification against MATLAB BADS.** A systematic check of the port against the MATLAB reference (`acerbilab/bads`),
  settling the reach and effect of each candidate defect. The starting point
  is the [survey](results/2026-09-23-codebase-survey.md): its candidate
  table (only partly looked at, never compared with MATLAB), and a finding
  of its section on the tests: the seed of the initial Sobol design, which
  ignores all but the integer part of `u0` (whether MATLAB's `uint64`
  product saturates needs MATLAB itself). The previously evaluated points
  that `contraints_check` keeps have an item of their own above.
  PyVBMC's MATLAB-comparison helpers (`pyvbmc/testing/_compare_matlab.py`:
  `randn2` and the draws that reproduce MATLAB's random stream) come with
  it, for the comparisons that need MATLAB's own numbers.
- [ ] **Exact step-by-step replay and numerical oracles**, after PyVBMC's
  (`dev/scripts/golden_replay.py`, `pyvbmc/testing/oracles/`), after the
  bug hunt, so that they do not pin today's defects; the random draws go
  through one generator per run (`bads.rng`), which replay needs. The
  population comparison of `dev/scripts/population.py` checks
  distributions, not trajectories, until then.
- [ ] **Profiler**, after PyVBMC's (`dev/scripts/profile_run.py` and kin),
  once PyBADS times its search, poll and GP-training stages separately:
  today its timer covers only the whole run and the target's evaluations.
- [ ] **Porting gaps** listed in `pybads/bads/README.md` (periodic
  variables, benchmarking on neurobench).
- [ ] **Coding-agent skill**, after PyVBMC's (`skills/pyvbmc/SKILL.md`): a
  `skills/pybads/SKILL.md` that points a coding agent to the parts of the
  documentation relevant to its task, linked from the README.
- [ ] **gpyreg releases after 1.3.3.** PyBADS's minimum gpyreg
  (`pyproject.toml`) and its CI pin (`GPYREG_PIN`) name one release, 1.3.3
  as of 2026-09-25 ([assessment](results/2026-09-25-gpyreg-1.3.3.md)).
  Each new release moves both, after the population comparison
  (`dev/scripts/population.py compare`) against the current reference
  shows that it has no effect on PyBADS, or explains the one it has.
- [ ] **For gpyreg's maintainers.** gpyreg lists pytest and
  pytest-rerunfailures among its runtime dependencies (`pyproject.toml`,
  every release from 1.0.4 to 1.3.3), so installing PyBADS still installs
  them.
