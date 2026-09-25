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
  it becomes a duplicate training input of the GP. At low noise that makes
  the training covariance nearly singular, a plausible cause of the
  `LinAlgError` crashes behind `plans/gp-update-guards.md`. On Linux at
  `8fc1dff` (gpyreg 1.3.3), one exact repeat was evaluated in
  `ellipsoid_D10` seed 7, one of the crashing seeds, and one in
  `sphere_D3_homo` seed 0; none in `ellipsoid_D3` seed 20. With a target
  noise SD, repeats are common: 227 in seeds 0-89 of `ellipsoid_D3_hetero`
  at `1c8c71d` (Linux), in 58 of the 90 runs, each merged into a row of
  the log (until `032dfcb`, 188 of them into another point's row), and
  the noise-variance fix of `020d6a8` tripled them (seeds 0-19: 17 at
  `685da15`, 53 at `1c8c71d`). Dropping them, as MATLAB does, needs two
  more changes that MATLAB has: the ES search must return an empty set
  when no candidate is left, and the search step must accept one (survey,
  candidate table); without them runs stop with `IndexError` or
  `UnboundLocalError`. With the three changes on `8afbe16`, the median
  error of `ellipsoid_D3_hetero` over 90 seeds falls from 0.46 to 0.33
  (`no_repeats.patch` in
  [experiments/population_ellipsoid_hetero_linux_20260925/](experiments/population_ellipsoid_hetero_linux_20260925/README.md)).
  To settle:
  - look for duplicate rows in `gp.X` just before the failing call of the
    crashing runs (`ellipsoid_D3` seed 20, `ellipsoid_D10` seeds 7, 13 and
    26, Windows, gpyreg 1.3.1);
  - count the repeats over the default suite;
  - fix the removal, as MATLAB does it. That moves results, so it is gated
    by the population comparison against the current reference of the
    platform (`README.md`), and the seeded tests are re-checked over their
    seeds.

  The survey's subsection "Found while fixing the tests" also records the
  defect.
- [ ] **A Windows reference after `97b2c66`.** `8afbe16` (the GP mean prior)
  and `97b2c66` (the bound of the GP length scales) change the runs of
  every configuration, and `032dfcb` those with target noise, so
  `experiments/population_targetnoise_20260925/` no longer stands for the
  current code on Windows. The replacement is the default suite at 30
  seeds on Windows, with its null check, as for the Linux reference
  (`experiments/population_linux_gpfixes_20260925/`).
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
  0.0008), now along the two steep axes. Still open:
  - the evaluated points that `contraints_check` keeps (the item above):
    dropping them, as MATLAB does, lowered the median from 0.46 to 0.33 on
    top of `8afbe16`, not yet measured with the length bound;
  - a run of MATLAB BADS on this problem, which would show whether correct
    noise handling alone gives such runs;
  - the bounds of the GP mean, which the port fixes by the initial design
    and MATLAB leaves infinite (a row of the survey's candidate table).
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
