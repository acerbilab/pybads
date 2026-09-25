# PyBADS: open work

Updated 2026-09-24. The list describes scope, not priority or execution
order.

- [ ] **Tooling, CI, seeded runs and the benchmark harness.** In progress:
  [plan](plans/tooling-and-rng.md), whose Worklog records each phase.
- [ ] **Investigate the crashes on unguarded GP updates.** In the benchmark
  reference `dev/experiments/population_baseline_20260924/` (default suite
  at 500 D, gpyreg 1.3.1), 2 of 540 runs stopped with `LinAlgError:
  Singular matrix for L Cholesky decomposition`, raised by gpyreg's
  training Cholesky factorization on a GP update that PyBADS does not
  guard (`_robust_gp_fit_` retries only its own fits):
  - `ellipsoid_D3`, seed 20, at 150 evaluations: `_poll_step_` →
    `_get_target_from_gp_` → `gp.set_hyperparameters`;
  - `ellipsoid_D10`, seed 7, at 951 evaluations: `_search_step_` →
    `add_and_update_gp` → `gp.update`.

  Both reproduce from their seeds at `2226883`
  (`PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.1 .venv/Scripts/python.exe dev/scripts/population.py run --only ellipsoid_D3 --seeds 20 --out <dir>`).
  In the reference that replaced it,
  `dev/experiments/population_generator_20260924/` (draws through a
  generator), those seeds finish, and 2 other runs stop at a third
  unguarded call: `ellipsoid_D10` seeds 13 and 26, where
  `local_gp_fitting` catches the failure of `gp.update(hyp=hyp_gp)` and
  its recovery, `gp.set_hyperparameters(old_hyp_gp)`, fails in turn
  (reached from `_search_step_` and from `_poll_step_`; they reproduce the
  same way at `c85cddb`). The recovery restores the old hyperparameters on
  the new training set, which `local_gp_fitting` assigns to `gp.X` and
  `gp.y` directly before the update; its MATLAB counterpart is not
  checked.
  Both are gaps of the port: MATLAB BADS (`../bads` at `019f0b4`) guards
  both calls. `UpdateTarget` (`bads.m`) predicts through `gppred`
  (`utils/gppred.m`), which catches a failed prediction per hyperparameter
  sample, and then falls back to the incumbent's `fval` and `fsd` when the
  prediction is not finite; the port kept that fallback but not the
  `try`, and gpyreg raises where MATLAB's prediction returned NaN.
  `gpupdate(..., 'add', ...)` (`private/gpupdate.m`) tries a rank-1
  posterior update, then a full recomputation, each in a `try`, and on
  failure clears the posterior with `exitflag = -2`; the port calls
  `gp.update(compute_posterior=True)` bare. From gpyreg 1.3.3, a `fit`,
  `update` or `set_hyperparameters` that raises leaves the GP as it was
  before the call (data, bounds, priors and posteriors), so a guard around
  them keeps a consistent GP without the new point, an alternative to
  MATLAB's cleared posterior. With gpyreg 1.3.3 no run of the suite
  crashes (`dev/experiments/population_gpyreg133_20260924/`), but the
  crashing runs pass through the low-noise regime whose predictions
  gpyreg 1.3.2 changed, so they follow other trajectories there and the
  calls are no safer. To settle: restore those guards, and what the GP
  left by a failed call means downstream in PyBADS.
- [ ] **Bug hunt and verification against MATLAB BADS (deferred).** A
  systematic check of the port against the MATLAB reference (`acerbilab/bads`),
  settling the reach and effect of each candidate defect. The starting point
  is the [survey](results/2026-09-23-codebase-survey.md): its candidate
  table (only partly looked at, never compared with MATLAB) and the tests
  that check less than they appear to.
  PyVBMC's MATLAB-comparison helpers (`pyvbmc/testing/_compare_matlab.py`:
  `randn2` and the draws that reproduce MATLAB's random stream) come with
  it, for the comparisons that need MATLAB's own numbers.
- [ ] **Exact step-by-step replay and numerical oracles**, after PyVBMC's
  (`dev/scripts/golden_replay.py`, `pyvbmc/testing/oracles/`), once the
  random draws go through a generator (tooling plan, Phase 8) and after the
  bug hunt, so that they do not pin today's defects. The population
  comparison of `dev/scripts/population.py` checks distributions, not
  trajectories, until then.
- [ ] **Profiler**, after PyVBMC's (`dev/scripts/profile_run.py` and kin),
  once PyBADS times its search, poll and GP-training stages separately:
  today its timer covers only the whole run and the target's evaluations.
- [ ] **Porting gaps** listed in `pybads/bads/README.md` (periodic
  variables, benchmarking on neurobench).
- [ ] **gpyreg releases after 1.3.3.** PyBADS's minimum gpyreg
  (`pyproject.toml`) and its CI pin (`GPYREG_PIN`) name one release, 1.3.3
  as of 2026-09-25 ([assessment](results/2026-09-25-gpyreg-1.3.3.md)).
  Each new release moves both, after the population comparison
  (`dev/scripts/population.py compare`) against the current reference
  shows that it has no effect on PyBADS, or explains the one it has.
- [ ] **conda-forge recipes, at the next release.**
  `conda-forge/pybads-feedstock` (`recipe/meta.yaml`): run requirements
  `gpyreg >=1.3.3`, without pytest, pytest-mock and pytest-rerunfailures
  and without the stale cma, corner, dill, imageio and plotly;
  `test.requires` gains pytest and pytest-rerunfailures, since its test
  command `python -m pytest --pyargs pybads --reruns=5 -x -vv` runs the
  tests of the installed package; `python_min` 3.10. It needs
  `conda-forge/gpyreg-feedstock`, at gpyreg 1.0.2 on 2026-09-24, to reach
  gpyreg 1.3.3 first.
- [ ] **For gpyreg's maintainers.** gpyreg lists pytest and
  pytest-rerunfailures among its runtime dependencies (`pyproject.toml`,
  every release from 1.0.4 to 1.3.3), so installing PyBADS still installs
  them.
