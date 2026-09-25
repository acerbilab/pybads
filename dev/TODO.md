# PyBADS: open work

Updated 2026-09-25. The list describes scope, not priority or execution
order.

- [ ] **Investigate the crashes on unguarded GP updates.** In progress in
  a separate session since 2026-09-25. In the first benchmark reference, `dev/experiments/population_baseline_20260924/`
  (default suite at 500 D, gpyreg 1.3.1, draws through NumPy's global
  stream), 2 of 540 runs stopped with `LinAlgError:
  Singular matrix for L Cholesky decomposition`, raised by gpyreg's
  training Cholesky factorization on a GP update that PyBADS does not
  guard (`_robust_gp_fit_` retries only its own fits):
  - `ellipsoid_D3`, seed 20, at 150 evaluations: `_poll_step_` →
    `_get_target_from_gp_` → `gp.set_hyperparameters`;
  - `ellipsoid_D10`, seed 7, at 951 evaluations: `_search_step_` →
    `add_and_update_gp` → `gp.update`.

  Both reproduce from their seeds at `2226883`
  (`PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.1 python dev/scripts/population.py run --only ellipsoid_D3 --seeds 20 --out <dir>`).
  In `dev/experiments/population_generator_20260924/` (the same suite with
  the draws through a generator, gpyreg 1.3.1), those seeds finish, and 2
  other runs stop at a third unguarded call: `ellipsoid_D10` seeds 13 and
  26, where `local_gp_fitting` catches the failure of
  `gp.update(hyp=hyp_gp)` and its recovery,
  `gp.set_hyperparameters(old_hyp_gp)`, fails in turn (reached from
  `_search_step_` and from `_poll_step_`; both reproduce from their seeds
  with the package code of `c85cddb`). The recovery restores the old
  hyperparameters on the new training set, which `local_gp_fitting`
  assigns to `gp.X` and `gp.y` directly before the update; its MATLAB
  counterpart is not checked.
  The first two are gaps of the port: MATLAB BADS (`../bads` at `019f0b4`)
  guards both calls. `UpdateTarget` (`bads.m`) predicts through `gppred`
  (`utils/gppred.m`), which catches a failed prediction per hyperparameter
  sample, and then falls back to the incumbent's `fval` and `fsd` when the
  prediction is not finite; the port kept that fallback but not the
  `try`, and gpyreg raises where MATLAB's prediction returned NaN.
  `gpupdate(..., 'add', ...)` (`private/gpupdate.m`) tries a rank-1
  posterior update, then a full recomputation, each in a `try`, and on
  failure clears the posterior with `exitflag = -2`; the port calls
  `gp.update(compute_posterior=True)` bare. From gpyreg 1.3.3, a `fit`,
  `update` or `set_hyperparameters` that raises restores the GP's state as
  it was when the call started (data, bounds, priors and posteriors). For
  `_get_target_from_gp_`, which changes no data, a guard therefore keeps a
  consistent GP. `add_and_update_gp` and `local_gp_fitting` assign the new
  data to `gp.X` and `gp.y` before they call `gp.update`, so a failed
  update leaves the new data beside the old posteriors; passing the data
  through `gp.update` (its `X_new` and `y_new` arguments) would let a
  failure restore the GP without them. With gpyreg 1.3.3 no run of the
  suite crashes (`dev/experiments/population_gpyreg133_20260924/`, the
  current reference), but the crashing runs pass through the low-noise
  regime whose predictions gpyreg 1.3.2 changed, so they follow other
  trajectories there, and this does not show whether the calls still fail.
  To settle: restore those guards, the data passed through `gp.update`,
  and what the GP left by a failed call means downstream in PyBADS.
- [ ] **Make the tests check what they appear to.** The survey's section
  "Tests that check less than they appear to" lists the defects:
  `pybads/testing/bads/poll/test_poll_mads.py` names its functions
  `*_test`, so pytest collects none of them (they pass when called
  directly); `test_sphere_opt` has its non-box constraint reversed with
  respect to MATLAB's `runtest.m` and passes only through its loose
  tolerance; `test_high_dim_opt` asserts nothing; the other optimization
  tests pass whenever the error is below 1, and most are unseeded
  (`random_seed` makes them deterministic); `pybads/testing/run_tests.py`
  imports paths that no longer exist, and no test reads
  `pybads/testing/bads/*.dat`. A fix to a test moves no result, so the
  suite is its check, not the population comparison; `AGENTS.md`, "Tests
  and their traps", and the survey's section record each fix. Work on a
  branch off `dev-next` (`dev-tests`, whose pushes get the CI smoke run),
  merged into `dev-next` by a pull request, and keep off the files of the
  item on unguarded GP updates, worked on in parallel:
  `bads/gaussian_process_train.py`, the GP call sites in `bads/bads.py`
  and the survey's candidate table.
- [ ] **Bug hunt and verification against MATLAB BADS.** A systematic check of the port against the MATLAB reference (`acerbilab/bads`),
  settling the reach and effect of each candidate defect. The starting point
  is the [survey](results/2026-09-23-codebase-survey.md): its candidate
  table (only partly looked at, never compared with MATLAB) and the tests
  that check less than they appear to.
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
