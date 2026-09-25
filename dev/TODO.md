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
