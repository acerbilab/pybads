# PyBADS: open work

Updated 2026-09-23. The list describes scope, not priority or execution
order.

- [ ] **Crash with user-specified noise.** With `specify_target_noise=True`,
  a run crashes when a search or poll evaluates a point already in the log,
  because `FunctionLogger._record` then returns an array instead of a scalar;
  `test_he_noisy_sphere_opt` fails in about half of its runs and passes in
  CI only through `--reruns=5`. Mechanism and reproduction:
  [survey](results/2026-09-23-codebase-survey.md), "Observed failures".
- [ ] **Bug hunt and verification against MATLAB BADS (deferred).** A
  systematic check of the port against the MATLAB reference (`acerbilab/bads`),
  settling the reach and effect of each candidate defect. The starting point
  is the [survey](results/2026-09-23-codebase-survey.md): its candidate
  table (only partly looked at, never compared with MATLAB) and the tests
  that check less than they appear to.
- [ ] **Porting gaps** listed in `pybads/bads/README.md` (periodic
  variables, benchmarking on neurobench).
