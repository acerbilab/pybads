# PyBADS: open work

Updated 2026-09-24. The list describes scope, not priority or execution
order.

- [ ] **Tooling, CI, seeded runs and the benchmark harness.** In progress:
  [plan](plans/tooling-and-rng.md), whose Worklog records each phase.
- [ ] **Bug hunt and verification against MATLAB BADS (deferred).** A
  systematic check of the port against the MATLAB reference (`acerbilab/bads`),
  settling the reach and effect of each candidate defect. The starting point
  is the [survey](results/2026-09-23-codebase-survey.md): its candidate
  table (only partly looked at, never compared with MATLAB) and the tests
  that check less than they appear to.
- [ ] **Porting gaps** listed in `pybads/bads/README.md` (periodic
  variables, benchmarking on neurobench).
- [ ] **conda-forge recipes, at the next release.**
  `conda-forge/pybads-feedstock` (`recipe/meta.yaml`): run requirements
  `gpyreg >=1.3.1`, without pytest, pytest-mock and pytest-rerunfailures
  and without the stale cma, corner, dill, imageio and plotly;
  `test.requires` gains pytest and pytest-rerunfailures, since its test
  command `python -m pytest --pyargs pybads --reruns=5 -x -vv` runs the
  tests of the installed package; `python_min` 3.10. It needs
  `conda-forge/gpyreg-feedstock`, at gpyreg 1.0.2 on 2026-09-24, to reach
  gpyreg 1.3.1 first.
- [ ] **For gpyreg's maintainers.** gpyreg lists pytest and
  pytest-rerunfailures among its runtime dependencies (`pyproject.toml`,
  every release from 1.0.4 to 1.3.1), so installing PyBADS still installs
  them.
