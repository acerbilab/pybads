# Task: periodic variables (`periodic_vars`) in PyBADS 1.5

Port MATLAB BADS's `PeriodicVars` (KD-B1-6). The GP kernel goes into gpyreg;
PyBADS's minimum gpyreg moves to the release that carries it (coordinated
by the PI).

PI decisions (2026-09-28):
- Consistent units: a periodic length scale is in u-units (the kernel
  matches the ordinary one at short range); no MATLAB prior shift (`gpdefBads.m:278-284`);
  `len_scale`/`poll_scale` unchanged. Catalogued as a deliberate difference.
- gpyreg: fixed `periods` on all ARD kernels (SE, Matérn, RQ-ARD).
- A sixth example notebook, after MATLAB's `bads_examples.m` Example 5.

Branches: pybads `claude/todo-discussion-q8c9im` (base `dev-next` 7c01d77e);
gpyreg `claude/todo-discussion-q8c9im` (base `main` 280d8c0).
Envs: `/home/user/pybads/.venv`; gpyreg clones `/home/user/gpyreg` (work),
`/home/user/gpyreg-v1.3.3` (reference); MATLAB BADS `/home/user/acerbilab/bads`.

## A. gpyreg kernel (Opus sub-agent, dispatched)
- [~] A1 `periods=None` on `SquaredExponential`, `Matern`, `RationalQuadraticARD`: per periodic dim `(p/π)² sin²(πΔ/p)` replaces `Δ²`; `inf` = not periodic; all-inf ≡ None
- [~] A2 `periods=None` path bit-identical (separate branch, no refactor of the old path)
- [~] A3 Isotropic kernels refuse `periods`; `GP.quad` refuses a periodic kernel
- [~] A4 Tests: periodicity, p→∞ limit, gradients vs numdifftools, validation, GP fit/predict, refusals
- [~] A5 Docstrings, [release notes](../gpyreg/docsrc/source/release_notes.rst), gpyreg AGENTS.md
- [~] A6 gpyreg suite + pre-commit clean; commit; push

## B. PyBADS core
- [x] B1 [`BADS.__init__`](pybads/bads/bads.py): validate `periodic_vars` (0-based ints, unique, in range; bools refused) instead of refusing it
- [x] B2 [`period_check`](pybads/utils/period_check.py): wrap into `[lb, ub)`
- [x] B3 Call sites: initial design (mask, W4-11), search, poll (assign, W3-35), ES loop ([es_search.py](pybads/search/es_search.py)); wrap again after `force_to_grid`
- [x] B4 [`udist`](pybads/search/grid_functions.py): per-coordinate shortest way round
- [x] B5 [`ucov`](pybads/search/es_search.py): fix the periodic shift
- [x] B6 [GP](pybads/bads/gaussian_process_train.py): kernel `periods = (ub - lb)/scale` in u-space, only when periodic vars exist
- [~] B7 Tests: unit (period_check, udist, ucov, validation, GP periods) + end-to-end runs
- [ ] B8 Fingerprint `4146a986863602cb` unchanged with the gpyreg branch; full suite; pre-commit

## C. Evidence
- [x] C1 Periodic problems + `periodic` suite in [benchmark_targets.py](dev/scripts/benchmark_targets.py); [dev/README.md](dev/README.md) entry
- [ ] C2 Population, periodic on vs off (`--options '{"periodic_vars": null}'`), 30 seeds, Linux
- [ ] C3 Results note in `dev/results/` + records in `dev/experiments/`

## D. Documentation (Opus sub-agent after B)
- [ ] D1 Option description ([advanced .ini](pybads/bads/option_configs/advanced_bads_options.ini)), `BADS` docstring
- [ ] D2 [FAQ](docsrc/source/faq.md): periodic answer, MATLAB-differences bullet
- [ ] D3 Example 6 notebook, generated script, docs toctree
- [x] D4 [CHANGELOG](CHANGELOG.md) (Added; Requirements/Upgrading for gpyreg); [catalogue](pybads/bads/README.md) KD-B1-6, KD-B1-5, open porting work
- [x] D5 [TODO](dev/TODO.md) (porting gaps, gpyreg release, what's new), [ledger](dev/results/2026-09-28-port-correctness-review.md) (W3-35, W4-11, udist loose end), [AGENTS.md](AGENTS.md)

## E. Verification
- [ ] E1 `/doublecheck`
- [ ] E2 Push both branches; report

## Success criteria
- `periodic_vars` optimizes across the wrap; default runs bit-identical (fingerprint)
- gpyreg and PyBADS suites pass; pre-commit clean
- Periodic-on runs at least as good as periodic-off on the periodic suite

## Notes
- Fingerprint with gpyreg 1.3.3 after B1-B6: `4146a986863602cb` (unchanged).
- gpyreg minimum/pin and CHANGELOG "Requirements"/"Upgrading" line move at gpyreg's tag: recorded in [TODO](dev/TODO.md) "gpyreg releases after 1.3.3".
- Examples toctree globs `_examples/*`: a new notebook needs no docs entry.
- Tolerances of the two new tests in [test_bads_optimization.py](pybads/testing/bads/test_bads_optimization.py) are provisional until the seed sweep (B7).
