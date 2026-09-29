# TASK: PyBADS code work that doesn't depend on the release

The four `dev/TODO.md` items that need no release, MATLAB, macOS or Windows:
loose ends of the port review, stage timers + profiler, replay + oracles, the
GP on a one-point training set. Branch `claude/todo-discussion-ad0tsh` (on
`dev-next` at `bec8a57a`). Executors: Opus sub-agents; orchestrator merges,
pushes and runs `/doublecheck` at the end.

## Prerequisites
- [x] venv `.venv` with gpyreg v1.3.3 editable (`../gpyreg`) and PyBADS `.[dev]` (NumPy 2.4.6, SciPy 1.17.1, Python 3.11); pre-commit hooks installed, tree passes
- [x] Clones: `../gpyreg` (v1.3.3), `../acerbilab/pyvbmc` (`dev-next`), `../acerbilab/bads` (MATLAB)
- [x] Baseline at `bec8a57a`: 767 passed (3 min 25 s); fingerprint `4146a986863602cb` (1 BLAS thread and default, this container)

## Research (done)
- [x] Loose ends: ledger corrections: item 3 is reached at default options with `non_box_cons`; item 5 returns an uninitialized row; item 6 is false (`transvars.m:65` clips too)
- [x] Stage timers + profiler: exclusive stage stack; a wrapper probe kept the fingerprint
- [x] Replay: exact only on one machine/kernel/thread count (runs part at eval 24-39 under another OpenBLAS kernel), so a dev tool; PyVBMC's oracles are self-generated, not MATLAB's
- [x] One-point GP: only the initial fit sees one point; mean prior centred at 0.5; six RuntimeWarnings reach users

## PI rulings (2026-09-28): all recommendations accepted
- Loose ends: 1 document (option description); 2 document (TODO "Porting gaps", KD-B1-6); 3 keep PyBADS's count + KD entry, correct the ledger; 4 drop; 5 fix (0-row placeholders); 6 drop, correct the ledger; 7 keep refusing, name `specify_target_noise=True` in the message + KD entry; 8 fix (one line); 9 delete `y_max`, update `Y_max` on a merge, keep `cache_count`; 10 take W2-36's measurement; 11 document in KD-B1-8
- Timers: internal only (`optim_state`, `iteration_history["timer"]`); exclusive accounting (stages + target = `total_time`); fine stage set; port `profile_run`/`profile_suite`/`profile_compare`; stage totals in `population.py` records
- Replay: dev tool `dev/scripts/replay.py`, parent-commit gate, nothing committed; CI test pinning the initial design
- Oracles: self-generated, about ten components, CI-wide; MATLAB-derived ones stay in TODO
- One-point GP: skip the fit on one distinct point, MATLAB's starting values, mean prior at y1 (SD 1); two points unchanged; refit warnings on zero-spread inputs → gpyreg TODO

## Phases (sequential unless noted; one heavy process at a time)
- [x] P1 Loose ends 1-9, 11 (Opus; main checkout) — fingerprint + suite
  `7110bba` (5), `5ac25aa` (7), `28fef97` (8), `c0aeaf4` (9), `c246982` (docs: 1, 2, 3, 6, 11, ledger, TODO); 778 passed; fingerprint `4146a986863602cb` (1 thread and default)
- [x] P5a Replay tool + initial-design pin (Opus; worktree) — merged `31060da`; replay identical at HEAD, parts under Sandybridge / 4 threads
- [x] P1b W2-36 measurement (Opus; worktree) — `fb830a4`; no flag in 15 tests; `sphere_D3_hetero` paired fraction solved 0.50 → 0.61 (McNemar p 0.03 after Holm over five); kept as MATLAB, adoption for the PI
- [x] P2 Stage timers + profiler (Opus) — fingerprint + suite; timing campaign on a quiet machine
  `cf371d4` (timers), `a4a423b` (profiler), results note; 811 passed; fingerprint `4146a986863602cb` (1 thread and default); `optimize()` body now in `_optimize_()`
- [x] P3 One-point GP (Opus) — fingerprint; geometry suite + noisy thin band, 30 seeds, base vs change
  `73d517a` (fix), `d9772a0` (`thinband` suite), records; 823 passed; fingerprint `4146a986863602cb` (1 thread and default); replay 8/8 identical; no flag, the 144 one-point runs change; refits on zero-spread inputs still warn (every `sphere_band_D3` run; a one-point refit at D = 1): gpyreg item
- [x] P5b Self-generated oracles (Opus) — after P3
  `d75efa2` (6 states, 14 oracles, 580 KB), `d82c075` (generator), records; 913 passed; fingerprint `4146a986863602cb`; floors over threads 1/2/4, Haswell, Sandybridge; the 3 deterministic post-refit GPs (cond. 1e15-1e18) platform-bound; wheel `--pyargs` ok
- [x] Records: `dev/TODO.md`, ledger "Open ends", `pybads/bads/README.md`, `CHANGELOG.md`, `AGENTS.md`/`dev/README.md` where tooling is added

## Verification
- [x] Test suite green; fingerprint unchanged where nothing may move; gates where results move — at `38491bf`: 913 passed, 18 skipped (platform-bound oracles); fingerprint `4146a986863602cb` (1 thread and default); pre-commit clean; dev tests 29 passed; replay of `bec8a57` against HEAD 8 of 8 identical
- [!] `/doublecheck` (comprehensive, 5 Opus reviewers): 3 must-fix (W2-36 adoption not recorded as open; oracle `--rebaseline` dead-ends off the generating machine; `profile_suite.py` relative `--out` loses runs), ~13 should-fix (BADS reference cycle via the stage timer; replay recorder fragility; oracle `--exact` silent skips, gpyreg key, option coupling; stale `BADS.optimize` pointers; changelog gaps; doc slips) — fixes pending the PI

## Fixes after the double-check (PI, 2026-09-29: fix the findings; W2-36 keeps MATLAB's behaviour; the one-point side effects stay with gpyreg)
- [ ] F1 package code: `BADS` reference cycle; stale `BADS.optimize` pointers; changelog gaps (`(f, sd)` message, `FunctionLogger`, one-point entry); comments; tests (stage balance on `output_fcn` exits, one point at level 2, median rule at N = 5) — worktree `fix-code`
- [ ] F2 dev tooling: `profile_suite.py` relative `--out`; replay recorder robustness and `check` warnings; gate recipe in dev/README; population/benchmark guards, thread variables; stage-times note facts; TODO item for shared dev helpers — worktree `fix-tools`
- [ ] F3 oracles: no platform-bound references in fixtures (rebaseline works anywhere); honest `--check --exact`; gpyreg and CPU features in the key; removed options tolerated; hedge margins; AGENTS.md gate sentences; shipped-tests changelog line — worktree `fix-oracles`
- [ ] Merge F1-F3; F4 records: W2-36 ruling; ledger, wave notes, KD "Settled by", one-point README fixes
- [ ] Gates on the merged head; focused re-review

## Success criteria
Each item fixed, documented or dropped per the rulings; the four TODO items closed or narrowed to what needs MATLAB; suite green.

## Notes
- After merging P1b, P5a: 817 passed; replay of `cf371d4^` (before the timers) against HEAD: 8 of 8 identical
