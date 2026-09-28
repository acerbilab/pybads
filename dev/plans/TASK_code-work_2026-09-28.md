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
- [~] P1 Loose ends 1-9, 11 (Opus; main checkout) — fingerprint + suite
- [~] P5a Replay tool + initial-design pin (Opus; worktree, parallel with P1)
- [ ] P1b W2-36 measurement (Opus; heavy, ~25-45 min) — population compare, 5 noisy configs, 90 seeds
- [ ] P2 Stage timers + profiler (Opus) — fingerprint + suite; timing campaign on a quiet machine
- [ ] P3 One-point GP (Opus) — fingerprint; geometry suite + noisy thin band, 30 seeds, base vs change
- [ ] P5b Self-generated oracles (Opus) — after P3
- [ ] Records: `dev/TODO.md`, ledger "Open ends", `pybads/bads/README.md`, `CHANGELOG.md`, `AGENTS.md`/`dev/README.md` where tooling is added

## Verification
- [ ] Test suite green; fingerprint unchanged where nothing may move; gates where results move
- [ ] `/doublecheck` (comprehensive)

## Success criteria
Each item fixed, documented or dropped per the rulings; the four TODO items closed or narrowed to what needs MATLAB; suite green.

## Notes
