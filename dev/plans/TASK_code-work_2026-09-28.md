# TASK: PyBADS code work that doesn't depend on the release

The four `dev/TODO.md` items that need no release, MATLAB, macOS or Windows.
Branch `claude/todo-discussion-ad0tsh` (on `dev-next` at `bec8a57a`).

## Prerequisites
- [x] venv `.venv` with gpyreg v1.3.3 editable (`../gpyreg`) and PyBADS `.[dev]` (NumPy 2.4.6, SciPy 1.17.1, Python 3.11)
- [x] Clones: `../gpyreg` (v1.3.3), `../acerbilab/pyvbmc` (`dev-next`), `../acerbilab/bads` (MATLAB)
- [~] Baseline: test suite green, fingerprint hash recorded (1 BLAS thread and default)

## 1. Research and decisions (discuss with PI before implementing)
- [~] Loose ends: proposal per item (fix / document / drop)
- [~] Stage timers + profiler: design after PyVBMC's `profile_run.py`
- [~] Golden replay: design after PyVBMC's `golden_replay.py`
- [~] One-point GP: options for priors/bounds, PyBADS vs gpyreg
- [ ] PI rulings recorded here

## 2. Implementation
- [ ] Loose ends (per rulings)
- [ ] Stage timers (fingerprint unchanged)
- [ ] Profiler scripts
- [ ] Golden replay + test
- [ ] One-point GP fix + thin-band test (gate on a config that reaches it)
- [ ] `dev/TODO.md`, ledger "Open ends", `pybads/bads/README.md`, `CHANGELOG.md` updated

## 3. Verification
- [ ] Test suite green; fingerprint where nothing may move; gate where results move
- [ ] `/doublecheck`

## Success criteria
Each item fixed, documented or dropped per the PI's ruling; the four TODO items closed or narrowed; CI-equivalent suite green.

## Notes
