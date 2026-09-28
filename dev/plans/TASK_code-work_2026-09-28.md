# TASK: PyBADS code work that doesn't depend on the release

The four `dev/TODO.md` items that need no release, MATLAB, macOS or Windows.
Branch `claude/todo-discussion-ad0tsh` (on `dev-next` at `bec8a57a`).

## Prerequisites
- [x] venv `.venv` with gpyreg v1.3.3 editable (`../gpyreg`) and PyBADS `.[dev]` (NumPy 2.4.6, SciPy 1.17.1, Python 3.11)
- [x] Clones: `../gpyreg` (v1.3.3), `../acerbilab/pyvbmc` (`dev-next`), `../acerbilab/bads` (MATLAB)
- [x] Baseline at `bec8a57a`: 767 passed (3 min 25 s); fingerprint `4146a986863602cb` (1 BLAS thread and default, this container)

## 1. Research and decisions (discuss with PI before implementing)
- [x] Loose ends: proposal per item (fix / document / drop). Ledger corrections: item 3 is reached at default options with `non_box_cons`; item 5 returns an uninitialized row; item 6 is false (`transvars.m:65` clips too)
- [x] Stage timers + profiler: exclusive stage stack, probe kept the fingerprint
- [x] Golden replay: exact replay holds only on one machine/kernel/thread count (runs part at eval 24-39 under another OpenBLAS kernel), so a dev tool; PyVBMC's oracles are self-generated, not MATLAB's
- [x] One-point GP: only the initial fit sees one point; its mean prior is centred at 0.5; six RuntimeWarnings reach users
- [~] PI rulings recorded here (asked 2026-09-28)

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
