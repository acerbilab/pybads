<!-- Report of reviewer (b) of the doublecheck of wave 4, the fixes of O and of the other rows (briefs/wave4_doublecheck.md), reading PyBADS at 81385ac in /home/user/pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave4/doublecheck/b_O/. -->


# Wave 4 doublecheck: the fixes of O and of the other rows

## 1. Coverage

**Read completely:**
- The brief: the common part and scope (b).
- `verification/wave4.md` (all 573 lines).
- Fix reports `A_W4-21_…W4-27.md`, `D_W4-29.md` and `C_…W4-28.md` (all of C, including the rows that are not mine).
- The W4-21 gate files: `w4-21_vs_reference.md`, `geometry_w4-21_vs_base.md`, both `_fields.txt`, `w4-21_changed.txt` and `geometry_w4-21_changed.txt`. Also `orchestrator/changed_runs.py`, `w430_init_stop.py` and its `.out`, and `changelog_entries/w421_points.txt`.
- The diffs of all 16 commits in my scope.
- Code at 81385ac:
  - `constraints_check.py`, `rounding.py`, `grid_functions.py` (`force_to_grid`), `acq_fcn_lcb.py` and its `__init__.py`, `search_hedge.py`, `poll_mads_2n.py`
  - `es_search.py`: `__init__` and the generation loop
  - `gaussian_process_train.py`: the geometry after a refit, and the return paths of `_robust_gp_fit_` and `add_and_update_gp`
  - `optimize_result.py`: `fsd` and `yval_vec`
  - `bads.py`: the checks (800-935), the start of `optimize`, the final estimate and the `"done"` call (1640-1860), and `_poll_step_` (2283-2688)
  - The `.ini` descriptions.
- The tests these commits added or changed, in `test_search.py`, `test_empty_search.py`, `test_bads_inputs.py` (430-700), `test_stobads.py`, `test_gaussian_process_train.py` and `test_noisy_runs.py`.
- MATLAB at 74919c0:
  - `uCheck.m`, `force2grid.m`, `pollMADS2N.m`, `acqLCB.m`, `ESupdate.m`, `searchHedge.m`
  - `searchES.m`:90-140
  - `acqPortfolio.m`:60-72
  - `gpupdate.m`:270-330 and 380-420
  - `bads.m`:1125-1200 and 1440-1480
  - The `hypweight` lines of `gpdefStationaryNew.m` and `gpHyperSVGD.m`.

**Skimmed:**
- The `Unreleased` changelog entries named by my rows.
- The Fig. 1 captions in `README.md` and `index.rst`.
- `variables_transformer.py` (the default log rule).
- The `a14524d..8c8d6f8` package diff: docstrings, messages and #79's checks only.

**Not reached (other reviewers' scope):** `known_differences.md`, `matlab_side_defects.md`, `dev/TODO.md`, `fp_all.out`, the API page.

**Scripts and outputs** are in `/home/user/dc4/b_O`: `c1`–`c11` `.py`/`.out`, `w415_*`, `w430_init_stop_81385ac.out`, `pycln_check/` and `pylint_unused.out`.

## 2. What holds

**W4-21 (`86512c9`), the code.**
- `contraints_check` bins with `round_half_away`. Against `ucheck_ref.py` (`c4_ucheck.out`):
  - Same bins in 3000/3000 random sets on grids no finer than a bin, and 3000/3000 on grids of 2^-21 to 2^-24.
  - Identical rows in all 6000 when the input is given sorted and unique. Otherwise only the representative within a bin differs, which is W3-1's recorded difference.
  - Every edge case matches: halves on both sides of zero, 0.49999999999999994 on either side, the next double above 1/2, bins -0/+0, evaluated points on halves, and candidates half a bin apart.
- `round_half_away` equals exact half-away rounding on 60,019 values, and NaN and ±inf pass through.
- `force_to_grid` still uses it, and equals 8c8d6f8's expression on 40,000 points, halves included.
- The ES split equals `searchES.m:111` (MATLAB's linspace and round) for every integer mu from 1 to 4096: [683, 682] at `n_search_iter` 3, [1, 0] at 4096 (`c5_split.out`).
- The reason for `pybads/rounding.py` holds: `function_logger/__init__` imports `constraints_check` first, and importing `pybads.search` would reach `from pybads.function_logger import FunctionLogger` while that package is only partly initialized.
- The tests' docstrings and the bin comments in `test_search.py` are true.

**W4-21 gates.**
- `w4-21_vs_reference.md` compares `population_linux_wave3_20260927` (pybads `a14524d`) with `86512c9`, 540 runs.
- `geometry_w4-21_vs_base.md` compares `geo_base_8c8d6f8` with `86512c9`, 210 runs.
- Both reached the changed code: 34 and 30 runs changed, deterministic configurations among them.
- What the gate shows of `ellipsoid_D3_homo`:
  - 13 runs changed, but only 9 end at other points.
  - Solved 22 → 19 of 30, while the median error of the 30 runs fell from 0.0684 to 0.0641.
  - KS 0.10 (p 0.999) on the error and 0.067 on the counts; signed-rank p 0.77; median paired log10 ratio +0.000 [+0.000, +0.000].
  - In the wave-3 reference, three runs sit at 0.085–0.098, just under the tolerance of 0.1, and seven lie within a factor 1.5 of it (`c6_homo.out`).
  - So the 0.10 fall is three runs crossing the threshold, with nothing else moving. The W4-21 population is not committed, so the three seeds cannot be named.
- The ledger's reading is right except as in F3.

**W4-17 (`2f15781`).**
- `n` is gone, and pylint finds no unused variable or import in `acq_fcn_lcb.py`.
- `acq_fcn_lcb`'s docstring is true: the poll calls it with the default schedule (`bads.py:2443`). So is `update_hedge`'s.
- The commit message's "`n` only in the catch branch" is loose: `acqLCB.m:46` also uses `n`. No record relies on "only".

**W4-18 (`6e24519`) and W4-29 (`bd793f2`).**
- The ranges hold: `n·γ ≤ 1` with n the length of `search_method` (n = 1 and 3 checked); `hedge_beta` in [0, inf); `hedge_decay` in [0, 1].
- The `hedge_beta` message names `1e-3 / options['tol_fun']`. A `tol_fun` of -1e-3, `np.float64(0)` or 1e-320 is refused through it.
- The defaults are accepted.
- The comments' formulas are true, and the MATLAB lines they cite are the unchecked uses (`searchHedge.m:45-46`, `acqPortfolio.m:69`).
- Exceptions: F1.

**W4-19 (`36c9ec1`).**
- At creation (`bads.py:926-933`) and on a callable's value at each call (`acq_fcn_lcb.py:52-60`), the check refuses booleans, `None` returned, strings, Python and NumPy complex, 0, negatives, ±inf, NaN, arrays of two, and `Decimal`.
- It accepts positive Python or NumPy numbers and one-element arrays of any shape (and a one-element list), converted with `.item()` at each call (`c1`, `c3`).
- The target is not called before the refusal.

**W4-22 (`3b7e64c`).**
- The three imports are gone.
- pycln 2.6.0 from the pre-commit environment leaves all three unchanged at `8c8d6f8`, so the removal by hand was needed.

**W4-23 (`65e2434`).** `search_dist = 0.0` (`bads.py:2097`). Only `_update_search_stats_` reads it.

**W4-24 (`4f535b8`).** `force_to_grid`'s docstring is true against `force2grid.m:3-5`. For `round_half_away`'s docstring, see F7(a).

**W4-25 (`36e8b70`).**
- It refuses 0, negatives, non-integers, NaN, ±inf, booleans, strings, complex numbers and arrays.
- It converts whole-number floats and NumPy ints and stores an `int`.
- MATLAB has no check (`setupoptions.m:26`, `searchES.m:125`).
- Exception: F2.

**W4-27 (`6f673a2`).** The message is logged at DEBUG on the `"BADS"` logger, and the test asserts exactly one record at DEBUG. `searchES.m` prints nothing.

**W4-15 (`6c36782`).**
- `poll_best_improvement > 0` holds if and only if `u_poll_best` was replaced by a polled point (`bads.py:2302-2303`, `2531-2537`, `2586-2594`).
- The test asserts that the incumbent changed and the poll is marked.
- I reran the nine `v_f1_selfmove` runs at `b61a880` and at `6c36782`: the six self-moves are gone, each has three fewer search rebuilds, and all nine are identical in every evaluated point, value and result (`w415_*.out`, `w415_compare.out`).

**W4-16 (`fa5d842`).**
- The code matches `gpupdate.m:299` and `313-317`, with MATLAB's `hypweight` = 1/N (`gpdefStationaryNew.m:207`, `gpHyperSVGD.m:14`).
- The test asserts two samples and transcribes `gpupdate.m`.

**W4-26 (`684d2e0`).**
- The optim_state at `"done"` equals the result (`u`, `yval`, `fval`, `fsd`) at levels 1 and 2, in these cases:
  - after later iterations;
  - within the first iteration (budgets 38 and 40);
  - with `noise_final_samples=0`.
- At level 0 it is kept in step through `_update_incumbent_`.
- `iteration_history` holds the final `fval` and `fsd` at the returned iterate in all eight noisy runs, as MATLAB's `iterList` does (`bads.m:1150-1159`). Nothing needs to go to the PI (`c7`, `c10`).

**W4-28 (`ffaf424`).**
- `_poll_step_` has one return, and every GP function it calls returns the GP it was given.
- In 40 polls over four 200-evaluation runs (levels 0, 1, 2 and `stobads`), the returned GP is the one given (`c8`).
- Every test wrapper of `_poll_step_` returns its result.

**W4-30 (`4b84a2d`).**
- For a stop at `"init"`:
  - At level 1, `fsd` is `noise_size`: 1.0, 2.5, or the first element of a pair, including when the noise test raises the level.
  - At level 2, it is the target's SD at the incumbent.
- The orchestrator's `w430_init_stop.py`, rerun at 81385ac, gives the same `fsd` values (`fval` differs by W4-1's design).
- Gap: F6.

**W4-20 (`5442a6c`).**
- The caption is true for a linearly mapped variable. At default the steps in u are ±mesh_size along one coordinate, and in x they scale with the plausible box (steps of 1 and 10 for widths 2 and 20, `c9`).
- `README.md` and `index.rst` carry the same text.
- Exception: F5.

**The PI's question on the checks** (`c1`, `c2`, `c3`):
- **Booleans:** refused everywhere. A one-element boolean array is accepted by `hedge_beta` and `hedge_decay`.
- **`None`:** stands for the default.
- **Strings:** refused.
- **Python complex:** refused.
- **NumPy complex:** refused by `sqrt_beta` and `n_search_iter`. Accepted by the three hedge checks whatever the imaginary part, for instance `0.1-5j`.
- **NumPy integer and float scalars:** accepted in range. `n_search_iter` becomes an `int`; the others are stored as given, which the run handles.
- **Whole-number floats:** converted for `n_search_iter`.
- **One-element arrays:**
  - `n_search_iter` refuses them.
  - `sqrt_beta` accepts them and converts at each call.
  - The hedge checks store the array, and some of these arrays stop the run (F1).
- **Larger arrays, inf, -inf, NaN:** refused by all five checks.
- **Range limits:**
  - `hedge_gamma`: 0 and 1/n accepted; the next double above 1/2 refused at n = 2. At n = 3, 1/3 and the next double above it are both accepted, since 3γ rounds to 1.
  - `hedge_beta`: 0 and 1e308 accepted; -1e-12 refused.
  - `hedge_decay`: 0 and 1 accepted; the next double above 1 and -1e-12 refused.
  - `n_search_iter`: 1 accepted, 0 refused.
  - `sqrt_beta`: 1e-300 accepted, 0 refused.
- **Against the texts:** the changelog and "Upgrading" lines hold except as in F1 and F2. The `search_acq_fcn` description is covered in F7(b). The `Raises` section holds except for F2's `TypeError`.

## 3. Findings

### F1. The checks of `hedge_gamma`, `hedge_beta` and `hedge_decay` accept values that the run cannot use, or uses differently
- Where: `pybads/bads/bads.py:871-925` at 81385ac; the failures occur at `pybads/search/search_hedge.py:65`, `69`, `83` and `173`.
- Kind: defect of a fix
- Severity: minor
- What is ruled and what is true. W4-18 and W4-29 refuse a value outside the range, and the changelog says `BADS` raises `ValueError` for "a `hedge_gamma` that is not a number from 0 to 1/n" and "a `hedge_beta` that is not a finite number at least 0". Instead (`c1_checks.out`, `c2_accepted_runs.out`, runs at D = 2 of at most 80 evaluations):
  - **One-element arrays** of any shape are accepted and stored as arrays.
    - `hedge_decay=np.array([0.5])` or `[[0.5]]` stops the run in its first search with `ValueError: setting an array element with a sequence` (`search_hedge.py:173`).
    - `hedge_gamma=np.array([[0.1]])` and `hedge_beta=np.array([[1.0]])` stop it with `shape mismatch` (`:83`).
    - Stopping at the first search with an unrelated error is what the checks at creation were meant to prevent.
  - **NumPy complex scalars** pass whatever their imaginary part, for instance `np.complex128(0.1-5j)`. The run goes on with complex probabilities and gains cast to real, and a `ComplexWarning`.
  - **A one-element boolean array** (`np.array([True])`) is accepted by `hedge_beta` and `hedge_decay`, although booleans are refused.
  - **`Decimal` and `Fraction`** are accepted: `Decimal('0.25')` as `hedge_gamma` raises `TypeError` at `:69`, and `Fraction(1, 4)` as `hedge_beta` at `:65`, both in the first search.
  - The comments at `bads.py:883`, `903` and `921` name "a complex number" among what the `except` catches; a NumPy complex never reaches it.
  - The ledger's "Found while fixing" records the complex values and that "a one-element array is accepted and stored as an array", but not that such arrays stop the run.
  - By contrast, W4-19's `_is_positive_finite_real` (NumPy array, size 1, dtype kind `iuf`, finite) refuses all of these.
- Would the correction move results: no. Default values are Python floats.
- Proposed correction: check each value as `_is_positive_finite_real` does (size 1, kind `iuf` and not boolean, finite, in range) and store `float(np.asarray(v).item())`. Failing that, record in "Found while fixing" and `dev/TODO.md` that such arrays stop the run at the first search.

### F2. `n_search_iter` of 2**63 or more raises `TypeError`, not `ValueError`
- Where: `pybads/bads/bads.py:862` at 81385ac.
- Kind: defect of a fix
- Severity: minor
- What is ruled and what is true. The ruling refuses what is not a positive integer, with `ValueError`. For a Python int of 2**63 or more, `np.isfinite(n_search_iter)` raises `TypeError: ufunc 'isfinite' not supported`, and `2**70` is refused with it although it is a positive integer (`c1_checks.out`). Meanwhile `1e308` is accepted and stored as a 309-digit int. Any value above `n_search` (4097 at default) is accepted and gives mu = 0: the ES search then draws nothing, and a run of 80 evaluations ended at 40 without an error. That case is outside the ruling; I note it for the PI.
- Would the correction move results: no.
- Proposed correction: test `isinstance(v, (int, np.integer))` before `np.isfinite` (or use `math.isfinite`). Whether to refuse `n_search_iter > n_search` is for the PI.

### F3. The ledger's reading of W4-21's default-suite gate: the per-configuration counts and "the median errors held or lower"
- Where: `dev/experiments/port_review_20260925/verification/wave4.md:480` at 81385ac.
- Kind: number does not recompute
- Severity: minor. The brief repeats the ledger's reading ("13 runs ending at other points").
- What is stated and what is true.
  - The ledger reads "25 of them ending at other points (`ellipsoid_D3_homo` 13, `ellipsoid_D3_hetero` 10, `rosenbrock_D2` 4, …)". Those counts are the changed runs, which sum to 34. The runs ending at other points are 9, 7, 4, 2, 2, 0, 0 and 1 (`w4-21_changed.txt`).
  - "The median errors held or lower" is false for `ellipsoid_D3`: its median error over 30 runs rose from 2.34e-6 to 3.15e-6 (`w4-21_changed.txt`; signed-rank statistic 0, so both of its changed pairs moved the same way).
- Would the correction move results: no.
- Proposed correction: "34 of 540 runs changed, 25 of them ending at other points (changed and elsewhere: `ellipsoid_D3_homo` 13 and 9, `ellipsoid_D3_hetero` 10 and 7, `rosenbrock_D2` 4 and 4, `ellipsoid_D3` 2 and 2, `ellipsoid_D3_unbounded` 2 and 2, `sphere_D3_homo` 1 and 1, `multisensory_s1_D6_homo` and `sphere_D3_hetero` 1 and 0 each), the median errors held or lower but `ellipsoid_D3`'s, 2.3e-6 → 3.2e-6, …".

### F4. W4-21's changelog entry does not carry what the gate measured, as its ruling asks
- Where: `wave4.md:93` (the ruling) and `CHANGELOG.md:649-657` at 81385ac.
- Kind: ruling not implemented
- Severity: minor
- What is ruled and what is true. The ruling says "The changelog's 'Points evaluated again' extended with what the gate measures". Fix agent A also proposed "Add the gate's measure". The entry, and `changelog_entries/w421_points.txt`, gained only the rounding rule. No record gives a reason. The wording is reviewer (c)'s.
- Would the correction move results: no.
- Proposed correction: one clause with the gate's outcome (34 of 540 runs of the default suite changed, none flagged), or a line under "Choices within the rulings" saying why the entry carries no measure.

### F5. Fig. 1's caption: the steps scale with the plausible box only for a variable that is mapped linearly
- Where: `README.md:121` and `docsrc/source/index.rst:53` at 81385ac; the default log rule is at `pybads/variable_transformer/variables_transformer.py:157-175`.
- Kind: false statement
- Severity: minor
- What is stated and what is true.
  - The caption says the poll's steps "scale with the plausible box in the original coordinates".
  - At default (`nonlinear_scaling=True`), a variable whose bounds are all positive and whose `pub/plb` is at least 10 is mapped through a log. Its steps in x grow with its value and are unequal on the two sides: steps of ±1 in u gave +27.0 and −2.70 at x2 = 3, with a plausible box of [0.1, 10] (`c9_poll_steps.out`).
  - The W4-20 row names the log transform; the ruled clause leaves it out.
- Would the correction move results: no.
- Proposed correction: "…and, for a variable mapped linearly, scale with the plausible box in the original coordinates, drawn here; a variable with positive bounds whose plausible box spans a factor of 10 or more is mapped through a log, where the steps grow with its value."

### F6. The description of `fsd` names only the stop at `"init"` as a case without an estimate
- Where: `pybads/bads/optimize_result.py:30-36` at 81385ac.
- Kind: false statement (by implication)
- Severity: minor
- What is stated and what is true. The description implies that every other noisy run reports an estimate. But a noisy run whose initial design uses up `max_fun_evals` ends within its first iteration with no final sample left. It then reports `fsd = noise_size` at level 1, or the target's SD at the incumbent at level 2, with `yval_vec` None (`c7`: level 1 gives 1.0, level 2 gives 0.7, at budgets 3 and 10; `c11`: at D = 2 level 1, a budget of 33 gives `fsd=1.0`, while 34 takes one sample). The `yval_vec` description already names this budget case.
- Would the correction move results: no.
- Proposed correction: "For a noisy run that takes no final samples, one that `output_fcn` stops in its initialization or whose `max_fun_evals` leaves no evaluation for a final sample after the initial design, it is not an estimate: …".

### F7. Small wording errors in the texts of the rows
- Where: `pybads/rounding.py:21-24`; `pybads/bads/option_configs/advanced_bads_options.ini:183`; `wave4.md:105`; all at 81385ac.
- Kind: false statement
- Severity: minor
- What is stated and what is true.
  - (a) `round_half_away`'s Returns section says `r : np.ndarray`, but a scalar input returns `np.float64` (`c11_misc.out`).
  - (b) The `search_acq_fcn` description says `sqrt_beta` is "a positive number", but the check also refuses `inf`. The changelog and the docstrings say "positive finite".
  - (c) The W4-22 row says pycln keeps the imports "as third-party imports". `multiprocessing.sharedctypes` is standard library. pycln 2.6.0 keeps all three (checked); the reason is side effects it cannot rule out.
- Would the correction move results: no.
- Proposed correction: (a) "r : np.ndarray or np.float64"; (b) "a positive finite number"; (c) "which pycln keeps, since it cannot prove that importing them has no side effects".

## 4. Outside my scope

- `bads.py:6` imports `matplotlib.pyplot` and never uses it (pylint W0611, `pylint_unused.out`). It is the package's only matplotlib import, and `pyproject.toml:11` requires matplotlib. W4-22's commit says "nothing … uses matplotlib but `bads.py`".
- Reviewer (a): W4-14's ruling says it closes "the default fsd of such a run", but a run whose design uses up the budget still reports `noise_size` (F6; `c7`, `c11`).
- The same `TypeError` for a Python int of 2**63 or more affects W3's checks of `max_fun_evals` (`bads.py:808`) and `accelerate_mesh_steps` (`:841`).
- Reviewer (c): the changelog says the ES split is "as MATLAB BADS does". With `n_search` = 4096, the quotient is an odd integer only at `n_search_iter` 4096. At 3, MATLAB's `es.mu` is 1365.33 (`setupvars.m:186` does not round it down), and `searchES.m:117`'s `randn(N, nvars)` gets a non-integer size (A's item, unverified).
- Reviewer (d): `w430_init_stop.out` ran against `/home/user/pybads-fix-C`, fix agent C's worktree, and its script prints no gpyreg path. The rerun at 81385ac gives the same `fsd` values.
- Scratch: `at_b61a880/` and `at_6c36782/` are `git archive` extracts of `pybads/` and can be left out of the copy, as can `__pycache__/`.
