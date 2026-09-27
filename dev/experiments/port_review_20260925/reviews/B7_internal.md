<!-- Report of the B7 internal reviewer (function logger, initial design, utilities, internal-correctness track), wave 4 of the port review, reading PyBADS at 0d866e8 in ../pybads-review and gpyreg v1.3.3 (98ab5a4), in a cloud session, with the complete history; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave4/B7_internal/. Nothing in it is verified. -->

# B7 internal review: function logger, initial design, utilities

On this track, "port discrepancy" means PyBADS does not do what its own documentation or the mathematics says. It does not mean a checked difference from MATLAB: I did not open the MATLAB code.

## 1. Coverage

**Read in full (at `0d866e8`, in `/home/user/pybads-review`):**
- `pybads/function_logger/function_logger.py`, `pybads/init_functions/init_sobol.py`, `pybads/utils/period_check.py` and the three package `__init__`s.
- In `pybads/bads/bads.py`:
  - `__init__` (the random `x0` at 322-339, the logger's construction at 361-369);
  - `_init_optim_state_` (the start `u0` and its grid at 686-712, the refusal of periodic variables at 643-647, the refusal of `fun_values` at 758-763, the uncertainty levels);
  - `_init_mesh_` (1035-1177) and `_init_optimization_` (1179-1284);
  - in `optimize()`, the final samples (1657-1670) and `overhead` (1716-1720);
  - the three `period_check` call sites (1130, 1837, 2220) and the search and poll calls of the logger (1887, 2355).
- `pybads/function_logger/constraints_check.py`, as the consumer of the design.
- The readers of the log in `pybads/bads/gaussian_process_train.py`: `_get_gp_training_options` (`n_eff`), `get_grid_search_neighbors`, `_get_fevals_data`.
- `optimize_result.py` (`func_count`, `overhead`).
- The option descriptions I needed in both `.ini` files.
- Docs: `docsrc/source/api/classes/function_logger.rst`, `api/advanced_docs.rst` (a toctree only; it has no section on the target function), `index.rst`, `quickstart.rst`.
- `CHANGELOG.md` (the entries on merges, repeats, the noise test, overhead and the generator).
- Tests: `testing/function_logger/test_function_logger.py`, the relevant parts of `test_run_control.py`, `test_noisy_runs.py` and `test_bads_seed.py`.
- The sheet and the counterpart map.

**Skimmed:** the rest of the main loop, `search/grid_functions.py` (`force_to_grid`, `grid_units`), `utils/timer/timer.py`.

**Not reached:**
- The MATLAB code. For the History lines I only listed commits with `git log --since=2022-02-11`: `init/initSobol.m`, `init/private/*` and `utils/periodCheck.m` have none; `private/funlogger.m` changed in `d4fead5`; `private/evalinitmesh.m` changed in `75ec49f`, `d4fead5` and `a21f2ee`.
- The text of the BADS paper, which is not available here. Where I cite it, it is from memory and marked so.

**Checks run** (scripts and logs in the B7_internal scratch folder; every one printed `pybads`/`gpyreg` from the review worktree and the v1.3.3 clone):
- `s1_seed.py`: the seed's derivation.
- `s2_design_seed.py`: the design across seeds and random starts.
- `s3_merge_reach.py`: an instrumented `_record` over 9 seeded runs of at most 200 evaluations, at levels 0, 1 and 2.
- `s4_checks.py`: the checks of the target's outputs.
- `s5_growth.py`: `cache_size` 3 against 500, plus `finalize` and `add`.
- `s6_size.py`: a table of design sizes.
- `s7_small.py`: an empty design, and a start at `plb`.
- `s8_complex.py`: a complex value, and the `n_eff` offset.
- `s9_periodic.py`: the refusal of periodic variables.

## 2. Answers to the first questions

### 1. The evaluation

**The point.**
- `__call__` squeezes `x` to shape `(D,)` and maps it from `u` space to the original space with `variable_transformer.inverse_transf`. BADS always passes a transformer (bads.py:361-369).
- The target gets a new 1-D ndarray of D elements in the original space (s4). That matches "a vector input" (class docstring) and "takes as input a vector" (quickstart).
- `__call__`'s docstring does not say that its `x` is in the transformed space (F10).

**Outputs by level.**
- At level 2 (`he_noise_flag`, fixed when the logger is created) the target must return exactly a `tuple` of two. A list, an array, a 1-tuple or a 3-tuple is refused with the `specify_target_noise` message.
  - `specify_target_noise` forces `uncertainty_handling=True` (bads.py:855-860), so no noise test runs and the level stays 2.
- At levels 0 and 1 the target must return one value; a tuple is refused as not a scalar.
- When the noise test sets level 1, the logger's own `uncertainty_handling_level` keeps its construction value. Nothing reads it.
- The class docstring says the SD is returned "optionally … (if the function fun is stochastic)". That is imprecise: the SD is required at level 2 and refused at level 1, which is also stochastic.

**Checks of the value.**
- 0-d arrays, size-1 arrays and one-element lists are converted.
- NaN, ±inf, `None`, a complex with a non-zero imaginary part, and a tuple are refused with `ValueError`.
- `True` is accepted.

**Checks of the SD.** It must be finite and positive; 0-d and size-1 arrays are converted. The departures from the documented `ValueError` are F8.

**An error raised by the target** is re-raised with its own type. Its `args` gain the point in the original space. No count, row or time is recorded.

**The count.** `func_count` goes up by one per completed evaluation, whether or not it is recorded; the noise test and the final samples are counted. It is the result's `func_count` ("Number of evaluations of the objective functions").

**Option descriptions and docs.**
- `uncertainty_handling` ("if None, determine at runtime; on with specify_target_noise") matches bads.py:855-870 and 1055-1068.
- `specify_target_noise` ("noise estimate (SD) as second output") matches.
- `function_logger.rst` is an `autoclass :members:`. It renders the class docstring plus `add`, `finalize` and `reset_fun_eval_time`, three methods no code calls. It does not render `__call__`.

### 2. The record

**Rows.** Every recorded evaluation (the start `x0`, the design, the search, the poll) gets a new row at every level. The row holds `X` (in `u`), `X_orig`, `Y`, `Y_orig`, `S` (level 2 only), `X_flag`, `n_evals = 1` and `fun_eval_time`. `Xn` and `X_max_idx` are always equal.

**Growth.** When a row index reaches the array size, every array grows by 50% (at least one row), `n_evals` and `S` included. Runs with `cache_size` 3 and 500 are identical at levels 0 and 2 (s5).

**The merge (KD-B7-3).** The code matches the entry:
- the repeat is precision-weighted into the one row that matches in every coordinate;
- the merged value is returned with the new observation's SD;
- `Y_orig` keeps the first observation.

No run can reach it any more (F5).

**Evaluations that are not recorded.** These are the noise test (default; levels 0 and 1, before its verdict) and the `noise_final_samples` at the returned point (levels 1 and 2). Each one:
- adds no row;
- adds 1 to `n_evals` of the last row equal to the point (the start's row, or the returned point's row);
- is averaged into that row's `fun_eval_time`;
- adds its time to `total_fun_eval_time`, which `_init_mesh_` sets back after the noise test (1061-1065);
- is counted in `func_count`.

The branch for a point with no matching row, which would leave only `func_count` and the total time, was taken by none of the 9 runs. The effects of this bookkeeping are F6 and F7.

**Other fields.**
- `X_flag` marks the filled rows.
- `Y_max` is updated after each new row but not after a merge; nothing reads it.
- `y_max` is never set, and `cache_count` counts only `add` calls. No code outside the logger reads either.
- The per-row `fun_eval_time` has no reader either: `t_train` at gaussian_process_train.py:92 is unused.
- `finalize` has no caller (F9). `add` has no caller: `fun_values` is refused (bads.py:758-763), consistent with KD-B1-4 (F10).

**The readers in other slices get what they expect.**
- They read only filled rows (`X[:X_max_idx+1]` or `X[X_flag]`).
- At level 2, `S` holds SDs, which they square into variances; at levels 0 and 1 they get `None`.
- `_init_mesh_` takes the argmin over `Y[:Xn+1]`; at level 2, `_init_optimization_` reads `S` at the lowest `Y`.
- The search's condition is `len(Y[X_flag]) > D`.
- `func_count` feeds the budget, LCB's `t` and the refit timing.

The one exception is `n_eff` (F6).

### 3. The initial design

**The seed.**
- The seed comes from the integer parts of the first 11 coordinates of `u0`, as the docstring says.
- In a BADS run `u0` is always finite, so the branch that draws the seed from `rng` never runs.
- As a result the design depends on neither `random_seed` nor any start inside the plausible box (F1), and for some starts it depends on the platform (F2).

**The size.**
- `fun_eval_start` is D at default. In a noisy run it becomes `max(20, fes)` capped at `max_fun_evals` (1098-1103), then `min(fes, max_fun_evals - 1)` (1105-1110).
- `init_sobol` rounds it up to `2**ceil(log2(fes))` and doubles it when that equals D (F3).
- Its second return value is the exponent, not the number of points; the caller discards it (F4).

**The map.** `plb + s*(pub - plb)`, with the transformed plausible bounds −1 and 1. The design is uniform on [−1, 1)^D in `u`, and log-uniform in `x` for a variable on a log scale. `random_base2` keeps the first point and a power-of-two size, so the set that `init_sobol` returns is the balanced set that the comment on Owen (2020) describes.

**What `_init_mesh_` does with the design.**
- It keeps the first `n_left` points in Sobol order, a prefix of the sequence. That breaks the balance only when the budget is below the design (KD-B2-6).
- `period_check` returns its input unchanged.
- `force_to_grid` moves points by at most 2^-11 in `u`.
- `contraints_check`:
  - projects the points onto the search bounds, which does nothing inside the plausible box;
  - removes duplicates within the set and points in the bin of a logged point;
  - removes points that violate `non_box_cons`;
  - returns the survivors sorted by bin.

  So the design is evaluated in lexicographic order, not Sobol order. That makes no difference for a full set.
- The start becomes the row with the lowest raw observation over `x0` and the design (at every level; the first row wins on ties). `eff_starting_points = Xn + 1`.

**Inputs at default.** Every input it can receive is handled:
- D = 1 gives 2 points.
- An empty design (`max_fun_evals=2` with the noise test) passes through as a `(0, D)` array. The run ends at 2 evaluations, and `non_box_cons` receives an empty array (s7).
- The scrambling depends on the SciPy version too.

### 4. The utilities

**`period_check`** returns `x` unchanged. For the inputs it can receive today (`periodic_vars` is `None` at the design and an all-False mask at the search and the poll), that is exactly what a periodic check with no periodic variable should do.

**KD-B1-6 matches the code:** bads.py:643-647 refuses any `periodic_vars` that is not `None`. Three details:
- An empty list is refused too, although it names no periodic variable (`_variable_transformer_` even tests `len(periodic_vars) != 0`).
- With a random `x0`, `_variable_transformer_` runs at bads.py:326, before the refusal. So its periodic branch is reached, with no effect, and an out-of-range index raises `IndexError` there instead of the `ValueError` (s9). The entry's "unreachable" is therefore loose for that branch.
- The poll throws away `period_check`'s return value (F11).

## 3. Findings

### F1. The initial design depends on neither `random_seed` nor the start (for any start inside the plausible box)
- **Location:** pybads/init_functions/init_sobol.py:52-64, 69; pybads/bads/bads.py:322-339, 712, 1112-1120. MATLAB: init/initSobol.m (the sheet, KD-B7-1 and C3: the seed there comes from `num2str` of the first 10 values).
- **Category:** random draws
- **Proposed classification:** port discrepancy. The derivation is left open by KD-B7-1, and it contradicts the documented generator contract.
- **Confidence:** high
- **Reached at default options:** yes: every run, every level.
- **History:** `initSobol.m` has no commit since 2022-02-11. The seed lines date from `c7c88ab` (2022-06-02); the `rng` branch was rewritten in `1d075ab` (2026-09-25).
- **What the code does, and why it is wrong:**
  - The seed is the product of the character codes of `array2string(u0[:11].astype(uint64))`, mod 997, plus 1.
  - `u0` is the start in `u` space, where the plausible box is [−1, 1]^D. Every coordinate in (−1, 1) has integer part 0, so every start inside the plausible box gives the string "0 0 … 0" and one seed per D:
    - 49, 948, 967, 241, 748 and 843 for D = 1 to 6;
    - 1 for every D ≥ 8. There the int64 product 2^(9k−5)·3^k wraps to exactly 0.
  - The seed changes only when a coordinate is at or beyond a plausible bound.
  - The `rng` branch (line 64) runs only when `u0` is not finite. That never happens: `__init__` replaces a non-finite `x0` by a random draw first.
  - So in `u` space the design is a fixed point set per D. The documentation says otherwise:
    - the `rng` attribute in the BADS docstring is "The generator of every random draw of the run";
    - `index.rst` and CHANGELOG [1.1.0] say "Every random draw of a run comes from one NumPy random generator".
  - The scrambling actually draws from a second generator, the `default_rng(seed)` inside SciPy, seeded from `u0`.
- **Consequence if real:** runs that differ in seed and start share the whole design. The start after the design is the best of `x0` and the design, so restarts (the usual advice for BADS) often begin from the same point. In the check, D = 3, with 20 seeds and a random `x0`: the designs were identical in all 20 runs, and 15 of the 20 started from the same design point. Restarts are less diverse; no single run is wrong.
- **Suggested reproduction:** `s2_design_seed.py` (the output above); `s1_seed.py` for the seeds.
- **Test adequacy:** no test checks that the design varies with the seed or the start. `test_bads_seed.py` checks reproducibility and that the global random state is untouched, and SciPy's generator never touches the global state, so `test_seeded_run_leaves_global_state_untouched` passes.

### F2. For a start on or below a lower plausible bound, the design depends on the platform
- **Location:** init_sobol.py:55 (`astype(np.uint64)`), 57-61. MATLAB: no counterpart (MATLAB derives its seed from `num2str`).
- **Category:** random draws
- **Proposed classification:** port discrepancy
- **Confidence:** medium. The x86-64 behaviour is observed; AArch64 is by reading.
- **Reached at default options:** only for a start with a coordinate at `u ≤ −1` among the first 11:
  - `x0` equal to `plb`, for example `x0 = lb` with the plausible bounds left unspecified, which BADS accepts and does not move;
  - or `x0` between `lb` and `plb`.
- **History:** `c7c88ab`.
- **What the code does, and why it is wrong:**
  - Converting a negative double to an unsigned integer is undefined in C (C11 6.3.1.4). NumPy 2.4 on x86-64 gives 18446744073709551615 for −1.0 and for −1.5, with no warning. AArch64's `fcvtzu` saturates to 0.
  - `array2string` pads every element to the widest one, so the string fills with spaces. The product then wraps to 0, and the seed is 1 on x86-64.
  - On AArch64 the string is "0 0", which gives an interior start's seed (948 at D = 2).
  - Observed on x86-64: `x0 = (−2, 0.5)` with `plb = −2` gives a different design from `x0 = (−1.9, 0.5)` (s7).
- **Consequence if real:** a seeded run with such a start differs between x86-64 and Apple-silicon macOS from its first design point on.
- **Suggested reproduction:** `np.array([-1.0]).astype(np.uint64)` on both architectures; `s1_seed.py`.
- **Test adequacy:** none. CI runs macOS, but no test compares designs across platforms.

### F3. The design is doubled when its rounded size equals D (W0-18, left open by the sheet)
- **Location:** init_sobol.py:73-76; advanced_bads_options.ini:22. MATLAB: init/initSobol.m (`Ninit` points, per the sheet).
- **Category:** defaults
- **Proposed classification:** unsure (possibly intentional)
- **Confidence:** high on the facts
- **Reached at default options:** yes when D is 1, 2, 4, 8 or 16: 2D points instead of D. In a noisy run, only at D = 32.
- **History:** Python since `c7c88ab`; the option description has documented the doubling since wave 0 (`0c56d86`).
- **What the code does, and why it matters:**
  - `n = ceil(log2(fes))`, and `n += 1` when `2**n == D`.
  - Nothing states a purpose. Owen's comment argues only for a power of two. From memory, the paper's initialization is `x0` plus n_init = D Sobol points; I did not check this against the text.
  - Without the doubling, the default design already has at least D points, and with `x0` that makes the D + 1 rows the search's condition needs (bads.py:1399-1403).
  - The rule compares with D, not with `fes`. So it does not guarantee more than D points (`fes = 3`, D = 5 gives 4), and user sizes jump (s6):
    - D = 8: `fes = 4` gives 4 points, `fes = 5` gives 16;
    - D = 16: `fes = 8` gives 8, `fes = 9` gives 32.
- **Consequence if real:** D extra evaluations at the start of every default run whose D is a power of two (for example 4 design points instead of 2 at D = 2, in runs of about 55 evaluations). A user-set `fun_eval_start` below D can get up to 3.6 times its value.
- **Suggested reproduction:** `s6_size.py`.
- **Test adequacy:** `test_initial_design_within_budget` checks only the cap; no test states the design's size.

### F4. `init_sobol`'s docstring misdescribes what it returns and takes
- **Location:** init_sobol.py:7-15, 35-36, 44-49, 80.
- **Category:** indexing/shape
- **Proposed classification:** port discrepancy (docstring)
- **Confidence:** high
- **Reached at default options:** yes, the call is; but bads.py:1112 discards the second value.
- **History:** `4e6a001` (2022-11-15) added the second return value.
- **What the code does, and why it is wrong:**
  - The second return value, documented as "n_samples: Number of samples", is the exponent: 2 for 4 points.
  - `fun_eval_start` is documented as "Number of initial function evaluations", but it is only the size before the rounding and doubling of F3.
  - The parameter defaults are types (`u0=np.ndarray`, and so on).
  - `lb` and `ub` are unused.
  - `plb` is described as "Lower bounds for the parameters".
- **Consequence if real:** none now. A caller that trusted the documented second value would be wrong.
- **Suggested reproduction:** `s1_seed.py` prints "second return value: 2" for 4 rows.
- **Test adequacy:** there is no test of `init_sobol`.

### F5. KD-B7-3's merge can no longer be reached (since W3-1), so the entry's stated reason no longer holds
- **Location:** function_logger.py:406-436; constraints_check.py:34-49, called at bads.py:1141, 1850 and 2237. MATLAB: private/funlogger.m:117-129 (per the sheet).
- **Category:** state/caching
- **Proposed classification:** unsure. The facts are certain; whether to keep the merge as a guard is a ruling.
- **Confidence:** high
- **Reached at default options:** no, and at no option.
  - Every recorded evaluation after the start (design, search, poll) goes through `contraints_check`. Since `149d528` it removes any candidate whose `tol_mesh/2` bin holds a logged point.
  - The only repeats left, the noise test and the final samples, take the path that does not record.
  - `add`, which could merge, is never called.
- **History:** the merge dates from PyVBMC's form (`c7c88ab`), with the row fix of 2026. W3-1 (`149d528`, part of `0d866e8`) removed the only way to reach it.
- **What the code does, and which entry it contradicts:** KD-B7-3 says that at level 2 PyBADS merges and returns the merged value, and that `add_and_update_gp` then adds that value beside the earlier row. Its reason is that returning the observation instead "changes 20 runs and worsens 16". That was measured while repeats still reached the logger: the CHANGELOG counts 100 repeats in 8800 evaluations on the noisy 3-D ellipsoid. At `0d866e8` neither form changes any run. The same holds on the PyBADS side for the entry's "At levels 0 and 1 a repeat is a new row".
- **The same staleness in the changelog.** Its Unreleased section has "Repeated points with user-specified noise", a fix measured on 54 runs "where 54 runs made such a merge". The same section also has "Points evaluated again: … neither repeats any now". Read together, the two entries say both that merges happen and that repeats no longer do.
- **Consequence if real:** no effect on results. The merge is dead code, and so is its `ValueError` guard ("More than one match"). The sheet and the changelog overstate.
- **Suggested reproduction:** `s3_merge_reach.py`. Over 3 seeded level-2 runs (164, 200 and 200 evaluations) the merge path was hit 0 times and there were 0 duplicate rows; the only row with `n_evals > 1` was the returned point's (the 10 final samples). Levels 1 and 0 were alike, except that at level 0 row 0 has 2 from the noise test.
- **Test adequacy:** the `test_record_duplicate_with_user_noise_*` tests call `_record` directly, so they pass whether or not a run can reach the merge.

### F6. `n_eff` counts the noise test but `eff_starting_points` does not
- **Location:** function_logger.py:390-402; bads.py:1055-1068, 1175; gaussian_process_train.py:1133, 1148-1160.
- **Category:** cross-module
- **Proposed classification:** unsure
- **Confidence:** high on the arithmetic, low on the intent
- **Reached at default options:** yes, at levels 0 and 1 with `uncertainty_handling=None`. Not when that option is given, and not with `specify_target_noise` (no noise test then).
- **History:** the `n_evals` increment on the path that does not record dates from `c7c88ab`/`9915bbf` (2022-11-13); the comment on the budget fraction dates from `fef6c14`.
- **What the code does, and why it is wrong:**
  - The noise test finds the start's row and sets `n_evals[0] = 2`.
  - So `n_eff = sum(n_evals[X_flag])` equals the number of rows plus 1, while `eff_starting_points` is the number of rows after the design.
  - The comment calls `x` "The fraction of the budget used after the initial design", but its numerator counts one evaluation made before the design.
- **Consequence if real:** `init_N` is computed at `x` shifted by `1/n_budget`:
  - at D = 2 (`n_budget` = 65): 123 instead of 128 at the first fit, 77 instead of 81 after 10 evaluations;
  - at D = 6: 124 instead of 128, and 93 instead of 96.

  That is a few percent fewer starting draws for the hyperparameter fit. Runs at default options and runs that set `uncertainty_handling` also differ in this respect. Small.
- **Suggested reproduction:** `s8_complex.py` (the arithmetic); `s3_merge_reach.py` shows `n_evals[0] = 2` at level 0.
- **Test adequacy:** none.

### F7. `overhead` counts the noise test's evaluation as the optimizer's time
- **Location:** bads.py:1059-1065, 1716-1720; optimize_result.py:71-72.
- **Category:** cross-module
- **Proposed classification:** possibly intentional. The code comment and `test_target_time_counts_every_evaluation_but_the_noise_test` say it follows MATLAB, which does not time the test.
- **Confidence:** high
- **Reached at default options:** yes, at levels 0 and 1.
- **History:** the restore of the total time dates from `8aecb6a` (wave 2).
- **What the code does, and why it is inconsistent:**
  - The target's time is set back after the noise test, but the run's time spans it.
  - So the noise test counts as optimizer time, although the documentation defines overhead as "taken by the optimizer, compared to function time".
  - Meanwhile `func_count` counts the test (KD-B2-6), and `n_evals[0]` and `fun_eval_time[0]` include it.
- **Consequence if real:** the overhead is overstated by one evaluation's time over the target's total. For example, 10 s per evaluation, 100 evaluations and 50 s of optimizer time report 0.06 instead of 0.05. Negligible for cheap targets.
- **Test adequacy:** the test asserts the current behaviour.

### F8. Some malformed target outputs do not raise the documented `ValueError`
- **Location:** function_logger.py:138-143, 157-179, 438-456; compare `add` at 268-277, which does check `np.isscalar(fsd)`.
- **Category:** control flow
- **Proposed classification:** port discrepancy (against the docstring)
- **Confidence:** high
- **Reached at default options:** no; only by a target with a malformed output.
- **History:** the check of the value since `9037851`, the check of the SD since `c7c88ab`.
- **What the code does (s4):**
  - At level 2 the SD is handled less generously than the value:
    - an SD given as `[0.5]` raises `TypeError`, although a value given as `[1.0]` is converted;
    - `None` or a string raises `TypeError`;
    - a two-element list raises `ValueError` with "truth value … ambiguous".
  - At any level:
    - a string value raises `TypeError`;
    - a Python complex with zero imaginary part passes `np.isreal`, then raises `TypeError` at line 450, after `Xn` and `X_max_idx` have advanced and `X` has been written (a NumPy `complex128` is stored with a `ComplexWarning`, its imaginary part dropped);
    - an array of size other than 1 raises `ValueError`, but with "Error in executing the logged function" appended, which blames the target for a run that succeeded.
- **Consequence if real:** confusing errors. The half-written row matters only to a caller that catches the error and continues.
- **Test adequacy:** `test_call_invalid_sd_value` tests only `inf`.

### F9. `finalize` is never called and leaves `n_evals` untrimmed; `reset_fun_eval_time` is dead
- **Location:** function_logger.py:283-301.
- **Category:** state/caching
- **Proposed classification:** port discrepancy
- **Confidence:** high
- **Reached at default options:** no. Nothing in `pybads/` has called either method since `c7c88ab`.
- **History:** `c7c88ab`, `8ff10f5` and `0c56d86` (the comment).
- **What the code does:**
  - After a run the log keeps its preallocated NaN rows.
  - `finalize` trims seven arrays but not `n_evals`. Afterwards `n_evals[X_flag]`, the read at gaussian_process_train.py:1133, raises `IndexError`. A later call grows the arrays to unequal lengths (X 5 rows, `n_evals` 12).
  - `reset_fun_eval_time` rebuilds `fun_eval_time` at `cache_size` rows, which is shorter than the other arrays once they have grown.
  - The docs page presents both methods as public interface.
- **Consequence if real:** none in a run. A user who calls `finalize` gets an inconsistent logger.
- **Suggested reproduction:** `s5_growth.py`.
- **Test adequacy:** `test_finalize` checks every length except that of `n_evals`.

### F10. The docstrings of `add` and `__call__` do not say that `x` is in the transformed space; `add` also diverges from `__call__` (unreached)
- **Location:** function_logger.py:79-81, 196-281.
- **Category:** indexing/shape
- **Proposed classification:** port discrepancy (docstring)
- **Confidence:** high
- **Reached at default options:** no. `fun_values` is refused (KD-B1-4), and `add` has no caller.
- **History:** `c7c88ab`.
- **What the code does, and why it is wrong:**
  - Both methods take `x` in `u` space. Their docstrings say only "the point at which the function has been evaluated" or "will be evaluated", while the `fun_values` description ("Earlier fcn evaluations with X and Y fields") implies points in the original space.
  - `add` also differs from `__call__`:
    - a value given as a one-element array is refused, where `__call__` accepts it;
    - when the logger holds SDs, a missing SD silently becomes 1;
    - when it does not, a given SD is silently dropped.
- **Consequence if real:** a future port of `fun_values` built on `add` would log the points in the wrong space.
- **Test adequacy:** `test_add_parameter_transform` uses `x = 0`, where the transform is the identity, so it cannot tell the two spaces apart.

### F11. The poll discards the return value of `period_check` (latent)
- **Location:** bads.py:2220-2225 (compare 1130-1135 and 1837-1842); period_check.py:4-6. MATLAB: utils/periodCheck.m (not read).
- **Category:** control flow
- **Proposed classification:** port discrepancy (latent)
- **Confidence:** high
- **Reached at default options:** no; the stub returns its input, and periodic variables are refused.
- **History:** `c7c88ab`.
- **What the code does:**
  - The design and the search assign the result of `period_check`; the poll does not.
  - The call sites pass different kinds of argument: the design passes the option's index list (`None`), the search and the poll `optim_state`'s boolean mask.
- **Consequence if real:** once `period_check` is implemented, it would have no effect on the poll.
- **Test adequacy:** none.

## 4. Test adequacy notes
- **The merge tests** in `test_function_logger.py` call `_record` directly, so they exercise a path that no run takes (F5).
- **Configurations BADS never builds.** `test_record_duplicate_fsd` and `test_call_noisy_function_level_1` construct a logger with `noise_flag=True` at level 1.
- **`test_finalize`** mirrors the trimming code and leaves out `n_evals` (F9).
- **`test_add_parameter_transform`** runs at `x = 0`, where the transform is the identity (F10). It is also called at module import.
- **`test_target_time_counts_every_evaluation_but_the_noise_test`** asserts the current overhead convention (F7).
- **The design.** `test_initial_design_within_budget` checks only the cap. No test checks the design's size (F3), its dependence on the seed or the start (F1), or that it is the same across platforms (F2). `test_bads_seed.py` checks only reproducibility, and would pass even though the design ignores the seed, as it does.
- **The SD check.** `test_call_invalid_sd_value` covers only `inf`; nothing tests malformed SDs (F8).
