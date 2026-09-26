<!-- Report of the B2 comparison reviewer (main loop, termination, noisy re-evaluation and final estimate, MATLAB-comparison track), wave 2 of the port review, reading PyBADS at fef6c14 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave2/B2_comparison/. Nothing in it is verified. The sandbox's clone was shallow (oldest commit ce3a0b3) when this reviewer began; it dated its lines on the complete history, fetched during its run. -->

# B2 comparison review: main loop, termination, noisy re-evaluation and final estimate

PyBADS at `fef6c14` (`/home/user/pybads-review`), MATLAB BADS at `74919c0`, gpyreg v1.3.3. Every check ran from `/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/review/B2_comparison`, and each printed `pybads.__file__` = `/home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__` = `/home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. The Python lines are dated with the full history (back to `c7c88ab`), as the orchestrator's note allowed.

## 1. Coverage

**Read completely, Python:**
- `pybads/bads/bads.py`:
  - the methods in scope: `_init_mesh_` (1010-1137), `_init_optimization_` (1139-1246), `optimize` (1248-1676), `_re_evaluate_history_` (2777-2824), `_check_mesh_overflow_`, `_log_column_headers`, `_setup_logging_display_format`, `_display_function_log_`;
  - also `_update_incumbent_`, `_update_search_bounds_`, `_eval_improvement_`, `_get_target_from_gp_`, the logger setup (224-232), and the parts of `_init_optim_state_` the loop reads (576-970).
- `_search_step_` and `_poll_step_`: read for what they return and what state they change.
- `pybads/utils/iteration_history.py`, `pybads/bads/optimize_result.py`.
- For object identity and writes to `optim_state`: `local_gp_fitting`, `get_grid_search_neighbors`, `add_and_update_gp`, `_robust_gp_fit_` and `_get_gp_training_options` in `gaussian_process_train.py`.
- `FunctionLogger.__call__` and `_record`; `init_sobol.py`.
- Every option the slice reads, in both `.ini` files.

**Read completely, MATLAB:**
- `bads.m` 140-290 (the defaults), 300-330, and 405-1194. The bodies of the search (511-740) and the poll (767-1046) were read only for the state they hand to the loop.
- The subfunctions `EvalImprovement`, `UpdateIncumbent`, `UpdateTarget`, `UpdateSearch` (tail), `reevaluateIterList`, `updateSearchBounds`, `meshOverflowCheck`, `FinalEstimate`.
- `private/evalinitmesh.m`.
- `private/funlogger.m` ('iter'/'single'), `private/gpupdate.m` ('add', 'nearest', the rebuild), `utils/gppred.m`.
- `private/setupvars.m`: `FunValues` and the state set up there; `private/bads_output.m` 20-25.

**Skimmed:** the update branch of `gpdefBads.m`, `uCheck.m`, `force2grid.m`, the centre of `searchES.m`, and the tests under `pybads/testing/bads/` that touch the slice.

**Not reached:**
- `fixedbads.m`/`expandvars` (KD-B1-7), `scatterplot.m` (KD-B2-2), restarts (KD-B2-1), the Sto-BADS branches (KD-S-1).
- The search and poll algorithms themselves (slices B3, B4) and the GP internals (B5, B6).

**Checked, no finding:**
- `optimize` throws away `_poll_step_`'s return value (1410). This is harmless: the poll never binds `gp` to another object. `local_gp_fitting` restores the GP in place, and `_robust_gp_fit_` and `add_and_update_gp` return the object they were given.

**Observation, not a finding:**
- `IterationHistory.record` grows its array one slot at a time through `__setitem__`, which deep-copies the whole array. Every record therefore copies again every object already stored, the GPs included: 200 records made 20100 deep copies (`ih_copy.py`). This costs time only; it did not matter in a 200-evaluation run (`ih_time.py`).

## 2. Answers to the first questions

### Q1. The initialization

**Evaluation of `x0` and the noise test**
- `x0` is evaluated as in `evalinitmesh.m:29-35`.
- The noise test runs only when `uncertainty_handling` is left empty (`bads.py:1030`, since W0-17), as in MATLAB (`evalinitmesh.m:38-50`).
- `specify_target_noise` sets the option to True in `__init__` (`bads.py:827-834`), so level 2 never takes the test, as in MATLAB (`evalinitmesh.m:11-18`).
- The second evaluation counts in `func_count` and is not stored (`record_duplicate_data=False`), like MATLAB's direct `funwrapper` call plus `funccount+1`.
  - PyBADS also increments that row's `n_evals`, which `_get_gp_training_options` reads as `n_eff` (a B5 effect with no MATLAB counterpart).
- **The threshold differs (F1):** PyBADS uses eps·`tol_fun`, MATLAB sqrt(eps)·`TolFun`.

**What a noisy result changes**
- Level 1, and `fun_eval_start = min(max(20, ·), max_fun_evals)`, as MATLAB's `Ninit` (`evalinitmesh.m:93-95`). MATLAB changes a local copy; PyBADS changes the option for good, but nothing reads it afterwards.
- The GP noise model set before the test (`gp_noisefun` `[1,0,0]`) is equivalent to the level-1 one (`[1,2,0]`): gpyreg ignores `scale_user_provided` without `user_provided_add`.

**Size of the initial design**
- The cap `min(fun_eval_start, max_fun_evals - 1)` is MATLAB's formula (`evalinitmesh.m:101`).
- **But `init_sobol` rounds up to a power of two after the cap (F2),** so the budget can be exceeded. The power-of-two size and the doubling are settled (KD-B7-1, C2); the cap is not.

**Incumbent after the design**
- The argmin of the observations, ties to the first, as `evalinitmesh.m:120-123`.
- PyBADS takes the argmin over the whole function logger. That would also cover imported points if `fun_values` worked (F12); today it cannot be reached.

**Options a noisy run changes**
- Same set, amounts and order as `bads.m:431-445`:
  - `tol_stall_iters` doubled;
  - `n_train_max` at least 200;
  - `n_train_min` doubled;
  - `mesh_overflow_warning` doubled;
  - `min_failed_poll_steps` = inf;
  - `mesh_noise_multiplier` = 0;
  - `noise_size` defaults to 1 (and is forced to 1 at level 2, KD-B5-8);
  - the final samples reserved as `min(noise_final_samples, max_fun_evals - func_count)`, then taken off `max_fun_evals`.
- A deterministic run gets `noise_size = sqrt(tol_fun)`, as in MATLAB; PyBADS also switches Sto-BADS off (KD-S-1).
- After F2's overshoot the reserve can go negative and *raise* `max_fun_evals`.

**Incumbent's `fval` and `fsd` at the start**
- `fval` is the best observation.
- `fsd` is 0, `noise_size[0]`, or `S` at the argmin of `Y`, as `bads.m:446-457`.

**Other**
- `max_fun_evals = 1` gives 2 evaluations with the test, as in MATLAB.
- `output_fcn('init')` is called after the option changes and the first GP fit; MATLAB calls it before them (F11).

### Q2. The loop and its termination

**Head of the loop** (`bads.py:1313-1349` vs `bads.m:486-507`): same mesh size, search size integer (when locked), search mesh size, search bounds and sufficient improvement. With `sloppy_improvement=False` the run crashes at the first pass (F6).

**When a search runs**
- Condition: `search_count < search_n_try` and more than D points. PyBADS counts the points in the logger; MATLAB counts the GP's training set (`size(gpstruct.y,1)`, although its comment says "stored points").
- The two are equivalent at default options. The training set holds at least `min(N, max(n_train_min, n_train_max - buffer_ntrain))` points, which exceeds D whenever N does. They differ only with user-set training-set sizes of at most D.
- `search_count` starts at `search_n_try`, so the first iteration is poll-only, as in `setupvars.m:173`.

**When a poll runs:** the poll decision (`search_count` 0 or n_try, `skip_poll_after_search`, the spree and its mesh expansion) and `u = ubest` before the poll match `bads.m:743-769`.

**Termination criteria:** same order and conditions at the end of every pass, and a later criterion overrides an earlier message, as in MATLAB. With Python's 0-based `poll_iteration` = MATLAB `iter` − 1:
- `max_fun_evals`: `func_count >= max_fun_evals`.
- `max_iter`: `poll_iteration >= max_iter - 1` ⇔ `iter >= MaxIter`. It is also checked on search-only passes, so a run ends at the first pass of round `max_iter`, as in MATLAB.
- `tol_mesh`: the poll mesh size after the poll.
- Stall: `poll_iteration > T - 1` with index `poll_iteration - T`. This reads MATLAB's `iterList(iter - T)` (1-based): the re-estimated value recorded T rounds earlier, compared with the current incumbent through the quantile improvement.
- Messages: equivalent texts. The output function's stop has a message of its own (F11).

**What is recorded:** after a poll or on termination, at index `poll_iteration`: `u`, `yval`, `fval`, `fsd` and the hyperparameters (plus extras: `x`, the mesh sizes, a GP copy, `func_count`). Same moment and content as `bads.m:1088-1094`.

**Iteration count:** it grows after a poll that does not end the run, as in MATLAB.

**Where they differ:**
- The accelerated mesh reduction, in the poll body, is tested one iteration later than in MATLAB (F3).
- The display: `display = "notify"`/`"final"` act as `"iter"` (F7); `f_vals` crashes the display (F8); the Actions column can show stale or merged actions (F9).
- A run that ends in its initialization reports `iterations = 0` where MATLAB reports 1 (F10).

### Q3. Noisy runs: the re-estimation, the incumbent and the final estimate

**The re-estimation, `_re_evaluate_history_`, matches `reevaluateIterList`:**
- Same condition: level > 0, a poll pass, from the second iteration.
- It covers all recorded iterates (as many rows as MATLAB's `optimState.iter`).
- It works on one copy of the working GP, carried from one iterate to the next as MATLAB's local `gpstruct` is.
- Each iterate uses the hyperparameters recorded at the end of its iteration.
- The rebuild takes the nearest neighbours of each iterate among all logged points, using the working GP's `len_scale`/`effective_radius`. MATLAB's `gpupdate` without a refit also keeps `lenscale`/`effectiveradius`.
- No refit and no random draws; latent predictions; skipped when nothing was evaluated since the last one (`lastreeval`).
- Its write to `optim_state["ntrain"]` is rewritten before anything reads it.
- **On a failed rebuild** it records NaN, as MATLAB (whose `gppred` retries the same inference and gives NaN), **except for the current iterate, which keeps its estimate (F5).**

**Moving the incumbent after the re-estimation:** the same as `bads.m:1111-1117`, including MATLAB's partial move: `u` moves and `ubest` does not (F4).

**The final choice and estimate:**
- Same condition as `bads.m:1138`, and the re-estimation runs again.
- The iterate is chosen by `q_beta = fval + sqrt(2)·erfcinv(2·final_quantile)·fsd` (3.09·fsd at 1e-3), the minimum over iterates 2..n, skipping NaN (`nanargmin`, as MATLAB's `min`). An all-NaN slice cannot occur, because the last iterate keeps its estimate.
- `noise_final_samples` evaluations at the chosen point, not stored:
  - level 1: the mean and its SEM (ddof = 1); with one sample, the iterate's biased `yval` is added;
  - level 2: the precision-weighted mean and `1/sqrt(total precision)`, without `yval`.
- The estimate is recorded at the chosen iterate, and `x`, `fval`, `fsd` and `yval_vec`/`ysd_vec` reach the result under the KD-B1-8 conventions.
- Differences: MATLAB takes the samples only when `fval` is requested (`nargout > 1`), PyBADS always. With one sample at level 1, `yval_vec` has shape (2, 1) (F13).
- A run that ends within its first iteration leaves the reserved evaluations unused, on both sides.
- The formulas were checked against returned samples at levels 1 and 2 (`final_est.py`).

## 3. Findings

### F1. The noise test's threshold `tol_noise` is eps·`tol_fun`, not MATLAB's sqrt(eps)·`TolFun`
- Location: `pybads/bads/option_configs/advanced_bads_options.ini:13`, read at `pybads/bads/bads.py:1037`; MATLAB: `bads.m:195`, `private/evalinitmesh.m:43`.
- Category: defaults
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes. The test runs at default (`uncertainty_handling = None`), at level 0 before it decides. Its outcome differs only for a target whose two evaluations at `x0` differ by more than 2.2e-19 and at most 1.5e-11.
- History: MATLAB's line is unchanged since `fd3f7a2` (2017-03-29). The Python value (`tolnoise = np.spacing(1.0) * self.get("tolfun")`) dates from `c7c88ab` (2022-06-02). The two never agreed.
- What the code does:
  - PyBADS declares a target noisy when the repeat at `x0` differs by more than 2.2e-19, about one thousandth of an ulp at |f| ≈ 1.
  - MATLAB's threshold is 1.49e-11.
  - The option's description is MATLAB's ("Min variab[i]lity for a fcn to be considered noisy"). No record explains the change.
- Consequence if real:
  - A deterministic target with last-bit nondeterminism is run as a noisy one: for example, a summation whose order varies between calls, multithreaded reductions, or GPU arithmetic.
  - That means at least 20 initial points (32 after rounding), the stall window doubled, at least 100 training points, 10 final samples reserved, a GP that infers noise on noise-free data, and a result reported as "stochastic".
  - MATLAB treats such targets as deterministic, unless |f| is large enough (around 1e5) for one ulp to exceed its own threshold.
- Suggested reproduction: `tol_noise2.py`, a sphere + 0.1 that adds one ulp on alternate calls, D = 2, 100 evaluations.
  - The two calls differ by 1.1e-16, which is above PyBADS's threshold and below MATLAB's.
  - PyBADS reports `target_type: stochastic`, with a 33-point start, `tol_stall_iters` 10 and `n_train_min` 100; its first GP fit logs a Cholesky failure.
- Test adequacy: no test covers the threshold. `test_declared_deterministic_target_takes_no_noise_test` only checks when the test runs.

### F2. The initial design ignores the budget cap, so a run can exceed `max_fun_evals`; a noisy run's reserve for the final samples then goes negative and raises the budget
- Location: `pybads/bads/bads.py:1075-1088` with `pybads/init_functions/init_sobol.py:73-76`; `bads.py:1175-1184`. MATLAB: `private/evalinitmesh.m:93-104`, `bads.m:441-442`.
- Category: control flow
- Proposed classification: port discrepancy. The part that does not count the noise-test evaluation is a suspected defect in both.
- Confidence: high
- Reached at default options: no. It needs a small `max_fun_evals`:
  - deterministic: `max_fun_evals` below about 2 + 2^⌈log2 D⌉ (twice that when the rounding gives D);
  - noisy: `max_fun_evals` below 34 for D ≤ 32.
  - Levels 0 and 1 are reached; level 2 has no noise test, but the rounding overshoot is the same.
- History:
  - MATLAB's lines date from 2017, apart from 75ec49f's wording.
  - PyBADS has passed the capped number to a rounding `init_sobol` since `c7c88ab` (2022-06-02; `ninit = np.minimum(ninit, maxfunevals - 1)`, then `random_base2(ceil(log2(ninit)))`).
  - The two never agreed.
- What the code does:
  - PyBADS caps the design at `max_fun_evals - 1`, as MATLAB does, but `init_sobol` then draws 2^⌈log2 n⌉ points, doubled when that equals D. The cap no longer holds.
  - `noise_final_samples = min(nfs, max_fun_evals - func_count)` can then go negative, and `max_fun_evals -= nfs` *raises* the budget.
  - Both sides size the design as if `x0` were the only evaluation before it, forgetting the noise test, so each overshoots by one when `max_fun_evals ≤ fun_eval_start + 1`.
  - KD-B7-1 settles the power-of-two size, not the cap.
- Consequence if real: a user budget smaller than the design is exceeded, and a noisy run loses its final estimate. Examples from `budget_init.py`:

| Run | PyBADS | MATLAB |
|---|---|---|
| D = 2, deterministic, `max_fun_evals=3` | 6 evaluations | 4 |
| D = 3, deterministic, `max_fun_evals=4` | 6 evaluations | 5 |
| D = 2, noisy, `max_fun_evals=25` | 34 evaluations; `noise_final_samples` −9, `max_fun_evals` raised to 34, no final samples | 22 evaluations, 3 final samples reserved |
| D = 2, noisy, `max_fun_evals=40` | ends after the design at 34 evaluations, without a single poll evaluation or final sample | 20-point design, 10 final samples |

- Suggested reproduction: `budget_init.py` (output as above).
- Test adequacy: only `max_fun_evals=1` is tested (`test_one_function_evaluation`).

### F3. The accelerated mesh reduction is tested from one iteration later than in MATLAB
- Location: `pybads/bads/bads.py:2422-2426` (the poll body, reading `iteration_history`); MATLAB: `bads.m:976-982`.
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes (`accelerate_mesh = True`, `accelerate_mesh_steps = 3`), at every level.
- History:
  - MATLAB's lines are unchanged since 2017 (`0c2bb80`, `7dcdc2b`).
  - PyBADS has had `iter > steps` with the 0-based `iter` since `c7c88ab` (2022-06-02). In `8e59038` the same off-by-one was fixed for the stall criterion (`> tolstalliters - 1`), and later for `max_iter`, but never here.
- What the code does:
  - The index `iter - steps` points at the same iterate as MATLAB's.
  - The condition `optim_state["iter"] > steps` (0-based) is MATLAB's `iter > steps + 1`. At MATLAB's iteration 4, a failed poll is not compared with iteration 1, although the history already holds that row.
- Consequence if real: in that one iteration, when the poll fails and the improvement since iteration 1 is below `tol_fun`, MATLAB halves the mesh once more and PyBADS does not. The next poll then runs at twice the mesh size. The effect is small but reached by default runs.
- Suggested reproduction: `accel_mesh.py`.
  - Polls failed at 0-based iterations 0-11. The test ran at 4-11.
  - MATLAB's condition would run it at 3-11.
- Test adequacy: none. `test_bads_logger` only checks that the stalling message appears.

### F4. The re-estimation's move to a better iterate moves the search centre and the incumbent's value, but not the incumbent (both sides)
- Location: `pybads/bads/bads.py:1531-1541` (it sets `self.u`, and `self.best_u`, an attribute nothing reads), then `1406` (`self.u = self.u_best`); MATLAB: `bads.m:1111-1117` (sets `u`, not `ubest`), `769` (`u = ubest`), `539` (`UpdateTarget(ubest, ...)`).
- Category: state/caching
- Proposed classification: suspected defect in both
- Confidence: medium (the Python mirrors MATLAB; whether MATLAB meant it is not recorded)
- Reached at default options: yes, at levels 1 and 2, whenever a re-estimated iterate beats the current one by more than `tol_fun`.
  - In 6 seeded runs of a noisy sphere (D = 2, 200 evaluations, 12-16 iterations), each run had 5 to 9 search steps that started with `u ≠ u_best` (`noisy_move.py`).
- History:
  - MATLAB's block dates from 2017; `75ec49f` (2022-05-09) only added "skip first".
  - The Python line was written in `c7c88ab` (2022-06-02) as `self.best_u = self.u.copy() # TODO in Matlab is not done`. The comment was dropped in `8e59038`. The line has matched MATLAB's behaviour since, only because `best_u` is a dead attribute.
- What the code does:
  - The incumbent's `yval`, `fval`, `fsd` and target hyperparameters become the chosen iterate's.
  - Only `u` moves to that iterate, so the next round's searches centre there, with the GP rebuilt around it.
  - The incumbent `u_best`/`ubest` stays where it was. The target is predicted there with the chosen iterate's hyperparameters, and `udist` is measured from there.
  - If no search succeeds, the poll is around the old point, whose improvement is measured against another point's value. The round then records the old location with that value until the next re-estimate replaces it.
  - `optim_state["u"]`/`["fval"]` (MATLAB `optimState.u`/`.fval`, read by the target's fallback) are not updated either.
  - A move of the incumbent would set `ubest` too.
- Consequence if real: the incumbent's location and its value disagree for up to a round after each move.
  - A variant that also moves `u_best` changed the final error of 6 of 10 seeded runs (noisy sphere, D = 2, 150 evaluations).
  - The medians were 0.0430 and 0.0432 (`move_variant.py`); a direction is not established.
- Suggested reproduction: `noisy_move.py`. In seed 0, round 6: the search was centred at [0.097, 0.148], the incumbent and that round's poll at [0.158, 0.154], and the round recorded [0.158, 0.154].
- Test adequacy: `test_iteration_history_keeps_the_gps_as_recorded` checks only the GPs; nothing checks where the move puts the incumbent.

### F5. A failed rebuild of the current iterate during the re-estimate keeps its estimate, where MATLAB gives NaN
- Location: `pybads/bads/bads.py:2811-2815`; MATLAB: `bads.m:1378-1412` with `private/gpupdate.m:340-354` and `utils/gppred.m:39-54`, `82-88` (with an empty `post`, `gppred` re-runs the same inference inside `try`, which leaves NaN).
- Category: control flow
- Proposed classification: possibly intentional. W1-35 in `fef6c14` (2026-09-26) is described by the docstring, the commit message and `test_noisy_re_estimate_after_failed_rebuild`, but no known-differences entry lists it.
- Confidence: high
- Reached at default options: only when a rebuild raises `LinAlgError` at the current iterate, at levels 1 and 2.
- History: MATLAB is unchanged. The Python line is from `fef6c14`; before it, the NaN (from `e004c79`) could crash the choice.
- What the code does:
  - MATLAB would give the incumbent a NaN value and SD. No search or poll could then count as an improvement until the next re-estimate, the stall criterion could not fire, and the final choice would skip the iterate.
  - PyBADS keeps the estimate recorded at the end of the iteration.
- Consequence if real: the two diverge in that rare path; PyBADS's choice keeps the run working.
- Suggested reproduction: the existing test, which injects the failure.
- Test adequacy: the test pins the departure.

### F6. `sloppy_improvement=False` crashes at the first pass of the loop
- Location: `pybads/bads/bads.py:1339-1349`; MATLAB: `bads.m:504-510`.
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no (`sloppy_improvement=False`), at every level.
- History: the `.copy()` line dates from `c7c88ab` (2022-06-02); MATLAB's lines from 2017.
- What the code does: without `np.maximum`, `tol_improvement * mesh_size**1.5` is a Python float, and `.copy()` raises `AttributeError`. MATLAB supports the option.
- Consequence if real: the option cannot be used.
- Suggested reproduction: `sloppy.py`, which fails with `AttributeError: 'float' object has no attribute 'copy'` at line 1349.
- Test adequacy: none.

### F7. `display = "notify"` and `display = "final"` print every iteration
- Location: `pybads/bads/bads.py:224-232`; `basic_bads_options.ini:2-3` lists the values. MATLAB: `bads.m:317-328`, and the `prnt` tests at `evalinitmesh.m:52`, `bads.m:1172`.
- Category: control flow
- Proposed classification: port discrepancy. KD-B2-3 settles the mechanism and leaves the content open.
- Confidence: high
- Reached at default options: no (the default is `"iter"`).
- History: in the code since `c7c88ab` (2022-06-02).
- What the code does: both values fall through to the INFO level. MATLAB's `'notify'` prints only the opening line; `'final'` adds the final message.
- Consequence if real: the display shows every iteration whether or not the user asked for less.
- Suggested reproduction: `display_levels.py`, where "notify", "final" and "iter" each log 20 messages.
- Test adequacy: none.

### F8. A run with the `f_vals` option crashes at its first display
- Location: `pybads/bads/bads.py:2840-2869` (with `cache_active` the format has 8 fields), `2871-2894` (it gets 6 or 7 arguments), `602-604`; MATLAB: no counterpart (`f_vals` is PyBADS-only, KD-B1-4).
- Category: control flow
- Proposed classification: port discrepancy (a PyBADS-only option that cannot be used)
- Confidence: high
- Reached at default options: no (`f_vals` given).
- History: the cache format dates from `c7c88ab`/`8e59038` (last changed in `ba65e15`, 2022-11-10), `_display_function_log_` from `9037851`.
- What the code does:
  - `f_vals` only fills a cache and turns `cache_active` on, and nothing else reads the cache.
  - `_display_function_log_` then fills a string into a `{:12.6f}` field and raises, in `_init_mesh_`, even with `display="off"`: the string is formatted before it reaches the logger.
- Consequence if real: the option always crashes the run.
- Suggested reproduction: `fvals.py`, which raises `ValueError: Unknown format code 'f' for object of type 'str'` at `bads.py:2886`.
- Test adequacy: none.

### F9. The display's Actions column shows only the last action appended
- Location: `pybads/bads/bads.py:2473-2484`, `2892` (`logging_action[-1]`); MATLAB: `bads.m:1021-1028` (`action` rebuilt at each poll: "Train", "(failed)", ", skip").
- Category: control flow
- Proposed classification: port discrepancy (display only)
- Confidence: high
- Reached at default options: yes, with display "iter", in some iterations.
- History: since `c7c88ab`/`8e59038`.
- What the code does:
  - A poll that neither trains nor skips shows whatever was appended last. That is a stale "Train" when no search printed in between.
  - A poll that trains and skips shows only "Skip".
- Consequence if real: the display misreports the actions of some iterations.
- Suggested reproduction: `display_actions4.py`, with refits after the first few calls suppressed and `search_n_try=0`. Iterations 6-13 show "Train" without training.
- Test adequacy: none.

### F10. A run that ends in its initialization reports `iterations = 0`; MATLAB reports 1
- Location: `pybads/bads/optimize_result.py:120-121` with `bads.py:820`; MATLAB: `bads.m:482` (`iter = 1` before the loop), `private/bads_output.m:21`.
- Category: indexing/shape
- Proposed classification: possibly intentional (tests pin 0). It contradicts KD-B1-8's "counts as MATLAB's `output.iterations` does, from 1" for this case.
- Confidence: high
- Reached at default options: no. It needs `max_fun_evals=1` or an `output_fcn` that stops at 'init'.
- History: `95da7f1` (#71).
- What the code does: PyBADS reports 0 iterations for a run that never enters the loop; MATLAB reports 1.
- Consequence if real: the count differs by one in those runs only.
- Suggested reproduction: `one_eval.py`, which reports `iterations 0` in the three cases.
- Test adequacy: `test_one_function_evaluation` and `test_output_fcn_stops_run_at_init` assert 0.

### F11. The output function differs from MATLAB in timing, message and the effect of a false return
- Location: `pybads/bads/bads.py:1287-1300`, `1411-1429`; MATLAB: `bads.m:424-428`, `1037-1039`.
- Category: control flow
- Proposed classification: possibly intentional (#71)
- Confidence: high
- Reached at default options: no (`output_fcn` given).
- History: `95da7f1` (2026-09-26).
- What the code does:
  - The call at 'init' comes after the noisy option changes and the first GP fit; MATLAB calls it before the changes.
  - A stop gets "Optimization terminated by options['output_fcn']." MATLAB keeps the stale "…after initialization." message.
  - A false return at 'init' cannot reopen a run that ended in its initialization; MATLAB assigns the return value to `isFinished_flag`, so it can.
- Consequence if real: better messages in PyBADS; the `optim_state` passed at 'init' differs from MATLAB's.
- Suggested reproduction: `test_output_fcn_stops_run_at_init`.
- Test adequacy: tests pin PyBADS's behaviour.

### F12. `fun_values` cannot be used; if it worked, the initial incumbent would be chosen among the imported points too
- Location: `pybads/bads/bads.py:755-760` (`np.isreal` on arrays in a boolean test), `787` (`range(len())`), `286` against `290` (`_init_optim_state_` runs before `self.function_logger` exists), and `1114-1116` (argmin over the whole logger). MATLAB: `private/setupvars.m:127-165`, `evalinitmesh.m:120-123` (minimum over `x0` and the design only).
- Category: cross-module
- Proposed classification: port discrepancy. The setup code is B1's; the choice of the incumbent is this slice's.
- Confidence: high
- Reached at default options: no (`fun_values` given).
- History: the lines date from `c7c88ab`/`8e59038`; MATLAB's from 2017.
- What the code does: `fun_values` raises `ValueError` ("truth value of an array … ambiguous") at line 755. Once repaired, `_init_mesh_` would take its argmin over the imported points too, where MATLAB puts them into the GP's data only.
- Consequence if real: the option always crashes the run today.
- Suggested reproduction: `fvals.py`.
- Test adequacy: none.

### F13. With one final sample at level 1, `yval_vec` has shape (2, 1)
- Location: `pybads/bads/bads.py:1608-1612` (`np.vstack`); MATLAB: `bads.m:1464-1466` (`[yval_vec, yval]`, a 1×2 row like the other cases).
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no (`noise_final_samples=1`, level 1).
- History: since `c7c88ab`.
- What the code does: that result's `yval_vec` has shape (2, 1), where every other case gives (n,). The estimate itself is right.
- Consequence if real: a script that indexes `yval_vec` gets an array where it expects a number.
- Suggested reproduction: `final_est.py`.
- Test adequacy: `test_final_estimate_from_one_sample` covers only level 2.

## 4. Test adequacy notes

- **`test_noisy_runs.py`, the final-estimate tests** (`…weights_samples_by_precision`, `…from_one_sample`, `…without_target_noise`) recompute the implementation's formula from the returned samples. That formula matches MATLAB's `FinalEstimate`, but no test checks:
  - which iterate `final_quantile` chooses, against a transcription;
  - the move of the incumbent after the in-loop re-estimate (F4);
  - the one-sample, level-1 case (F13).
- **`test_run_control.py`:**
  - `test_one_function_evaluation` and `test_output_fcn_stops_run_at_init` pin `iterations == 0` (the implementation's convention, F10).
  - `test_iterations_count_from_one` claims MATLAB's count, and covers only runs that end on `max_iter`.
- **`test_gp_update_failures.py::test_noisy_re_estimate_after_failed_rebuild`** pins W1-35's departure from MATLAB (F5).
- **No test covers:**
  - the noise test's threshold (F1);
  - the design size against the budget (F2);
  - when the accelerated mesh reduction starts (F3);
  - the display levels, the Actions column or `f_vals` (F7-F9);
  - `sloppy_improvement=False` (F6);
  - `fun_values` (F12).
- **`test_iteration_history.py`** tests the container only, not the repeated deep copies made as it grows.
