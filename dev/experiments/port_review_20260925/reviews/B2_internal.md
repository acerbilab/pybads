<!-- Report of the B2 internal reviewer (main loop, termination, noisy re-evaluation and final estimate, internal-correctness track), wave 2 of the port review, reading PyBADS at fef6c14 in ../pybads-review and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave2/B2_internal/. Nothing in it is verified. The sandbox's clone was shallow while this reviewer ran (its oldest commit ce3a0b3, 2022-11-22), so its "predates ce3a0b3" dates no line more precisely. -->

# B2 internal review: main loop, termination, noisy re-evaluation and final estimate

## 1. Coverage

**Read completely** (PyBADS at `fef6c14`, `/home/user/pybads-review`):
- `pybads/bads/bads.py`: `_init_mesh_` (1010-1137), `_init_optimization_` (1139-1246), `optimize` (1248-1676), `_re_evaluate_history_` (2777-2824), `_check_mesh_overflow_` (2826-2834), the display (`_log_column_headers`, `_setup_logging_display_format`, `_display_function_log_`, 2836-2894) and the messages of `optimize`.
- Also in `bads.py`, for what the loop depends on: `_init_optim_state_` (576-970), `_eval_improvement_`, `_update_incumbent_`, `_get_target_from_gp_`, `_search_step_`, `_poll_step_`, `_is_gp_refit_time_`, `_is_poll_stop_`, `_record_gp_refit_`.
- `pybads/utils/iteration_history.py`, `pybads/bads/optimize_result.py`, both `.ini` files, `pybads/init_functions/init_sobol.py`.
- `FunctionLogger.__call__` and `_record`.
- In `gaussian_process_train.py`: `init_and_train_gp`, `local_gp_fitting`, `_robust_gp_fit_`, `add_and_update_gp`, `get_grid_search_neighbors`, `_get_gp_training_options`.
- gpyreg `GaussianNoise.__init__` and `hyperparameter_count`.
- The known-differences sheet and the counterpart map.
- Tests: `test_noisy_runs.py`, the relevant tests of `test_run_control.py`, `test_bads_logger.py`, and the test list of `test_iteration_history.py`.

**Skimmed:** `__init__`, `_bounds_check_`, the CHANGELOG `[Unreleased]` entries on these topics, `docsrc/source` (quickstart, options page, OptimizeResult and IterationHistory pages), and the text of example notebook 3 about `fval` and `fsd`.

**Not reached:**
- The MATLAB code (not opened, per the track).
- The BADS paper, which is not in the environment. My references to it come from memory and are marked as such.
- `bads_dump.py`, and the ES search internals (slice B3).
- The git history: the worktree is a shallow clone whose root is `ce3a0b3` (2022-11-22). "Predates `ce3a0b3`" below means written by that date.

All scripts and outputs are in `/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/review/B2_internal`. Every script printed the review worktree's `pybads` and the gpyreg v1.3.3 clone.

## 2. Answers to the first questions

### Q1. The initialization

**Evaluations of `x0` and the noise test**
- `x0` is evaluated once, at its gridded `u0` (1023).
- The noise test runs only when `uncertainty_handling` is `None`, the default (1030-1041). A `specify_target_noise` run has already set the option to `True` (830-834), so it takes no test.
- The second evaluation is not recorded in the training data (`record_duplicate_data=False`), but it counts in `func_count` (function_logger.py:192, 384-398).
- The threshold is `|y1 - y2| > tol_noise = eps*tol_fun`, about 2.2e-19, as the option's description says.
- A noisy result changes two things:
  - the uncertainty level becomes 1;
  - `fun_eval_start` becomes `min(max(20, fun_eval_start), max_fun_evals)` (1067-1071).
- Everything else changes after the design, in `_init_optimization_`.
- `optim_state["gp_noisefun"]` keeps the level-0 value `[1,0,0]`, set in `_init_optim_state_` before the test, instead of the level-1 value `[1,2,0]`. This has no effect: in gpyreg 1.3.3, `scale_user_provided` without `user_provided_add` adds nothing (noise_functions.py:43-48, 58-61).
- With `max_fun_evals=1`, the run returns after these one or two evaluations, as the CHANGELOG documents.

**Size of the initial design**
- The design has `2**ceil(log2(min(fun_eval_start, max_fun_evals-1)))` points, doubled when that equals D (init_sobol.py:73-76). At D < 32, a noisy run gets 32 points.
- This matches the description of `fun_eval_start`, except that the budget cap is applied before the rounding and does not count the noise test (F3).
- The points are forced to the search grid and filtered by `contraints_check`, so fewer may be evaluated.

**Incumbent after the design**
- The incumbent is the argmin of every observation so far, `x0` included; at level 2 these are the merged values (1114-1119). Its `fval` is that observation.

**Options a noisy run changes** (after the design, 1155-1184, in this order):
- `tol_stall_iters` ×2.
- `n_train_max` = max(200, ·).
- `n_train_min` ×2.
- `mesh_overflow_warning` ×2.
- `min_failed_poll_steps` = inf (already the default).
- `mesh_noise_multiplier` = 0.
- `noise_size` = 1.0 if it is `None` or at level 2 (KD-B5-8).
- `noise_final_samples` = min(nfs, `max_fun_evals` − `func_count`), then `max_fun_evals` −= nfs. This reservation can go negative (F3).

Each of these matches the option's description where one exists. The doubling of `mesh_overflow_warning` is not documented.

**The incumbent's `fval` and `fsd` at the start**
- Level 0: the raw minimum and 0; `noise_size` defaults to sqrt(`tol_fun`).
- Level 1: the raw minimum and `noise_size[0]`, which is 1.0 by default.
- Level 2: the merged minimum and the SD the target returned there.

The raw minimum stays the noisy incumbent's value through two iterations (F10). The display prints `fsd` as `nan` on the two iteration-0 lines of noisy runs, because `fsd` is set after them. The docstrings of `_init_mesh_` and `_init_optimization_` are accurate, if loose.

### Q2. The loop and its termination

**Head of each pass (1312-1349)**
- Mesh size: `2**msi`.
- With `search_size_locked`, the search mesh is `2**min(0, 2*msi-10)`, and the search bounds are recomputed.
- Sufficient improvement: `tol_improvement*mesh**forcing_exponent`, raised to at least `tol_fun` under `sloppy_improvement`. With `sloppy_improvement=False` the run crashes (F4).

**When a search and a poll run**
- A search runs when `search_count < search_n_try` and more than D points exist. The first iteration has none, because the count starts at `search_n_try`.
- The poll decision is taken when `search_count` is 0 or `search_n_try`.
  - With `skip_poll_after_search` and a successful search in the round, there is no poll: the spree count goes up, and every `search_mesh_expand` rounds (0, off, by default) the mesh expands, after the overflow check.
  - Otherwise the poll runs and the spree resets.
- An iteration is a series of rounds that a poll ends; the counter moves only after a poll.
- The loop discards the `gp` that `_poll_step_` returns. This is harmless: `local_gp_fitting` and `add_and_update_gp` modify the GP in place and return the same object.
- Line 1406 (`self.u = self.u_best`) matters only after a move by the re-evaluation, and it undoes that move (F1).
- Line 1425 resets `best_gp_hyp` at every pass (F2).

**What is recorded (1469-1500)**
- Records are made at each poll and at termination, at index `poll_iteration`: `u`, `x`, `yval`, `fval`, `fsd`, `gp_hyp_full`, a deep copy of `gp`, and `func_count`.
- `mesh_size` is the mesh after the poll, i.e. the next iteration's, while `search_mesh_size` is the one used during the iteration.
- `init_N` and `ntrain` are recorded by refits.
- Seven declared keys are never recorded: `iter`, `lcbmax`, `Ns_gp`, `timer`, `optim_state`, `n_eff`, `logging_action`.

**Termination (checked at every pass, 1427-1467)**
- `func_count >= max_fun_evals` (the reduced value at levels 1 and 2). It can already be exceeded by the design (F3).
- `poll_iteration >= max_iter-1` (F9).
- `mesh_size < tol_mesh`. The tolerance is rounded up to 2^-19 for 1e-6; since the mesh is a power of 2, this ends at the same mesh as the raw value would.
- The stall criterion:
  - It applies for iteration k ≥ T, against the base at index k−T, with the improvement taken at `improvement_quantile`, below `tol_fun`.
  - The index reads a recorded iterate T entries back, as it should.
  - Because the check runs at every pass, it fires after T−1 complete iterations and one search (F9).
- `output_fcn` is called at `"init"`, after each poll and at `"done"`, as its description says. Its message is replaced by any other criterion met in the same pass; the precedence is stall > `tol_mesh` > `max_iter` > `max_fun_evals` > `output_fcn`.
- `exit_flag=2` is dead code.
- Each message matches its criterion.
- The mesh acceleration inside the poll indexes the history with a stricter guard than the stall criterion does (F13, slice B4).

### Q3. Noisy runs: the re-estimation, the incumbent and the final estimate

**`_re_evaluate_history_`**
- It re-estimates every recorded iterate, index 0 and the current one included.
- It uses one deep copy of the working GP. For each iterate it sets the hyperparameters recorded at the end of that iteration, which are the working GP's at that moment.
- It rebuilds the copy without a refit, from the nearest neighbours of `u_i` in all current data, and predicts the latent mean and SD at `u_i`.
- The neighbours are chosen with the working GP's `len_scale` and `effective_radius`, not with iterate i's.
- If a rebuild fails, the iterate gets NaN, except the last one, which keeps its recorded estimate.
- The call is skipped when no evaluation has been made since the last one.
- All of this matches the docstring. It also overwrites `optim_state["ntrain"]`, which only feeds a diagnostic.

**End of the iteration (1502-1541)**
- The current iterate's re-estimate becomes the incumbent's.
- The improvement of every iterate from index 1 onward is computed, skipping NaNs, and the incumbent moves when it exceeds `tol_fun`.
- The move is lost at the next pass (F1), and the target hyperparameters it sets are never used (F2).

**End of the run (1558-1636)**
- The history is re-evaluated once more.
- The returned point minimizes `q_beta = fval + Φ⁻¹(1−final_quantile)·fsd`, the upper quantile (3.09 SD at 1e-3), over the iterates from index 1 onward (F11), skipping NaNs.
- `noise_final_samples` fresh evaluations are taken at that point, without recording them in the log.
  - Level 2: precision-weighted mean, with `fsd = (Σ1/σ²)^(-1/2)`. This is the correct estimate and SD for independent Gaussian noise of known SD.
  - Level 1: the mean and its standard error, with ddof=1. With one sample, the iterate's biased observation `YVAL` is added in, as the comment says.
- The estimate overwrites `fval` and `fsd` at the chosen index of the history.
- What reaches the result:
  - `x` and `fval`/`fsd` of the estimate, which is an unbiased estimate of the mean objective at `x` with its standard error, as notebook 3 documents; the OptimizeResult docstring's "standard deviation of objective function" is looser;
  - `yval_vec` when nfs > 0, and `ysd_vec` at level 2;
  - `func_count` (samples included), `iterations`, and the last poll's `mesh_size`.
- A run that ends within its first iteration returns the raw minimum with `fsd = noise_size` and leaves its reserved evaluations unused (F3, F10).
- `overhead` counts the final samples as optimizer time (F6).

## 3. Findings

### F1. The re-evaluation's move of the incumbent writes `self.best_u`, not `self.u_best`; the next pass puts the incumbent back and keeps the moved iterate's estimate
- Location: pybads/bads/bads.py:1536 (the typo), 1531-1541, 1406; MATLAB: not opened (internal track).
- Category: state/caching
- Proposed classification: port discrepancy (a slip in the port: the surrounding lines show the intent)
- Confidence: high
- Reached at default options: yes, at levels 1 and 2, from the second iteration, whenever an earlier iterate's re-estimate beats the incumbent's by more than `tol_fun`.
- History: line 1536 predates `ce3a0b3` (2022-11-22). W0-1 (`e004c79`) edited the lines around it and kept it.
- What the code does:
  - It sets `self.u`, `yval`, `fval`, `fsd` and `best_gp_hyp` to the chosen iterate.
  - Nothing ever reads `best_u`; the incumbent's location lives in `u_best`, which is read at 1406, 1738 (target), 1884 (search distance) and 2408.
  - In the next pass, the first search rebuilds the GP and runs the ES search around the moved `self.u`, while the target is predicted at the old `u_best`.
  - Unless a search improves, line 1406 then resets `self.u` to the old point, and the rest of the iteration, the poll included, runs there. Improvements are judged against the moved iterate's `fval`/`fsd`, which are lower than the old point's own estimate by more than `tol_fun`.
  - `optim_state["u"/"yval"/"fval"/"fsd"]` are not updated either. They are read by the fallback of `_get_target_from_gp_` (2667-2670) and passed to `output_fcn`.
- What it should do: move `u_best` too (as `_update_incumbent_` does), so that the iteration continues from the better iterate.
- Consequence if real:
  - The mechanism that returns a noisy run to a better earlier iterate is undone.
  - The iteration's poll then runs around one point with the value of another. This biases the poll toward failure and contraction of the mesh.
  - It happens often. With 6 seeds (noisy sphere, D=2, noise SD 0.5, 200 evaluations) there were 43 moves:
    - in no case did the next poll run at the moved-to iterate;
    - 16 polls ran around the old point with the moved `fval`;
    - in 24 cases a search moved the incumbent first;
    - in 3 the run ended first.
  - With `best_u` aliased to `u_best`, 5 of the 6 runs changed trajectory; errors and evaluation counts moved in both directions. The effect is not measured.
- Suggested reproduction: `f1_move_revert.py`, an instrumented subclass that detects `u != u_best` at the next step (output above). `f1_ab.py` and `f1_check_fix.py` show the change with the alias.
- Test adequacy: no test checks the incumbent after a move. `test_iteration_history_keeps_the_gps_as_recorded` checks only that the GP objects are distinct.

### F2. The target hyperparameters that the re-evaluation moves to the chosen iterate never reach a target that is used
- Location: pybads/bads/bads.py:1424-1425 (reset at every pass), 1512-1514, 1537-1541, 1737-1744, 2103; MATLAB: not opened.
- Category: state/caching
- Proposed classification: unsure. The comment at 1537-1538 states an intent that the code does not carry out.
- Confidence: high on the mechanism, medium on the intent.
- Reached at default options: yes, at levels 1 and 2 (at every move).
- History: line 1425 predates `ce3a0b3`; the comment at 1537-1538 was written in `e004c79` (2026-09-26).
- What the code does:
  - Line 1425 ("GP hyperparameters at end of iteration") runs at the end of every pass and sets `best_gp_hyp` to the working GP's current hyperparameters.
  - After a move, the moved vector is used only by the next search's target. The search writes that target into `optim_state["f_target*"]` and nothing reads it: the poll recomputes its own target before the probability of improvement (2234-2266), and no search module reads it.
  - By the time of the poll, which starts from `best_gp_hyp` (2103), 1425 has overwritten the vector.
  - In deterministic runs too, "end of iteration" really means "end of the last pass".
- What it should do: the comment says the target's hyperparameters should move to the chosen iterate. Whether the target should use the end-of-iteration hyperparameters is for B4 and the comparison track.
- Consequence if real: the poll after a move predicts its target with the working GP's hyperparameters instead of the chosen iterate's. The size of the effect is not measured.
- Suggested reproduction: `f_hyp_overwrite.py`. Over 3 seeds there were 22 moves; the moved vector differed from the working GP's in 22 of 22, and was used by the next poll in 0 of 20.
- Test adequacy: none.

### F3. The initial design exceeds a small `max_fun_evals`; in a noisy run the reservation for the final samples then goes negative and raises `max_fun_evals`
- Location: pybads/bads/bads.py:1075-1078 (the cap, before rounding), init_sobol.py:73-76 (rounding and doubling), bads.py:1030-1037 (the uncounted noise test), 1176-1184; MATLAB: not opened.
- Category: control flow
- Proposed classification: port discrepancy. The rounding is PyBADS's own (KD-B7-1); whether the uncounted noise test is also in MATLAB is unsure.
- Confidence: high
- Reached at default options: no. It needs a `max_fun_evals` smaller than the rounded design plus 2 (for example 5 at D=2 or 3, 7 at D=5), or, at levels 1 and 2, smaller than 34 plus `noise_final_samples`.
- History: the Python lines predate `ce3a0b3`, except the power-of-two rounding (`cdc2e0f`, 2023-06-10).
- What the code does:
  - The design is capped at `max_fun_evals − 1` points before `init_sobol` rounds it up to a power of two (and doubles it when it equals D). The cap also ignores the noise test's evaluation.
  - A negative reserved `noise_final_samples` is subtracted from `max_fun_evals`, which therefore grows.
  - The description says `max_fun_evals` is the "Max number of target fcn evals".
- Consequence if real (measured):

  | Case | `max_fun_evals` | Evaluations made | Notes |
  |---|---|---|---|
  | D=2 | 5 | 6 | |
  | D=3 | 5 | 6 | |
  | D=5 | 7 | 10 | |
  | Noisy, D=2 | 25 | 33 | `noise_final_samples` becomes −8, `max_fun_evals` becomes 33, no final samples |
  | Noisy, D=2 | 38 | 33 | 5 samples reserved, none taken; returned `fval` is the raw minimum of 33 noisy draws, with `fsd = 1.0` (`noise_size`) |

- Suggested reproduction: `f2_budget.py` (output above).
- Test adequacy: `test_one_function_evaluation` covers only `max_fun_evals=1`.

### F4. `sloppy_improvement=False` makes every run fail at its first pass
- Location: pybads/bads/bads.py:1339-1349; MATLAB: not opened.
- Category: control flow
- Proposed classification: port discrepancy (a Python slip)
- Confidence: high
- Reached at default options: no (`sloppy_improvement=False`).
- History: 1339-1349 were written by `ce3a0b3` (2022-11-22), with `cdc2e0f` (2023-06-10).
- What the code does: with `mesh_size_integer` a Python int (0 at the first pass) and no `np.maximum`, `self.sufficient_improvement` is a Python float, so `.copy()` raises `AttributeError: 'float' object has no attribute 'copy'`.
- Consequence if real: the option is unusable.
- Suggested reproduction: `f3b.py` (the traceback points to line 1349).
- Test adequacy: no test sets the option.

### F5. The `f_vals` option makes every run fail at the first display line, and otherwise does nothing
- Location: pybads/bads/bads.py:581-604, 2840-2869 (the cache header and an 8-field format), 2871-2894, 1023; MATLAB: not opened.
- Category: control flow
- Proposed classification: port discrepancy (an unported cache whose remains are broken)
- Confidence: high
- Reached at default options: no (`f_vals` given).
- History: predates `ce3a0b3`.
- What the code does:
  - A finite `f_vals` turns on `cache_active`, which selects a format with 8 fields, `{:5.0f}/{:5.0f}` and three `{:12.6f}`.
  - `_display_function_log_` passes it 6 or 7 values, so the call fails with `ValueError: Unknown format code 'f' for object of type 'str'`. The `.format` runs before `logger.info`, so it fails with `display="off"` too.
  - Otherwise, `x0` is evaluated anyway (1023), so "Evaluated function values at X0" has no effect on the run.
- Suggested reproduction: `f_fvals.py`.
- Test adequacy: none.

### F6. `overhead` counts the noise test and the final samples as optimizer time
- Location: pybads/bads/bads.py:1648-1658; function_logger.py:384-398 (a non-recorded evaluation adds nothing to `total_fun_eval_time`, 440); MATLAB: not opened.
- Category: cross-module
- Proposed classification: unsure
- Confidence: high on the mechanism.
- Reached at default options: yes. Every default run has the noise test (one evaluation); every noisy run has `noise_final_samples` evaluations.
- History: predates `ce3a0b3`.
- What the code does: `overhead = total_time/total_fun_eval_time − 1`, where the denominator leaves out the evaluations made with `record_duplicate_data=False`. The docstring describes the overhead as "compared to function time".
- Consequence if real: with 60 evaluations of 50 ms each, 10 of them final samples, the reported overhead is 0.311, against 0.092 from the target's measured time.
- Suggested reproduction: `f_overhead.py`.
- Test adequacy: none.

### F7. `display="notify"` and `display="final"`, offered by the option's description, behave as `"iter"`
- Location: pybads/bads/bads.py:224-232; basic_bads_options.ini:2-3; MATLAB: not opened.
- Category: defaults
- Proposed classification: port discrepancy against the description. KD-B2-3 settles the logger mechanism, not which levels exist.
- Confidence: high
- Reached at default options: no (only when those values are set).
- History: predates `ce3a0b3`; `"full"` was added in #71. `"full"` is not in the description.
- What the code does: only `"off"`, `"iter"` and `"full"` are handled; any other value keeps INFO. Each of `"final"`, `"notify"` and `"iter"` logged the same 18 lines.
- Suggested reproduction: `f3_sloppy_maxiter.py`.
- Test adequacy: none.

### F8. The final message and `yval_vec` misdescribe the final estimate of noisy runs with 0 or 1 final samples
- Location: pybads/bads/bads.py:1559-1562, 1606-1615, 1660-1671; optimize_result.py:28-32; MATLAB: not opened.
- Category: state/caching
- Proposed classification: port discrepancy (display and documentation)
- Confidence: high
- Reached at default options: no (`noise_final_samples` 0 or 1, at levels 1 and 2).
- History: 1559-1562 was edited in #71 and `068e57f`; the rest predates `ce3a0b3`.
- What the code does:
  - With `noise_final_samples=0`, the local `yval_vec` is taken before the final choice. The message "Observed function value at minimum" then prints the last incumbent's observation (−0.1005 in the run), not the returned point's (−0.3686).
  - At level 2 with one sample, the message calls the sample and its SD a "GP mean ± SEM".
  - At level 1 with one sample, the message says "from 2 samples", one of which is `YVAL`, and the result's `yval_vec` has shape (2,1) with `YVAL` in it. The docstring says "Final sampled observations at the solution".
- Suggested reproduction: `f_final.py`.
- Test adequacy: `test_final_estimate_from_one_sample` covers level 2 only.

### F9. Termination is checked at every pass, so `max_iter` and `tol_stall_iters` count the current iteration as complete after its first search
- Location: pybads/bads/bads.py:1439-1465; MATLAB: not opened.
- Category: control flow
- Proposed classification: possibly intentional. #71 and AGENTS.md say the iteration count follows MATLAB; the comparison track should settle where the check sits.
- Confidence: medium
- Reached at default options: the stall criterion yes; `max_iter` (200·D) rarely.
- History: the checks predate `ce3a0b3`; the `max_iter` condition was changed in #71.
- What the code does:
  - `max_iter=N` ends the run at the first search of iteration N: N−1 polls. With `max_iter=2` the run made 1 poll, then 1 search.
  - The stall criterion fired on a search pass after T−1 = 4 complete iterations since its base, in 4 of 4 seeded runs that ended on it.
  - The descriptions read "Max number of iterations" and "Max iterations with no significant change".
- Suggested reproduction: `f3_sloppy_maxiter.py`, `f_stall.py`.
- Test adequacy: `test_iterations_count_from_one` checks only the count.

### F10. In a noisy run, the incumbent's value through the first two iterations is the raw minimum of the initial design
- Location: pybads/bads/bads.py:1114-1119, 1186-1195, 1503-1507 (the re-evaluation requires `poll_iteration > 0`); MATLAB: not opened.
- Category: control flow
- Proposed classification: suspected defect in both / possibly intentional. The brief's default course has the re-estimation starting from the second iteration.
- Confidence: low
- Reached at default options: yes, at levels 1 and 2.
- History: predates `ce3a0b3`.
- What the code does:
  - The minimum of 33 or more noisy draws, a biased-low order statistic, with `fsd = noise_size`, is what searches and polls must beat until the end of the second iteration.
  - From memory, the paper describes the noisy incumbent's value as the GP's estimate.
  - In one seed the incumbent stayed at −0.739 (`fsd` 1.00) through two polls, where the GP mean was +0.33 and the true value 0.555. Both polls failed, and the mesh went from 1 to 0.25.
- Suggested reproduction: `f_first_iters.py`.
- Test adequacy: none.

### F11. The final choice and the end-of-iteration move exclude the first iterate
- Location: pybads/bads/bads.py:1523, 1580; MATLAB: not opened.
- Category: indexing/shape
- Proposed classification: possibly intentional
- Confidence: low
- Reached at default options: yes, at levels 1 and 2.
- History: predates `ce3a0b3`.
- What the code does: `[1:]` is marked "Skip the first iteration", with no stated reason. The iterate's re-estimate uses the current data like any other. The description of `final_quantile` says nothing of it.
- Consequence if real: the first iterate can never be returned, and a two-iteration run has only one candidate.
- Suggested reproduction: a noisy run whose best re-estimate is at index 0.
- Test adequacy: none.

### F12. `min_fun_evals` and `min_iter` are documented but no code reads them
- Location: advanced_bads_options.ini:281-284; no reads in `pybads/`; MATLAB: not opened.
- Category: defaults
- Proposed classification: unsure (likely PyVBMC leftovers; KD-B1-5 does not list them)
- Confidence: high
- Reached at default options: they never act; setting them changes nothing.
- History: predates `ce3a0b3`.
- What the code does: their descriptions ("Min number of fcn evals", "Min number of iterations") promise a limit on termination that no code applies. This overlaps with B1.
- Suggested reproduction: grep for the two names.
- Test adequacy: none.

### F13. The mesh acceleration's guard skips a comparison that the stall criterion would make (slice B4's code)
- Location: pybads/bads/bads.py:2422-2446; MATLAB: not opened.
- Category: indexing/shape
- Proposed classification: unsure
- Confidence: low
- Reached at default options: yes (`accelerate_mesh` is on by default).
- History: predates `ce3a0b3`.
- What the code does: `iter > accelerate_mesh_steps`, with the base at `iter − steps`, never uses history index 0. The first acceleration can therefore come at the 5th iteration, where "after 3 stalled iterations" would allow the 4th. The stall criterion's guard is `>=`.
- Suggested reproduction: log the iterations at which the acceleration is checked.
- Test adequacy: none.

## 4. Test adequacy notes
- `test_noisy_runs.py`: the final-estimate tests (`..._weights_samples_by_precision`, `..._without_target_noise`, `..._from_one_sample`) recompute the formulas from `yval_vec` and `ysd_vec`, which mirrors the implementation. None checks which point is returned or where the incumbent is after a re-evaluation (F1, F2). `test_final_estimate_recorded_at_its_iterate` only checks that the history matches the result.
- `test_iterations_count_from_one` checks the reported count, not how many polls ran (F9). `test_one_function_evaluation` is the only budget test (F3).
- No test runs `sloppy_improvement=False`, `f_vals`, `display="notify"` or `"final"`, or checks `overhead` (F4 to F7).
- `test_iteration_history.py` tests the container's mechanics only. Nothing tests what the loop records, or its indices: `mesh_size` is the next iteration's, and `search_mesh_size` the current one's.
