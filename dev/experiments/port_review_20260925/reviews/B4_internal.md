<!-- Report of the B4 internal reviewer (poll, mesh, incumbent and target, internal-correctness track), wave 3 of the port review, reading PyBADS at 8aecb6a in ../pybads-review and gpyreg v1.3.3 (98ab5a4), in a cloud session, with the complete history; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave3/B4_internal/. Nothing in it is verified. -->

# B4 internal review: poll, mesh, incumbent and target

Python cited at `8aecb6a` (`/home/user/pybads-review`). The checks are in `/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/review/B4_internal/check*.py`. Every one printed the review worktree's `pybads/__init__.py` and `/home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. I did not open any MATLAB code.

## 1. Coverage

**Read completely:**
- `pybads/bads/bads.py`:
  - the functions `_poll_step_` (2088-2500), `_eval_improvement_`, `_is_poll_stop_`, `_get_target_from_gp_`, `_update_incumbent_`, `_check_mesh_overflow_` and `_is_gp_refit_time_`;
  - the whole of `_search_step_` and `_update_search_stats_`;
  - the main loop (1300-1560);
  - the mesh setup in `_init_optim_state_` (590-720) and the changes a noisy run makes to the options (1135-1200).
- `pybads/poll/poll_mads_2n.py`, `pybads/function_logger/constraints_check.py`, `pybads/search/grid_functions.py`, `pybads/acquisition_functions/acq_fcn_lcb.py`.
- `basic_bads_options.ini` in full, and `advanced_bads_options.ini` lines 1-200 (every option the poll reads).
- `local_gp_fitting`: its snapshot and restore, and how `poll_scale`, `len_scale` and the effective radius are computed (`gaussian_process_train.py` 252-300, 485-597).
- The variance clipping in gpyreg's `predict`.
- The known-differences sheet.

**Skimmed:**
- Tests: `test_poll_mads.py`, `test_search.py::test_incumbent_constraint_check`, parts of `test_run_control.py`, the target tests in `test_gp_update_failures.py`, the skip test in `test_bads_logger.py`, `test_bads_seed.py`.
- `docsrc/source/index.rst`.

**Not reached:**
- The counterpart map (not needed on this track).
- The Sto-BADS branches, beyond where the default path passes through them.
- The use of `poll_scale` in ES-ell, beyond a glance.

## 2. Answers to the first questions

### Q1. The poll set

**How the basis is built (`poll_mads_2n`):**
- The off-diagonal entries are drawn in {1−n_max, …, n_max−1}, which is LTMADS's open interval, and the matrix is cut to its strictly lower-triangular part. The diagonal is ±n_max with random signs. Its rows are permuted and the result transposed.
- The columns are not permuted separately, although the comment says "rows and columns". This only reorders the directions, so the set is unchanged. The signs are irrelevant too, since the negation of the basis is added.
- The basis is nonsingular, so the basis and its negation form a maximal positive basis: the set does positively span.

**The two defects that make it a coordinate poll:**
- The integer bound is `n_max = max(1, round(search_mesh_size/mesh_size))`. At every reachable default state the search mesh is finer than the poll mesh (2^(2k−10) against 2^k, with k ≤ 0), so n_max is always 1 (F1). The basis is then a signed permutation of the identity, and the poll set is always {u ± mesh_size·e_i}. That is a fixed coordinate poll, not a dense set of LTMADS directions.
- The division by `poll_scale` is undone exactly by the multiplication in `_poll_step_` (2146-2150), so the GP's geometry never reaches the poll vectors (F2).
- Measured: in default runs on Rosenbrock (D = 3) and an ellipsoid (D = 4), n_max was always 1 and every poll direction was ±e_i, although the log-range of `poll_scale` reached 5.1 (`check1`).

**The points dropped:**
- `contraints_check(..., proj=False)` drops points outside the hard bounds and points that violate `non_box_cons`.
- Its removal of points already evaluated is a no-op (F6).
- `force_poll_mesh` is off, and with n_max = 1 the points already lie on the search mesh whenever the incumbent does.

**An empty set, or one that runs out:**
- An empty set breaks out of the loop at 2188, before the rebuild and the target. The poll counts as failed and the mesh shrinks.
- A set that runs out, or a spent budget, ends the loop, and the poll is then judged on the points evaluated.
- The refill branch `np.vstack(u_poll, u_poll_new)` (2180) has the wrong signature but can never run: `B` is filled once and never emptied.

### Q2. The order and the early stop

**The order:**
- At each step the remaining points are ranked by LCB, `f_mu − sqrt_beta·f_s`, with ν = 0.2 and δ = 0.1, computed on the current GP.
- At level 0 that GP never takes the poll's own evaluations: it is rebuilt only at the first step or when a refit is due. The ranking is therefore effectively static, and the target is predicted at polled points the GP does not hold (F5).
- At levels 1 and 2, each point is added with `add_and_update_gp` before the next ranking.

**The target the stop uses:**
- It is predicted at `u_poll_best`, which is the incumbent until a polled point improves on it, under `gp_poll_hyp_best`.
- `gp_poll_hyp_best` starts as `best_gp_hyp` (the GP's hyperparameters at the end of the previous pass) and becomes the GP's current hyperparameters when a polled point improves.

**The probability of improvement:**
- PoI_i = Φ((f_target − SI − μ_i)/σ_i), where σ_i is the latent SD and SI is the sufficient improvement, max(mesh^1.5, tol_fun) at default.
- p_less is meant to be ∏(1 − PoI) over the D+1 largest PoI among the remaining points, the next candidate included. Because of the sort, it is taken over the last D+1 points in the order of `u_poll` (F3).

**The stop rule (`_is_poll_stop_`), with `complete_poll` off:**
- After a good poll: stop if the GP is unreliable, otherwise stop when p_less > 1 − tol_poi.
- Without a good poll: the stop also needs a reliable GP, `consecutive_skipping` or no skip in the previous iteration, poll_count ≥ `min_failed_poll_steps` (∞ by default, and forced to ∞ at levels 1 and 2), and p_less > 1 − tol_poi. So a default poll without a good point is never cut short.

**What makes the GP "unreliable":**
- the calibration test (or no statistics since the last refit);
- a failed rebuild in the poll;
- any non-finite γ, which includes any remaining point with a predictive SD of zero (F9).

**Consistency with the option descriptions:**
- The descriptions of `tol_poi` and `consecutive_skipping` hold, with two exceptions:
  - `tol_poi = 0` does not always complete polling, because the unreliable-GP stop ignores it.
  - With `consecutive_skipping=False`, no skip is allowed at iteration 0, since `last_skipped = −1` is not less than `iter − 1 = −1`. This is minor and needs non-default options.
- `skip_poll = True` reads as though skipping without success is on, but it is unread (KD-B1-5a), and `min_failed_poll_steps = ∞` turns that skipping off.

### Q3. The improvement, the move and the mesh

**`_eval_improvement_`:**
- It returns (f_base − f_new) + sqrt(s_base² + s_new²)·Φ⁻¹(q). That is the q-quantile of the difference of two independent Gaussians, correct under independence. The code's own comment flags the correlation as uncorrected.
- At level 0 both SDs are 0, so the result is the plain difference at any q, except at q ∈ {0, 1}, where 0·∞ gives NaN. That is outside the useful range.
- q < 0.5 is conservative, as the description says.

**Success and the move:**
- A poll succeeds when its best improvement exceeds SI (`certain_good_poll`).
- With `sloppy_improvement` on, any positive improvement moves the incumbent, and a move that is not a success still shrinks the mesh.
- `sloppy_improvement` also sets the `tol_fun` floor on SI (main loop, 1337-1343). Its description does not say so.
- `_update_incumbent_` sets u, `u_best`, yval, fval and fsd in both places (`self` and `optim_state`). It does not move `best_gp_hyp`; the loop resets that at the end of each pass.

**The mesh:**
- A success adds one to `mesh_size_integer`, capped at `max_poll_grid_number = 0`. The overflow is counted first, and the warning fires once, at ceil(`mesh_overflow_warning`), which is doubled under noise.
- A failure subtracts one, and one more when the improvement since `iteration_history[iter − accelerate_mesh_steps]` is below `tol_fun`. That comparison starts at 0-based iter ≥ 3, and the tests use the same quantile.
- The search mesh follows: `min(ssi, 2·msi − 10)`, recomputed with the lock at the next pass.
- All of this matches the option descriptions and a forcing function of Δ^(3/2).

### Q4. The target

**The formula:**
- With `uncertain_incumbent` on (the default), or at levels 1 and 2: a deep copy of the GP takes `hyp_best` (the posterior is recomputed), predicts μ and fs² at u, and gives f_target = μ − sd_level·sqrt(fs² + tol_fun²), with sd_level = 0.1 unless `adaptive_incumbent_shift` is on.
- With `alternative_incumbent`: f_target = μ − sqrt(D/func_count)·f_target_s.
- On a `LinAlgError`, the current GP is used (KD-B4-2).
- A non-finite prediction falls back to fval and fsd for μ and σ, but f_target still uses the non-finite fs² (F7).
- With `uncertain_incumbent=False` at level 0, the branch returns Python floats and the callers' `.item()` raises (F4).

**Which point each stage uses:**
- The search computes the target at `u_best` under `best_gp_hyp` and stores it, but nothing reads it: LCB does not use the target, and neither does the search's improvement.
- Only the poll's PoI reads `f_target`.

**The docstring:**
- It says the target is shifted below the prediction "when the target function is stochastic". It is shifted in deterministic runs too, since `uncertain_incumbent` is on by default.
- It calls `f_target_s` a "GP variance/noise". It is the latent SD.

## 3. Findings

### F1. The "LTMADS" poll basis always reduces to the ± coordinate directions: n_max is always 1
- Location: `pybads/poll/poll_mads_2n.py:22` (also 25, 30); `pybads/bads/bads.py:2137-2150`. MATLAB: not consulted.
- Category: formula
- Proposed classification: unsure. The lines look transliterated, with MATLAB's comments kept; only the comparison track can tell a port discrepancy from a defect in both.
- Confidence: high that the poll reduces to a coordinate poll; medium that this is a defect.
- Reached at default options: yes, at every poll and every uncertainty level. n_max > 1 needs a search mesh coarser than the poll mesh (for example `search_grid_number < 0`), which no default state produces.
- History: written in `c7c88ab` (2022-06-02); the draws moved to `rng` in `1d075ab`.

**What the code does:**
- `n_max = max(1, round(search_mesh_size/mesh_size))`. With `search_size_integer = min(0, 2k − 10)` and k ≤ 0, the ratio is 2^(k−10) ≤ 2^−10, so n_max = 1.
- The lower entries are then drawn from {0}, and the diagonal is ±1.
- The poll set is always u ± mesh_size·e_i. It is deterministic, and the directions are identical at every iteration.

**What the specification says:**
- The docstring claims "dense refining directions in the hypertangent cone".
- In LTMADS (Audet and Dennis 2006) and in the paper's MADS definitions (P_k = {x_k + Δ^mesh_k·v}, with Δ^poll ≳ Δ^mesh·‖v‖), the integer bound is Δ^poll/Δ^mesh = `mesh_size/search_mesh_size` = 2^(10−k), which grows as the mesh shrinks, and the step is Δ^mesh·v.
- The code uses the reciprocal ratio, and steps of `mesh_size·v`.

**Consequence if real:**
- The poll is a coordinate search (GPS with D = [I, −I]): positively spanning but not dense, so the Clarke-stationarity argument for MADS does not apply.
- The search usually compensates. In a monkeypatched LTMADS variant (bound `mesh/search`, steps on the search mesh), on a nonsmooth diagonal ridge in D = 2, one seed of three stalled early with the coordinate poll (0.0166 after 71 evaluations, against 1.3e-4). Rosenbrock (D = 3) was comparable (`check12`).

**Reproduction:** `check1_poll_dirs.py`. In default runs every poll had `n_max = {1}` and "all coordinate directions: True".

**Test adequacy:** `test_poll_mads_2n_ones` and `test_poll_mads_2n` assert n_max = 1 at the default mesh sizes, which enshrines the degenerate case. `test_poll_mads_2n_dense` uses a configuration that defaults never reach.

### F2. The GP's `poll_scale` never changes the poll vectors: the division and the multiplication cancel
- Location: `pybads/poll/poll_mads_2n.py:36-37`; `pybads/bads/bads.py:2145-2150`. MATLAB: not consulted.
- Category: formula
- Proposed classification: possibly intentional. The comment "Counteract subsequent multiplication by pollscale" says the division exists to cancel it. Unsure against the specification.
- Confidence: high that it has no effect; low that it is a defect.
- Reached at default options: yes, at all levels.
- History: `c7c88ab` (2022-06-02).

**What the code does:**
- `B_new = D/poll_scale`, then `vv = B_new·mesh_size·poll_scale = D·mesh_size`, for any n_max.
- So `gp_rescale_poll` ("GP-based geometric scaling factor of poll vectors") and the clipping of `poll_scale` to [search_mesh_size, ub − lb] in `local_gp_fitting` (564-571) have no effect on the poll. They act only through ES-ell (`es_search.py:295`).
- The option description, and (as I recall it) the BADS paper's poll vectors rescaled by the normalized GP length scales, say the poll is GP-scaled. `AGENTS.md` also claims `poll_scale` "drives the poll basis".

**Consequence if real:** the poll steps are equal in every coordinate whatever the GP's anisotropy. On the D = 4 ellipsoid the length scales differed by a factor of about 160.

**Reproduction:** `check1`, which records `vv/mesh_size` against `ptp(log poll_scale)`.

**Test adequacy:** `_check_poll_set` in `test_poll_mads.py` multiplies by `poll_scale` again before checking, which mirrors the implementation. No test checks that the poll vectors follow the GP's length scales.

### F3. p_less sorts along the wrong axis, so it takes the last D+1 points in `u_poll` order instead of the D+1 largest PoI
- Location: `pybads/bads/bads.py:2281-2284`. MATLAB: not consulted.
- Category: indexing/shape
- Proposed classification: port discrepancy. The comment "sort descend" states the intent, and a MATLAB `sort` of a column vector sorts along it.
- Confidence: high.
- Reached at default options: yes. It matters after a good poll with more than D+1 points left, so for D ≥ 3, at all levels. With a finite `min_failed_poll_steps` it also affects skipping.
- History: `c7c88ab` (2022-06-02).

**What the code does:**
- `f_mu` and `fs` from `acq_fcn_lcb` have shape (N, 1), and so does `f_pi`. `np.sort(f_pi)` sorts along the axis of length 1, which changes nothing. `[::-1]` then only reverses the rows.
- The product is thus over an arbitrary subset: `u_poll` is in lexicographic order after `contraints_check`.
- Toy example: PoI = [0.9, 0.01, 0.02, 0.03, 0.04] with D = 3 gives p_less = 0.904 as coded, against 0.091 over the largest four.

**Consequence if real:** after a good poll, a high-PoI point that is left out of the product makes p_less too large, so the poll stops early. The measured effect is small:
- the shape (N, 1) is confirmed;
- the coded and intended products differed in 4 to 12 steps per run, and the threshold decision flipped in 1 to 3 steps, all of them outside good polls (`check8`);
- substituting the intended product changed no run of 12 at level 0 (D = 3 to 6), 4 at levels 1 and 2, or 2 with `min_failed_poll_steps=2` (`check9`, `check9b`).

**Reproduction:** `check8_pless_sort.py`.

**Test adequacy:** no test covers p_less.

### F4. `uncertain_incumbent=False` on a deterministic target crashes at the first poll
- Location: `pybads/bads/bads.py:2696-2699` (returns floats), read at 2245, 2249, 1748 and 1752 (`.item()`). MATLAB: not consulted.
- Category: control flow
- Proposed classification: port discrepancy (a Python-only type error).
- Confidence: high.
- Reached at default options: no. It needs `uncertain_incumbent=False` at level 0 (a deterministic target, or `uncertainty_handling=False`).
- History: the `.item()` calls come from `c7c88ab` and `9037851` (2022-09-22); the else-branch from `c7c88ab` and `cdc2e0f`.

**What the code does:** the branch returns `optim_state["fval"]` and `fval − tol_fun` as Python floats. The poll calls `f_target_mu.item()` on them, which raises `AttributeError: 'float' object has no attribute 'item'`. The option is documented as a supported switch.

**Consequence if real:** every such run fails at iteration 1.

**Reproduction:** `check4_options.py`. The traceback points to `bads.py:2245`. `alternative_incumbent`, `complete_poll`, `tol_poi=0`, `force_poll_mesh` and a finite `min_failed_poll_steps` all ran.

**Test adequacy:** no test sets `uncertain_incumbent=False`.

### F5. At level 0 the poll's GP never takes the poll's evaluations, and after an improving point the target is predicted where the GP has no data
- Location: `pybads/bads/bads.py:2309-2332` (the GP is updated only when level > 0), 2242-2244 and 2344-2348 (target at `u_poll_best`). MATLAB: not consulted.
- Category: state/caching
- Proposed classification: possibly intentional (the design rebuilds only at the first step or a refit). Unsure.
- Confidence: medium on the mechanism; low on the impact.
- Reached at default options: yes, at level 0 after a polled point improves.
- History: `c7c88ab` (2022-06-02).

**What the code does:**
- After an improving point, `u_poll_best` is that point. The target is μ(u_poll_best) − 0.1·sqrt(fs² + tol_fun²) from a GP that has not seen y(u_poll_best).
- With `uncertain_incumbent` on, the incumbent is meant to be treated as uncertain: its value should be the GP's posterior at the incumbent given its data.
- Observed:
  - on the ellipsoid, a prediction of −1.24e5 where the observed value was 1.307e5 (SD 2.3e5);
  - elsewhere, −8.1e-7 against 6.0e-7 with an SD of 0.
- The remaining points' LCB and PoI ignore the poll's own observations too.

**Consequence if real:** the early-stop decision is taken against a target that can be far too low, which stops early, or too high, which wastes evaluations. Using a copy of the GP that holds the point changed 1 run of 9 (ellipsoid D = 4, seed 2: 200 evaluations became 191, fval 1.66e-7 became 2.47e-7) (`check10`).

**Reproduction:** `check3_target_level0.py`, `check10_target_obs.py`.

**Test adequacy:** none.

### F6. The poll set keeps points that were already evaluated: `contraints_check`'s set difference is a no-op (cross-module, slice B3)
- Location: `pybads/function_logger/constraints_check.py:39-43`, called from `pybads/bads/bads.py:2166-2174`. MATLAB: not consulted.
- Category: cross-module
- Proposed classification: port discrepancy. The repository's own test comment says MATLAB's `uCheck` removes them, and calls it a candidate defect in the survey.
- Confidence: high.
- Reached at default options: yes, at all levels.
- History: the correct setdiff of `c7c88ab` was replaced by the `np.unique` form in `8e59038` (2022-06-03).

**What the code does:** `np.unique(vstack(u1, u2), return_index=True)` returns the first occurrence of each row. Every row of `u1` comes first, so none is ever removed. Its comment says "Remove previously evaluated vectors".

**Consequence if real:** the poll can spend an evaluation on a known point, adding a duplicate row to the log and the GP. At the mesh cap, the poll that follows a successful poll contains the old incumbent. Measured: 1 poll re-evaluation in about 60 on Rosenbrock D = 3 (seeds 1 and 2), a design point at u − e1 with mesh 1. None on the ellipsoid, the sphere or Rosenbrock D = 2 (`check2`, `check6`). The search is affected too.

**Reproduction:** `check2_reevals.py`, `check6_reeval_detail.py`.

**Test adequacy:** `test_incumbent_constraint_check` asserts that no evaluated row is removed, which enshrines the defect.

### F7. The target's fallback for a non-finite prediction still uses the non-finite variance
- Location: `pybads/bads/bads.py:2672-2695`. MATLAB: not consulted.
- Category: formula
- Proposed classification: unsure.
- Confidence: high on the mechanism.
- Reached at default options: only when the GP predicts a non-finite value (rare; gpyreg clips s² at 0).
- History: the fallback is from `157bd09` and `c7c88ab`, and was reshaped in `685da15` (2026-09-25).

**What the code does:** μ and σ fall back to fval and fsd, but `f_target = μ − sd_level·sqrt(fs2 + tol_fun²)` keeps the original `fs2`. Measured: fs² = NaN gives f_target = NaN, and fs² = inf gives −inf (`check5`).

**Consequence if real:** the poll's γ is non-finite, so p_less = 0 and the GP is declared unreliable, and a good poll stops. The search's target is unused. The impact is small.

**Test adequacy:** `test_target_fallback_to_incumbent` calls `f_target.item()` but never checks that it is finite; it would pass with NaN.

### F8. A NaN LCB is evaluated first, and the "randomly choose index" fallback cannot fire
- Location: `pybads/bads/bads.py:2257-2270` (the search's copy is at 1805-1818). MATLAB: not consulted.
- Category: control flow
- Proposed classification: port discrepancy. `np.argmin` returns the first NaN, whereas MATLAB's `min` omits NaN.
- Confidence: medium.
- Reached at default options: only with a NaN prediction (rare).
- History: `c7c88ab`.

**What the code does:** `np.argmin` always returns a finite scalar index, so the checks for `None`, size < 1 and non-finite never hold (verified in `check11`). If some LCB is NaN, that point is chosen.

**Consequence if real:** a wasted evaluation, in rare states.

**Test adequacy:** none.

### F9. An unreliable GP, including any remaining point with a zero predictive SD, stops a good poll whatever `tol_poi` says
- Location: `pybads/bads/bads.py:2273-2288`, 2597-2601. MATLAB: not consulted.
- Category: control flow
- Proposed classification: possibly intentional.
- Confidence: low (on whether it matters).
- Reached at default options: yes, at level 0, after a good poll.
- History: `c7c88ab`.

**What the code does:**
- A zero σ at any remaining point makes γ = ±inf, so p_less = 0 and `do_gp_calibration = True`. After a good poll that stops the poll, although PoI ∈ {0, 1} is well defined there.
- The description of `tol_poi` says 0 "always complete[s] polling". That does not hold on this path.
- Zero SDs at remaining poll points were frequent: in 6 to 50 poll steps per run in 9 default runs. None of them fell after a good poll, so no stop was observed (`check7`).

**Test adequacy:** none.

### F10. The poll turns off NumPy's divide warnings for the whole process and never restores them
- Location: `pybads/bads/bads.py:2271-2272`. MATLAB: no counterpart.
- Category: state/caching
- Proposed classification: port discrepancy (a Python-only side effect).
- Confidence: high.
- Reached at default options: yes (the root logger's level is WARNING).
- History: `f9e9326` (2022-11-02).

**What the code does:** `np.seterr(divide="ignore")` changes the global error state. After a run, `np.geterr()["divide"]` is `'ignore'`, and 1/0 in user code warns no more (`check11`). The condition tests the root logger's level, not the `BADS` logger's.

**Test adequacy:** `test_seeded_run_leaves_global_state_untouched` checks only NumPy's random state.

## 4. Test adequacy notes

- `test_poll_mads.py` checks the construction as coded: integer entries once multiplied back by `poll_scale`, a maximum of n_max, and a determinant of ±n_max^D. It asserts n_max = 1 at the default mesh sizes. Nothing checks that the directions are dense across calls, that they scale with the GP, or that the integer bound follows poll size over mesh size (F1, F2).
- `test_incumbent_constraint_check` asserts the no-op removal of evaluated points (F6).
- `test_target_fallback_to_incumbent` accepts a NaN `f_target` (F7).
- No test covers p_less or the order of the poll, `uncertain_incumbent=False` (F4), the level-0 target in the poll (F5), or NumPy's global error state (F10).
- `test_accelerated_mesh_reduction_counts_iterations_from_one` checks the extra reduction on a start at the minimum, where every poll fails. It does not separate the improvement test from the iteration count.
