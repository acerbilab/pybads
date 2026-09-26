<!-- Report of the B4 comparison reviewer (poll, mesh, incumbent and target, MATLAB-comparison track), wave 3 of the port review, reading PyBADS at 8aecb6a in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session, with the complete history; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave3/B4_comparison/. Nothing in it is verified. -->

# B4 comparison review: poll, mesh, incumbent and target

## 1. Coverage

**Read completely.**
- Python at `8aecb6a`, in `pybads/bads/bads.py`: `_poll_step_` (2088-2500), `_eval_improvement_` (2018-2037), `_is_poll_stop_` (2588-2615), `_get_target_from_gp_` (2634-2701), `_update_incumbent_` (2717-2730), `_check_mesh_overflow_`, `_is_gp_refit_time_` and `_record_gp_refit_` (read for the unreliability flag), the whole of `_search_step_` (1686-2016), and the main loop (1242-1560).
- Other Python files: `pybads/poll/poll_mads_2n.py`, `pybads/function_logger/constraints_check.py`, `force_to_grid`, `acq_fcn_lcb`, and in `gaussian_process_train.py` the functions `local_gp_fitting`, `add_and_update_gp` and the return path of `_robust_gp_fit_` (to check that the poll's GP object is modified in place). Also the duplicate handling in `FunctionLogger._record`, the poll's options in both `.ini` files, and gpyreg 1.3.3's `GP.predict` (for output shapes and variance clamping).
- MATLAB at `74919c0`: `poll/pollMADS2N.m`, and in `bads.m` lines 430-510, the search stage 511-740, the poll stage 767-1046, the iteration end 1048-1133, and `savegpstats`, `IsRefitTime`, `EvalImprovement`, `UpdateIncumbent`, `UpdateTarget`, `reevaluateIterList` and `meshOverflowCheck`. Also `utils/uCheck.m`, `force2grid.m`, `gppred.m`, `mygp.m` (1-200), `gppredcheck.m`, `acq/acqLCB.m`, `private/gpupdate.m` (1-470), `search/searchES.m` (130-170), and a grep of `private/setupvars.m`.
- History: `git diff 9490a84 74919c0 -- bads.m` shows that no MATLAB line of this slice changed after 2022-02-11. All of them date from 2017.

**Not opened.** `poll/pollGPS2N.m`, `poll/private/*` and `private/covmatadapt.m`. They are unused at MATLAB's defaults, and KD-B4-1 and the counterpart map cover them.

**Checks run.** Scripts are in the scratchpad directory, and each printed that it imported the review worktree and gpyreg 1.3.3.
- `check_poll_basis.py`: the port against a transcription of `pollMADS2N`.
- `check_ucheck.py`, `check_pless.py`, `check_uncertain_incumbent_off.py`, `check_quantile.py`: reproductions of F2, F1, F3 and F7.
- `instrument_runs.py`: 20 seeded default runs, each capped at 200 evaluations: Rosenbrock D=2 and D=4, ellipsoid D=3 and D=6, Ackley D=3, two seeds each, at level 0 and at level 1.
- `instrument_tolpoi.py` (8 runs with `tol_poi=0.01`) and `instrument_subset.py` (8 default runs).

## 2. Answers to the first questions

### 1. The poll set
- **Basis.** `poll_mads_2n` draws the same basis as `pollMADS2N`:
  - `rng.integers(1, 2*n_max) - n_max` is `randi(2*nmax-1) - nmax`, which gives integers in [1-n_max, n_max-1]. The strictly lower-triangular part matches.
  - The diagonal is ±n_max with random signs, as in MATLAB. The transcription check found the same set of entry values for n_max = 1, 3, 4 and 8.
- **Permutations.** MATLAB permutes both rows and columns and then transposes. Python permutes only the rows (`rng.permutation(D)`, line 34). The column permutation only reorders the directions, and the set of directions is the same (checked). After `uCheck` or `contraints_check`, both poll sets come out in lexicographic order of their rounded rows. So the only difference is in which random numbers are drawn.
- **Scaling.**
  - Python divides column-wise by `poll_scale`, then `_poll_step_` multiplies by `mesh_size` and by `poll_scale` again (2146-2150). This is the same as `bads.m:803`.
  - The two `poll_scale` factors cancel exactly on both sides, so the GP's geometry never shapes the poll vectors.
  - At default options `n_max` is always 1: `search_size_integer = min(0, 2·msi − 10)` with `msi ≤ max_poll_grid_number = 0`, so the ratio is at most 2^-10 and rounds to 0. The basis is then a signed permutation matrix, and the poll points are u ± Δ e_i on both sides.
- **Points dropped.**
  - Bounds: the hard bounds in u space. Matches.
  - `non_box_cons`: matches.
  - `force_poll_mesh`: off by default. When on, it uses `force_to_grid` on the search mesh, as `force2grid` does.
  - Points already evaluated are **not** dropped: see F2.
- **Empty set, or a set that runs out.**
  - Both sides break before the refit test when the set is empty after filtering (`bads.m:819`, Python 2188). The poll then fails and the mesh shrinks.
  - Both stop when the set empties, at `max_fun_evals`, or after 2D evaluations. Python's `poll_count < 2D` and MATLAB's `pollcount <= 2*nvars` are equivalent, because the set never holds more than 2D points.
  - `np.vstack(u_poll, u_poll_new)` at line 2180 is wrong syntax but cannot be reached.

### 2. The order and the early stop
- **Order.** Both sides evaluate the point with the lowest LCB on the current GP. That GP is rebuilt at the first poll step, rebuilt again at any refit, and at level ≥ 1 updated by each add. LCB does not use the target on either side.
  - `np.argmin(z)` differs from MATLAB's `min` only when `z` contains NaN: MATLAB skips NaN, NumPy returns the first NaN. gpyreg clamps the variance at 0, so this is not reached.
  - The random-index fallbacks are dead code on both sides.
- **Stop rule.** `_is_poll_stop_` follows `bads.m:876-895`:
  - `tol_poi`, `consecutive_skipping`, `min_failed_poll_steps` and `complete_poll` all match.
  - `last_skipped = -1` against MATLAB's `lastskipped = 0` is consistent with Python's 0-based iteration count.
  - An unreliable GP stops the poll at once after a good poll, and allows no stop without one. Matches.
  - The target (`u_poll_best` under `gp_poll_hyp_best`) and the sufficient improvement (from the loop head) are the same on both sides.
- **The probability `p_less` differs: F1.** Python does not sort the probabilities, and it takes D+1 of them where MATLAB takes D.
- **Unreliability flag.** The flag comes from `_is_gp_refit_time_`, which is slice B5's. With `poll_training=False` it differs from MATLAB (F6). A failed rebuild marks the GP unreliable (KD-B5-2).
- At default options no stop happens without a good poll, since `min_failed_poll_steps` is infinite.

### 3. The improvement, the move and the mesh
- **Improvement.** `_eval_improvement_` computes `EvalImprovement` term by term, at every quantile q in (0, 1). MATLAB raises for q outside (0, 1); Python does not (F7). Python returns a 1-element array, which is harmless.
- **Success and move.** A successful poll is one whose improvement exceeds the sufficient improvement. The incumbent moves on any positive improvement when `sloppy_improvement` is on. Both match.
  - `_update_incumbent_` does what `UpdateIncumbent` does, and it also sets `u` and `u_best`, which MATLAB sets through `u = ubest` (`bads.m:955`).
  - The success lists (`u_success` and the others) match.
- **Mesh.** All of these match:
  - Expansion: `msi + 1`, capped at `max_poll_grid_number`, with the overflow check made before the increment.
  - Reduction: `msi − 1`.
  - Accelerated reduction: an extra −1 when the improvement since `iter − accelerate_mesh_steps` is below `tol_fun`. Python's `iter >= A` corresponds to MATLAB's 1-based `iter > A`, and Python's index `iter − A` addresses the same stored iteration.
  - Search mesh: `min(ssi, msi·multiplier − number)`.
  - Mesh size: recomputed at the end of the poll. The search mesh size is recomputed only at the loop head, on both sides.
- **The state the poll leaves.** `reset_gp` answers a move of the incumbent once. MATLAB's `pollmoved_flag` persists across loop passes (F5).

### 4. The target
- **Default path.** At level ≥ 1, or with `uncertain_incumbent` on (the default), both sides predict at the given point under `hyp_best`. The search predicts at `u_best` with `best_gp_hyp` (MATLAB's `fhyp`); the poll at `u_poll_best` with `gp_poll_hyp_best` (MATLAB's `fpollhyp`).
  - MATLAB keeps the current posterior and swaps in the kernel and mean of `hyp_best`. Python recomputes the posterior under `hyp_best`: see F4, the open half of KD-B4-2.
  - The `LinAlgError` fallback matches KD-B4-2.
- **Terms of the formula.**
  - `np.max(fs2, axis=0)` is not MATLAB's `max(fs2, 0)`, but gpyreg already clamps the variance, so it makes no difference.
  - A non-finite prediction falls back to `optim_state["fval"]` and `["fsd"]`. The target itself still uses the raw `fs2`, exactly as `bads.m:1321` does, and `optim_state["fval"]` goes stale after a re-estimate the same way on both sides.
  - `alternative_incumbent`, `sd_level·sqrt(fs2 + tol_fun²)` and the deterministic formula `fval − tol_fun` all match. The deterministic branch crashes, though (F3).
- **Where the target is used.** At default options the target enters only the poll's `p_less`, on both sides. MATLAB's `searchES` reads it only for `acqNegEIMin` and `acqNegPIMin`, and nothing in Python reads `optim_state["f_target"]` outside the poll (line 2274). So the search's target computation has no effect at default options.

## 3. Findings

### F1. The poll's probability that no point improves (`p_less`) uses unsorted probabilities, and D+1 of them instead of MATLAB's D
- Location: `pybads/bads/bads.py:2281-2284`. MATLAB: `bads.m:862-869`.
- Category: indexing/shape.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: yes, at every poll step and at all levels. It decides something only after a good poll with a reliable GP.
- History: the MATLAB lines date from 2017 and are unchanged. The Python lines are unchanged since `c7c88ab` (2022-06-02), apart from formatting. gpyreg's `predict` has returned (N, 1) columns since 2021. Python never matched MATLAB here.
- What the code does:
  - `f_mu` and `fs` from `acq_fcn_lcb` are (n, 1) columns, and every run showed `gamma_z` with shape (n, 1).
  - `np.sort(f_pi)` sorts along the last axis, which has length 1, so it sorts nothing. `[::-1]` only reverses the rows.
  - `f_pi[0:min(D+1, n)]` therefore takes the **last** D+1 poll points in the set's order, which is lexicographic after `contraints_check`, and not the most probable ones.
  - MATLAB sorts the row vector in descending order and takes `1:min(nvars,end)`: the D largest probabilities. The Python `D + 1` looks like a misreading of the 1-based range.
- Consequence:
  - When more than D+1 points remain (possible for D ≥ 3 in the first D−2 steps after a good poll), a point with a high probability of improvement can be left out, and the poll can stop too early.
  - When fewer remain, Python multiplies over more terms and stops less often.
  - Measured:
    - 20 default runs: 80 stop decisions where `p_less` mattered, no flipped decision.
    - 8 more default runs: in 3 of the 13 decisions with more than D+1 points left, the largest probability was left out, but no decision flipped.
    - 8 runs with `tol_poi=0.01`: no flipped decision.
  - The rule is wrong, but its measured effect is small because the default threshold, 1 − 1e-6/D, is extreme.
- Reproduction: `check_pless.py`. With D=3, six points, the first with probability 0.69 and the others near 0: PyBADS gives `p_less = 0.9999999961` and stops; MATLAB's rule gives 0.309 and does not stop.
- Test adequacy: no test checks `p_less`.

### F2. The poll set keeps points already evaluated: the check in `contraints_check` removes nothing
- Location: `pybads/function_logger/constraints_check.py:33-43`, which the poll calls at `bads.py:2166-2174`. MATLAB: `utils/uCheck.m:17-27`.
- Category: control flow.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: yes, at all levels. The function is slice B3's; the poll depends on it.
- History: MATLAB `uCheck.m` is unchanged since 2017 (`a8817f9`). The first port, `c7c88ab` (2022-06-02), had a working set difference based on L1 distance. `8e59038` (2022-06-03) replaced it with `np.unique` over `[u1; u2]`, which returns each row's first occurrence. A row that is both a candidate and already evaluated occurs first in `u1`, so it is kept. The test comment cites the codebase survey, which lists this as a candidate defect; the sheet does not list it.
- Consequence for the poll:
  - At level 0, no poll set in the 20 runs contained an evaluated point (0 of 103 poll sets).
  - At level 1, 4 of 97 poll sets did, and 4 of 635 poll evaluations went to points already in the log. MATLAB never re-polls such a point.
  - In those two runs, the repeated rows then sat in 128 of 138 and 98 of 232 rebuilt training sets.
  - At level 0, an evaluated point that stayed in the set would be evaluated last in a complete failed poll, and a repeated input with little noise invites the Cholesky failures of KD-B6-6.
- Reproduction: `check_ucheck.py`. Four poll points, two of them already in the log: all four are kept.
- Test adequacy: `test_search.py::test_incumbent_constraint_check` asserts the defective behaviour.

### F3. `uncertain_incumbent=False` on a deterministic target crashes at the first poll
- Location: `bads.py:2696-2699` returns Python floats; the callers call `.item()` on them at 2245 and 2249 (poll) and 1748 and 1752 (search). MATLAB: `bads.m:1328-1332`, which works.
- Category: control flow.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: no. It is reached with `uncertain_incumbent=False` at level 0.
- History: the branch and `f_target.item()` date from `c7c88ab`, and `f_target_mu.item()` from `9037851` (2022-09-22). `optim_state["fval"]` is a Python float after `_init_mesh_`, which calls `.item()` at line 1108. The branch has never run.
- Consequence: `AttributeError: 'float' object has no attribute 'item'` at line 2245, whether the target returns a float or `np.float64`. The option is unusable.
- Reproduction: `check_uncertain_incumbent_off.py` raises for both target types.
- Test adequacy: no test sets `uncertain_incumbent=False`.

### F4. The target under `hyp_best`: MATLAB mixes the current posterior with `hyp_best`'s kernel and mean; PyBADS recomputes the posterior
- Location: `bads.py:2659-2662`. MATLAB: `bads.m:1300-1302`, through `utils/gppred.m:39-47` and `utils/mygp.m:122-123`, `146-187`. When `post` is not empty, `mygp` reuses `alpha`, `L` and `sW` and evaluates `Ks`, `kss` and the mean under the new `hyp`.
- Category: formula.
- Proposed classification: port discrepancy. This is the half that KD-B4-2 leaves open. MATLAB's hybrid is not a GP prediction under any single set of hyperparameters, so MATLAB's form looks unintended.
- Confidence: medium. MATLAB was emulated with gpyreg's posterior. With the current hyperparameters the emulation reproduces `gp.predict`, and this was checked at every use.
- Reached at default options: yes, but it matters only when `hyp_best` differs from the current hyperparameters at a poll stop decision after a good poll. That is, a refit inside the poll after its best point was found. It also happens after a move to an earlier iterate at level ≥ 1, and in MATLAB's first poll, which uses the defined hyperparameters (KD-B6-5), though no stop can occur there.
- History: MATLAB's lines are from 2017, unchanged. Python predicted from the current GP at `c7c88ab`, recomputes under `hyp_best` since `9037851`, and gained the fallback in `685da15`. It never matched MATLAB.
- Consequence:
  - Level 0: none of 28 relevant decisions had different hyperparameters.
  - Level 1: 9 of 52 did, and MATLAB's target would flip 4 stop decisions.
  - The hybrid targets are far from PyBADS's: 1.25e6 against 143 on Rosenbrock D=4; 1.24 against −0.08, −0.70 against 0.11, and −3.19 against 0.56 on the ellipsoids.
- Reproduction: `instrument_runs.py`, the `hyp_diff` and `flip_hyb` columns.
- Test adequacy: the tests in `test_gp_update_failures.py` cover only the fallbacks.

### F5. MATLAB's `pollmoved_flag` persists across loop passes; PyBADS's `reset_gp` answers a move once
- Location: `bads.py:2498`, `1734-1737` and `1987`. MATLAB: `bads.m:956-958` and `1049`. The flag is set only inside the poll and is read at the end of every loop pass.
- Category: state/caching.
- Proposed classification: port discrepancy. The MATLAB side looks unintended.
- Confidence: medium.
- Reached at default options: yes, at all levels.
- History: MATLAB `0c2bb80a` (2017); Python since `c7c88ab`. They never matched. The comment at line 1734, "as in MATLAB BADS", holds only for the first rebuild.
- What differs: after a poll that moved the incumbent, MATLAB clears the posterior at the end of every loop pass until a poll that does not move. So every later search rebuilds the local GP, without a refit. PyBADS rebuilds once, which the first search does anyway because `search_count == 0`.
- Consequence: without a refit the rebuild keeps the hyperparameters, so it changes the posterior only when the nearest-neighbour set differs from the GP's incrementally grown set, which becomes possible once more than about `n_train_min` points exist. Measured: in 95 such searches over the 20 runs, the set never differed. No effect within 200 evaluations; longer runs may differ.
- Reproduction: `instrument_runs.py`, the `sticky_checks` and `sticky_diff` columns.
- Test adequacy: none.

### F6. With `poll_training=False`, the poll neither records a refit it does not make nor clears the unreliability flag; MATLAB does both
- Location: `bads.py:2195-2200`, and `2516-2586` (`refit_allowed`). MATLAB: `bads.m:822-823`, `1242-1252`.
- Category: control flow.
- Proposed classification: possibly intentional. The changelog of `fef6c14` documents it ("Refits without poll training … MATLAB BADS does the same"), but the sheet has no entry for it.
- Confidence: high.
- Reached at default options: no. It is reached with `poll_training=False` after the first iteration.
- History: the Python change is from `fef6c14` (2026-09-26). MATLAB is unchanged since 2017.
- What differs: MATLAB's `IsRefitTime` sets `lastfitgp`, resets `gpstats` and sets `unrelgp_flag = 0` before `refitgp_flag` is forced to false. PyBADS leaves all three alone, so the stop rule reads the flag as computed.
- Consequence: different stop decisions and refit timing, only with that option.
- Test adequacy: not checked against MATLAB.

### F7. `_eval_improvement_` accepts `improvement_quantile` outside (0, 1), which MATLAB refuses
- Location: `bads.py:2027-2037`. MATLAB: `bads.m:1269-1271`.
- Category: defaults.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: no. It is reached with `improvement_quantile ≤ 0` or `≥ 1`.
- History: MATLAB `d04640ae` (2017); Python has lacked the check since `c7c88ab`.
- Consequence: at q = 0 or q = 1, `erfcinv` returns ±inf. At level 0 the improvement becomes 0·inf = NaN, so the incumbent never moves, and the run ends without an error. On a 2-D sphere the run used all 100 evaluations and ended at the best initial point, f = 7.57.
- Reproduction: `check_quantile.py`.
- Test adequacy: none.

## 4. Test adequacy notes
- `test_search.py::test_incumbent_constraint_check` asserts that evaluated rows are kept. It mirrors the defect of F2, not `uCheck`.
- `test_poll_mads.py` checks integrality, determinant ±n_max^D and the negation. That is structure consistent with LTMADS, but it does not check the entry ranges or the permutations against MATLAB, and it does not check that `poll_scale` cancels in `_poll_step_`.
- No test covers `p_less` (F1), the `uncertain_incumbent=False` branch (F3), the target under a `hyp_best` that differs from the current GP's hyperparameters (F4), or the stickiness of the move flag (F5).
- `test_run_control.py::test_accelerated_mesh_reduction_counts_iterations_from_one` checks the iteration offset of the accelerated reduction against MATLAB's rule, which is adequate.
