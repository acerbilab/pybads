<!-- Report of the B3 comparison reviewer (search, MATLAB-comparison track), wave 3 of the port review, reading PyBADS at 8aecb6a in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session, with the complete history; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave3/B3_comparison/. Nothing in it is verified. -->

# B3 comparison review: search

Scripts and outputs are in `/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/review/B3_comparison/`. Every script printed `pybads.__file__` = `/home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__` = `/home/user/gpyreg-v1.3.3/gpyreg/__init__.py`.

## 1. Coverage

**Read completely, Python at `8aecb6a`:**
- `pybads/search/search_hedge.py` and `pybads/search/es_search.py`.
- `pybads/search/grid_functions.py`: `force_to_grid` and `udist`.
- `pybads/acquisition_functions/acq_fcn_lcb.py` and `pybads/function_logger/constraints_check.py`.
- `pybads/bads/bads.py`: the head of the `optimize` loop (1242-1420), `_search_step_` (1686-2016), `_eval_improvement_`, `_update_search_bounds_`, `_update_incumbent_`, `_update_search_stats_`, `_get_target_from_gp_` (read only where the search depends on it), and the search state set up in `_init_optim_state_` (600-770).
- Every search option in the two `.ini` files.

**Read completely, MATLAB at `74919c0`:**
- `bads.m`: 478-770 (the loop head, the search stage and the poll decision) and 1199-1430 (`savegpstats`, `IsRefitTime`, `EvalImprovement`, `UpdateIncumbent`, `UpdateTarget`, `UpdateSearch`, `updateSearchBounds`).
- `searchHedge.m`, `searchES.m`, `acqLCB.m`, `acqPortfolio.m`, `ESupdate.m`, `uCheck.m`, `force2grid.m`, `udist.m` and `ucov.m`.
- `setupvars.m:165-192` and `evalinitmesh.m:95-125`.

**Skimmed:**
- `FunctionLogger.__call__`. `X` is in u space, and so is MATLAB's `optimState.U`.
- The poll's and `_init_mesh_`'s calls of `contraints_check`.
- The callers of `udist` in `gaussian_process_train.py` and `gpdefBads.m`.
- gpyreg's `predict` signature: it returns the latent variance by default, as `gppred`'s `fs2` does.

**Not reached:** the internals of `local_gp_fitting` and `add_and_update_gp` (B5), `_is_gp_refit_time_` and `_save_gp_stats_` (B5), the Sto-BADS branches (slice S), and the poll beyond its call of `contraints_check` (B4).

**History:** none of the MATLAB search files changed after 2022-02-11, except in these ways:
- `acqPortfolio.m` lost commented-out lines in `d4fead5`.
- The search stage of `bads.m` was edited in `d4fead5`: `gpTrainingSet` was renamed `gpupdate`, `ysearch_sd` was added, and the `HessianUpdate` block was removed.

All Python discrepancies below therefore date from the Python side.

**The sheet's B3 entries (KD-B3-1 to KD-B3-4) match the code as they describe it,** and so do the search draw sites of KD-B1-1.

## 2. Answers to the first questions

### 1. The search set

**The choice of strategy matches `searchHedge.m:30-60`.** The gains start at `g = [10, 0]`. The probabilities are `p = softmax(β(g − max g))·(1 − nγ) + γ`, with β = 1e-3/`tol_fun` and γ = 0.125. The strategy is drawn as the first index with `rand < cumsum(p)`, with a uniform fallback. `phat` is Inf for the strategies not chosen, or all ones when γ = 0.

**`ES-ell` matches `searchES.m:86-94`,** apart from the rotated-GP branch, which is not the default. Its covariance is the diagonal of `poll_scale`, normalized to unit norm.

**`ES-wcm` differs from `searchES.m:41-84` in the number of best points.** The weights (`log(μ+½) − log(1..⌊μ⌋)`, normalized), the eigen-rescaling (eigenvalues clipped at 0, plus `mesh_size²` jitter, divided by their sum under the sum rule) and `sqrtsigma = diag(√λ)Eᵀ` all match. But Python takes ⌊μ⌋+1 best training points where MATLAB takes ⌊μ⌋ (F3). Both sides' `ucov` drop the weights (F4).

**The scale and the initial population match.**
- The scale is `mesh_size · search_factor`.
- The initial population is N = `n_search/n_search_iter` = 2048 draws, half of them at half scale (`poll_mesh_multiplier^[-1,0]`).
- `es_start` = 0.25 is the reproduction scale.

**The `es_beta` scale update and the fraction of new points never run at default.** With `n_search_iter` = 2 the update needs 0 < i < 1. The count behind it is wrong for `n_search_iter` ≥ 3 (F7).

**Selection and reproduction differ.** The offspring's parents are shifted one rank toward worse candidates compared with `ESupdate.m` (F2). This is reached in every search.

**Ordering and the returned point mostly match.**
- The candidates are sorted by LCB.
- The returned point is the best candidate. An empty set is returned when no candidate is left, as `searchES.m:209` does.
- Python keeps every earlier candidate in `us_candidates` where MATLAB trims to λ. The top N are the same.
- Python's `np.argsort` is not stable, while MATLAB's `sort` is. This matters only for exact ties.
- When the last iteration's offspring are all removed, Python returns an empty set, where MATLAB returns the best of the earlier iteration (F6).

### 2. The grid and the checks

**`force_to_grid` matches `force2grid.m`.** NumPy rounds halves to even and MATLAB rounds them away from zero. For the search bounds this makes no difference, because the correction after rounding (`lb_search < lb` gives `+= search_mesh_size`) yields the same bound either way. Candidates are continuous draws, so exact halves have zero probability.

**`_update_search_bounds_` matches `updateSearchBounds`,** at the loop head and at setup.

**`contraints_check` matches `uCheck.m` except in one step:**
- The projection onto `lb_search`/`ub_search` (proj = 1) and the bounds check (proj = 0, on the actual bounds) match.
- Rounding to bins of `tol_mesh/2` (`tol_mesh` is 2^ceil(log2 1e-6), as in `setupvars.m:105`) matches.
- One point is kept per bin, in lexicographic order, as `setdiff(...,'rows')` does.
- The `non_box_cons` filter `C <= 0` in original space matches.
- **But points already evaluated are not removed** (F1). What remains is every candidate bin, evaluated ones included.

**`udist` matches `udist.m`** for non-periodic variables: the squared scaled Euclidean distance. Python returns the pairwise matrix, which is what MATLAB's callers build with their 3-D `temp` trick. The periodic branch is unreachable (KD-B1-6).

### 3. The choice, the evaluation and the decision

**LCB matches `acqLCB.m`:** `t = func_count + 1` and `√β = √(0.4·log(D t² π²/0.6))`, applied to the latent SD. Since the ES returns a single point on both sides, the LCB choice in `_search_step_` is trivial. The real selection happens inside the ES.

**Error handling differs:**
- MATLAB catches a failed acquisition. `bads.m:583-589` then picks a random index, and `searchES.m:149-168` uses random `z`.
- Python has no `try`: an exception stops the run. KD-B5-3 lists these sites as having "no recorded decision", and I could not construct a reachable failure.
- NaN values sort last on both sides.
- Python's random-search branch in the ES is dead and is misused when a candidate set is empty (F6).
- A non-default float LCB parameter crashes (F9).

**Evaluation, the GP update and the noisy estimate match `bads.m:624-665`:**
- The point is added to the GP except at the round's last search.
- At level ≥ 1, a copy of the GP is rebuilt around the point (without a refit) and predicts there. A failed rebuild gives NaN, and the search then counts as a failure.
- `search_dist` is taken from the incumbent before the move, with `len_scale`.

**The decision matches `bads.m:676-712`:**
- The improvement comes from `_eval_improvement_` with `improvement_quantile` (at 0.5 it is `fval − f_search`).
- Success means an improvement above `tol_improvement·mesh^1.5` (with `sloppy_improvement`, at least `tol_fun`). An incremental improvement is any improvement > 0.
- On either, the incumbent moves, `gp = new_gp` at level ≥ 1, and the GP is rebuilt at the next step (the counterpart of `gpstruct.post = []`).

**An empty search set gives y = `yval`, f = `fval`, SD = 0 and distance 0 on both sides, and a failure at default.** Python also forces a failure for `improvement_quantile` > 0.5, where MATLAB counts an incremental improvement and moves to a stale `usearch` (F10).

**A failed GP add leaves the point out of the GP until the next rebuild** (KD-B5-1).

**One B2 item the search depends on:** `do_search_step_flag` counts the points in the function log (`bads.py:1349-1353`), where MATLAB counts `size(gpstruct.y,1)`. The two are equal in practice, since the training set holds every point while there are at most `n_train_min` = 50 of them.

### 4. The hedge's rewards and the search statistics

**`update_hedge` follows the `'upd'` branch in structure:**
- Each gain is updated as `g ← decay·g + er/phat/mesh_size`, with `decay = 0.1^(1/(2D))`.
- With γ > 0, only the chosen strategy gets a nonzero reward; the others decay, since er/Inf = 0.
- The chosen strategy is scored from the search's estimate: y at level 0, the rebuilt GP's estimate at level ≥ 1, compared against `fval` before the move.
- At level 0, where the SD is 0, er = max(0, `fval_old − f`), exactly as in MATLAB.

**It differs in these places:**
- The expected-reward formula at levels 1 and 2 has a misplaced parenthesis (F5).
- γ = 0 crashes, and so does MATLAB, for another reason (F8).
- On an empty set Python skips the update, where MATLAB decays the gains with a stale `usearch` (F10).

**`_update_search_stats_` matches `UpdateSearch` line for line:**
- `search_scale_success` (√2), `search_scale_incremental` (2) and `search_scale_failure` (√½, floored at `search_factor_min` = 0.5).
- The reset to 1 when `search_count == search_n_try`.
- `sd_level` ×2 / ×4 / halved and floored at `incumbent_sigma_multiplier` under `adaptive_incumbent_shift`.
- The statistics lists.

**The loop head matches `bads.m:490-507`:** the mesh size, `search_size_integer` (when locked), `search_mesh_size`, the search bounds and the sufficient improvement.

## 3. Findings

### F1. `contraints_check` does not remove points already evaluated
- Location: pybads/function_logger/constraints_check.py:33-43; MATLAB: utils/uCheck.m:17-27 (the `setdiff` at 25)
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, at every level, in every ES iteration, in the final check of `_search_step_` (bads.py:1787), in the poll (bads.py:2166) and in the initial design (bads.py:1091).
- History (comparison track): `uCheck.m` is unchanged since `a8817f9` (2017). The first port, `c7c88ab` (2022-06-02), removed matches with an L1 test and so agreed with MATLAB, apart from the per-bin dedup. `8e59038` (2022-06-03) replaced it with the current `np.unique` code, and it has never matched since. `pybads/testing/bads/search/test_search.py:16-31` notes the behaviour as a "candidate defect, in dev/results/2026-09-23-codebase-survey.md". It is not on the sheet.
- What the code does:
  - It stacks the candidates' bins `u1` over the log's bins `u2` and takes `np.unique(..., return_index=True)`. That returns each distinct row's first occurrence, which for any row of `u1` is in `u1`, whether or not it is also in `u2`.
  - It then keeps `idx < len(u1)`, which is every distinct candidate bin.
  - MATLAB's `setdiff(u1,u2,'rows')` drops the candidates whose bin was evaluated.
- Consequence if real:
  - Candidates at already evaluated points stay in the ES population and can be chosen and evaluated again: wasted evaluations and duplicate GP training points.
  - Default runs: on the sphere at D = 1 (30 evaluations), 1 or 2 of the 16 search evaluations re-evaluate an existing point exactly (4/4 seeds), the incumbent included.
  - At D = 2-4, 0 repeats in 16 deterministic runs (sphere, Rosenbrock, ellipsoid, up to 200 evaluations).
  - Noisy sphere at D = 3: one poll point repeated in 3 runs.
- Suggested reproduction:
  - `check_ucheck.py` gives candidates [[0,0],[.5,.25],[.125,.125]×2,[2,0]] with [0,0] and [.5,.25] already evaluated.
  - Python keeps [0,0], [.125,.125], [.5,.25] and [1,0] (the last after projection).
  - A transcription of `uCheck.m` keeps [.125,.125] and [1,0].
  - `probe_dups_lowD.py 1` shows the repeated evaluations in runs.
- Test adequacy: `test_incumbent_constraint_check` asserts the defective behaviour: only the duplicate is removed.

### F2. The ES reproduction's selection mask is shifted by one parent rank
- Location: pybads/search/es_search.py:70-74 (used at 208-217); MATLAB: utils/ESupdate.m:18-21, search/searchES.m:198-201
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, in every search (one reproduction per search at `n_search_iter` = 2), at every level.
- History (comparison track): the Python lines were written in `c7c88ab` (2022-06-02) and are unchanged in substance. `ESupdate.m` is unchanged since `12f7ff8` (2017). The two never agreed.
- What the code does:
  - `cw` holds MATLAB's 1-based start positions and is used unconverted as 0-based indices into `idx`, whose element 0 stays 0.
  - `cumsum(idx[0:-1])` then equals `[0, M(1), M(2), …]`, where `M` is MATLAB's 1-based `selectmask`.
  - So Python's parent for offspring j ≥ 1 is MATLAB's parent for offspring j−1, plus one rank. It should be `M − 1`.
- Consequence if real:
  - At μ = λ = 2048, the best candidate gets 1 offspring instead of 17, and ranks 1, 2, 3… get 17, 12, 10… instead of 12, 10, 9…. The allocation slides one rank toward worse parents.
  - I ran both masks on 14 GP states captured from a Rosenbrock D = 4 run, 5 generator seeds each, 70 pairs per strategy.
  - The MATLAB mask found a lower best LCB in 56% (ES-wcm) and 53% (ES-ell) of pairs. The Python mask did so in 27% and 19%; the rest were ties.
  - The median difference is small (−2e-4, −5e-4). The mean difference (−143, −71) comes from the early states.
- Suggested reproduction: `check_esupdate.py` (a transcription of `ESupdate.m`; Python equals `[0] + MATLAB[:-1]` for every μ tried) and `paired_mask.py`.
- Test adequacy:
  - `test_search_selection_mask` expects sum = 885072. That is the sum of MATLAB's 1-based mask, and Python matches it only because of the shift. A correct 0-based port sums to 883024 and would fail the test.
  - With μ = 1 the shift has no effect on what is drawn.

### F3. ES-wcm builds its covariance from ⌊μ⌋+1 best training points, not ⌊μ⌋
- Location: pybads/search/es_search.py:240-248 (247); MATLAB: search/searchES.m:55-62
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, in every ES-wcm search (the strategy chosen most often), at every level.
- History (comparison track): written in `c7c88ab`; `searchES.m` is unchanged since `603da99` (2017). The two never agreed.
- What the code does:
  - `y_idx[0 : floor(mu+1)]` takes ⌊μ⌋+1 rows, while the weights have ⌊μ⌋ entries.
  - Broadcasting in `ucov` hides the mismatch.
  - MATLAB takes `index(1:floor(mu))`.
- Consequence if real: in 16 states of a Rosenbrock D = 3 run, the normalized search covariance differs from a transcription of `searchES.m` method 1 by 1-17% (relative Frobenius). A transcription with ⌊μ⌋+1 rows reproduces Python exactly.
- Suggested reproduction: `check_wcm_cov.py`.
- Test adequacy: none. `test_search` checks only shapes and `gp.y >= z`.

### F4. `ucov` ignores its weights: the "weighted covariance" is unweighted, on both sides
- Location: pybads/search/es_search.py:324-329; MATLAB: utils/ucov.m:19
- Category: formula
- Proposed classification: suspected defect in both
- Confidence: high (that the weights have no effect); medium (that weighting was intended)
- Reached at default options: yes, in every ES-wcm search.
- History (comparison track):
  - `ucov.m` is unchanged since `12f7ff8` (2017).
  - Python transcribed it literally in `c7c88ab` and rewrote it in `e404fcc` (2023-07-06), still unweighted.
  - Python and MATLAB agree.
- What the code does:
  - MATLAB computes `sum(bsxfun(@times, weights, ushift'*ushift), 3)`, which is `Σw · UᵀU = UᵀU`. Python's `matmul(uᵀ, w[:,None,None]*u)` summed over axis 0 gives the same result.
  - The comments ("Compute weighted covariance matrix wrt u0"), the computed CMA-style log weights and the name "wcm" point to `Σᵢ wᵢ sᵢsᵢᵀ`.
- Consequence if real: in the same 16 states, weighting would change the normalized covariance by 22-56%. Fixing it would move PyBADS away from MATLAB, so this needs a decision, not a port fix.
- Suggested reproduction: `check_ucov.py` (Python `ucov` equals the unweighted `SᵀS`, not the weighted sum) and the last lines of `check_wcm_cov.py`.
- Test adequacy: `test_u_cov` checks only the shape.

### F5. The hedge's expected reward uses exp(−γ²/(2√(2π))) instead of φ(γ) = exp(−γ²/2)/√(2π)
- Location: pybads/search/search_hedge.py:151-155; MATLAB: acq/acqPortfolio.m:64
- Category: formula
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, at uncertainty levels 1 and 2, for the chosen strategy's reward. At level 0 its SD is 0 and the `er = max(0, fval_old − f)` branch runs instead. Strategies not chosen get er/Inf = 0 either way.
- History (comparison track): the formula has been wrong since `c7c88ab`. `acqPortfolio.m:64` is unchanged (`d4fead5` removed comments only). The two never agreed.
- What the code does: `np.exp(-0.5*(gamma_z**2)/np.sqrt(2*np.pi))` puts √(2π) inside the exponent.

  | γ | Python er/σ | MATLAB er/σ |
  |---|---|---|
  | 0 | 1.000 | 0.399 |
  | −2 | 0.405 | 0.0085 |
  | +2 | 2.40 | 2.01 |

- Consequence if real: failed searches still earn sizeable rewards, and the hedge discriminates less. On a noisy sphere at D = 3 (level 1), the median reward ratio Python/MATLAB was 2.6-2.8. The share of ES-wcm choices moved only slightly in 3 seeded runs (0.82/0.61/0.69 against 0.74/0.60/0.67 with MATLAB's formula), so the effect is small.
- Suggested reproduction: `check_reward.py`.
- Test adequacy: no test exercises `update_hedge`'s formula.

### F6. When every offspring of the last ES iteration is removed, the ES returns an empty set, not the best earlier candidate
- Location: pybads/search/es_search.py:170-175 (the branch overwrites `z_candidates`), 185, 196-197; MATLAB: search/searchES.m:168-182
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no. It needs a `non_box_cons` that rejects every offspring of an iteration after the first (the check removes nothing else in bulk).
- History (comparison track): the overwrite has been there since `c7c88ab`. Before `0c56d86` (2026-09-26) the empty `us` raised `IndexError` at `us[0]`; since then an empty set is returned. `searchES.m` is unchanged since 2017.
- What the code does:
  - When a candidate set is empty (`z_new.size == 0`), the "random search" branch sets `z_candidates = rng.random(0)`, discarding the accumulated values, and logs "Something went wrong with the acquisition function, random search is performed".
  - The append then leaves `z_candidates` empty while `us_candidates` still holds the earlier rows, so `us` becomes empty and the loop breaks.
  - MATLAB keeps `zold` and returns the earlier best.
  - The `z_new is None` case of the branch is dead, and would crash at `z_new.copy()`.
  - The warning is also logged, spuriously, whenever a candidate set is empty.
- Consequence if real: a wasted search round (a failure) under a tight constraint, and a misleading warning.
- Suggested reproduction: `check_edge.py` (b). A constraint that accepts everything at its first call and nothing afterwards makes `ESSearchWM` return shape (0, 3); MATLAB would return the best of iteration 1.
- Test adequacy: `test_empty_search.py` patches the whole check to return nothing, so it never covers an empty later iteration.

### F7. The ES's fraction of new candidates is miscounted, which breaks the `es_beta` scale adaptation for `n_search_iter` ≥ 3
- Location: pybads/search/es_search.py:177-205 (177, 182-185, 191-192); MATLAB: search/searchES.m:170-193
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no. At `n_search_iter` = 2 the scale update never runs. It needs `n_search_iter` ≥ 3 and is badly off from 4.
- History (comparison track): written in `c7c88ab`. MATLAB is unchanged since 2017. The two never agreed.
- What the code does:
  - `n_new = sum(z_idx[0:ntest+1] > nold)` takes ntest+1 entries and misses position `nold`. MATLAB's 1-based `index(1:ntest) > nold` corresponds to `z_idx[:ntest] >= nold` in 0-based indexing.
  - `us_candidates` accumulates all iterations, untrimmed, so from the third iteration on, old rows at positions above `nold` count as "new".
  - `nold` at i = 0 is μ, not 0. This is harmless.
  - `n_search_iter` = 0 would return an uninitialized `np.empty` row.
- Consequence if real: with lam = 2048 and a drifting z, the fraction per iteration is 0.56 / 0.43 / 0.39 / 0.37 in MATLAB and 0.56 / 0.77 / 0.87 / 0.93 in Python, so the scale grows where MATLAB's would shrink.
- Suggested reproduction: `check_frac.py`.
- Test adequacy: none.

### F8. `hedge_gamma = 0` crashes: `update_hedge` slices the point's coordinates (MATLAB crashes too, differently)
- Location: pybads/search/search_hedge.py:127, 132-134; MATLAB: acq/acqPortfolio.m:40, 44-51
- Category: indexing/shape
- Proposed classification: suspected defect in both
- Confidence: high
- Reached at default options: no (`hedge_gamma = 0`).
- History (comparison track): written in `c7c88ab`. MATLAB is unchanged since 2017.
- What the code does:
  - `u_search` is 1-D, so `u_search[min(i, len-1):]` drops coordinates for i ≥ 1, and `gp.predict` raises "The dimension of input data 2 doesn't match GP's input dimension 3".
  - MATLAB's `u(min(iHedge,end),:)` picks a row, but line 47 reads `gpstructnew`, which is undefined in `acqPortfolio`, so MATLAB errors as well.
- Consequence if real: a user who sets `hedge_gamma = 0` gets a crash at the second search.
- Suggested reproduction: `check_opts.py`, or `check_edge.py` (d).
- Test adequacy: none.

### F9. A plain float as the LCB parameter crashes the search
- Location: pybads/acquisition_functions/acq_fcn_lcb.py:42; MATLAB: acq/acqLCB.m:10-21
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no (`search_acq_fcn = ('acq_LCB', <float>)`). The poll always passes `None`.
- History (comparison track): written in `c7c88ab`. `acqLCB.m` is unchanged since 2017.
- What the code does:
  - `~np.isfinite(1.0) or (1.0).size` raises `AttributeError: 'float' object has no attribute 'size'`. A `np.float64` works.
  - Non-finite values are refused, which MATLAB accepts as any numeric scalar.
  - A string name (MATLAB `feval`s it) fails inside `np.isfinite`.
  - Separately, `_search_step_` (bads.py:1801) ignores the parameter. That is harmless, since it scores a single candidate.
- Consequence if real: a crash at the first search for the natural way of setting the option.
- Suggested reproduction: `check_opts.py`: `('acq_LCB', 1.0)` raises `AttributeError`, `('acq_LCB', np.float64(1.0))` runs.
- Test adequacy: none.

### F10. Empty search set: Python fails the search and skips the hedge update; MATLAB, off its defaults, does neither
- Location: pybads/bads/bads.py:1956-1958, 1996; MATLAB: bads.m:667-672, 676-681, 722-725 (`usearch` is first assigned at 621)
- Category: control flow
- Proposed classification: possibly intentional (Python avoids a MATLAB defect; not on the sheet)
- Confidence: medium
- Reached at default options: no. The search set must be empty (only through `non_box_cons`, given F1). The move also needs `improvement_quantile` > 0.5, with an incumbent SD > 0.
- History (comparison track): the forced failure was written in `0c56d86` (2026-09-26), whose message and comment say "as in MATLAB BADS … on every path". The hedge guard dates from `c7c88ab`. MATLAB is unchanged.
- What the code does:
  - With an empty set, MATLAB's improvement is `σ·x0(q)`, which is positive for q > 0.5. It then declares an incremental search and moves the incumbent to the stale `usearch` of an earlier search, with that search's point but f = `fval` and SD 0.
  - MATLAB also runs the hedge update with the stale `usearch`, which decays all gains. If the first search is empty, `usearch` is undefined and MATLAB errors.
  - Python always counts a failure and leaves the gains untouched. "As in MATLAB" holds only at q ≤ 0.5.
- Consequence if real: none at default. The code differs from MATLAB off its defaults, and the comment overstates the match.
- Suggested reproduction: reasoning from `bads.m:667-725`; there is no MATLAB here to run it.
- Test adequacy: `test_empty_search_set_is_a_failed_search` checks the failure at default q only.

## 4. Test adequacy notes

- **`test_incumbent_constraint_check`** (`test_search.py:16-44`) asserts that previously evaluated rows survive `contraints_check`: it encodes F1 and would fail on a correct port.
- **`test_search_selection_mask`** checks `sum(mask) == 885072`. That is MATLAB's 1-based `selectmask` sum, which the shifted Python mask reproduces (F2). A correct 0-based mask sums to 883024. With μ = 1 the test cannot show the shift in what is drawn.
- **`test_u_cov`** checks only the shape of `ucov`, not its value, so neither F3 nor F4 would be caught.
- **`test_search` and `test_search_hedge`** check only shapes and `gp.y >= z`. There is no test of the ES-wcm covariance, the ES scale update or the reward formula of `update_hedge` (F5).
- **`test_empty_search.py`** replaces the constraint check wholesale or returns an empty set from the hedge. It does not cover an empty later ES iteration (F6) or `improvement_quantile` > 0.5 (F10).
- **`test_failed_searches_floor_the_search_factor`** follows `UpdateSearch` as specified (floor, scale factors, reset) and would catch errors in `_update_search_stats_`.
- **Outside the numerical scope:** `ESSearch.__init__` (es_search.py:48-49) calls `logging.basicConfig(stream=sys.stdout, ...)` at every construction. When the user's root logger has no handlers, this configures it globally.
