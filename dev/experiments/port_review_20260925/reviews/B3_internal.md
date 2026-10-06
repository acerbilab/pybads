<!-- Report of the B3 internal reviewer (search, internal-correctness track), wave 3 of the port review, reading PyBADS at 8aecb6a in ../pybads-review and gpyreg v1.3.3 (98ab5a4), in a cloud session, with the complete history; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave3/B3_internal/. Nothing in it is verified. -->

# B3 internal review: search

## 1. Coverage

**Read completely** (worktree `/home/user/pybads-review` at `8aecb6a`):
- `pybads/search/search_hedge.py`, `pybads/search/es_search.py`, `pybads/search/grid_functions.py`.
- `pybads/acquisition_functions/acq_fcn_lcb.py`, `pybads/function_logger/constraints_check.py`.
- In `pybads/bads/bads.py`:
  - the head of the loop in `optimize`, lines 1242-1530;
  - `_search_step_`, 1686-2016;
  - `_eval_improvement_` and `_sto_success_improvement_`, 2018-2086;
  - `_save_gp_stats_`, `_is_gp_refit_time_`, `_record_gp_refit_`, `_get_target_from_gp_`;
  - `_update_search_bounds_`, `_update_incumbent_`, `_update_search_stats_`, 2502-2788;
  - the search-grid, bounds and state setup in `_init_optim_state_`, 600-760.
- The `.ini` entry and description of every option these functions read.
- The tests `pybads/testing/bads/search/test_search.py` and `test_empty_search.py`.
- The B3 entries (and related entries) of the known-differences sheet, and the B3 rows of the counterpart map.

**Skimmed:**
- The poll, `bads.py:2088-2300`, as far as it calls `contraints_check` and LCB.
- In `gaussian_process_train.py`: the GP geometry (525-596) and the use of `udist` in `get_grid_search_neighbors`.
- `FunctionLogger.__call__` and `_record`; the `period_check` stub; gpyreg `GP.predict` (it clips variances at 0); `docsrc/source/index.rst`.
- The git history of the lines reported below.

**Not reached:**
- The BADS paper. arXiv and NeurIPS are blocked by the egress proxy, so what I say about the paper comes from memory and is kept general.
- MATLAB, which the internal track does not open.
- The internals of `local_gp_fitting` and `add_and_update_gp` (slice B5).

**Checks run** (all in `/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/review/B3_internal`; each prints `pybads.__file__` = review worktree and `gpyreg.__file__` = `/home/user/gpyreg-v1.3.3`):

| Script | What it checks |
|---|---|
| `shapes.py` | shapes passed through the search |
| `cc_check.py` | `contraints_check` on known inputs |
| `dup_evals.py`, `dup_where.py`, `dup_source.py`, `dup_init.py` | repeated evaluations, where they come from |
| `selmask.py` | the ES selection mask |
| `ucov_check.py`, `wcm_shape.py` | the ES-wcm covariance |
| `ei_check.py` | the hedge reward against the expected improvement |
| `gamma0.py` | a run with `hedge_gamma = 0` |
| `lcb_check.py` | `acq_fcn_lcb` with numeric `sqrt_beta` |
| `empty_gen.py` | the ES when its second generation loses every candidate |
| `variants.py` (output `variants_out.txt`) | 10 seeded runs per variant, `max_fun_evals = 200`, of the port against patched variants |

## 2. Answers to the first questions

### Q1. The search set
**Hedge** (`search_hedge.py:65-85`):
- The probabilities are a softmax of `hedge_beta·(g − max g)`, mixed with the uniform distribution: `p·(1 − n·γ) + γ`.
- One uniform draw from `bads.rng` picks the strategy through the cumulative sum; a random index is the fallback when rounding leaves no match.
- The gains start at `[10, 0]`, a hard-coded prior for the first strategy listed. With `hedge_beta = 1e-3/tol_fun = 1`, ES-wcm is chosen with probability 0.875 at the first search and never less than γ = 0.125.
- `hedge_gamma` has no description of its own: the comment line above it is the section header.
- Only ES-wcm and ES-ell can be dispatched (KD-B3-1).

**ES-wcm** (`es_search.py:232-277`):
- μ = n/2 over the GP's local training set, with weights log(μ+½) − log i for i = 1..⌊μ⌋, normalized.
- It selects the ⌊μ⌋+1 best points by `gp.y`, one more than it has weights (F3).
- `ucov` takes the scatter about the incumbent but ignores the weights (F2).
- The eigendecomposition clips negative eigenvalues, adds `mesh_size²` as jitter, and normalizes to unit trace (sum rule) or to a largest eigenvalue of 1. `sqrt_sigma = Λ^½Eᵀ` is correct, so the draws have covariance C.
- Apart from F2 and F3, this is the CMA-ES-style weighted covariance about the incumbent that the paper describes.

**ES-ell:**
- The covariance is `diag(poll_scale²/Σpoll_scale²)`, which has unit trace.
- It ignores the sum-rule flag of `search_method`. That only matters with a non-default flag of 0; unsure whether it is intended.

**Scale and counts:**
- `sqrt_sigma` is multiplied by `mesh_size·search_factor`.
- Generation 0 has λ = `n_search/n_search_iter` = 2048 points around u: half at ×0.5 (`poll_mesh_multiplier⁻¹`) and half at ×1.
- Offspring get ×`es_start` (0.25) around their selected parents.
- The `es_beta` one-fifth rule runs only for 0 < i < `n_search_iter` − 1, so never at default (and it is miscounted, F7).
- There are two generations, so at most 4096 candidates.

**Selection and reproduction:**
- All generations are pooled, and the best min(pool, λ) by LCB are kept. This is equivalent to (μ+λ) selection.
- Offspring are allocated by rank weights ∝ 1/√rank. The allocation is shifted by one rank (F4).
- The number of offspring is min(λ, number of parents), so it shrinks when the checks remove candidates. That may be intended; unsure.

**Ordering and return:**
- Candidates are ordered by an ascending `argsort` of LCB (unstable; NaN last).
- `contraints_check` returns its rows in lexicographic order, so ties break by coordinate.
- The ES returns the best candidate (1-D) and its LCB, or a `(0, D)` empty set. A later generation that is emptied by the checks empties the whole set (F9).

The structure matches the docs and the paper: an evolution strategy scored by the GP's LCB, two proposal covariances, and a Hedge choice.

### Q2. The grid and the checks
**`force_to_grid`:**
- It rounds to the nearest multiple of `search_mesh_size`, anchored at 0 in u space.
- `np.round` rounds halves to even; this is immaterial.

**`_update_search_bounds_`:**
- `lb` is rounded to the grid and raised one step if the rounding went below `lb`; `ub` is lowered the same way.
- The result is the largest box on the grid inside the transformed hard bounds.

**`contraints_check`, step by step:**
1. It projects candidates onto `[lb_search, ub_search]`, so they stay on the grid. The poll uses `proj=False`, which drops out-of-bounds points instead.
2. It removes exact duplicates, keeping the first occurrence.
3. The "previously evaluated" block rounds candidates and logged points to bins of `tol_mesh/2`, where `tol_mesh` = 2⁻¹⁹ in u space. It does not remove candidates that match logged points. It only merges candidates that share a bin, and it reorders the set lexicographically (F1).
4. It keeps the points with `non_box_cons(X) <= 0` in the original space, so `True`, positive values and NaN are all dropped, as the `BADS` docstring requires.

What remains is in bounds, on the grid, unique per bin and feasible, but it can include points already evaluated.

**`udist`:**
- It returns the squared Euclidean distance in units of the length scales. Its callers take the square root, or compare with radius².
- Its periodic branch is unreachable (KD-B1-6), and wrong as written: it indexes rows of the distance matrix by variable index.

### Q3. The choice, the evaluation and the decision
**Choice:**
- The step recomputes the target, which LCB does not use: the search pays for one deepcopy of the GP per step for nothing.
- It re-checks the single ES candidate, which is redundant.
- `acq_fcn_lcb` uses √β = √(ν·2·ln(D t² π²/(6δ))), with ν = 0.2, δ = 0.1 and t = `func_count` + 1, on the latent SD (`add_noise=False`). This is GP-UCB as the paper describes it.
- A numeric `sqrt_beta` set through `search_acq_fcn` applies only inside the ES; the step always uses the default schedule, which is harmless with one candidate. A plain Python float raises (F8).
- A NaN LCB sorts last in the ES. The random fallbacks in the ES and at `bads.py:1809-1818` cannot trigger: the step tests the index returned by `argmin`, which is always finite (see also F9).

**Evaluation:**
- The point is evaluated through the function logger.
- GP statistics are saved with the observation SD.
- The point is added to the GP, except at the last search of a round.
- At levels 1 and 2, a copy of the GP is rebuilt around the point without a refit, and its latent mean and SD become the estimate; a failed rebuild gives NaN.

**Decision:**
- The improvement is the `improvement_quantile` quantile of N(fval − f, fsd² + f_sd²), with the two estimates taken as independent. At q = 0.5 this is fval − f.
- Success means an improvement above `tol_improvement·mesh^1.5`, floored at `tol_fun` when `sloppy_improvement` is on. The option's description does not mention that floor.
- Incremental means an improvement above 0 with `sloppy_improvement` on.
- Either moves the incumbent (u, u_best, yval, fval, fsd) and sets `reset_gp`; at level 1 or 2 the rebuilt copy becomes the working GP.

**When something fails:**
- An empty search set means no evaluation, a failure, no hedge update and a contracted `search_factor`.
- A NaN estimate is a failure, and the chosen strategy's gain only decays.

### Q4. The hedge's rewards and the search statistics
**`update_hedge`**, with γ > 0 (the default):
- Only the chosen strategy is rewarded: g_i ← decay·g_i + er/(p_i·mesh_size). The others have p = ∞, so they only decay (Exp3 importance weighting).
- er = max(0, fval_old − f) when fs = 0, which covers every deterministic run.
- Otherwise er is the "expected reward" σ(γΦ(γ) + φ(γ)), with φ mis-coded (F5).
- The prediction used is the step's own estimate: the observed y at level 0, the estimate from the rebuilt copy at levels 1 and 2.

**`update_hedge`**, with γ = 0:
- Every strategy is updated, and a non-chosen one is scored at a slice of the chosen point. This crashes (F6).

**`_update_search_stats_`:**
- It logs log(search_factor) before updating it.
- Success multiplies the factor by √2, incremental by 2, failure by √½ with a floor at `search_factor_min` = 0.5.
- The factor is reset to 1 after the last search of a round.
- Under `adaptive_incumbent_shift` (off by default), `sd_level` is doubled on success, multiplied by 4 on an incremental search, and halved on failure down to `incumbent_sigma_multiplier`. Only the target reads `sd_level`.
- All of this matches the option descriptions, including that an incremental search expands more than a successful one.

## 3. Findings

### F1. `contraints_check` does not remove previously evaluated points, and reorders the set
- Location: `pybads/function_logger/constraints_check.py:33-43`; MATLAB: `utils/uCheck.m` (per the map; not opened).
- Category: control flow.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: yes. Every ES generation, every search step, the poll and the initial design go through it, at every uncertainty level.
- History: written in `c7c88ab` (2022-06-02); the logic has been unchanged since `f9e9326` (2022-11-02). MATLAB history not checked (internal track).
- **What the code does:**
  - It stacks `[u1; u2]` (candidates first, then logged points) and keeps `np.unique(..., return_index=True)` indices below `len(u1)`.
  - A candidate that matches a logged point has its first occurrence in `u1`, so it is kept.
  - The block only merges candidates that share a `tol_mesh/2` bin.
  - The indices are left unsorted, since `np.sort` is commented out, so the output is in lexicographic order.
- **What it should do:** remove candidates that match a logged point, as the comment "Remove previously evaluated vectors (within tol_mesh)" says, and keep the input order.
- Consequence if real:
  - At an optimum on the bounds, where projection makes the candidates coincide, the search keeps re-evaluating an evaluated point.
  - f = Σ(x+1)² on [0,5]^D (3 seeds): **17-19 of 43** evaluations were repeats in D = 2, and **9 of 27** in D = 1.
  - All repeats came from the search: 18 of 25 searches at seed 1 in D = 2, 9 of 16 in D = 1. None came from the poll.
  - With the removal patched in: no repeats, the same final value and the same count; the searches evaluate new neighbours instead.
  - Interior optima (sphere, D = 3): no candidate set held an evaluated point.
  - The reordering changes tie-breaking only.
- Suggested reproduction: `cc_check.py`. On U = [[0.5,0.25] (evaluated), [0.75,0], [0,0] (evaluated), [0,0], [0,1e-8], [0.25,−0.25]] the output is [[0,0],[0.25,−0.25],[0.5,0.25],[0.75,0]]: both evaluated rows are kept, and the order is lexicographic.
- Test adequacy: `test_incumbent_constraint_check` asserts this behaviour, and its comment says MATLAB would remove the rows and names a survey candidate defect. The known-differences sheet has no entry for it.

### F2. `ucov` ignores the weights: the ES-wcm covariance is an unweighted scatter
- Location: `pybads/search/es_search.py:324-329`, called from `251-259`; MATLAB: `utils/ucov.m`, `search/searchES.m` method 1.
- Category: formula.
- Proposed classification: port discrepancy (the error is in NumPy broadcasting).
- Confidence: high.
- Reached at default options: yes, at every search that picks ES-wcm (at least 12.5% of searches, 87.5% at the start), at every level.
- History: the weights have been ignored since `c7c88ab`. `e404fcc` (2023-07-06) rewrote the broadcast for high D and kept the defect.
- **What the code does:**
  - `w.reshape(-1,1,1) * u_shift` broadcasts to (n_w, n_b, D), and `matmul` gives w_k·(u_shiftᵀu_shift) for every k.
  - Summing over k gives (Σw)·u_shiftᵀu_shift = u_shiftᵀu_shift.
- **What it should do:** Σᵢ wᵢ(Uᵢ−u)(Uᵢ−u)ᵀ, the weighted covariance matrix that the comment and the paper describe, with the best points weighted most.
- Consequence if real:
  - The shape of the search covariance changes. Because C is about n/2 times larger than the weighted form, the `mesh_size²` jitter is relatively n/2 times weaker.
  - On a rotated 4-D ellipsoid (`wcm_shape.py`), the port's min/max eigenvalue ratio was 5-100× smaller than the weighted one's, and the leading directions differed (|cos| 0.53-0.63 at some searches).
  - Final values over 10 seeds on 3-D Rosenbrock and a 4-D ellipsoid: no difference beyond the spread between runs.
- Suggested reproduction: `ucov_check.py`. `ucov(U, 0, w, …)` equals `(U−u)ᵀ(U−u)` exactly: `port == unweighted: True`.
- Test adequacy: `test_u_cov` checks only the shape.

### F3. ES-wcm selects ⌊μ⌋+1 best points for ⌊μ⌋ weights
- Location: `pybads/search/es_search.py:242-248`; MATLAB: `search/searchES.m` method 1.
- Category: indexing/shape.
- Proposed classification: port discrepancy (a 1-based `1:floor(mu)` converted twice).
- Confidence: medium-high.
- Reached at default options: yes, as F2.
- History: `c7c88ab`.
- **What the code does:** the weights run over `arange(1, floor(mu+1))`, which is 1..⌊μ⌋, but the selection is `y_idx[0:floor(mu+1)]`, which is ⌊μ⌋+1 rows.
- **What it should do:** use the ⌊μ⌋ best points. For example, with 17 training points, 9 points are selected for 8 weights.
- Consequence if real:
  - The extra point, the worst of the selection, enters the scatter with full weight because of F2.
  - If F2 were fixed on its own with per-row weights, the size mismatch would raise, so the two are coupled.
- Suggested reproduction: print `len(weights)` and `len(idx_sel)` inside `_initialize_`.
- Test adequacy: none.

### F4. The ES selection mask is shifted by one rank: the best parent gets one offspring
- Location: `pybads/search/es_search.py:70-74`, used at `208-217`; MATLAB: `utils/ESupdate.m`.
- Category: indexing/shape.
- Proposed classification: port discrepancy (`cw = cumsum(w) − w + 1` holds 1-based start positions and is used as 0-based).
- Confidence: high.
- Reached at default options: yes, at the one reproduction step of every search, at every level.
- History: `c7c88ab`.
- **What the code does:**
  - `idx[cw] = 1; cumsum(idx[:-1])` yields [0, 1×w₀, 2×w₁, …], so parent 0 gets 1 offspring and parent k gets w_{k−1}.
  - The mask has λ+1 entries.
  - With μ = λ = 2048, w starts [17, 12, 10, …], and the offspring counts of parents 0..5 are [1, 17, 12, 10, 9, 8].
- **What it should do:** give parent k w_k offspring, i.e. `np.repeat(arange, w)`.
- Consequence if real:
  - The best candidate's neighbourhood is under-sampled in the second generation.
  - Final values over 10 seeds on two problems: no difference beyond the spread between runs.
- Suggested reproduction: `selmask.py`, which prints the counts above.
- Test adequacy: `test_search_selection_mask` asserts a golden sum of 885072, which mirrors the implementation and would lock in the shift.

### F5. The hedge's expected reward uses a mis-parenthesized normal density
- Location: `pybads/search/search_hedge.py:152-155`; MATLAB: `acq/acqPortfolio.m`, `'upd'` branch.
- Category: formula.
- Proposed classification: unsure (a defect against the mathematics; whether MATLAB shares it is for the comparison track).
- Confidence: high that the formula is wrong.
- Reached at default options: yes, in every noisy run: levels 1 and 2, where the estimate's SD is above 0. Level 0 takes the fs = 0 branch.
- History: written in `c7c88ab` and never changed.
- **What the code does:** `exp(-0.5*γ²/sqrt(2π))`.
- **What it should do:** use φ(γ) = exp(−γ²/2)/√(2π), which gives the expected improvement σ(γΦ(γ) + φ(γ)).
- Consequence if real:
  - The rewards are inflated, most for points worse than the incumbent: the ratio to the expected improvement is 2.5 at γ = 0, 48 at γ = −2 and 424 at γ = −3.
  - The hedge therefore credits a strategy whose point was 2-3 SD worse.
  - Noisy 3-D sphere, 10 seeds: final errors identical to two decimals.
- Suggested reproduction: `ei_check.py`, which prints the port's reward against the expected improvement for γ in [−3, 2].
- Test adequacy: nothing tests `update_hedge` values.

### F6. `hedge_gamma = 0` crashes at the first search
- Location: `pybads/search/search_hedge.py:127`, `132-134`; MATLAB: `acq/acqPortfolio.m`, `'upd'` branch.
- Category: indexing/shape.
- Proposed classification: port discrepancy (a 1-D point indexed as if it were a matrix of rows).
- Confidence: high.
- Reached at default options: no, only with `hedge_gamma = 0` (any level).
- History: `c7c88ab`.
- **What the code does:**
  - `u_search` is a 1-D point, so `u_search[min(i, len−1):]` slices coordinates.
  - For a non-chosen strategy with index ≥ 1, `gp.predict` receives D−1 values: `ValueError: The dimension of input data 2 doesn't match GP's input dimension 3`.
  - With γ = 0 and gains starting at [10, 0], ES-wcm is chosen almost surely, so the run stops at its first search.
  - Even without the crash, the non-chosen strategy would be scored at the chosen strategy's point.
- **What it should do:** run, since `hedge_gamma = 0` is a legitimate value.
- Suggested reproduction: `gamma0.py`, which shows the traceback above.
- Test adequacy: none.

### F7. The fraction of new points behind the ES scale update is miscounted
- Location: `pybads/search/es_search.py:177`, `182-192`, `199-205`; MATLAB: `search/searchES.m`.
- Category: indexing/shape.
- Proposed classification: port discrepancy.
- Confidence: medium-high (by reading).
- Reached at default options: no. Only with `n_search_iter` ≥ 3; issue (c) below needs ≥ 4.
- History: `c7c88ab`.
- **What the code does:** `n_new = sum(z_idx[0:ntest+1] > nold)`.
  - (a) With 0-based indices the new points start at index nold, which `>` excludes.
  - (b) It examines ntest+1 entries instead of ntest.
  - (c) From the third generation on, the pool holds every earlier generation while `nold = min(pool, λ)`, so older points with indices in (nold, len(pool)) count as new, and `frac`, and with it the scale, is biased up.
  - (d) An empty generation gives 0/0: the scale becomes NaN, and so do the offspring.
- Consequence if real: a wrong step-size adaptation in multi-generation searches.
- Suggested reproduction: instrument `frac` against the true fraction of new points among the top ntest, with `n_search_iter = 4`.
- Test adequacy: none.

### F8. `acq_fcn_lcb` rejects a plain numeric `sqrt_beta`
- Location: `pybads/acquisition_functions/acq_fcn_lcb.py:42`; MATLAB: `acq/acqLCB.m`.
- Category: control flow.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: no, only with a numeric second element in `search_acq_fcn`.
- History: `c7c88ab`.
- **What the code does:**
  - `2.0` or `2` raises `AttributeError: 'float' object has no attribute 'size'`.
  - An array raises NumPy's ambiguous-truth `ValueError` before the intended message.
  - Only NumPy scalars work, although the docstring says `sqrt_beta: float`.
  - The docstring also calls the returned SD "GP variance".
- Suggested reproduction: `lcb_check.py`.
- Test adequacy: none.

### F9. An emptied later ES generation discards every candidate, behind a misleading warning
- Location: `pybads/search/es_search.py:170-185`; MATLAB: `search/searchES.m`.
- Category: control flow.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: rarely. It needs every offspring of the second generation removed, which in practice only `non_box_cons` can do (at any level).
- History: `c7c88ab`.
- **What the code does:**
  - When a generation has no candidates left, the "random search" fallback replaces the whole pooled `z_candidates` with `rng.random(0)`.
  - `z` then stays empty while `us_candidates` holds the earlier generations, so the ES returns an empty set and the search fails.
  - The warning says "random search is performed", which never happens.
  - In generation 0 the fallback is overwritten at once.
  - A NaN LCB, which is the real failure of the acquisition, is never caught.
- **What it should do:** keep the earlier generations' candidates.
- Suggested reproduction: `empty_gen.py`, which empties the second generation: the unpatched call returns `(3,)`, the patched one `(0, 3)`.
- Test adequacy: `test_es_search_without_candidates_returns_an_empty_set` empties every generation, so it cannot see this.

## 4. Test adequacy notes
- `test_search_selection_mask` pins the implementation's own output (sum 885072), which carries F4.
- `test_incumbent_constraint_check` asserts F1's behaviour.
- `test_u_cov` checks only the shape, so F2 and F3 pass.
- `test_search` and `test_search_hedge` assert only that `gp.y >= z`.
- No test covers:
  - the rewards of `update_hedge`, or `hedge_gamma = 0`;
  - a numeric `sqrt_beta`;
  - `n_search_iter` ≥ 3;
  - a second generation emptied on its own.
- `test_failed_searches_floor_the_search_factor` agrees with the option descriptions.
