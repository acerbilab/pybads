<!-- Report of the S reviewer (Sto-BADS, internal track), wave 0 of the port review, reading PyBADS at ab4dded in ../pybads-review; saved verbatim from its final message on 2026-09-25. Its check scripts are kept on the orchestrator's machine only (dev/scripts/runs/LOCAL.md). Nothing in it is verified. -->

# S internal review: Sto-BADS

## 1. Coverage

**Read completely** (review worktree `C:\Users\luigi\Documents\GitHub\pybads-review` at `ab4dded`):
- `pybads/bads/bads.py`: 150–270 (`__init__`), 950–1172 (`_init_mesh_`, `_init_optimization_`), 1174–1570 (`optimize`), 1572–2349 (`_search_step_`, `_eval_improvement_`, `_sto_success_improvement_`, `_poll_step_`), 2351–2690 (`_is_gp_refit_time_`, `_is_poll_stop_`, `_get_target_from_gp_`, `_update_incumbent_`, `_update_search_stats_`, `_re_evaluate_history_`, `_check_mesh_overflow_`).
- `pybads/bads/option_configs/advanced_bads_options.ini`, the whole file.
- `pybads/search/search_hedge.py`, the whole file. `pybads/search/es_search.py`, 80–216 (`__call__`). `pybads/function_logger/constraints_check.py`.
- `pybads/testing/bads/test_gp_update_failures.py`, 680–880, and `pybads/testing/bads/scripts/bads_quadratic_noisy_example.py`.

**Skimmed:** the class docstring (1–150), `optimize_result.py` (only its reads of `optim_state`).

**Not reached:** the internals of `gaussian_process_train.py` (I took the `needs_rebuild` markers as the code describes them), `poll_mads_2n`, `basic_bads_options.ini`.

**Grep for the options:** the Sto-BADS branches read only `stobads`, `opp_stobads`, `stobads_frame_size_scaling_power` and the constructor argument `gamma_uncertain_interval`. Outside `bads.py` and the `.ini`, no code, test or `docsrc` page mentions them.

**Paper:** I read the arXiv version, 1911.01012: Section 2.1, Definitions 1–3, Algorithm 1 and Assumption 2. The number in the brief, 1903.07664, is an unrelated astrophysics paper. What the paper specifies:
- Parameters: γ > 2 and τ = 1/2.
- An estimate is ε_f-accurate when |f_x − f(x)| ≤ ε_f δ_p².
- **Success:** f_s − f_0 ≤ −γ ε_f δ_p² for some point. The point is accepted and δ_p ← τ⁻²δ_p.
- **Certain failure:** f_s − f_0 ≥ γ ε_f δ_p² for **all** poll points. x is kept and δ_p ← τ²δ_p.
- **Uncertain failure:** anything else. x is kept and δ_p ← τδ_p.
- Assumption 2(ii): the estimate's error variance must be ≤ κ_F² δ_p⁴, which the method meets by resampling.

**Checks run** (scripts and logs in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pybads\10994841-8958-4975-b288-1fd2bcd6013b\scratchpad\review\S_internal\`). Each printed that it imported the review worktree's PyBADS and gpyreg v1.3.3.
- `s1_rule.py`: direct calls of the rule.
- `s2_empty_search.py`: an empty search set.
- `s3_instrument.py` and `s5_instrument_more.py`: instrumented runs of ≤200 evaluations.
- `s4_nan_search.py` and `s4b_nan_search_trace.py`: a failed rebuild of the search GP.
- `s6_positional.py`: where an 8th positional argument goes.
- `s7_level2.py`: a run with `specify_target_noise`.

## 2. Answers

### Q1. What `_sto_success_improvement_` returns and compares

**What it computes** (bads.py:1935–1956):
- mu = f_base − f_new
- ε = sqrt(s_base² + s_new²)
- u = γ · ε · frame_size^p, with γ = 1.96 when `gamma_uncertain_interval` is None, and p = `stobads_frame_size_scaling_power` (default 2)
- frame_size is `self.mesh_size` = 2^mesh_size_integer, which is ≤ 1 because `max_poll_grid_number = 0`. It is the poll step length, the right analogue of δ_p.

**The three outcomes:**
- **1** if mu ≥ u. A tie with u = 0 counts as success.
- **−1** if mu ≤ −u.
- **0** otherwise. This includes every case with a NaN, because both comparisons are then False (checked in s1: `(0, nan, .3, nan, 1) -> 0`, `(nan, -1, nan, .3, 1) -> 0`).

**Does the formula match its documentation and Sto-MADS?**
- It matches the option comment ("γ·ε·frame_size**power") exactly.
- The sign convention matches Sto-MADS.
- With ε read as the SD of the difference, γ = 1.96 is a z-value. That fits the GP scheme: the deterministic analogue of Sto-MADS's γ > 2 would be γ > √2 relative to the combined SD.
- The δ² factor does not fit. In Sto-MADS, ε_f δ² is the accuracy the estimates are forced to reach. If the GP SD stands in for the estimate's accuracy, the analogue of the threshold is γ·ε. Multiplying by δ² applies δ² twice (F3).

**Which quantities it compares:**
- f_new and s_new are latent GP estimates:
  - in the search, the mean and SD from `new_gp`, a copy of the GP with the new observation, rebuilt around `u_search`;
  - in the poll, the current GP after `add_and_update_gp`.
- f_base and s_base are the stored incumbent values `self.fval` and `self.fsd`.
  - Until the incumbent first moves, and until `_re_evaluate_history_` runs (only from the end of the second poll iteration, `poll_iteration > 0`), these are an **observation**: the minimum of ≥20 noisy initial draws, biased low. The SD is `noise_size` (1.0 unless the user sets it) at level 1, or the target's reported SD at level 2.
  - In s4 the incumbent at the 4th search still had fval = −1.64, fsd = 1.0 on a sphere whose minimum is 0.
  - After a move, they are the prediction of the GP of that moment, stale within a search round.
- So the two estimates are not on the same footing: an observation against a GP estimate at first, then predictions from different GPs, with their covariance ignored. The default noisy path feeds the same inputs to `_eval_improvement_`, so this is shared with BADS's own scheme and not specific to Sto-BADS. Sto-MADS, by contrast, re-estimates f_0 at every iteration.

### Q2. With `opp_stobads` on: what an uncertain outcome does

**Search:**
- **Outcome 0:** `is_search_improved` is True (1829). The incumbent moves to the single evaluated search point, even when its estimate is worse (mu < 0). fval and fsd become `new_gp`'s estimate, `gp = new_gp`, and `reset_gp = True`.
  - It is logged as "Incremental search", so `search_stats` gets 0.5 and `search_factor` is multiplied by 2 (more than the ×√2 of a success).
  - `search_success` is not incremented, so the poll is not skipped.
  - The search never changes the mesh.
- **Outcome −1:** a failure, nothing moves.

**Poll:**
- The decision uses `sto_success` of the **last evaluated** poll point (2214–2222, 2243).
- If that outcome is 0, the incumbent is set to `u_poll_best` (the poll point with the largest positive mean improvement, or the incumbent itself), and `is_poll_moved` is True. Because `certain_good_poll` is False, the mesh contracts (−1, and −2 when stalling).
- If the poll evaluates no point, `sto_success` keeps its initial 0, and the poll "moves" the incumbent to itself.
- The mesh update does not tell certain from uncertain failure: both are ÷2. Sto-MADS uses ÷4 and ÷2, and success is ×2 here against ×4 there. This follows BADS's own mesh scheme, so I do not report it. The commented-out block at 2276–2280 would have added the extra reduction for certain failures.

**Is that what "Move incumbent even for the uncertain unsuccess" describes?**
- For the search, literally yes, but the description does not say that the move can be uphill by up to γεδ², or that it widens the search (F4).
- For the poll, only loosely: the point moved to is not the point whose outcome was uncertain, and an earlier success is discarded (F1).

**A NaN estimate is classified 0, "uncertain"** (F2):
- In the search, with opp on, the incumbent moves to the search point and takes fval = fsd = NaN. This was reproduced. It contradicts the comment at 1753–1755 ("so the search counts as failed"), which holds only for the default path and for opp off.
- While the incumbent's fval is NaN, every comparison is 0:
  - any further search in the round is accepted unconditionally;
  - the poll can find no improvement (NaN > x is False), classifies every point 0 and contracts the mesh.
- The state clears only at the end-of-iteration re-evaluation, and not at all in iteration 0.
- In the poll, a NaN on the last point (failed `add_and_update_gp`) gives 0 and wipes an earlier success, with opp on or off.

### Q3. State skipped or double-counted on the Sto-BADS path

- **`certain_good_poll`** is overwritten at every poll point (2222), not raised monotonically from the best point as on the default path (2207–2210). This drives both the mesh update and `_is_poll_stop_`.
  - After a success followed by a non-success, the early stop for a good poll is lost.
  - Under noise, `min_failed_poll_steps = inf` means the stop without a good poll never fires, so the poll runs through all 2D directions (F1).
- **`search_success`, `search_spree`, `u_success`:** consistent. Only outcome 1 counts, so an uncertain search never skips the poll.
- **Search statistics:** "incremental" on the Sto+opp path includes moves to worse estimates, and still multiplies `search_factor` by 2 (F4).
- **`reset_gp`:** set True after a poll that "moved" the incumbent to itself (outcome 0 with no improvement, or no point evaluated). Harmless: the first search after a poll rebuilds anyway (`search_count == 0`).
- **Iteration history:** can record a NaN incumbent (F2).
- **Hedge rewards:** `update_hedge` uses `fval_old` and `f_mu_search`, independent of the Sto outcome, so it is the same on both paths. A NaN gives a reward of 0 on both.
- **`_init_optimization_`** (1130–1132) silently turns `stobads` off for a deterministic target. The option comment does not say so. It is consistent with the code comment, so I do not report it.

## Findings

### F1. The Sto poll decides from the last evaluated poll point, not from the best or from any point
- Location: pybads/bads/bads.py:2213–2222, 2243–2256, 2257–2274; MATLAB: no counterpart
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- Reached at default options: no. It needs `stobads=True` on a noisy target (either `opp_stobads` setting).
- **What the code does:**
  - `sto_success` and `certain_good_poll = sto_success == 1` are reassigned for every evaluated poll point, while the move goes to `u_poll_best`, the point with the largest mean improvement.
  - A success found mid-poll is discarded when a later point is a certain failure (no move with either `opp` setting, mesh contracted), or when a later point is uncertain or NaN (with opp on: move but contract).
  - Poll points are evaluated in LCB order, so the last point tends to be the least promising.
- **What it should do:** Sto-MADS declares a poll successful if the condition holds for **some** poll point, and a certain failure only if it fails for **all** of them (Algorithm 1, lines 18–23). The default path makes `certain_good_poll` monotone in the best improvement.
- Consequence if real: lost mesh expansions and lost moves, and complete polls where the default path would stop early. This triggers whenever the poll continues past a success. In s5, polls went on for 4 and 5 points after their first success.
- Suggested reproduction: run `s3_instrument.py` (rosen D=2, seed 13, `opp_stobads=False`, 200 evaluations). Output: `poll 11: outcomes=[-1, 1, -1, -1], moved=False, dmsi=-1, best_is_incumbent=False`. The best point was a success, yet the incumbent stayed and the mesh contracted.
- Test adequacy: no test sets `stobads`.

### F2. With `opp_stobads`, a NaN GP estimate counts as "uncertain" and moves the incumbent to a point with NaN value
- Location: pybads/bads/bads.py:1752–1757 with 1948–1956, 1828–1830, 1856–1860; poll at 2184–2185 and 2214–2222; MATLAB: no counterpart
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- Reached at default options: no. It needs `stobads=True` (with `opp_stobads=True`, the default) on a noisy target, and a failed rebuild of the search GP (`needs_rebuild`). The poll variant needs a failed `add_and_update_gp` on the last poll point, with opp on or off.
- **What the code does:**
  - On a failed rebuild the search sets the estimates to NaN, meaning "no estimate, the search counts as failed" (the comment at 1753–1755).
  - The rule returns 0 for a NaN. `opp_stobads` turns that into an accepted move with fval = fsd = NaN, logged as an incremental search, and the failed `new_gp` becomes `gp`.
  - While fval is NaN, every later comparison returns 0: searches in the same round are accepted without any comparison, and the poll cannot recognise an improvement.
- **What it should do:** the rule should treat a NaN as failure (−1), as the default path does.
- Consequence if real: one or more moves without any comparison, and a poll wasted with a contracted mesh. The state recovers at `_re_evaluate_history_` (not in iteration 0). It triggers only when the GP rebuild fails, which is rare.
- Suggested reproduction: `s4_nan_search.py` / `s4b_nan_search_trace.py` inject a failed rebuild of the search GP on the 4th search, 120 evaluations. Output:
  - `stobads` False, and `stobads` True with opp False: `moved=False`, and the search counts as failure (0.0).
  - `stobads` True with opp True: `moved=True, fval=nan, fsd=nan`, `search_stats … 0.5`, `reset_gp=True`. The following poll classified all 4 points `f_base=nan … -> 0`, ended with `fval=nan` and contracted the mesh.
- Test adequacy: `test_noisy_search_after_failed_rebuild_counts_as_failure` and `test_noisy_poll_after_failed_add_counts_no_improvement` (test_gp_update_failures.py:743, 794) run at default options and watch `_eval_improvement_`, which the Sto path never calls. A Sto variant of the search test would fail its "no move to the point" assertion.

### F3. The uncertainty threshold multiplies the GP SDs by frame_size², so the three outcomes lose Sto-MADS's meaning
- Location: pybads/bads/bads.py:1935–1946; advanced_bads_options.ini:62–63; MATLAB: no counterpart
- Category: formula
- Proposed classification: suspected defect (the option comment documents the formula, so it may be a deliberate forcing-function design)
- Confidence: medium
- Reached at default options: no. It needs `stobads=True` on a noisy target.
- **Why the δ² factor is double-counted:**
  - In Sto-MADS, ε_f is a constant, and the estimates are made ε_f δ²-accurate by resampling (Definition 1, Assumption 2(ii)). The half-width of the uncertainty interval, γ ε_f δ², is therefore γ times the estimates' actual accuracy.
  - PyBADS puts the estimates' actual SDs in ε, which do not shrink like δ², and then multiplies by δ² again.
  - Replacing the sampled estimates with GP estimates therefore gives the rule γ·ε, which `stobads_frame_size_scaling_power = 0` recovers.
- **What the rule becomes:** with power 2 it is a forcing function scaled by the noise, not an uncertainty interval. After a few contractions it reduces to sign(mu).
- **Consequence if real:** with equal true values and independent Gaussian errors of the SDs the rule uses:

  | mesh size | P(outcome 1) | P(outcome 0) |
  |---|---|---|
  | 1 | 0.025 | 0.95 |
  | 2⁻¹ | 0.31 | 0.38 |
  | 2⁻² | 0.45 | 0.10 |
  | ≤ 2⁻³ | ≥ 0.49 | ≤ 0.02 |

  - In the s5 runs (sphere D=4 and D=6, rosen D=3; 200 evaluations each), every outcome 1 had mu/ε < 1.96, with median mu/ε between 0.10 and 0.19, and 88–100% below 1.
  - In the traced run, ε ≈ 0.35. There the Sto test for success is looser than BADS's own sufficient-improvement test, max(δ^1.5, tol_fun), at every mesh size: the Sto threshold is smaller whenever ε < 1/(1.96√δ).
  - "Certain" outcomes are therefore close to coin flips for most of a run.
- Suggested reproduction: `s1_rule.py` (the table above) and `s5_instrument_more.py` (the ratios above).
- Test adequacy: none.

### F4. `opp_stobads` accepts search points with worse estimates and then widens the search by the "incremental" factor
- Location: pybads/bads/bads.py:1828–1833, 1851–1853, 2624–2629; advanced_bads_options.ini:60–61, 98; MATLAB: no counterpart
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: medium
- Reached at default options: no. It needs `stobads=True` on a noisy target; `opp_stobads=True` is the default.
- **What the code does:**
  - Outcome 0 covers mu ∈ (−γεδ², γεδ²), so the search moves to points whose estimate is worse than the incumbent's. At δ = 1 that is up to about 2 combined SDs worse.
  - Such a move is labelled "incremental", which multiplies `search_factor` by `search_scale_incremental = 2`.
  - On the default path, "incremental" requires a positive mean improvement.
  - The option comment says only "move incumbent even for the uncertain unsuccess". Sto-MADS never moves on an uncertain failure.
- Consequence if real: uphill drift of the incumbent early in a run, each uphill move followed by a wider search. Moves to a worse estimate among uncertain searches in the 200-evaluation runs:

  | target | uphill moves / uncertain searches |
  |---|---|
  | sphere D=2 | 1/3 |
  | sphere D=3 | 4/5 |
  | sphere D=4 | 9/17 |
  | sphere D=6 | 10/15 |
  | rosen D=2 | 6/10 |
  | rosen D=3 | 3/6 |

- Suggested reproduction: `s3_instrument.py` / `s5_instrument_more.py` (the counts above).
- Test adequacy: none.

### F5. `gamma_uncertain_interval` is an undocumented 8th positional parameter placed before `options`
- Location: pybads/bads/bads.py:163–164, 251, 1937–1940; MATLAB: `bads(fun,x0,lb,ub,plb,pub,nonbcon,options)` takes OPTIONS 8th
- Category: defaults
- Proposed classification: suspected defect
- Confidence: high
- Reached at default options: yes, by any call that passes options positionally.
- **What the code does:**
  - The class docstring lists `options` right after `non_box_cons` and does not mention `gamma_uncertain_interval`.
  - A MATLAB-style call `BADS(fun, x0, lb, ub, plb, pub, None, opts)` binds `opts` to γ, and the options are silently ignored.
  - With `stobads` on, a dict as γ would raise a TypeError in the rule.
  - The rule tests the attribute `self.gamma_uncertain_interval` for None but uses the argument as the value. Harmless, since both call sites pass the attribute.
- Consequence if real: a user's options are silently dropped (for example, `max_fun_evals` falls back to 500·D).
- Suggested reproduction: `s6_positional.py`. Output: `gamma_uncertain_interval = {'display': 'off', 'max_fun_evals': 7, 'stobads': True}`, `options max_fun_evals = 1000 stobads = False display = iter`.
- Test adequacy: no test passes 8 positional arguments.

### F6. An empty search set ends in UnboundLocalError on `u_search` in every configuration
- Location: pybads/bads/bads.py:1781–1786, 1857 (Sto + opp), 1890 (all paths); MATLAB: no counterpart
- Category: control flow
- Proposed classification: suspected defect
- Confidence: medium (the crash is certain; how often it can happen is low)
- Reached at default options: in principle, but I found no natural trigger. `ESSearch` returns `us[0]`, which already raises IndexError when it has no candidate, and its point has passed the same constraint check. Periodic variables (`period_check`) are the one unverified way in.
- **What the code does:** the empty-set branch sets `f_mu_search = self.fval`, `f_sd_search = 0` and never assigns `u_search`.
  - The default path fails at the return statement.
  - The Sto+opp path classifies the empty set as 0 (or 1 when fsd = 0) and fails at `_update_incumbent_`.
- Consequence if real: the run crashes.
- Suggested reproduction: `s2_empty_search.py` empties the set on the 3rd search. Output: `UnboundLocalError` at 1890 for `stobads` False and for `stobads` True with opp False, and at 1857 for `stobads` True with opp True.
- Test adequacy: none.

### F7. A successful poll appends a bound method to `optim_state["u_success"]`
- Location: pybads/bads/bads.py:2268 (`self.u_best.copy` without the call); MATLAB: no counterpart
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: high
- Reached at default options: yes (the shared path, on every successful poll).
- **What the code does:** it stores a bound method where the incumbent's coordinates should be, which is not what the list's name promises.
- Consequence if real: none at present. Nothing reads `u_success`, and deep-copying treats the method as atomic.
- Suggested reproduction: inspect `optim_state["u_success"]` after any run with a successful poll.
- Test adequacy: none.

## 3. Test adequacy notes

- **No test reaches Sto-BADS.** Nothing under `pybads/testing/` sets `stobads`, `opp_stobads`, `stobads_frame_size_scaling_power` or `gamma_uncertain_interval`. `scripts/bads_quadratic_noisy_example.py` writes a dump named "stobads_noise" but never turns the option on.
- **The failure-injection tests cover only the default path.** The NaN-estimate tests in `test_gp_update_failures.py` (743, 794) observe `_eval_improvement_`, which the Sto path bypasses. Run with `stobads=True`, the search test would catch F2.
- **Missing tests:**
  - No test drives a poll sequence of outcomes such as [1, −1] through `_sto_success_improvement_`, which would catch F1.
  - No test checks the rule's outcomes against its specification (NaN, ties, the δ scaling), which would catch F2 and F3.
  - No test checks how `BADS.__init__` binds its positional arguments, which would catch F5.
- A run with `specify_target_noise=True` and `stobads=True` (`s7_level2.py`, 150 evaluations) completed without error. Outcome counts: search 1: 11, 0: 12, −1: 44; poll −1: 40.
