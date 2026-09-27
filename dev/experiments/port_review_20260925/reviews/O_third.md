<!-- Report of the O reviewer (improvement, acquisition and geometry, the third reader: re-derivation, then MATLAB comparison), wave 4 of the port review, reading PyBADS at 0d866e8 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session, with the complete history; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave4/O_third/. Nothing in it is verified. -->

# O third review: improvement, acquisition and geometry

I found one defect, and it is in Sto-BADS (F1). Every formula in the slice matches MATLAB BADS line for line and matches my own derivation. That holds for the improvement at any quantile and at each call site, the sufficient improvement, the final quantile choice, the lower confidence bound (LCB) and its schedule, the Hedge's probabilities and update, and `len_scale`, `poll_scale` and `effective_radius` together with their uses. Three differences from MATLAB exist but are recorded somewhere other than the sheet; they are listed at the end of the answers.

The papers could not be opened: arxiv.org and proceedings.neurips.cc are blocked by the network proxy. Where I cite the BADS paper or Sto-MADS, it is from memory, and I say so each time.

## 1. Coverage

**Read completely (Python at `0d866e8`):**
- `pybads/bads/bads.py`:
  - the loop head (1360-1398: mesh sizes and the sufficient improvement);
  - the search/poll decision, the end of an iteration (the stall test, 1504-1519), the recording, the re-estimate and the move (1558-1605), and the final choice (1621-1690);
  - `_search_step_` (1747-2082), `_eval_improvement_` (2084-2103), `_sto_success_improvement_` (2105-2152) and `_poll_step_` (2154-2553);
  - `_is_poll_stop_`, `_update_incumbent_`, `_update_search_stats_`, `_re_evaluate_history_`;
  - the option checks at 800-845 and the noisy setup at 1195-1250.
- `pybads/acquisition_functions/acq_fcn_lcb.py`, `pybads/search/search_hedge.py`, `pybads/search/es_search.py`, `pybads/search/grid_functions.py` (`udist`), `pybads/poll/poll_mads_2n.py`.
- `pybads/bads/gaussian_process_train.py`: `local_gp_fitting` (250-598), `get_grid_search_neighbors`, the geometry's starting values (1079-1081); `_robust_gp_fit_` only to confirm that a single hyperparameter set comes back.
- The option lines of both `.ini` files that these functions read.
- `pybads/utils/iteration_history.py`, and gpyreg's `RationalQuadraticARD` and the signature of `GP.predict`.

**Read completely (MATLAB at `74919c0`):**
- `bads.m`: 149-290, 428-1165 (loop head, search, poll, end of iteration, final choice), 1255-1300 (`EvalImprovement`, `UpdateIncumbent`), 1340-1420 (`UpdateSearch`, `reevaluateIterList`).
- `acq/acqLCB.m`, `acq/acqPortfolio.m`, `search/searchHedge.m`, `search/searchES.m`, `poll/pollMADS2N.m`, `utils/udist.m`, `private/gpupdate.m`.
- `gpdef/gpdefBads.m` 1-200, the header of `gpml_fast/covRQard_fast.m`, and the `'iter'` branch of `private/funlogger.m`.

**Tests read:** `test_stobads.py`; the Hedge and LCB tests in `search/test_search.py`; `poll/test_poll_mads.py`; the final-estimate tests in `test_noisy_runs.py`.

**Histories:** `git log -L` on every formula, on both sides.

**Not reached:** the BADS paper and Sto-MADS (network blocked). `_get_target_from_gp_` and `_is_gp_refit_time_` belong to slices B4 and B5; I read them only where a formula here depends on them.

**Checks run.** The scripts and logs are in `/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad/wave4/O_third`; each printed the review worktree's `pybads` and the gpyreg 1.3.3 clone.
- `check_formulas.py`: Python transcriptions of `EvalImprovement`, `acqLCB`, `acqPortfolio` `'upd'`, `searchHedge`'s probabilities, the geometry of `gpupdate.m` and `udist.m`, compared with the port on the same inputs, plus Monte Carlo checks of the quantile and of the expected improvement.
- `check_runtime.py` (`run_runtime_*.log`): 4 seeded runs of 150 evaluations (D = 3 at level 0, D = 3 at level 1, D = 1 at level 0, D = 2 at level 1 with `improvement_quantile` = 0.2). At every refit and every Hedge update it compared the port with the transcriptions, and it recorded every call of `_eval_improvement_`.
- `check_stobads*.py` (`run_stobads*.log`): 3 seeds, 200 evaluations each, with Sto-BADS and `opp_stobads` on.
- `check_empty_predict.py`: the LCB on an empty candidate set returns (0, 1) arrays and does not raise.

## 2. Answers to the first questions

### Q1. The improvement

**Derivation.** Take two independent Gaussian estimates, F_b ~ N(f_b, s_b²) at the base point and F_n ~ N(f_n, s_n²) at the new point. The improvement I = F_b − F_n is N(f_b − f_n, s_b² + s_n²). Its q-quantile is z = (f_b − f_n) + √(s_b² + s_n²)·Φ⁻¹(q), where Φ⁻¹(q) = −√2·erfcinv(2q).
- A q below 0.5 is conservative.
- At level 0 the SDs are 0, so z is the difference f_b − f_n.
- The formula assumes the two estimates are independent. GP predictions under one posterior are positively correlated, so the true variance is s_b² + s_n² − 2c. The code's comment "needs to be corrected" can only mean this covariance, or that the two estimates often come from different GPs, which have no joint covariance.
- At q = 0.5 the median is f_b − f_n whatever the variance, so "for q=0.5 it does not matter" is exact.

**Code.** `_eval_improvement_` (`bads.py:2084-2103`) is `EvalImprovement` (`bads.m:1257-1280`) line for line. It computes exactly this quantile under independence: the maximum difference from the transcription and from `norm.ppf` is 8.9e-16 over q ∈ {1e-3, 0.1, 0.25, 0.5, 0.75, 0.9}. At q = 0.1 the Monte Carlo quantile is −0.3402 against −0.3408 from the formula. MATLAB rejects a q outside (0, 1) at every call; PyBADS rejects it when `BADS` is created (`bads.py:812-819`), with the same effect.

**Call sites.** Each one matches `bads.m` in its base, SDs, quantile and threshold:

| Call site | Python | MATLAB | Base, new point | Compared with |
|---|---|---|---|---|
| Search | `bads.py:1980-1996` | `bads.m:676-690` | (`fval`, `f_mu_search`, `fsd`, `f_sd_search`) | success above `search_sufficient_improvement`; incremental above 0 with `sloppy_improvement` |
| Poll | `2392-2413`, `2437-2441` | `931-961` | (`fval`, `f_poll`, `fsd`, `f_sd_poll`) | best above 0 with `sloppy_improvement` moves; best above the sufficient improvement expands the mesh |
| Stall | `1504-1519` | `1078-1084` | iterate `tol_stall_iters` back, against the incumbent | below `tol_fun` |
| Accelerated mesh | `2491-2511` | `976-982` | iterate `accelerate_mesh_steps` back, against the incumbent after the poll's move | below `tol_fun` |
| Re-estimate | `1575-1605` | `1107-1118` | the incumbent against every iterate, first skipped | above `tol_fun` |

- The 0-based `optim_state["iter"]` maps correctly onto MATLAB's 1-based `iter`: `poll_iteration > T - 1` is MATLAB's `iter > T`, and index `iter - T` in both.
- In the runs, the base was the incumbent (`self.fval`, `self.fsd`) in all 259 search calls and all 146 poll calls. Every call passed `improvement_quantile`.

**Sufficient improvement.** It is `tol_improvement · mesh_size^forcing_exponent` (1 · Δ^1.5), floored at `tol_fun` when `sloppy_improvement` is on (`bads.py:1387-1398`, `bads.m:502-507`).
- This is what the three option descriptions say.
- Δ is the poll size (`MeshSize`), not the search mesh.
- It matches the BADS paper's forcing function (Δ_poll)^{3/2}, as far as I recall the paper.
- At unit mesh the sufficient improvement is 1 in the units of f, which depends on the scale of f. That is BADS's design, and MATLAB is the same.
- The line has not changed in MATLAB since 2017. The Python line dates from `c7c88ab`; `8aecb6a` only dropped a `.copy()`.

### Q2. The final choice and Sto-BADS's rule

**Final choice.** `q_beta = fval + √2·erfcinv(2q)·fsd`, and √2·erfcinv(2q) = Φ⁻¹(1 − q), which is 3.0902 at q = 1e-3 (checked). This is the upper (1 − q) quantile of each iterate's Gaussian re-estimate: the run returns the iterate with the lowest 99.9 % upper bound.
- The iterates are all those recorded, after `_re_evaluate_history_`, with the first skipped and NaN skipped.
- `bads.py:1628-1650` matches `bads.m:1137-1152`, where MATLAB's `min` also ignores NaN.
- `np.nanargmin` works on the object arrays that `IterationHistory` holds (checked). The current iterate keeps its estimate (KD-B2-4), so the choice is never over NaN only.
- MATLAB added the skip of the first iterate in `75ec49f` (2022-05-09), after PyBADS began and before the first ported algorithm. The Python has skipped it since `8e59038` (2022-06-03); `c7c88ab` did not.

**Sto-BADS's rule.** `_sto_success_improvement_` computes μ = f_b − f_n, ε = √(s_b² + s_n²) (the SD of the difference under independence), and the half-width h = γ·ε·Δ^p. By default γ = 1.96 (overridable by `gamma_uncertain_interval`), p = `stobads_frame_size_scaling_power` = 2, and Δ = `mesh_size`, the poll or frame size.
- It returns 1 if μ ≥ h, −1 if μ ≤ −h or if the estimate is NaN, and 0 otherwise.
- This is what the docstring and the option descriptions say.
- Against Sto-MADS as I recall it: the success test f_s − f_0 ≤ −γ·ε_f·δ_p² matches, with ε_f replaced by the SD of the difference, and γ = 1.96 where Sto-MADS requires γ > 2 on a per-estimate accuracy bound. Both points belong to the open question of what ε should be.
- The certain and uncertain failures both shrink the mesh, as a failed Sto-MADS iteration does. Moving on an uncertain outcome (`opp_stobads`) is PyBADS's own.
- The calls pass the right values: the search passes (`fval`, `f_mu_search`, `fsd`, `f_sd_search`, `mesh_size`) and the poll passes (`fval`, `f_poll`, `fsd`, `f_sd_poll`, `mesh_size`).
- The poll keeps its best outcome and moves to the successful point with the largest improvement (W0-10).
- The one defect is F1.
- One observation for the open question of which uncertain outcomes should move the incumbent. The search moves on every uncertain outcome, including points estimated worse (μ < 0), and records such a move as an "incremental" search, which doubles `search_factor` (`bads.py:2010`, `2043`). On a noisy sphere this was 7 of 31, 9 of 46 and 10 of 40 search moves in 200-evaluation runs. The poll, by contrast, moves only to a point with a positive improvement.

### Q3. The acquisition and the Hedge

**LCB.** z = μ − √(ν·β_t)·s, with β_t = 2·ln(D·t²·π²/(6δ)), ν = 0.2 and δ = 0.1.
- t = `func_count` + 1, the index of the evaluation about to be made.
- D is the number of variables. The BADS paper puts the dimension where Srinivas et al.'s Theorem 1 has |D|, the size of a finite domain.
- s is the latent SD: gpyreg's `predict` without noise, MATLAB's `fs`.
- `acq_fcn_lcb.py:40-72` equals `acqLCB.m` for D ∈ {1, 3, 10} and several counts. MATLAB's lines date from 2017.

**Where the LCB is called:**
- The evolution-strategy (ES) search passes `search_acq_fcn[1]` (`es_search.py:154-157`).
- The search step (`bads.py:1864`) passes no parameter. It ranks the single point the ES search returns (MATLAB's `searchES` also returns one row), so the parameter makes no difference there.
- The poll (`2313`) ignores `poll_acq_fcn` (KD-B1-5 (b)).

**Hedge.** With n strategies:
- p = (1 − n·γ)·softmax(β·g) + γ.
- The chosen strategy's reward is the expected improvement of its point below `fvalold`: fs·(γ_z·Φ(γ_z) + φ(γ_z)) with γ_z = (fvalold − f)/fs. At fs = 0 it is max(0, fvalold − f). The expected-improvement formula agrees with Monte Carlo.
- The reward is weighted by 1/p (as in Exp3), and a strategy not chosen gets 0 (phat = ∞).
- The reward is divided by `MeshSize`, and the gains update as g ← `decay`·g + reward.
- `update_hedge` and `__call__` (`search_hedge.py:61-83`, `121-175`) equal `acqPortfolio.m` `'upd'` and `searchHedge.m`: the maximum difference is 0.0 over 200 random updates and over the 166 updates of the runs.
- The prediction for the chosen strategy is that of its search point. At level 0 it is the observation with SD 0. At levels 1 and 2 it is the GP rebuilt around the point; a NaN there gives a reward of 0, as MATLAB's NaN does. The other strategy gets (0, 1), with weight 0 when γ > 0.
- `fval_old` is the incumbent before any move, which is MATLAB's `fvalold`. The mesh size passed is that of the loop head.

### Q4. The geometry

**Derivation, for the rational-quadratic ARD kernel k(r) = σ²·(1 + r²/(2α))^(−α), with r = ‖(u − u′)/ℓ‖:**
- `len_scale` = ℓ = exp(log ℓ), in `u` units. `udist` = Σ((u − u′)/ℓ)², the squared distance in length scales.
- k falls to e⁻¹ at r = √(2α(e^{1/α} − 1)). Divided by √2, this gives the effective radius R = √(α(e^{1/α} − 1)), which tends to 1 as α → ∞ (the squared-exponential kernel falls to e⁻¹ at √2). That is the convention of `gpupdate.m`'s Matérn constants (1/√2 for Matérn 1). I checked k(√2·R) = e⁻¹ for α from 0.01 to 1e4.
- As α → 0, R grows without bound (5e20 at α = 0.01), so the radius test then keeps every point up to `n_train_max`. Both sides do the same.
- The training set is the points with d² ≤ (`gp_radius`·R)², clamped to [max(`n_train_min`, `n_train_max` − `buffer_ntrain`), `n_train_max`] and to the size of the log (`gaussian_process_train.py:1216-1231`, `gpupdate.m:86-110`).
- `poll_scale` = (ℓ_d / geometric mean of ℓ)^ρ, with ρ = `gp_rescale_poll`. It is clipped to [`search_mesh_size`, (ub − lb)/scale], with infinite bounds replaced by the plausible ones.
- That clip compares a dimensionless ratio with lengths in `u` units, and MATLAB's own comment reads "Perhaps this should just be PUB - PLB?". Both sides do the same.
- Its uses: ES-ell's `sqrt_sigma` = diag(ps/‖ps‖)·`mesh_size`·`search_factor` (`es_search.py:282-291`, `searchES.m:86-94`), and the poll vectors, where the division and the multiplication cancel (settled by W3-25).

**Code.** At all 33 refits of the four runs, the temporary data equalled the transcription of `gpupdate.m:283-328` on the GP's hyperparameters, with maximum difference 0. `udist` differs from `udist.m` by 3.6e-15. A rebuild without a refit left the geometry unchanged, which the runs asserted. The starting values (1, ones(D), 1.0) are those of `gpdefBads.m:186-190`.

**Units.** Both sides work in `u` space with scale 1. gpyreg's hyperparameter vector [log ℓ, log σ, log α] is the `hyp` of `covRQard_fast`.

**Differences, none reached at default:**
- *D = 1.* MATLAB sets `lenscale` = 1 because `ncovlen` (= D = 1) fails `ncovlen > 1`. PyBADS has used ℓ since `fef6c14`. The Python agrees with the derivation and differs from MATLAB. This is recorded as a fix of a defect shared with MATLAB (CHANGELOG, "One-dimensional problems"), not on the sheet.
- *Several hyperparameter samples* (not reachable, KD-B5-4). `poll_scale` would sum over samples without MATLAB's `hypweight` (`gaussian_process_train.py:556`), and `effective_radius` would be one array entry per sample instead of a value from the weighted α (`584-593`). These are latent only.

**Other differences recorded outside the sheet.**
- `acq_fcn_lcb` refuses a `sqrt_beta` that is zero, negative, non-finite or a function name. MATLAB accepts any numeric scalar (0 is the posterior mean) and names. This is in the CHANGELOG's upgrading list (wave 3), not on the sheet.
- When every LCB value is NaN, PyBADS picks a random index, where MATLAB's `min` returns index 1. In the search this is the same point, since the set has one row. In the poll it is not reachable in practice.

## 3. Findings

### F1. With Sto-BADS and `opp_stobads`, an uncertain poll whose points are all estimated worse "moves" the incumbent to itself and is marked as moved
- **Location:** `pybads/bads/bads.py:2455-2459` (with `2172-2178`, `2401-2407`, `2551`, `1470-1471`). MATLAB: no counterpart (KD-S-1).
- **Category:** control flow.
- **Proposed classification:** a suspected defect of Sto-BADS, which has no MATLAB counterpart. None of the four classes fits; the closest is unsure.
- **Confidence:** high that it happens; low on how much it matters.
- **Reached at default options:** no. It needs `stobads=True` (with `opp_stobads`, True by default) at uncertainty level 1 or 2.
- **History:** there is no MATLAB counterpart. The Python has been there since `9037851` (2022-09-22), whose rule `if opp_stobads and sto_success > -1: _update_incumbent_(u_poll_best, …)` already fell back to the incumbent. `0c56d86` (W0-10, 2026-09-26) rewrote the rule and kept the fallback.
- **What the code does:**
  - `u_poll_best` starts as the incumbent (`2173`) and changes only for a point with `poll_improvement > poll_best_improvement`, where `poll_best_improvement` starts at 0.
  - When no point succeeds and some point is uncertain (`sto_poll == 0`), the poll calls `_update_incumbent_(u_poll_best, …)` and sets `is_poll_moved = True`.
  - If every uncertain point is estimated worse (−h < μ ≤ 0), that "move" goes to the incumbent itself.
  - The poll is still recorded as moved, so `reset_gp` is set at the end of every pass until the next poll (`1470-1471`). Every search of the following round or rounds then rebuilds the local GP.
- **What it should do:** the comment and the changelog say that "an uncertain poll moves to the best polled point". Either it moves to a polled point, or it does not move and is not marked as moved.
- **Context:** the search, on the same outcome, moves to its point even when μ < 0 (`2010`). Which uncertain outcomes should move the incumbent is the open design question; the false move flag is a defect whichever way that question is decided.
- **Consequence if real:** extra rebuilds of the local GP without a refit, around an unchanged incumbent, so the searches use a GP whose training set is re-selected rather than extended. The numerical effect is small and there is a cost in time.
  - On a flat noisy target (`0.01·Σx²` + N(0, 1), D = 3, 200 evaluations), 1, 2 and 3 of the 3, 6 and 3 uncertain polls of seeds 0 to 2 moved to the incumbent itself.
  - These caused 3, 5 and 14 search rebuilds that nothing else required.
  - On the unit-scale sphere no poll was uncertain.
- **Suggested reproduction:** `check_stobads_flat.py` and `check_stobads_rebuilds.py` in the scratch directory. They wrap `_update_incumbent_` and count, within `_poll_step_`, the calls whose `u_new` equals `self.u` and whose `fval_new` equals `self.fval`. Output as above.
- **Test adequacy:** `test_stobads.py::test_uncertain_poll_moves_only_with_opp_stobads` asserts only that `_update_incumbent_` was called (`bool(poll["moves"])`), so it passes with a move to the incumbent itself.

## 4. Test adequacy notes
- `test_stobads.py::test_uncertain_poll_moves_only_with_opp_stobads` mirrors the implementation: it checks that the move function was called, not that the incumbent changed (F1).
- `search/test_search.py::test_hedge_reward_is_the_expected_improvement` is written from the specification, but only with g = 0, phat = 1 and a mesh size of 1. The decay, the importance weight 1/p, the division by the mesh size and the zero reward of the strategy not chosen go untested. `test_search_hedge` checks only shapes and `gp.y >= z`.
- `test_lcb_sqrt_beta_schedule_and_callable` checks `acqLCB`'s formula, which is sound.
- No test compares `len_scale`, `poll_scale` or `effective_radius` after a refit with `gpupdate.m`'s formulas. `test_poll_mads.py` checks only that `poll_scale` is divided out of the basis.
- No test exercises `_eval_improvement_` at `improvement_quantile` ≠ 0.5.
- No test checks which iterate the final quantile rule returns, including the skip of the first. `test_final_estimate_recorded_at_its_iterate` checks only where the final estimate is recorded.
- The optimization tests' tolerances, which are set from sweeps over seeds, would not detect an error in these formulas of small effect.
