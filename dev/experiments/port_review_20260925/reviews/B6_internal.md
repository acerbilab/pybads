<!-- Report of the B6 internal reviewer (the GP model and its gpyreg objects, internal-correctness track), wave 1 of the port review, reading PyBADS at 95da7f1 in ../pybads-review and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave1/B6_internal/. Nothing in it is verified. -->

# B6 internal review: GP model and its gpyreg objects

Scripts and their outputs are in `/tmp/claude-0/-home-user-pybads/4f4200d5-5a60-5de0-9b26-94c4f1e930f9/scratchpad/wave1/B6_internal/`. Every script prints `pybads.__file__` (the review worktree) and `gpyreg.__file__` (the v1.3.3 clone). Runs are seeded, use one BLAS thread and at most 200 evaluations. Python lines are cited at `95da7f1`, gpyreg lines at v1.3.3.

## 1. Coverage

**Read completely**
- `pybads/bads/gaussian_process_train.py`. The functions the slice names, plus `_robust_gp_fit_`, `_get_gp_training_options`, `get_grid_search_neighbors`, `_estimate_noise_` and `add_and_update_gp` where the model depends on them.
- `pybads/stats/get_hpd.py`.
- gpyreg `covariance_functions.py` (`RationalQuadraticARD`, `_bounds_info_helper`, `_target_spread`), `mean_functions.py` and `noise_functions.py`.
- In gpyreg `gaussian_process.py`: `set_bounds`, `get_priors`/`set_priors`/`_write_prior_block`, `update`/`__apply_update`, `fit`, `__recompute_normalization_constants`, `__prior_masks`, `__compute_log_priors`, `__gp_obj_fun`, `predict` and `__training_cholesky`.
- gpyreg `f_min_fill.py` (the design of the fit).
- In `bads.py`: the GP setup (`915-957`), `_init_optimization_` (`1105-1215`), every `predict(` site (`1830`, `2249`, `2597`/`2605`, `2745`), the search and poll rebuild conditions, `_get_target_from_gp_`, `_save_gp_stats_` and `_is_gp_refit_time_`.
- `acq_fcn_lcb.py`; `search_hedge.update_hedge`; the GP tests in `pybads/testing/bads/`.
- MATLAB: only GPML `cov/covRQard.m`, which the slice names as the kernel reference.

**Skimmed**
- gpyreg `__core_computation`, `slice_sample.py` (signatures only) and `predict_full`.

**Not reached**
- The BADS paper. arxiv.org and papers.nips.cc are blocked by the egress proxy. The priors are therefore judged against the code comments, the option descriptions and the mathematics. From memory, the paper's empirical-Bayes forms are a mean at the 90th percentile with width (q90−q50)/5, log σ_f ~ N(log SD(y), 2²) and log α ~ N(1, 1). The code matches these, but I could not check them against the text.
- The refit policy and training-set choice (B5), except where noted.

**Dead code noted, no finding**
- `_estimate_noise_` (`1170-1209`) sorts descending, which is PyVBMC's convention; BADS's HPD region is the lowest targets. Its result `sn2hpd` is returned (`bads.py:1185`, `1244`) and never used.
- For B5: `gp.temporary_data["len_scale"]` is 1.0 when D = 1 (`455-463`), so at D = 1 the training distances are not scaled by the length scale.

## 2. Answers to the first questions

### Q1. The kernel

Yes: the kernel, its parameterization and its derivatives are GPML's `covRQard`.

- **Parameterization.** gpyreg's `RationalQuadraticARD` takes hyp = [log ℓ_1..log ℓ_D, log σ_f, log α], with σ_f² = exp(2h) and natural logs, in GPML's order.
- **Kernel.** k = σ_f²(1 + r²/(2α))^(−α), with r² = Σ((x−x′)/ℓ_d)².
- **Derivatives**, as in GPML:
  - ∂K/∂log ℓ_i = σ_f² M^(−α−1)(Δx_i/ℓ_i)²
  - ∂K/∂log σ_f = 2K
  - ∂K/∂log α = K(½r²/M − α log M)
- **Checks** (`k1_kernel.py`), over 50 random cases with D = 1–6 and log α ∈ [−5, 5]:
  - K against a transcription of `covRQard.m`: largest relative difference 9e-13 (the rounding of the transcription's `sq_dist`).
  - Cross-covariance: 1.6e-14. Diagonal: exact.
  - dK against GPML's derivative formulas: 1.8e-12. dK against central finite differences: 5e-9.
- **The fit objective** (`k2_objective.py`), for a GP built as PyBADS builds it (`[cov(D+2), noise, mean]`), at level 0 and at level 2 with `s2`:
  - The negative log marginal likelihood matches an independent multivariate-normal computation to 10 digits.
  - The log prior equals the sum of Gaussian log densities truncated to the bounds.
  - The analytic gradient matches finite differences to 3e-10 and 5e-10.
  - `gp._GP__gp_obj_fun(h, False, True)`, the call at `741`, equals `log_posterior`. That call is reached only with `use_slice_sampler=True`.
- Reached at default at every uncertainty level. The shape α is used in one more place, the effective radius (F7).

### Q2. The hyperpriors and bounds

gpyreg Gaussian priors are (mean, SD). The noise and output scale are log SD.

| hyperparameter | prior in `_gp_hyp` (initial fit only) | prior at each rebuild (`local_gp_fitting`) | bounds (set once in `_gp_hyp`, never updated) |
|---|---|---|---|
| log ℓ_d | N(−1, 2) | iso: N(½(log d_max + log d_min), ½(log d_max − log d_min)), where d are the distances between distinct training inputs in u units | [log tol_mesh, log min(100, 10(ub−lb)/scale)], e.g. [−13.17, 3.91] |
| log σ_f | N(1, 2) | N(log std(y_local), 2) | [log tol_fun, log(1e6·tol_fun/tol_mesh)] = [−6.91, 20.08] |
| log α | N(1, 1) | unchanged | [−5, 5] |
| log σ_n | N(log noise_size, 1 or noise_size[1]). Level 0: noise_size = √tol_fun, so −3.45. Level 1: 0. Level 2: log tol_fun = −6.91 | centre log noise_size + mesh_noise_multiplier·log Δ (0.5 at level 0, 0 at levels 1–2); SD kept | [log tol_fun − 1, 5] = [−7.91, 5] |
| m | N(median(hpd_y), std(hpd_y)) | N(q90_hazen(y_local), (q90 − median)/5), or the previous width when the range is 0 (KD-B6-2 matches the code) | [min(hpd_y) − h/2, max(hpd_y) + h/2], from the lowest 80% of the initial design |

- **The initial fit's priors only set a starting point.** The first rebuild was a refit in every run checked (sphere, shifted quadratic, Rosenbrock D=3; `r11`), so the initial priors matter only as the start of that refit.
- **What gpyreg computes.** Each Gaussian prior is truncated to its bounds and divided by its mass inside them. That mass is a difference of cdf values, or of sf values when lb > μ, computed in linear space. The constant does not move the MAP.
- **Consistency.**
  - The length-scale prior lies inside its bounds by construction.
  - The mean prior leaves its frozen bounds routinely (F1). When it is more than about 37.5 SD out, the mass underflows to 0 and the objective becomes −inf (F2).
  - With noise_size above e^5 the noise prior lies outside the noise bounds (F5).
  - The output-scale and noise bounds together admit models that cannot be factorized in double precision (F4).
  - At level 0 the noise prior's centre ends up to 2.1 SD below the noise floor: −10.0 against −7.91 at the final mesh 2⁻¹⁹. The comment "Increase minimum noise" says this floor is intended, so it is not reported.
  - The comment "Lower maximum constant mean" (`959`) does not describe what the code sets.
- **Other defects.** The prior sampler does not sample the priors (F3). Degenerate inputs crash (F6, F8). One option has no effect (F9).

### Q3. The noise and the prediction

- **Level 0.** `GaussianNoise(constant_add)`: σ_n² = exp(2h), one hyperparameter.
- **Level 1.** The flags `[1, 2, 0]` give constant_add=True, user_provided_add=False and scale_user_provided=True. gpyreg ignores the scale flag when user_provided_add is off, so σ_n² = exp(2h) alone, as "Infer noise" says. `gp.s2` then holds NaN (the logger's unset `S`), and the noise function never reads it.
- **Level 2.** σ²_n,i = exp(2h) + s_i², where s_i is the SD the target returned. The SDs are squared at `1159`, `1130` and `1243`. Checked row by row in a 150-evaluation run (`r6_noise_levels.py`: 216 rebuilds, 107 added points): gp.s2 = S², gp.y = Y, and the noise function's output equals exp(2h) + s2, all with a difference of exactly 0.
- **Caveat.** The posterior that predictions use can carry the noise multiplied by 10–1000 (F4).
- **Predictions.** Every call uses `predict()` defaults, so it gets the latent mean and latent variance (add_noise=False) from the single hyperparameter sample. The call sites are:
  - the LCB in search and poll (`acq_fcn_lcb.py:48`) and the PoI of the poll (`bads.py:2203-2216`);
  - the target (`2597`/`2605`);
  - the noisy search and poll estimates (`1830`, `2249`);
  - the history re-evaluation (`2745`);
  - the hedge update (`search_hedge.py:133`).

  The latent quantities are what LCB, PoI and an estimate of f need.
- **Calibration statistics (checked, not a finding).** The GP calibration statistics (`_save_gp_stats_`) compare noisy observations with the latent SD. A counterfactual that records the observation SD √(fs² + sn²) changed nothing in 6 level-1 runs (D = 2, 3 × 3 seeds): the same 44 refits per dimension and the same results (`r14_calibration.py`). With `normalpha_level = 1e-6` this makes no difference.

## 3. Findings

### F1. The constant-mean bounds are fixed on the initial design, while the mean prior is re-centred at each rebuild and routinely falls outside them
- Location: `pybads/bads/gaussian_process_train.py:866`, `958-962` (bounds from gpyreg's recommendation on the lowest 80% of the initial design); `315-324` (the prior re-centred); `bads.py:1185` (the only call of `init_and_train_gp`); `local_gp_fitting` never sets bounds. MATLAB: not examined (internal track).
- Category: state/caching
- Proposed classification: unsure
- Confidence: high on the mechanism; medium that it is unintended.
- Reached at default options: yes, at every level, with the default `gp_mean_fun='const'`.
- History: the bounds date from `c7c88ab` (2022-06-02). The re-centred prior has been applied only since `ab4dded` (2026-09-25, #66). Before #66 the prior stayed at the initial one, inside the bounds, so the inconsistency became reachable with #66. MATLAB history not examined.
- What the code does:
  - The mean's bounds are [min(hpd_y) − h/2, max(hpd_y) + h/2] of the initial design's lowest 80%, for the whole run.
  - The prior follows the local training set: N(q90, (q90 − q50)/5), in the comment's words "an empirical prior, re-centred at each rebuild".
  - An empirical prior that follows the data, inside bounds that do not, can lie outside its own support. The MAP is then pinned at a bound, and the prior no longer centres the mean at the 90th percentile.
  - The comment "Lower maximum constant mean" describes an upper cap, not what is set.
- What was measured (`r1_priors_in_run.py`, `r11_mean_bounds_cf.py`):
  - Sphere, D=2, optimum inside the plausible box: the prior centre is outside the bounds at 28 of 37 rebuilds, by up to 4.5 SD. At all 5 refits the mean sits at its upper bound 3.44 while the prior centre is 6.5–7.0; with widened bounds the same refits give 6.5–18.
  - Linear + quadratic, D=2, optimum at −50 outside the plausible box [−1, 1]: bounds [−308, 186], local y in [−5000, −653], fitted m = 182, above every local target.
  - Rosenbrock shifted by 4, D=3: bounds [1753, 1.4e5], local y ≤ 25, m = 1753. The counterfactual with widened bounds (same data, same generator state) fits m = 3.9.
- Consequence if real: away from the data the GP reverts to the bound instead of the 90th percentile. Early in the sphere run that reversion level is lower than intended, so predictions far from the data are more optimistic. When the optimum lies below the initial design's values, it is orders of magnitude above all local data. The log marginal likelihood is similar with and without the widened bounds, so the effect is on extrapolation (LCB and PoI away from the data), not on the fit to the data. The effect on results was not measured. F1 also sets up F2.
- Suggested reproduction: `r11_mean_bounds_cf.py` (outputs as quoted above).
- Test adequacy: `test_gp_mean_prior_recentred_at_each_rebuild` checks the centre and SD only. No test relates the prior to the bounds.

### F2. When the mean prior is more than about 37.5 SD outside its bounds, gpyreg's normalization underflows: the log prior is +inf, the fit objective −inf, and the random design's mean values are infinite
- Location: gpyreg `gaussian_process.py:2045-2107` (the mass inside the bounds as a cdf/sf difference), `2170` (`log_norm = Σ log`), `2354`; gpyreg `f_min_fill.py:208-217` (design draws through the ppf/isf of the truncated prior). Triggered by PyBADS through `gaussian_process_train.py:315-324` and `962`. MATLAB: not examined.
- Category: cross-module
- Proposed classification: unsure. The comparison track should check whether MATLAB's priors are normalized over the bounds at all; if not, this is a port discrepancy.
- Confidence: high
- Reached at default options: yes, with a target whose optimum lies far below the initial design's values. Rosenbrock shifted by 4 (plausible box [−1, 1], bounds [−10, 10], 200 evaluations): the constant is 0 at 4 rebuilds (1 refit) at D=2 and at 34 rebuilds (4 refits) at D=3. In a Gaussian well plus Rosenbrock, D=2 seed 1: 6 rebuilds (1 refit) (`r4_underflow_runs2.py`).
- History: the normalization is gpyreg's. The PyBADS path to it exists since `ab4dded` (2026-09-25).
- What the code does:
  - At z = 37 the constant is 5.7e-300; at z = 38 it is 0.0, with "divide by zero in log", and `log_posterior` is +inf (`k3_norm.py`). The objective is then −inf at every hyperparameter, with a finite gradient.
  - `f_min_fill` maps the design's uniform draws through ppf(0), so the mean coordinate of every random design point is −inf (`k5_design.py`).
  - The constant does not depend on the hyperparameters. It should be computed in log space (logcdf/logsf), or left out of the objective the optimizer sees.
- Consequence if real (`r5_counterfactual.py`, each such refit redone with the zero constant set to 1, same data and generator state):
  - Each affected refit took 2–12 s instead of 0.01–0.13 s. L-BFGS-B runs to its iteration limit on a flat −inf objective: 2503 iterations against 15 in `k3`. The D=3 run took 36 s against about 2 s.
  - The returned hyperparameters are not the MAP. At D=2: log σ_f 13.7 against 11.9, log α 2.06 against 0.46, negative log posterior 1460 against 1359. At D=3 the difference goes either way.
  - The predictive mean changed by ≤ 0.11.
  - With the slice sampler (non-default), the log posterior is +inf everywhere.
- Suggested reproduction: `k3_norm.py`, `k5_design.py`, `r5_counterfactual.py`.
- Test adequacy: no test covers a prior outside its bounds.

### F3. `_get_random_samples_from_priors_` draws N(exp μ, exp σ) for log hyperparameters, and raises on a block without a prior
- Location: `gaussian_process_train.py:717-719` (the exponentiation), `713` (`value[0]` on `None`); callers `193` (initial fit failures 1–2), `402` (second fit), `655` (retries in `_robust_gp_fit_`).
- Category: formula
- Proposed classification: unsure. It is a defect against its own docstring; whether MATLAB's `gppriorrnd` differs is for the comparison track.
- Confidence: high
- Reached at default options: yes, through the retries after a `LinAlgError` in a fit. There were 6 calls in the Rosenbrock D=2 and D=4 runs, 1 in Ackley D=4, and none in the sphere and noisy-sphere runs (`r7_sampler_reach.py`). Fits fail inside L-BFGS-B (`fit:1974`; `r8_fit_failures.py`).
- History: `9037851` (2022-09-22).
- What the code does:
  - The priors are Gaussian on the log hyperparameters, so a draw from the prior is N(μ, σ). The code exponentiates both parameters of every block whose name contains "log" and stores the draw as a log value.
  - Over 20,000 draws (`k4_prior_samples.py`):
    - log ℓ, prior N(−1.20, 2.5): draws with mean 0.29 and SD 12.2.
    - log σ_f, prior N(3.91, 2): mean 50.0, SD 7.5; 99.995% of draws lie above the upper bound 20.7.
    - log α, prior N(1, 1): N(2.72, 2.73).
    - log σ_n, prior N(−5.53, 1): N(0.008, 2.70).
    - The mean (no "log" in its name) is drawn correctly.
  - In the runs, the draws of log σ_f were 2212 and 2221.
  - A block with no prior raises `TypeError`. With `gp_mean_fun='negquad'` the run stops at the first retry (`r13_negquad_tb.py`).
- Consequence if real: the retry's own starting point, 0.5·(draw + previous), clipped to the bounds by `f_min_fill.py:88`, has σ_f at its upper bound (5e8) and often the length scales and noise at a bound as well. Only the fit's random design, which gpyreg samples correctly, is left to diversify the retry. The effect on results is expected to be small; it was not measured.
- Suggested reproduction: `k4_prior_samples.py`.
- Test adequacy: `test_initial_fit_recovers_from_failure` reaches this function but asserts only that `fval` is finite.

### F4. At default options a large share of GP posteriors carry the noise multiplied by 10–1000 by gpyreg's Cholesky retries, and the fitted noise does not show it
- Location: gpyreg `gaussian_process.py:3584-3666` (`__training_cholesky`: the noise multiplied by 10 per failed attempt, up to 10 attempts, and kept as `sn2_mult`), used by `predict`. PyBADS's bounds: `gaussian_process_train.py:915` (noise ≥ log tol_fun − 1) and `936-941` (σ_f ≤ 1e6·tol_fun/tol_mesh). MATLAB: KD-B6-1 leaves the Cholesky handling open.
- Category: cross-module
- Proposed classification: unsure
- Confidence: high that it happens, as measured.
- Reached at default options: yes, at level 0 (`r15_sn2mult.py`):
  - Rosenbrock D=2: 23 of 66 GP states (×10: 19, ×100: 4).
  - Rosenbrock D=4: 59 of 240 (×10: 35, ×100: 13, ×1000: 11).
  - Sphere D=2 and Ackley D=4: none.
- History: the bounds date from `c7c88ab`; the multiplier is gpyreg's.
- What the code does:
  - In the affected states (`r16`), log σ_f ≈ 13.1 (σ_f ≈ 4.8e5) while the local std(y) is 340–2150, σ_n ≈ 0.005–0.011, and the uninflated matrix is numerically singular (eigenvalue ratio about −1e-16).
  - The bounds admit σ_f/σ_n up to e^(20.08+7.91) ≈ 1.4e12, a variance ratio of about 2e24 against 1/eps = 4.5e15.
  - The objective at such points is the likelihood of the inflated model, so it is discontinuous in the hyperparameters. This is plausibly why fits fail inside L-BFGS-B.
  - `get_hyperparameters`, and so PyBADS's high-noise check, report the uninflated noise.
- Consequence if real: the effective noise SD is up to √1000 ≈ 32 times the fitted one (e.g. 0.052 against 0.0052). The posterior no longer interpolates the data to the fitted precision, and the latent variances near the data are larger. The effect on results was not measured.
- Suggested reproduction: `r16_sn2mult_detail.py`.
- Test adequacy: none.

### F5. The noise upper bound is the constant log SD 5 (SD 148), below the prior centre whenever noise_size > 148
- Location: `gaussian_process_train.py:915`; the prior at `976`, `303-306`; `basic_bads_options.ini:16-17` ("Base observation noise magnitude (SD)").
- Category: defaults
- Proposed classification: possibly intentional
- Confidence: high on the mechanism; medium on its importance.
- Reached at default options: at level 1, whenever the target's noise SD exceeds about 148, which depends on the target's scale. With `noise_size > 148` the prior is outside the bounds by construction.
- History: `c7c88ab` (renamed in `cdc2e0f`, 2023-06-10).
- What the code does (`r10_noise_bound.py`; target 1e4·|x|² plus noise of SD 500, D=2):
  - With `noise_size=500`, the prior N(6.21, 1) lies beyond the bound, and the fitted log noise SD is 5.0 at all 13 refits.
  - With the default noise_size, the fitted value is 0.15–0.38 for 7 refits (the noise absorbed into σ_f ≈ e^10.5), then 5.0.
  - Nothing ties the bound to noise_size or to the targets.
- Consequence if real: the GP's noise is underestimated (3.4 times in SD here), and the rest goes into the latent function. The final points in r10 were comparable (|x| < 0.05); not measured further.
- Suggested reproduction: `r10_noise_bound.py`.
- Test adequacy: `test_noise_size_forms` uses noise_size 2 and checks only the prior.

### F6. A target that is flat on the initial design stops the run with a ValueError from `set_priors`
- Location: `gaussian_process_train.py:960-961` (SD = std(hpd_y) = 0); `357` (log std(y) = −inf at a rebuild); raised by gpyreg `_write_prior_block` (`gaussian_process.py:171-188`).
- Category: control flow
- Proposed classification: unsure
- Confidence: high
- Reached at default options: yes, with a target whose lowest 80% of initial-design values are equal, such as a plateau over the plausible box.
- History: `c7c88ab`, reformatted in `157bd09`.
- What the code does:
  - The run stops at initialization with "The prior of mean_const has a sigma that is zero or negative" (`r9_degenerate.py`). With x0 lower by 1e-3, the run proceeds.
  - gpyreg's own recommended bounds handle equal targets (`covariance_functions.py:10-50`, "a range of one is assumed instead"); PyBADS's own std bypasses that.
  - The ValueError is not caught (KD-B5-3 covers only LinAlgError).
  - The rebuild path (`357`) was not reproduced: it needs at least `n_train_min` equal neighbours.
- Consequence if real: no optimization on such targets.
- Test adequacy: none.

### F7. The effective radius corresponds to an RQ kernel without the factor 2, not to the kernel gpyreg computes
- Location: `gaussian_process_train.py:493-506`; used at `1111`.
- Category: formula
- Proposed classification: unsure (the intended definition is not documented).
- Confidence: low
- Reached at default options: yes (`use_effective_radius=True`).
- History: `c7c88ab`.
- What the code does:
  - √(α(e^(1/α) − 1)) is the distance at which (1 + r²/α)^(−α) = e^(−1). That is the kernel whose squared-exponential limit is exp(−r²), at one length scale.
  - For gpyreg's (1 + r²/(2α))^(−α), the corresponding radius (k = e^(−1/2)) is √(2α(e^(1/(2α)) − 1)).
  - At the code's radius, gpyreg's kernel is 0.58 at log α = 1, 0.54 at 0, 0.46 at −1 and 0.40 at −2 (`k6_effrad.py`). The radii compared (code against gpyreg-consistent) are 1.10 against 1.05, 2.28 against 1.46, 14.8 against 3.26, and about 1e31 at the bound log α = −5.
- Consequence if real: the training radius is larger than one consistent with the kernel, much larger at small α. The training-set size moves only between `n_train_min` and `n_train_max`, so the effect is bounded and small.
- Test adequacy: none.

### F8. `fit_lik=False` stops the run: gpyreg has no "delta" prior
- Location: `gaussian_process_train.py:896-898`, `979`.
- Category: control flow
- Proposed classification: unsure
- Confidence: high
- Reached at default options: no (`fit_lik=False`).
- History: `c7c88ab`.
- What the code does: it raises "Unknown hyperprior type delta" at initialization (`r9`), so the option "Fit the likelihood term" cannot be switched off. Equal noise bounds, which gpyreg treats as fixed, would express a known noise.
- Test adequacy: none.

### F9. `upper_gp_length_factor` has no effect
- Location: `gaussian_process_train.py:908-913`, overwritten unconditionally at `930-933`.
- Category: defaults
- Proposed classification: possibly intentional (a dead option; PyBADS-only, KD-B1-4).
- Confidence: high
- Reached at default options: no (default 0).
- History: `c7c88ab`; the overwrite's log was fixed in `ab4dded`.
- What the code does: with 0.05 the length-scale bounds are identical to the default, [−13.17, 4.605] (`r9`). The option's description promises an upper bound from the plausible box.
- Test adequacy: none.

### F10. The non-constant mean options: `'negquad'` is concave, and most accepted names cannot be built
- Location: `bads.py:935-955` (12 names accepted); `gaussian_process_train.py:545-573` (3 built), `964-970`.
- Category: defaults
- Proposed classification: unsure (PyBADS's own options, KD-B1-4).
- Confidence: high (analytic, plus the crash in F3).
- Reached at default options: no.
- History: `c7c88ab`.
- What the code does:
  - gpyreg's `NegativeQuadratic`, m0 − ½Σ((x − x_m)/ω)², is PyVBMC's mean for log densities. In a minimizer it predicts ever lower values away from x_m, the wrong curvature, which draws the LCB outward. It also crashes at the first retry (F3).
  - 9 of the 12 names that `_init_optim_state_` accepts ("se", "negquadse", …) raise "Unknown mean function!" when the GP is built.
- Test adequacy: `test_meanfun_name_to_mean_function` only mirrors the name-to-class mapping.

## 4. Test adequacy notes

- `test_cov_identifier_to_covariance_function` covers identifiers 2, 3, [3, k] and 0, but not 1 (`RationalQuadraticARD`), the only one a run reaches. No PyBADS test compares the kernel or the objective's gradient with GPML.
- `test_gp_mean_prior_recentred_at_each_rebuild` and `test_gp_log_lengthscale_bounds` restate the formulas, citing `gpdefBads`. Neither checks that a prior lies within its bounds (F1, F2, F5).
- `test_initial_fit_recovers_from_failure` exercises `_get_random_samples_from_priors_` but asserts only that the result is finite (F3).
- `test_noise_size_forms` checks the noise prior but not the noise bounds (F5).
- `test_gp_noise_variances_with_target_noise` checks gp.s2 = S² row by row against what the target returned, so it tests the specification rather than mirroring the code.
- No test covers an optimum far below the initial design's range, targets that are flat on the initial design, noisy targets with SD above 148, `fit_lik=False` or `upper_gp_length_factor > 0`.
