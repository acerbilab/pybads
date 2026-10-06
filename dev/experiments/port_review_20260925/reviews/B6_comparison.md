<!-- Report of the B6 comparison reviewer (the GP model and its gpyreg objects, MATLAB comparison track), wave 1 of the port review, reading PyBADS at 95da7f1 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave1/B6_comparison/. Nothing in it is verified. -->

# B6 comparison review: GP model and its gpyreg objects

## 1. Coverage

**Read completely.**
- Python: `pybads/bads/gaussian_process_train.py` (all of it, at `95da7f1`) and `pybads/stats/get_hpd.py`.
- MATLAB: `gpdef/gpdefBads.m`, `gpml_fast/covRQard_fast.m`, `sq_dist_fast.m`, `infPrior_fast.m`, `infExact_fastrobust.m`; `utils/likGaussHe.m`, `mygp.m`, `gppred.m`, `gppriorrnd.m`, `gpset.m`, `prctile1.m`, `gpHyperOptimize.m`, `udist.m`; `acq/acqLCB.m`; `private/gpupdate.m` (lines 1-120 and 230-419); GPML `priorGauss.m` and `meanConst.m`, and the header of `covRQard.m`.
- gpyreg v1.3.3:
  - `covariance_functions.py`: `RationalQuadraticARD`, `_bounds_info_helper`, `_target_spread`.
  - `mean_functions.py`: `ConstantMean`, `NegativeQuadratic`, `_bounds_info_helper`.
  - `noise_functions.py`.
  - `gaussian_process.py`: `set_bounds`, `get_priors`/`set_priors`/`_write_prior_block`, `set_hyperparameters`, `fit`, `predict`, `__recompute_normalization_constants`, `__prior_masks`, `__compute_log_priors`, `__compute_nlZ`, `__gp_obj_fun`, `__training_cholesky`, `__core_computation`.
  - `f_min_fill.py`: how the design is drawn from the priors.

**Skimmed.**
- `bads.py`: the GP setup (`_init_optim_state_` 840-960), `_init_optimization_`, and the six `predict` call sites.
- gpyreg `GP.update` (only its handling of `s2`); `SliceSampler`'s signature.
- MATLAB `update_posterior.m`, the `minimizebnd.m` header, `funlogger.m`'s `S`, and the defaults in `bads.m`.
- The tests in `test_gaussian_process_train.py` and `test_gp_update_failures.py`.

**Not reached.** GPML's `sq_dist.m` and `solve_chol.m` (I used transcriptions instead); `gpdefStationaryNew.m`, `exact_inference_*.m`, `infExact_fast.m` and `fminbayes.m` (not used at MATLAB's defaults); the body of `minimizebnd`.

**Scripts** (all seeded, one BLAS thread), in `/tmp/claude-0/-home-user-pybads/4f4200d5-5a60-5de0-9b26-94c4f1e930f9/scratchpad/wave1/B6_comparison/`:
- `check_kernel.py`, `objective_check.py`, `predict_check.py`, `prior_samples.py`: component checks against MATLAB transcriptions.
- `run_sphere.py`, `run_ackley.py`, `mean_bounds_effect.py`: the mean's bounds (F1).
- `prior_sample_reach.py`: how often the prior sampler is reached (F2).
- `chol_mult.py`, `chol_effect.py`: noise inflation in the Cholesky factorization (F3).
- `plateau_init2.py`, `flat_region.py`: equal targets (F4).
- `first_rebuild.py`: whether the first rebuild refits (F5).

## 2. Answers to the first questions

### Q1. The kernel
Yes, the kernel matches.

- **Parameterization.** gpyreg's `RationalQuadraticARD` with PyBADS's hyperparameters computes GPML's `covRQard` as BADS calls it through `covRQard_fast`: hyp = [log ℓ_1..D, log sf, log α], and K = sf²(1 + r²/(2α))^(−α), with sf² = exp(2·hyp[D]).
- **Against a transcription** of `covRQard_fast.m` and `sq_dist_fast.m`, over 200 random cases with D from 1 to 7 and log α in [−5, 5]:
  - maximum relative error 2.3e-13 on K, 9.2e-15 on K(X, Z) and 4.6e-13 on dK;
  - dK has all D+2 derivatives, in the same log units and order: `K/M·C` for the length scales, `2K` for the output scale, `K(0.5·D2/M − α·log M)` for the shape;
  - central finite differences agree to 1.3e-8, and the diagonal equals sf².
- **The whole objective.** gpyreg's log posterior for the GP PyBADS builds (RQ ARD, `ConstantMean`, `GaussianNoise`) was compared with a transcription of `infPrior_fast` ∘ `infExact_fastrobust` with `likGaussHe`, `meanConst` and `priorGauss` (prior variance = gpyreg SD²). With unbounded hyperparameters, so that gpyreg's normalization constants are 1, it agrees to 6e-13 in value and 1e-12 in gradient at uncertainty level 0, and to 2e-15 at level 2.

### Q2. The hyperpriors and bounds
The table gives MATLAB's priors as (mean, variance) and gpyreg's as (mean, SD).

| Hyperparameter | MATLAB, definition → update at each rebuild | PyBADS, `_gp_hyp` → `local_gp_fitting` | Bounds, MATLAB / PyBADS |
|---|---|---|---|
| log ℓ | N(−1, 2²) → `iso`: N(½(uu+ll), ((uu−ll)/2)²) | N(−1, 2) → `iso`: (cov_mu, cov_sigma): same. `ard` is not ported (F6) | [log TolMesh, log min(100, 10·range)], same on both sides (`test_gp_log_lengthscale_bounds`) |
| log sf | N(1, 2²) → N(log std(y), 2²), with std using N−1 | (1, 2) → (log np.std(y), 2), with ddof 0 (F7, negligible) | [log TolFun, log(1e6·TolFun/TolMesh)], same |
| log α | N(1, 1²), never updated | (1, 1), never updated | [−5, 5], same |
| log sn | N(log NoiseSize(1), NoiseSize(2)² or 1) → centre log NoiseSize(1) + MeshNoiseMultiplier·log(mesh) | same, including TolFun at level 2 and multiplier 0 at levels above 0 | [log TolFun − 1, 5], same |
| mean | N(0, 1), never fitted under → N(prctile1(y, 90), (yrange/2)²) | (median(hpd_y), std(hpd_y)), fitted at initialization (F5) → (hazen 90th percentile, yrange/2): same. KD-B6-2 and KD-B6-3 hold | MATLAB (−∞, ∞) (`gpdefBads.m:173`) / PyBADS [min(hpd_y) − h/2, max(hpd_y) + h/2] of the initial design, fixed for the run: **F1** |

- **Percentile.** `prctile1` equals NumPy's `"hazen"` percentile to 4e-16 over 2000 random cases.
- **The mean's prior can fall outside its bounds.** It does, at default options:
  - 10-D sphere: bounds [0.438, 5.46]; by iteration 5 the prior centre is 0.25 and the fitted mean sits at 0.4384.
  - 6-D Ackley, both seeds: bounds [6.08, 13.5]; the prior centre is 2.1 and the fitted mean is 6.078; the prior's mass inside the bounds is down to 1e-32.
- **What gpyreg computes then.**
  - Each Gaussian prior is divided by its mass inside [lb, ub] (`gaussian_process.py:2045-2107`), and the sum of the log masses is subtracted (`:2170`, `:2354`).
  - While the mass is positive this is a constant, so the fit is the MAP of a prior truncated to the bounds. The mean is pinned at its lower bound.
  - Once the prior's centre is more than about 38 SDs outside the bounds, the survival function underflows and the mass is 0. The log prior is then +∞ at every point (nlZ = −∞), and `f_min_fill` maps the mean's design draws to +∞ (`isf(0)`, `f_min_fill.py:212`), which gives NaN objectives. The fit moves on gradients alone (reproduced, F1 (c)).
  - MATLAB's `infPrior_fast` adds unnormalized `priorGauss` terms on an unbounded mean, so neither the pinning nor the underflow can occur there.
- **Comments.** "Lower maximum constant mean" (line 959) describes neither the two-sided bounds nor MATLAB. "in BADS are all zeros" (863-865) is true of MATLAB, but the code starts elsewhere (F5).

### Q3. The noise and the prediction
- **Level 2.** The GP noise is exp(2θ) + S², where S is the SD the target returns; the logger's SDs are squared at `gaussian_process_train.py:1130`, `1159` and `1243`. This is `likGaussHe.m:26-27` with `s = gpstruct.s = S` (`gpupdate.m:111`), in the same units. `test_gp_noise_variances_with_target_noise` checks the squaring.
- **Levels 0 and 1.** The noise is the fitted constant alone. `[1,0,0]` gives it directly. `[1,2,0]`, which the option `uncertainty_handling=True` sets, gives the same, because `scale_user_provided` acts only together with `user_provided_add` (`noise_functions.py:43-46`). MATLAB has no `optimState.S` without SpecifyTargetNoise (`funlogger.m:49`), so `s = []`.
- **Prediction.** All six call sites use `gp.predict` with `add_noise=False`: `acq_fcn_lcb.py:48`, `bads.py:1830`, `2249`, `2597`/`2605`, `2745`, and `search_hedge.py:133`. That is the latent mean and variance, which is what MATLAB takes: `[~,~,fmu,fs2] = gppred` at `bads.m:653`, `919`, `1302` and `1401`, `acqPortfolio.m:46`, and `fmu`/`fs2` in `acqLCB.m:26-32`.
  - Numerically, gpyreg's prediction equals the latent GP formula to 2e-11 in both noise regimes, at levels 0 and 2.
  - The exception is a posterior whose noise gpyreg multiplied to get the factorization through (F3): it predicts from a model with 10–100 times the fitted noise.

## 3. Findings

### F1. The constant mean is bounded to the initial design's range for the whole run; MATLAB leaves it unbounded
- Location: `pybads/bads/gaussian_process_train.py:866`, `958-962` (with the re-centred prior at `315-324`); gpyreg `mean_functions.py:498-499`, `gaussian_process.py:2045-2107`, `2170`, `2354`, `f_min_fill.py:205-212`. MATLAB: `gpdef/gpdefBads.m:173` (and `219-222`; line 223 is a commented-out `bounds.mean{1}(1) = min(y)`), `utils/gpset.m:119-121`.
- Category: defaults.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: yes, at any uncertainty level, whenever the local targets fall below min(hpd_y) − h/2 of the initial design. It happened in every default run I made on the 10-D sphere (from iteration 5) and the 6-D Ackley (seeds 0 and 1, from iteration 5).
- History:
  - MATLAB line 173 has not changed since 2017; `d4fead5` touched only lines 164-165.
  - The Python matched MATLAB at `c7c88ab` (2022-06-02): bounds `(-inf, inf)`, prior `(0, 1)`. It diverged at `9037851` (2022-09-22), which took gpyreg's recommended bounds on `hpd_y`.
  - The underflow described below can occur only since `8afbe16` (2026-09-25), which re-centres the prior at each rebuild while the bounds stay.
  - The message of commit `2210046` records NaN hyperpriors from this cause as open. The item is not on the known-differences sheet.
- What the code does, what it should do, and why:
  - `_gp_hyp` sets the bounds of `mean_const` to gpyreg's `[min(hpd_y) − h/2, max(hpd_y) + h/2]` of the initial design's best 80%, and nothing resets them.
  - `local_gp_fitting` moves the prior to the 90th percentile of the local targets, with SD (y90 − y50)/5. As the run improves, that prior falls below the lower bound and the MAP of the mean is pinned there.
  - When the prior is more than about 38 SD below the bound, its mass rounds to 0. The log prior is then +∞ (nlZ = −∞) and design draws for the mean are +∞, giving NaN; the fit degenerates.
  - MATLAB's mean is unbounded, and its prior is unnormalized.
- Consequence if real: the GP's mean sits above what the local data and the prior support, so predictions away from the data are too high. The searched region looks worse than MATLAB's GP would say, and the fit is worse.
  - Refitting the last GP of the Ackley run: with PyBADS's bounds the mean is 6.078 and nlZ −152.5; with MATLAB's bounds the mean is 2.211 and nlZ −159.2.
  - The median prediction two length scales from the incumbent is 2.62 against 2.25.
  - Once the mass underflows, the fit is broken.
- Suggested reproduction: `mean_bounds_effect.py` (Ackley D=6, seed 0, 200 evaluations). Cases (a) and (b) are above. Case (c) moves the prior 40 SD below the bound: log prior +∞, nlZ −∞, "invalid value" warnings, mean at the bound. `run_ackley.py` and `run_sphere.py` print the bounds and priors per iteration.
- Test adequacy: `test_gp_mean_prior_recentred_at_each_rebuild` checks only the prior's centre and width; no test checks the mean's bounds or whether the prior lies within them.

### F2. `_get_random_samples_from_priors_` draws log-hyperparameters from N(exp(μ), exp(σ))
- Location: `gaussian_process_train.py:705-726` (the exponentials at 717-719). Callers: 193 (retry of the initial fit), 402 (second fit), 655 (`_robust_gp_fit_` retry). MATLAB: `utils/gppriorrnd.m:75` → `priorGauss` (`sqrt(s2)*randn+mu`); `private/gpupdate.m:389-395`; `utils/gpHyperOptimize.m:150-158`.
- Category: random draws / formula.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: yes. It is reached on a second fit (high noise or low mean, `use_slice_sampler` off by default) and on fit retries after a `LinAlgError`.
  - 6-D Ackley: 4-5 draws per run, through the second fit.
  - 4-D Rosenbrock: 6-7 draws per run, through `_robust_gp_fit_`.
  - 10-D sphere: none.
- History: the Python was written in `9037851` (2022-09-22) and has never matched. MATLAB's logic is unchanged: the same lines were at `utils/gpTrainingSet.m:497-502` and `gpHyperOptimize.m:151` before `d4fead5`.
- What the code does, what it should do, and why: for every key containing "log", the code exponentiates the prior's centre and SD and uses the draw as the log-hyperparameter. MATLAB samples N(μ, σ²) in log units. Measured over 20000 draws:

  | Hyperparameter | Prior (μ, σ) | Drawn by PyBADS (mean, SD) |
  |---|---|---|
  | log ℓ | (−0.8, 1.5) | (0.43, 4.5) |
  | log sf | (1.10, 2) | (3.0, 7.3) |
  | log α | (1, 1) | (2.7, 2.7) |
  | log sn | (−3.45, 1) | (0.03, 2.7) |
  | mean (no log) | | drawn correctly |

  In the Rosenbrock runs the drawn log sf is about 370 against a prior centre of 5.9. After averaging and clipping, that start sits at the upper bound log(1e9).
- Consequence if real: the second start and the retry starts are poor and often clipped to bounds (sf ≈ 1e9), so the second fit rarely improves on the first. A retry after a Cholesky failure starts from an extreme output scale. Not quantified.
- Suggested reproduction: `prior_samples.py` (above); `prior_sample_reach.py` (callers and drawn values in default runs).
- Test adequacy: `test_initial_fit_recovers_from_failure` exercises this function through an injected failure but asserts only that `fval` is finite.

### F3. A failed Cholesky factorization silently multiplies the GP noise (up to 1e9) in the fit objective and in the posterior used for prediction; MATLAB (CholAttempts = 0) raises
- Location: gpyreg `gaussian_process.py:3584-3666` (`__training_cholesky`, the ×10 retries at 3636-3657), used by the fit's objective and by posteriors. PyBADS catches only `LinAlgError`, which comes after ten attempts (`gaussian_process_train.py:611`, `515`, `524`, `1255`); `chol_attempts` is unread. MATLAB: `gpml_fast/infExact_fastrobust.m:36`, `77-80`; `gpdef/gpdefBads.m:314`; `bads.m:272` (`CholAttempts = 0`); `utils/gpHyperOptimize.m:73-176` (catch, then a restart with more noise); `private/gpupdate.m:340-354` (`post = []`).
- Category: cross-module.
- Proposed classification: port discrepancy. The library was substituted, and KD-B6-1 leaves "the Cholesky handling" open.
- Confidence: high on the mechanism; medium on the consequence.
- Reached at default options: yes.
  - 4-D Rosenbrock, seed 0: 349 of 1954 factorizations were inflated (up to ×1e9) and 6 failed after the ten attempts. 4 of the 13 GPs stored at iteration ends (iterations 4, 7, 8, 9) hold posteriors with `sn2_mult` 10–100, whose unmodified factorization fails.
  - 2-D Rosenbrock, seed 1: similar.
  - 10-D sphere: none.
- History: gpyreg has behaved this way since the first port. The MATLAB lines have not changed since before 2022. The two never agreed.
- What the code does, what it should do, and why:
  - MATLAB: when the factorization fails, the objective errors. `gpHyperOptimize` restarts from a prior draw with the noise start raised, and a failed posterior leaves `post = []`, so the GP is rebuilt.
  - gpyreg: it evaluates and keeps the model with 10^k times the noise, so the fit can converge where MATLAB's cannot, and predictions use the inflated posterior.
- Consequence if real: at iteration 8 the effective noise SD is 3.7e-3 against a fitted 3.7e-4. The residuals at the 10 best training points reach 7e-3, against their range of 1e-3, so the GP no longer resolves the local structure it is used for. The objective is also discontinuous where the inflation starts. The direction of the effect on results is unknown.
- Suggested reproduction: `chol_mult.py` (counts), `chol_effect.py` (stored posteriors, whether an unmodified factorization succeeds, and residuals).
- Test adequacy: `test_gp_update_failures.py` injects `LinAlgError` only; no test observes `sn2_mult > 1`.

### F4. Equal best targets in the initial design make `_gp_hyp` raise `ValueError`
- Location: `gaussian_process_train.py:960-961` (SD = `np.std(hpd_y)`), and `357`/`363` (log std(y) = −∞ at a rebuild); gpyreg `_write_prior_block` rejects both. MATLAB: `gpdefBads.m:171` (N(0,1), cannot fail), `219-222` and `293-295`.
- Category: defaults / control flow.
- Proposed classification: port discrepancy. The rebuild case is a suspected defect in both.
- Confidence: high for PyBADS (reproduced); medium for MATLAB (reasoned from the code, no MATLAB run).
- Reached at default options: only with such targets. The best round(0.8N) initial targets must be equal, for example an x0 and a Sobol design that all fall on a penalty plateau. Reproduced with D=3 and a target that returns 1e3 outside a small ball: all 5 initial targets are 1000, and `_gp_hyp:994` raises "The prior of mean_const has a sigma that is zero".
- History: introduced in `9037851`; `c7c88ab` had N(0, 1).
- What the code does, what it should do, and why: MATLAB's definition cannot fail. Its update gives −∞/NaN prior terms, the fit falls into `gpHyperOptimize`'s catch, and the run carries on with a degenerate GP. PyBADS stops, because KD-B5-3 lets the `ValueError` propagate. The later case (all local targets equal, so the output-scale prior is at −∞) raises the same way. I did not reach it in default runs; they end on stalls first (`flat_region.py`).
- Consequence if real: the run crashes at initialization.
- Suggested reproduction: `plateau_init2.py`.
- Test adequacy: none.

### F5. PyBADS fits a GP at initialization under the definition priors and from different starting values; MATLAB's first fit uses the updated priors
- Location: `gaussian_process_train.py:109-205`, `857-901`; `bads.py:1185-1205`. MATLAB: `bads.m:465-469`; `gpdefBads.m:48`, `153`, `164-165`; `gpupdate.m:276`, `374`.
- Category: defaults.
- Proposed classification: port discrepancy (the initial fit is an open item of KD-B6-1).
- Confidence: high on the difference; low on its effect.
- Reached at default options: yes, in every run.
- History: the initial fit dates from `c7c88ab`. MATLAB changed `hyp.mean` from 0 to the median of the lowest 80% in `d4fead5` (2022-10-31), after the Python of `9037851`.
- What the code does, what it should do, and why:
  - MATLAB's first fit starts from [0, 0, 0, log NoiseSize, median of the lowest ceil(0.8N) targets].
  - PyBADS starts from [log std(hpd_X) per dimension, log std(hpd_y), 0, log noise_size, median(hpd_y)], with hpd = round(0.8N). For N = 9 (the default at D = 4) that is 7 targets against MATLAB's 8.
  - PyBADS fits under the priors of the table above, bounded mean included.
  - The first rebuild then refits from these values; in my runs it refits at `func_count` 6 (D=3) and 10 (D=6).
- Consequence if real: the effect is on the start of the first refit and on `best_gp_hyp` until it is updated; small.
- Suggested reproduction: `first_rebuild.py`.
- Test adequacy: none.

### F6. `gp_cov_prior = 'ard'` is not ported and is silently ignored
- Location: `gaussian_process_train.py:329-351`; MATLAB `gpdefBads.m:254-274` (with `error` for an unknown value).
- Category: defaults.
- Proposed classification: port discrepancy (unported).
- Confidence: high.
- Reached at default options: no; only with `gp_cov_prior='ard'`.
- History: never ported (`c7c88ab`).
- What the code does, what it should do, and why: with `'ard'` PyBADS keeps N(−1, 2) for the whole run, where MATLAB sets a per-dimension empirical prior. Any other value is accepted where MATLAB errors.
- Consequence if real: a user who sets the option gets a different prior, without a message.
- Suggested reproduction: read the branch; any run with `gp_cov_prior='ard'` keeps the definition prior.
- Test adequacy: none.

### F7. The output-scale prior's centre uses `np.std` with ddof 0; MATLAB's `std` uses N−1
- Location: `gaussian_process_train.py:357`; MATLAB `gpdefBads.m:293`.
- Category: formula.
- Proposed classification: port discrepancy.
- Confidence: high.
- Reached at default options: yes.
- History: the same since `c7c88ab`.
- What the code does, what it should do, and why: the centre is log of the population standard deviation, where MATLAB uses the sample standard deviation.
- Consequence if real: the centre moves by ½·log(N/(N−1)), about 0.005–0.01 at N = 50–110, against a prior SD of 2. Negligible.
- Suggested reproduction: compare `np.std(y)` with `np.std(y, ddof=1)` on any training set.
- Test adequacy: none.

## 4. Test adequacy notes
- `test_gp_mean_prior_recentred_at_each_rebuild` and `test_gp_log_lengthscale_bounds` are written against MATLAB's formulas, but nothing checks the mean's bounds (F1) or the other priors (output scale, noise update).
- `test_initial_fit_recovers_from_failure` and `test_gp_update_failures.py` inject `LinAlgError` and assert that the run carries on. They mirror the retry code, and would catch neither F2's wrong-unit draws nor gpyreg's silent noise inflation (F3).
- `test_cov_identifier_to_covariance_function` and `test_meanfun_name_to_mean_function` check class identity only. No test compares `RationalQuadraticARD` or the objective with GPML; my checks pass.
- `test_bads_optimization.py` checks final accuracy within tolerances set from seed sweeps, so it would catch F1–F3 only through a loss of accuracy.

## Notes beyond my slice
- **B5, the noise-nudge retry.** `_robust_gp_fit_` (`gaussian_process_train.py:661-675`) raises the noise lower bound by the cumulative nudge (+1, +2, +3, …). MATLAB raises it by `nudge(2) = 0` (`gpHyperOptimize.m:161-164`). After 5 consecutive failures the lower bound (−7.9 + 15) passes the upper bound 5, and gpyreg's `set_bounds` would raise `ValueError`. Reasoned from the code, not reproduced.
- **B5, the optimizer.** gpyreg's optimizer (a random design of `init_N` 128→8 points, then L-BFGS-B at tolerance 1e-5) differs from MATLAB's 1–2 starts with TolFun 0.1, TolX 1e-4, MaxFunEval 150. I noted this and did not review it.
- **Python-only options.** `upper_gp_length_factor > 0` sets bounds that lines 930-933 overwrite, so it has no effect. `fit_lik=False` writes a `"delta"` prior, which gpyreg refuses with `ValueError`; MATLAB also refuses fixed noise. The slice-sampler path's private call `gp._GP__gp_obj_fun(hyp, False, True)` matches gpyreg 1.3.3's signature and returns the log posterior; it is reached only with `use_slice_sampler=True`.
