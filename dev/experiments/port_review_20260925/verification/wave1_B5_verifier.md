<!-- Report of the verifier of wave 1, slice B5 (the two B5 reports and the survey rows it was given as B5-R1 to B5-R3), reading PyBADS at 95da7f1 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave1/B5_verifier/. -->

# Wave 1 verification: B5

I verified 23 items: the 14 internal findings (I-F1…F14), the 12 comparison findings (C-F1…F12) and the three extra rows (B5-R1…R3). Where both reports describe the same behaviour I verified it once. Two splits: I-F7 into (a) the SD used and (b) the scale-blind test; I-F12 into (a) D = 1 and (b) the multi-sample sum.

- **Where I disagree with the reports:**
  - I-F10 is shared with MATLAB, not a port discrepancy.
  - The comparison report's "not reached at default" for C-F2/C-F4 is wrong: fit failures are common in deterministic runs.
  - I-F1's "worse on all 5 seeds" does not replicate.
  - B5-R2's negquad clause does not hold.
  - I-F7's "the test never fired" is not reproduced.
- **Setup:** all checks use PyBADS at `95da7f1` and gpyreg v1.3.3. Every script prints `pybads.__file__ = /home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__ = /home/user/gpyreg-v1.3.3/gpyreg/__init__.py`.
- **Where things are:** scripts and outputs are in `/tmp/claude-0/-home-user-pybads/4f4200d5-5a60-5de0-9b26-94c4f1e930f9/scratchpad/wave1/B5_verifier/`. My own transcriptions of MATLAB's `swtest.m`, `gppredcheck.m`, `IsRefitTime` and `prctile1.m` are in `matlab_transcriptions.py`. On platykurtic samples its Shapiro-Wilk branch matches scipy to |Δlog p| ≤ 1.9e-8.
- **Dating:** no MATLAB commit after 2022-06 changes any line cited below. `d4fead5` only renames `gpTrainingSet` to `gpupdate`, and `74919c0` fixes an unrelated typo.
- **The sheet:** no finding contradicts an entry.

## 1. Summary

| Finding | Classification | Reached at default | Dating | Confidence |
|---|---|---|---|---|
| I-F1 (`plb`/`pub` swapped) | confirmed port discrepancy | yes, but only on problems unbounded in every variable; all levels | never agreed (Py `c7c88ab` 2022-06; MATLAB 2017) | high |
| I-F2 = C-F4 (prior draw N(e^μ, e^σ)) | confirmed port discrepancy | yes, level 0 (fit failures in 21 of 53 deterministic refits); not seen at level 1 (0 of 29) | never agreed (Py `9037851` 2022-09; `gppriorrnd.m` 2018) | high |
| I-F3 = C-F2 (cumulative raise of the noise lower bound) | confirmed port discrepancy | yes, level 0; not seen at level 1 | never agreed (Py `c7c88ab`; MATLAB 2017) | high |
| I-F4 = C-F3 (all tries fail: crash, no exit −1) | confirmed port discrepancy | no (at most 2 failures in a row in 82 refits) | never agreed | high |
| I-F5 = C-F8 (χ² quantiles halved) | confirmed port discrepancy | yes, all levels (n = 2 checks, unreliable flag only) | never agreed (`c7c88ab`; 2017) | high |
| I-F6 = C-F7 (one stat read as none; periodic refit at period+1) | confirmed port discrepancy | yes, all levels | never agreed (current form `9037851`; 2017) | high |
| I-F7a = C-F6 = B5-R3 first clause (latent SD; 1e-6 replacement) | confirmed port discrepancy | yes, all levels | never agreed (latent SD `c7c88ab`, replacement `cdc2e0f` 2023-06; MATLAB 2017) | high |
| I-F7b (the n ≥ 3 test ignores scale and location) | design question, shared with MATLAB | yes, n ≥ 3 | the two agree | high |
| C-F9 (scipy `shapiro` in place of `swtest`) | confirmed port discrepancy (a substitution not on the sheet) | yes, level 0 (5 of 163 checks); 0 of 125 at level 1 | never agreed (`c7c88ab`; 2017) | medium-high |
| I-F8 = C-F1 (`reset_gp` never cleared by a rebuild) | confirmed port discrepancy | yes, all levels | never agreed (`c7c88ab`; 2017) | high |
| I-F9 = C-F11 (the `&` slip) | confirmed, inert | yes (the condition is always true) | never agreed (`c7c88ab`; 2017) | high |
| I-F10 (`poll_training=False` records refits it cancels) | confirmed shared defect (not a port discrepancy) | no | the two agree (Py `c7c88ab`; MATLAB 2017) | high on the code, medium that it is a defect |
| I-F11 = C-F10 = B5-R1 last sentence (geometry after a rescued refit) | confirmed port discrepancy, inert today | no (0 of 393 rebuilds) | never agreed (`c7c88ab`; kept by `685da15`) | high |
| B5-R1, the retry itself | design question (left open in KD-B5-2) | no | never agreed (`c7c88ab`; MATLAB 2017) | high |
| I-F12a (D = 1: `len_scale` = 1) | design question; Python matches MATLAB | yes, at D = 1 | the two agree | high |
| I-F12b (multi-sample sum) | confirmed, inert | no (one sample, KD-B5-4) | `c7c88ab`; MATLAB differs since 2017 | high |
| I-F13 (`init_N` schedule: 0/0 or extrapolation) | confirmed defect | no (small `max_fun_evals`) | Python only (`8ff10f5` 2022-11) | high |
| I-F14 (uncapped retry of the initial fit) | confirmed defect | no | Python only (`9037851`) | high |
| C-F5 (refit starts from gpyreg's design) | design question | yes, all levels | never agreed (`c7c88ab`; defaults `8ff10f5`) | high on the difference, medium on its size |
| C-F12a/b (percentile rule; no minimum-points stop) | confirmed port discrepancy | (a) yes, level 0; (b) no | never agreed (`c7c88ab`; MATLAB 2017) | high |
| C-F12c (exit flag 0 instead of 1) | confirmed, inert (display only) | yes | never agreed | high |
| B5-R2 (the retry reads `tmp_gp`) | confirmed defect with the slice sampler, inert at default; the negquad clause is not a defect | no | `c7c88ab`; gpyreg 1.3.3 changed what a failed fit leaves | high |
| B5-R3 second clause (`_re_evaluate_history_` uses each stored GP's geometry) | holds at `95da7f1` (outside the slice) | levels 1 and 2 | never agreed (`9037851`; 2017) | high |

## 2. Per finding

### I-F1: `optim_state["plb"]`/`["pub"]` are swapped
- **Lines:**
  - `bads.py:650-651` put the lower plausible bound in `optim_state["pub"]` and the upper one in `optim_state["plb"]`.
  - The only reader is `gaussian_process_train.py:477-489`.
  - MATLAB `private/setupvars.m:67-68` stores them correctly, and `private/gpupdate.m:301-308` uses them.
- **Check (`v01_pollscale_unbounded.py`):**
  - Unbounded 3-D quadratic, seeds 11 and 12: `poll_scale=[-2. -2. -2.]` at every refit, while the length scales were for example `[92.059 43.359 6.201]`.
  - With the names put right: `poll_scale=[2. 1.345 0.164]`.
  - With bounds of ±20: `[4.773 1.126 0.186]`.
  - The poll divides by `poll_scale` and multiplies back (`poll_mads_2n.py:37`, `bads.py:2082-2086`), so it is unaffected. ES-ell (`es_search.py:289-293`) becomes isotropic.
- **Reach:** only problems unbounded in every variable. PyBADS refuses any mix of bounded and unbounded variables (new item N2).
- **Consequence (`v01b`, `v01c`):**
  - My 3-D target, seeds 20-25: worse as is on 3 of 6 (log10 differences `[-0.93 -1.49 0.19 -0.59 0.62 0.9]`).
  - The reviewer's 4-D ellipsoid class at seeds 30-35: also 3 of 6 (`[0.18 0.69 0.72 -0.31 -0.84 -1.95]`).
  - The report's "all 5 seeds worse" does not replicate. The defect is real; the direction of its effect is not established.
- **Dating:** Python `c7c88ab`, never changed; MATLAB lines from 2017. Never agreed.
- **Disposition: fix.**
  - Swap the two assignments at `bads.py:650-651`, and correct the AGENTS.md sentence that records the swap as a fact.
  - Default runs with finite bounds do not move: gate them with the fingerprint.
  - The effect needs a population comparison with a configuration that has fully unbounded problems.

### I-F2 = C-F4: `_get_random_samples_from_priors_` exponentiates the mean and SD of log priors
- **Lines:** `gaussian_process_train.py:717-719`. MATLAB `utils/gppriorrnd.m:75` calls `priorGauss(mu,s2)`, which returns `sqrt(s2)*randn+mu`.
- **Check (`v02_prior_sampler.py`, 4000 draws from a run's GP):**
  - `covariance_log_lengthscale[0]`: prior N(−2.974, 4.273); draws have mean −1.318, SD 70.69, and 91% fall outside the bounds.
  - `noise_log_scale`: prior N(−3.454, 1); draws have mean 0.024, SD 2.71.
  - `mean_const` (no "log" in its name) is drawn correctly.
- **Reach:** the retries after a fit failure use it, and those are frequent.
  - `v04_fit_failure_rate.py`: 21 of 53 refits in deterministic runs had at least one `LinAlgError`, at most 2 in a row. Noisy runs had 0 of 29.
  - The comparison report's "rare" is wrong for level 0.
- **Consequence:** the retry start `0.5·(draw + previous)` is one row of gpyreg's 8-128-point design, so the effect is diluted. I did not isolate its size.
- **Dating:** added in `9037851` (the `c7c88ab` sampler was different); `gppriorrnd.m` last changed in 2018. Never agreed.
- **Disposition: fix.** Draw N(μ, σ) for every Gaussian prior. This moves default runs, so it needs a population comparison at default; combine it with the other `_robust_gp_fit_` fixes.

### I-F3 = C-F2: the noise lower bound rises by the cumulative nudge
- **Lines:**
  - `gaussian_process_train.py:668-675`: the bound read from `tmp_gp`, which already holds the earlier raises, gets `+ noise_nudge`, where `noise_nudge` is 1, 2, 3, … cumulatively. The bound is therefore lb₀ + k(k+1)/2 after k failures.
  - `nudge[1]` is never read.
  - MATLAB `utils/gpHyperOptimize.m:161-166`: the bound gets `+ nudge(2)`, which is 0 by default, and the start gets `+ noiseNudge`.
- **Check (`v03_robust_fit.py`, failures injected into `GP.fit`, next to a MATLAB-rule column):**
  - Noise lower bound per try: `-7.908, -6.908, -4.908, -1.908, 2.092`. MATLAB keeps −7.908.
  - After 4 failures the returned noise is `2.092`.
  - The GP handed back keeps all 69 rows and its original bounds, while the fit saw 69 → 55 rows.
  - On this data set the fit also failed twice without injection, and landed at the nudged bound −4.908.
- **Reach (`v04`):** 13 of the 21 refits with a failure ended with the noise exactly at the nudged bound.
- **Consequence (`v05_nudge_effect.py`, MATLAB's bound rule against as is, same seeds):**

  | Run | log10(f as is / f MATLAB rule) |
  |---|---|
  | rosen3, seed 41 | +0.31 |
  | rosen3, seed 42 | +0.07 |
  | rosen3, seed 43 | +0.23 |
  | ell4, seed 40 | +0.48 |
  | ell4, seed 41 | −0.25 |
  | ell4, seed 42 | +0.63 |

  As is was worse in 5 of 6.
- **Dating:** Python since `c7c88ab` (the code moved in `9037851`); MATLAB `6c93629` (2017). Never agreed.
- **Disposition: fix.** Raise the bound from the entry bound by `nudge[1]`, as MATLAB does. It moves default runs, so it needs a population comparison at default. Fix it with I-F4: once the `ValueError` at the fifth failure is gone, the `UnboundLocalError` at the tenth becomes the exit.

### I-F4 = C-F3: when every try fails, there is no exit −1
- **Lines:** `res` is bound only at `gaussian_process_train.py:607`, and 702 returns it. MATLAB `gpHyperOptimize.m:61-62, 197-209` keeps the evaluated start of each run and returns the best one with exit flag −1.
- **Check (`v03`):**
  - 5 injected failures: `RAISED ValueError: Lower bound above upper bound for the hyperparameter(s) noise_log_scale.`
  - With `noise_nudge=[0,0]` and 10 failures: `RAISED UnboundLocalError: cannot access local variable 'res'`.
  - Reading the code confirms that, without either error, the last unfitted start would be returned.
- **Reach:** not at default (at most 2 failures in a row seen).
- **Dating:** `c7c88ab`; MATLAB 2017. Never agreed.
- **Disposition: fix, with I-F3.** Default runs do not move: gate with the fingerprint and a failure-injection test.

### I-F5 = C-F8: the χ² bounds lack the factor 2
- **Lines:** `bads.py:2491`, `gammaincinv(v/2, y)`. MATLAB `gppredcheck.m:20`, `2*gammaincinv(x,v/2)`, with MATLAB's argument order.
- **Check (`v06_refit_test.py`):**
  - For v = 2 the port's bounds are `[5.0e-07, 14.51]`; the true ones are `[1.0e-06, 29.02]`.
  - With z = (3, 3.2), Python says unreliable and my transcription of MATLAB does not.
  - A refit at n = 2 is impossible, since it needs more than 2D evaluations since the last one, so only the poll-stop flag changes.
- **Dating:** `c7c88ab`; MATLAB 2017. Never agreed.
- **Disposition: fix** (use `chi2.ppf`), in the same change as I-F6 and I-F7a. It moves poll-stop decisions, so it needs a population comparison at default.

### I-F6 = C-F7: one stat is read as none, and the periodic refit comes at period + 1
- **Lines:** `bads.py:2447-2451` (`gp_iter_idx[-1] == 0`) and `2513` (`gp_iter_idx >= refit_period`, where `gp_iter_idx` = n − 1). MATLAB `gppredcheck.m:9, 19` and `bads.m:1243` (`gpstats.last >= refitperiod`).
- **Check (`v06`):**
  - n = 1 with z = 0.1: Python `(False, True)`, MATLAB `(False, False)`.
  - n = 10 at D = 2: Python `(False, False)`, MATLAB `(True, False)`.
- **Consequence:**
  - `v07_latent_sd.py`: each 200-evaluation run had 7-14 periodic refits one evaluation late.
  - `v20_n1_pollstop.py`: 1 poll-stop decision changed in 15 checks at n = 1 over 4 runs (ell2: a good poll stopped at once).
- **Dating:** the current form is from `9037851`; MATLAB 2017. Never agreed.
- **Disposition: fix** (n = the number of stats), with I-F5. It needs a population comparison at default.

### I-F7a = C-F6 = B5-R3 first clause: the calibration z-scores use the latent SD
- **Lines:**
  - `bads.py:1784-1786` and `2233-2235` store `fs` from `acq_fcn_lcb.py:48-49`, which is `predict` without noise.
  - `bads.py:2481-2482` replaces every SD that `np.isclose(0.0, ·)` calls zero, which means up to 1e-8 by its default tolerance, with 1e-6.
  - MATLAB `bads.m:629, 906` store `ys`, the SD of the observation (`acqLCB.m:29`).
- **Check (`v07b_sd_ratio.py`):**
  - Predictive/latent SD ratio at the 10/50/90 percentiles: rosen3 `[1. 1.02 6.72]`, ell3 `[1. 3.97 7.63]`, noisy sphere `[2.52 8.29 10.96]`.
  - The latent SD was at or below 1e-8 in 29 of 105 stats (rosen3) and 6 of 99 (ell3).
  - The comparison report's figures match mine. The internal report's median of about 1000-3000 is not reproduced: ratios that large arise only at the replaced points.
- **Refit verdicts (`v07`):**
  - Under MATLAB's rule, the verdict changed with the SD in 7, 12, 2 and 20 checks per run.
  - Python refits before the period, with n = 0 being the first refit on empty stats, shared with MATLAB: `[0, 9, 9, 9]`, `[0, 9, 9]`, `[0, 9]` and `[0]`.
  - Every one at n = 9 had a replaced near-zero latent SD. So the n ≥ 3 test does fire, contrary to I-F7's "never fired", but only through the 1e-6 replacement.
- **Dating:** the latent SD since `c7c88ab`; the replacement since `cdc2e0f` (2023-06-10); MATLAB 2017. Never agreed.
- **Disposition: fix.** Store the predictive SD, √(fs² + noise); that also removes the need for the replacement. It needs a population comparison at default, together with I-F5, I-F6 and C-F9.

### I-F7b: the n ≥ 3 test ignores scale and location
- **What is true:** Shapiro-Wilk tests composite normality. MATLAB's `swtest.m` does the same ("normal with unspecified mean and variance"), and both call it a calibration check.
- **Classification:** shared with MATLAB, so a design question.
- **The decision:** keep MATLAB's normality test, or add a scale test such as χ² on Σz² for every n. Only the second changes behaviour, and it would need a population comparison.

### C-F9: `scipy.stats.shapiro` in place of `swtest.m`, which switches to Shapiro-Francia when the kurtosis is above 3
- **Lines:** `bads.py:2505-2506`; MATLAB `swtest.m:130-160`.
- **Check (`v06`, my transcription):**
  - t(1.5) samples, n = 10/20/40: disagreements 38, 72 and 72 of 2000.
  - N(0,1) with an outlier of 8: disagreements 31 (n = 20) and 449 (n = 40).
  - `v21_sw_vs_sf_runs.py`, on the z-scores the port actually tests: 4 of 84 (rosen3) and 1 of 79 (ell3) verdicts disagree, and 0 of 125 in the noisy run. `swtest` rejects fewer; the rejections come from the zero-SD outliers of I-F7a.
- **Dating:** `c7c88ab`; `swtest.m` 2017. Never agreed.
- **Disposition: fix** (port `swtest`), or decide to keep scipy and record it on the sheet. It needs a population comparison at default, with I-F7a.

### I-F8 = C-F1: `reset_gp` is cleared only at the end of a poll
- **Lines:** `bads.py:1933` sets it, `2418` sets it to `is_poll_moved`, `1114` is the only `False`, and `1677` and `2150` read it. MATLAB `bads.m:524-526, 707, 826-829, 1047`: `post = []` asks for one rebuild, and the rebuild fills `post`.
- **Check (`v08_reset_gp.py`, as is against clearing the flag after each rebuild):**

  | Run | Rebuilds, as is | Rebuilds, cleared | log10(f as is / f cleared) |
  |---|---|---|---|
  | rosen3, seed 60 | 51 | 25 | +0.00 |
  | rosen3, seed 61 | 82 | 44 | −0.50 |
  | ell4, seed 60 | 104 | 57 | +1.80 |
  | ell4, seed 61 | 140 | 53 | −1.41 |

  Results move in both directions. In deterministic polls, the polled points reach the GP only through these rebuilds.
- **Tests:** `test_gp_update_failures.py:589` sets `reset_gp = False` in `_probe`, as the comparison report says.
- **Dating:** `c7c88ab`; MATLAB 2017. Never agreed.
- **Disposition: fix toward MATLAB** (clear the flag in the rebuild), gated by a population comparison at default. If the comparison favours the current behaviour, keep it and record it on the sheet as deliberate.

### I-F9 = C-F11: the `&` slip
- **Lines:** `bads.py:1789-1793`; `search_hedge.py:48, 62` (a count over the whole run). MATLAB `bads.m:633` uses the round's `searchcount`.
- **Check (`v09_and_slip.py`):**
  - `size=3 count=50: expression -> True; 'and' form -> False`.
  - The expression is True whenever size > 0, which always holds where it is evaluated.
- **Why inert:** it only adds the point at the last search of a round. The poll rebuilds at `poll_count == 0` anyway, and `update_hedge` reads the GP only when `hedge_gamma == 0`; the default is 0.125.
- **Dating:** `c7c88ab`; MATLAB 2017. Never agreed.
- **Disposition: fix.** Use `self.optim_state["search_count"] < self.options["search_n_try"]`, not `and` with the hedge's count. It should move nothing: gate with the fingerprint.

### I-F10: with `poll_training=False` the poll records refits it then cancels
- **Lines:** `bads.py:2128-2136` with `2517-2518`. MATLAB `bads.m:822-823` has the same order: `IsRefitTime` resets `lastfitgp` and `gpstats`, and then `PollTraining` cancels the refit.
- **Check (`v10_poll_training_off.py`, ell3 seed 70):** refits recorded 9, performed 3. With the option on: 10 and 10.
- **Where I disagree:** the report's "port discrepancy" is wrong. The comparison report's Q2 is right that the two sides are the same.
- **Dating:** Python `c7c88ab`; MATLAB 2017. The two agree.
- **Disposition: decide the design.** Keep it (MATLAB-faithful) and document it, or apply the override before the refit is recorded. A change needs a population comparison with `poll_training=False`.

### I-F11 = C-F10 = B5-R1 last sentence: a rescued refit keeps the rejected refit's geometry
- **Lines:**
  - `gaussian_process_train.py:455-506` write the geometry from the refit.
  - `520-523` put back the priors and the previous hyperparameters.
  - `540-541` clear the markers.
  - MATLAB `gpupdate.m:279-354` keeps the new hyperparameters with their geometry and sets `post = []`.
- **Check (`v11_retry_geometry.py`, first posterior update after a refit made to fail):**
  - The GP holds length scales `[50. 24.7957 7.0131]`, while `temporary_data len_scale` is `[50. 15.1506 4.2335]`.
  - `poll_scale` from the held hyperparameters would be `[2.4316 1.2058 0.3411]`, against `[3.3905 1.0274 0.2871]` stored.
  - The effective radius would be 1.0448, against 1.0718 stored.
  - Markers `[]`, exit flag −2.
- **Why inert today:**
  - `v12_retry_reach.py`: no exit flag −2 in 393 rebuilds (44 refits) over 4 default runs.
  - Without points removed in `_robust_gp_fit_`, the fit already computed this very posterior successfully, so the update cannot fail there.
- **Dating:** `c7c88ab`; `685da15` added the restore and kept the geometry; MATLAB 2017. Never agreed.
- **Disposition: fix.** Whichever way B5-R1 is decided, the geometry should come from the hyperparameters the GP holds. Default runs do not move: gate with the fingerprint, and add a geometry check to `test_local_fit_recovered_failure_is_unchanged`.

### B5-R1: the retry with the previous hyperparameters on the new training set
- **What is true:**
  - The retry exists; MATLAB has none and clears the posterior instead (`gpupdate.m:347-350`).
  - Without a refit it cannot succeed. `hyp_gp` is `old_hyp_gp` (line 508), and `set_hyperparameters` goes to the same `update(hyp=...)` on the same data (gpyreg `gaussian_process.py:873-935`). The computation is deterministic, so a failure repeats; the comment at 525-528 says so.
  - With a refit the retry can succeed, which is where I-F11 applies.
  - KD-B5-2 leaves the retry open.
- **Disposition: decide the design.** Options:
  - skip the retry when there was no refit (it is pure cost);
  - keep the retry with a refit, but recompute the geometry (I-F11);
  - or do as MATLAB does: keep the refit's hyperparameters and geometry and set the rebuild markers.

  It is not reached at default: gate with the fingerprint and an injection test.

### I-F12a: at D = 1, `len_scale` is 1
- **What is true:**
  - `gaussian_process_train.py:455` tests whether there is more than one length scale.
  - MATLAB sets `ncovlen = ncov − 2`, which is 1 at D = 1 for `covRQard` (`gpdefBads.m:51`), and `gpupdate.m:285-292` then sets `lenscale = 1` too. Python matches MATLAB.
  - `v13_oned.py`: fitted length scale 0.492, `len_scale` 1.0.
- **Disposition: decide the design.** Keep MATLAB's behaviour, or use the length scale at D = 1. Only the second moves D = 1 runs, and it would need a population comparison of D = 1 problems.

### I-F12b: the multi-sample sum
- **What is true:** `len_scale += len_scale + exp(...)` gives `[8. 11.]` for two samples, where MATLAB's weighted sum gives `[3. 4.]` (`v13`). There is always one sample (KD-B5-4).
- **Disposition: fix in passing.** Gate with the fingerprint.

### I-F13: the `init_N` schedule divides by zero, or extrapolates
- **Lines:** `gaussian_process_train.py:1048-1058`. Python only.
- **Check (`v14_small_budget.py`, `eff_starting_points` = 5):**
  - D = 2 and 3 with `max_fun_evals` = 5: `RAISED ValueError: cannot convert float NaN to integer`. It also raised at D = 2 with 2, and at D = 3 with 2 and 3.
  - `max_fun_evals` = 3 or 4 gives `init_N` 413 or 968, above the 128 of `gp_train_n_init`.
- **Dating:** the current formula is from `8ff10f5`.
- **Disposition: fix.** Guard the denominator and clip x to [0, 1]. Gate with the fingerprint and a small-budget test.

### I-F14: the retry loop of the initial fit has no cap
- **Lines:** `gaussian_process_train.py:162-210`.
- **Check (`v15_initial_fit_loop.py`):** with a fit that always raises, `init_and_train_gp kept retrying: stopped by the check after 50 calls`.
- **Dating:** `9037851`.
- **Disposition: fix.** Cap the retries, then raise or fall back. Gate with the fingerprint and an injection test.

### C-F5: a refit starts from gpyreg's design, not from MATLAB's local runs
- **What is true:**
  - `_get_gp_training_options` sets `init_N` from 128 falling to 8, and `opts_N` to 1, or 2 on a second fit.
  - gpyreg `gaussian_process.py:1885-1920` optimizes from the best point of the design. With `opts_N = 2`, it overwrites the second start with a low-noise design point.
  - MATLAB `gpupdate.m:371-408` and `gpHyperOptimize.m:47-75` run one local optimization from each of hyp0(1) and, with a second fit, hyp0(2).
  - The defaults were tuned deliberately in `8ff10f5` ("changed configuration of init_n"), but no record names this a departure from MATLAB; KD-B1-4 leaves it open.
- **Check (`v16_start_points.py`, 22 refits, the same data fitted both ways):**
  - Of the 14 refits where both fits succeeded: the same optimum in 8; the design better in 3, by up to 17886.56 in the negative log posterior; the previous-hyperparameter start better in 3, by up to 0.56.
  - In the other 8, one fit or both raised `LinAlgError`.
- **Disposition: decide the design.** Keep the design and record it on the sheet, or move to MATLAB's scheme. The second needs a population comparison at default.

### C-F12: details of the retry
- **(a) The percentile rule** (`gaussian_process_train.py:635`, linear interpolation; MATLAB `prctile1`). Points removed, from `v17_percentile.py`:

  | n | linear (the port) | `prctile1` (MATLAB) |
  |---|---|---|
  | 5 | 1 | 0 |
  | 10 | 1 | 0 |
  | 49 | 3 | 2 |
  | 50 | 3 | 2 |

  NumPy's `"hazen"` method equals `prctile1` at every size I checked. Removal starts at the second failure, which happened in 10 of 53 deterministic refits.
- **(b) The minimum-points stop:** MATLAB stops when fewer than `nvars` points remain (`gpHyperOptimize.m:71`); the port has no such stop. Not reached at default.
- **(c) The exit flag:** 0 after retries (MATLAB gives 1). It is read only by the display (`bads.py:2407`), so it is inert.
- **Dating:** `c7c88ab`; MATLAB 2017.
- **Disposition: fix (a) and (b)** (`method="hazen"`; stop below D points). (a) moves default runs, so it needs a population comparison at default, together with I-F2/I-F3.

### B5-R2: the retry reads `tmp_gp` after a failed fit
- **What holds:**
  - The retry nudges bounds read from `tmp_gp.get_bounds()` (line 671).
  - With `use_slice_sampler=True`, it samples on `tmp_gp`'s data (lines 651-653, 741).
  - The v1.3.2 source (`git show v1.3.2:…`) assigns `self.X`, `self.y` and `self.s2` before the fit can fail, sets the filled bounds, and puts back nothing but `df`. v1.3.3 puts the GP back as it was before the call.
  - `v18_tmp_gp_reads.py` (1.3.3): `fit 3: given X rows 64, tmp_gp holds 69` and `slice sampler reads the data of tmp_gp: 69 rows`.
  - At default, no bound is NaN and the sampler is off, so nothing depends on the gpyreg version.
- **What does not hold:** the negquad clause. The bounds that `_gp_hyp` leaves unset are filled by the initial fit and kept (`v18`: `gp_mean_fun=negquad: NaN bounds held by the GP at a refit: lower 0, upper 0`).
- **The remedy:** the fit already receives X, Y and s2 explicitly. What would need the retry's data is the slice sampler.
- **Disposition: fix** when the slice-sampler path is touched: sample on a GP that holds the retry's X, Y and s2, together with new item N1. Gate with the fingerprint and a test with `use_slice_sampler=True`.

### B5-R3, second clause: `_re_evaluate_history_` uses each stored GP's geometry
- **It holds at `95da7f1`.** `bads.py:2733` passes `gps[i]` to `local_gp_fitting`, whose `get_grid_search_neighbors` reads that GP's `len_scale` and `effective_radius` (`gaussian_process_train.py:1100, 1111`). MATLAB `bads.m:1384-1395` swaps only `hyp` into the current `gpstruct`.
- **In my one noisy run (`v19_reeval_neighbours.py`):** no neighbour set differed (all 150 points fall inside every set at `n_train_max` = 200), although the length scales did (`[37.939 14.516 7.391]` stored against `[50. 21.991 10.759]` current).
- **Disposition:** pass to the owner of `_re_evaluate_history_` (B2).

## 3. New items met while verifying (unverified beyond what is stated)

- **N1:** with `use_slice_sampler=True`, `_robust_gp_fit_` raises at the third consecutive fit failure: `ValueError: The initial starting point X0 is outside the bounds.` The start rises by k while the nudged bound rises by k(k+1)/2 (start − LB = −0.493 at try 3). The error comes from `SliceSampler`, and only `LinAlgError` is caught, so the run stops. Reproduced once (`v18`).
- **N2 (B1):** the half-bounds check (`bads.py:524-533`) uses `np.any` across all variables. It therefore refuses a problem that mixes fully bounded and fully unbounded variables, which its own message allows. My mixed case in `v01` was refused. MATLAB's `boundscheck.m` and `transvars.m` have no such refusal (read, not run).
- **N3 (B2):** `_re_evaluate_history_` does not copy `gps[i]` before calling `local_gp_fitting`, which changes GPs in place. So the stored GPs in `iteration_history` have their training sets and priors rewritten. Seen by reading only.
- **N4:** in `v02`, the prior of `mean_const`, N(14.98, 2.92), lay outside its bounds [0.17, 9.61]: the prior is re-centred at each rebuild while the bounds stay fixed from initialization. This is probably the matter named in #70's title; not investigated.
- **N5 (seen again, B2/B7):** with `max_fun_evals` of 3 or 4, a run makes 6 evaluations (`v14`).
- **Not verified:** the comparison report's Q3 claim that at level 2 the high-noise check uses a base noise of 1.0 even when the user sets `noise_size`.
