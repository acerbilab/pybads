<!-- Report of the B5 comparison reviewer (the GP training set and refit policy, MATLAB comparison track), wave 1 of the port review, reading PyBADS at 95da7f1 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave1/B5_comparison/. Nothing in it is verified. -->

# B5 comparison review: GP training set and refit policy

## 1. Coverage

**Read completely.**
- Python (`/home/user/pybads-review` @ `95da7f1`), `pybads/bads/gaussian_process_train.py`: `init_and_train_gp`, `local_gp_fitting`, `_robust_gp_fit_`, `_get_random_samples_from_priors_`, `_get_gp_training_options`, `get_grid_search_neighbors`, `_get_fevals_data`, `_estimate_noise_`, `add_and_update_gp`. I skimmed `_gp_hyp` and `_get_samples_from_slice_sampler_`, which are B6.
- Python, `pybads/bads/bads.py`: `_search_step_` (1643-1962), `_poll_step_` (2030-2270; the rest skimmed), `_save_gp_stats_`, `_is_gp_refit_time_`, `_is_poll_stop_`, `_record_gp_refit_`, `_update_incumbent_`, the noisy option changes in `_init_optimization_` (1120-1165), and the main loop (1255-1420).
- Python, other files: the relevant `.ini` options, `FunctionLogger.__call__`/`_record`, `IterationHistory`, `udist`, `acq_fcn_lcb`, `ESSearchHedge.update_hedge`.
- gpyreg: `GP.fit` (1496-2010) and `f_min_fill`.
- MATLAB (`/home/user/bads` @ `74919c0`), in full: `private/gpupdate.m`, `utils/gpHyperOptimize.m` (its `gpminimize` only skimmed), `gppredcheck.m`, `swtest.m`, `gppriorrnd.m`, `prctile1.m`, `gpdef/gpdefBads.m`, `udist.m`, `acq/acqLCB.m`.
- MATLAB, in part: `bads.m` 140-300, 430-1050, 1195-1260, 1285-1293 and 1376-1414; `gppred.m`; the header of `mygp.m`; the noise test in `evalinitmesh.m`; `funlogger.m` 95-130; `acqPortfolio.m` 'upd'.
- Git histories: MATLAB since 2022-02-11 (`bfe8e22`, `8515191`, `75ec49f`, `c4d2b9a`, `d4fead5`, `a21f2ee`, `019f0b4`, `74919c0`), and `git log -L`/`-S` on every Python line cited below.

**Skimmed or not reached.** `update_posterior.m` (skimmed: KD-B5-1 settles its absence) and `minimizebnd.m` (only its error handling). I did not read `gpHyperSVGD.m` or `gpHyperSample.m` (KD-B5-4). I did not run the test suite.

**Checks run.** Scripts and outputs are in `/tmp/claude-0/-home-user-pybads/4f4200d5-5a60-5de0-9b26-94c4f1e930f9/scratchpad/wave1/B5_comparison/`.
- `matlab_ref.py` transcribes `swtest.m`, `gppredcheck.m` and `IsRefitTime`. On platykurtic samples its SW branch matches `scipy.stats.shapiro` to |Δlog p| ≤ 1.3e-8.
- The checks are `s1`–`s9`. The runs are default, seeded, at most 200 evaluations, one thread each: rosen3, ellip3, ellip4, and a noisy sphere at level 1.

**Seen outside the slice, not reviewed** (passed to their owners):
- B3: `update_hedge` computes `np.exp(-0.5*gamma_z**2/np.sqrt(2*np.pi))`, where `acqPortfolio.m:64` has `exp(-0.5*gammaz.^2)/sqrt(2*pi)`.
- B4: the poll's `p_less` multiplies the `D+1` largest PoIs (`bads.py:2212-2214`), where `bads.m:869` takes `nvars`.
- Harmless:
  - `_estimate_noise_` sorts by descending y (PyVBMC's convention), but its output `sn2hpd` is never read.
  - `len_scale += len_scale + exp(...)` (`gaussian_process_train.py:458`) doubles the running sum. It cannot be reached, because there is always one hyperparameter row.
  - The `res` returned by `_robust_gp_fit_` is gpyreg's sampling result, which is always `None` and unused.

## 2. Answers to the first questions

### Q1. The training set
Yes, point for point, apart from the order of tied rows.
- **Distance:** `udist` uses `gp.temporary_data["len_scale"]`, the ARD length scales from the last refit (1 before the first), as `gpupdate.m:91` uses `gpstruct.lenscale`.
- **Count:** `ntrain = min(n_train_max, #(d ≤ radius²))`, then the max with `n_train_min` and `n_train_max − buffer_ntrain`, then the min with the number of stored points, as in `gpupdate.m:97-104`.
- **Radius:** `gp_radius·effective_radius`, the latter updated only at a refit.
- **Noisy runs:** the changes (`n_train_max ≥ 200`, `n_train_min ×2`) match `bads.m:433-434`.
- **Sort:** `np.argsort` is not stable; MATLAB's `sort` is.

A transcription of `'nearest'` run on the same data at every rebuild (`s6`) gave the same set in 541 of 541 rebuilds. The row order differed in 182, from ties such as ± poll pairs and level-1 repeats. There was no tie at the cut-off.

Other points:
- MATLAB's cache is circular at 1e4 rows; the Python logger grows without limit. This matters only beyond 9999 evaluations.
- Neither side records the noise-test evaluation.
- At level 1, `get_grid_search_neighbors` hands the GP an `s2` of NaNs, since `S` is never filled. This is harmless: at level 1 the noise function does not read `s2`.
- The rebuilds happen more often than in MATLAB (F1). The set chosen at each rebuild is right; how often it is re-chosen is not.

### Q2. When to refit
The pieces that match:
- The minimum time `lastfitgp < func_count − min_refit_time`, the condition `func_count > D`, and the refit period `max(10, 2D)` below 200 evaluations, `5D` above.
- A refit leaves the same state: `lastfitgp` set, the stats reset, the unreliable flag cleared. The `poll_training` override after the reset is also the same.
- A NaN z-score means unreliable on both sides.

The pieces that do not match:
- **The statistic** divides by the latent SD, where MATLAB divides by the predictive SD of the observation (latent plus noise). A zero SD is replaced by 1e-6 (F6).
- **The count** is off by one: one stat is read as none, and the periodic refit comes at n = period + 1 (F7).
- **Test for n < 3:** the χ² quantiles are half the true ones (F8).
- **Test for n ≥ 3:** `scipy.stats.shapiro` replaces `swtest.m`, which uses Shapiro–Francia on leptokurtic samples (F9).
- **Threshold:** `p < α` against `α ≥ p`, which differ only at equality.

On the same trajectory, the refit verdict differed from MATLAB's rule in 11 of 123 checks (rosen3, seed 0, `s4b`). Seven were periodic refits one evaluation late; four were extra refits caused by a zero latent SD.

### Q3. How a fit is attempted
It does not do what `gpHyperOptimize.m` does.
- **Starting points** come from gpyreg's design: 128 draws from the priors at first, falling to 8, plus the given rows. The best one is optimized, or the best two with the second replaced by gpyreg's low-noise heuristic. MATLAB optimizes from the previous hyperparameters, and from the second-fit point when there is one (F5).
- **The second-fit and retry points** are drawn with a broken prior sampler (F4).
- **The noise nudge:** the starting point is nudged as in MATLAB (+1, +2, …). The lower bound, which MATLAB leaves unchanged (`nudge(2) = 0`), is raised by the cumulative starting-point nudge: +1, +3, +6, +10. The run stops with a `ValueError` at the fifth consecutive failure (F2).
- **Point removal** starts at the second failure, as in MATLAB: the higher point of the closest pair, plus the points above the 95th percentile. The percentile method differs, and MATLAB's stop below `nvars` points is missing (F12).
- **After the last failure** MATLAB returns the best starting hyperparameters with exit flag −1. The port raises `UnboundLocalError` (F3).

At level 2, the high-noise check uses a base noise of 1.0 even when the user sets `noise_size`; MATLAB keeps the user's value. This is minor and needs a non-default input.

## 3. Findings

### F1. `reset_gp` is never cleared by a rebuild, so after any improving search or moved poll the GP is rebuilt at every search and every poll step until a poll ends
- Location: `pybads/bads/bads.py:1674-1679`, `2147-2152` (the rebuild conditions), `1933` (set to True), `2418` (`= is_poll_moved`), `1114` (the only False); MATLAB: `bads.m:523-525`, `707`, `826-829`, `1049` (rebuild while `post` is empty; a rebuild fills `post`).
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, at all levels.
- History: the Python lines date from `c7c88ab` (2022-06-02) and are unchanged in substance. The MATLAB lines date from 2017 (`6c93629`…`e424df6`); `d4fead5` only renamed `gpTrainingSet` to `gpupdate`. The two never agreed.
- What it does:
  - MATLAB's `gpstruct.post = []` has the next step rebuild once, and the rebuild clears it. The Python stand-in is only reset at the end of the poll (`2418`).
  - Once a search improves the incumbent (success or incremental), every later search of the round and every poll step rebuilds with `local_gp_fitting`. If the poll moved, the whole next round does too.
  - In a deterministic poll, MATLAB does not add poll points to the GP (`bads.m:908` is noisy-only), so its poll GP lacks them. Here the rebuild at each poll step includes the points already polled.
  - The LCB ordering, the PoI stopping rule and the target therefore change. Whether the poll GP holds the poll points depends on whether the last search improved or the last poll moved.
- Consequence:
  - Poll rebuilds were 45/65/68 for rosen3 (seeds 0-2); 33/45/54 of them are ones MATLAB's rule would not make. With the flag cleared after each rebuild they were 16/22/16. Figures for ellip4 were 73/81/67 against 15/17/16.
  - Results change: rosen3 final f was 7.3e-6/5.7e-6/9.3e-7 as is, against 2.0e-5/2.9e-5/2.1e-4 with the flag cleared; ellip4 2.2e-6/1.1e-8/6.4e-7 against 4.9e-7/4.5e-9/1.6e-6.
  - Each extra rebuild costs a posterior computation. The two noisy runs ended at the same estimate either way.
- Reproduction: `s4_runs.py` (a subclass with a `reset_gp` property, counting rebuilds that MATLAB's rule would not make; mode `clear_reset`).
- Test adequacy: no test would catch it. `test_gp_update_failures._probe` sets `self.reset_gp = False` before probing (line 588), so `test_poll_leaves_unmarked_gp` and `test_search_leaves_unmarked_gp` never see the state that triggers it.

### F2. A failed fit raises the noise lower bound by the cumulative starting-point nudge; MATLAB raises it by `nudge(2)`, which is 0
- Location: `pybads/bads/gaussian_process_train.py:661-675`; MATLAB: `utils/gpHyperOptimize.m:6-7`, `160-167`.
- Category: formula
- Proposed classification: port discrepancy (a defect)
- Confidence: high
- Reached at default options: no. It is reached only when gpyreg's `fit` raises `LinAlgError` at a refit, which is rare because gpyreg already retries with jitter internally. It is reached at every level.
- History: the Python dates from `c7c88ab` (2022-06-02). MATLAB's lines 160-167 predate 2022; the 2022 commits `bfe8e22` and `8515191` touched only the removal of `s`. Never agreed.
- What it does:
  - MATLAB, per failure: `noiseNudge += nudge(1)`, `lb(end-1) += nudge(2)`, `theta0(end-1) += noiseNudge`. With `NoiseNudge=[1 0]` the start moves by +1, +2, … and the bound stays put.
  - Python adds the cumulative `noise_nudge` to the bound of `tmp_gp`, whose bounds already hold the earlier raises. The bound therefore rises by k(k+1)/2.
- Consequence:
  - `s2_robust_fit.py` and `s2_big.py` (a 49-point set) measured the noise lower bound at −7.91, −6.91, −4.91, −1.91, then +2.09: a noise SD of at least 8 on data of unit scale.
  - At the fifth consecutive failure, `set_bounds` raises `ValueError: Lower bound above upper bound for noise_log_scale`. `_robust_gp_fit_` does not catch it, so the run stops.
  - A retry that succeeds after three or four failures returns a GP forced to high noise, for example a returned noise of +2.09.
- Reproduction: `s2_robust_fit.py` (outputs above).
- Test adequacy: no test injects fit failures in `_robust_gp_fit_`.

### F3. When every try fails, `_robust_gp_fit_` raises `UnboundLocalError` instead of returning the best starting hyperparameters with exit flag −1
- Location: `pybads/bads/gaussian_process_train.py:605-610`, `702`; MATLAB: `utils/gpHyperOptimize.m:61-62`, `179-209`.
- Category: control flow
- Proposed classification: port discrepancy (a defect)
- Confidence: high
- Reached at default options: no. It needs 10 consecutive fit failures, and at the default nudge F2 stops the run first. It is reached with `noise_nudge=[0,0]`, or a training set large enough to survive nine removals.
- History: Python `c7c88ab`; MATLAB unchanged since before 2022. Never agreed.
- What it does:
  - `res` is bound only by a successful `fit`, so after 10 failures the return raises. Even without the crash, the port would return the last retry start, which was never fitted (a nudged random draw); `local_gp_fitting` would then compute the posterior and geometry from it.
  - MATLAB keeps `theta(:,iRun) = theta0` and `fval = f0` for each run, returns `argmin fval` (normally the previous hyperparameters, clipped), and sets exitflag −1.
- Consequence: the run stops with an exception where MATLAB carries on.
- Reproduction: `s2c_all_fail_big.py` gives "after 10 tries: RAISED UnboundLocalError: cannot access local variable 'res'". On the 5-point set of `s2b` the repeated removal empties the set first ("argmin of an empty sequence").
- Test adequacy: none.

### F4. `_get_random_samples_from_priors_` draws each log-hyperparameter from N(exp(μ), exp(σ)) instead of N(μ, σ)
- Location: `pybads/bads/gaussian_process_train.py:712-724`. The function belongs to B6 per the map; the policy uses it at `402` (second fit) and `655` (retries). MATLAB: `utils/gppriorrnd.m:66-78`, called at `gpupdate.m:390` and `gpHyperOptimize.m:150`.
- Category: formula
- Proposed classification: port discrepancy (a defect)
- Confidence: high
- Reached at default options: only on a second fit (a high-noise or low-mean fit, both rare in the runs; 0 of 45 refits in `s7`) and on retries after failures. It is reached at every level. The initial fit also uses it after a failure (B6).
- History: Python `9037851` (2022-09-22); `gppriorrnd.m` last changed in 2018. Never agreed.
- What it does: for keys containing "log", both the mean and the SD are exponentiated, and the draw is stored as the log value. MATLAB draws the log-hyperparameter from its Gaussian prior.
- Consequence: the second-fit and retry starting points come from the wrong distribution. For example, the noise prior N(−3.45, 1) gives draws of mean 0.03 and SD 2.66; the log length scale prior N(−0.12, 0.83) gives mean 0.94 and SD 2.30. The effect is diluted, because each such point enters gpyreg's design as one row among 8-128 (F5).
- Reproduction: `s3_prior_samples.py` (4000 draws against `gp.hyper_priors`; table above).
- Test adequacy: none.

### F5. The starting points of a refit are gpyreg's design (128 falling to 8 prior draws, plus the given rows), not MATLAB's local runs from the previous hyperparameters and the second-fit point
- Location: `pybads/bads/gaussian_process_train.py:1047-1071` (`init_N`, `opts_N`), `439-449`, `607-609`; gpyreg `gaussian_process.py:1885-1920` (design, and the second start replaced by a low-noise design point), `f_min_fill.py:92-240`. MATLAB: `private/gpupdate.m:371-408`, `utils/gpHyperOptimize.m:47-75`, `197-200`.
- Category: control flow
- Proposed classification: possibly intentional. The `gp_train_*` options come from PyVBMC, and KD-B1-4 leaves their effects open for B5/B6.
- Confidence: high on the difference, medium on its size.
- Reached at default options: yes, at every refit and every level.
- History: the Python policy dates from `c7c88ab`; the defaults (`gp_train_n_init=128`, `_final=8`, `"rand"`) from `8ff10f5` (2022-11-04). MATLAB's `gpfit` changed after 2022 only in comments (`d4fead5`). Never agreed.
- What it does:
  - MATLAB runs one local optimization from the previous hyperparameters, and a second from the second-fit point if there is one, then keeps the lower. Its first fit starts from `gpdefBads`' initial values; the port's first refit starts from the fit made at initialization (KD-B6-1).
  - The port evaluates init_N prior draws (Student-t with df ≤ 3) plus the given rows, and optimizes only the best one. With `opts_N = 2`, gpyreg overwrites the second start with the best of the lowest-noise 20% of the design, so the explicit second-fit point is optimized only if it ranks first.
- Consequence:
  - Over 45 refits (`s7`, rosen3 and ellip3, seeds 0-1), the previous hyperparameters ranked first in the design only 22 times; at the first two refits they ranked 51-74th.
  - Against a MATLAB-like fit on the same data (`init_N=0`, starts = given rows), the final negative log posterior was often identical. It was sometimes lower (by up to 789) and sometimes higher (by up to 1.1).
  - The MATLAB-like fit raised `LinAlgError` in 12 refits where the design avoided it. The design also costs up to 123 extra objective evaluations per refit.
- Reproduction: `s7_starts.py`.
- Test adequacy: `test_get_gp_training_options_opts_N` checks only `opts_N == 1`, which mirrors the implementation.

### F6. The calibration statistic divides by the latent SD, and a zero SD is replaced by 1e-6; MATLAB divides by the predictive SD of the observation
- Location: `pybads/bads/bads.py:1784-1786`, `2233-2235` (they store `fs` from `acq_fcn_lcb`, which is `gp.predict` without noise, `acq_fcn_lcb.py:48-54`), `2480-2484` (the 1e-6 replacement); MATLAB: `bads.m:628-629`, `905-906` (store `ys`), `acq/acqLCB.m:26-29` (`ys = sqrt(ys2)`, the predictive output variance, `mygp.m:17`), `utils/gppredcheck.m:11`.
- Category: formula
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, at all levels, most strongly at levels 1 and 2.
- History: the latent `fs` is passed since `c7c88ab`; the 1e-6 replacement was added in `cdc2e0f` (2023-06-10). MATLAB's lines date from 2017. Never agreed.
- What it does: z = (y − μ)/σ_f instead of (y − μ)/√(σ_f² + σ_n²). The z-scores are inflated by a factor that varies from point to point. When σ_f is exactly 0 (tiny mesh near the data), z ≈ 1e6·(y − μ); MATLAB's σ can never be 0, since the noise is at least e^(log TolFun − 1).
- Consequence:
  - The ratio of predictive to latent SD at the saved points had a median of 1.0-6.8 in deterministic runs and 8.6 in noisy runs.
  - Along the same trajectories, the MATLAB verdict with latent SD against predictive SD differed in 8-20 of roughly 130-160 checks per run.
  - In rosen3 seed 0, four refits (at n = 9, where the period is 10) came only from zero latent SDs replaced by 1e-6. Shapiro then gave p ≈ 3e-7 < 1e-6; with MATLAB's SD those z-scores would have been finite and moderate.
- Reproduction: `s4_runs.py` (the ratio, and the verdicts with each SD); `s4c_n9.py` (the zero-SD refits).
- Test adequacy: none.

### F7. The stats count is off by one: one stat is read as "no stats" (unreliable), and the periodic refit waits for n = period + 1
- Location: `pybads/bads/bads.py:2447-2458`, `2510-2515` (`gp_iter_idx` is n − 1); MATLAB: `utils/gppredcheck.m:4-9`, `19-28` (n = 1 goes to the χ² test); `bads.m:1242-1244` (`gpstats.last >= refitperiod`).
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, at all levels.
- History: in `c7c88ab` the key `'gp_iter'` did not exist, so the stats always read as empty. The current form comes from `9037851` (2022-09-22). MATLAB's lines date from 2017. Never agreed.
- What it does: MATLAB tests n = 1 with a χ² test on one degree of freedom, and refits periodically when n ≥ refitperiod. The port treats n = 1 as no stats, and refits periodically only when n − 1 ≥ refitperiod.
- Consequence:
  - The periodic refits come one evaluation late: 7 of the 11 disagreeing refit verdicts in `s4b`.
  - The n = 1 case, which follows every refit, flags the GP unreliable; this changed no poll-stop decision in 6 runs (`s9`). It cannot trigger a refit at default, because `min_refit_time` has not passed.
- Reproduction: `s1_refit_check.py` cases A and C (n = 10 for D = 2: MATLAB refits, Python does not); `s4b_disagreements.py`.
- Test adequacy: none.

### F8. The χ² bounds for n < 3 lack the factor 2
- Location: `pybads/bads/bads.py:2491-2493`; MATLAB: `utils/gppredcheck.m:20-22` (`2*gammaincinv(x,v/2)`).
- Category: formula
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes. In practice only n = 2 reaches it (n = 1 goes elsewhere, F7), and there it changes only the unreliable flag used by the poll's stopping rule.
- History: Python `c7c88ab`; MATLAB 2017. Never agreed.
- What it does: `gammaincinv(v/2, y)` is half of the χ² quantile. At α = 1e-6 and v = 2 the bounds are [5e-7, 14.5] instead of [1e-6, 29.0].
- Consequence: with z = (3, 3) the port flags unreliable and MATLAB does not. No poll decision changed in the runs.
- Reproduction: `s1_refit_check.py` case B.
- Test adequacy: none.

### F9. `swtest.m` is replaced by `scipy.stats.shapiro`, which lacks MATLAB's Shapiro–Francia branch for leptokurtic samples
- Location: `pybads/bads/bads.py:2505-2506`; MATLAB: `utils/swtest.m:130-160`, `272`, via `gppredcheck.m:30`.
- Category: formula
- Proposed classification: port discrepancy (a substitution not on the sheet)
- Confidence: medium
- Reached at default options: yes, whenever n ≥ 3.
- History: Python `c7c88ab`; `swtest.m` last changed in 2017/2018. Never agreed.
- What it does: MATLAB tests with Shapiro–Francia when the biased kurtosis exceeds 3, and Shapiro–Wilk otherwise. scipy always uses Shapiro–Wilk (AS R94).
- Consequence:
  - Decisions at α = 1e-6 differ on heavy-tailed z-scores. With one z = 8 outlier among N(0,1) values, MATLAB rejects 3 of 2000 samples at n = 20 and scipy 40; at n = 40, MATLAB 890 and scipy 1323. For t(1.5), up to 73 of 2000 decisions differ.
  - In the default runs, no disagreement arose from this cause alone.
- Reproduction: `s1_refit_check.py` case D.
- Test adequacy: none.

### F10. After a refit whose posterior fails, the retry with the previous hyperparameters keeps the refit's geometry and clears the markers
- Location: `pybads/bads/gaussian_process_train.py:453-506` (geometry written), `513-523` (retry), `540-541` (markers popped); MATLAB: `private/gpupdate.m:279-354` (new hyperparameters and geometry kept together, `post = []`).
- Category: state/caching
- Proposed classification: port discrepancy. This bears on the part of KD-B5-2 that the sheet leaves open (the retry).
- Confidence: high
- Reached at default options: no. It needs a `LinAlgError` in `gp.update(hyp=hyp_gp)` after a successful refit. It is reached at every level.
- History: the retry dates from `c7c88ab`; `685da15` (2026-09-25) added the restore on a double failure, but not a rollback of the geometry. MATLAB's lines date from 2017 (the Debug print was added in `d4fead5`).
- What it does: the GP holds the previous hyperparameters, while `len_scale`, `poll_scale` and `effective_radius` are those of the refit it discarded. `lastfitgp` has already been set, so nothing refits until the next refit time.
- Consequence: the training-set distance, the radius, the poll basis and the ES-ell search all use a geometry that the GP's hyperparameters do not have. Measured: len_scale [50, 4.83] against [50, 9.40] from the held hyperparameters; poll_scale [3.22, 0.31] against [2.31, 0.43].
- Reproduction: `s8_retry_geometry.py`.
- Test adequacy: `test_local_fit_recovered_failure_is_unchanged` does not check `temporary_data`'s geometry against the hyperparameters.

### F11. The search adds its point at every search: the condition `size > 0 & count < n_try` is always true
- Location: `pybads/bads/bads.py:1789-1793`; `pybads/search/search_hedge.py:48`, `62` (the count accumulates over the run); MATLAB: `bads.m:633`.
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, at all levels.
- History: Python `c7c88ab`; MATLAB 2017. Never agreed.
- What it does:
  - The expression parses as `size > (0 & count) < n_try`, which is always true. Even with `and`, the hedge's count is cumulative, not the round's `search_count`.
  - MATLAB skips the add after the last search of a round.
- Consequence: an extra full posterior recomputation per round. On failure it also sets `needs_rebuild`. Decisions are not affected, since the next step rebuilds anyway and `update_hedge` reads the GP only when `hedge_gamma = 0`.
- Reproduction: inspection. The expression evaluates to `True` for any `count`.
- Test adequacy: none.

### F12. Retry details of `_robust_gp_fit_`: percentile method, no minimum-point stop, exit flag
- Location: `pybads/bads/gaussian_process_train.py:634-636`, `604-616`, `685-700`; MATLAB: `utils/gpHyperOptimize.m:137` (`prctile1`, hazen), `71` (`if size(y,1) < nvars; break`), `102`, `202-209` (success per run).
- Category: formula / control flow
- Proposed classification: port discrepancy
- Confidence: medium
- Reached at default options: no; only after fit failures, at every level.
- History: Python `c7c88ab`. The MATLAB lines are unchanged except the `s` removal (`bfe8e22`/`8515191`, May 2022, before `c7c88ab`).
- What it does:
  - `np.percentile(Y, 95)` is linear interpolation. It removes the maximum for n ≤ 10, where `prctile1` removes none, and 3 points against 2 at n ≈ 50.
  - There is no stop when fewer than D points remain; a fit on a single point was observed.
  - A fit that succeeds after retries reports 0; MATLAB reports 1. Only the display reads it.
- Consequence: a smaller training set on the retry path. Minor.
- Reproduction: the percentile table from the `prctile1` check (n = 5, 10, 49 and 50); `s2_robust_fit.py` (n train 5 → 3 → 2 → 1).
- Test adequacy: none.

## 4. Test adequacy notes
- `test_gp_update_failures._probe` sets `self.reset_gp = False` and stubs `_is_gp_refit_time_` before counting rebuilds. The "leaves unmarked" tests therefore confirm the implementation's intent and hide F1.
- No test compares `_is_gp_refit_time_` with `gppredcheck`/`IsRefitTime` (F6-F9), or checks what is stored in `gp_stats`.
- No test injects `fit` failures into `_robust_gp_fit_`, so the noise nudge, removal and exhaustion paths (F2, F3, F12) are not covered. `test_initial_fit_recovers_from_failure` covers only `init_and_train_gp`.
- `test_get_gp_training_options_*` assert the sampler name and `opts_N == 1`, which mirrors the code; the starting points are not tested (F5).
- `test_bads_optimization.py` checks final errors with tolerances from seed sweeps (KD-T-1). None of F1-F12 would fail it: F1 moves results in both directions within those tolerances.
