<!-- Report of the B5 internal reviewer (the GP training set and refit policy, internal-correctness track), wave 1 of the port review, reading PyBADS at 95da7f1 in ../pybads-review and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave1/B5_internal/. Nothing in it is verified. -->

# B5 internal review: GP training set and refit policy

## 1. Coverage

**Read in full:**
- `pybads/bads/gaussian_process_train.py`, the whole file.
- In `pybads/bads/bads.py`: `_init_mesh_`, `_init_optimization_`, the main loop of `optimize` (1214-1641), `_search_step_`, `_poll_step_`, `_save_gp_stats_`, `_is_gp_refit_time_`, `_is_poll_stop_`, `_record_gp_refit_` and `_re_evaluate_history_`. From `_init_optim_state_`, the bounds, noise-level and GP-setting parts (600-700, 790-830, 880-960).
- `acquisition_functions/acq_fcn_lcb.py`, `search/grid_functions.py` (`udist`), `poll/poll_mads_2n.py`, ES-ell in `search/es_search.py` (287-297), the `count` attribute in `search/search_hedge.py`.
- `FunctionLogger` (`__init__`, `__call__`, `_record`), `IterationHistory`, `stats/get_hpd.py`.
- Every option of the slice in both `.ini` files.
- In gpyreg v1.3.3: `GP.update`, `GP.fit`, `f_min_fill`, `GaussianNoise.compute`, and the `RationalQuadraticARD` formula.

**Skimmed:** `_gp_hyp` (B6), `_bounds_check_`, and the tests named in part 4.

**Not reached:**
- The BADS paper: arxiv.org and neurips.cc are blocked by the egress proxy. I judged against docstrings, option descriptions and the mathematics instead.
- MATLAB: not opened, as the internal track requires.

**Scripts:** all under `/tmp/claude-0/-home-user-pybads/4f4200d5-5a60-5de0-9b26-94c4f1e930f9/scratchpad/wave1/B5_internal/`. Each prints `pybads.__file__` (the review worktree) and `gpyreg.__file__` (the v1.3.3 clone). All runs are seeded, at most 200 evaluations, one BLAS thread.

**Two observations outside the slice:**
- With tiny budgets (`max_fun_evals` of 3 or 4 at D=2), a run makes 6 evaluations, which is more than `max_fun_evals` (initial design, B2/B7).
- `_estimate_noise_` has two problems, but no effect, because nothing reads the `sn2hpd` it returns:
  - it ranks the "HPD" points in descending order of y, the PyVBMC convention, which picks the worst points of a minimization;
  - `get_hpd` ranks the other way (ascending).

## 2. Answers to the first questions

### Q1. The training set

- **Distance.** `udist` gives the squared Euclidean distance in u space, each coordinate divided by `gp.temporary_data["len_scale"]`. Those are the ARD length scales of the last refit: 1 before the first refit, and always 1.0 at D = 1 (F12). The training set is chosen with the previous GP's length scales and then refitted on.
- **Radius.** It is `gp_radius` (3) × the effective radius √(α(e^{1/α}−1)), compared with the squared distance. For gpyreg's RQ kernel (1+r²/(2α))^{−α}, this is the distance at which the correlation falls to e^{−1}, divided by the SE distance for the same fall (√2). It tends to 1 as α→∞, so it is consistent. At small α the radius is effectively infinite and the cap binds.
- **Number of points.** It is computed as:
  - min(`n_train_max`, number of points within the radius);
  - then max(`n_train_min`, `n_train_max − buffer_ntrain`, that);
  - then min(that, number of rows in the logger).
  - Defaults: floor max(50, 10D−50) and cap 50+10D. Under uncertainty: floor max(100, `n_train_max`−100) and cap ≥ 200.
  - This matches the three option descriptions ("minimum 200 under uncertainty", "doubled under uncertainty", "Max number … removed if too far").
- **Order.** Ascending distance, nearest first. `np.argsort` defaults to an unstable sort, so exact ties at the cutoff are broken arbitrarily. Exact ties are plausible for mirrored poll points on a dyadic mesh, but they matter only when the cutoff binds.
- **Data.** All recorded rows. At level 2, the variances S². At a user-declared level 1, S = 1 is stored, but the noise model ignores s2 at that level (`parameters[1] = 0`), so there is no effect.

### Q2. When to refit

- **Condition.** A refit needs all three of:
  - `func_count − lastfitgp > min_refit_time`. This is strict, so at least 2D+1 evaluations, one more than a literal reading of "Minimum fcn evals before refitting";
  - `func_count > D`;
  - (number of stats − 1 ≥ `refit_period`) or "uncalibrated".
- **Period.** `refit_period` is max(10, 2D) below 200 evaluations and 5D afterwards. So at D = 1 the period drops from 10 to 5 after 200 evaluations, while for D ≥ 3 it grows; whether that is intended cannot be judged internally.
  - One stat is saved per evaluation, so the default interval is 11 evaluations for D ≤ 5 and 2D+1 for D ≥ 6.
- **Statistic.** z = (y − μ)/s, where μ and s are the GP's *latent* mean and SD at the point before it was evaluated (`acq_fcn_lcb` calls `predict` without noise). An s of about 0 is replaced by 1e−6.
- **Test, by number of stats since the last refit:**
  - 0 or 1: flagged "uncalibrated" (F6);
  - 2: a two-sided χ² test on Σz², whose quantiles lack a factor 2 (F5);
  - 3 or more: Shapiro–Wilk at `normalpha_level` = 1e−6.
- **Is it a correct test?** It is the "normality test" the option names. It is not the calibration test the docstring names, because Shapiro–Wilk is blind to scale and bias (F7). In practice it never fired: 0 flags in about 300 checks with n ≥ 3. So refits are periodic.
- **State.** A positive decision calls `_record_gp_refit_` (sets `lastfitgp`, clears `gp_stats`) *before* the refit is attempted, and returns `do_gp_calibration = False`. If the caller then does not refit (F10), the state is wrong. `do_gp_calibration` also drives `_is_poll_stop_`.

### Q3. How a fit is attempted

- **Starting points of a refit.**
  - `hyp0` is the GP's current hyperparameters (one row).
  - With a second fit (`double_refit`, high noise, or mean below min y), a second row is added: 0.5·(prior draw + current). The noise is reset to N(−2,1) if the noise was high, the mean to median(y) if the mean was low, and the row is clipped to the bounds.
  - `gp.fit` then builds an `init_N`-point design: the `hyp0` rows plus draws through the priors truncated to the bounds (`"rand"`). `init_N` follows a cubic from 128 down to 8 over the first min(`max_fun_evals`, `n_train_max`) − (initial points) evaluations.
  - The design is ranked by the negative log posterior. L-BFGS-B then starts from the best point, or with a second fit from the best plus the best of the lowest-noise 20% of `X0[2:]`.
  - So the carefully built second-fit start becomes an optimizer start only if it ranks first, or wins that low-noise pick. The prior draw inside it is broken (F2).
- **After a `LinAlgError`** (up to 10 tries):
  - the start becomes 0.5·(broken prior draw + previous);
  - the noise start moves by +k after k failures, while the noise lower bound grows by k(k+1)/2 (F3);
  - from the 2nd failure on, the worse point of the closest pair (Euclidean in u) and all points above the 95th percentile of y are removed, but only from the data given to the fit. The GP handed back keeps every point and its original bounds, and its posterior is computed on the full set with hyperparameters fitted on the reduced set.
  - `remove_points_after_tries = 1` thus means removal starts at the 2nd failure. That is ambiguous against "after this number of failures".
- **After the last failure.** The 5th consecutive failure raises `ValueError`. The documented return of −1 is unreachable (F4).
- **Measured frequency of fit failures:**
  - deterministic runs: 17% of refits (26/149) and 38% (26/69) in two sets of runs had at least one failure, and at most 2 in a row;
  - noisy runs: 0 of 43 refits;
  - after a failure, the nudged noise bound binds: 50% of refits after 1 failure, 89% after 2.
- **Initial fit** (`init_and_train_gp`): retries without a cap (F14).

## 3. Findings

### F1. `optim_state["plb"]`/`["pub"]` are swapped, so the poll scale is −2 in every unbounded dimension and ES-ell ignores the GP length scales
- Location: pybads/bads/bads.py:650-651 (`optim_state["pub"] = plausible_lower`, `["plb"] = plausible_upper`); pybads/bads/gaussian_process_train.py:477-489 (reads); pybads/search/es_search.py:289-293 (use). MATLAB: private/gpupdate.m, poll-scale block (not opened, internal track).
- Category: cross-module
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, with infinite bounds. `lower_bounds=None`/`upper_bounds=None` gives ±inf (bads.py:229-233). Half bounds are refused, so only fully unbounded variables are affected.
- History: both sets of lines date from c7c88ab (2022-06-02); MATLAB history not checked.
- What happens: an infinite `ub` is replaced by `optim_state["pub"]`, which is the lower plausible bound (−1). An infinite `lb` is replaced by the upper plausible bound (+1). The cap `(ub_bounded − lb_bounded)/scale` becomes −2, so `poll_scale = min(max(ll, search_mesh), −2) = −2` whatever the length scales. The intended cap is `pub − plb = +2`.
- Why only ES-ell is affected: the poll cancels `poll_scale` (poll_mads_2n.py:37 divides by it, bads.py:2082-2086 multiplies by it), so the poll is unaffected. ES-ell uses `poll_scale` as its ellipse, which becomes isotropic when every variable is unbounded.
- Consequence: the ES-ell search loses its GP-informed shape on unbounded problems.
- Reproduction (ran):
  - `check_pollscale.py`: unbounded 3-D ellipsoid, fitted `len_scale` [70.9, 7.3, 100], `poll_scale` [−2, −2, −2] at every refit. A bounded run gives [1.93, 0.16, 3.24].
  - `unbounded_effect.py`: 4-D ellipsoid (scales 1–100), 150 evaluations, seeds 0-4, as is versus the two names swapped back. log10 fval as is: −5.01, −4.54, −4.64, −4.56, −4.95. Swapped back: −6.36, −4.99, −5.90, −5.97, −5.35. All 5 seeds are worse as is, by 0.45 to 1.4 orders of magnitude.
- Test adequacy: none. All optimization tests use finite bounds (`get_test_opt_conf`: ±100). AGENTS.md records the swap as a fact, not as a defect.

### F2. `_get_random_samples_from_priors_` draws log hyperparameters from N(e^μ, e^σ)
- Location: pybads/bads/gaussian_process_train.py:705-726 (717-719). MATLAB: utils/gppriorrnd.m (not opened).
- Category: formula
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes. It runs at every fit failure in a refit (17-38% of deterministic refits), at every second fit, and at the first two failures of the initial fit.
- History: the `exp` lines come from 9037851 (2022-09-22).
- What happens: for keys containing "log" it sets `mean = exp(mu)` and `sd = exp(sigma)`, draws a normal, and stores the draw as the log value. The priors are Gaussian on the log values, so the draw should be N(μ, σ) directly. No reading makes exponentiating both the mean and the SD correct.
- Consequence: the retry start (0.5·(draw + previous)) and the second-fit start land at near-random corners of the box after clipping. For example, the log output scale lands at its upper bound of 20.1. That start is only one member of gpyreg's design, which draws its own points correctly, so the fit usually recovers. The losses are:
  - after a failure, the warm start from the previous hyperparameters;
  - most of the intended low-noise or median-mean second start.
- Reproduction (ran, `check_prior_sampler.py`, 4000 draws):
  - log output scale: prior N(11.1, 2); draws have mean 67355 and SD 7.3;
  - log length scale: prior N(−2.04, 3.40); draws have mean ≈0.2 and SD ≈30;
  - log noise: prior N(−3.45, 1); draws have mean 0.00 and SD 2.73;
  - `mean_const` (no log) is correct.
- Test adequacy: no test checks the distribution of the draws.

### F3. The noise nudge raises the lower bound twice over and never reads the nudge's second component
- Location: pybads/bads/gaussian_process_train.py:661-683. MATLAB: utils/gpHyperOptimize.m (not opened).
- Category: formula / defaults
- Proposed classification: unsure (the comparison track should check what the second component of `NoiseNudge` does in `gpHyperOptimize.m`)
- Confidence: high on what the code does; medium that it is unintended.
- Reached at default options: yes, after any fit failure in a refit.
- History: c7c88ab.
- What happens:
  - `noise_nudge` accumulates 1, 2, 3, …, and that cumulative value is added to a bound that was *already* nudged. The bound is therefore lb₀ + k(k+1)/2 after k failures (lb₀ + 1, + 3, + 6, …).
  - The start point is rebuilt fresh each try, plus k.
  - For a one-element option, the code builds `nudge[1] = 0.5·nudge[0]` and then never reads it. The default is `[1, 0]`.
- Consequence: the raised bound binds. The fitted log noise sits exactly at the nudged bound in 50% of refits after one failure and 89% after two, which is lb₀ + 3 (noise SD about 20× the floor), in about 6-10% of deterministic refits. Points removed after the 2nd failure add to this.
- Reproduction (ran):
  - `check_robust_fit.py`: bound per try −7.91, −6.91, −4.91, −1.91, 2.09; training points 59, 59, 55, 51, 47.
  - `nudge_effect.py`: the fractions above.
- Test adequacy: no test injects failures into the refit's `gp.fit`.

### F4. `_robust_gp_fit_` cannot return its documented "all attempts failed" result
- Location: pybads/bads/gaussian_process_train.py:603-702 (`res` is assigned only at 607; returned at 702). Nothing around the call at 439 catches the error.
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no. It needs 5 consecutive failures in one refit, and at most 2 were observed in 218 refits.
- History: c7c88ab/9037851.
- What happens: at the 5th failure, the bound nudge of F3 (lb₀ + 15) exceeds the noise upper bound of 5, and `set_bounds` raises `ValueError` ("Lower bound above upper bound"), which stops the run. With wider bounds, 10 failures would raise `UnboundLocalError` on `res`. Even without either error, the function would return the last *unfitted*, nudged candidate as `hyp_gp`, and `local_gp_fitting` would compute the geometry and the posterior from it. The intended result is exit −1 with the previous hyperparameters.
- Consequence: a run crash in a pathological case.
- Reproduction (ran, `check_robust_fit.py`, `GP.fit` patched to always raise): `ValueError` after 5 calls.
- Test adequacy: none.

### F5. The χ² quantiles of the two-point calibration test lack the factor 2
- Location: pybads/bads/bads.py:2491-2493. MATLAB: utils/gppredcheck.m (not opened).
- Category: formula
- Proposed classification: unsure (MATLAB may share it)
- Confidence: high
- Reached at default options: yes, whenever exactly 2 stats exist since the last refit.
- History: c7c88ab.
- What happens: `gammaincinv(v/2, p)` is half of `chi2.ppf(p, v)`. With α = 1e−6 and n = 2, the bounds are [5e−7, 14.5] instead of [1e−6, 29.0]. The false-alarm rate for a calibrated GP is 7.1e−4 instead of 1e−6.
- Consequence: small in practice. At n = 2, a refit is impossible by default (it needs n > 2D), so only the poll-stop rule is affected. The latent-SD z-scores of F7 are so inflated that the correct test also flags most of these checks: 9 of the 15 n = 2 checks in one run, against 10 for the code.
- Reproduction (ran, `check_refit_time.py`): z = (3, 3.2) has χ² p = 6.6e−5 > α, and the code flags it.
- Test adequacy: none.

### F6. The branch commented "empty stats" also catches one stat, so one z-score always means "uncalibrated"
- Location: pybads/bads/bads.py:2446-2454 (`gp_iter_idx[-1] == 0`). The `n < 3` branch at 2490 therefore never sees n = 1.
- Category: control flow (off by one)
- Proposed classification: unsure (it looks like a 0-based translation of a count equal to 0)
- Confidence: medium
- Reached at default options: yes. Measured on ell4: 15 of 15 calls with n = 1 returned `True`.
- History: 9037851.
- What happens: `iter_gp` holds 0, 1, 2, …, so a last entry of 0 means one entry, not none.
- Consequence:
  - In the poll right after a refit, a certain good first poll ends the poll at once, instead of stopping only when p_less > 1 − `tol_poi`. Such events were rare in the runs measured (0 of 6 good-poll checks happened at n = 1).
  - With `min_refit_time` below 1, it would trigger refits.
- Reproduction (ran, `check_refit_time.py`): with 1 stat of |z| = 0.1, `do_gp_calibration` is `True`; with 2 stats, `False`.
- Test adequacy: none.

### F7. The calibration test is scale- and bias-blind, and its z-scores use the latent SD instead of the predictive SD
- Location: pybads/bads/bads.py:2473-2506; the stats come from 1784-1786 and 2233-2235 (`fs` of `acq_fcn_lcb.py:52-53`, latent).
- Category: formula
- Proposed classification: possibly intentional (the option does say "normality test")
- Confidence: high on the behavior.
- Reached at default options: yes, at every check with n ≥ 3.
- History: 9037851.
- What the docstring promises: "checks the calibration of the GP prediction".
- What the code does:
  - Shapiro–Wilk ignores location and scale: z-scores 1000× too large, or biased by +50 SD, pass with p = 0.996;
  - the observations y are standardized by the latent SD only.
- Measured |z_latent|/|z_predictive|: median ≈1000-3000 on deterministic Rosenbrock, ≈7 on a noisy sphere.
- Consequence:
  - the n ≥ 3 test never fired (0 of ≈300 checks), so refits are purely periodic;
  - the n = 2 χ² test, which is scale-sensitive, fires most of the time.
- Reproduction (ran): `check_refit_time.py`, `zscore_scale.py`, `n2_checks.py`.
- Test adequacy: none.

### F8. `reset_gp` is set on a move and never cleared after the rebuild it requests
- Location: pybads/bads/bads.py:1114, 1677, 1933, 2150, 2418; the poll calls `add_and_update_gp` only at level > 0 (2237).
- Category: state/caching
- Proposed classification: unsure
- Confidence: high on the behavior; medium that it is unintended.
- Reached at default options: yes.
- History: c7c88ab.
- What happens: after a move or an improving search, *every* later search step and *every* poll iteration rebuilds the GP, until the poll's end resets the flag.
- Measured (ell4, seed 0, 200 evaluations): 24 search rebuilds and 63 poll-iteration rebuilds were triggered only by the flag, with the incumbent unchanged since the previous rebuild. 18 search rebuilds were real moves.
- Consequence in deterministic runs: polled points enter the GP within a poll only through these rebuilds. So the LCB order and the probability-of-improvement (PoI) stop see the points already polled after a move, and not after a failed round. There is also extra cost.
- Reproduction (ran): `count_rebuild_reasons.py`, `count_stale_reset.py`.
- Test adequacy: none.

### F9. The `&` slip in the search's add condition, and the counter it would use once fixed
- Location: pybads/bads/bads.py:1789-1793; pybads/search/search_hedge.py:48, 62.
- Category: control flow
- Proposed classification: port discrepancy (latent)
- Confidence: high
- Reached at default options: yes, but the condition is always true.
- History: c7c88ab.
- What happens: Python reads the condition as `size > (0 & count) < n_try`, which reduces to `size > 0 and 0 < n_try`, always true. AGENTS.md already names this slip. New here: `search_es_hedge.count` counts searches over the whole run (−1, then +1 per call). Written with `and`, the condition would stop adding search points after the first `search_n_try` searches of the run. The per-round counter is `optim_state["search_count"]`.
- Consequence: none now, since the last search of a round is also added and the poll rebuilds anyway. A plain `&` → `and` fix would break the search.
- Test adequacy: none.

### F10. With `poll_training=False`, the poll records refits that it then cancels
- Location: pybads/bads/bads.py:2128-2136 together with 2517-2518.
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no; only with `poll_training=False`.
- History: 685da15/9037851.
- What happens: `_is_gp_refit_time_` has already reset `gp_stats` and set `lastfitgp` before the poll forces `refit_flag = False`.
- Consequence: fewer real refits. The search's refits are delayed by the recorded time.
- Reproduction (ran, `poll_training_off.py`, ell4): 15 refits with the option on; with it off, 14 recorded but only 7 performed.
- Test adequacy: `test_poll_refit_gives_way_to_poll_training` checks only the refit flag.

### F11. After a failed rebuild is rescued with the previous hyperparameters, the geometry belongs to the rejected refit
- Location: pybads/bads/gaussian_process_train.py:453-506, 512-541.
- Category: state/caching
- Proposed classification: unsure (this is the unsettled part of KD-B5-2)
- Confidence: high
- Reached at default options: rarely; it needs the posterior update after a refit to fail.
- What happens: the GP ends with the entry hyperparameters, but `len_scale`, `poll_scale` and `effective_radius` come from the refit that was rejected. The markers are cleared and the refit counts as done, so no refit is forced.
- Reproduction (ran, `retry_geometry.py`): the GP holds log ℓ [4.200, 2.797, 0.474], while `len_scale` is [4.200, 2.828, 0.434].
- Test adequacy: `test_local_fit_recovered_failure_is_unchanged` does not compare the geometry with the hyperparameters.

### F12. At D = 1 the length scale is always 1.0; with several samples it would be summed wrongly
- Location: pybads/bads/gaussian_process_train.py:455-463.
- Category: formula
- Proposed classification: unsure
- Confidence: high
- Reached at default options: yes for D = 1. The multi-sample sum is never reached (there is always one sample).
- History: c7c88ab.
- What happens:
  - The test `len(log_lengthscale) > 1` is presumably meant to tell ARD from isotropic kernels, but it also fails at D = 1.
  - The multi-sample line, `len_scale += len_scale + exp(...)`, doubles the total and has no weights.
- Consequence: small. Distances for the training set and the search statistics are not in length-scale units at D = 1; `ntrain` came out 60 where it might be 50.
- Reproduction (ran, `oned_lenscale.py`): fitted ℓ = 1.62, `len_scale` = 1.0.
- Test adequacy: none.

### F13. The `init_N` schedule divides 0 by 0, or extrapolates, when the budget does not exceed the initial design
- Location: pybads/bads/gaussian_process_train.py:1048-1058.
- Category: formula
- Proposed classification: unsure
- Confidence: high
- Reached at default options: no; only small `max_fun_evals`.
- What happens:
  - When min(`max_fun_evals`, `n_train_max`) equals the number of initial points, 0/0 gives NaN, and `round` raises `ValueError`.
  - When it is below that number, x < 0 and the cubic gives `init_N` of 413 or 968, above `gp_train_n_init` = 128.
  - The lambda also mixes `x_` and `x`, which is harmless.
- Reproduction (ran, `small_budget.py`): the crash at D=2 with `max_fun_evals` 2 or 5, and at D=3 with 2, 3 or 5.
- Test adequacy: none.

### F14. The retry loop of the initial fit has no cap
- Location: pybads/bads/gaussian_process_train.py:162-210.
- Category: control flow
- Proposed classification: port discrepancy (MATLAB fits no GP at this point, per KD C5)
- Confidence: high
- Reached at default options: no.
- What happens: failures 1 and 2 retry from a broken prior draw (F2), failure 3 from zeros, and failure 4 onwards from further draws forever. A fit that keeps failing hangs the run.
- Test adequacy: `test_initial_fit_recovers_from_failure` injects only one failure.

## 4. Test adequacy notes

- **`test_search.py::test_grid_search_neighbors`** uses 3 points, finite bounds and `len_scale` 1, and checks only the order. It exercises neither the radius nor the buffer, nor the ARD length scales, nor ties.
- **`test_gaussian_process_train.py::test_get_gp_training_options_*`** assert only `sampler` and `opts_N`. The `init_N` schedule is untested (F13).
- **`test_gp_update_failures.py`** injects failures into `update`, `set_hyperparameters` and the initial fit, never into the refit's `gp.fit`. So `_robust_gp_fit_`'s retries, nudges and removal of points (F3, F4) are uncovered, as is the prior sampler (F2).
- **No test** covers `_is_gp_refit_time_` or `_save_gp_stats_` (F5-F7), the rebuild triggers (F8), `poll_scale` with infinite bounds (F1), or D = 1 geometry (F12).
- **The optimization tests** all use finite bounds (±100). They pass at seed-swept tolerances, which cannot detect the F1 or F2 losses.
