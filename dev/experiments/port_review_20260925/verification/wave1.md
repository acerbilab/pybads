# Wave 1 ledger: B5 and B6

The verified findings of wave 1 of the port review
([plan](../../../plans/port-correctness-review.md), "Wave 1 pickup"): slice
B5, the GP training set and refit policy, and slice B6, the GP model and its
gpyreg objects, each read on both tracks. The reports are
`reviews/B5_internal.md`, `reviews/B5_comparison.md`,
`reviews/B6_internal.md` and `reviews/B6_comparison.md`. Each slice was
verified by a fresh Opus agent that had not written either report, with
checks of its own: `wave1_B5_verifier.md` and `wave1_B6_verifier.md`. The
verifiers were also given the open rows of the survey's candidate table of
their slice that neither report covered, as B5-R1 to B5-R3 and B6-R1. Every
agent read PyBADS at the freeze `95da7f1`, MATLAB BADS at `74919c0` and
gpyreg v1.3.3 (`98ab5a4`), in a cloud session (Linux, Python 3.11.15,
NumPy 2.4.6, SciPy 1.17.1). The scripts and their outputs are under
`scripts/wave1/<slice>_<track>/` and `scripts/wave1/<slice>_verifier/`,
formatted by the pre-commit hooks after they ran. The reports cite them at
the sandbox's scratch paths.

Lines are at `95da7f1`, MATLAB lines at `74919c0`. The fix pass of wave 0 on
`dev-port-review` touches none of the code these rows describe (in
`gaussian_process_train.py` it changed only comments and the unreachable
substitution of non-finite values, W0-6), so every row holds there as well,
at shifted line numbers. "B5-I F1" is finding F1 of the B5 internal report, "B5-C"
the B5 comparison report, and likewise "B6-I" and "B6-C". "Survey" names the
row of the candidate table of `dev/results/2026-09-23-codebase-survey.md`
that describes the same behavior. The dispositions are the verifiers'
recommendations, proposals until the PI rules; the gate is the one of
`AGENTS.md`, "Numerical gates", that a fix would need. "Default run" says
whether a run at default options reaches the code, and at which uncertainty
level (0 deterministic, 1 noise inferred, 2 `specify_target_noise`).

## B5: training set, refit policy, fit attempts

| Id | Source | What | Classification | Default run | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|
| W1-1 | B5-I F1 | `optim_state["pub"]` holds the transformed lower plausible bound and `["plb"]` the upper one (`bads.py:650-651`); `gaussian_process_train.py:477-489`, their only reader, then caps `poll_scale` at `-2` in every unbounded variable, so the ES-ell search is isotropic. The poll divides by `poll_scale` and multiplies back, so it is unaffected. MATLAB stores them right (`setupvars.m:67-68`, `gpupdate.m:301-308`) | confirmed port discrepancy | yes, but only on problems unbounded in every variable (a mix is refused, survey row `bads.py:500-507`); all levels | never agreed (`c7c88ab`; MATLAB 2017) | `bads.py:626-627` | fix: swap the two assignments, and correct `AGENTS.md`, which states the swap as a fact. The effect on results is not established: worse as is on 3 of 6 seeds in each of two targets, where the reviewer saw 5 of 5 | fingerprint (bounded runs unchanged); population with fully unbounded configurations |
| W1-2 | B5-I F8, B5-C F1 | `reset_gp`, set by a move or an improving search, is cleared only at the end of a poll (`bads.py:1114`, `1933`, `2418`), so every later search and every poll step rebuilds the GP until then; MATLAB's `post = []` asks for one rebuild, which fills `post` (`bads.m:523-525`, `707`, `826-829`). In deterministic polls the polled points reach the GP only through these rebuilds | confirmed port discrepancy | yes, all levels | never agreed (`c7c88ab`; MATLAB 2017) | — | fix toward MATLAB: clear the flag at the rebuild. Clearing it halves the rebuilds and moves results in both directions (4 runs); if the population favours the current behavior, keep it and put it on the sheet | population at default |
| W1-3 | B5-I F7(a), B5-C F6, B5-R3 (first clause) | the calibration statistics store the latent SD of the prediction (`bads.py:1784-1786`, `2233-2235`, from `acq_fcn_lcb.py:48-49`), and every SD that `np.isclose` calls zero (below 1e-8) is replaced by 1e-6 (`2481-2482`); MATLAB stores the SD of the observation, noise included (`bads.m:629`, `906`; `acqLCB.m:29`) | confirmed port discrepancy | yes, all levels | never agreed (latent SD `c7c88ab`, the replacement `cdc2e0f`, 2023-06-10; MATLAB 2017) | `_save_gp_stats_` calls (first clause) | fix: store `sqrt(fs**2 + sn**2)`, which removes the need for the replacement; with W1-4 to W1-6. The early refits at n = 9 all come from a replaced SD, so the n ≥ 3 test does fire (B5-I's "never fired" is not reproduced). B6-I's counterfactual at level 1 changed no refit in 6 runs | population at default (W1-3 to W1-6 together) |
| W1-4 | B5-I F6, B5-C F7 | the stats count is off by one: `gp_iter_idx[-1] == 0` reads one stat as none (unreliable, `bads.py:2447-2451`), and the periodic refit waits for n = period + 1 (`2513`); MATLAB tests n = 1 with χ² and refits at n ≥ `refitperiod` (`gppredcheck.m:9`, `19`; `bads.m:1243`) | confirmed port discrepancy | yes, all levels: 7-14 periodic refits one evaluation late per 200-evaluation run; 1 poll-stop decision changed in 15 checks at n = 1 | never agreed (the current form `9037851`; MATLAB 2017) | — | fix, with W1-3 | as W1-3 |
| W1-5 | B5-I F5, B5-C F8 | the χ² bounds of the test at n < 3 are `gammaincinv(v/2, p)`, half the quantiles (`bads.py:2491-2493`); MATLAB has `2*gammaincinv(...)` (`gppredcheck.m:20-22`). At n = 2 a refit is impossible by default, so only the poll's unreliable flag changes | confirmed port discrepancy | yes, all levels (n = 2) | never agreed (`c7c88ab`; MATLAB 2017) | — | fix (`chi2.ppf`), with W1-3 | as W1-3 |
| W1-6 | B5-C F9 | `scipy.stats.shapiro` replaces `swtest.m`, which switches to Shapiro-Francia when the kurtosis exceeds 3 (`bads.py:2505-2506`; `swtest.m:130-160`). On the z-scores of default runs 4 of 84 and 1 of 79 verdicts differ at level 0, 0 of 125 at level 1, all from the replaced SDs of W1-3 | confirmed port discrepancy (a substitution not on the sheet) | yes, level 0 | never agreed (`c7c88ab`; MATLAB 2017) | — | PI: port `swtest`, or keep scipy and put it on the sheet. W1-3 removes the outliers that make the two differ | as W1-3 |
| W1-7 | B5-I F7(b) | the n ≥ 3 test is a normality test, blind to the scale and location of the z-scores; MATLAB's is the same | design question, shared with MATLAB | yes (n ≥ 3) | the two agree | — | PI: keep MATLAB's test, or add a scale test (χ² on Σz²) at every n | none if kept; population if changed |
| W1-8 | B5-I F10 | with `poll_training=False`, `_is_gp_refit_time_` records a refit (`lastfitgp`, stats reset) that the poll then cancels (`bads.py:2128-2136`, `2517-2518`): 9 recorded, 3 performed. MATLAB has the same order (`bads.m:822-823`) | confirmed shared defect | no (`poll_training=False`) | the two agree (`c7c88ab`; MATLAB 2017) | — | PI: keep it, faithful to MATLAB, and document it; or apply the override before the refit is recorded | population with `poll_training=False` if changed |
| W1-9 | B5-I F9, B5-C F11 | the condition for adding the search point to the GP, `u_search.size > 0 & self.search_es_hedge.count < n_try` (`bads.py:1789-1793`), is always true; written with `and` it would use the hedge's count over the whole run and stop adding after `search_n_try` searches. MATLAB uses the round's `searchcount` (`bads.m:633`) | confirmed, inert (the poll rebuilds at its first step; a noisy search estimates from a rebuilt copy) | yes (always true) | never agreed (`c7c88ab`; MATLAB 2017) | `bads.py:1657-1661` | fix with `optim_state["search_count"] < search_n_try`, not with `and` | fingerprint |
| W1-10 | B5-I F11, B5-C F10, B5-R1 (last sentence) | when the posterior update after a refit fails and the retry with the previous hyperparameters succeeds, `len_scale`, `poll_scale` and `effective_radius` stay those of the refit the GP no longer holds (`gaussian_process_train.py:455-506`, `520-523`), and the markers are cleared (`540-541`); MATLAB keeps the refit's hyperparameters with their geometry and sets `post = []` (`gpupdate.m:279-354`) | confirmed port discrepancy, inert today (no exit flag -2 in 393 rebuilds: without points removed, the fit has already computed that posterior) | no | never agreed (`c7c88ab`; `685da15` added the restore and kept the geometry) | the `local_gp_fitting` retry row (last sentence) | fix: the geometry from the hyperparameters the GP holds, whatever W1-11 decides | fingerprint; an injection test that compares the geometry with the hyperparameters |
| W1-11 | B5-R1 | the retry with the previous hyperparameters on the new training set, which MATLAB lacks (`gpupdate.m:347-350`), cannot succeed without a refit: it repeats the same deterministic `update` (`gaussian_process_train.py:508`, `520-528`) | design question (left open in KD-B5-2) | no | never agreed (`c7c88ab`; MATLAB 2017) | the `local_gp_fitting` retry row | PI: skip the retry when there was no refit; keep it after a refit with W1-10; or do as MATLAB (keep the refit's hyperparameters and geometry, set the markers) | fingerprint; an injection test |
| W1-12 | B5-I F3, B5-C F2 | after each failed fit the noise lower bound, read from a GP that already holds the earlier raises, is raised by the cumulative nudge: lb₀ + k(k+1)/2 after k failures (`gaussian_process_train.py:668-675`); `noise_nudge[1]` is never read. MATLAB raises the start by the cumulative `nudge(1)` and the bound by `nudge(2)`, 0 by default (`gpHyperOptimize.m:161-166`). The fifth failure puts the bound above 5 and `set_bounds` raises `ValueError` | confirmed port discrepancy | yes, level 0: 21 of 53 deterministic refits had a failure, 13 of those 21 ended with the noise at the nudged bound (B5-C's "not reached" is wrong); none at level 1 (0 of 29) | never agreed (`c7c88ab`; MATLAB 2017) | the `_robust_gp_fit_` row "(at `1a21844`, line 680)" | fix: the bound from the entry bound plus `nudge[1]`; with W1-13, W1-14 and W1-16. MATLAB's rule gave the lower final value in 5 of 6 runs | population at default (with W1-13, W1-14, W1-16) |
| W1-13 | B5-I F4, B5-C F3 | `_robust_gp_fit_` has no exit for "all tries failed": the fifth failure raises `ValueError` (W1-12), the tenth `UnboundLocalError` for `res` (`gaussian_process_train.py:607`, `702`), and otherwise the last unfitted start would be returned; MATLAB returns the best evaluated start with exit flag -1 (`gpHyperOptimize.m:61-62`, `197-209`) | confirmed port discrepancy | no (at most 2 failures in a row in 82 refits) | never agreed (`c7c88ab`; MATLAB 2017) | the `_robust_gp_fit_` and forced-refit row | fix, with W1-12 | fingerprint; a failure-injection test |
| W1-14 | B5-C F12 | the retry's details: the points above the 95th percentile are cut with NumPy's linear percentile, where MATLAB's `prctile1` is `"hazen"` (1 point removed against 0 at n = 5 and 10, 3 against 2 at n = 50; `gaussian_process_train.py:635`, `gpHyperOptimize.m:137`); no stop below D points (`gpHyperOptimize.m:71`); exit flag 0 after retries, where MATLAB gives 1 (read only by the display) | confirmed port discrepancy (percentile, stop); confirmed, inert (exit flag) | percentile yes, level 0 (removal from the second failure, 10 of 53 refits); stop no | never agreed (`c7c88ab`; MATLAB 2017) | — | fix the percentile (`method="hazen"`) and the stop; with W1-12 | as W1-12 |
| W1-15 | B5-C F5 | a refit starts from gpyreg's design (`init_N` prior draws, 128 falling to 8, plus the given rows) and optimizes its best point, or its best two with the second replaced by a low-noise design point (`gaussian_process_train.py:1047-1071`; gpyreg `gaussian_process.py:1885-1920`); MATLAB runs one local optimization from the previous hyperparameters and one from the second-fit point (`gpupdate.m:371-408`, `gpHyperOptimize.m:47-75`). Over 14 refits fitted both ways: the same optimum in 8, the design better in 3 (by up to 17887), MATLAB's starts better in 3 (by up to 0.56) | design question (tuned in `8ff10f5`; no record calls it a departure from MATLAB, and KD-B1-4 leaves the `gp_train_*` options open) | yes, all levels | never agreed (`c7c88ab`; defaults `8ff10f5`) | — | PI: keep the design and put it on the sheet, or move to MATLAB's starts | none if kept; population if changed |
| W1-16 | B5-I F2, B5-C F4, B6-I F3, B6-C F2 | `_get_random_samples_from_priors_` exponentiates the centre and the SD of every prior on a log hyperparameter and draws the log value from N(e^μ, e^σ) (`gaussian_process_train.py:717-719`): 99.995% of the draws of the log output scale lie above its upper bound. A block without a prior raises `TypeError` (`713`), which stops a `negquad` run at its first retry. MATLAB draws N(μ, σ²) in log units and keeps a block without a prior (`gppriorrnd.m:66-78`) | confirmed port discrepancy | yes, level 0: 1 to 7 draws in each of the B6 verifier's four default runs (none in B6-C's sphere D=10), through the retries of `_robust_gp_fit_` (`655`) and the second fit (`402`); also the initial fit's failures (`193`) | never agreed (`9037851`; `gppriorrnd.m` 2018) | — | fix: N(μ, σ) in log units, a block without a prior left at its value; the effect is diluted (one row of gpyreg's design) and not isolated | population at default, on targets with retries (Rosenbrock), with W1-12 |
| W1-17 | B5-I F12(a) | at D = 1 `len_scale` is 1, not the fitted length scale (`gaussian_process_train.py:455`); MATLAB does the same (`gpdefBads.m:51`, `gpupdate.m:285-292`) | design question, Python matches MATLAB | yes, at D = 1 | the two agree | — | PI: keep MATLAB's behavior, or use the length scale at D = 1 | none if kept; population of D = 1 problems if changed |
| W1-18 | B5-I F12(b) | `len_scale += len_scale + np.exp(...)` doubles the running sum and has no weights (`gaussian_process_train.py:458`); one hyperparameter sample always (KD-B5-4) | confirmed, inert | no | `c7c88ab`; MATLAB's weighted sum since 2017 | `gaussian_process_train.py:413` | fix in passing | fingerprint |
| W1-19 | B5-I F13 | the `init_N` schedule divides 0 by 0 when `min(max_fun_evals, n_train_max)` equals the initial design (`ValueError` from `round(nan)`), and extrapolates the cubic above 128 when it is below it (413, 968); the cubic mixes `x_` and `x`, harmlessly (`gaussian_process_train.py:1048-1058`) | confirmed defect (Python only) | no (a small `max_fun_evals`) | `8ff10f5` | `gaussian_process_train.py:979-983` | fix: guard the denominator, clip x to [0, 1] | fingerprint; a small-budget test |
| W1-20 | B5-I F14 | the retry loop of the initial fit has no cap (`gaussian_process_train.py:162-210`): a fit that keeps failing hangs the run | confirmed defect | no | `9037851` | the `init_and_train_gp` row (a B6 row, closed here) | fix: cap the retries, then raise or fall back | fingerprint; an injection test |
| W1-21 | B5-R2 | after a failed fit the retry reads `tmp_gp`: its bounds (`gaussian_process_train.py:671`) and, with `use_slice_sampler=True`, its data (`651-653`, `741`), 69 rows where the retry fits 64. At default no bound is NaN and the sampler is off, so nothing depends on what gpyreg leaves after a failure (its data up to 1.3.2, the GP before the call from 1.3.3). The row's clause on `negquad` does not hold: the initial fit fills its bounds | confirmed defect with the slice sampler; inert at default | no (`use_slice_sampler=True`) | `c7c88ab`; gpyreg 1.3.3 changed what a failed fit leaves | the `_robust_gp_fit_` row "(at `2f3d949`)" | fix when the slice-sampler path is touched: sample on the retry's data (with N1 below) | fingerprint; a test with `use_slice_sampler=True` |

## B6: the GP model

| Id | Source | What | Classification | Default run | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|
| W1-22 | B6-R1 | at each rebuild `local_gp_fitting` computes the noise prior's new centre, `log(noise_size) + mesh_noise_multiplier*log(mesh_size)`, and never writes it back (`gaussian_process_train.py:300-306`; `364` sets the definition prior); MATLAB updates it at every training (`gpdefBads.m:207`, `gpupdate.m:276`). At level 0 the centre stays at -3.45 all run, where MATLAB's falls to -6.2 (sphere D=2) and -7.3 (Rosenbrock D=3). Both B6 reports state the opposite in their answers to Q2 | confirmed port discrepancy | yes, level 0 only (levels 1 and 2 set the multiplier to 0) | never agreed (no revision writes it back; MATLAB since `31a39f3`, 2017) | `local_gp_fitting` "(at `8afbe16`, lines 298-306)" | fix: one line, and a level-0 test of the centre after a rebuild. The update moved 6 runs in both directions | population at default (with W1-29) |
| W1-23 | B6-I F1, B6-C F1 | the constant mean is bounded for the whole run by gpyreg's recommendation on the initial design's lowest 80%, `[min - h/2, max + h/2]` (`gaussian_process_train.py:958-962`), while its prior is re-centred at each rebuild (`315-324`): the prior leaves the bounds routinely and the fitted mean is pinned at a bound (sphere D=2: prior outside at 26 of 29 rebuilds; Ackley D=6: by up to 26 SD; a shifted Rosenbrock: mean 1753 with local y ≤ 53). MATLAB's mean is unbounded (`gpdefBads.m:173`) | confirmed port discrepancy | yes, every level | matched MATLAB at `c7c88ab`; diverged at `9037851`; the prior leaves the bounds since `ab4dded` (#66); MATLAB unchanged since 2017. #70 (`2210046`) records it as open | `_gp_hyp` "(at `8afbe16`, lines 968-971)" | fix: `(-inf, inf)`, as MATLAB. Unbounded, 3 of 3 whole runs ended lower (sphere 9.4e-8 against 1.6e-6; Ackley 0.0138 against 0.0194; shifted Rosenbrock 2.9e-7 in 122 evaluations and 1.8 s against 1.9e-5 in 152 and 55 s). Not on the sheet; KD-B6-2 says nothing about the bounds | population at default (`ackley_D6`, `sphere_D10`, `sphere_nonbox_D3` reach it) |
| W1-24 | B6-I F2, B6-C F1(c) | when a prior lies more than about 38 SD outside its bounds, gpyreg's normalization (the mass inside the bounds, as a difference of cdf or sf values in linear space, `gaussian_process.py:2045-2107`, `2170`, `2354`) underflows to 0: the log prior is +inf, the objective -inf, and the design's draws for that hyperparameter are infinite (`f_min_fill.py:205-217`). Refits then take seconds instead of hundredths and do not return the MAP. MATLAB's priors carry no such constant | confirmed defect (gpyreg's), reached through W1-23 | yes, level 0, target-dependent (a shifted Rosenbrock; Ackley D=4) | reachable since `ab4dded` | the `_gp_hyp` row (its NaN log prior) | fix: in gpyreg, the log mass in log space (logcdf, logsf); in PyBADS, W1-23 removes the route at default | the gate of a gpyreg release; with W1-23 |
| W1-25 | B6-I F4, B6-C F3 | gpyreg's Cholesky retries multiply the noise by 10 per failed attempt, up to 10 attempts (`gaussian_process.py:3584-3666`), and the posterior keeps the inflation (`sn2_mult`) for prediction, while `get_hyperparameters` reports the fitted noise; the bounds of the output scale and the noise admit matrices singular in double precision. MATLAB raises at the first failure (`CholAttempts = 0`, `bads.m:272`; `infExact_fastrobust.m:80`), restarts the fit with a nudge and empties `post` | confirmed port discrepancy (substituted library; KD-B6-1 leaves the Cholesky handling open) | yes, level 0, target-dependent (Rosenbrock D=2: 609 of 1725 factorizations inflated, up to ×1e9, and 73 of 116 GP states handed on; sphere D=2: 4 of 53; Ackley D=6: none) | never agreed (gpyreg's since the port; MATLAB since 2017) | — | PI: (a) keep it and document it under KD-B6-1; (b) treat `sn2_mult > 1` as a failure, as MATLAB (a gpyreg switch; `chol_attempts` is unread); (c) narrow the bounds, away from MATLAB. The effect on results is not measured | none for (a); population at default, and a gpyreg release's gate, for (b) or (c) |
| W1-26 | B6-I F6, B6-C F4 | a target whose lowest 80% of initial values are equal stops the run in `_gp_hyp`: the mean prior's SD is `np.std(hpd_y) = 0` and gpyreg's `set_priors` raises `ValueError` (`gaussian_process_train.py:960-961`); at a rebuild, `log(np.std(gp.y)) = -inf` would raise the same way (`357`, not reproduced). MATLAB's definition prior, N(0, 1), cannot fail, and it fits no GP at initialization | confirmed port discrepancy (initialization); needs MATLAB (the rebuild case) | only with such targets (a plateau over the plausible box) | matched MATLAB at `c7c88ab` (N(0, 1)); diverged at `9037851` | — (related: wave 0, "Found while verifying", the thin feasible region) | fix: a positive fallback SD, and keep the previous prior at `357`, as KD-B6-2 does | fingerprint; a plateau test |
| W1-27 | B6-C F5 | PyBADS fits a GP at initialization (`bads.py:1185-1205`), under the definition priors and from `[log std(hpd_X), log std(hpd_y), 0, log noise_size, median(hpd_y)]` with hpd = `round(0.8 N)`; MATLAB only defines the GP, from zeros, `log NoiseSize` and the median of the lowest `ceil(0.8 N)` (`bads.m:465-469`, `gpdefBads.m:164-165`). The first rebuild refits from that fit, and the first target is predicted under its hyperparameters, where MATLAB uses the unfitted values (`bads.m:469`, `539`) | confirmed port discrepancy (KD-B6-1 leaves the initial fit open); also W0-7 | yes, every run | PyBADS's own since `c7c88ab`; MATLAB changed its starting mean in `d4fead5` (2022-10-31), after `9037851` | — | PI: document the fit as deliberate, or follow MATLAB; W0-7 goes to the fix pass regardless. The effect is not measured | none if kept; population at default if changed |
| W1-28 | B6-C F6 | `gp_cov_prior = "ard"` is not ported and any other value is accepted: the definition prior stays all run (`gaussian_process_train.py:329-351`); MATLAB sets a per-dimension empirical prior and raises on an unknown value (`gpdefBads.m:254-274`) | confirmed port discrepancy (unported) | no | never ported (`c7c88ab`) | — | PI: port `"ard"`, or refuse it with a message; refuse unknown values either way | fingerprint; population with `"ard"` if ported |
| W1-29 | B6-C F7 | the output-scale prior's centre is `log(np.std(gp.y))` with ddof 0 (`gaussian_process_train.py:357`); MATLAB's `std` divides by N-1 (`gpdefBads.m:293`): 0.005 to 0.01 against a prior SD of 2 | confirmed port discrepancy, negligible | yes | never agreed (`c7c88ab`) | — | fix (`ddof=1`), with W1-22 | with W1-22 |
| W1-30 | B6-I F5 | the noise's upper bound is a log SD of 5 (SD 148), below the prior's centre when `noise_size` exceeds 148 (`gaussian_process_train.py:915`); MATLAB has the same bound (`gpdefBads.m:161`). With noise of SD 500 the fitted noise sits at the bound | design question, shared with MATLAB | only when the noise SD exceeds about 148 (level ≥ 1) | the two agree (`c7c88ab`; MATLAB 2017) | — | PI: keep MATLAB's constant and document it, or warn when `noise_size > e^5`, or tie the bound to `noise_size` | fingerprint |
| W1-31 | B6-I F7 | the effective radius `sqrt(alpha*(exp(1/alpha) - 1))` (`gaussian_process_train.py:504`) is said not to match gpyreg's kernel | not a defect: it equals `gpupdate.m:317`, and gpyreg's kernel is e^-1 at √2 times it, the convention of MATLAB's Matérn constants | yes | the two agree | — | correct the record; optionally a comment naming the convention | none |
| W1-32 | B6-I F8 | `fit_lik=False` stops the run: gpyreg has no `"delta"` prior (`gaussian_process_train.py:896-898`, `979`); MATLAB refuses fixed noise too (`gpdefBads.m:140`) | confirmed, inert (both refuse; only the message differs) | no | `c7c88ab`; MATLAB 2017 | — | keep, with MATLAB's message ("Fixed noise not supported") and "unsupported" in the option's description | fingerprint |
| W1-33 | B6-I F9 | `upper_gp_length_factor` sets bounds that `gaussian_process_train.py:930-933` overwrite unconditionally (`908-913`): no effect at any value | confirmed, inert (a dead PyBADS-only option) | no | `c7c88ab` | — | correct the record (KD-B1-4 lists it as read by code; it belongs with KD-B1-5), and remove the branch or document it | fingerprint |
| W1-34 | B6-I F10 | `_init_optim_state_` accepts 12 names of `gp_mean_fun` (`bads.py:935-955`), of which 9 raise "Unknown mean function!" when the GP is built (`gaussian_process_train.py:564-571`); `"negquad"` is PyVBMC's concave mean for log densities, wrong for a minimizer, has no priors in `_gp_hyp`, and stops at the first retry (W1-16) | confirmed defect (the names); design question (`negquad`) | no | `c7c88ab` | `bads.py:880-893` | fix: accept only the names that can be built (a stricter interface, refused when `BADS` is created); PI: fix or remove `negquad` | fingerprint |

## Notes on the reports

- The verifiers corrected the reports in six places: both B6 reports on the
  noise prior (W1-22); B5-C's "not reached at default" for the nudge and the
  prior sampler (W1-12, W1-16); B5-I's "worse on all 5 seeds" for W1-1, which
  did not replicate; B5-I F10, which MATLAB shares (W1-8); B5-I's "the n ≥ 3
  test never fired" (W1-3); and B6-I F7, not a defect (W1-31). B6-I F5 (W1-30)
  is shared with MATLAB, which that track does not read.
- Two differences the preparatory agent saw in passing, kept from the
  reviewers, belong to these slices, and the reviewers found both: (d) the GP
  fit at initialization (B6-C F5, W1-27) and (f) the χ² bounds without the
  factor 2 (B5-I F5, B5-C F8, W1-5). Two more, of other slices, were met in
  passing: (e) `p_less` over the `D+1` largest probabilities (B5-C, slice B4)
  and (g) a budget below the initial design (B5-I, slice B2).
- W0-7 (the starting GP mean over `round(0.8 N)`) is named inside B6-C F5
  (W1-27). W0-8 (the unstable sort of the training set) is described by
  both B5 reports in their answers to Q1, not as a finding: over 541
  rebuilds the transcription of MATLAB's `'nearest'` selected the same set
  every time, the row order differed in 182, and no tie fell at the cut-off
  (B5-C, `s6_nearest.py`). Both rows go to the GP fix pass, as ruled.
- A proposal for grouping the fixes: the calibration test (W1-3 to W1-6),
  under one population comparison; the fit retries (W1-12 to W1-14, W1-16),
  under another; the priors and bounds (W1-22, W1-23, W1-29, with W0-7 and
  W0-8); and the fixes that must move nothing (W1-9, W1-10, W1-13, W1-18 to
  W1-21, W1-26, W1-32 to W1-34), each under the fingerprint. W1-1 and W1-2
  each need their own comparison.

## Survey rows of B5 and B6

Every open row of the candidate table that belongs to these slices is closed
by a row above: `bads.py:626-627` (W1-1); `gaussian_process_train.py:413`
(W1-18); `bads.py:1657-1661` (W1-9); the `local_gp_fitting` retry (W1-10,
W1-11); `_robust_gp_fit_` and the forced refit (W1-13, W1-12);
`gaussian_process_train.py:979-983` (W1-19); `_robust_gp_fit_` at
`2f3d949` (W1-21); `_robust_gp_fit_` at `1a21844`, line 680 (W1-12); the
`_save_gp_stats_` calls (W1-3; the second clause, on `_re_evaluate_history_`,
holds at `95da7f1` and belongs to B2, below); `init_and_train_gp` (W1-20);
`bads.py:880-893` (W1-34); `local_gp_fitting` at `8afbe16`, lines 298-306
(W1-22); `_gp_hyp` at `8afbe16`, lines 968-971 (W1-23, W1-24). The rows on
the search after a failed rebuild (B3) and on `_get_target_from_gp_` (B4)
touch the GP but belong to slices of wave 3, which has its pass ahead.

## Found while verifying

Reproduced or read by a verifier or the orchestrator, but outside the
reports' findings; each is left to the wave of its slice, all of which have
their pass ahead, or proposed here.

- With `use_slice_sampler=True`, `_robust_gp_fit_` stops the run at the
  third consecutive fit failure with `ValueError: The initial starting point
  X0 is outside the bounds.` from `SliceSampler`: the start rises by k and
  the nudged bound by k(k+1)/2 (B5 verifier, N1, reproduced once). Slice B5;
  the fix of W1-12 and W1-21.
- `_re_evaluate_history_` selects the neighbours of each stored GP with that
  GP's `len_scale` and `effective_radius` (`bads.py:2733`), where MATLAB
  swaps only `hyp` into the current GP (`bads.m:1384-1395`): the second
  clause of the `_save_gp_stats_` survey row, holding at `95da7f1`; in one
  noisy run no neighbour set differed (B5 verifier). Its in-place change of
  the stored GPs (N3) is W0-1. Slice B2, wave 2, to be checked against W0-1's
  fix (`d6e3f61`), proposed as a re-evaluation from a copy of the working GP.
- At level 2, the high-noise check of `local_gp_fitting` takes the base 1,
  whatever `noise_size` the user sets (`bads.py:1132-1139`), where MATLAB
  reads `NoiseSize` (B5-C, answer to Q3; read by the orchestrator). It is
  intentional, by the PI's ruling of 2026-09-25 in #71: the comment there,
  the changelog's `noise_size` entry and the survey row fixed in `7b50a3a`.
  Not on the sheet: an entry to add.
- The half-bounds check refuses a problem that mixes fully bounded and fully
  unbounded variables (B5 verifier, N2, reproduced): the survey row
  `bads.py:500-507`, slice B1, wave 2. It limits W1-1 to problems unbounded
  in every variable.
- A run with `max_fun_evals` of 3 or 4 at D = 2 makes 6 evaluations (B5-I;
  B5 verifier, N5): wave 0's "Found while verifying" item on a budget below
  the initial design, slice B2, wave 2.
- `update_hedge` computes `exp(-0.5*g**2/sqrt(2*pi))`, where `acqPortfolio.m:64`
  has `exp(-0.5*g.^2)/sqrt(2*pi)` (B5-C, not verified): the survey row
  `search/search_hedge.py:141`, slice B3, wave 3.
- `_estimate_noise_` ranks the high-density points by descending y, PyVBMC's
  convention, and nothing reads its `sn2hpd` (B5-I, B5-C, B6-I): a candidate
  for removal with the other PyVBMC leftovers.
- gpyreg's optimizer (L-BFGS-B at a tolerance of 1e-5 from the best design
  point) against MATLAB's `TolFun` 0.1, `TolX` 1e-4 and 150 evaluations
  (B6-C, not reviewed): with W1-15.
- The sheet: no entry for W1-23 or W1-22; KD-B6-2's "Otherwise the re-centred
  prior follows MATLAB" holds for the centre and the width, not for the
  bounds under them; KD-B1-4 lists `upper_gp_length_factor` as read by code
  (W1-33) (B6 verifier). No finding contradicts an entry outright (both
  verifiers).

## Rulings (PI, 2026-09-26)

The orchestrator proposed a disposition for every row, following its
verifier's recommendation unless it says why not. The PI ruled on the rows
marked "(PI)" (W1-25 measured behind a switch in gpyreg that is off by
default; W1-27 kept; W1-17 a defect; W1-8 left to the orchestrator; W1-6,
W1-28 and W1-34 as proposed) and accepted every other proposal as written.
As in wave 0, a fix is one commit per row on `dev-port-review`, with a test
that fails at `95da7f1` and passes at the commit, and a changelog line in
every commit a user can notice. The fix pass is not started.

**Fix, moving nothing** (each under the fingerprint):

- W1-9: count with the round's `optim_state["search_count"]`; the example in
  `AGENTS.md`, "MATLAB logicals", then names a fixed slip and is reworded.
- W1-10 and W1-11: retry with the previous hyperparameters only after a
  refit, and take the geometry from the hyperparameters the GP holds.
  Without a refit the retry repeats a failed deterministic computation, so
  skipping it changes no run. This settles the open part of KD-B5-2.
- W1-13: when every try fails, `_robust_gp_fit_` returns the best start it
  evaluated with exit flag -1, as MATLAB (committed within the retries
  batch below).
- W1-18: MATLAB's weighted sum over the samples.
- W1-19: guard the denominator of the `init_N` schedule and clip its
  argument, so that `init_N` stays between the final and the initial size.
- W1-20: cap the retries of the initial fit, then stop with a clear message.
- W1-21: the retry's slice sampler samples on the retry's data. N1 comes
  from W1-12's rising bound and should go with it; the test with
  `use_slice_sampler=True` checks both.
- W1-26: a positive fallback wherever a prior's width or centre comes from
  the spread of the targets (the mean's SD in `_gp_hyp`; at a rebuild, the
  output scale's centre keeps the previous prior, as KD-B6-2 does for the
  mean), with wave 0's thin feasible region as a second reproduction if it
  is the same mechanism. KD-B6-2 is extended.
- W1-31: a comment naming the convention.
- W1-32: MATLAB's message ("Fixed noise not supported") when `BADS` is
  created, and "unsupported" in the option's description.
- W1-33: remove the overwritten branch; the option moves to KD-B1-5.
- W1-34 (PI: accepted): accept only `"zero"` and `"const"`, and refuse
  every other name when `BADS` is created, `"negquad"` included: it has the
  wrong shape for a minimizer, has no priors, and stops at the first retry.
  A stricter interface: a changelog entry and an "Upgrading from" line.
- W1-28 (PI: accepted): refuse `gp_cov_prior="ard"` and unknown values with
  a message, and put the unported `"ard"` on the sheet (a changelog entry and an "Upgrading
  from" line). This departs from the verifier, which proposed porting it:
  it is off by default, and a port needs its own population comparison; a
  `TODO.md` item keeps the port.
- W1-8 (PI: the orchestrator's call; revised from "keep"): with
  `poll_training` off, the poll neither performs nor records the refit, so
  that the search's next refit is not delayed by one that did not happen. It
  is a defect MATLAB shares, but off by default, so MATLAB's tuning does not
  bear on it. A unit test and the fingerprint (default runs do not reach it),
  a changelog line, and an entry in `matlab_side_defects.md`.

**Fix, moving results**, in batches, each ending in a population comparison
on Linux (this platform and versions are those of
`population_linux_gpfixes_20260925`) against the end of the batch before; the
fingerprint recorded at every commit, and a batch's steps compared one by
one only if its comparison flags something. In this order:

0. Baseline: the head of `dev-port-review` against
   `population_linux_gpfixes_20260925` (at `97b2c66`; #67, #70, #71 and the
   wave 0 fix pass came after it), which serves as the pass's baseline if it
   reproduces the records `compare` reads. Otherwise a new baseline at the
   head.
1. The fit retries: W1-12 (the bound raised from the entry bound by
   `nudge[1]`, so not at all at the default `[1, 0]`), W1-13, W1-14 (the
   `"hazen"` percentile, the stop below D points, exit flag 1), W1-16 (draws
   of N(μ, σ) in log units; a block without a prior keeps its value). This
   comes first because the second fit that MATLAB and PyBADS make when the
   mean falls below the targets (`gpupdate.m:384-393`) draws from W1-16's
   sampler, and W1-23 changes where the mean goes.
2. The priors and bounds: W1-23 (the mean unbounded, as MATLAB; this also
   closes W1-24's route at default), W1-22, W1-29, W0-7, and W0-8 (first
   under the fingerprint, and into this comparison if it moves).
3. The calibration test: W1-3 (the SD of the observation, as MATLAB's `ys`;
   the fix checks which noise MATLAB's prediction adds at level 2, and the
   replacement by 1e-6 goes), W1-4, W1-5. W1-6 (PI: accepted): keep scipy's
   Shapiro-Wilk and put the substitution on the sheet, with the verifier's disagreement
   rates as its evidence. In default runs the two tests differ only on the
   replaced SDs that W1-3 removes; revisit if this comparison is flagged.
4. W1-2 alone: clear `reset_gp` at the rebuild it asks for (a failed
   rebuild is already marked). If the comparison flags a worsening, it comes
   back to the PI, to keep and put on the sheet.
5. W1-1 alone: swap the two assignments (the default suite's
   `ellipsoid_D3_unbounded` reaches it) and correct `AGENTS.md`.
6. W1-17 (PI: a defect, fixed; revised from "keep"): use the fitted length scale at D = 1 too. MATLAB's
   `ncovlen > 1` (`gpupdate.m:285-292`) is meant to tell per-dimension
   length scales from an isotropic one, and at D = 1 it takes the one length
   scale of the ARD kernel for an isotropic kernel's; the port copied the
   test. PyBADS's kernel is always ARD, so the test goes, together with
   W1-18 on the same lines. The effect is bounded: at D = 1 the training set
   has between 50 and 60 points whatever the radius. The default suite has
   no D = 1 configuration, so its fingerprint and comparison stay unchanged,
   and the gate is a comparison on a few 1-D configurations (30 seeds). An
   entry in `matlab_side_defects.md`.

W0-1, once ruled, is gated on top of whichever batches have landed by then;
the fixes table records the order.

**Keep, and record:**

- W1-15, the refit's starts from gpyreg's design, with its optimizer's
  tolerances: on the sheet, as deliberate. On the same data the design found
  the better optimum more often, and by far more, and it avoided fit
  failures that MATLAB's starts met.
- W1-25, gpyreg's inflation of the noise (PI: measure, with the switch
  opt-in in gpyreg): kept for now, on the sheet under KD-B6-1. A switch in
  gpyreg, off by default so that no user of gpyreg (PyVBMC among them)
  changes, makes a failed factorization an error, as MATLAB's
  `CholAttempts = 0` (the fit's random design skips a draw that fails; the
  optimizer's run fails). It goes to gpyreg on its own branch and pull
  request. After batch 1, one population comparison with the switch on
  (from the branch, recorded as exploratory) against batch 1's end. Turning
  it on in PyBADS would change PyBADS's default runs, so that is a separate
  ruling on that evidence, with its own gate.
- W1-27, the GP fit at initialization (PI: kept; fitting the GP on the
  initial design makes sense): on the sheet, as deliberate, by that ruling.
  W1-20 and W1-26 are fixed within it, and W0-7 regardless.
- W1-7 and W1-30, which PyBADS shares with MATLAB: kept, as MATLAB has
  them, and collected in `matlab_side_defects.md`, created with this pass,
  as shared design observations, beside W1-8 and W1-17 (shared defects that
  PyBADS fixes). W1-30 also gets a warning when `BADS` is created with a
  `noise_size` above e^5.
- The base noise of 1 at level 2 ("Found while verifying"): on the sheet,
  by the ruling of #71.

**Out of this pass:** W1-24 goes to gpyreg, as an issue and a pull request
on its own branch (the log mass in log space, and the design's draws from a
prior far outside its bounds), gated as a gpyreg release. By the same
principle as W1-25's switch, the log-space mass is taken only where the
linear one underflows, so that no fit that succeeds today changes. The items of
"Found while verifying" left to waves 2 and 3 go to them with their slices.

## Fix pass

Not started (PI, 2026-09-26), but for step 0. How it runs (PI,
2026-09-26): the fixes go on `dev-port-review-w1`, and one pull request
into `dev-next` carries wave 1's records and fixes once the pass is done;
each fix is made by an agent in a git worktree of its own, one commit per
row, and the orchestrator reviews each diff and cherry-picks it; the
gpyreg side of W1-25 and W1-24 goes to `acerbilab/gpyreg` on a branch of
its own, from this session. The records of wave 1 were
rebased onto `dev-next` at `e004c79`, which carries wave 0 and its fix pass
(#72) and W0-1 (#73), so the pass starts from there: `dev-port-review` is
superseded, and W0-1 is in the base rather than gated on top of the
batches.

**Step 0, the baseline** (2026-09-26, Linux, the platform and versions of
the Linux references): the default suite × seeds 0-29 at `ac3dfed` (the
head of `dev-port-review-w1`, the package code of `e004c79`) is the new
Linux reference,
[`population_linux_wave0_20260926`](../../population_linux_wave0_20260926/README.md),
against which the batches compare. Against the previous Linux reference it
flags the number of evaluations of `ellipsoid_D3_homo`,
`multisensory_s1_D6_homo` and `ellipsoid_D3_hetero`, fewer, with no error
test flagged: W0-1's flags on Windows. Its step at `138a141` (the fix pass
of wave 0 without W0-1) reproduces the previous reference exactly in every
field `compare` reads. Its null check flags nothing. The fingerprint of
`dev/scripts/fingerprint.py` at `ac3dfed` on this platform, for the fixes
that must move nothing: `bfbc6d6737e99d88`.

**Found while fixing** (2026-09-26):

- **W1-35. W0-1's re-estimate crashes a noisy run whose rebuild fails.**
  `_re_evaluate_history_` (`bads.py`, about line 2780, from W0-1 at
  `e004c79`) sets an iterate's recorded hyperparameters on a copy of the
  working GP with `compute_posterior=False` and rebuilds it around the
  iterate. When the rebuild's posterior fails, `local_gp_fitting` puts back
  the GP it was given, which has no posterior, and `predict` raises
  `TypeError`. The inject gate (`gp_update_failures.py --inject 0.02`,
  seeds 0-9) crashed 50 of 50 noisy runs at `bc57dd6` and 48 of 50 at
  `4ba6787`, all there, after 49 to 490 evaluations; the stress run at
  `a83bd51`, before W0-1, finished every run with 310 restores in the
  re-estimate, since the stored GP it rebuilt kept its posterior. Default
  runs do not reach it (no guarded computation fails over the default
  suite: `dev/plans/gp-update-guards.md`, Phase 3), but W1-25's switch makes
  failed factorizations real. MATLAB records NaN: a failed `gpupdate`
  clears `post` (`private/gpupdate.m:340-349`), and `gppred` recomputes it,
  fails and keeps its NaN (`utils/gppred.m:22-56`). Its `min` and `max`
  skip a NaN, and the incumbent becomes NaN when the current iterate fails
  (`bads.m:1101-1104`). NumPy's `argmin` and `argmax` pick a NaN instead of
  skipping it: the end-of-iteration improvement (`bads.py`, about lines
  1517-1535) and the final choice (about 1573-1580). Ruling (PI,
  2026-09-26, the third of three options, after MATLAB's NaN throughout and
  keeping the stale estimate): a past iterate whose re-estimate fails gets
  NaN, as in MATLAB, and the two choices skip NaN; the current iterate
  keeps its estimate when its own re-estimate fails, so that the incumbent
  is never NaN. Under the fingerprint (default runs do not reach it), with
  the inject gate, before W1-25's measurement.
- The inject gate's script counted a restore of `local_gp_fitting` by a
  failed call after the failed update, the retry that W1-11 removes when
  there is no refit; it now reads the `needs_refit` marker on the GP
  returned (`4ca8e41`), which gives the old counts at a commit that always
  retries.
