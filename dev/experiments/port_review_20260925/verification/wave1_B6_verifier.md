<!-- Report of the verifier of wave 1, slice B6 (the two B6 reports and the survey row it was given as B6-R1), reading PyBADS at 95da7f1 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave1/B6_verifier/. -->

# Wave 1 verification: B6

Scripts and outputs are in `/tmp/claude-0/-home-user-pybads/4f4200d5-5a60-5de0-9b26-94c4f1e930f9/scratchpad/wave1/B6_verifier/`. Each `v_*.py` has a matching `.out`, and `common.py` holds the shared helpers. Every script printed `pybads.__file__ = /home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__ = /home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. All runs were seeded, used one BLAS thread, ran one at a time, and stayed at or under 200 evaluations. Python lines are at `95da7f1`, MATLAB at `74919c0`, gpyreg at v1.3.3.

**Notation.** I-Fn is the internal report, C-Fn the comparison report.

**Where the two reports describe the same behaviour** (verified once each):
- I-F1 = C-F1.
- I-F2 = C-F1(c).
- I-F3 = C-F2.
- I-F4 = C-F3.
- I-F6 = C-F4.
- I-F8 and I-F9 also appear in the comparison's "Python-only options" note.

**Correction to both reports.** Both Q2 tables say the noise prior's centre is updated at each rebuild, and it is not (B6-R1). The internal report's "−10.0 against −7.91 at the final mesh" is MATLAB's value. PyBADS's centre stays at −3.45, 4.5 SD above that floor.

## 1. Summary

| Finding | Classification | Reached at default | Dating | Confidence |
|---|---|---|---|---|
| **B6-R1** noise-prior centre computed, never written | confirmed port discrepancy | yes, level 0 only (levels 1–2 coincide with MATLAB) | never agreed: MATLAB has updated it since `31a39f3` (2017); the port has computed and discarded it since `c7c88ab` | high |
| **I-F1 = C-F1** mean bounds fixed on the initial design (MATLAB: unbounded) | confirmed port discrepancy | yes, every level (measured at level 0) | matched at `c7c88ab`; diverged at `9037851`; the prior leaves the bounds since `ab4dded` (#66); MATLAB unchanged since 2017 | high |
| **I-F2 (= C-F1(c))** normalization underflows to 0 | confirmed defect (gpyreg's linear-space constant), reached only through F1 | yes, level 0, target-dependent | reachable since `ab4dded`; MATLAB's priors carry no such constant | high |
| **I-F3 = C-F2** prior sampler draws N(e^μ, e^σ); a block without a prior raises | confirmed port discrepancy (plus a crash for `negquad`) | yes, at level 0 through fit retries and the second fit | never agreed: introduced in `9037851`; before that it was a slice-sampler draw | high |
| **I-F4 = C-F3** Cholesky retries inflate the noise ×10^k | confirmed port discrepancy (substituted library; open under KD-B6-1) | yes, level 0, target-dependent | never agreed: gpyreg's since the port; MATLAB `CholAttempts=0` since 2017 | high (mechanism), medium (effect) |
| **I-F5** noise upper bound log SD 5 | design question, shared with MATLAB | only when noise SD > ~148 (level ≥ 1) | always agreed (`gpdefBads.m:161`, 2017; `c7c88ab`) | high |
| **I-F6 = C-F4** flat initial design raises `ValueError` | confirmed port discrepancy (initialization case); the rebuild case needs MATLAB | only with such targets | matched MATLAB's N(0,1) at `c7c88ab`; diverged at `9037851` | high (PyBADS) |
| **I-F7** effective radius | not a defect | yes | always agreed (`gpupdate.m:317`) | high |
| **I-F8** `fit_lik=False` raises | confirmed, inert (both refuse; only the message differs) | no | `c7c88ab`; MATLAB errors since 2017 | high |
| **I-F9** `upper_gp_length_factor` has no effect | confirmed, inert (dead PyBADS-only option) | no | overwritten unconditionally since `c7c88ab` | high |
| **I-F10** mean-function names; `negquad` | confirmed defect (non-default names); design question (`negquad`) | no | `c7c88ab` | high |
| **C-F5** GP fitted at initialization, other start values | confirmed port discrepancy (open under KD-B6-1) | yes, every run | the initialization fit is PyBADS's own since `c7c88ab`; MATLAB changed its start mean in `d4fead5` (2022-10-31), after `9037851` | high (difference), effect not measured |
| **C-F6** `gp_cov_prior='ard'` ignored, unknown values accepted | confirmed port discrepancy (unported) | no | never ported (`c7c88ab`) | high |
| **C-F7** `np.std` with ddof 0 | confirmed port discrepancy, negligible | yes | never agreed (`c7c88ab`) | high |

## 2. Per finding

### B6-R1. The noise prior's centre is never updated
- **Lines.** `gaussian_process_train.py:300-306` builds `prior_noise = (prior_noise[0], (mu_noise_prior, prior_noise[1][1]))`, and nothing writes it back to `gp_priors`. Line 364 then calls `gp.set_priors(gp_priors)` with the definition prior (`_gp_hyp:976`).
- **MATLAB.** `gpdefBads.m:207`, `gpstruct.prior.lik{end}{2} = log(NoiseSize(1)) + options.MeshNoiseMultiplier*log(MeshSize)`, runs at every `gpupdate` (`gpupdate.m:276`), and `gpstruct.inf` is rebuilt with the new prior (line 314).
- **Check.** `v_r1_noise_prior.py` wraps `local_gp_fitting` and reads the prior the GP holds after each call:
  - Level 0, sphere D=2: "distinct prior centres held by the GP: [-3.453878] … MATLAB centre range: [-6.226, -3.454]" (29 rebuilds).
  - Level 0, Rosenbrock D=3: held −3.454, MATLAB's −7.266 to −3.454 (85 rebuilds).
  - Level 1: 0.0 on both sides. Level 2: −6.908 on both sides.
- **History.** A loop over every revision of the file found no write of `gp_priors["noise_log_scale"]` in any of them, from `c7c88ab` to `95da7f1`. The MATLAB line dates from `31a39f3` (2017) and was edited in `a3b6ebd` (2021). The two never agreed. `8afbe16`, which the row cites, is not an ancestor of `95da7f1`; it reached this line as `ab4dded` (#66). The same lines are at 300-306.
- **Consequence.**
  - At level 0 the port keeps the noise prior at √tol_fun (log −3.45, SD 1) all run. MATLAB lowers it by ½·log(mesh size).
  - In `v_r1_counterfactual.py` a copy of `local_gp_fitting` with the one missing line moved results in both directions. The table gives fval, evaluations in parentheses where they differ:

    | run | port | with the update |
    |---|---|---|
    | sphere D=2 s0 | 1.55e-6 | 1.08e-7 |
    | sphere D=2 s1 | 7.8e-8 | 1.74e-6 |
    | Rosenbrock D=3 s0 | 1.45e-5 (97) | 2.05e-5 (111) |
    | Rosenbrock D=3 s1 | 1.9e-7 | 5.3e-5 |
    | Ackley D=4 s0 | 2.09e-4 | 5.49e-4 |
    | Ackley D=4 s1 | 5.61e-4 | 3.82e-4 |

  - Six runs give no direction.
- **Tests.** `test_noise_size_forms` (`test_noisy_runs.py:190-204`) checks the centre only in noisy runs, where MATLAB's update leaves it unchanged, so it passes with this defect present.
- **Recommended disposition: fix.**
  - Add `gp_priors["noise_log_scale"] = prior_noise` after line 306, and a level-0 test that the centre equals log(noise_size) + 0.5·log(mesh_size) after a rebuild.
  - It changes default deterministic runs, so it needs the population comparison at default options.

### I-F1 = C-F1. The constant mean's bounds come from the initial design and never change; MATLAB leaves the mean unbounded
- **Lines.** `_gp_hyp:958-962` sets gpyreg's `[min(hpd_y) − h/2, max(hpd_y) + h/2]`, and nothing resets it. The prior is re-centred at `315-324`.
- **MATLAB.** `gpdefBads.m:173` sets `[-Inf; Inf]`, unchanged since 2017. Line 223, `bounds.mean{1}(1) = min(y)`, is commented out.
- **Check.** `v_f1_mean_bounds.py`:
  - Sphere D=2: bounds [0.2365, 3.202] throughout; the prior centre is outside them at 26/29 rebuilds, by at most 4.7 SD. At 4 of the 5 refits the fitted mean is 3.202, the upper bound, with the prior at 7.0 to 7.8. Unbounded, it fits 6.6 to 7.7.
  - Ackley D=6: 37/123 rebuilds outside, by up to 26 SD; the mean is pinned at the lower bound 9.964 while the prior centre is 3.3–4.3.
  - Rosenbrock shifted by 4, D=3: bounds [1753, 1.43e5], local y ≤ 53, fitted mean 1753. Unbounded, the mean follows the data (28.96 → 31.6).
  - Whole runs, unbounded (MATLAB) against the port: sphere 9.4e-8 against 1.55e-6; Ackley 0.0138 against 0.0194; shifted Rosenbrock 2.9e-7 in 122 evaluations and 1.8 s against 1.85e-5 in 152 evaluations and 55 s.
- **History.** Confirmed as the comparison states. `c7c88ab` had `(-np.inf, np.inf)` and prior `(0., 1.)`. `9037851` introduced the gpyreg bounds. Before `ab4dded` the re-centred mean prior was computed and not written (at `ab4dded~1` there is no `gp_priors["mean_const"] =`), so the prior stayed inside the bounds. Commit `2210046` (#70) already records this in its message. It is not on the sheet, and KD-B6-2 says nothing about the bounds.
- **Where I disagree.** C-F1 says predictions away from the data are "too high". That holds on Ackley and Rosenbrock. On the sphere early in the run the mean is pinned at the upper bound, below the prior, so the reversion level is lower than intended.
- **Recommended disposition: fix.**
  - Set the mean's bounds to (−∞, ∞) as MATLAB does, in `_gp_hyp:962`. This also takes away F2's route.
  - It changes default runs, so it needs the population comparison at default (the Ackley D6 and sphere D10 configurations reach it, according to `2210046`).

### I-F2 (= C-F1(c)). Underflow of the prior's mass inside its bounds
- **Lines.** gpyreg `gaussian_process.py:2045-2107` computes the mass as a difference of cdf or sf values in linear space; `2170` takes `log_norm`; `2354` computes `lp -= log_norm`.
- **Check.** `v_f2_underflow.py`: the sf difference is 4.6e-308 at z = 37.5 and 0.000e+00 at z = 38, while logsf is still finite (−726.56). With a zero constant, "log posterior at hyp0: inf".
- **In a default run.** `v_f2_counterfactual.py`, shifted Rosenbrock D=3: each of the 6 refits with a zero constant was redone on copies (same data, same generator state), with a zero constant replaced by 1.

  | refit | time, port | time, constant 1 | max \|Δhyp\| |
  |---|---|---|---|
  | 0 | 7.60 s | 0.05 s | 0.24 |
  | 1 | 18.83 s | 0.05 s | 11.2 |
  | 2 | 3.91 s | 0.04 s | 3.0 |
  | 3 | 0.30 s | 0.01 s | 0.30 |
  | 4 | 11.50 s | 0.01 s | 2.3 |
  | 5 | 11.57 s | 0.01 s | 0.18 |

  - The objective with a finite constant is sometimes better at the port's point (refit 1: 1492 against 3001) and sometimes worse (refit 2: 47430 against 47291). Neither is reliably the MAP.
  - The mean at the training points moved by at most 1e-3.
- **Toy fit.** On a noisy toy the two fits coincided (42 iterations each), so the slowdown depends on the case.
- **Design draws.** Both reports are right, each in its own case. A prior below the bounds gives `isf(0) = +∞`, a prior above them gives `ppf(0) = −∞` (`f_min_fill.py:205-217`), and the draws are not clipped.
- **Reach.** Also at default in Ackley D=4: the `lp -= masks["log_norm"]` warning appeared in the B6-R1 runs.
- **Recommended disposition: fix,** in two places:
  - In gpyreg: compute the log mass in log space (logsf/logcdf). That is a gpyreg release, so it needs that release's gate.
  - In PyBADS: F1's fix removes the route at default. Once the mean is unbounded, the other priors I saw lie within a few SD of their bounds.

### I-F3 = C-F2. `_get_random_samples_from_priors_`
- **Lines.** `717-719` exponentiate the centre and the SD of every block whose name contains "log"; `713` indexes a `None` prior. The callers are `193`, `402` and `655`.
- **MATLAB.** `gppriorrnd.m:75` → `priorGauss`: `sqrt(s2)*randn+mu` in log units; an empty prior keeps the current value (lines 67-68).
- **Check.** `v_f3_prior_samples.py`, 20000 draws through the function itself:
  - Draws of log sf, prior (3.91, 2.00): (49.939, 7.490), matching N(e^3.91, e^2) = (49.899, 7.389).
  - Draws of log sn, prior (−3.45, 1): (0.036, 2.702).
  - The mean is drawn correctly: (4.995, 0.499).
  - "share of log sf draws above its upper bound 20.72: 0.99995".
  - `negquad`: "raises TypeError - 'NoneType' object is not subscriptable". A whole `negquad` run (Rosenbrock D=2, seed 1) stops with the same error.
- **Reach.** `v_reach.py`: the sampler is reached in every default run:

  | run | draws | caller |
  |---|---|---|
  | sphere D=2 | 1 | `_robust_gp_fit_` |
  | Rosenbrock D=2 s1 | 7 | `_robust_gp_fit_` |
  | Rosenbrock D=4 | 5 | `_robust_gp_fit_` |
  | Ackley D=6 | 6 | the second fit in `local_gp_fitting` |

  Log sf draws went as high as 1396.
- **History.** The function was introduced in `9037851`. At `c7c88ab`, `get_random_sample_from_prior` was a slice-sampler draw from the posterior, so the port never drew from the prior.
- **Recommended disposition: fix.**
  - Draw N(μ, σ) in log units, and leave a block without a prior at its current value.
  - It changes default runs whenever a fit is retried or a second fit is made, so it needs the population comparison at default, on targets that trigger retries (Rosenbrock-like). Confirm the retries actually happen in the benchmark configurations.

### I-F4 = C-F3. Noise inflation in gpyreg's Cholesky retries
- **Lines.** gpyreg `gaussian_process.py:3584-3666` multiplies the noise by 10 per failed attempt, up to 10 attempts, keeps `sn2_mult`, and `predict` uses it (1321).
- **MATLAB.** `bads.m:272` sets `CholAttempts = 0`, and `infExact_fastrobust.m:80` raises `if kmax <= 0` on the first failure. `gpHyperOptimize.m:73-176` catches the error and restarts from a prior draw with a noise nudge. `gpupdate.m:347-350` sets `post = []`.
- **Check.** `v_reach.py`, default runs:

  | run | factorizations | inflated | failed after 10 attempts | inflated states handed on |
  |---|---|---|---|---|
  | Rosenbrock D=2 s1 | 1725 | 609 (up to ×1e9) | 7 | 73/116 |
  | Rosenbrock D=4 | 2263 | 555 | 5 | 82/252 (up to ×1e4) |
  | sphere D=2 | 808 | 18 | 1 | 4/53 |
  | Ackley D=6 | 1573 | 2 | 0 | 0/213 |

- **Detail.** `v_f4_detail.py`: "sn2_mult=100: N=46 fitted sn=0.00466, effective sn=0.0466; sf=2.42e+05, std(y)=1.13e+03; uninflated K+sn2 I fails; |mean-y| at the 10 best points: max 0.0103 (their range 0.0123)". The GP no longer resolves the structure near the incumbent. The effect on results was not measured.
- **Recommended disposition: decide the design.** The options are:
  - (a) keep gpyreg's inflation, and document it as a deliberate difference under KD-B6-1;
  - (b) treat `sn2_mult > 1` as a failure, as MATLAB does (restart the fit with a nudge; a posterior handed on with `sn2_mult > 1` gets a rebuild), which needs a gpyreg switch like `chol_attempts` (unread today, KD-B1-5(b));
  - (c) narrow the output-scale/noise bounds that admit singular matrices, which departs from MATLAB.

  Any choice other than (a) moves default runs, so it needs the population comparison at default, plus the gpyreg release gate if gpyreg changes.

### I-F5. The noise upper bound is log SD 5
- **Lines.** `_gp_hyp:915`. MATLAB has the same, `gpdefBads.m:161` `[log(TolFun)-1; 5]`, since 2017.
- **Check.** `v_f5_noise_bound.py`, level 1, target 1e4|x|² plus noise of SD 500 (log 6.21):
  - With `noise_size=500`, the fitted log sn is 5.0 at all 15 refits.
  - With the default `noise_size`, it runs −0.05 to 0.26 for 5 refits and then sits at 5.0.
- **Classification.** Design question, shared with MATLAB. The internal track did not see that MATLAB has the same bound. In PyBADS the prior above the bound is only about 1 SD out, so there is no underflow; the consequence is the same as in MATLAB.
- **Recommended disposition: decide the design.**
  - Keep MATLAB's constant and document it (or warn when noise_size > e^5), or tie the bound to noise_size.
  - A change that depends only on `noise_size` leaves default runs unchanged, so it needs the fingerprint gate, plus a configuration with a large noise_size for the effect.

### I-F6 = C-F4. A target flat on the initial design stops the run
- **Lines.** `_gp_hyp:960-961` (SD = `np.std(hpd_y)` = 0). The rebuild case is `357` (log 0 = −∞). gpyreg's `_write_prior_block` rejects both a zero σ and an infinite μ (`gaussian_process.py:167-182`).
- **Check.** `v_misc.py F6`: a plateau of 1000 over the plausible box, at D=2 and D=3, gives "raises ValueError - The prior of mean_const has a sigma that is zero or negative…".
- **MATLAB.** The definition prior N(0,1) (`gpdefBads.m:171`) cannot fail, and there is no fit at initialization. What its first update does with yrange = 0 and log std = −∞ (prior terms become −∞/NaN inside `gpHyperOptimize`'s try) needs MATLAB.
- **Recommended disposition: fix** (initialization case).
  - Use a positive fallback SD (1, as for `len(hpd_y) ≤ 1`, or gpyreg's `_target_spread`), and guard line 357 by keeping the previous prior, as KD-B6-2 does.
  - It leaves non-degenerate runs unchanged, so it needs the fingerprint gate plus a plateau test.

### I-F7. The effective radius: not a defect
- **Lines.** `504`, `sqrt(alpha*(exp(1/alpha)-1))`; MATLAB `gpupdate.m:317` is identical.
- **Check.** `v_misc.py F7`: for gpyreg's (= GPML's `covRQard`) kernel, "k(sqrt2*r_e)=0.367879 (e^-1=0.367879)" at every log α from −2 to 5.
- **MATLAB's convention.** Its Matérn constants follow the same rule: the root of k = e⁻¹, divided by √2, gives 0.876179713323453 against MATLAB's 0.876179713323485, and 0.918524648109246 against 0.918524648109253. SE gives 1.
- **Why the report is wrong.** The formula is consistent with the kernel gpyreg computes. The report's alternative, k = e^−½ at r_e, is a different convention. The factor 2 it misses is absorbed by the √2.
- **Recommended disposition: correct the record.** Optionally add a comment naming the convention. No gate.

### I-F8. `fit_lik=False`
- **Check.** "raises ValueError - Unknown hyperprior type delta".
- **MATLAB.** It also refuses: `bads.m:466` sets `gplik = log(TolFun)`, and `gpdefBads.m:140` answers `error('Fixed noise not supported.')`.
- **Classification.** Confirmed, inert: both sides refuse, and only the message differs.
- **Recommended disposition: keep and document.** Raise MATLAB's clear message and say "unsupported" in the `.ini` description. Fingerprint gate.

### I-F9. `upper_gp_length_factor`
- **Check.** `v_misc.py F9`: factors 0, 0.05 and 5.0 all give the bounds [[-13.17, -13.17], [3.912, 3.912]]. Lines 908-913 are overwritten unconditionally at 930-933, which has been so since `c7c88ab` (`ab4dded` only added the missing log).
- **Classification.** Confirmed, inert.
- **Sheet.** KD-B1-4 lists the option under "read by code"; it is read, but it has no effect, so it belongs with KD-B1-5.
- **Recommended disposition: correct the record.** Also remove the dead branch or document it. Fingerprint gate.

### I-F10. Mean-function options
- **Lines.** `bads.py:935-955` accepts 12 names; `_meanfun_name_to_mean_function:564-571` builds 3.
- **Check.** "se: raises ValueError - Unknown mean function!", and the same for "negquadse"; "zero" runs. `negquad` crashes at the first retry, through F3.
- **Why `negquad` is a design question.** It is m0 − ½Σ((x−x_m)/ω)², PyVBMC's mean for log densities, which turns the wrong way for a minimizer. `_gp_hyp` gives it no priors.
- **Recommended disposition: fix** the accepted list to the names that can be built, and **decide the design** of `negquad` (give it priors and flip it, or remove it). Fingerprint gate.

### C-F5. A GP fit at initialization
- **Lines.** `bads.py:1185-1205`; `init_and_train_gp`; `get_hpd` takes `round(0.8N)`.
- **MATLAB.** It only defines the GP (`bads.m:465-469`). Its start values are zeros, log NoiseSize and the median of the lowest `ceil(0.8N)` (`gpdefBads.m:164-165`, since `d4fead5`, 2022-10-31; before that the mean started at 0).
- **Check.**
  - The first rebuild was a refit in every trace of `v_f1_mean_bounds.py` (index 0 is a refit), so the initial fit's result reaches the run as the start of that refit.
  - By reading: at default (`uncertain_incumbent=True` on both sides), `_get_target_from_gp_` predicts the first iteration's target under `best_gp_hyp`, the initial fit's hyperparameters (`bads.py:1205`, `2589-2597`). MATLAB uses `fhyp`, the unfitted definition values (`bads.m:469`, `539`).
  - This initialization fit is also what exposes F6.
- **Effect.** Not measured.
- **Recommended disposition: decide the design** (document it as deliberate, or follow MATLAB). A change needs the population comparison at default.

### C-F6. `gp_cov_prior`
- **Check.** `v_misc.py CF6`: "ard: ran; distinct length-scale prior centres over rebuilds: 1 e.g. [(-1.0, -1.0)]", and the same for "foo". 'iso' gave 12 distinct centres.
- **MATLAB.** `gpdefBads.m:254-274` implements 'ard' and errors on anything else.
- **Recommended disposition: fix.** Port 'ard' and raise on unknown values. That needs the fingerprint gate for default runs ('iso'), plus a population run with `gp_cov_prior='ard'`.

### C-F7. ddof
- **Lines.** `357` uses `np.std(gp.y)` (ddof 0); MATLAB `gpdefBads.m:293` uses `std` (N−1).
- **Magnitude.** The shift is ½·log(N/(N−1)): 0.0101 at N=50, 0.0046 at N=110, against a prior SD of 2.
- **Recommended disposition: fix** (ddof=1). It moves default runs at rounding level, so it needs the population comparison; it can go with B6-R1.

## 3. Met while verifying (unverified unless stated)

1. **Noise nudge (B5, read only, not reproduced).** `_robust_gp_fit_:668-675` raises the noise lower bound by the cumulative nudge (+1, +2, +3, …). MATLAB raises it by `nudge(2) = 0` (`gpHyperOptimize.m:161-166`). After 5 consecutive failures the lower bound is −7.91 + 15, above the upper bound 5. gpyreg's `set_bounds` then raises `ValueError` (its docstring, `gaussian_process.py:338-339`), which KD-B5-3 lets through. This is the comparison reviewer's B5 note.
2. **Doubled length-scale sum (inert, read only).** `local_gp_fitting:458` has `len_scale += len_scale + np.exp(...)`, which doubles the running sum for more than one hyperparameter sample. It is inert with PyBADS's single sample (KD-B5-4).
3. **Sheet coverage (read).** The sheet has no entry for the mean bounds (F1) or the noise-prior update (B6-R1). KD-B6-2's "Otherwise the re-centred prior follows MATLAB" is true of the centre and width, but not of the bounds under them. No finding contradicts an entry outright. KD-B1-4 groups `upper_gp_length_factor` as a read option; see I-F9.
