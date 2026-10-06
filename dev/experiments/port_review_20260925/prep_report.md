<!-- Written by the preparatory agent of the port review (wave 0), reading PyBADS at ab4dded and MATLAB BADS at 74919c0; saved verbatim from its final message on 2026-09-25 (the three parts of that message are kept as known_differences.md, counterpart_map.md and prep_report.md). -->

# Preparatory report

## What was produced

The sheet above has 30 entries: 9 in B1, 3 in B2, 4 in B3, 2 in B4, 5 in B5, 4 in B6, 3 in B7, 1 in S and 1 for the tests, plus 8 claims that did not check out. I verified each entry against the Python code at `ab4dded` and the MATLAB code at `74919c0`. The map has one row for each of the 114 `*.m` files outside `gpml-matlab-v3.6-2015-07-07/`, plus the `bads.m` sub-table. I wrote no tracked file. Scratch output is in the scratchpad directory under `review/prep/`: `mfiles.txt`, `mfuncs.txt`, `grep_map.txt`, `mrefs.txt`, `opt_names.txt`, `opt_reads.txt`, `mapped.txt`, `pyonly.txt`.

## Mismatches with the slice table, and proposed corrections

1. **`utils/gppredcheck.m`** is in B4, but its counterpart is inlined in `_is_gp_refit_time_`, which is in B5. Move it to B5, together with **`utils/swtest.m`**, which a default run reaches and which is substituted by `scipy.stats.shapiro`.
2. **`search/grid_functions.py`** is in B7 "where B3 does not read it". The only function that remains is `grid_units`, whose counterpart `utils/gridunits.m` is in B1. Put `force_to_grid` and `udist` in B3, put `grid_units` in B1, and drop the file from B7.
3. **`search/searchWCM.m`** (B3) is not the counterpart of `ESSearchWM`: that is `searchES` method 1. It is unported and unused by MATLAB's defaults. Mark it unported.
4. **Files the table lists as counterparts that no MATLAB code calls:**
   - `utils/private/fminbayes.m` (B6);
   - `utils/gpHyperSample.m` (B5; not the counterpart of `_get_samples_from_slice_sampler_`);
   - `gpdef/private/gpdefStationaryNew.m`, `gpml_fast/exact_inference_*.m` and `gpml_fast/infExact_fast.m` (B6; reached only through `gpdefStationaryNew.m` or tests);
   - `private/covmatadapt.m` (B4);
   - `poll/private/*.m` (B4);
   - `utils/xCheck.m` (B3);
   - `search/private/*.m` (B3).

   Mark them all "unused in MATLAB".
5. **`utils/update_posterior.m`** (B5) has no Python counterpart, since the rank-1 path is not taken (KD-B5-1). The comparison is against its absence.
6. **Files a default run reaches that the table does not mention:**
   - `utils/evalbool.m` → B1 (no counterpart);
   - `utils/maskindex.m` → B1 (`variables_transformer.maskindex`);
   - `utils/swtest.m` → B5;
   - `utils/prctile1.m` → B6;
   - `gpml_fast/infPrior_fast.m` and `gpml_fast/sq_dist_fast.m` → B6;
   - `init/private/i4_bit_hi1.m` and `i4_bit_lo0.m` → B7 (the pattern "i4_sobol*" misses them).

   B6's "infExact_fast*" matches the unused `infExact_fast.m` and misses the default `infPrior_fast.m`. GPML library functions reached at default (`priorGauss`, `meanConst`, `sq_dist`, `solve_chol`) are out of scope as library, but B6 should know that they are the reference for the priors and the mean.
7. **Python code that no slice covers, or covers without a counterpart:**
   - `pybads/__init__.py` and the subpackage `__init__.py` files (exports);
   - `bads/option_configs/options_confs.py` (a path helper; B1 by folder);
   - `option_configs/test_options.ini` and `test_options2.ini` (test fixtures shipped inside the package; not BADS options).

   Covered, but without a MATLAB counterpart:
   - `gaussian_process_train._estimate_noise_` (B5; from PyVBMC; its result `sn2hpd` is unused);
   - `_get_fevals_data` and `_get_gp_training_options` (B5; they feed gpyreg's fit, and MATLAB fits no GP at initialization);
   - `stats/get_hpd.py` (B7; from PyVBMC; used only by `_gp_hyp`, so B6 is its natural home);
   - `decorators/handle_0D_1D_input.py` (B1; imported at `variables_transformer.py:3` but never applied, so it is out of scope in effect).

   `ESSearchCMA` (B3) is unreachable.
8. **The "(unported?)" entries, resolved:**
   - B2: `fixedbads.m` is unported (PyBADS refuses fixed variables); `scatterplot.m` is unported (plotting is out of scope).
   - B3: every other search and acquisition function is unported and unused by MATLAB's defaults. `acqPortfolio.m` is not among them: its 'upd' branch is reached at default and is ported as `update_hedge`.
   - B5: `gpHyperSVGD.m` is unported and unused by the defaults (`gpSamples = 0`).
   - B7: `initLHS.m` is unported (it is also `initSobol`'s fallback); `initRand.m` is unported and unused by the defaults.
9. **Not in the table and out of scope:** `bads_examples.m`, `hetsphere.m`, `rosenbrocks.m`, `install.m`, `private/checklist.m`, `private/runtest.m` (the source of the tests' problems), `gpml_fast/test_*.m`, and the other kernels (`covSEard_fast.m`, `covMaternard_fast.m`, `covPPERard*.m`, `ard_ratquad_covariance_fast.m`).
10. **The plan's M table** says `a21f2ee` moved `Ninit` "from `nvars` to `10 + nvars`". `019f0b4` set it back to `nvars` (`bads.m:198` at `74919c0`; `git log -L198,198:bads.m`). Note the revert in the `a21f2ee` and `019f0b4` rows.

## Left off the sheet, and why

- **Survey rows still open that the other sources touch:**
  - the recovery with the previous hyperparameters on the new training set (the plan's Open Question 4 keeps it "to the bug hunt");
  - the `output_fcn` TODO at `bads.py:2345` (its survey row is "seen");
  - the iteration count from 0: `AGENTS.md` at `ab4dded` states it as fact, but the survey row is open;
  - `_re_evaluate_history_` using the stored GPs;
  - `optim_state["plb"]`/`["pub"]` swapped;
  - the Sobol seed from `u0`;
  - the target's recomputation under `hyp_best`;
  - the search ranking by the previous GP after a failed rebuild;
  - `min_iter`, `min_fun_evals`, `success`, `exit_flag` (their row is "not looked at").
- **Differences that no record decides:**
  - `fit_lik=False`: both sides refuse it (`gpdefBads.m:139-140`, "Fixed noise not supported"; PyBADS fails in gpyreg with "Unknown hyperprior type delta"). It is recorded only as an observed failure.
  - `swtest.m` → `scipy.stats.shapiro`: MATLAB's `swtest` switches to Shapiro–Francia for leptokurtic samples.
  - `cache_size`: MATLAB's 1e4 caps a circular buffer; PyBADS's 500 is the initial size of a log that grows.
  - `_bounds_check_` does more than `boundscheck.m` (plausible bounds estimated from an `x0` set, effective bounds, `x0` moved inside them).
  - A user value of `None` versus MATLAB's empty → default.
- **Seen in passing while checking entries.** These are neither verified as findings nor in the survey as far as I read it. They are listed so the orchestrator can check whether the reviewers find them:
  - (a) `search_factor_min` is unread, where MATLAB clamps the search factor on a failed search at default options (`bads.m:1366` against `bads.py:2635-2645`).
  - (b) `tol_noise` is `np.spacing(1.0)*tol_fun` (`advanced_bads_options.ini:13`), where MATLAB has `sqrt(eps)*TolFun` (`bads.m:195`).
  - (c) The random `x0` is uniform in the original plausible box (`bads.py:254-259`); MATLAB's is uniform in the transformed box (`setupvars.m:83`).
  - (d) PyBADS fits a GP at initialization on all points (`bads.py:1145-1154`); MATLAB's first fit comes at the first poll.
  - (e) `p_less` takes the `D+1` largest probabilities (`bads.py:2140-2142`), where MATLAB takes `nvars` (`bads.m:869`).
  - (f) The chi-square inverse at `bads.py:2420` lacks MATLAB's factor 2 (`gppredcheck.m:20`).
  - (g) `fun_eval_start` is capped at `max_fun_evals - 1` before the power-of-two rounding (`bads.py:1009-1012`), so the design can exceed that cap.
- **The freeze.** Line numbers are at `ab4dded`. The main checkout has uncommitted changes to `bads.py` (+129/-?), `gaussian_process_train.py`, `optimize_result.py`, both `.ini` files, `AGENTS.md` and `CHANGELOG.md`, and they touch `output_fcn`, the iteration count and `noise_size`. Re-check the sheet's line citations, KD-B1-8 and claims C1/C2 against the freeze revision before waves 1-4.

## Coverage

- **Read in full:**
  - records: `AGENTS.md` (the worktree's, which differs from the main checkout's in two passages); `CHANGELOG.md`; `pybads/bads/README.md`; `pybads/decorators/README.md`, `pybads/stats/README.md`; `dev/plans/gp-update-guards.md`; `dev/plans/tooling-and-rng.md`; `dev/results/2026-09-23-codebase-survey.md`; `dev/results/2026-09-25-gpyreg-1.3.3.md`; `dev/README.md`; `dev/TODO.md` (for exclusions); `dev/experiments/population_ellipsoid_hetero_linux_20260925/README.md`; `population_linux_gpfixes_20260925/README.md`; `docsrc/source/` `index`, `quickstart`, `documentation`, `development` and all `api/*.rst`;
  - Python: `bads.py`, `gaussian_process_train.py`, `function_logger.py`, `constraints_check.py`, `search_hedge.py`, `es_search.py`, `grid_functions.py`, `poll_mads_2n.py`, `acq_fcn_lcb.py`, `init_sobol.py`, `period_check.py`, `rng.py`, `get_hpd.py`, `variables_transformer.py`, `options.py`, `optimize_result.py`, both `.ini` files;
  - MATLAB: `bads.m`, `setupoptions.m`, `setupvars.m`, `boundscheck.m`, `evalinitmesh.m`, `funlogger.m`, `gpupdate.m`, `gpdefBads.m`, `initSobol.m`, `searchES.m`, `searchHedge.m`, `pollMADS2N.m`, `acqPortfolio.m`, `acqLCB.m`, `gppredcheck.m`, `bads_output.m`, `i4_sobol_generate.m`, and the small utilities (`force2grid`, `gridunits`, `origunits`, `udist`, `uCheck`, `xCheck`, `periodCheck`, `ucov`, `ugdist`, `maskindex`, `weightedsum`, `prctile1`, `evalbool`).
- **Read in part:** `gpHyperOptimize.m` (1-330), `gppred.m` (30-95), `mygp.m` (110-130), `fixedbads.m` (1-60), `infExact_fastrobust.m` and `covRQard_fast.m` (by grep), `searchWCM.m`, `initLHS.m`, `initRand.m`, `acqHedge.m` (heads).
- **Help lines and call sites only (grep):** every other MATLAB file, the other `dev/experiments/*/README.md` (for decisions), the example notebooks (for "MATLAB"), and gpyreg (the prior parameterization and the `GaussianNoise` flags).
- **Not read:** `iteration_history.py`, `kde1d.py`, `kldiv_mvn.py`, `timer.py`, `bads_dump.py`, `__main__.py`, `function_examples.py` (beyond its first functions), `docsrc` `installation`/`about_us`/`examples`.
