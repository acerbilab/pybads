<!-- Written by the preparatory agent of the port review (wave 0), reading PyBADS at ab4dded and MATLAB BADS at 74919c0; saved verbatim from its final message on 2026-09-25 (the three parts of that message are kept as known_differences.md, counterpart_map.md and prep_report.md). Its Python line citations were carried to 95da7f1 on 2026-09-26 (refresh_citations.py), and KD-B1-8 and the path conventions edited to match; the rest is the agent's text. -->

# Known differences between PyBADS and MATLAB BADS

**Path conventions.** Python paths are relative to the PyBADS repository root, at commit `95da7f1`, the review's freeze (written at `ab4dded`; the citations were carried to `95da7f1` by `refresh_citations.py --base ab4dded`, and the entries that #71 touched were read again). MATLAB paths are relative to the root of MATLAB BADS (`acerbilab/bads`), at commit `74919c0` (v1.1.3). gpyreg paths are relative to gpyreg at v1.3.3 (`98ab5a4`). Line ranges are inclusive. "MATLAB" means MATLAB BADS at `74919c0`, "default" means the default options of either side, and "reached at default" means that a default run executes the code.

**What this sheet is.** It lists only differences that are *settled and deliberate*: the ones a record decided, or that the code marks as a deliberate departure. A reviewer who finds one of them, as described here, does not report it as new. Each entry is a claim a reviewer may challenge. If the code differs from the description, or the stated reason does not hold, that is a finding. Open findings, undecided questions, suspected defects and items marked "not yet fixed" in earlier records are left off on purpose. **If a difference is not on this sheet, that does not mean the code is correct or matches MATLAB.** Where an entry settles only part of a behaviour, it says which part stays open. The records named under "Why" are provenance. A reviewer can check an entry without them.

Kinds: deliberate change | unported feature | removed feature | substituted library | Python-only feature.

## B1: setup, options, bounds, transform, result

**KD-B1-1. Every random draw comes from one `numpy.random.Generator`, `bads.rng`, created from the `random_seed` option; MATLAB draws from its global stream**
- Python: `pybads/rng.py:6-26` (`get_rng`); `pybads/bads/bads.py:215-216`, `961-975` (`_init_rng_`), `254-259` (random `x0`), `1771`, `2200` (fallback indices); `pybads/poll/poll_mads_2n.py:25`, `30`, `34`; `pybads/search/search_hedge.py:71`, `75`; `pybads/search/es_search.py:132`, `172`, `212`; `pybads/init_functions/init_sobol.py:64`; `pybads/bads/gaussian_process_train.py:150-152`, `167-201`, `409`, `607-609`, `722`, `750-758` (rng passed to `gp.fit` and `SliceSampler`); `pybads/bads/optimize_result.py:151`; `pybads/bads/option_configs/basic_bads_options.ini:22-23`.
- MATLAB: `private/setupvars.m:83`; `bads.m:589`, `857`; `search/searchES.m:117`, `168`, `201`; `search/searchHedge.m:48`, `50`; `poll/pollMADS2N.m:10`, `14`, `17`; `init/initSobol.m:14`; `private/gpupdate.m:374`, `392`; `utils/gppriorrnd.m:75`; `private/bads_output.m:25` (`output.rngstate = rng`).
- What differs: MATLAB has no seed option. It draws with `rand`, `randn`, `randi` and `randperm` from the global stream and reports the stream state. PyBADS builds one generator when the `BADS` object is created, passes it to every draw site, and never touches NumPy's global stream. The one exception is `random_seed=None`: the generator is then seeded from four draws of the global stream. The result reports `random_seed`, not a stream state. No draw is meant to reproduce MATLAB's numbers. This entry settles where the draws come from. It does not settle what is drawn or from which distribution; those remain for each slice to compare.
- Why: CHANGELOG `[1.1.0]`, "Added: Seeded runs through a random generator" and the Upgrading lines; `dev/plans/tooling-and-rng.md`, Phase 8 "Contract" and Decisions ("One generator per run", "`random_seed` no longer seeds NumPy's global stream"); `AGENTS.md`, "Randomness goes through one numpy.random.Generator".
- Kind: deliberate change (and a Python-only option, `random_seed`).
- Slice: B1 (the draw sites belong to every slice).

**KD-B1-2. Options live in two `.ini` files, with snake_case names, several of them renamed**
- Python: `pybads/bads/option_configs/basic_bads_options.ini`, `advanced_bads_options.ini`; `pybads/bads/bads.py:195-210`.
- MATLAB: `bads.m:149-161` (basic `defopts`), `187-290` (advanced).
- What differs: the option set is split between a basic and an advanced `.ini` file. MATLAB CamelCase names become snake_case. Renames that are not a plain case change: `Ninit`→`fun_eval_start`, `Ndata`→`n_train_max`, `MinNdata`→`n_train_min`, `BufferNdata`→`buffer_ntrain`, `MeshOverflowsWarning`→`mesh_overflow_warning`, `Nsearch`→`n_search`, `Nsearchiter`→`n_search_iter`, `Nbasis`→`n_basis`, `TolPoI`→`tol_poi`, `ESbeta`/`ESstart`→`es_beta`/`es_start`, `gpSVGDiters`→`gp_svd_iters`, `NormAlphaLevel`→`normalpha_level`, `InitFcn`→`init_fun`, `gpdefFcn`→`gp_def_fcn`, and `gp*`→`gp_*`. `PeriodicVars` and `OutputFcn` are basic in MATLAB and advanced in PyBADS. Values also change form: function handles become strings (`init_fun = "init_sobol"`, `poll_method = 'poll_mads_2n'`, `search_acq_fcn = ('acq_LCB', None)`); `'on'`/`'off'`/`'yes'`/`'no'` become Python booleans; `nvars` becomes `D`. `search_method = [('ES-wcm',1), ('ES-ell',1)]` lists the members of the search hedge, and the hedge itself is always used. MATLAB names `@searchHedge` explicitly (`bads.m:239`). Default values are not settled by this entry: B1 compares them one by one.
- Why: the `.ini` files are the documented option interface (`docsrc/source/api/options/bads_options.rst` includes them verbatim); `AGENTS.md`, "Options are layered".
- Kind: deliberate change (interface).
- Slice: B1.

**KD-B1-3. User option values are used verbatim, `.ini` expressions are evaluated with `D`, and an unknown option name raises**
- Python: `pybads/bads/options.py:31-52` (user options stored as given), `91-118` (`.ini` values `eval`'d with `D` bound through `exec`; user-set keys skipped), `120-148` (unknown names raise `ValueError`); `pybads/bads/bads.py:198-210`.
- MATLAB: `private/setupoptions.m:21-50` (string values of the listed fields, the user's included, are `eval`'d, with `evalbool` as fallback); unknown fields are kept and ignored.
- What differs: a user's string such as `"200*D"` stays a string in PyBADS, where MATLAB evaluates `'200*nvars'`. A misspelt option raises in PyBADS and is ignored in MATLAB. Not settled: what PyBADS does with a user value of `None`, where MATLAB replaces an empty field by the default (`setupoptions.m:5-9`). No record decides that.
- Why: `AGENTS.md`, "Options are layered … the dict is used verbatim, so a user's `"200*D"` stays a string … An unknown name raises `ValueError`".
- Kind: deliberate change.
- Slice: B1.

**KD-B1-4. Options that exist on one side only**
- Python: the two `.ini` files.
- MATLAB: `bads.m:161` (`OptimToolbox`), `188` (`Debug`), `189` (`TrueMinX`).
- What differs:
  - *MATLAB only:* `OptimToolbox`, which chooses between `fmincon` and `minimizebnd` for the GP hyperparameters (`utils/gpHyperOptimize.m:235-283`). PyBADS has no counterpart because its optimizer is gpyreg's (KD-B6-1). `Debug` and `TrueMinX` only print or plot (`bads.m:1002-1011`, `private/gpupdate.m:61-63`, `351-353`, `private/scatterplot.m`).
  - *PyBADS only, and read by code:* `random_seed` (KD-B1-1); `stobads`, `opp_stobads`, `stobads_frame_size_scaling_power` (KD-S-1); `gp_mean_fun` (`'const'` is MATLAB's fixed `@meanConst`, `gpdef/gpdefBads.m:167`; `'zero'` and `'negquad'` are PyBADS's own); `gp_train_n_init`, `gp_train_n_init_final`, `gp_train_init_method`, `gp_tol_opt`, `hpd_frac`, `upper_gp_length_factor`, `gp_quadratic_mean_bound`, `tol_sd`, `use_slice_sampler`, `gp_hyp_sampler`, `hyp_run_weight`, `fun_evals_per_iter`, `noise_shaping` (all options of the gpyreg-based GP layer or taken from PyVBMC's); `init_mesh_size_integer` (default 0, which is MATLAB's fixed `MeshSizeInteger = 0`, `private/setupvars.m:41`); `f_vals`; `hessian_update`, `hessian_method` (read only by a no-op branch, `bads.py:1859-1864`, under the `.ini` heading "Adaptive basis (unsupported)"; MATLAB v1.1.3 has no such option).
  - *PyBADS only, and read by no code:* see KD-B1-5.
  - This entry settles only that these options exist on one side. Their effects are not settled: `gp_train_*`, `gp_tol_opt` and `hpd_frac` act at default options and are compared under B5/B6.
- Why: `AGENTS.md`, "Many options do nothing. Some are PyVBMC or MATLAB leftovers"; KD-B1-1, KD-S-1 and KD-B6-1 for the named groups.
- Kind: removed feature (MATLAB-only options); Python-only feature (the others).
- Slice: B1.

**KD-B1-5. Options that are parsed and have no effect**
- Python: the two `.ini` files. None of the names below is read in `pybads/` outside `testing/`, except where a no-op read is cited.
- MATLAB: as listed per group.
- What differs:
  - (a) *No reads on either side:* `skip_poll` (`SkipPoll`), `search_improve_frac` (`SearchImproveFrac`), `gp_cluster` (`gpCluster`). MATLAB reads none of these either.
  - (b) *PyBADS hard-codes MATLAB's default choice, so the option does nothing:* `poll_method` (always `poll_mads_2n`, KD-B4-1), `poll_acq_fcn` (always LCB, KD-B3-2), `gp_def_fcn` (always the RQ ARD kernel, KD-B6-1), `gp_method` (always nearest neighbours), `chol_attempts` (the Cholesky factorization is gpyreg's, KD-B6-1).
  - (c) *MATLAB reads the counterpart only away from its defaults:* `n_basis` (read only by `poll/private/pollBMADS2N.m`, which nothing calls), `gp_samples` and `gp_svd_iters` (KD-B5-4), `rotate_gp` (MATLAB also marks it unsupported, `gpdef/gpdefBads.m:105-108`).
  - (d) *No MATLAB counterpart; PyVBMC or porting leftovers:* `gp_cov_fun` (overridden by `optim_state["gp_cov_fun"] = 1`, `bads.py:917`), `diagnostics`, `hessian_alternate`, `cov_sample_thresh`, `gp_sample_widths`, `weighted_hyp_cov`, `tol_cov_weight`, `gp_sample_thin`, `stable_gp_sampling`, `gp_tol_optmcmc`, `nsgp_max`, `nsgp_maxwarmup`, `nsgp_maxmain`, `stable_gp_samples`, `gp_tol_optactive`, `gp_tol_optmcmcactive`, `tol_gp_var`, `tol_gp_varmcmc`, `active_sample_gp_update`, `sample_extra_vp_means`, `integrate_gp_mean`, `tol_skl`, `tol_stable_warmup`, `variational_sampler`, `kl_gauss`, `k_warmup`, `stable_gp_vpk`, `max_repeated_observations`, `repeated_acq_discount`, `sgd_step_size`, `rank_criterion`, `ns_search`, `gp_stochastic_step_size`, `heavy_tail_search_frac`, `mvn_search_frac`, `hpd_search_frac`, `box_search_frac`, `search_cache_frac`, `empirical_gp_prior`, `tol_gp_noise`, `gp_length_prior_mean`, `gp_length_prior_std`, `init_design`, `bandwidth`, `out_warp_thresh_base`, `out_warp_thresh_mult`, `out_warp_thresh_tol`, `temperature`, `separate_search_gp`, `noise_shaping_threshold`, `noise_shaping_factor`, `acq_hedge_iter_window`, `acqhedge_decay`, `active_search_bound`, `tol_bound_x`, `recompute_lcb_max`, `double_gp`, `warp_every_iters`, `incremental_warp_delay`, `warp_tol_reliability`, `warp_proto_scaling`, `warp_cov_reg`, `warp_proto_corr_thresh`.
  - (e) *Read only by a branch that does nothing or refuses:* `plot` (KD-B2-2), `restarts` (KD-B2-1), `search_optimize` (KD-B3-4), `acq_hedge` (KD-B3-3), `fitness_shaping` (KD-B5-5), `hessian_update`/`hessian_method` (KD-B1-4), `warp_func` ≠ 0 (KD-B6-4), `periodic_vars` (KD-B1-6), `init_fun` other than `"init_sobol"` (KD-B7-2).
  - This list covers only the options settled as having no effect. It is not a list of every option that no code reads.
- Why: `AGENTS.md`, "Many options do nothing … Grep for an option's reads before relying on it"; the entries cited in (b), (c) and (e).
- Kind: removed feature ((b), (c)); Python-only feature ((d)).
- Slice: B1.

**KD-B1-6. Periodic variables are not supported**
- Python: `pybads/bads/bads.py:616-620` (a non-`None` `periodic_vars` raises `ValueError`); `pybads/utils/period_check.py:4-6` (a stub that returns its input); `pybads/bads/gaussian_process_train.py:353` (TODO); `pybads/search/es_search.py:142` (TODO). The periodic branches of `udist`, `ucov` and `_init_optim_state_` (`bads.py:625-629`, `727-742`) are unreachable.
- MATLAB: `bads.m:152` (`PeriodicVars`); `private/setupvars.m:49-57`, `107-116`; `utils/periodCheck.m`; `gpdef/gpdefBads.m:58-81`, `277-284`; `utils/udist.m`; `utils/ucov.m`; `gpml_fast/covPPERard_fast.m`.
- What differs: MATLAB wraps periodic variables into their range and uses a periodic kernel. PyBADS refuses them.
- Why: `pybads/bads/README.md` ("Support for periodic variables"); `dev/TODO.md`, "Porting gaps"; the error message and TODO comments above.
- Kind: unported feature.
- Slice: B1 (option); B7 (`period_check`).

**KD-B1-7. Fixed variables are refused; MATLAB removes them and runs a smaller problem**
- Python: `pybads/bads/bads.py:413-423`.
- MATLAB: `private/boundscheck.m:39-40`; `bads.m:351-382`, `1480-1488` (`expandvars`); `private/fixedbads.m`.
- What differs: a variable whose bounds are all equal makes PyBADS raise `ValueError`. MATLAB fixes it and optimizes the others.
- Why: the code comment "Fixed variables (all bounds equal) are not supported" and the error message.
- Kind: unported feature.
- Slice: B1 (the check); `fixedbads.m` and `expandvars` are in B2's MATLAB list.

**KD-B1-8. The result is an `OptimizeResult` dict, not MATLAB's six outputs**
- Python: `pybads/bads/optimize_result.py:8-162`; `pybads/bads/bads.py:1638-1641`.
- MATLAB: `bads.m:1` (`[x,fval,exitflag,output,optimState,gpstruct]`), `1185-1194`; `private/bads_output.m`.
- What differs: PyBADS returns a scipy-style dict (`x`, `x0`, `fval`, `fsd`, `yval_vec`, `ysd_vec`, `func_count`, `iterations`, `mesh_size`, `message`, `target_type`, `problem_type`, `total_time`, `overhead`, `random_seed`, `algorithm`, `version`, `fun`, `non_box_cons`, `success`, `status`). The `BADS` object keeps the run's state. It has no `exitflag` output, no `rngstate` (see KD-B1-1) and no `maxconstraint`. Not settled: the meaning of `success`. `iterations` counts as MATLAB's `output.iterations` does, from 1, since #71. `yval_vec` is `None` for a deterministic run and with `noise_final_samples = 0`, where MATLAB returns the incumbent's observation (`bads.m:1134`, `bads_output.m:37`), as the docstring of `OptimizeResult` documents (ledger of wave 0, W0-4).
- Why: the class docstring ("based on `scipy.optimize.OptimizeResult`"); `docsrc/source/api/classes/optimize_result.rst`; `docsrc/source/quickstart.rst`.
- Kind: deliberate change (interface).
- Slice: B1.

## B2: main loop, termination, noisy re-evaluation, final estimate

**KD-B2-1. Restarts are not implemented**
- Python: `pybads/bads/bads.py:1241`, `1509-1513` (`if self.restarts > 0: pass`).
- MATLAB: `bads.m:201`, `479`, `1121-1127` ("Multiple starts (deprecated)").
- What differs: with `restarts > 0`, MATLAB resets the mesh and continues after termination. PyBADS stops. Both default to 0.
- Why: the code keeps MATLAB's own label, "Multiple starts (deprecated)".
- Kind: unported feature.
- Slice: B2.

**KD-B2-2. Plotting is not implemented**
- Python: `pybads/bads/bads.py:1386-1388` (`plot == "scatter"`: `pass`), `2389` (TODO: profile plot); `advanced_bads_options.ini:3`.
- MATLAB: `bads.m:187`, `988-1015` (`'profile'`, `utils/landscapeplot.m`), `1054-1057` (`'scatter'`, `private/scatterplot.m`).
- What differs: `plot` has no effect.
- Why: TODO comments; `dev/plans/port-correctness-review.md` puts plotting code out of scope.
- Kind: unported feature.
- Slice: B2 (out of scope as non-numerical).

**KD-B2-3. Messages go through Python logging, to the `BADS` logger**
- Python: `pybads/bads/bads.py:166-167`, `218-226`, `2764-2822`; `pybads/bads/gaussian_process_train.py:19`.
- MATLAB: `bads.m:311-328` (`prnt` levels) and `fprintf` throughout.
- What differs: `display` sets the level of a logger (`"off"`→WARN, `"iter"`→INFO, and `"full"`, which exists only in PyBADS, →DEBUG) instead of choosing which `fprintf` calls run. This entry settles the mechanism. The content and format of the display remain open to comparison.
- Why: CHANGELOG `[Unreleased]`, Fixed, "Messages on the BADS logger".
- Kind: deliberate change.
- Slice: B2.

## B3: search

**KD-B3-1. The search hedge chooses only between ES-wcm and ES-ell; the other search methods are not ported**
- Python: `pybads/search/search_hedge.py:86-119` (dispatch by name; anything else raises "not implemented yet"); `pybads/search/es_search.py:219-297` (`ESSearchWM` = `searchES` method 1, `ESSearchELL` = method 2), `277-284` (`ESSearchCMA`, `searchES` method 5 `'ES-cma+'`, which is unreachable); `pybads/bads/bads.py:1710-1724`.
- MATLAB: `bads.m:239`; `search/searchES.m:3-12`, `39-70` (methods 1-5: `ES-wcm`, `ES-ell`, `ES-eye`, `ES-cov`, `ES-cma+`); `search/searchCMA.m`, `searchCombine.m`, `searchCrossover.m`, `searchGauss.m`, `searchGrid.m`, `searchMax.m`, `searchMaxAcq.m`, `searchNewton.m`, `searchOptim.m`, `searchWCM.m`, `search/private/*.m`.
- What differs: only MATLAB's default search set exists in PyBADS. `ES-eye`, `ES-cov` and the other search functions are absent, and `ES-cma+` cannot be selected.
- Why: `AGENTS.md`, "Extension points are hard-coded" (`ESSearchCMA` unreachable; a new search method needs a subclass, an `elif` and an option entry).
- Kind: unported feature.
- Slice: B3.

**KD-B3-2. Only the LCB acquisition exists, called directly**
- Python: `pybads/bads/bads.py:1754-1758` (search), `2183-2187` (poll); `pybads/search/es_search.py:158-168` (`search_acq_fcn` must be `'acq_LCB'`; TODO "handle other acqs fcns: acqNegEIMin, acqNegPIMi").
- MATLAB: `bads.m:269-270`, `577-578`, `852`; `search/searchES.m:147`, `156-165`; `acq/acqNegEI.m`, `acqNegEQI.m`, `acqNegPI.m`, `acqNegSqEI.m`, `acqRnd.m`, `acq/private/*.m`.
- What differs: `PollAcqFcn` and `SearchAcqFcn` can name other acquisition functions in MATLAB. In PyBADS the poll always uses LCB and the search accepts only LCB. Both default to LCB.
- Why: `AGENTS.md`, "LCB is called directly at the search and poll call sites"; the TODO above.
- Kind: unported feature.
- Slice: B3.

**KD-B3-3. The acquisition hedge (`AcqHedge`) is not implemented**
- Python: `pybads/bads/bads.py:905-906`, `1750`, `1760`, `1908-1910`, `1939`, `2182`, `2189`; `advanced_bads_options.ini:182`.
- MATLAB: `bads.m:271`, `569-573`, `684-686`, `716-719`, `845-848`; `acq/acqPortfolio.m` ('acq' branch; its help line says "(unsupported)"); `acq/acqHedge.m`; `search/searchES.m:139-141` (`error('Hedge not supported here.')`).
- What differs: `acq_hedge=True` does nothing. Both default to off. This entry does not cover the *search* hedge's reward update, which is ported (`ESSearchHedge.update_hedge` ↔ `acqPortfolio.m` 'upd', reached at default).
- Why: code comments "not yet supported (even in Matlab)"; MATLAB's own "(unsupported)" label.
- Kind: unported feature.
- Slice: B3.

**KD-B3-4. Local optimization of the acquisition function (`SearchOptimize`) is not implemented**
- Python: `pybads/bads/bads.py:1776-1778` (TODO; `pass`).
- MATLAB: `bads.m:248`, `596-616` (`fmincon` on the acquisition function).
- What differs: `search_optimize=True` does nothing. Both default to off.
- Why: TODO comment "(generally it does not improve results)", which echoes MATLAB's comment at `bads.m:594-595`.
- Kind: unported feature.
- Slice: B3.

## B4: poll, mesh, incumbent, target

**KD-B4-1. The poll always uses LTMADS (`poll_mads_2n`); the other poll methods are not ported**
- Python: `pybads/bads/bads.py:2073-2079`.
- MATLAB: `bads.m:206`, `791-798` (`feval(options.PollMethod{:}, …)`); `poll/pollGPS2N.m`; `poll/private/pollBADS2N.m`, `pollBMADS2N.m`.
- What differs: `poll_method` is ignored (KD-B1-5). Both default to MADS 2N.
- Why: `AGENTS.md`, "Many options do nothing … `poll_method`".
- Kind: unported feature.
- Slice: B4.

**KD-B4-2. When the posterior under the best iteration's hyperparameters cannot be computed, the target is predicted from the current GP**
- Python: `pybads/bads/bads.py:2594-2605` (`try` around `set_hyperparameters(hyp_best)` and `predict`; on `LinAlgError`, `gp.predict` on the unchanged GP).
- MATLAB: `bads.m:1296-1312` (`UpdateTarget` sets `gptemp.hyp = hyp` and keeps `gptemp.post`, so it never refactorizes and cannot fail at this point).
- What differs: PyBADS has a failure path that MATLAB lacks, and on that path it uses the GP's own hyperparameters and posterior. **Settled:** only this fallback. **Not settled:** the recomputation under `hyp_best` itself, where MATLAB reuses the current posterior. That recomputation is why the call can fail, and it gives different targets at default options.
- Why: `dev/plans/gp-update-guards.md`, Design "Target (call 1)" and Open Question 2.
- Kind: deliberate change.
- Slice: B4.

## B5: GP training set and refit policy

**KD-B5-1. Adding a point recomputes every posterior in full, and a failed add leaves the point out of the GP until the next rebuild**
- Python: `pybads/bads/gaussian_process_train.py:1212-1265` (`add_and_update_gp`: `gp.update(X_new=…, y_new=…, s2_new=…, hyp=…)`; on `LinAlgError` gpyreg restores the GP and `temporary_data["needs_rebuild"]` is set); `pybads/bads/bads.py:1788-1807` (search), `2237-2257` (noisy poll: when the GP did not grow, the poll's estimate is NaN and the point counts as no improvement).
- MATLAB: `private/gpupdate.m:39-83` ('add': tries a rank-1 update with `utils/update_posterior.m` first, except under `SpecifyTargetNoise`; the point is appended to `x` and `y` whatever happens), `340-354` (full recomputation inside `try`; `post = []` on failure); `bads.m:633-641`, `908-924`.
- What differs: there is no rank-1 path. On failure the point is dropped from the GP, not kept beside an empty posterior. It stays in the function logger and enters the GP at the next rebuild. The noisy poll's NaN matches MATLAB's NaN prediction.
- Why: survey (`dev/results/2026-09-23-codebase-survey.md`), candidate row `add_and_update_gp`, status "by design"; `dev/plans/gp-update-guards.md`, Open Questions 1 and 5. `dev/TODO.md` keeps the rank-1 update as an open item for a possible change. Its absence is known, not new.
- Kind: deliberate change.
- Slice: B5.

**KD-B5-2. A failed rebuild restores the GP as it was on entry, marks it, and forces a refit at the next rebuild**
- Python: `pybads/bads/gaussian_process_train.py:261-265` (snapshot), `522-538` (restore; `needs_rebuild` and `needs_refit` set; exit flag -2), `540-541` (markers cleared once a posterior is left on the new set); `pybads/bads/bads.py:1668-1679` (search), `2132-2152` (poll; with `poll_training` off after the first iteration, the forced refit gives way), `2166-2169` (the poll treats the GP as unreliable after its own rebuild fails), `2552-2567` (`_record_gp_refit_`).
- MATLAB: `private/gpupdate.m:340-354` (the new data and the failed rebuild's hyperparameters and `pollscale` stay, with `post = []`); `bads.m:523-536`, `826-839` (rebuild while `post` is empty), `1223-1254` (refit only when `gppredcheck` finds the NaN predictions unreliable and `MinRefitTime` has passed).
- What differs: PyBADS throws away the new data and hyperparameters on failure and refits at the very next rebuild, whatever `min_refit_time` says. The markers in `gp.temporary_data` stand in for MATLAB's empty `post`. **Not settled:** the retry with the previous hyperparameters on the new training set (`gaussian_process_train.py:520-523`), which MATLAB lacks. That retry is still an open question.
- Why: survey, candidate row "`local_gp_fitting`, and `bads.py`, the forced refit", status "by design (… Open Question 7)"; `dev/plans/gp-update-guards.md`, Design and Open Questions 3 and 7.
- Kind: deliberate change.
- Slice: B5.

**KD-B5-3. Only `LinAlgError` is caught at the guarded GP calls; MATLAB's `try` catches any error**
- Python: `pybads/bads/gaussian_process_train.py:515`, `524`, `1255`; `pybads/bads/bads.py:2598`. The same policy applies at the older guards, `gaussian_process_train.py:206` and `621`.
- MATLAB: `private/gpupdate.m:52-64`, `340-354`.
- What differs: a `ValueError` or other error from gpyreg stops a PyBADS run, where MATLAB would catch it. The decision covers the GP-update guards only. MATLAB's other `try`/`catch` sites (`bads.m:567-586` around the acquisition, `1229-1233` around `gppredcheck`; `acq/acqLCB.m:34-38`; `search/searchES.m:138-150`; `utils/gppred.m:44-54`) have no recorded decision.
- Why: `dev/plans/gp-update-guards.md`, Design, "Scope of the catch".
- Kind: deliberate change.
- Slice: B5.

**KD-B5-4. GP hyperparameters are optimized, never sampled: there is one hyperparameter set**
- Python: `gp_samples` and `gp_svd_iters` are unread; `pybads/bads/gaussian_process_train.py:224` ("Missing port: sample for GP for debug"), `362` ("only optimization supported"), `423` ("Matlab uses hyperSVGD when using multiple samples … not implemented"), `991` (`gp_s_N = 0`).
- MATLAB: `bads.m:254` (`gpSamples = 0`); `private/gpupdate.m:411-414`; `utils/gpHyperSVGD.m`; the weighted sums over `hypweight` throughout.
- What differs: with `gpSamples > 0`, MATLAB fits several hyperparameter samples by SVGD. PyBADS ignores the option. At the default, 0, both optimize a single set.
- Why: the code comments above.
- Kind: unported feature.
- Slice: B5.

**KD-B5-5. Fitness shaping is not implemented**
- Python: `pybads/bads/gaussian_process_train.py:275-278` (TODO; `pass`); `pybads/bads/bads.py:1794` (TODO); `advanced_bads_options.ini:311-312`.
- MATLAB: `bads.m:279-280` (under the heading "GP warping parameters (unsupported)"); `private/gpupdate.m:43-47`, `252-256`; `utils/fitnessTransform.m`.
- What differs: `fitness_shaping=True` does nothing. Both default to off.
- Why: TODO comments; MATLAB's own "(unsupported)" heading.
- Kind: unported feature.
- Slice: B5.

## B6: GP model and its gpyreg objects

**KD-B6-1. The GP is a gpyreg `GP` with a hard-wired rational-quadratic ARD kernel, not GPML plus `gpml_fast`**
- Python: `pybads/bads/gaussian_process_train.py:88-109` (GP construction), `797-798` (identifier 1 → `RationalQuadraticARD`), `811-996` (`_gp_hyp`: bounds and priors in gpyreg's units, where a Gaussian prior is `(mean, SD)`), `167-201` and `607-609` (`gp.fit` with the options of `_get_gp_training_options`, `999-1075`), `741` (the private `gp._GP__gp_obj_fun`, on the slice-sampler path); `pybads/bads/bads.py:915-933` (`optim_state["gp_cov_fun"] = 1`; `gp_noisefun` → `GaussianNoise` flags).
- MATLAB: `bads.m:260` (`gpdefFcn = {@gpdefBads,'rq',[1,1]}`); `gpdef/gpdefBads.m` (a GPML struct; `priorGauss` takes `(mean, variance)`; `likGaussHe`; inference `infPrior_fast` + `infExact_fastrobust` with `CholAttempts`, `309-315`); `gpml_fast/covRQard_fast.m`; `private/gpupdate.m:359-419` (`gpfit`: one or two starting points, `optimset('TolFun',0.1,'TolX',1e-4,'MaxFunEval',150)`); `utils/gpHyperOptimize.m` (`fmincon` or `minimizebnd`); `utils/gppred.m`; `utils/mygp.m`.
- What differs: the GP library and every object it involves (hyperparameter vector, priors, bounds, likelihood, inference, optimizer, prediction) are gpyreg's. The kernel cannot be changed (`gp_cov_fun` and `gp_def_fcn` have no effect). It equals MATLAB's default (`'rq'`, ARD). **Settled:** only the substitution and the hard-wired kernel. **Open to comparison:** every hyperparameter, bound and prior in the units each side uses; the optimizer's starting points and tolerances; the Cholesky handling; the inference; and how PyBADS calls gpyreg. That includes the GP fit at initialization (`init_and_train_gp`), where MATLAB only defines the GP (`bads.m:465-469`).
- Why: `AGENTS.md` ("The GP layer is the lab's `gpyreg`"; "`gp_cov_fun` is overridden by a hard-coded rational-quadratic ARD kernel"; "gpyreg internals"); `dev/plans/port-correctness-review.md`, Decisions (gpyreg's internals out of scope; its use in scope; `covRQard_fast.m` as the reference for `RationalQuadraticARD`) and "Two facts about the GP layer".
- Kind: substituted library.
- Slice: B6 (and B5).

**KD-B6-2. A zero range of the training targets keeps the previous width of the GP-mean prior**
- Python: `pybads/bads/gaussian_process_train.py:310-324` (`mean_sd = y_range / 2` only when `y_range > 0`).
- MATLAB: `gpdef/gpdefBads.m:219-222` (variance `yrange.^2/4` whatever `yrange` is).
- What differs: when `gp_mean_range_fun` gives 0, MATLAB sets a zero-variance prior and PyBADS keeps the previous width. Otherwise the re-centred prior follows MATLAB (the fix of `8afbe16`).
- Why: the code comment "A zero range, which MATLAB leaves to fail, keeps the previous width", written with `8afbe16` (survey row for the GP-mean prior, status "fixed in `8afbe16`").
- Kind: deliberate change.
- Slice: B6.

**KD-B6-3. With `gp_fixed_mean`, the GP mean is not fixed**
- Python: `pybads/bads/gaussian_process_train.py:322`, `325-326` (TODO).
- MATLAB: `gpdef/gpdefBads.m:168-172` (a delta prior on the mean), `220-231` (the mean hyperparameter set to `ymean`).
- What differs: with `gp_fixed_mean=True`, PyBADS re-centres a Gaussian prior and keeps its width. MATLAB fixes the mean at `ymean`. Both default to off.
- Why: the TODO comment; survey row for the GP-mean prior ("… which the port leaves as a `TODO`"), status "fixed in `8afbe16`".
- Kind: unported feature.
- Slice: B6.

**KD-B6-4. Warped likelihoods and output warping are unsupported on both sides**
- Python: `pybads/bads/gaussian_process_train.py:308`, `356-363` (with `warp_func` ≠ 0, `sd_y` is never assigned and the first rebuild fails), `868`, `903`, `952-953`, `988` ("Missing port: output warping"); `advanced_bads_options.ini:248` (`warp_func`), `343-353` (unread `warp_*` options).
- MATLAB: `bads.m:279-281`; `gpdef/gpdefBads.m:116-118` (`error('Warped likelihoods not supported at the moment.')`), `210-215`, `287-291`; `warp/*.m`.
- What differs: neither side supports warping. MATLAB refuses it with a message when the GP is defined. PyBADS fails without one at the first rebuild.
- Why: `dev/plans/port-correctness-review.md`, out-of-scope list ("`warp/` (unsupported on both sides)"); the code comments.
- Kind: removed feature.
- Slice: B6 (out of scope).

## B7: function logger, initial design, utilities

**KD-B7-1. The initial design is a scrambled Sobol set from `scipy.stats.qmc.Sobol` with a power-of-two number of points**
- Python: `pybads/init_functions/init_sobol.py:16-21` (docstring), `66-78` (`Sobol(D, seed=seed).random_base2(ceil(log2(fun_eval_start)))`; the comment cites Owen (2020) on keeping Sobol sets to powers of two); `pybads/bads/bads.py:1039-1054`.
- MATLAB: `private/evalinitmesh.m:98-104` (`Ninit` points); `init/initSobol.m:16` (`i4_sobol_generate(nvars,Ninit,seed)`: the unscrambled sequence, and `seed` is a skip index into it); `init/private/i4_sobol*.m`, `i4_bit_*.m`.
- What differs: the generator (scipy's scrambled Sobol, where the seed seeds the scrambling), and the size, `2**ceil(log2(fun_eval_start))` points instead of `Ninit` (see also claim C2). **Not settled:** how the seed is derived from `u0` (the integer parts of its first 11 coordinates, against MATLAB's character codes of `num2str` of the first 10). That derivation is an open candidate.
- Why: the docstring and the Owen comment; `AGENTS.md`, Architecture ("a Sobol initial design of `2**ceil(log2(fun_eval_start))` points").
- Kind: substituted library (and a deliberate change of the design size).
- Slice: B7.

**KD-B7-2. Only the Sobol initial design exists; LHS and uniform designs are not ported**
- Python: `pybads/bads/bads.py:1045-1091` (any other `init_fun` raises "Initialization function not implemented yet").
- MATLAB: `bads.m:199`; `init/initLHS.m`, `init/initRand.m`, `init/private/lhs.m`; `init/initSobol.m:18-21` (Latin hypercube as the fallback when Sobol generation raises).
- What differs: there is no alternative design and no LHS fallback.
- Why: `AGENTS.md`, "the initial design is selected by `init_fun == "init_sobol"`"; the error message.
- Kind: unported feature.
- Slice: B7.

**KD-B7-3. With target noise, a repeated point is merged into its own row, and the merged value is returned**
- Python: `pybads/function_logger/function_logger.py:398-428` (precision-weighted merge into the row that matches in every coordinate; returns the merged value with the new observation's SD).
- MATLAB: `private/funlogger.m:117-129` (each evaluation is a new row, and the call returns the observation itself).
- What differs: at level 2, PyBADS keeps one row per point and returns the merged value. Until the next rebuild, `add_and_update_gp` then adds that value beside the point's earlier row. Returning the observation, as MATLAB does, was tested and not adopted. At levels 0 and 1, a repeat is a new row on both sides.
- Why: survey, candidate row "`function_logger.py`, `__call__` (at `1a21844`, line 193)", status "seen; MATLAB's form tested, not adopted"; `dev/experiments/population_ellipsoid_hetero_linux_20260925/README.md` ("Returning the observation … changes 20 runs and worsens 16 of them", p = 0.0019); CHANGELOG `[1.1.0]` and `[Unreleased]` (the merge, and the row fix of `032dfcb`); `AGENTS.md`, `FunctionLogger` bullet.
- Kind: deliberate change.
- Slice: B7.

## M: MATLAB changes since the port began

No entry of its own. Entries that touch the eight commits: KD-B5-1 (the rank-1 `'add'` of `private/gpupdate.m`, as rewritten in `d4fead5`) and KD-B5-2 (a failed rebuild in `gpupdate.m`). The default `Ninit` that `a21f2ee` changed to `10 + nvars` was set back to `nvars` in `019f0b4` (`bads.m:198`), and PyBADS's `fun_eval_start = D` matches that.

## S: Sto-BADS

**KD-S-1. Sto-BADS is PyBADS's own**
- Python: `pybads/bads/bads.py:163`, `251` (constructor argument `gamma_uncertain_interval`), `212-213`, `1167-1169` (switched off for deterministic targets), `1869-1904` (search), `1985-2028` (`_sto_success_improvement_`), `2279-2328` (poll); `advanced_bads_options.ini:60-63`, `69-70` (`stobads` False, `opp_stobads` True, `stobads_frame_size_scaling_power` 2).
- MATLAB: no counterpart.
- What differs: when `stobads` is on, an optional success rule based on uncertainty intervals, after Sto-MADS (Audet, Dzahini, Kokkolaras and Le Digabel, 2021), replaces the improvement tests of search and poll. It is off by default.
- Why: `dev/plans/port-correctness-review.md`, slice S ("none: Sto-BADS is PyBADS's own"); the option descriptions and the docstring's reference.
- Kind: Python-only feature.
- Slice: S.

## O: third reader

No entry of its own. For this slice, see KD-B3-3 (the search hedge's reward is ported; the acquisition hedge is not), KD-B4-2, KD-B5-1, KD-B5-2 and KD-B6-1.

## Tests (no slice; for test-adequacy notes)

**KD-T-1. The optimization tests use MATLAB's `runtest.m` problems, but with tolerances set from seed sweeps**
- Python: `pybads/testing/bads/test_bads_optimization.py:1-17`.
- MATLAB: `private/runtest.m:11` (`tolerr = [0.1 0.1 1 1]`).
- What differs: each test's tolerance is ten times the largest error over seeds 0-99, rounded, or `runtest.m`'s tolerance when that is lower. `test_sphere_opt` uses `runtest.m`'s constraint and start point.
- Why: survey, sections "Tests that checked less than they appeared to" and "The seed sweep behind the tolerances".
- Kind: deliberate change.
- Slice: none.

## Claims that did not check out

- **C1.** `AGENTS.md` (Architecture) says initialization evaluates `x0` "a second time as a noise test when `uncertainty_handling` is `None`". The code also runs the test with `uncertainty_handling=False`: `False` gives level 0 (`bads.py:893-902`), and the test runs at any level below 1 (`bads.py:997-1005`). A noisy target can then switch a run the user declared deterministic to level 1. MATLAB tests only when the option is empty (`private/evalinitmesh.m:38-50`).
- **C2.** `AGENTS.md` (Architecture) gives the Sobol design as "`2**ceil(log2(fun_eval_start))` points". `init_sobol.py:73-75` raises the exponent by one when that number equals `D`. At default options (`fun_eval_start = D`), a `D` that is a power of two therefore gets `2D` points (for example 2 points at D = 1, 4 at D = 2, 8 at D = 4). No record explains the extra step.
- **C3.** `dev/plans/tooling-and-rng.md` (Phase 8, Contract) says "The Sobol seed keeps its MATLAB derivation from the digits of `u0`". PyBADS takes the integer parts of the first 11 coordinates (`init_sobol.py:55-62`). MATLAB takes the character codes of `num2str` of the first 10 values (`init/initSobol.m:11-12`), and in MATLAB the seed is a skip index into the unscrambled sequence, not a scrambling seed (`init/private/i4_sobol_generate.m`).
- **C4.** The comment at `bads.py:916`, "Squared exponential kernel with separate length scales", sits above `optim_state["gp_cov_fun"] = 1`, which selects `RationalQuadraticARD` (`gaussian_process_train.py:797-798`). That kernel is MATLAB's default (`'rq'`).
- **C5.** `gaussian_process_train.py:176-177`, "Initialize the hyper-params. to zero after the second failure (like in BADS)": the branch runs at the third failure (`training_failures == 3`). MATLAB fits no GP at initialization (`bads.m:465-469` only defines it; its first fit comes at the first poll through `IsRefitTime`), so it has no such retry. Its zero initial covariance hyperparameters (`gpdefBads.m:48`) are starting values, not a fallback.
- **C6.** Several "Missing port" comments name features that MATLAB BADS does not have; they come from PyVBMC's GP code:
  - `gaussian_process_train.py:955` ("we only implement the mean functions that gpyreg supports"; BADS has only `@meanConst`, `gpdefBads.m:167`);
  - `:996` ("meanfun == 14 hyperprior case");
  - `:1173` ("noise_shaping");
  - `:1271` ("intmean part").

  The output-warping ones (`:878`, `:913`, `:962`, `:998`) do name a BADS feature that BADS refuses (KD-B6-4).
- **C7.** Stale MATLAB pointers:
  - `gaussian_process_train.py:267`, "(Matlab: gpTrainingSet)": since `d4fead5` this is `private/gpupdate.m`, method `'nearest'`.
  - `:359`, "Matlab gpdefbads line-code 302": the warped signal-variance prior is at `gpdefBads.m:287-291`.
  - `function_logger.py:289`, "in the original matlab version X and Y get deleted": MATLAB's `'done'` trims `X` and `Y` and removes `U` (`private/funlogger.m:132-147`).
- **C8.** CHANGELOG `[Unreleased]`, Fixed, "Failed GP updates": "The run carries on, as in MATLAB BADS: … A failed rebuild is retried at the next step with refitted hyperparameters". The refit forced after a failed rebuild is PyBADS's own (KD-B5-2): MATLAB refits only through `gppredcheck` after `MinRefitTime`. "As in MATLAB" holds only for the run carrying on and the rebuild at the next step.
