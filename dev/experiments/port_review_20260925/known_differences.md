<!-- Written by the preparatory agent of the port review (wave 0), reading PyBADS at ab4dded and MATLAB BADS at 74919c0; saved verbatim from its final message on 2026-09-25 (the three parts of that message are kept as known_differences.md, counterpart_map.md and prep_report.md). Its Python line citations were carried to 95da7f1 on 2026-09-26 (refresh_citations.py), and KD-B1-8 and the path conventions edited to match; the entries and edits that cite the rulings of wave 1 (verification/wave1.md) were added by the orchestrator from 2026-09-26, at the same revisions. For wave 2 the Python citations were carried to fef6c14 on 2026-09-26 (refresh_citations.py --base 95da7f1, the citations that it leaves to a reading by hand mapped by the same diff), the entries of wave 1's rulings read against it, and the path conventions and a note on the claims edited to match; the entries and edits that cite the rulings of wave 2 (verification/wave2.md) were made by the orchestrator on 2026-09-26, with the lines of `fef6c14` where they cite its code and the commits of wave 2's fix pass, on `dev-port-review-w2`, where they cite code that the pass wrote. For wave 3 the Python citations were carried to 8aecb6a on 2026-09-26 (refresh_citations.py --base fef6c14, the two citations that it leaves to a reading by hand mapped by the same diff, and the citations of wave 2's entries labelled with fef6c14 carried by hand), and the entries of wave 2's rulings read against it; the entries and edits that cite the rulings of wave 3 (verification/wave3.md) were made by the orchestrator on 2026-09-27, with the lines of `8aecb6a` where they cite its code and the commits of wave 3's fix pass, on `dev-port-review-w3`, where they cite code that the pass wrote, and KD-B4-3 (W3-24's LTMADS directions) went with W3-24's revert; the doublecheck of wave 3 (verification/wave3.md, "Doublecheck") added KD-B3-7, KD-B3-8 and KD-B4-4 to KD-B4-6, and corrected KD-B3-3, KD-B3-5, KD-B4-1, KD-B4-2 and KD-B5-2, on 2026-09-27, at the same revisions. For wave 4 the Python citations were carried to 0d866e8 on 2026-09-27 (refresh_citations.py --base 8aecb6a; of the five citations whose lines wave 3's fix pass rewrote, the three of the acquisition's call sites, the one of KD-B5-2 labelled with 8aecb6a among them, and the one of the empty search set mapped by the same diff, and the one of the ES search's fallback draw, which W3-9 removed, dropped from KD-B1-1), and the entries of wave 3's rulings read against it; the citations of the doublecheck's new and edited entries, written at 8aecb6a, were carried to 0d866e8 by hand when dev-next was merged into dev-port-review-w4 (KD-B3-3, KD-B4-4, KD-B4-5, KD-B4-6); the entries and edits that cite the rulings of wave 4 (verification/wave4.md) were made by the orchestrator on 2026-09-27, with the lines of `0d866e8` where they cite its code and the commits of wave 4's fix pass, on `dev-port-review-w4`, with their lines there, where they cite code that the pass wrote; the rest is the agent's text. -->

# Known differences between PyBADS and MATLAB BADS

**Path conventions.** Python paths are relative to the PyBADS repository root, at commit `0d866e8`, `dev-next` after the fix pass of wave 3 (#77), which wave 4 reviews (written at `ab4dded`; the citations were carried to `95da7f1`, the review's freeze, by `refresh_citations.py --base ab4dded`, and the entries that #71 touched were read again; then to `fef6c14` by `refresh_citations.py --base 95da7f1`, and the entries of wave 1's rulings were read again; then to `8aecb6a` by `refresh_citations.py --base fef6c14`, and the entries of wave 2's rulings were read again; then to `0d866e8` by `refresh_citations.py --base 8aecb6a`, and the entries of wave 3's rulings were read again), except in the claims C1 to C8, which cite `95da7f1`. MATLAB paths are relative to the root of MATLAB BADS (`acerbilab/bads`), at commit `74919c0` (v1.1.3). gpyreg paths are relative to gpyreg at v1.3.3 (`98ab5a4`). Line ranges are inclusive. "MATLAB" means MATLAB BADS at `74919c0`, "default" means the default options of either side, and "reached at default" means that a default run executes the code.

**What this sheet is.** It lists only differences that are *settled and deliberate*: the ones a record decided, or that the code marks as a deliberate departure. A reviewer who finds one of them, as described here, does not report it as new. Each entry is a claim a reviewer may challenge. If the code differs from the description, or the stated reason does not hold, that is a finding. Open findings, undecided questions, suspected defects and items marked "not yet fixed" in earlier records are left off on purpose. **If a difference is not on this sheet, that does not mean the code is correct or matches MATLAB.** Where an entry settles only part of a behaviour, it says which part stays open. The records named under "Why" are provenance. A reviewer can check an entry without them.

Kinds: deliberate change | unported feature | removed feature | substituted library | Python-only feature.

## B1: setup, options, bounds, transform, result

**KD-B1-1. Every random draw comes from one `numpy.random.Generator`, `bads.rng`, created from the `random_seed` option; MATLAB draws from its global stream**
- Python: `pybads/rng.py:6-26` (`get_rng`); `pybads/bads/bads.py:276-277`, `1019-1033` (`_init_rng_`), `324-339` (random `x0`), `1878`, `2327` (fallback indices); `pybads/poll/poll_mads_2n.py:62`, `67`, `72`; `pybads/search/search_hedge.py:71`, `75`; `pybads/search/es_search.py:128`, `219`; `pybads/init_functions/init_sobol.py:56-61` at `efe5e95` (the seed of the design's scrambling, one draw, since W4-1); `pybads/bads/gaussian_process_train.py:157-159`, `176-212`, `449`, `670-672`, `815`, `843-851` (rng passed to `gp.fit` and `SliceSampler`); `pybads/bads/optimize_result.py:168`; `pybads/bads/option_configs/basic_bads_options.ini:22-23`.
- MATLAB: `private/setupvars.m:83`; `bads.m:589`, `857`; `search/searchES.m:117`, `168`, `201`; `search/searchHedge.m:48`, `50`; `poll/pollMADS2N.m:10`, `14`, `17`; `init/initSobol.m:14`; `private/gpupdate.m:374`, `392`; `utils/gppriorrnd.m:75`; `private/bads_output.m:25` (`output.rngstate = rng`).
- What differs: MATLAB has no seed option. It draws with `rand`, `randn`, `randi` and `randperm` from the global stream and reports the stream state. PyBADS builds one generator when the `BADS` object is created, passes it to every draw site, and never touches NumPy's global stream. The one exception is `random_seed=None`: the generator is then seeded from four draws of the global stream. The result reports `random_seed`, not a stream state. No draw is meant to reproduce MATLAB's numbers. This entry settles where the draws come from. It does not settle what is drawn or from which distribution; those remain for each slice to compare.
- Why: CHANGELOG `[1.1.0]`, "Added: Seeded runs through a random generator" and the Upgrading lines; `dev/plans/tooling-and-rng.md`, Phase 8 "Contract" and Decisions ("One generator per run", "`random_seed` no longer seeds NumPy's global stream"); `AGENTS.md`, "Randomness goes through one numpy.random.Generator".
- Kind: deliberate change (and a Python-only option, `random_seed`).
- Slice: B1 (the draw sites belong to every slice).

**KD-B1-2. Options live in two `.ini` files, with snake_case names, several of them renamed**
- Python: `pybads/bads/option_configs/basic_bads_options.ini`, `advanced_bads_options.ini`; `pybads/bads/bads.py:249-264`.
- MATLAB: `bads.m:149-161` (basic `defopts`), `187-290` (advanced).
- What differs: the option set is split between a basic and an advanced `.ini` file. MATLAB CamelCase names become snake_case. Renames that are not a plain case change: `Ninit`→`fun_eval_start`, `Ndata`→`n_train_max`, `MinNdata`→`n_train_min`, `BufferNdata`→`buffer_ntrain`, `MeshOverflowsWarning`→`mesh_overflow_warning`, `Nsearch`→`n_search`, `Nsearchiter`→`n_search_iter`, `Nbasis`→`n_basis`, `TolPoI`→`tol_poi`, `ESbeta`/`ESstart`→`es_beta`/`es_start`, `gpSVGDiters`→`gp_svd_iters`, `NormAlphaLevel`→`normalpha_level`, `InitFcn`→`init_fun`, `gpdefFcn`→`gp_def_fcn`, and `gp*`→`gp_*`. `PeriodicVars` and `OutputFcn` are basic in MATLAB and advanced in PyBADS. Values also change form: function handles become strings (`init_fun = "init_sobol"`, `poll_method = 'poll_mads_2n'`, `search_acq_fcn = ('acq_LCB', None)`); `'on'`/`'off'`/`'yes'`/`'no'` become Python booleans; `nvars` becomes `D`. `search_method = [('ES-wcm',1), ('ES-ell',1)]` lists the members of the search hedge, and the hedge itself is always used. MATLAB names `@searchHedge` explicitly (`bads.m:239`). Default values are not settled by this entry: B1 compares them one by one.
- Why: the `.ini` files are the documented option interface (`docsrc/source/api/options/bads_options.rst` includes them verbatim); `AGENTS.md`, "Options are layered".
- Kind: deliberate change (interface).
- Slice: B1.

**KD-B1-3. User option values are used verbatim, `.ini` expressions are evaluated with `D`, and an unknown option name raises**
- Python: `pybads/bads/options.py:36-67` (user options stored as given), `106-133` (`.ini` values `eval`'d with `D` bound through `exec`; user-set keys skipped), `135-163` (unknown names raise `ValueError`); `pybads/bads/bads.py:252-264`.
- MATLAB: `private/setupoptions.m:21-50` (string values of the listed fields, the user's included, are `eval`'d, with `evalbool` as fallback); unknown fields are kept and ignored.
- What differs: a user's string such as `"200*D"` stays a string in PyBADS, where MATLAB evaluates `'200*nvars'` (for `max_fun_evals`, which must be a positive integer, it is refused: W2-18, `b29b9b5`). A misspelt option raises in PyBADS and is ignored in MATLAB. A user value of `None` stands for the default, as MATLAB's empty field does (`setupoptions.m:5-9`): the option is not set (`Options.__init__`, `unset_user_options`; W2-19, `49ac8aa`). The options whose default is `True` or `False`, and `uncertainty_handling`, take only booleans (`Options.validate_boolean_options`, called from `BADS.__init__`): MATLAB's strings `'on'`, `'off'`, `'yes'` and `'no'`, which `setupoptions.m` evaluates with `evalbool`, are refused with `ValueError`, not converted, since user values are used verbatim.
- Why: `AGENTS.md`, "Options are layered … the dict is used verbatim, so a user's `"200*D"` stays a string … An unknown name raises `ValueError`"; the PI's ruling on W2-19 (`verification/wave2.md`, "Rulings").
- Kind: deliberate change.
- Slice: B1.

**KD-B1-4. Options that exist on one side only**
- Python: the two `.ini` files.
- MATLAB: `bads.m:161` (`OptimToolbox`), `188` (`Debug`), `189` (`TrueMinX`).
- What differs:
  - *MATLAB only:* `OptimToolbox`, which chooses between `fmincon` and `minimizebnd` for the GP hyperparameters (`utils/gpHyperOptimize.m:235-283`). PyBADS has no counterpart because its optimizer is gpyreg's (KD-B6-1). `Debug` and `TrueMinX` only print or plot (`bads.m:1002-1011`, `private/gpupdate.m:61-63`, `351-353`, `private/scatterplot.m`).
  - *PyBADS only, and read by code:* `random_seed` (KD-B1-1); `stobads`, `opp_stobads`, `stobads_frame_size_scaling_power` (KD-S-1); `gp_mean_fun` (`'const'` is MATLAB's fixed `@meanConst`, `gpdef/gpdefBads.m:167`; `'zero'` is PyBADS's own; since W1-34 (`43138f2`) every other name, `'negquad'` included, is refused when `BADS` is created); `gp_train_n_init`, `gp_train_n_init_final`, `gp_train_init_method`, `gp_tol_opt`, `hpd_frac`, `gp_quadratic_mean_bound`, `tol_sd`, `use_slice_sampler`, `gp_hyp_sampler`, `hyp_run_weight`, `fun_evals_per_iter`, `noise_shaping` (all options of the gpyreg-based GP layer or taken from PyVBMC's); `init_mesh_size_integer` (default 0, which is MATLAB's fixed `MeshSizeInteger = 0`, `private/setupvars.m:41`); `hessian_update`, `hessian_method` (read only by a no-op branch, `pybads/bads/bads.py:1970-1975`, under the `.ini` heading "Adaptive basis (unsupported)"; MATLAB v1.1.3 has no such option).
  - *PyBADS only, and refused:* `f_vals`, which could not work (it selected a display format that the display could not fill, and its values reached nothing), is refused with `ValueError` when `BADS` is created if it holds a finite value (W2-7, `652b25c`); one without, such as an empty list, stands for `None` (wave 2's doublecheck, `verification/wave2.md`).
  - *PyBADS only, and read by no code:* see KD-B1-5.
  - *On both sides, unported in PyBADS:* `fun_values` (MATLAB's `FunValues`, which imports earlier evaluations into the function log, `private/setupvars.m:126-167`): a non-empty value is refused with `ValueError` (W2-6, `0ba1241`); the port is a `dev/TODO.md` item.
  - This entry settles only that these options exist on one side. Their effects are not settled: `gp_train_*`, `gp_tol_opt` and `hpd_frac` act at default options and are compared under B5/B6.
- Why: `AGENTS.md`, "Many options do nothing. Some are PyVBMC or MATLAB leftovers"; KD-B1-1, KD-S-1 and KD-B6-1 for the named groups; the PI's rulings on W2-6 and W2-7 (`verification/wave2.md`).
- Kind: removed feature (MATLAB-only options); Python-only feature (the others).
- Slice: B1.

**KD-B1-5. Options that are parsed and have no effect**
- Python: the two `.ini` files. None of the names below is read in `pybads/` outside `testing/`, except where a no-op read is cited.
- MATLAB: as listed per group.
- What differs:
  - (a) *No reads on either side:* `skip_poll` (`SkipPoll`), `search_improve_frac` (`SearchImproveFrac`), `gp_cluster` (`gpCluster`). MATLAB reads none of these either.
  - (b) *PyBADS hard-codes MATLAB's default choice, so the option does nothing:* `poll_method` (always `poll_mads_2n`, KD-B4-1), `poll_acq_fcn` (always LCB, KD-B3-2), `gp_def_fcn` (always the RQ ARD kernel, KD-B6-1), `gp_method` (always nearest neighbours), `chol_attempts` (the Cholesky factorization is gpyreg's, KD-B6-1).
  - (c) *MATLAB reads the counterpart only away from its defaults:* `n_basis` (read only by `poll/private/pollBMADS2N.m`, which nothing calls), `gp_samples` and `gp_svd_iters` (KD-B5-4), `rotate_gp` (MATLAB also marks it unsupported, `gpdef/gpdefBads.m:105-108`).
  - (d) *No MATLAB counterpart; PyVBMC or porting leftovers:* `gp_cov_fun` (overridden by `optim_state["gp_cov_fun"] = 1`, `pybads/bads/bads.py:949`), `upper_gp_length_factor` (its branch in `_gp_hyp` was overwritten by the next lines, and W1-33, `0889426`, removed it), `min_iter` and `min_fun_evals` (PyVBMC's, since `c7c88ab`; MATLAB's termination has no such options, W2-35), `diagnostics`, `hessian_alternate`, `cov_sample_thresh`, `gp_sample_widths`, `weighted_hyp_cov`, `tol_cov_weight`, `gp_sample_thin`, `stable_gp_sampling`, `gp_tol_optmcmc`, `nsgp_max`, `nsgp_maxwarmup`, `nsgp_maxmain`, `stable_gp_samples`, `gp_tol_optactive`, `gp_tol_optmcmcactive`, `tol_gp_var`, `tol_gp_varmcmc`, `active_sample_gp_update`, `sample_extra_vp_means`, `integrate_gp_mean`, `tol_skl`, `tol_stable_warmup`, `variational_sampler`, `kl_gauss`, `k_warmup`, `stable_gp_vpk`, `max_repeated_observations`, `repeated_acq_discount`, `sgd_step_size`, `rank_criterion`, `ns_search`, `gp_stochastic_step_size`, `heavy_tail_search_frac`, `mvn_search_frac`, `hpd_search_frac`, `box_search_frac`, `search_cache_frac`, `empirical_gp_prior`, `tol_gp_noise`, `gp_length_prior_mean`, `gp_length_prior_std`, `init_design`, `bandwidth`, `out_warp_thresh_base`, `out_warp_thresh_mult`, `out_warp_thresh_tol`, `temperature`, `separate_search_gp`, `noise_shaping_threshold`, `noise_shaping_factor`, `acq_hedge_iter_window`, `acqhedge_decay`, `active_search_bound`, `tol_bound_x`, `recompute_lcb_max`, `double_gp`, `warp_every_iters`, `incremental_warp_delay`, `warp_tol_reliability`, `warp_proto_scaling`, `warp_cov_reg`, `warp_proto_corr_thresh`.
  - (e) *Read only by a branch that does nothing or refuses:* `plot` (KD-B2-2), `restarts` (KD-B2-1), `search_optimize` (KD-B3-4), `acq_hedge` (KD-B3-3), `fitness_shaping` (KD-B5-5), `hessian_update`/`hessian_method` (KD-B1-4), `warp_func` ≠ 0 (KD-B6-4), `periodic_vars` (KD-B1-6), `init_fun` other than `"init_sobol"` (KD-B7-2).
  - This list covers only the options settled as having no effect. It is not a list of every option that no code reads.
- Why: `AGENTS.md`, "Many options do nothing … Grep for an option's reads before relying on it"; the entries cited in (b), (c) and (e).
- Kind: removed feature ((b), (c)); Python-only feature ((d)).
- Slice: B1.

**KD-B1-6. Periodic variables are not supported**
- Python: `pybads/bads/bads.py:324-333` at `b78f782` (`BADS.__init__`: an empty `periodic_vars` is set to `None`, and any other value that is not `None` raises `ValueError`, before the random draw of `x0`, the first transform of the variables; `periodic_vars`, found while verifying wave 4, `b78f782`); `pybads/utils/period_check.py:4-6` (a stub that returns its input); `pybads/bads/gaussian_process_train.py:386` (TODO); `pybads/search/es_search.py:138` (TODO). The periodic branches of `udist`, `ucov`, `_variable_transformer_` and `_init_optim_state_` (`pybads/bads/bads.py:1002-1006`, `737-752`) are unreachable.
- MATLAB: `bads.m:152` (`PeriodicVars`); `private/setupvars.m:49-57`, `107-116`; `utils/periodCheck.m`; `gpdef/gpdefBads.m:58-81`, `277-284`; `utils/udist.m`; `utils/ucov.m`; `gpml_fast/covPPERard_fast.m`.
- What differs: MATLAB wraps periodic variables into their range and uses a periodic kernel. PyBADS refuses them. An empty `periodic_vars` names none on both sides (MATLAB's `isempty`, `private/setupvars.m:107-108`).
- Why: `pybads/bads/README.md` ("Support for periodic variables"); `dev/TODO.md`, "Porting gaps"; the error message and TODO comments above; the PI's ruling on `periodic_vars` (`verification/wave4.md`, "Found while verifying" and "Rulings"), which moved the refusal before the first transform, since a random `x0` reached `_variable_transformer_`'s periodic branch before it.
- Kind: unported feature.
- Slice: B1 (option); B7 (`period_check`).

**KD-B1-7. Fixed variables are refused; MATLAB removes them and runs a smaller problem**
- Python: `pybads/bads/bads.py:505-515`.
- MATLAB: `private/boundscheck.m:39-40`; `bads.m:351-382`, `1480-1488` (`expandvars`); `private/fixedbads.m`.
- What differs: a variable whose bounds are all equal makes PyBADS raise `ValueError`. MATLAB fixes it and optimizes the others.
- Why: the code comment "Fixed variables (all bounds equal) are not supported" and the error message.
- Kind: unported feature.
- Slice: B1 (the check); `fixedbads.m` and `expandvars` are in B2's MATLAB list.

**KD-B1-8. The result is an `OptimizeResult` dict, not MATLAB's six outputs**
- Python: `pybads/bads/optimize_result.py:8-184`; `pybads/bads/bads.py:1742-1745`.
- MATLAB: `bads.m:1` (`[x,fval,exitflag,output,optimState,gpstruct]`), `1185-1194`; `private/bads_output.m`.
- What differs: PyBADS returns a scipy-style dict (`x`, `x0`, `fval`, `fsd`, `yval_vec`, `ysd_vec`, `func_count`, `iterations`, `mesh_size`, `message`, `target_type`, `problem_type`, `total_time`, `overhead`, `random_seed`, `algorithm`, `version`, `fun`, `non_box_cons`, `success`, `status`). The `BADS` object keeps the run's state. MATLAB's `exitflag` is the result's `status`: 0 when the run ends on `max_fun_evals` or `max_iter` or is stopped by `output_fcn`, 1 on `tol_mesh`, 2 on the stall criterion (`bads.m:423`, `1062-1083`; W2-12, `d964576`, where `status` was listed and never set), and `success` is `status > 0`, the convention of scipy's `OptimizeResult` too (W2-13, `877d63c`). There is no `rngstate` (see KD-B1-1) and no `maxconstraint`. `iterations` counts as MATLAB's `output.iterations` does, from 1, since #71, except that a run that ends in its initialization (`max_fun_evals=1`, or `output_fcn` stopping at `"init"`) reports 0, where MATLAB reports 1 (`bads.m:482`, `bads_output.m:21`): 0 says that no iteration ran (the PI's ruling on W2-32). `fun` and `non_box_cons` are the objects passed, not copies (W2-14, `1fb162e`), where MATLAB stores `func2str(fun)` (`bads_output.m:4`). `yval_vec` is `None` for a deterministic run and with `noise_final_samples = 0`, where MATLAB returns the incumbent's observation (`bads.m:1134`, `bads_output.m:37`), as the docstring of `OptimizeResult` documents (ledger of wave 0, W0-4).
- Why: the class docstring ("based on `scipy.optimize.OptimizeResult`"); `docsrc/source/api/classes/optimize_result.rst`; `docsrc/source/quickstart.rst`; the PI's rulings on W2-12, W2-13 and W2-32 (`verification/wave2.md`).
- Kind: deliberate change (interface).
- Slice: B1.

**KD-B1-9. A start that `non_box_cons` rejects once put on the mesh is refused**
- Python: `pybads/bads/bads.py:700-709` (`_init_mesh_`, since `1bee482`, "Check gridizied non-box-cons").
- MATLAB: `private/evalinitmesh.m:22-26` tests the constraint at `optimState.x0`, the start in original coordinates before it is put on the mesh (`private/setupvars.m:84-87`, `101`), and evaluates the point on the mesh without testing it.
- What differs: PyBADS tests `non_box_cons` at the start a second time, after `force_to_grid`, and raises `ValueError` if the point on the mesh violates it, where MATLAB evaluates that point. Settled only for the start; how the design's and the poll's infeasible points are handled is compared under B3, B4 and B7.
- Why: the commit `1bee482` ("Check gridizied non-box-cons"); the B1 verifier of wave 2 (V6, "Accepted as intended"), `verification/wave2.md`, "Notes on the reports".
- Kind: deliberate change.
- Slice: B1.

**KD-B1-10. The transform's self-test tolerates an error relative to the bounds' magnitude**
- Python: `pybads/variable_transformer/variables_transformer.py`, the self-test of `VariableTransformer.__init__` (lines 214-236; since W2-5, `a2b8d38`, a tolerance of `1e-6 · max(1, |b|)`).
- MATLAB: `utils/transvars.m:30`, `169-178` (an absolute tolerance, 1e-6).
- What differs: both sides check that the inverse of the transform returns each finite bound. MATLAB's absolute tolerance refuses valid bounds of large magnitude (from about 1e10, or an upper bound from about 1e9 on a log scale), through rounding alone; PyBADS accepts them. A shared defect that PyBADS fixes (`matlab_side_defects.md`).
- Why: the PI's ruling on W2-5 (`verification/wave2.md`, "Rulings").
- Kind: deliberate change.
- Slice: B1.

**KD-B1-11. With `non_box_cons`, a random start that violates the constraint is drawn again**
- Python: `pybads/bads/bads.py`, `BADS.__init__`, the draw of a start that is not finite (since W2-11, `c3d7815`; lines 308-329).
- MATLAB: `private/setupvars.m:83-85` (one draw in the plausible box), `private/evalinitmesh.m:22-26` (the error).
- What differs: when `x0` is not given or not finite, MATLAB draws one start in the plausible box and stops with an error when it violates `non_box_cons`. PyBADS draws again, up to 1000 draws in all, and then raises `ValueError` as before; a run whose first draw is feasible draws the same start as MATLAB would from the same numbers. A shared defect that PyBADS fixes (`matlab_side_defects.md`).
- Why: the PI's ruling on W2-11 (`verification/wave2.md`, "Rulings").
- Kind: deliberate change.
- Slice: B1.

## B2: main loop, termination, noisy re-evaluation, final estimate

**KD-B2-1. Restarts are not implemented**
- Python: `pybads/bads/bads.py:1320`, `1607-1611` (`if self.restarts > 0: pass`).
- MATLAB: `bads.m:201`, `479`, `1121-1127` ("Multiple starts (deprecated)").
- What differs: with `restarts > 0`, MATLAB resets the mesh and continues after termination. PyBADS stops. Both default to 0.
- Why: the code keeps MATLAB's own label, "Multiple starts (deprecated)".
- Kind: unported feature.
- Slice: B2.

**KD-B2-2. Plotting is not implemented**
- Python: `pybads/bads/bads.py:1475-1477` (`plot == "scatter"`: `pass`), `2519` (TODO: profile plot); `advanced_bads_options.ini:3`.
- MATLAB: `bads.m:187`, `988-1015` (`'profile'`, `utils/landscapeplot.m`), `1054-1057` (`'scatter'`, `private/scatterplot.m`).
- What differs: `plot` has no effect.
- Why: TODO comments; `dev/plans/port-correctness-review.md` puts plotting code out of scope.
- Kind: unported feature.
- Slice: B2 (out of scope as non-numerical).

**KD-B2-3. Messages go through Python logging, to the `BADS` logger**
- Python: `pybads/bads/bads.py:220-221`, `279-298`, `2915-2973`; `pybads/bads/gaussian_process_train.py:19`.
- MATLAB: `bads.m:311-328` (`prnt` levels) and `fprintf` throughout.
- What differs: `display` sets the level of a logger instead of choosing which `fprintf` calls run. The level follows MATLAB's mapping of the first three letters, lower case (since W2-15, `37be0d9`): `"off"` and `"none"` show the warnings only (WARNING), `"notify"` and any other value also the opening message (level 25), `"final"` also the final message (level 22), `"iter"` and `"all"` also the iteration lines (INFO), and `"full"`, which exists only in PyBADS, the debug messages too (DEBUG). This entry settles the mechanism. The content and format of the display remain open to comparison.
- Why: CHANGELOG `[Unreleased]`, Fixed, "Messages on the BADS logger".
- Kind: deliberate change.
- Slice: B2.

**KD-B2-4. When the re-estimate of the current iterate fails, it keeps its estimate; MATLAB records NaN**
- Python: `pybads/bads/bads.py:2890-2894` (`_re_evaluate_history_`).
- MATLAB: `bads.m:1097-1104`; `utils/gppred.m:22-56` (a failed posterior gives NaN).
- What differs: in a noisy run, the re-estimate of each iterate rebuilds a copy of the working GP around it. When that rebuild fails, a past iterate gets NaN, as in MATLAB, and the move and the final choice skip NaN; the current iterate keeps its recorded estimate, so that the incumbent is never NaN, where MATLAB's incumbent becomes NaN. The NaN of past iterates stays in `iteration_history` (W2-39; the `IterationHistory` documentation page says so).
- Why: the PI's ruling on W1-35 (`verification/wave1.md`, "Fix pass"); the changelog, "Re-evaluation of the iterates in noisy runs"; `test_noisy_re_estimate_after_failed_rebuild`; W2-30 (`verification/wave2.md`).
- Kind: deliberate change.
- Slice: B2.

**KD-B2-5. The output function: a stop has a message of its own, a stop is final, and the `"init"` call comes after the setup of a noisy run**
- Python: `pybads/bads/bads.py`, `optimize()` and `_init_optimization_` (the calls of `output_fcn` with `"init"`, `"iter"` and `"done"`, since #71, `95da7f1`).
- MATLAB: `bads.m:427` (the `"init"` call), `431-457` (the changes of a noisy run and the first GP), `1062-1083` (the stop).
- What differs: a stop by `output_fcn` ends the run with the message "terminated by options['output_fcn']", where MATLAB keeps the message of an earlier criterion; a false return at `"init"` cannot reopen a run that ended there, where MATLAB assigns the return value. The `"init"` call comes after the options of a noisy run are changed and the first GP is fitted, and MATLAB's before; moving it would split `_init_optimization_`, for nothing a run computes.
- Why: #71 (`95da7f1`) and its changelog lines; the PI's ruling on W2-33 (`verification/wave2.md`, "Rulings").
- Kind: deliberate change.
- Slice: B2.

**KD-B2-6. The budget counts the noise test, and the initial design keeps within it after its rounding**
- Python: `pybads/bads/bads.py`, `_init_mesh_` (the design keeps its first points within `max_fun_evals` minus the evaluations made) and `_init_optimization_` (the reserve for the final samples floored at 0), since W2-27 (`381bf32`); lines 1071-1078, 1165-1178.
- MATLAB: `private/evalinitmesh.m:37-42` (the noise test), `98-104` (`Ninit = min(options.Ninit, MaxFunEvals - 1)`).
- What differs: MATLAB caps the design at `MaxFunEvals - 1` without counting the noise test at `x0`, so a budget below the design takes one evaluation more than `MaxFunEvals`. PyBADS rounds the design up to a power of two (KD-B7-1) and then keeps its first points within the evaluations left, the noise test counted, so that only a run with `max_fun_evals=1` and the noise test exceeds `max_fun_evals`, by the noise test, as in MATLAB, and a noisy run's reserve for its final samples is never negative. At the default budgets the cap does not bind. The design's doubling when its size equals D is deliberate (KD-B7-1).
- Why: the PI's ruling on W2-27 (`verification/wave2.md`, "Rulings"); the counting of the noise test is a shared defect that PyBADS fixes (`matlab_side_defects.md`); the PI's ruling on W4-3, keep (`verification/wave4.md`).
- Kind: deliberate change.
- Slice: B2 (the cap); B7 (the design).

**KD-B2-7. A move to an earlier iterate after the re-estimate moves the incumbent's location with its value**
- Python: `pybads/bads/bads.py`, `optimize()`, the move after `_re_evaluate_history_` (since W2-25, `a9fbb97`: `_update_incumbent_` with the iterate's `u`, `yval`, `fval` and `fsd`; lines 1528-1545).
- MATLAB: `bads.m:1111-1118` (sets `u`, `yval`, `fval`, `fsd` and the target's hyperparameters, not `ubest`), `769` (`u = ubest` at the next iteration).
- What differs: when the re-estimate of a noisy run finds an earlier iterate better by more than `TolFun`, MATLAB moves the incumbent's value to it and leaves `ubest` at the old incumbent, so that the next search's target is predicted at the old point and a poll that no successful search precedes runs around the old point while it is judged by the other iterate's value. PyBADS moves the incumbent, its location with its value. As in MATLAB, only the target's hyperparameters move with it, and the working GP stays. A shared defect that PyBADS fixes (`matlab_side_defects.md`).
- Why: the PI's ruling on W2-25, option (b) (`verification/wave2.md`, "Rulings"); the port's own TODO at `c7c88ab` ("TODO in Matlab is not done").
- Kind: deliberate change.
- Slice: B2.

**KD-B2-8. A noisy run that ends within its first iteration takes the final samples it reserved, at the incumbent**
- Python: `pybads/bads/bads.py`, `optimize()`, the final estimate (since W4-14, `b61a880`: the iterate `final_idx` is 0, the incumbent, when `optim_state["iter"]` is 0).
- MATLAB: `bads.m:1138` (the final estimate only at `iter > 1`), `448-452`.
- What differs: MATLAB takes the final samples only after its first iteration, so a noisy run that ends within it, on `MaxIter` 1 or on a `MaxFunEvals` that the initial design nearly uses up, leaves the evaluations it reserved for them unused and reports the incumbent's single observation. PyBADS takes the reserved samples at the incumbent, the run's only iterate, and reports their estimate, as after a later iteration. A run that `output_fcn` stops at `"init"` takes none, on both sides, and its `fsd` is not an estimate (W4-30, `4b84a2d`). A shared defect that PyBADS fixes (`matlab_side_defects.md`).
- Why: the PI's rulings on W4-14, option (a), and W4-30 (`verification/wave4.md`, "Rulings").
- Kind: deliberate change.
- Slice: B2.

**KD-B2-9. `optim_state` keeps the incumbent's values after the re-estimate and after the final estimate; MATLAB's `optimState` keeps older ones**
- Python: `pybads/bads/bads.py`, `optimize()`: the re-estimate at the end of an iteration of a noisy run (W3-33, `43ee8ed`) and the final estimate before the `"done"` call of `output_fcn` (W4-26, `684d2e0`) set `optim_state`'s `u`, `yval`, `fval` and `fsd` with the object's.
- MATLAB: `bads.m:1111-1118` (the move to an earlier iterate sets the variables, not `optimState.fval`), `1150-1165` (the final estimate writes `iterList.fval` and `iterList.fsd` at the chosen index).
- What differs: the copy of `optim_state` that `output_fcn` receives holds the incumbent's values and, at `"done"`, the returned point with its final estimate; MATLAB's `optimState` keeps the values of an earlier iteration there. No result reads these entries. `iteration_history` holds the final estimate at the chosen iterate, as MATLAB's `iterList` does.
- Why: the PI's rulings on W3-33 (`verification/wave3.md`) and W4-26 (`verification/wave4.md`).
- Kind: deliberate change.
- Slice: B2.

## B3: search

**KD-B3-1. The search hedge chooses only between ES-wcm and ES-ell; the other search methods are not ported**
- Python: `pybads/search/search_hedge.py:86-119` (dispatch by name; anything else raises "not implemented yet"); `pybads/search/es_search.py:229-291` (`ESSearchWM` = `searchES` method 1, `ESSearchELL` = method 2; `ESSearchCMA`, method 5 `'ES-cma+'`, which no search method reached and which failed when called, was removed by W3-13, `4865fad`); `pybads/bads/bads.py:1820-1834`.
- MATLAB: `bads.m:239`; `search/searchES.m:3-12`, `39-70` (methods 1-5: `ES-wcm`, `ES-ell`, `ES-eye`, `ES-cov`, `ES-cma+`); `search/searchCMA.m`, `searchCombine.m`, `searchCrossover.m`, `searchGauss.m`, `searchGrid.m`, `searchMax.m`, `searchMaxAcq.m`, `searchNewton.m`, `searchOptim.m`, `searchWCM.m`, `search/private/*.m`.
- What differs: only MATLAB's default search set exists in PyBADS. `ES-eye`, `ES-cov`, `ES-cma+` and the other search functions are absent.
- Why: `AGENTS.md`, "Extension points are hard-coded" (a new search method needs a subclass, an `elif` and an option entry); the PI's ruling on W3-13 (`verification/wave3.md`).
- Kind: unported feature.
- Slice: B3.

**KD-B3-2. Only the LCB acquisition exists, called directly**
- Python: `pybads/bads/bads.py:1864-1869` (search), `2313-2318` (poll); `pybads/search/es_search.py:154-164` (`search_acq_fcn` must be `'acq_LCB'`; TODO "handle other acqs fcns: acqNegEIMin, acqNegPIMi").
- MATLAB: `bads.m:269-270`, `577-578`, `852`; `search/searchES.m:147`, `156-165`; `acq/acqNegEI.m`, `acqNegEQI.m`, `acqNegPI.m`, `acqNegSqEI.m`, `acqRnd.m`, `acq/private/*.m`.
- What differs: `PollAcqFcn` and `SearchAcqFcn` can name other acquisition functions in MATLAB. In PyBADS the poll always uses LCB and the search accepts only LCB. Both default to LCB.
- Why: `AGENTS.md`, "LCB is called directly at the search and poll call sites"; the TODO above.
- Kind: unported feature.
- Slice: B3.

**KD-B3-3. The acquisition hedge (`AcqHedge`) is not implemented**
- Python: `pybads/bads/bads.py:936-937`, `1860`, `1871`, `2027-2029`, `2058`, `2312`, `2320`; `advanced_bads_options.ini:182`.
- MATLAB: `bads.m:271`, `569-573`, `684-686`, `716-719`, `845-848`; `acq/acqPortfolio.m` ('acq' branch; its help line says "(unsupported)"); `acq/acqHedge.m`; `search/searchES.m:139-141` (`error('Hedge not supported here.')`).
- What differs: `acq_hedge=True` is not implemented, and `BADS` refuses it with `ValueError` when it is created, by the PI's ruling after the doublecheck of wave 3 (`verification/wave3.md`, "Doublecheck"); a run with it stopped with `UnboundLocalError` at its first improving search, whose branch for the acquisition hedge names no search (`pybads/bads/bads.py:2027-2031`), as in 1.1.0. MATLAB's search swallows its own error "Hedge not supported here." and falls back. Both default to off. This entry does not cover the *search* hedge's reward update, which is ported (`ESSearchHedge.update_hedge` ↔ `acqPortfolio.m` 'upd', reached at default): its expected reward took `exp(-0.5*g**2/sqrt(2*pi))` for the standard normal density of `acqPortfolio.m:64` until W3-6 (`d79ab75`), and an empty search set decays its gains, as MATLAB's update does, since W3-11 (`4388e6d`; KD-B3-5).
- Why: code comments "not yet supported (even in Matlab)"; MATLAB's own "(unsupported)" label.
- Kind: unported feature.
- Slice: B3.

**KD-B3-4. Local optimization of the acquisition function (`SearchOptimize`) is not implemented**
- Python: `pybads/bads/bads.py:1883-1885` (TODO; `pass`).
- MATLAB: `bads.m:248`, `596-616` (`fmincon` on the acquisition function).
- What differs: `search_optimize=True` does nothing. Both default to off.
- Why: TODO comment "(generally it does not improve results)", which echoes MATLAB's comment at `bads.m:594-595`.
- Kind: unported feature.
- Slice: B3.

**KD-B3-5. An empty search set is a failed search on every path; MATLAB moves to a stale point at `ImprovementQuantile` > 0.5, and stops on a first empty search**
- Python: `pybads/bads/bads.py:2016-2070` (`_search_step_`, the branch for an empty set: W0-15, `0c56d86`; the hedge's update with no point, W3-11, `4388e6d`); `pybads/search/search_hedge.py` (`update_hedge` with `u_search=None`: every gain decays).
- MATLAB: `bads.m:667-725` (the search stage; `722-725`, the hedge's update), `1257-1282` (`EvalImprovement`); `acq/acqPortfolio.m:56-69`.
- What differs: a search set is empty when every candidate violates `non_box_cons` or, on both sides since W3-1 (`149d528`), was already evaluated. PyBADS counts a failed search and decays the hedge's gains, as MATLAB does at `ImprovementQuantile` ≤ 0.5 (the default) or without noise. At `ImprovementQuantile` > 0.5 in a noisy run, MATLAB counts an incremental search and moves the incumbent to the previous search's point `usearch`, with its `fval` and an SD of 0; when the run's first search set is empty, `usearch` is undefined and MATLAB stops with an error. MATLAB's hedge update scores the stale `usearch`, with a reward of 0; PyBADS scores no point, and the gains are the same.
- Why: the PI's ruling on W3-11 (`verification/wave3.md`): the move and the error are MATLAB defects that PyBADS avoids (`matlab_side_defects.md`), and W0-15's ruling counts an empty set as a failed search.
- Kind: deliberate change.
- Slice: B3.

**KD-B3-6. A generation of the ES search that adds no candidate leaves its scale unchanged; MATLAB's scale becomes NaN**
- Python: `pybads/search/es_search.py`, `ESSearch.__call__` (the scale is updated only when `ntest > 0`; W3-9, `a77d95d`, and W3-8, `c788617`).
- MATLAB: `search/searchES.m:170-193` (`frac = nnew/ntest`, 0/0 when `uCheck` removed every candidate of the generation).
- What differs: from `n_search_iter` = 3 (the default is 2), a generation that `non_box_cons` or the removal of evaluated points empties makes MATLAB's scale NaN, and `uCheck`'s projection, whose `min` and `max` ignore NaN, sends every later candidate of the search to the corner `UBsearch` (by reading). PyBADS keeps the scale and reproduces the kept candidates at it.
- Why: the rulings on W3-8 and W3-9 (PI, 2026-09-27, `verification/wave3.md`); `matlab_side_defects.md`.
- Kind: deliberate change.
- Slice: B3.

**KD-B3-7. The search's `sqrt_beta` is `None`, a callable or a positive finite number; MATLAB also takes a schedule's name and any numeric scalar**
- Python: `pybads/acquisition_functions/acq_fcn_lcb.py` (`check_sqrt_beta`; W3-10, `599115b`, and W4-19, `36c9ec1`); `pybads/bads/bads.py`, `_init_optim_state_` (the check of `search_acq_fcn[1]` when `BADS` is created, W4-19).
- MATLAB: `acq/acqLCB.m:10-21`.
- What differs: MATLAB's `acqLCB` takes an empty value (the schedule of Srinivas et al.), a function handle or a function's name, which it calls with `feval(sqrtbetat, t, nvars)`, or any numeric scalar, zero, negative and non-finite values included, and uses a function's value unchecked. PyBADS takes `None`, a callable or a positive finite real number, and raises `ValueError` for anything else, a name included, when `BADS` is created; the search raises `ValueError` when a callable returns a value that is not a positive finite real number.
- Why: the PI's rulings on W3-10 (`verification/wave3.md`) and W4-19 (`verification/wave4.md`): a value that is not positive and finite gives no lower confidence bound, and PyBADS has no schedules to name.
- Kind: deliberate change.
- Slice: B3.

**KD-B3-8. With `hedge_gamma = 0` the search hedge scores every search at the search point, where MATLAB stops on an undefined variable, and `BADS` refuses the hedge's parameters outside their ranges**
- Python: `pybads/search/search_hedge.py`, `update_hedge` (W3-7, `4d357e4`); `pybads/bads/bads.py`, `_init_optim_state_` (the ranges of `hedge_gamma`, W4-18 `6e24519`, and of `hedge_beta` and `hedge_decay`, W4-29 `bd793f2`).
- MATLAB: `acq/acqPortfolio.m:40` (`u(min(iHedge,end),:)`, one row for every search), `47` (`gpstructnew`, whose assignment is commented out), `69` (the decay of the gains); `search/searchHedge.m:45-46` (the probabilities); none of the three parameters is checked.
- What differs: at `hedge_gamma = 0` the searches that the hedge did not choose are scored by the GP at the search point, taken as a row, which is MATLAB's evident intent; MATLAB stops with an error at its first search there, and PyBADS stopped too, with `ValueError`, until W3-7. MATLAB runs with any value of the hedge's parameters, and so did PyBADS: a `HedgeGamma` above 1/n, n the number of searches, makes the hedge favor the search of lower gain, and above 1/(n - 1) gives some searches a negative probability; a negative `HedgeBeta` inverts the hedge, and a non-finite one makes its probabilities NaN; a `HedgeDecay` above 1 makes the gains grow until they overflow, and a negative one makes them alternate in sign. PyBADS refuses, when `BADS` is created, a `hedge_gamma` outside `[0, 1/n]`, a `hedge_beta` that is not a finite number at least 0 and a `hedge_decay` outside `[0, 1]`. Shared defects that PyBADS fixes (`matlab_side_defects.md`).
- Why: the PI's rulings on W3-7 (`verification/wave3.md`) and on W4-18 and W4-29 (`verification/wave4.md`).
- Kind: deliberate change.
- Slice: B3.

## B4: poll, mesh, incumbent, target

**KD-B4-1. The poll always uses MADS 2N (`poll_mads_2n`, MATLAB's `pollMADS2N`); the other poll methods are not ported**
- Python: `pybads/bads/bads.py:2203-2209`.
- MATLAB: `bads.m:206`, `791-798` (`feval(options.PollMethod{:}, …)`); `poll/pollGPS2N.m`; `poll/private/pollBADS2N.m`, `pollBMADS2N.m`.
- What differs: `poll_method` is ignored (KD-B1-5). Both default to MADS 2N, whose basis is the signed coordinate directions at every default state on both sides; LTMADS's tilted directions, tried in wave 3 (W3-24), were reverted after their gate (`verification/wave3.md`, "Fix pass"; `matlab_side_defects.md`).
- Why: `AGENTS.md`, "Many options do nothing … `poll_method`".
- Kind: unported feature.
- Slice: B4.

**KD-B4-2. The target is predicted from the posterior recomputed under the best iteration's hyperparameters, and from the current GP when that posterior cannot be computed**
- Python: `pybads/bads/bads.py:2719-2730` (`try` around `set_hyperparameters(hyp_best)` and `predict`; on `LinAlgError`, `gp.predict` on the unchanged GP).
- MATLAB: `bads.m:1296-1312` (`UpdateTarget` sets `gptemp.hyp = hyp` and keeps `gptemp.post`, so it never refactorizes and cannot fail at this point).
- What differs: PyBADS recomputes the posterior of a copy of the GP under `hyp_best`, the hyperparameters of the best iteration, and predicts the target from it. MATLAB's `UpdateTarget` keeps the current posterior and evaluates the kernel and mean under `hyp` (`bads.m:1301`, `utils/gppred.m:39-47`, `utils/mygp.m:122-123`, `146-187`), a hybrid that is no GP prediction under one set of hyperparameters: emulated under the hyperparameters of 1 to 3 iterations earlier, it gave means of 1.2e3 to 2.8e7 where the observed values were at most 5e-3 (W3-21). PyBADS computes the prediction that MATLAB's code intends. The two agree when `hyp_best` is the current set; the sets differed in none of 21 decisions of three runs at level 0 and three at level 1 (10 and 11 decisions), and in 5 of 13 decisions of five other runs at level 1 (W3-21). The recomputation is why the call can fail, and PyBADS has a failure path that MATLAB lacks, on which it uses the GP's own hyperparameters and posterior. **Settled:** the recomputation (W3-21 (a)) and the fallback.
- Why: `dev/plans/gp-update-guards.md`, Design "Target (call 1)" and Open Question 2; the PI's ruling on W3-21 (`verification/wave3.md`).
- Kind: deliberate change.
- Slice: B4.

**KD-B4-4. A target whose prediction is not finite is computed from the incumbent's SD; MATLAB keeps the prediction's variance**
- Python: `pybads/bads/bads.py:2732-2758` (`_get_target_from_gp_`, the fallback to the incumbent's `fval` and `fsd`; the target from `fsd**2` since W3-23, `dac062e`).
- MATLAB: `bads.m:1309-1311`, `1321`.
- What differs: both sides replace a prediction that is not finite by the incumbent's `fval` and `fsd`, but MATLAB's target keeps the prediction's variance, so that a NaN variance gives a NaN target and an infinite one `-Inf`; PyBADS computes the target from the incumbent's SD. A shared defect that PyBADS fixes (`matlab_side_defects.md`); no run has shown a non-finite prediction.
- Why: the PI's ruling on W3-23 (`verification/wave3.md`).
- Kind: deliberate change.
- Slice: B4.

**KD-B4-5. When every acquisition value is NaN, the search and the poll choose a candidate at random; MATLAB takes the first**
- Python: `pybads/bads/bads.py:1869` (search), `2318` (poll): `np.nanargmin` since W3-27 (`a1bf658`), and a random choice through `bads.rng`, with a warning, when every value is NaN.
- MATLAB: `bads.m:581` (search), `853` (poll): `[~,index] = min(z)`, which skips NaN and returns the first index when every value is NaN.
- What differs: with some values NaN, both sides take the smallest of the others. With every value NaN, MATLAB takes the first candidate, and its fallback "randomly choose index" can never fire; PyBADS makes that fallback live. No such case has been observed.
- Why: the PI's ruling on W3-27 (`verification/wave3.md`, "Rulings" and "Choices within the rulings").
- Kind: deliberate change.
- Slice: B4.

**KD-B4-6. `improvement_quantile`, `accelerate_mesh_steps` and `n_search_iter` are checked when `BADS` is created**
- Python: `pybads/bads/bads.py:812-839` (`_init_optim_state_`, beside the warning for a quantile above 0.5): the checks of W3-31 (`ec1b2d0`) and W3-39 (`5d711bf`), and that of W4-25 (`36e8b70`).
- MATLAB: `bads.m:1269-1271` (`EvalImprovement`); `bads.m:976-979`, `private/setupvars.m:179-182`; `private/setupoptions.m:26`, `search/searchES.m:125` (`Nsearchiter`, unchecked).
- What differs: MATLAB refuses a quantile outside (0, 1) when it first evaluates an improvement, and lets NaN through, to NaN improvements; PyBADS refuses both when `BADS` is created. For `accelerate_mesh_steps`, MATLAB has no check: 0, a negative value and a finite value that is not an integer stop its run at the accelerated mesh reduction (`iterList` starts empty), and `Inf` runs without the reduction, since `iter > Inf` never holds. PyBADS refuses every value that is not a positive integer, `inf` included, and converts a whole-number float; its message names `accelerate_mesh=False`, the switch that turns the reduction off, as MATLAB's `AccelerateMesh` does (the PI's ruling after the doublecheck of wave 3). The stop on a value below 1 is a shared defect that PyBADS fixes (`matlab_side_defects.md`). `n_search_iter`, unchecked on both sides before, is refused when it is not a positive integer, and a whole-number float is converted, as for `accelerate_mesh_steps`.
- Why: the PI's rulings on W3-31 and W3-39 (`verification/wave3.md`, "Rulings" and "After the gates") and on W4-25 (`verification/wave4.md`).
- Kind: deliberate change.
- Slice: B4.

## B5: GP training set and refit policy

**KD-B5-1. Adding a point recomputes every posterior in full, and a failed add leaves the point out of the GP until the next rebuild**
- Python: `pybads/bads/gaussian_process_train.py:1315-1372` (`add_and_update_gp`: `gp.update(X_new=…, y_new=…, s2_new=…, hyp=…)`; on `LinAlgError` gpyreg restores the GP and `temporary_data["needs_rebuild"]` is set); `pybads/bads/bads.py:1897-1917` (search), `2367-2387` (noisy poll: when the GP did not grow, the poll's estimate is NaN and the point counts as no improvement).
- MATLAB: `private/gpupdate.m:39-83` ('add': tries a rank-1 update with `utils/update_posterior.m` first, except under `SpecifyTargetNoise`; the point is appended to `x` and `y` whatever happens), `340-354` (full recomputation inside `try`; `post = []` on failure); `bads.m:633-641`, `908-924`.
- What differs: there is no rank-1 path. On failure the point is dropped from the GP, not kept beside an empty posterior. It stays in the function logger and enters the GP at the next rebuild. The noisy poll's NaN matches MATLAB's NaN prediction.
- Why: survey (`dev/results/2026-09-23-codebase-survey.md`), candidate row `add_and_update_gp`, status "by design"; `dev/plans/gp-update-guards.md`, Open Questions 1 and 5. `dev/TODO.md` keeps the rank-1 update as an open item for a possible change. Its absence is known, not new.
- Kind: deliberate change.
- Slice: B5.

**KD-B5-2. A failed rebuild restores the GP as it was on entry, marks it, and forces a refit at the next rebuild**
- Python: `pybads/bads/gaussian_process_train.py:285-289` (snapshot), `494-526` (restore; `needs_rebuild` and `needs_refit` set; exit flag -2), `596-597` (markers cleared once a posterior is left on the new set); `pybads/bads/bads.py:1772-1783` (search), `2262-2279` (poll; with `poll_training` off after the first iteration, the forced refit gives way), `2296-2299` (the poll treats the GP as unreliable after its own rebuild fails), `2670-2685` (`_record_gp_refit_`).
- MATLAB: `private/gpupdate.m:340-354` (the new data and the failed rebuild's hyperparameters and `pollscale` stay, with `post = []`); `bads.m:523-536`, `826-839` (rebuild while `post` is empty), `1223-1254` (refit only when `gppredcheck` finds the NaN predictions unreliable and `MinRefitTime` has passed).
- What differs: PyBADS throws away the new data and hyperparameters on failure and refits at the very next rebuild, whatever `min_refit_time` says. The markers in `gp.temporary_data` stand in for MATLAB's empty `post`. The retry with the previous hyperparameters on the new training set, which MATLAB lacks, is made only after a refit (without one it would repeat the computation that failed), and the GP's geometry then comes from the hyperparameters it keeps: W1-11 (`f65bc91`) and W1-10 (`9ac1a47`), by the rulings of wave 1 (PI, 2026-09-26). After a failed rebuild the search ranks its candidates by the LCB of the restored GP, a consistent GP with finite predictions (`pybads/bads/bads.py:1864-1869`); in MATLAB the empty `post` makes `gppred` fail again, `acqLCB` sums over no prediction samples, every candidate scores 0, and the stable sort keeps `uCheck`'s first candidate, the lexicographically smallest, an arbitrary point far from the incumbent (W3-12; with an injected failure in 3-D, at an offset of (-142, 38, 93) grid units from the incumbent in PyBADS and (-1891, 1696, 291) under MATLAB's rule). The poll treats such a GP as unreliable on both sides.
- Why: survey, candidate row "`local_gp_fitting`, and `bads.py`, the forced refit", status "by design (… Open Question 7)"; `dev/plans/gp-update-guards.md`, Design and Open Questions 3 and 7; for the search's ranking, the PI's ruling on W3-12 (`verification/wave3.md`).
- Kind: deliberate change.
- Slice: B5.

**KD-B5-3. Only `LinAlgError` is caught at the guarded GP calls; MATLAB's `try` catches any error**
- Python: `pybads/bads/gaussian_process_train.py:496`, `511`, `1366`; `pybads/bads/bads.py:2723`. The same policy applies at the older guards, `pybads/bads/gaussian_process_train.py:217` and `675`.
- MATLAB: `private/gpupdate.m:52-64`, `340-354`.
- What differs: a `ValueError` or other error from gpyreg stops a PyBADS run, where MATLAB would catch it. The decision covers the GP-update guards only. MATLAB's other `try`/`catch` sites (`bads.m:567-586` around the acquisition, `1229-1233` around `gppredcheck`; `acq/acqLCB.m:34-38`; `search/searchES.m:138-150`; `utils/gppred.m:44-54`) have no recorded decision.
- Why: `dev/plans/gp-update-guards.md`, Design, "Scope of the catch".
- Kind: deliberate change.
- Slice: B5.

**KD-B5-4. GP hyperparameters are optimized, never sampled: there is one hyperparameter set**
- Python: `gp_samples` and `gp_svd_iters` are unread; `pybads/bads/gaussian_process_train.py:241` ("Missing port: sample for GP for debug"), `402` ("only optimization supported"), `463` ("Matlab uses hyperSVGD when using multiple samples … not implemented"), `1087` (`gp_s_N = 0`).
- MATLAB: `bads.m:254` (`gpSamples = 0`); `private/gpupdate.m:411-414`; `utils/gpHyperSVGD.m`; the weighted sums over `hypweight` throughout.
- What differs: with `gpSamples > 0`, MATLAB fits several hyperparameter samples by SVGD. PyBADS ignores the option. At the default, 0, both optimize a single set.
- Why: the code comments above.
- Kind: unported feature.
- Slice: B5.

**KD-B5-5. Fitness shaping is not implemented**
- Python: `pybads/bads/gaussian_process_train.py:300-303` (TODO; `pass`); `pybads/bads/bads.py:1904` (TODO); `advanced_bads_options.ini:311-312`.
- MATLAB: `bads.m:279-280` (under the heading "GP warping parameters (unsupported)"); `private/gpupdate.m:43-47`, `252-256`; `utils/fitnessTransform.m`.
- What differs: `fitness_shaping=True` does nothing. Both default to off.
- Why: TODO comments; MATLAB's own "(unsupported)" heading.
- Kind: unported feature.
- Slice: B5.

**KD-B5-6. A refit starts from gpyreg's design of prior draws, not from MATLAB's local runs**
- Python: `pybads/bads/gaussian_process_train.py:1143-1174` (`_get_gp_training_options`: `init_N`, from `gp_train_n_init` falling to `gp_train_n_init_final`, and `opts_N`), `479-489`, `670-672`; gpyreg `gaussian_process.py:1885-1920` (the design, and the second start replaced by a low-noise design point), `f_min_fill.py`.
- MATLAB: `private/gpupdate.m:371-408`, `utils/gpHyperOptimize.m:47-75` (one local optimization from the previous hyperparameters and one from the second-fit point, with `optimset('TolFun',0.1,'TolX',1e-4,'MaxFunEval',150)`).
- What differs: PyBADS evaluates `init_N` draws from the priors beside the given rows and optimizes the best one with gpyreg's L-BFGS-B (the best two on a second fit, the second replaced by gpyreg's low-noise pick). MATLAB runs its local optimizations from the previous hyperparameters and the second-fit point. On the same data (14 refits), the design reached the same optimum in 8, a better one in 3 (by up to 17887 in the negative log posterior) and a worse one in 3 (by at most 0.56), and it avoided fit failures that MATLAB's starts met. Whether the better fit gives a better optimization is not measured.
- Why: the ruling on W1-15 (PI, 2026-09-26, `experiments/port_review_20260925/verification/wave1.md`); the defaults tuned in `8ff10f5`; the optimizer is gpyreg's (KD-B6-1). This settles the `gp_train_*` options that KD-B1-4 leaves open.
- Kind: deliberate change.
- Slice: B5.

**KD-B5-7. The normality test of the GP's calibration is scipy's Shapiro-Wilk; MATLAB's `swtest` switches to Shapiro-Francia for leptokurtic samples**
- Python: `pybads/bads/bads.py:2622-2623` (`scipy.stats.shapiro`, in `_is_gp_refit_time_`).
- MATLAB: `utils/swtest.m:130-160`, `272`, through `utils/gppredcheck.m:30`.
- What differs: MATLAB tests with Shapiro-Francia when the kurtosis of the z-scores exceeds 3, and with Shapiro-Wilk otherwise; PyBADS always uses Shapiro-Wilk. At the level `normalpha_level = 1e-6`, the two decide differently on heavy-tailed z-scores; in default runs the verifier found 4 of 84 and 1 of 79 verdicts different at uncertainty level 0 and none of 125 at level 1, all on the near-zero standard deviations that the fix of W1-3 removes.
- Why: the ruling on W1-6 (PI, 2026-09-26).
- Kind: substituted library.
- Slice: B5.

**KD-B5-8. Under `specify_target_noise`, the high-noise check of a refit takes the base noise 1, whatever `noise_size`**
- Python: `pybads/bads/bads.py:1207-1214` (`noise_size` set to 1.0 under `specify_target_noise`), read by `pybads/bads/gaussian_process_train.py:414-423`.
- MATLAB: `private/setupoptions.m:100-101` (a warning that `NoiseSize` is ignored with `SpecifyTargetNoise`), `private/gpupdate.m:379-381` (the high-noise check reads `NoiseSize` all the same).
- What differs: MATLAB's check reads a user's `NoiseSize`, although its own warning says the option is ignored; PyBADS follows the warning, so that `noise_size=0`, which the warning proposes, does not make every refit a second fit.
- Why: the PI's ruling of 2026-09-25, in #71 (`7b50a3a`): the code comment; `CHANGELOG.md`, "`noise_size` with user-specified noise"; the survey's row fixed in `7b50a3a`.
- Kind: deliberate change.
- Slice: B5.

**KD-B5-9. With `poll_training` off, the poll neither records a refit that it does not make nor clears the flag of an unreliable GP**
- Python: `pybads/bads/bads.py:2255-2260` (`c9ebdc7`, W1-8).
- MATLAB: `bads.m:822-823` (the poll drops the refit when `PollTraining` is off), `1242-1252` (`IsRefitTime` has already set `lastfitgp`, reset the GP statistics and cleared `unrelgp_flag`).
- What differs: MATLAB records a refit that the poll then cancels, so that the next refit of the search waits for `MinRefitTime` counted from a refit that did not happen, and it clears the flag that marks the GP as unreliable; PyBADS does neither, and its stopping rule reads the calibration of the GP as it is. Off by default.
- Why: the PI's rulings on W1-8 (`verification/wave1.md`) and W3-30 (`verification/wave3.md`); `matlab_side_defects.md`, "With `PollTraining` off, the poll records a refit that it then cancels"; `CHANGELOG.md`, "Refits without poll training".
- Kind: deliberate change.
- Slice: B5 (the code is the poll's, B4).

**KD-B5-10. At D = 1 the training set's distances are in units of the fitted length scale; MATLAB's are in units of 1**
- Python: `pybads/bads/gaussian_process_train.py`, `local_gp_fitting` (`len_scale` from the fitted length scales at every D, since W1-17, `cfacb98`).
- MATLAB: `private/gpupdate.m:285-292` (the ARD length scales taken only when `gpstruct.ncovlen > 1`); `gpdef/gpdefBads.m:51`.
- What differs: at D = 1 MATLAB takes the length scale of the training set's distances as 1, since its test for an ARD kernel, `ncovlen > 1`, fails with one length scale; PyBADS takes the fitted one, as at every other D. The training set has between `n_train_min` and `n_train_max` points on both sides. A shared defect that PyBADS fixes (`matlab_side_defects.md`).
- Why: the PI's ruling on W1-17 (`verification/wave1.md`, "Rulings"); its gate, a 1-D suite of 30 seeds, changed 43 of 180 runs, with no flag. Recorded by wave 4 (O's third reading found it recorded elsewhere).
- Kind: deliberate change.
- Slice: B5.

## B6: GP model and its gpyreg objects

**KD-B6-1. The GP is a gpyreg `GP` with a hard-wired rational-quadratic ARD kernel, not GPML plus `gpml_fast`**
- Python: `pybads/bads/gaussian_process_train.py:95-116` (GP construction), `890-891` (identifier 1 → `RationalQuadraticARD`), `904-1092` (`_gp_hyp`: bounds and priors in gpyreg's units, where a Gaussian prior is `(mean, SD)`), `176-212` and `670-672` (`gp.fit` with the options of `_get_gp_training_options`, `1095-1178`), `834` (the private `gp._GP__gp_obj_fun`, on the slice-sampler path); `pybads/bads/bads.py:946-965` (`optim_state["gp_cov_fun"] = 1`; `gp_noisefun` → `GaussianNoise` flags).
- MATLAB: `bads.m:260` (`gpdefFcn = {@gpdefBads,'rq',[1,1]}`); `gpdef/gpdefBads.m` (a GPML struct; `priorGauss` takes `(mean, variance)`; `likGaussHe`; inference `infPrior_fast` + `infExact_fastrobust` with `CholAttempts`, `309-315`); `gpml_fast/covRQard_fast.m`; `private/gpupdate.m:359-419` (`gpfit`: one or two starting points, `optimset('TolFun',0.1,'TolX',1e-4,'MaxFunEval',150)`); `utils/gpHyperOptimize.m` (`fmincon` or `minimizebnd`); `utils/gppred.m`; `utils/mygp.m`.
- What differs: the GP library and every object it involves (hyperparameter vector, priors, bounds, likelihood, inference, optimizer, prediction) are gpyreg's. The kernel cannot be changed (`gp_cov_fun` and `gp_def_fcn` have no effect). It equals MATLAB's default (`'rq'`, ARD). **Settled:** only the substitution and the hard-wired kernel. **Open to comparison:** every hyperparameter, bound and prior in the units each side uses; the optimizer's starting points and tolerances; the Cholesky handling; the inference; and how PyBADS calls gpyreg. That includes the GP fit at initialization (`init_and_train_gp`), where MATLAB only defines the GP (`bads.m:465-469`). Wave 1 settled three of these (PI, 2026-09-26): the starting points (KD-B5-6), the fit at initialization (KD-B6-5) and the Cholesky handling (KD-B6-6).
- Why: `AGENTS.md` ("The GP layer is the lab's `gpyreg`"; "`gp_cov_fun` is overridden by a hard-coded rational-quadratic ARD kernel"; "gpyreg internals"); `dev/plans/port-correctness-review.md`, Decisions (gpyreg's internals out of scope; its use in scope; `covRQard_fast.m` as the reference for `RationalQuadraticARD`) and "Two facts about the GP layer".
- Kind: substituted library.
- Slice: B6 (and B5).

**KD-B6-2. A zero range of the training targets keeps the previous width of the GP-mean prior, and distances without spread the previous prior of the length scales**
- Python: `pybads/bads/gaussian_process_train.py:341-355` (`mean_sd = y_range / 2` only when `y_range > 0`).
- MATLAB: `gpdef/gpdefBads.m:219-222` (variance `yrange.^2/4` whatever `yrange` is).
- What differs: when `gp_mean_range_fun` gives 0, MATLAB sets a zero-variance prior and PyBADS keeps the previous width. Otherwise the re-centred prior follows MATLAB (the fix of `8afbe16`). Likewise, since W1-26 (`cd1831f`), targets with no spread give the mean's prior the SD 1 in `_gp_hyp`, and a rebuild keeps the previous centre of the output scale's prior where MATLAB centres it at `log(std(y)) = -Inf` (`gpdefBads.m:293-295`). Since W3-40 (`a14524d`), a rebuild whose pairwise distances have no spread (two distinct points) keeps the previous prior of the length scales (`gaussian_process_train.py`, the empirical prior of `gp_cov_prior = "iso"`), where MATLAB's `covsigma` is 0 (`gpdefBads.m:240-251`) and gpyreg refuses a zero sigma.
- Why: the code comment "A zero range, which MATLAB leaves to fail, keeps the previous width", written with `8afbe16` (survey row for the GP-mean prior, status "fixed in `8afbe16`"); the ruling on W1-26 (PI, 2026-09-26); the PI's ruling on W3-40 (`verification/wave3.md`, "After the gates").
- Kind: deliberate change.
- Slice: B6.

**KD-B6-3. With `gp_fixed_mean`, the GP mean is not fixed**
- Python: `pybads/bads/gaussian_process_train.py:353`, `356-357` (TODO).
- MATLAB: `gpdef/gpdefBads.m:168-172` (a delta prior on the mean), `220-231` (the mean hyperparameter set to `ymean`).
- What differs: with `gp_fixed_mean=True`, PyBADS re-centres a Gaussian prior and keeps its width. MATLAB fixes the mean at `ymean`. Both default to off.
- Why: the TODO comment; survey row for the GP-mean prior ("… which the port leaves as a `TODO`"), status "fixed in `8afbe16`".
- Kind: unported feature.
- Slice: B6.

**KD-B6-4. Warped likelihoods and output warping are unsupported on both sides**
- Python: `pybads/bads/gaussian_process_train.py:339`, `392-403` (with `warp_func` ≠ 0, `sd_y` is never assigned and the first rebuild fails), `961`, `1004`, `1047-1048`, `1084` ("Missing port: output warping"); `advanced_bads_options.ini:248` (`warp_func`), `343-353` (unread `warp_*` options).
- MATLAB: `bads.m:279-281`; `gpdef/gpdefBads.m:116-118` (`error('Warped likelihoods not supported at the moment.')`), `210-215`, `287-291`; `warp/*.m`.
- What differs: neither side supports warping. MATLAB refuses it with a message when the GP is defined. PyBADS fails without one at the first rebuild.
- Why: `dev/plans/port-correctness-review.md`, out-of-scope list ("`warp/` (unsupported on both sides)"); the code comments.
- Kind: removed feature.
- Slice: B6 (out of scope).

**KD-B6-5. PyBADS fits a GP on the initial design; MATLAB only defines it**
- Python: `pybads/bads/bads.py:1264-1284` (`init_and_train_gp` at initialization); `pybads/bads/gaussian_process_train.py:22-249`, `950-1002` (the starting values).
- MATLAB: `bads.m:465-469` (`gpdefBads` defines the GP with its starting values; the first fit comes at the first rebuild), `gpdef/gpdefBads.m:164-165`.
- What differs: PyBADS fits the hyperparameters on the initial design under the definition priors; MATLAB keeps the definition values until its first rebuild. Both refit at the first rebuild, so the initial fit reaches a run as one start of that refit and as the hyperparameters under which the first target is predicted.
- Why: the ruling on W1-27 (PI, 2026-09-26: fitting the GP on the initial design makes sense).
- Kind: deliberate change.
- Slice: B6.

**KD-B6-6. A failed Cholesky factorization multiplies the GP's noise; MATLAB treats it as an error**
- Python: gpyreg `gaussian_process.py:3584-3666` (`__training_cholesky`: the noise multiplied by ten per failed attempt, up to ten attempts, the multiplier kept in the posterior as `sn2_mult`), `1321` (`predict` uses it); `chol_attempts` is unread (KD-B1-5 (b)).
- MATLAB: `bads.m:272` (`CholAttempts = 0`); `gpml_fast/infExact_fastrobust.m:36`, `77-80` (an error at the first failure); `utils/gpHyperOptimize.m:73-176` (the fit restarts, with the noise's start nudged); `private/gpupdate.m:340-354` (`post = []`).
- What differs: where MATLAB's fit restarts and its posterior is rebuilt, gpyreg evaluates the objective and computes the posterior at 10 to 1e9 times the fitted noise, which `get_hyperparameters` does not show. It is reached at default options on some targets (on Rosenbrock D = 2, 73 of 116 GP states handed on were inflated; none on Ackley D = 6). Measured with gpyreg's switch (acerbilab/gpyreg#56) turned on at the end of wave 1's first batch (`verification/wave1.md`, "W1-25's measurement"): more evaluations and larger errors on the deterministic ellipsoids, a smaller error on the 2-D sphere, no crash.
- Why: the rulings on W1-25 (PI, 2026-09-26): kept for now; gpyreg's switch, off by default, that makes a failed factorization an error, stays off in PyBADS after its measurement, and the question is revisited once wave 1's fixes have all landed.
- Kind: substituted library.
- Slice: B6.

**KD-B6-7. `gp_cov_prior="ard"`, MATLAB's per-dimension prior over the length scales, is not ported, and is refused**
- Python: `pybads/bads/bads.py`, `_init_optim_state_` (since W1-28, `64616af`: any value other than `"iso"` raises `ValueError`); `pybads/bads/gaussian_process_train.py`, `local_gp_fitting` (the `"iso"` update of the length-scale prior).
- MATLAB: `gpdef/gpdefBads.m:254-274` (`'iso'` and `'ard'`; an error for any other value).
- What differs: MATLAB's `'ard'` sets an empirical prior per dimension; PyBADS refuses it when `BADS` is created, where 1.1.0 accepted it, and any other value, and kept the definition prior all run.
- Why: the ruling on W1-28 (PI, 2026-09-26): refuse rather than port; `dev/TODO.md` keeps the port.
- Kind: unported feature.
- Slice: B6.

**KD-B6-8. A fixed noise (`fit_lik=False`) is refused on both sides**
- Python: `pybads/bads/bads.py`, `_init_optim_state_` (since W1-32, `238afad`: `ValueError`, "Fixed noise not supported").
- MATLAB: `bads.m:466`, `gpdef/gpdefBads.m:139-140` (`error('Fixed noise not supported.')`).
- What differs: nothing but the moment: PyBADS refuses it when `BADS` is created, MATLAB when it defines the GP.
- Why: the ruling on W1-32 (PI, 2026-09-26).
- Kind: removed feature.
- Slice: B6.

## B7: function logger, initial design, utilities

**KD-B7-1. The initial design is a scrambled Sobol set from `scipy.stats.qmc.Sobol` with a power-of-two number of points, seeded from the run's generator**
- Python, at `efe5e95`: `pybads/init_functions/init_sobol.py:16-54` (docstring), `56-61` (`seed = get_rng(rng).integers(2**63)`, since W4-1), `66-73` (`Sobol(D, seed=seed).random_base2(m)`, `m = ceil(log2(fun_eval_start))`, raised by one when `2**m` equals D; the comment cites Owen (2020) on keeping Sobol sets to powers of two); `pybads/bads/bads.py:1214-1223` (the call in `_init_mesh_`).
- MATLAB: `private/evalinitmesh.m:98-104` (`Ninit` points); `init/initSobol.m:9-16` (`seed = mod(prod(uint64(num2str(u0(1:min(10,end))))),MaxSeed)+1`, a skip index into the unscrambled sequence of `i4_sobol_generate(nvars,Ninit,seed)`); `init/private/i4_sobol*.m`, `i4_bit_*.m`.
- What differs: the generator (scipy's scrambled Sobol, where the seed seeds the scrambling); its seed, one draw of `bads.rng`, so that `random_seed` decides the design whatever the start, where MATLAB's design depends on the start alone (KD-B1-1); and the size, `2**ceil(log2(fun_eval_start))` points instead of `Ninit`, twice as many when that number equals D (claim C2). What MATLAB's seed is for a start inside the plausible box needs MATLAB (`matlab_side_defects.md`, "Questions that need MATLAB").
- Why: the PI's rulings on W4-1, option (a), and W4-3, keep (`verification/wave4.md`); the docstring and the Owen comment; `AGENTS.md`, Architecture and the bullet on randomness.
- Kind: substituted library (and deliberate changes of the design's seed and size).
- Slice: B7.

**KD-B7-2. Only the Sobol initial design exists; LHS and uniform designs are not ported**
- Python: `pybads/bads/bads.py:1111-1165` (any other `init_fun` raises "Initialization function not implemented yet").
- MATLAB: `bads.m:199`; `init/initLHS.m`, `init/initRand.m`, `init/private/lhs.m`; `init/initSobol.m:18-21` (Latin hypercube as the fallback when Sobol generation raises).
- What differs: there is no alternative design and no LHS fallback.
- Why: `AGENTS.md`, "the initial design is selected by `init_fun == "init_sobol"`"; the error message.
- Kind: unported feature.
- Slice: B7.

**KD-B7-3. With target noise, a repeated point is merged into its own row, and the merged value is returned**
- Python: `pybads/function_logger/function_logger.py:406-436` (precision-weighted merge into the row that matches in every coordinate; returns the merged value with the new observation's SD).
- MATLAB: `private/funlogger.m:117-129` (each evaluation is a new row, and the call returns the observation itself).
- What differs: at level 2, PyBADS keeps one row per point and returns the merged value; MATLAB adds a row per evaluation. No run of `BADS` reaches the merge since W3-1 (`149d528`): every recorded evaluation after `x0` passes `contraints_check`, which removes a point already evaluated, and the noise test and the final samples record nothing; only a direct use of `FunctionLogger` merges. Returning the observation, as MATLAB does, was tested before W3-1 and not adopted. At levels 0 and 1, a repeat is a new row on both sides.
- Why: survey, candidate row "`function_logger.py`, `__call__` (at `1a21844`, line 193)", status "seen; MATLAB's form tested, not adopted"; `dev/experiments/population_ellipsoid_hetero_linux_20260925/README.md` ("Returning the observation … changes 20 runs and worsens 16 of them", p = 0.0019); CHANGELOG `[1.1.0]` and `[Unreleased]` (the merge, and the row fix of `032dfcb`); `AGENTS.md`, `FunctionLogger` bullet; the PI's ruling on W4-5 (`verification/wave4.md`), which closes the survey's row.
- Kind: deliberate change.
- Slice: B7.

**KD-B7-4. The log of evaluations grows when it is full; MATLAB's is a ring of `CacheSize` rows**
- Python: `pybads/function_logger/function_logger.py:309-321` (`_expand_arrays`: the arrays grow by half when full); the description of `cache_size`, the initial size (`advanced_bads_options.ini`, W4-12 `f1247d0`).
- MATLAB: `private/funlogger.m:120-121` (the row index wraps at `CacheSize`, 1e4 by default).
- What differs: past `CacheSize` logged evaluations MATLAB overwrites its oldest rows, and PyBADS keeps every row. The two agree up to 9999 logged evaluations. MATLAB's ring never writes its last row, which it then reads (`matlab_side_defects.md`).
- Why: the PI's ruling on W4-12 (`verification/wave4.md`); the docstring of `cache_size`.
- Kind: deliberate change.
- Slice: B7.

**KD-B7-5. The noise test's second value goes through the logger's checks**
- Python: `pybads/bads/bads.py:1158-1184` at `46af65a` (`_init_mesh_`: the noise test through `function_logger(..., record_duplicate_data=False)`, which records no row, and the start's row left as it was, W4-6 `e7bd01d` and `46af65a`); `pybads/function_logger/function_logger.py` (the checks of a target's value).
- MATLAB: `private/evalinitmesh.m:41-47` (a direct call of the target).
- What differs: a NaN or infinite second value raises `ValueError`, as at every other evaluation; MATLAB reads NaN as deterministic (its comparison is false) and infinity as noisy. On both sides the test counts toward no record of the log; it counts in PyBADS's `max_fun_evals` (KD-B2-6), and the schedule of the GP's fits leaves it out of its budget.
- Why: the PI's rulings on W4-13 and W4-6 (`verification/wave4.md`).
- Kind: deliberate change.
- Slice: B7.

## M: MATLAB changes since the port began

No entry of its own. Entries that touch the eight commits: KD-B5-1 (the rank-1 `'add'` of `private/gpupdate.m`, as rewritten in `d4fead5`) and KD-B5-2 (a failed rebuild in `gpupdate.m`). The default `Ninit` that `a21f2ee` changed to `10 + nvars` was set back to `nvars` in `019f0b4` (`bads.m:198`), and PyBADS's `fun_eval_start = D` matches that.

## S: Sto-BADS

**KD-S-1. Sto-BADS is PyBADS's own**
- Python: `pybads/bads/bads.py:218`, `321` (keyword-only constructor argument `gamma_uncertain_interval`), `273-274`, `1246-1248` (switched off for deterministic targets), `1980-2014` (search), `2105-2152` (`_sto_success_improvement_`), `2409-2462` (poll); `advanced_bads_options.ini:60-63`, `69-70` (`stobads` False, `opp_stobads` True, `stobads_frame_size_scaling_power` 2).
- MATLAB: no counterpart.
- What differs: when `stobads` is on, an optional success rule based on uncertainty intervals, after Sto-MADS (Audet, Dzahini, Kokkolaras and Le Digabel, 2021), replaces the improvement tests of search and poll. It is off by default.
- Why: `dev/plans/port-correctness-review.md`, slice S ("none: Sto-BADS is PyBADS's own"); the option descriptions and the docstring's reference.
- Kind: Python-only feature.
- Slice: S.

## O: third reader

No entry of its own. For this slice, see KD-B3-3 (the search hedge's reward is ported; the acquisition hedge is not), KD-B3-5, KD-B3-7, KD-B3-8, KD-B4-2, KD-B4-5, KD-B5-1, KD-B5-2, KD-B5-10 and KD-B6-1.

## Tests (no slice; for test-adequacy notes)

**KD-T-1. The optimization tests use MATLAB's `runtest.m` problems, but with tolerances set from seed sweeps**
- Python: `pybads/testing/bads/test_bads_optimization.py:1-17`.
- MATLAB: `private/runtest.m:11` (`tolerr = [0.1 0.1 1 1]`).
- What differs: each test's tolerance is ten times the largest error over seeds 0-99, rounded, or `runtest.m`'s tolerance when that is lower. `test_sphere_opt` uses `runtest.m`'s constraint and start point.
- Why: survey, sections "Tests that checked less than they appeared to" and "The seed sweep behind the tolerances".
- Kind: deliberate change.
- Slice: none.

## Claims that did not check out

Checked at `95da7f1`, whose lines they cite. Wave 0's rulings settled all eight (`verification/wave0.md`, W0-17 to W0-21): C1 is fixed in the code (W0-17: the noise test runs only when `uncertainty_handling` is left empty); the records of C2, C3 and C8 and the comments of C4 to C7 are corrected; the doubling of C2 stays, by the PI's ruling on W4-3 (`verification/wave4.md`, "Rulings"), and the seed that C3 describes is replaced by one draw of the run's generator (W4-1, KD-B7-1).

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
