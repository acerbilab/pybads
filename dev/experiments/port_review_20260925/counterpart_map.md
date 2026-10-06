<!-- Written by the preparatory agent of the port review (wave 0), reading PyBADS at ab4dded and MATLAB BADS at 74919c0; saved verbatim from its final message on 2026-09-25 (the three parts of that message are kept as known_differences.md, counterpart_map.md and prep_report.md). -->

# Counterpart map

Slice (table) is the slice the plan's table assigns. "—" means the table does not mention the file. "No caller" means no MATLAB file outside GPML calls it.

| MATLAB file | Python counterpart(s) | Slice (table) | Note / proposed |
|---|---|---|---|
| `acq/acqHedge.m` | unported, unused by MATLAB's defaults (reached only through `acqPortfolio` 'acq' with `AcqHedge` on) | B3 (other acq, unported?) | resolved: unported (KD-B3-3) |
| `acq/acqLCB.m` | `acquisition_functions/acq_fcn_lcb.py`: `acq_fcn_lcb` | B3, O | |
| `acq/acqNegEI.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `acq/acqNegEQI.m` | unported, unused by MATLAB's defaults (no caller) | B3 (unported?) | resolved |
| `acq/acqNegPI.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `acq/acqNegSqEI.m` | unported, unused by MATLAB's defaults (no caller) | B3 (unported?) | resolved |
| `acq/acqPortfolio.m` | 'upd' branch (reached at default, `bads.m:722-725`): `search/search_hedge.py`: `ESSearchHedge.update_hedge` (`bads.py:1871-1879`); 'acq' branch: unported | B3, O | |
| `acq/acqRnd.m` | unported, unused (testing only; no caller) | B3 (unported?) | resolved |
| `acq/private/acqNegEIMin.m` | unported, unused by MATLAB's defaults (`searchES.m:156` special-cases the name) | B3 (unported?) | resolved |
| `acq/private/acqNegGEI.m` | unported; no caller | B3 (unported?) | resolved |
| `acq/private/acqNegPIMin.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `acq/private/acqThompson.m` | unported; no caller | B3 (unported?) | resolved |
| `bads.m` | `bads/bads.py`: `BADS` (sections and subfunctions below); `defopts` → `option_configs/*.ini` | B1, B2, B3, B4, B5, O | |
| `bads_examples.m` | none in `pybads/` (the notebooks in `examples/` are PyBADS's own) | — | out of scope (examples) |
| `gpdef/gpdefBads.m` | `gaussian_process_train.py`: `_gp_hyp` (first definition, `gpdefBads.m:21-196`); `local_gp_fitting` `293-364` (update branch, `198-305`); `_cov_identifier_to_covariance_function`; `_meanfun_name_to_mean_function`; `init_and_train_gp` `96-106` (likelihood) | B6 | |
| `gpdef/private/gpdefStationaryNew.m` | unported; no caller | B6 | mark unused |
| `gpml_fast/ard_ratquad_covariance_fast.m` | unported; callers only `gpdefStationaryNew.m` and a test | — | B6, unused |
| `gpml_fast/covMaternard_fast.m` | unused by MATLAB's defaults; gpyreg `Matern` (identifier 3) is unreachable, since `gp_cov_fun` is hard-wired to 1 | — | B6, unused by defaults |
| `gpml_fast/covPPERard.m` | unported (periodic kernel, KD-B1-6) | — | periodic, unported |
| `gpml_fast/covPPERard_fast.m` | unported (periodic kernel, KD-B1-6) | — | periodic, unported |
| `gpml_fast/covRQard_fast.m` | gpyreg `RationalQuadraticARD` (`gpyreg/covariance_functions.py`), selected at `gaussian_process_train.py:807-808` | B6 | |
| `gpml_fast/covSEard_fast.m` | unused by MATLAB's defaults; gpyreg `SquaredExponential` (identifier 2) is unreachable | — | B6, unused by defaults |
| `gpml_fast/exact_inference_fast.m` | unported; callers only `gpdefStationaryNew.m` and a test | B6 | unused |
| `gpml_fast/exact_inference_robust.m` | unported; caller only `gpdefStationaryNew.m` | B6 | unused |
| `gpml_fast/infExact_fast.m` | unported; callers only `exact_inference_robust.m` and `test_fast.m` | B6 ("infExact_fast*") | unused |
| `gpml_fast/infExact_fastrobust.m` | gpyreg exact inference (`GP.update`, `GP.fit`, `gpyreg/gaussian_process.py`) | B6 | |
| `gpml_fast/infPrior_fast.m` | gpyreg's log prior in its objective (`GP._GP__gp_obj_fun`, `GP.fit`) | — | add to B6 (reached at default) |
| `gpml_fast/sq_dist_fast.m` | inside gpyreg's `RationalQuadraticARD` | — | add to B6 (reached at default) |
| `gpml_fast/test_extensions_fast.m` | none (test) | — | out of scope |
| `gpml_fast/test_fast.m` | none (test) | — | out of scope |
| `hetsphere.m` | none (`he_noisy_sphere` in `test_bads_optimization.py:244` is a similar target, with noise SD `2 + sqrt(f)`) | — | out of scope (example) |
| `init/initLHS.m` | unported (an `InitFcn` alternative, and `initSobol.m`'s fallback) | B7 (unported?) | resolved (KD-B7-2) |
| `init/initRand.m` | unported, unused by MATLAB's defaults | B7 (unported?) | resolved (KD-B7-2) |
| `init/initSobol.m` | `init_functions/init_sobol.py`: `init_sobol` | B7 | |
| `init/private/i4_bit_hi1.m` | `scipy.stats.qmc.Sobol` (substituted) | — (not matched by "i4_sobol*") | add to B7 |
| `init/private/i4_bit_lo0.m` | `scipy.stats.qmc.Sobol` (substituted) | — | add to B7 |
| `init/private/i4_sobol.m` | `scipy.stats.qmc.Sobol` (substituted) | B7 | |
| `init/private/i4_sobol_generate.m` | `Sobol(...).random_base2` (`init_sobol.py:69-76`) | B7 | |
| `init/private/lhs.m` | unported (through `initLHS.m`) | — | B7, unported |
| `init/private/tau_sobol.m` | none; no caller | — | unused |
| `install.m` | none (MATLAB path setup) | — | out of scope |
| `poll/pollGPS2N.m` | unported, unused by MATLAB's defaults | — | B4, unported (KD-B4-1) |
| `poll/pollMADS2N.m` | `poll/poll_mads_2n.py`: `poll_mads_2n` | B4, O | |
| `poll/private/pollBADS2N.m` | unported; no caller | B4 | unused |
| `poll/private/pollBMADS2N.m` | unported; no caller | B4 | unused |
| `private/bads_output.m` | `bads/optimize_result.py`: `OptimizeResult.set_attributes` | B1 | |
| `private/boundscheck.m` | `bads.py`: `BADS._bounds_check_` | B1 | |
| `private/checklist.m` | none (comments only: developer notes) | — | out of scope |
| `private/covmatadapt.m` | unported; no caller (its own comment: "currently unused"); nearest Python code is the `hessian_update` stub, `bads.py:1788-1793` | B4 | mark unused |
| `private/evalinitmesh.m` | `bads.py`: `BADS._init_mesh_` | B2 | |
| `private/fixedbads.m` | unported (`bads.py:413-423` refuses fixed variables) | B2 (unported?) | resolved (KD-B1-7) |
| `private/funlogger.m` | `function_logger/function_logger.py`: `FunctionLogger` (`__call__`, `_record`, `add`, `finalize`) | B7 | |
| `private/gpupdate.m` | `gaussian_process_train.py`: `local_gp_fitting` ('nearest' through `get_grid_search_neighbors`; the `gpfit` subfunction → the second-fit logic plus `_robust_gp_fit_`; geometry at `462-516`), `add_and_update_gp` ('add'); methods 'grid', 'nogrid', 'global' unported | B5 (geometry also O) | |
| `private/runtest.m` | `testing/bads/test_bads_optimization.py` (its problems) | — | tests |
| `private/scatterplot.m` | unported (`plot="scatter"` is a no-op) | B2 (unported?) | resolved (KD-B2-2) |
| `private/setupoptions.m` | `bads/options.py`: `Options`; `bads.py`: `__init__` `195-213`, `_init_optim_state_` `832-863` | B1 | |
| `private/setupvars.m` | `bads.py`: `_init_optim_state_` (transform, grid, `u0`, `tol_mesh`, periodic check, `fun_values`, state), `__init__` `253-264` (random `x0`); ES weights → `search/search_hedge.py:56-59` | B1 | |
| `rosenbrocks.m` | `function_examples.py`: `rosenbrocks_fcn` | — | out of scope |
| `search/private/searchSeries.m` | unported; no caller | B3 | unused |
| `search/private/searchWCMmix.m` | unported; no caller | B3 | unused |
| `search/private/searchWCMmultiscale.m` | unported; no caller | B3 | unused |
| `search/private/searchWCMscale.m` | unported; no caller | B3 | unused |
| `search/searchCMA.m` | unported (its only caller, `covmatadapt.m`, has none); not the counterpart of `ESSearchCMA`, which is `searchES` method 5 | B3 (unported?) | resolved |
| `search/searchCombine.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `search/searchCrossover.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `search/searchES.m` | `search/es_search.py`: `ESSearch.__call__`, `ESSearchWM` (method 1), `ESSearchELL` (method 2), `ESSearchCMA` (method 5, unreachable), `ucov`; methods 3 and 4 unported | B3, O | |
| `search/searchGauss.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `search/searchGrid.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `search/searchHedge.m` | `search/search_hedge.py`: `ESSearchHedge.__call__` | B3 | |
| `search/searchMax.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `search/searchMaxAcq.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `search/searchNewton.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `search/searchOptim.m` | unported, unused by MATLAB's defaults | B3 (unported?) | resolved |
| `search/searchWCM.m` | unported, unused by MATLAB's defaults; not the counterpart of `ESSearchWM` (that is `searchES` method 1) | B3 (listed as counterpart) | mismatch |
| `utils/ESupdate.m` | `search/es_search.py`: `ESSearch._get_selection_idx_mask_` (also called from `setupvars.m`) | B3 | |
| `utils/evalbool.m` | none (the `.ini` files hold Python literals) | — | add to B1 (reached through `setupoptions.m:43`) |
| `utils/fitnessTransform.m` | unported (KD-B5-5) | — | B5, unused by defaults |
| `utils/force2grid.m` | `search/grid_functions.py`: `force_to_grid` | B3 | |
| `utils/gpHyperOptimize.m` | `gaussian_process_train.py`: `_robust_gp_fit_` (retries, noise nudge, point removal) around gpyreg `GP.fit` | B5 (optimizer: B6) | |
| `utils/gpHyperSVGD.m` | unported, unused by MATLAB's defaults (`gpSamples = 0`) | B5 (unported?) | resolved (KD-B5-4) |
| `utils/gpHyperSample.m` | none; no caller | B5 | mismatch: unused, not a counterpart |
| `utils/gpgrad.m` | unported; unused by MATLAB's defaults | — | |
| `utils/gphess.m` | none; no caller | — | |
| `utils/gppred.m` | gpyreg `GP.predict` | B6 | |
| `utils/gppredcheck.m` | `bads.py`: `_is_gp_refit_time_` (inlined, `2372-2435`) | B4 | mismatch → B5 |
| `utils/gppriorrnd.m` | `gaussian_process_train.py`: `_get_random_samples_from_priors_` | B6 | |
| `utils/gpset.m` | gpyreg `GP` constructor, `set_bounds`, `set_priors` | B6 | |
| `utils/gpstruct_check_old.m` | none; no caller | — | |
| `utils/gridunits.m` | `search/grid_functions.py`: `grid_units` (`bads.py:677-680`) through `VariableTransformer.__call__` | B1 | its Python file is assigned to B7 |
| `utils/hessianapprox.m` | none; no caller | — | |
| `utils/landscapeplot.m` | unported (`plot="profile"`) | — | out of scope (plotting) |
| `utils/likGaussHe.m` | gpyreg `GaussianNoise` (`gaussian_process_train.py:97-106`), with target SDs squared into `s2` (`1140`, `1169`, `1253`) | B6 | |
| `utils/maskindex.m` | `variable_transformer/variables_transformer.py`: `maskindex` | — | add to B1 (reached through `transvars.m`) |
| `utils/mcs_gp_optimizer.m` | none; referenced only in a disabled block of `gpHyperOptimize.m` | — | |
| `utils/minimizebnd.m` | gpyreg `GP.fit`'s optimizer; MATLAB uses `minimizebnd` only without the Optimization Toolbox, `fmincon` otherwise | B6 | |
| `utils/mygp.m` | gpyreg `GP.update` / `GP.predict` | B6 | |
| `utils/origunits.m` | `VariableTransformer.inverse_transf` | B1 | |
| `utils/periodCheck.m` | `utils/period_check.py`: `period_check` (a stub) | B7 | |
| `utils/prctile1.m` | `np.percentile(..., method="hazen")`, `gaussian_process_train.py:316-318` | — | add to B6 (reached at default) |
| `utils/private/fminbayes.m` | none; no caller | B6 | mismatch: unused |
| `utils/swtest.m` | `scipy.stats.shapiro` in `_is_gp_refit_time_` (`bads.py:2434-2435`) | — | add to B5, with `gppredcheck.m` |
| `utils/transvars.m` | `variable_transformer/variables_transformer.py`: `VariableTransformer` | B1 | |
| `utils/uCheck.m` | `function_logger/constraints_check.py`: `contraints_check` | B3 | |
| `utils/ucov.m` | `search/es_search.py`: `ucov` | B3 | |
| `utils/udist.m` | `search/grid_functions.py`: `udist` (callers in B3, B5, B6) | B3 | |
| `utils/ugdist.m` | none; no caller | — | |
| `utils/update_posterior.m` | none (the rank-1 update is not taken, KD-B5-1) | B5 | no counterpart |
| `utils/weightedsum.m` | none (one hyperparameter set); unused at MATLAB's defaults | — | |
| `utils/xCheck.m` | none; no caller | B3 | mark unused |
| `warp/infExactWarp.m` | unported (KD-B6-4) | — (plan: out of scope) | |
| `warp/likGaussWarpExact.m` | unported (KD-B6-4) | — | |
| `warp/warpLog.m` | unported; no caller | — | |
| `warp/warpPower.m` | unported (KD-B6-4) | — | |
| `warp/warpSqrt.m` | unported; no caller | — | |

**`bads.m`, by section and subfunction**

| `bads.m` lines | Python | Slice |
|---|---|---|
| `149-290` `defopts` | `option_configs/basic_bads_options.ini`, `advanced_bads_options.ini` | B1 |
| `302-382` inputs, display level, `x0`, fixed variables | `__init__`, `_bounds_check_`; fixed variables unported | B1 |
| `384-412` options, variables, logger | `__init__`, `_init_optim_state_`, `FunctionLogger` | B1 |
| `414-428` initial evaluations | `_init_mesh_` | B2 |
| `430-479` changes under uncertainty, `fsd`, incumbent, GP definition, `gpstats` | `_init_optimization_`; `init_and_train_gp` (which fits a GP; MATLAB only defines one) | B2, B6 |
| `481-510` loop head, mesh, sufficient improvement | `optimize` `1216-1261` | B2 |
| `511-740` search stage | `_search_step_`; portfolio update → `update_hedge` | B3 |
| `742-764` poll decision | `optimize` `1280-1316` | B2 |
| `767-1046` poll stage | `_poll_step_`, `_is_poll_stop_` | B4 |
| `1048-1133` iteration end, termination, `iterList`, re-evaluation | `optimize` `1324-1458` | B2 |
| `1135-1162` final re-evaluation and estimate | `optimize` `1462-1537` | B2, O |
| `1164-1194` output function, output struct | `optimize` `1208-1212` (at start only); `OptimizeResult` | B2, B1 |
| `savegpstats` `1199` | `_save_gp_stats_` | B5 |
| `IsRefitTime` `1223` | `_is_gp_refit_time_`, `_record_gp_refit_` | B5 |
| `EvalImprovement` `1257` | `_eval_improvement_` | B4, O |
| `UpdateIncumbent` `1285` | `_update_incumbent_` | B4 |
| `UpdateTarget` `1296` | `_get_target_from_gp_` | B4 |
| `UpdateSearch` `1342` | `_update_search_stats_` | B3 |
| `reevaluateIterList` `1378` | `_re_evaluate_history_` | B2 |
| `updateSearchBounds` `1417` | `_update_search_bounds_` | B3 |
| `meshOverflowCheck` `1429` | `_check_mesh_overflow_` | B2 |
| `FinalEstimate` `1443` | inline in `optimize` `1494-1537` | B2 |
| `expandvars` `1480` | unported (KD-B1-7) | B2 |
| `add2path` `1490` | none (MATLAB path) | — |
