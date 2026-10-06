# Slice part: B5, the GP training set and the refit policy

Title of the report: `# B5 <track> review: GP training set and refit policy`, with `<track>` "internal" or "comparison".

## The slice

Your slice is **B5, the training set of the local GP and the policy of its refits**: which points the GP is trained on, when its hyperparameters are refitted, how a fit is attempted and retried, and how a point is added between rebuilds. The GP model itself (kernel, mean, noise, their hyperpriors and bounds, the prior updates at each rebuild, the prediction) is slice B6, reviewed at the same time by others: read it where you need it to judge the policy, but report on it only where the policy depends on it.

Python (`{PYBADS_REVIEW}`):
- `pybads/bads/gaussian_process_train.py`: `local_gp_fitting` (the rebuild of the local GP: the training set, the refit branch, the retry with the previous hyperparameters, the restore after a failure, the markers `needs_rebuild` and `needs_refit`; its prior and bound updates are B6's), `get_grid_search_neighbors`, `add_and_update_gp`, `_robust_gp_fit_`, `_get_gp_training_options`, `_get_fevals_data`, `_estimate_noise_`;
- `pybads/bads/bads.py`: `_is_gp_refit_time_`, `_save_gp_stats_`, `_record_gp_refit_`, and the call sites of `local_gp_fitting` and `add_and_update_gp` in `_search_step_` and `_poll_step_` (read enough of both to judge when each is called);
- the options these read (grep the code), in `pybads/bads/option_configs/*.ini`.

MATLAB counterparts (`{BADS}`), for the comparison track:
- `private/gpupdate.m` (the whole file: `'nearest'`, `'add'`, the refit, the length scale, the poll scale, the effective radius);
- in `bads.m`, the subfunctions `IsRefitTime` and `savegpstats`, and the call sites of `gpupdate` in the search and poll stages;
- `utils/gppredcheck.m`, and `utils/swtest.m`, which PyBADS replaces by `scipy.stats.shapiro`;
- `utils/gpHyperOptimize.m`: the policy (starting points, their number and origin, the nudges after a failure, the removal of points after repeated failures, what a failure returns); `utils/minimizebnd.m` as far as the policy needs;
- `utils/update_posterior.m`: MATLAB's rank-1 update when a point is added; PyBADS does not take it (the sheet has the entry), so compare against its absence;
- unported and unused by MATLAB's defaults: `utils/gpHyperSVGD.m`, `utils/gpHyperSample.m`.

## How PyBADS reaches this code at default options

Every run trains a GP on the initial design (`init_and_train_gp`), then rebuilds the local GP around the incumbent at the first search of each round (`search_count == 0`) and after a move, refits its hyperparameters when `_is_gp_refit_time_` says so, and adds each new evaluation with `add_and_update_gp`. At default options the hyperparameters are optimized, not sampled (`gp_samples = 0`, the slice sampler off). A noisy run (uncertainty level 1 or 2) raises `n_train_min` and `n_train_max` and changes other options at the start of `optimize()`. Say for every finding whether a default run reaches it, at which uncertainty level, and which option or input reaches it otherwise.

## First questions

Answer each under its own heading:
1. **The training set.** Does `get_grid_search_neighbors` select the set of MATLAB's `'nearest'` method, point for point: the distance (`udist`, with which length scales), the number of points (`n_train_min`, `n_train_max`, the cap by the number of evaluations), the radius, the order and the ties? On the internal track: is the set what the options' descriptions and the BADS paper say it is?
2. **When to refit.** Does `_is_gp_refit_time_` decide as `IsRefitTime` and `gppredcheck` do: the minimum time between refits, the calibration test of the GP's predictions (its statistic, its test, its threshold), and what the decision leaves in the state? On the internal track: is the calibration test a correct test of what its comments say it tests?
3. **How a fit is attempted.** Does `_robust_gp_fit_`, with `_get_gp_training_options`, do what `gpHyperOptimize.m` does: the number and origin of the starting points, what each retry changes after a failure (the nudge of the noise's starting point and of its bound), when points are removed and which, and what the caller receives after the last failure?

---

The rest of the prompt is `wave1_common.md` (from "You are a reviewer") and the part of your track.
