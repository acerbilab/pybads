<!-- Report of fix agent E of wave 4's fix pass (the completion of W4-6, after W4-6's pick failed test_get_gp_training_options_small_budget), working in its own worktree on w4-fix-E from e7bd01d; saved verbatim from its final message on 2026-09-27. The commit hash is that of its branch; its scripts are copied to verification/scripts/wave4/fix_E/. -->

# Fix report: W4-6 (completion)

## Commit `faa7501`: fix: the GP's fit schedule counts its budget without the noise test, as it counts the evaluations used (W4-6)

Worktree `/home/user/pybads-fix-E`, branch `w4-fix-E`, one commit on `e7bd01d`. The pre-commit hooks passed.

**Files changed**
- `/home/user/pybads-fix-E/pybads/bads/bads.py` (`_init_mesh_`, +5 lines)
- `/home/user/pybads-fix-E/pybads/bads/gaussian_process_train.py` (`_get_gp_training_options`, +8/-3 lines)
- `/home/user/pybads-fix-E/pybads/testing/bads/test_gaussian_process_train.py` (+37 lines)

**What changed and why**
- **The bug left by `e7bd01d`:** the numerator of the budget fraction (`n_eff - eff_starting_points`) no longer counted the noise test. The denominator (`min(max_fun_evals, n_train_max) - eff_starting_points`) still counted it, because `max_fun_evals` counts evaluations (since W2-27). So a budget that x0, the noise test and the design had used up read as unused: x = 0, and `init_N` was 128 instead of 8.
- **The fix:** `_init_mesh_` records `optim_state["n_noise_test"]`: 1 in the noise-test branch, 0 in the `else` branch (which covers `uncertainty_handling` set to True or False). The budget becomes `n_budget = min(max_fun_evals - optim_state["n_noise_test"], n_train_max) - eff_starting_points`, so it counts points, as `n_eff` and `eff_starting_points` do.
- **Why a recorded count:** the code that runs the test keeps it, beside `eff_starting_points`, which `_init_mesh_` also records for this schedule. I considered two alternatives:
  - `func_count - sum(n_evals[X_flag])` gives the same number today, but only while every other evaluation adds to some row's count (final samples and merges both do).
  - Testing `options["uncertainty_handling"] is None` again would repeat `_init_mesh_`'s condition in a second place.
- **Two other tests edited:** `test_get_gp_training_options_samplers` and `_opts_N` call `_get_gp_training_options` without `_init_mesh_` and set `eff_starting_points` by hand. They now set `n_noise_test = 1` beside it, which is what `_init_mesh_` records for their default `uncertainty_handling=None`.

**Tests**
- **`test_get_gp_training_options_small_budget`**, unchanged:
  - At the parent `e7bd01d`: `8 failed, 54 deselected`, each failing at `assert all(n == bads.options["gp_train_n_init_final"] for n in seen)`.
  - At the commit: all 8 pass.
- **New test, `test_get_gp_training_options_budget_counts_points[None|False]`.** I added it because the existing test covers only a count of 1. At a budget no larger than the design, a count of 0 (or a hard-coded `-1`) also gives x = 1, so that test can't tell them apart.
  - Setup: D = 2, `max_fun_evals = 10`, after `_init_optimization_`. It adds seeded points to the log and calls `_get_gp_training_options` after each one.
  - What it checks: `init_N` equals `gp_train_n_init_final` at the last point the budget allows (the 9th with the noise test, the 10th without) and is larger one point before.
  - At the parent, run in a scratch copy of `e7bd01d` (imports checked): `FAILED ...budget_counts_points[None]` with `assert 9 == 8`; `1 failed, 1 passed`.
  - At the commit: both pass.
- **Whole file** `test_gaussian_process_train.py` at the commit: `64 passed`. This includes the `refit_case` fixture, whose run (`max_fun_evals=70`, D = 3) now moves.
- **`test_run_control.py -k "one_function or noise_test or initial_design"`:** 13 passed.

**Fingerprint at the commit:** `4146a986863602cb` (Linux, gpyreg v1.3.3, one BLAS thread, imported from `/home/user/pybads-fix-E/pybads/__init__.py`). The parent gives `c91725823bc62b29`.
- **Which runs move:** the fingerprint's runs use `max_fun_evals = 80`, which equals `n_train_max` at D = 3.
  - The three deterministic runs take the noise test, so their budget falls from 80 points to 79, and they move. For example, the `init_N` sequence of seed 0 goes from `[128, 55, 34, 20, 12, 9, 8]` to `[128, 55, 33, 20, 12, 9, 8]`.
  - The three noisy runs (`uncertainty_handling=True`, no noise test) are identical run for run.
- **Control:** with `max_fun_evals = 81`, all six runs are identical at the parent and at the commit.
- **Default runs:** `n_budget` changes only when `max_fun_evals - 1 < n_train_max`. That never holds at the default budget (500 D against 50 + 10 D, and noisy runs raise `n_train_max` to at least 200), so no default run moves.
- Logs are in the scratch directory `/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad/wave4/fix_E/`: `per_run_parent.log`, `per_run_fix.log`, `per_run_81_*.log` and the parent test outputs.

**Proposed changelog line:** no separate line, since this fix belongs to W4-6's entry, which no release has shipped. Optionally, reword its first sentence: "The noise test, the second evaluation of the starting point when `uncertainty_handling` is left empty, no longer counts in the schedule of the hyperparameter fits of the Gaussian process, neither among the evaluations used nor in the budget, whose fraction used after the initial design sets their number of starting points; in 1.1.0 the schedule ran one evaluation ahead." No "Upgrading from" line: the existing "Results differ from 1.1.0" covers it.

**Uncertain**
- `optim_state["n_noise_test"]` is now a required key of `_get_gp_training_options`, as `eff_starting_points` already is. Any caller that builds `optim_state` without `_init_mesh_` must set it.
- A gate configuration with the noise test and `max_fun_evals <= n_train_max` (50 + 10 D) will move. The default suite does not.

## Noticed outside the row
- The `_get_gp_training_options` docstring has a typo in the return type (`gp_train : dic`). It doesn't document `function_logger` or `second_fit`, and it lists `hyp_dict`, which the body never reads.
- `test_get_gp_training_options_samplers` and `_opts_N` assign `hyp_dict_none` and never use it.
