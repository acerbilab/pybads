<!-- Report of reviewer (a) of the doublecheck of wave 4, the fixes of B7 (briefs/wave4_doublecheck.md), reading PyBADS at 81385ac in /home/user/pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave4/doublecheck/a_B7/. -->


# Wave 4 doublecheck: the fixes of B7

## 1. Coverage

**Read in full**
- The brief (the common part and scope (a)).
- `verification/wave4.md`, the whole ledger.
- The fix agents' reports: `fixes/B_…W4-6.md`, `fixes/C_…W4-28.md` (all of it; the W4-14, `periodic_vars` and W4-7 parts are in my scope) and `fixes/E_W4-6_completion.md`.
- The code at `81385ac`:
  - `init_sobol.py`;
  - `function_logger.py`;
  - in `bads.py`: `__init__` (periodic check, random `x0`), `_init_optim_state_` (lines 930-1098), `_variable_transformer_`, `_init_mesh_`, `_init_optimization_`, and `optimize` from the start through the final estimate;
  - `_get_gp_training_options`;
  - `optimize_result.py`;
  - the `.ini` diff.
- The commit diffs of `8daf7ad`, `5dd92b7`, `29a258a`, `f1247d0`, `b61a880`, `b78f782`, `efe5e95`, `e7bd01d`, `46af65a` and `3a8096b`, and the messages of `4ea665a`, `2dc5807`, `fba29cd` and `e744ed9`.
- The pass's diffs of the seven test files in scope.
- The gate files: `w4-1_vs_w4-21.md`, `w4-6_vs_w4-1.md`, the three `geometry_*` comparisons of W4-1 and W4-6, the `_fields.txt` and `_changed.txt` files, `geometry_edgesphere_D2_steps.txt`, and `w41_fixed_designs.py` with its `.out`.
- MATLAB at `74919c0`: `funlogger.m`, `initSobol.m`, `evalinitmesh.m`, `setupvars.m:95-125`, `bads.m:440-452`, `bads.m:1120-1200` and `FinalEstimate` (`bads.m:1443-1480`).

**Skimmed:** the B7 verifier's report (sections F and N, and §3); the diffs of `AGENTS.md`, `dev/TODO.md` and the survey; KD-B7-1; the changelog's W4-14 entry; the W4-14 entry of `matlab_side_defects.md`; the plan's wave-4 worklog lines.

**Not reached:** the B7 reviewers' own reports, `i4_sobol.m`, and the fix agents' scripts. I ran no suite, no `fingerprint.py` and no `population.py`.

**Runs**, one at a time with one BLAS thread:
- 60 full runs of `ellipsoid_D3_homo`, the allowance.
- 150 runs of `edgesphere_D2` and 2 of `sphere_band_D2`.
- Small checks.

Everything is in `/home/user/dc4/a_B7`: the scripts `q1`–`q8`, `runcfg.py`/`runcfg.sh`, `q3_*` and `es_summary.py`, their `.out` files, and the per-run `*.jsonl` records. The `at_<rev>/` directories are `git archive` extractions; they can be dropped when the directory is copied.

## 2. What holds

**The rows**

- **W4-1 (`efe5e95`).** The seed is `get_rng(rng).integers(2**63)` (`init_sobol.py:61`), and the cast to `uint64` is gone: no `uint64` is left under `pybads/init_functions`, which settles W4-2 in the code.
  - The generator advances by exactly one `integers(2**63)` draw: 0 mismatches over D from 1 to 40 and several sizes, directly and inside `BADS` (`q4_rng_design.out`).
  - NumPy's global state is untouched in seeded runs at levels 0 and 1.
  - Seeds 42 and 43 give two designs. For each seed, six given starts give the same design: inside the box, on `plb`, below `plb`, near `lb`, and above `pub`.
  - Caveat: a random `x0` (`None`) is drawn from `bads.rng` before the design, so the same seed then gives another design. It is still decided by the seed. No statement in my scope says otherwise (see section 4).
  - `test_seeded_run_leaves_global_state_untouched`: all four of its configurations reach `init_sobol` (D = 3, 60 evaluations; a 4-point design, 32 points at level 1). The old int-seeded scipy draw did not touch the global state either, so this test passes before and after W4-1 alike.
  - scipy 1.17.1's `Sobol(seed=int)` raises no warning.
  - Comparison with MATLAB: `initSobol.m:9-15` seeds a skip index from `num2str(u0(1:10))`. The departure is recorded as such.
- **W4-2 (records only).** The cast is gone (above). The code change is exactly `3a8096b` plus W4-1.
- **W4-3 (records only).** The doubling is kept (`init_sobol.py:70-72`). Design sizes equal `2**ceil(log2(n))`, doubled at D, for D = 1, 2, 3, 4, 8, 16 at levels 0 and 1, and at small and larger budgets (`q8_design_periodic.out`).
- **W4-4 (`8daf7ad`).**
  - The second value is `samples.shape[0]`.
  - The docstring is true at every level. It states the power of two and the doubling at D. The cut to the budget is the caller's and is stated there: the comment at `bads.py:1236-1237` and the description of `fun_eval_start`, both true (`q8`).
  - Only `u0.size` is read, and `self.u` has shape `(D,)` (`bads.py:718`).
- **W4-5 (records only).** The merge code is unchanged, and no run reaches it. In six seeded level-2 runs (200 evaluations, `max_iter=1`, and `max_fun_evals=40`, which includes W4-14's new path) there were 0 merges (`q7_merge.out`). `add` has no caller.
- **W4-6 (`e7bd01d`, completed by `46af65a`)** — the PI's question.
  - Every path of `_init_mesh_` sets `n_noise_test` before the one reader, `_get_gp_training_options`. The count is 1 in the noise-test branch (set before the comparison, so also when the test finds noise) and 0 otherwise. The paths checked:
    - a deterministic start;
    - `uncertainty_handling` set to True or False;
    - `specify_target_noise`, which sets `uncertainty_handling=True`, so no test runs;
    - a start that the noise test finds noisy;
    - `max_fun_evals` from 1 to 5;
    - `output_fcn` stopping the run at `"init"` (the initial fit reads the count after it is set);
    - `fun_eval_start=0`.
  - At every call of the schedule in those runs, `func_count - n_noise_test == n_eff == points` (0 mismatches, `q1_noise_test_paths.out`). The noisy budget is cut by the final samples before the first fit, so `min(max_fun_evals - n_noise_test, n_train_max) - eff_starting_points` counts exactly what `n_eff` and `eff_starting_points` count.
  - The completion is right and complete. No other caller builds `optim_state` by hand except the two tests that now set the count.
  - `max_fun_evals=1` still makes two evaluations; this is shared with MATLAB (`evalinitmesh.m:38-47` runs the test before the check at 86) and already recorded (sheet line 158; `CHANGELOG.md:429`).
  - The ledger records `e7bd01d`'s failure truthfully: the body of `test_get_gp_training_options_small_budget` fails in all 8 cases at `e7bd01d` (D = 2, 3; `max_fun_evals` 2-5; `init_N` 128 instead of 8) and passes at `46af65a` (`q1b_*.out`).
  - `e7bd01d` against MATLAB: `evalinitmesh.m:41-42` calls the target directly.
- **W4-7 (records only; `e744ed9`).** The description of `overhead` is true. The noise test's time is taken back out of `total_fun_eval_time` (`bads.py:1167-1175`). MATLAB never times the test either: `funwrapper` is called directly, the `totalfunevaltime` accumulation is at `funlogger.m:130`, and the ratio is at `bads_output.m:48`.
- **W4-8 (`5dd92b7`).**
  - Well-formed outputs that 1.1.0 accepted are still accepted: Python and NumPy scalars (float, float32, int, bool) and 0-d, 1-element and (1,1) arrays, lists and tuples. At level 2 this includes `[0.5]`, which 1.1.0 refused with `TypeError`.
  - Every malformed value and SD raises `ValueError`, and leaves `Xn`, `X_max_idx`, `func_count`, `cache_count`, `X`, `Y`, `X_orig`, `Y_orig`, `S`, `n_evals`, `fun_eval_time`, `total_fun_eval_time`, `X_flag` and `Y_max` exactly as they were. The malformed cases tried: several elements, a string, `None`, Python and NumPy complex, NaN, ±inf, a tuple, empty inputs, a dict, an object, a ragged list, a huge int, and an SD of 0 or negative.
  - None of these refusals carries the target note (`q5_logger.out` against `q5_logger_v110.out`).
  - The checks match `funlogger.m:102-110`, including `true` accepted as a value or an SD.
- **W4-9 (`29a258a`).** After growth, `finalize`, a later call and `reset_fun_eval_time`, every array has the same length, at levels 0 and 2 and on an empty log (`q5`).
- **W4-10 (`4ea665a`).** The docstrings are true: `x` is taken in `u` space (squeeze, then `inverse_transf`), and `add` records a missing SD as 1 and drops a given one when the logger holds no SDs. The TODO line on `add` with `fun_values` is present.
- **W4-11 (records only).** `period_check` is a stub that returns `x`; its call sites are unchanged. The TODO line on the argument's form is present.
- **W4-12 (`f1247d0`).** "Grows by half" is true (`_expand_arrays` adds `ceil(Xn/2)` rows). The ring in `funlogger.m:120` never writes its last row.
- **W4-13 (records only).** The noise test still goes through the logger's checks (`bads.py:1170`).
- **W4-14 (`b61a880`).** In all four ways a run can end within its first iteration, at levels 1 and 2 (`q6_w414.out`):
  - a budget of 38;
  - `max_iter=1`;
  - the mesh (`tol_mesh=2`, status 1);
  - `output_fcn` returning True at `"iter"`.

  In each of these:
  - the reserved samples are taken at `result.x`;
  - `fval`/`fsd` equal their mean and standard error (precision-weighted at level 2);
  - `iteration_history[0]` holds the estimate;
  - a single sample appends `yval`.

  The `"init"` stop takes none (`iter == -1`). MATLAB takes none at `iter == 1` (`bads.m:1138`), and `FinalEstimate` (`1443-1480`) is transcribed exactly. The one gap is F1.
- **`periodic_vars` (`b78f782`).** The refusal (`bads.py:328-333`) runs before the first transform, both for a given `x0` (the transform is at `:656`) and for a random one (`:341`). `[5]` with a random `x0` now raises `ValueError`. `[]`, `np.array([])` and `()` become `None` with an all-False mask, as with `isempty` in `setupvars.m:107-108` (`q8`).

**The gates**

- **W4-1:** `86512c9` against `efe5e95` on the default suite, i.e. W4-21's step as ruled. Every run changed, so the gate reached the changed code. The ledger's reading matches the file: 0.63 → 0.43 with +0.268 [-0.221, +0.860]; 0.40 → 0.60; 0.10 → 0.23.
- **W4-6:** `efe5e95` against `46af65a`. 390 runs changed across 13 configurations, and the five noisy configurations, which set `uncertainty_handling=True` in `benchmark_targets.py:905`, are identical. The ledger's reading matches.
- **Geometry:** the `edgesphere_D2` flag is KS 0.6 with Holm p 0.000497, and the error p is 0.393, as the ledger says.

**The PI's questions**

- **`edgesphere_D2`.**
  - At `86512c9` all 30 runs used Sobol seed 948.
  - At `efe5e95` with the seed forced to 948 and no draw, the runs equal W4-21's in 30 of 30 (`func_count`, `x`, `fval`).
  - W4-1 as it is gives 46-55 evaluations, mean 48.5, and my reruns of fixed seeds 2 and 500 reproduce the `.out` (`es_summary.out`).
  - The `.out`'s "948+draw" line (same design, with W4-1's draw) gives 948's counts, so the design sets the counts, not the draw.
  - The ledger's attribution holds, apart from F2.
- **`ellipsoid_D3_homo`.** Nothing beyond the change of trajectories explains the fall. The references compared (`q3_records.out`):
  - `x0` are identical across the two references;
  - 13 runs go from solved to unsolved and 4 from unsolved to solved;
  - the tolerance of 0.1 sits at the median error (0.068 → 0.119, with a noise SD of 1 and `fsd` ≈ 0.3), and 13-14 of the 30 errors lie within a factor of 2 of it;
  - messages, evaluations and `fsd` are similar.

  My runs (`q3_analyze.out`):
  - W4-1's code with 967 forced equals `86512c9` on seeds 0-3 and reproduces W4-21's gate for this configuration: 0.63 solved, median 0.0641, 13 runs changed, 9 at other points. The intervening commits therefore moved nothing here.
  - `efe5e95` as it is equals the wave-4 reference records on seeds 0-1.
  - Over seeds 0-23, W4-1's one extra draw with the design held at 967 flips 10 of 24 solved flags (fraction 0.58 → 0.58, log10 ratio +0.04, p = 0.57). The design change alone flips 10 of 24 (0.58 → 0.50, +0.01, p = 0.86).
  - W4-1 against W4-21 over 30 seeds: +0.27, p = 0.13. The whole pass: +0.39, p = 0.088.
  - The solved flag is re-drawn by any perturbation, so the 0.73 → 0.43 fall is within that spread.

## 3. Findings

### F1. After W4-14, a noisy run that ends within its first iteration without reserved samples still reports a default `fsd`
- Where:
  - `pybads/bads/optimize_result.py:30-36` at `81385ac`;
  - `dev/experiments/port_review_20260925/verification/wave4.md:338-340` ("…closes with it, and so does the default `fsd` of such a run");
  - `dev/TODO.md`, whose small-budget item was removed whole.
- Kind: false statement
- Severity: minor
- What is stated, and what is true:
  - W4-14 takes samples only when some were reserved.
  - A run that ends in its first iteration with none reserved takes none, and the level-1 `fsd` is `noise_size` (1.0) or the level-2 SD at the incumbent. Two cases give this:
    - a `max_fun_evals` that the start, the test and the design use up: at D = 2, 33 with `uncertainty_handling=True`, or 34 with the noise test;
    - `noise_final_samples=0`.
  - Evidence (`q6_w414.out`): "budget 33 (none reserved) L1 it=1 … fsd=1.0"; "noise test finds noise, budget 34 … fsd=1.0"; "noise_final_samples 0, max_iter 1 L1 … fsd=1.0" (L2 0.5).
  - The `fsd` description (written by W4-30) names only the `"init"` stop, and the ruling says the default `fsd` closes. MATLAB is the same (`bads.m:448-452`, `1138`).
- Would the correction move results: no, it is a docstring and records. A code change would move only such runs.
- Proposed correction for the `fsd` description: "For a noisy run that takes no final samples and ends before or within its first iteration (`output_fcn` stopping it at `"init"`, `noise_final_samples = 0`, or a `max_fun_evals` that leaves no evaluation after the initial design), it is not an estimate: `noise_size` without `specify_target_noise`, and otherwise the standard deviation that the target returned at the incumbent." Also flag the ledger line where it stands, or keep a TODO line for this shared gap.

### F2. The ledger says each fixed `edgesphere_D2` design has "a narrow band of its own"; the `.out` does not support it
- Where: `dev/experiments/port_review_20260925/verification/wave4.md:501` (the "W4-6, completed" row).
- Kind: false statement
- Severity: minor
- What is stated, and what is true:
  - The `.out` ranges are 47-53, 46-53, 46-53, 48-55, 48-49 and 49-49 (seeds 1, 2, 3, 500, 997, 12345).
  - My rerun of seed 2 (`es_w41_force2.jsonl`): 46-53, SD 2.25 against 2.34 for W4-1's 30 designs and 0.56 for 948; 17 of 30 runs at 46, below 948's median of 47.
  - Seed 500: 27 runs at 48 and 3 at 52-55.
  - "948 was the cheapest" holds only for the mean.
  - The attribution itself stands: 948 was a single, low, narrow design, and 948+draw shows that the design sets the counts.
- Would the correction move results: no.
- Proposed correction: "…six other fixed designs cost 47.6 to 49.3 evaluations on average, with spreads from a single count (seed 12345) to 46-53 (seed 2) and 48-55 (seed 500): 948 was the cheapest of the seven on average and the narrowest, and with W4-1's draw kept (948+draw) its runs keep 948's counts, so the design, not the draw, sets them…"

## 4. Outside my scope
- (d) `AGENTS.md` ("the seed decides the design, whatever the start"), KD-B7-1 ("whatever the start") and the survey's row ("the start does not"): with a random `x0` the same seed gives another design, because the draw of `x0` comes first (`q4_rng_design.out`). They hold for every given start.
- (c) The changelog's W4-8 entry and its "Upgrading" line say that 1.1.0 "accepted a NumPy complex value". 1.1.0 also accepted a NumPy complex SD at level 2 (`q5_logger_v110.out`: `np.complex128(0.5+0j)` gives `S=0.5`).
- (b)/(d) The W4-21 fix-pass row says "34 runs changed, 25 of them ending at other points (ellipsoid_D3_homo 13, …)". The counts in the parentheses are runs changed and sum to 34. `w4-21_changed.txt` gives `ellipsoid_D3_homo` 13 changed, 9 of them at other points, and my reproduction agrees. Brief (b)'s "13 runs ending at other points" repeats the ledger's wording.
