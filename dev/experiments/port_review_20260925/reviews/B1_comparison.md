<!-- Report of the B1 comparison reviewer (setup, options, defaults, bounds, transform and result, MATLAB-comparison track), wave 2 of the port review, reading PyBADS at fef6c14 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave2/B1_comparison/. Nothing in it is verified. The sandbox's clone was shallow (oldest commit ce3a0b3) for part of this reviewer's run, which its "History note" describes; the complete history was fetched during the run, and c7c88ab is an ancestor of fef6c14. -->

# B1 comparison review: setup, options, defaults, bounds, transform and result

PyBADS at `fef6c14` (`/home/user/pybads-review`), MATLAB BADS at `74919c0` (`/home/user/bads`), gpyreg v1.3.3. Every script printed `pybads.__file__ = /home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__ = /home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. The scripts and their outputs are in `/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/review/B1_comparison/`: `opts_dump.py`, `check_bounds.py`, `check_transform.py`, `check_effbounds.py`, `check_effbounds2.py`, `check_narrow.py`, `check_tolnoise.py`, `check_searchfactor.py`, `check_rng.py`, `check_result.py`, `check_options.py`, `check_fvals.py` and `check_deepcopy.py`, each with a matching `.out` file.

## 1. Coverage

**Read completely**
- Python, `pybads/bads/bads.py`: lines 1–1010 (docstring, `__init__`, `_bounds_check_`, `_init_optim_state_`, `_variable_transformer_`, `_init_rng_`). I also read what the setup feeds or what reads it: `_init_mesh_` and `_init_optimization_` (1010–1247), the end of `optimize()` (1556–1676), `_update_search_bounds_`, `_update_search_stats_`, and the display helpers (2836–2894).
- Python, the rest of the slice: `pybads/bads/options.py`; both `.ini` files, `options_confs.py` and the two test `.ini` fixtures; `pybads/variable_transformer/variables_transformer.py`; `pybads/search/grid_functions.py`; `pybads/rng.py`; `pybads/bads/optimize_result.py`; every package `__init__.py`.
- Python, `FunctionLogger.__call__` and `_record`, only for how `func_count` and `total_fun_eval_time` are counted.
- Docs: `docsrc/source/api/classes/bads.rst`, `optimize_result.rst`, `options.rst`, `parameter_transformer.rst`, `api/options/bads_options.rst`, `quickstart.rst`; the `CHANGELOG.md` entries that touch this slice.
- MATLAB, `bads.m`: 1–480, 1040–1200, 1342–1376 (`UpdateSearch`), 1417–1528.
- MATLAB, the rest: `private/setupoptions.m`, `setupvars.m`, `boundscheck.m`, `bads_output.m`; `utils/transvars.m`, `origunits.m`, `gridunits.m`, `maskindex.m`, `evalbool.m`, `force2grid.m`; `private/evalinitmesh.m` 1–140; `private/funlogger.m` 1–160.
- The history of every line cited below, on both sides.

**Skimmed**
- `search/searchES.m` and `es_search.py`, for the role of the search factor only.
- The tests: `test_variable_transformer.py`, `test_bads_seed.py`, `test_run_control.py`, `test_noisy_runs.py` (names), `test_bads_optimization.py` (bounds only).

**Not reached**
- `private/fixedbads.m` and `expandvars`: unported, KD-B1-7.
- The main loop beyond what the setup feeds it.
- MATLAB VBMC's `boundscheck.m`, which PyBADS's `_bounds_check_` appears to follow (see F1). It is not in this environment.

**History note.** PyBADS's pre-restructure history (`9e63b0f` → `c7c88ab` → `8e59038` → `9037851`) is not an ancestor of `ce3a0b3` in the worktree, so `git log -L` stops at `ce3a0b3`. I dated older lines with `git show <c7c88ab|8e59038|9037851>:<path>` in `/home/user/pybads`. Every MATLAB line cited below predates 2022-02-11 unless stated.

## 2. Answers to the first questions

### Q1. The options and their defaults

**Defaults table.** Each MATLAB `defopts` entry (`bads.m:148-161` basic, `187-290` advanced) is listed against its PyBADS counterpart. The values were evaluated at D = 1, 2, 6, 20 on the PyBADS side (`opts_dump.out`) and by hand on the MATLAB side. "Read" means some PyBADS code outside `testing/` reads the option.

D-dependent entries. All are equal on both sides and read by PyBADS unless noted.

| MATLAB | PyBADS | D = 1 / 2 / 6 / 20 |
|---|---|---|
| `MaxIter` 200·nvars | `max_iter` | 200 / 400 / 1200 / 4000 |
| `MaxFunEvals` 500·nvars | `max_fun_evals` | 500 / 1000 / 3000 / 10000 |
| `TolStallIters` 4+floor(nvars/2) | `tol_stall_iters` | 4 / 5 / 7 / 14 |
| `Ninit` nvars | `fun_eval_start` | 1 / 2 / 6 / 20 (same value; the design size differs, KD-B7-1 and C2) |
| `Nbasis` 200·nvars | `n_basis` | 200 / 400 / 1200 / 4000 (unread on both sides at default, KD-B1-5(c)) |
| `TolPoI` 1e-6/nvars | `tol_poi` | 1e-6 / 5e-7 / 1.67e-7 / 5e-8 |
| `MeshOverflowsWarning` 2+nvars/2 | `mesh_overflow_warning` | 2.5 / 3 / 5 / 12 |
| `SearchNtry` max(nvars, floor(3+nvars/2)) | `search_n_try` | 3 / 4 / 6 / 20 (a NumPy float in PyBADS) |
| `Ndata` 50+10·nvars | `n_train_max` | 60 / 70 / 110 / 250 |
| `MinRefitTime` 2·nvars | `min_refit_time` | 2 / 4 / 12 / 40 |
| `HedgeDecay` 0.1^(1/(2·nvars)) | `hedge_decay` | 0.3162 / 0.5623 / 0.8254 / 0.9441 |

D-independent entries whose values match and that PyBADS reads:
- Basic: `Display`, `NonlinearScaling`, `CompletePoll`, `AccelerateMesh`, `OutputFcn` (advanced in PyBADS), `UncertaintyHandling`, `NoiseSize`, `SpecifyTargetNoise`, `NoiseFinalSamples`.
- Tolerances and initialization: `TolMesh`, `TolFun`, `InitFcn`.
- Poll: `PollMeshMultiplier`, `ForcePollMesh`, `AlternativeIncumbent`, `AdaptiveIncumbentShift`, `gpRescalePoll`, `ConsecutiveSkipping`, `SkipPollAfterSearch`, `MinFailedPollSteps`, `AccelerateMeshSteps`, `SloppyImprovement`.
- Improvement: `TolImprovement`, `ForcingExponent`, `IncumbentSigmaMultiplier`, `ImprovementQuantile`, `FinalQuantile`.
- Search: `Nsearch` (4096), `Nsearchiter`, `ESbeta`, `ESstart`, `SearchScaleSuccess`, `SearchScaleIncremental`, `SearchScaleFailure`, `SearchMethod` (KD-B1-2), `SearchGridNumber`, `MaxPollGridNumber`, `SearchGridMultiplier`, `SearchSizeLocked`, `SearchMeshExpand`, `SearchMeshIncrement`.
- GP: `MinNdata`, `BufferNdata`, `PollTraining`, `DoubleRefit`, `gpMeanPercentile`, `gpMeanRangeFun`, `gpRadius`, `UseEffectiveRadius`, `gpCovPrior`, `gpFixedMean`, `FitLik`, `SearchAcqFcn`, `NoiseNudge`, `RemovePointsAfterTries`, `gpWarnings`, `NormAlphaLevel`, `WarpFunc`.
- Noise and hedge: `UncertainIncumbent`, `MeshNoiseMultiplier`, `HedgeGamma`, `HedgeBeta` (= 1).

Values that match and that PyBADS does not read, or reads only in a no-op branch (all on the sheet):
- KD-B1-5(a): `SkipPoll`, `SearchImproveFrac`, `gpCluster`.
- KD-B1-5(b): `PollMethod`, `PollAcqFcn`, `gpdefFcn`, `gpMethod`, `CholAttempts`.
- KD-B1-5(c): `gpSamples`, `gpSVGDiters`, `RotateGP`, `Nbasis`.
- No-op branches: `Plot`, `Restarts`, `SearchOptimize`, `AcqHedge`, `FitnessShaping`.
- `PeriodicVars` is refused when set (KD-B1-6).
- `OptimToolbox`, `Debug` and `TrueMinX` exist only in MATLAB (KD-B1-4).

Entries that differ:
- **`TolNoise`.** MATLAB: `sqrt(eps)*TolFun` = 1.49e-11. PyBADS: `np.spacing(1.0)*tol_fun` = 2.22e-19. Read by both sides at default. **F3.**
- **`SearchFactorMin`.** 0.5 on both sides. MATLAB reads it at default (`bads.m:1366`); PyBADS reads it nowhere. **F4.** It is not on KD-B1-5.
- **`CacheSize`.** MATLAB 1e4, PyBADS 500. Harmless: PyBADS's logger grows its arrays (`function_logger.py:303-335`), while MATLAB's is a circular buffer that wraps only after 1e4 evaluations, which is D > 20 at the default budget.
- **`FunValues`.** Empty on both sides. A non-empty value crashes PyBADS (**F9**).

**`setupoptions.m` against PyBADS's processing.**
- **Empty values.** MATLAB gives any field missing or empty in the user's struct its default (5–9). PyBADS stores the user dict verbatim, `None` included (`options.py:48-51`), so `None` means "default" only where the default is `None`, or for `specify_target_noise` and `stobads`, which are special-cased. **F8.**
- **String values.** MATLAB strips `%` comments (12–19) and `eval`s the listed fields, the user's included, with `evalbool` as fallback (22–50). PyBADS evaluates only its `.ini` expressions, with `D` bound, and uses user values verbatim; that part is KD-B1-3. A consequence the entry does not spell out: a MATLAB-style string for a boolean option is truthy, so `uncertainty_handling='off'` gives level 1 (`check_options.out`), where MATLAB gives 0.
- **Unknown names.** Ignored by MATLAB, a `ValueError` in PyBADS (KD-B1-3).
- **Missing checks.** PyBADS has no counterpart to the `MaxFunEvals` positive-integer error (72–74) or the `ImprovementQuantile > 0.5` warning (76–78). **F10.** `OptimToolbox` detection is KD-B1-4.
- **Noise combinations.** These match (`bads.py:826-908` against `setupoptions.m:80-101` and `evalinitmesh.m:10-17`):
  - `specify_target_noise` empty becomes false.
  - `specify_target_noise` with `uncertainty_handling` empty sets `uncertainty_handling` true.
  - `specify_target_noise` with `uncertainty_handling` false is an error.
  - Without `specify_target_noise`, `noise_size[0] <= 0` is an error.
  - With `specify_target_noise`, `noise_size > 0` gives a warning.
  - The level is 2 with `specify_target_noise`, 1 when `uncertainty_handling` is truthy, and 0 otherwise, with the noise test only when `uncertainty_handling` is empty.
  - PyBADS-only: a `noise_size` of any size other than 1 or 2 is an error, and a warning when `noise_size > exp(5)`.
- **Rewrites under uncertainty.** The rewrite at the start of `optimize()` (`bads.py:1154-1200`) matches `bads.m:431-445` line for line, plus KD-B5-8.
- **Layering.** Basic file, then the user dict, then the advanced file skipping user keys. `self.get("tol_fun")` in `tol_noise` and `hedge_beta` sees a user `tol_fun`, as MATLAB's `options.TolFun` does.
- **Descriptions (minor).** `_read_config_file` splits a comment line at its first `=` or `:`, so `Options.descriptions` truncates the descriptions of `noise_size`, `periodic_vars` and `stobads_frame_size_scaling_power`. The docs include the files verbatim, so users still see the full text.

### Q2. The bounds and the starting point

**MATLAB's order.**
1. `bads.m:331-342`: an empty `x0` requires PLB and PUB; then `x0 = NaN(size(PLB))`.
2. `boundscheck.m`: scalars expanded to vectors (7–10); empty PLB or PUB becomes LB or UB with a warning, nothing moved (12–16); same sizes; row vectors; finite plausible bounds; real values; fixed variables when x0, LB, UB, PLB and PUB are all equal.
3. `setupvars.m`: order LB ≤ PLB < PUB ≤ UB (6–9); `nonbcon([PLB;PUB])` must return 2×1, inside try/catch (11–25); a caution for infinite bounds (27–38); `transvars`; search bounds; a random x0 in the transformed plausible box, gridized (79–85); otherwise x0 is kept and gridized (87); gridization overshoot adjusted (91–92); x0 inside the original bounds and u0 inside the transformed ones (95–99).
4. `evalinitmesh.m:22-26`: `nonbcon(x0) > 0` is an error.

**PyBADS's order.**
1. `__init__`: plb/pub default to lb/ub whenever omitted (181–184); `x0=None` then requires plb/pub (186–196); D is `x0.shape[1]`.
2. `_bounds_check_`:
   - Estimation of plb/pub from a multi-row x0 (339–378). It has no MATLAB counterpart and is reachable only with lb or ub `None`.
   - Shape (1, D), with no scalar expansion (380–400).
   - Finite plausible bounds; real values.
   - Fixed variables. The test ignores x0; both sides refuse, KD-B1-7.
   - plb == pub is an error.
   - x0 outside [lb, ub] is an error.
   - **Effective bounds**: x0 clamped into them, plausible bounds moved inside them, plausible box expanded to x0 (450–531). **F1.**
   - Half-bounded variables refused, with the test taken across variables (533–544). **F2.**
   - A weak `non_box_cons` shape check (546–556).
   - An infinite-bounds warning.
3. Back in `__init__`: random x0 in the transformed plausible box, not gridized until `_init_optim_state_` (260–274); `non_box_cons(x0) > 0` is an error (277–284).
4. `_init_optim_state_`: gridization and adjustment as in MATLAB (670–682); PyBADS-only, a `ValueError` if the gridized point violates `non_box_cons` (685–693); u0 inside the transformed bounds (699–705).

**What PyBADS refuses that MATLAB accepts:**
- scalar bounds with D > 1 (F5);
- half-bounded variables, and any problem that mixes bounded and unbounded variables (F2);
- a plausible box narrower than 1e-3 of the range next to a hard bound (F1, `check_narrow.out`);
- a gridized x0 that violates `non_box_cons`;
- `x0` containing `+inf` with a finite `ub`, which MATLAB replaces by a random point;
- non-empty `fun_values` (F9).

**What PyBADS accepts that MATLAB refuses (F12):**
- `x0=None` with lb/ub but no plb/pub;
- a multi-row `x0`, which then fails in `optimize()` with a broadcasting error;
- a `non_box_cons` returning (2, k).

**What PyBADS moves and MATLAB does not.** x0, plb and pub within 1e-3·(ub−lb) of a hard bound, measured in the original space (F1). The docstring says scalar bounds are replicated (they are not, F5) and that a coordinate may be unbounded while others are bounded (refused, F2). The docstring's `non_box_cons` example `lambda x: np.sum(x.^2,1)>1` is MATLAB syntax and not valid Python.

### Q3. The transform and the grid

- **The transform matches.** `VariableTransformer` computes exactly what a Python transcription of `transvars.m` (create, `'dir'`, `'inv'`) computes: bit-identical direct and inverse maps, transformed bounds and log flags, on 2000 random points plus the bound points. The cases were: linear with finite bounds, linear with infinite bounds, mixed log/linear, all log, a log variable with `ub = inf` (MATLAB can reach this; PyBADS refuses it at the bounds check), forced log flag 0, and a ratio of exactly 10 (`check_transform.out`).
- **Details that match.** The log rule (all four bounds > 0 and pub/plb ≥ 10), mu and gamma, the `log(abs(x)+(x==0))` form, the realmax cap, the clamps (direct to the transformed bounds, inverse to the original ones), the invertibility test, and `maskindex`. `grid_units` equals the direct transform, row by row.
- **Round trip.** The relative error is at most 4e-15.
- **What differs is the input.** `_bounds_check_` hands the transformer moved plausible bounds (F1), which can flip the log decision and shift the log-space box, and it never hands it a half-bounded variable (F2).
- **Gridization and search bounds.** Grid parameters, gridization of u0, the ± `search_mesh_size` adjustment, `lb_search`/`ub_search` and `tol_mesh` = 2^ceil(log2(1e-6)) are identical to `setupvars.m:40-45`, `70-76`, `87-92`, `105`.
- **Unreachable defects.** None of these is reachable from `BADS`:
  - a scalar `apply_log_t` raises `AttributeError` (`variables_transformer.py:102-105` reads `self.apply_log_t` before setting it);
  - `grid_units` and `maskindex` need 2-D input;
  - `_init_optim_state_` passes `scale` into `grid_units`'s `x0` parameter (672), which is harmless.

### Q4. The result and the seed

`OptimizeResult` against its docstring, the docs and `bads_output.m` with `bads.m`'s outputs, field by field:
- **`x`**: `inverse_transf(u)`, as MATLAB's `origunits(u)`. It is 1-D, while `x0` is (1, D).
- **`fval` and `fsd`**: the incumbent's values, with `fsd = 0` when deterministic; the final estimate when noisy (B2). They match MATLAB, and `f(x) == fval` in a deterministic run.
- **`iterations`**: `iter + 1` (KD-B1-8).
- **`func_count`**: counts the noise test and the final samples, as MATLAB's `funccount` does (55 against 54 logger rows in `check_result.out`).
- **`mesh_size`, `message`, `target_type`, `problem_type`, `algorithm`, `version`**: the same logic as MATLAB; some strings differ.
- **`yval_vec` and `ysd_vec`**: as documented (KD-B1-8).
- **`status`**: in the key whitelist and listed by KD-B1-8, but never set: `result["status"]` raises `KeyError`. **F7.**
- **`success`**: always `True`, also at `max_fun_evals` (MATLAB exitflag 0). **F7.**
- **`overhead`**: leaves out the time of the final samples and of merged level-2 repeats, which MATLAB counts. **F13.**
- **`total_time`**: starts in `optimize()`, so the setup is excluded; MATLAB's `t0` includes it. Minor.
- **`fun` and `non_box_cons`**: deep-copied, which makes `optimize()` raise after the run for some targets. **F6.**
- **`x0`**: the checked x0, which F1 may have moved, or the random x0 before gridization. It is not the first point evaluated. PyBADS-only field.
- **`maxconstraint` and `rngstate`**: absent (KD-B1-8, KD-B1-1).

`_init_rng_` with `get_rng`, against the option's description and the docstring (`check_rng.out`):
- Accepted as documented: `int`, `np.int64`, a float or `np.float32` that is a whole number (converted to int), `SeedSequence`, `Generator` (used as the same object), `BitGenerator`, a list, and `None`.
- `None` derives the generator from exactly four `uint32` draws of the global state (MT position 624 → 4). `np.random.seed` before construction fixes the run, and a seeded construction leaves the global state untouched.
- The reported seed is an int for integer seeds and `None` otherwise.
- `3.5`, `"3"`, `nan` and `inf` raise `TypeError`, as documented.
- Two small departures from the docstring: a negative integer raises `ValueError` (the docstring says `TypeError`), and `True` is accepted as seed 1. Neither affects results.

## 3. Findings

### F1. Effective bounds move x0 and the plausible bounds inward by 1e-3 of the linear range, which MATLAB BADS never does; for log-scaled variables this cuts decades off the plausible box
- Location: `pybads/bads/bads.py:450-531`, together with `181-184` (plb/pub default to lb/ub whenever omitted). MATLAB: `private/boundscheck.m:12-16` (PLB = LB, nothing moved), `private/setupvars.m:6-9`, `79-99` (x0 kept, only gridized).
- Category: defaults
- Proposed classification: port discrepancy. The block, with its `bads:TooCloseBounds`, `InitialPointsTooClosePB` and `InitialPointsOutsidePB` identifiers, looks taken from MATLAB VBMC's `boundscheck.m`, not BADS's; I could not check this, since VBMC is not in this environment. Computing the margin in the original space is a defect for log-transformed variables whatever the intent.
- Confidence: high
- Reached at default options: yes, at every uncertainty level, for inputs whose plb/pub or x0 lie within 1e-3·(ub−lb) of a finite hard bound. That includes every bounded problem that omits plb/pub, or sets them equal to lb/ub as the docstring recommends "where in doubt"; the package's example 2 omits them.
- History: MATLAB `boundscheck.m` is unchanged since `8ea0ecd` (2018-02-13). The `setupvars.m` lines cited are from 2017; its 2022 commits `d4fead5` and `019f0b4` touch other lines. No commit of MATLAB BADS has effective bounds (`git log -S_eff`). The Python block has been there since the first port, `c7c88ab` (2022-06-02); the plb←lb default in `__init__` since `9037851` (2022-09-22). The two never agreed.
- What the code does:
  - It computes LB_eff = lb + 1e-3·(ub−lb) and UB_eff = ub − 1e-3·(ub−lb), in the original space.
  - It clamps x0 into [LB_eff, UB_eff] (473–480).
  - It moves plb up to LB_eff and pub down to UB_eff (494–504).
  - If x0 then equals LB_eff or UB_eff, it expands the plausible box to x0 (506–519). That test compares x0 with LB_eff/UB_eff, not with plb/pub as its message says.
  - A narrow plausible box inside the margin then fails the ordering check (521–531).

  MATLAB takes PLB = LB and x0 as given (a point on the bound included) and only gridizes u0.
- Consequence if real (`check_effbounds.out`, `check_effbounds2.out`):
  - **[-5, 5] with plb = lb.** The plausible box becomes ±4.99 and the hard bounds sit at u = ±1.002 instead of ±1, off the search grid.
  - **[1, 10] with plb = lb.** MATLAB log-transforms (pub/plb = 10); PyBADS does not (9.991/1.009 = 9.90).
  - **[0.001, 1000] with plb = lb.** The plausible box becomes [1.001, 999], which is [0.00014, 0.9999] in MATLAB's u. The lower three decades lie outside the plausible box (hard bound at u = −3.0), and the initial design samples only x ≥ 1 (first evaluations 1.001, 1.96, 113).
  - **Same bounds with user plb = 0.01, pub = 100 and x0 = 0.01 (the optimum).** x0 is moved to 1.000999 and plb to 1.000999, with three warnings; the user's start and plausible box are overridden.
  - **x0 = lb = −5 with plb = −2.** x0 and plb both become −4.99, and that variable's u-scale changes by a factor 1.75.
  - **lb = 0, ub = 1, plb = 1e-4, pub = 5e-4.** Refused; MATLAB accepts it.

  The initial design, the random x0, the mesh scale and the GP's priors in plausible units all follow the moved box. The 1-D examples still found their optimum; the effect on harder problems is not measured.
- Suggested reproduction: `check_effbounds2.py`, whose output is above. A MATLAB check would be `bads(@(x)(log10(x)+2)^2, 0.01, 0.001, 1000, 0.01, 100)`, which should evaluate x = 0.01 first.
- Test adequacy: no test reaches it. `test_random_x0_is_uniform_in_the_transformed_box` picks bounds outside the margin, and the bounds of `test_bads_optimization.py` are far from its plausible boxes.

### F2. Half-bounded variables are refused, and because the test is taken across variables, any problem that mixes bounded and unbounded variables is refused too
- Location: `pybads/bads/bads.py:533-544`. MATLAB: no counterpart. `bads.m:15-21` documents LB(i) = −Inf and UB(i) = Inf separately, and `setupvars.m:27-38` only prints a caution.
- Category: control flow
- Proposed classification: port discrepancy (the half-bound refusal) and a suspected defect (the mixed case, which contradicts PyBADS's own docstring, `bads.py:64-66`: "while other coordinates may be bounded").
- Confidence: high
- Reached at default options: no. Inputs reach it: any infinite bound together with a finite one.
- History: MATLAB never had such a check. The Python has been identical since `c7c88ab` (2022-06-02). They never agreed.
- What the code does: it refuses when `np.any(isfinite(lb)) and np.any(~isfinite(ub)) or np.any(~isfinite(lb)) and np.any(isfinite(ub))`. That is the Python `and`/`or` slip on MATLAB logicals that AGENTS.md warns about: a per-variable test would combine elementwise before `any`. MATLAB accepts both cases. `transvars.m`, and PyBADS's own `VariableTransformer`, handle infinite bounds, including a log variable with `ub = inf`: bit-identical in `check_transform.out`.
- Consequence if real: `ValueError` "bads:HalfBounds" for lb = [0, −inf], ub = [1, inf] (`check_bounds.out`), and for lb = 0, ub = inf. Users must invent finite bounds.
- Suggested reproduction: `check_bounds.py`, cases "mixed" and "half-bounded".
- Test adequacy: none.

### F3. `tol_noise` is eps·tol_fun (2.2e-19) instead of MATLAB's sqrt(eps)·TolFun (1.5e-11), so almost-deterministic targets are declared noisy
- Location: `pybads/bads/option_configs/advanced_bads_options.ini:13`, read at `bads.py:1037`. MATLAB: `bads.m:195`, `private/evalinitmesh.m:43`.
- Category: defaults
- Proposed classification: port discrepancy (a transcription slip of `sqrt(eps)`)
- Confidence: high
- Reached at default options: yes (`uncertainty_handling = None`), at level 0 → 1, for targets whose second evaluation of x0 differs from the first by between 2.2e-19 and 1.5e-11. With |f| of order 1, any bit-level difference is enough.
- History: MATLAB since `fd3f7a2` (2017-03-29). Python `np.spacing(1.0) * self.get("tolfun")` since `c7c88ab` (2022-06-02), renamed in `cdc2e0f`. They never agreed.
- What the code does: a difference above 2.2e-19 switches the run to level 1. MATLAB's threshold is 1.49e-11.
- Consequence if real: a run on sphere+1 in D = 2 with 100 evaluations (`check_tolnoise.out`):

  | Jitter SD | PyBADS | MATLAB's threshold would give |
  |---|---|---|
  | 1e-13 or 1e-12 | level 1: all 100 evaluations, fval error about 1e-5 | level 0 |
  | 0 | level 0: 55 evaluations, error 0 | level 0 |

  Targets with thread-order summation, iterative solvers or other tiny nondeterminism get the noisy settings: 20+ initial points, n_train ≥ 200, doubled stall, 10 final samples.
- Suggested reproduction: `check_tolnoise.py`.
- Test adequacy: none. `test_univariate_input_and_opt` uses noise SD 0.1, far above both thresholds.

### F4. `search_factor_min` is never read: after failed searches the search factor is not floored at 0.5
- Location: `advanced_bads_options.ini:101`; `pybads/bads/bads.py:2759-2767` (`_update_search_stats_`, `failure` branch; this is B3's code, found through the defaults table). MATLAB: `bads.m:238`, `1366`: `searchfactor = max(SearchFactorMin, searchfactor*SearchScaleFailure)`.
- Category: defaults
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, at all levels, whenever a round has ≥ 3 consecutive failed searches (D ≥ 2).
- History: MATLAB since `6c93629` (2017-03-14). The Python option has been in the `.ini` since `c7c88ab` and never read; the update without the floor since `8e59038` (2022-06-03). They never agreed. It is not on KD-B1-5.
- What the code does: it multiplies by sqrt(0.5) at each failure, down to 0.707^(n_try−1) before the reset at the end of the round (0.177 at D = 6, 0.0014 at D = 20). This factor scales the ES sampling covariance (`es_search.py:128`, `searchES.m:105`); MATLAB keeps it ≥ 0.5.
- Consequence if real: in default runs of 200 evaluations (`check_searchfactor.out`), the share of searches that ran with a factor below 0.5 was:

  | Problem | Searches with factor < 0.5 | Smallest factor |
  |---|---|---|
  | Rosenbrock D = 2 | 8% | 0.354 |
  | Ellipsoid D = 6 | 49% | 0.177 |
  | Rosenbrock D = 6 | 15% | 0.177 |

  The late searches of a round sample a neighbourhood much tighter than MATLAB's.
- Suggested reproduction: `check_searchfactor.py`.
- Test adequacy: none.

### F5. Scalar bounds are not replicated across dimensions (docstring and MATLAB say they are)
- Location: `pybads/bads/bads.py:181-184`, `380-400`; docstring `63-64`. MATLAB: `private/boundscheck.m:7-10`.
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no. Inputs reach it: scalar `lower_bounds`, `upper_bounds`, `plausible_lower_bounds` or `plausible_upper_bounds` with D > 1.
- History: MATLAB since `42ae029` (2017-05-12). Python shape check (1, D) without expansion since `c7c88ab`. They never agreed.
- What the code does: `np.atleast_2d(scalar)` has shape (1, 1), which fails the (1, D) check with `ValueError`.
- Consequence if real: `BADS(f, np.zeros(3), -5, 5, -2, 2)` raises (`check_bounds.out`). D = 1 works.
- Suggested reproduction: `check_bounds.py`, case "scalar".
- Test adequacy: `test_1D_opt_scalar` covers only D = 1.

### F6. `OptimizeResult` deep-copies the target and the constraint, so `optimize()` raises after the run for targets that cannot be deep-copied
- Location: `pybads/bads/optimize_result.py:98-99`, `182-186`. MATLAB: `private/bads_output.m:4` (`output.function = func2str(fun)`).
- Category: state/caching
- Proposed classification: suspected defect (no MATLAB analogue)
- Confidence: high
- Reached at default options: no. It needs a bound-method or callable-object target; a plain function or lambda is copied by reference.
- History: in `optimize_result.py` since `ce3a0b3` (2022-11-22), at the latest.
- What the code does: `__setitem__` applies `copy.deepcopy` to every value. For a bound method, `deepcopy` copies `__self__`.
- Consequence if real: with `fun = model.nll`, where `model` holds a `threading.Lock`, all 30 evaluations run and then `optimize()` raises `TypeError: cannot pickle '_thread.lock' object`. The result is lost; only `bads.x` remains on the object (`check_deepcopy.out`). A target that holds a large dataset is copied whole. For a callable object, `result["fun"]` is a copy, not the user's object.
- Suggested reproduction: `check_deepcopy.py`.
- Test adequacy: none.

### F7. `status` is whitelisted, and listed by KD-B1-8 as returned, but never set; `success` is always `True`
- Location: `pybads/bads/optimize_result.py:62-84`, `159-162`. MATLAB: `bads.m:1` and `1062-1083` (`exitflag` 0/1/2).
- Category: state/caching
- Proposed classification: port discrepancy. This contradicts KD-B1-8's list of fields; the entry leaves the meaning of `success` open.
- Confidence: high
- Reached at default options: yes, in every run.
- History: MATLAB `exitflag` since 2017. Python since `ce3a0b3` (2022-11-22), with `success = True` and a TODO.
- What the code does: `result["status"]` raises `KeyError`, and `result.status` raises `AttributeError`. `success` is `True` also when the run stops at `max_fun_evals`, which MATLAB flags as exitflag 0 against 1 or 2. MATLAB's exit condition is available only through the message text.
- Consequence if real: callers written for scipy's convention (`res.status`, `res.success`) fail or get a constant.
- Suggested reproduction: `check_result.py`: "missing from whitelist: ['status']".
- Test adequacy: none.

### F8. A user value of `None` replaces the default instead of standing for it (MATLAB: empty → default)
- Location: `pybads/bads/options.py:48-51`. MATLAB: `private/setupoptions.m:5-9`.
- Category: defaults
- Proposed classification: port discrepancy. KD-B1-3 leaves this open; this finding records what PyBADS does.
- Confidence: high
- Reached at default options: no. It needs an option set to `None` by the user.
- History: MATLAB since `6c93629` (2017). Python since `c7c88ab`.
- What the code does (`check_options.out`):
  - `max_fun_evals`, `max_iter`, `fun_eval_start` or `search_n_try` = `None`: `TypeError` in `optimize()`.
  - `tol_mesh` or `tol_fun` = `None`: `TypeError` in `BADS()`.
  - `nonlinear_scaling=None`: silently disables the log transform, where MATLAB's default is on. The same applies to any boolean whose default is `True`; `fit_lik=None` is refused.
  - Only `specify_target_noise` and `stobads` map `None` to their default.
- Consequence if real: crashes, or silent changes of the transform.
- Suggested reproduction: `check_options.py`.
- Test adequacy: none.

### F9. The options for prior evaluations crash: `fun_values` (MATLAB's `FunValues`) at creation, `f_vals` at the first display line
- Location: `pybads/bads/bads.py:740-791` (`fun_values`: `not np.isreal(X)` on an array, then `range(len())`, then `self.function_logger` before it exists); `581-604`, `2840-2869` (`f_vals` sets `cache_active`, which selects a 7-field display format that the log call fills with 6 values). MATLAB: `private/setupvars.m:126-167`, `private/funlogger.m:52-83` (the evaluations are imported into the logger).
- Category: control flow
- Proposed classification: port discrepancy for `fun_values`; suspected defect for `f_vals` (PyBADS-only, KD-B1-4, which lists it as read by code).
- Confidence: high
- Reached at default options: no. It needs a non-empty `fun_values`, or `f_vals`.
- History: MATLAB since 2017 (`4a5a3d7` and earlier). Python `fun_values` validation since `c7c88ab`; the `range(len())` loop since `9037851` (2022-09-22). It never worked.
- What the code does: `fun_values={"X": [[1,1]], "Y": [[2]]}` raises `ValueError` ("truth value of an array … ambiguous"). `f_vals=[2.0]` raises `ValueError` ("Unknown format code 'f' for object of type 'str'") after 2 evaluations, and its values are never used as evaluations.
- Consequence if real: both documented options are unusable.
- Suggested reproduction: `check_bounds.py` (case "fun_values"), `check_fvals.py`.
- Test adequacy: none.

### F10. MATLAB's checks of `MaxFunEvals` and `ImprovementQuantile` are not ported
- Location: no counterpart in `pybads/bads/bads.py`, `__init__` or `_init_optim_state_`. MATLAB: `private/setupoptions.m:71-78`.
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no. It needs `max_fun_evals ≤ 0` or non-integer, or `improvement_quantile > 0.5`.
- History: MATLAB since 2017/2018 (`8ea0ecd`). Python never had these checks.
- What the code does: `max_fun_evals=0` or `−5` evaluates x0 and then fails with "cannot convert float NaN to integer". `30.5` is accepted and runs 31 evaluations. `improvement_quantile=0.9` gives no warning. MATLAB raises "needs to be a positive integer" and warns, respectively.
- Consequence if real: obscure errors instead of a clear refusal; no warning where MATLAB gives one.
- Suggested reproduction: `check_options.py`.
- Test adequacy: none.

### F11. `display` values `"notify"`, `"final"`, `"none"`, and any case variant act as `"iter"`; the `.ini` description lists `"notify"` and `"final"`
- Location: `pybads/bads/bads.py:224-232`; `basic_bads_options.ini:2`. MATLAB: `bads.m:312-328` (first three letters, lower case: notify → 1, off/none → 0, iter/all → 3, final → 2).
- Category: control flow
- Proposed classification: port discrepancy. KD-B2-3 settles the logger mechanism and leaves the display content open.
- Confidence: high
- Reached at default options: no. It needs `display` set to one of these.
- History: MATLAB since 2017. Python since `c7c88ab`.
- What the code does: only the exact strings `"off"`, `"iter"` and `"full"` are handled. `"final"`, `"notify"`, `"none"` and `"OFF"` leave the logger at INFO, which is the full iteration display (`check_options.out`).
- Consequence if real: the display is more verbose than asked. Not numerical.
- Suggested reproduction: `check_options.py`.
- Test adequacy: `test_bads_logger.py` checks `"full"` only.

### F12. Input validation that differs from MATLAB: multi-row `x0`, `x0=None` without plausible bounds, and the `non_box_cons` output check
- Location: `pybads/bads/bads.py:181-199` and `339-378` (multi-row x0 and the estimation of plb/pub from it); `546-556` (the `non_box_cons` check). MATLAB: `bads.m:331-342`; `boundscheck.m:18-27`; `setupvars.m:11-25`.
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no. Inputs reach it.
- History: MATLAB lines from 2017. The Python estimation branch since `c7c88ab`. The `non_box_cons` check was MATLAB-like in `c7c88ab` (`shape[0] != 2 or shape[1] != 1`) and became `shape[0] != 2 and ndim == 1` by `9037851` (2022-09-22). The plb←lb default dates from `9037851`.
- What the code does (`check_bounds.out`):
  - A 2-row x0 passes the checks (with lb `None`, plb/pub are estimated from it, a branch MATLAB BADS lacks); `u0` is flattened to 2·D, and `optimize()` fails with "operands could not be broadcast". MATLAB refuses: "should be row vectors".
  - `x0=None` with lb/ub only is accepted, taking the plausible box from the moved hard bounds (F1). MATLAB errors, as PyBADS's own "Raises" section says it should.
  - `x0=None` with a list `plb` gives `AttributeError`.
  - A scalar `non_box_cons` output gives `IndexError`; a (2, 2) output passes the check and fails at x0 with "truth value … ambiguous". MATLAB gives a clear error for anything but N×1.
- Consequence if real: confusing failures; one silent acceptance.
- Suggested reproduction: `check_bounds.py`.
- Test adequacy: none.

### F13. `overhead` leaves out the time of the final samples and of merged level-2 repeats, which MATLAB counts
- Location: `pybads/function_logger/function_logger.py:400-430` (neither the `record_duplicate_data=False` path nor the merge path adds to `total_fun_eval_time`); `pybads/bads/bads.py:1648-1658`. MATLAB: `private/funlogger.m:130` (`'single'` adds `t`), `private/bads_output.m:48`.
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: medium. The mechanism is certain, and the size is small in practice.
- Reached at default options: yes, at levels 1 and 2 (final samples); merged repeats at level 2 only. Both sides leave out the noise test at initialization.
- History: MATLAB line from `603da99` (2017-04-24). The Python accounting since the logger's port. The overhead lines were last touched in `95da7f1` (2026-09-26).
- What the code does: in a noisy run with 80 evaluations, `total_fun_eval_time` was 0.151 s against 0.173 s summed over all evaluations (`check_result.out`).
- Consequence if real: the reported `overhead` is larger than MATLAB's definition. Informational only.
- Suggested reproduction: `check_result.py`.
- Test adequacy: none.

## 4. Test adequacy notes

- `pybads/testing/variable_transformer/test_variable_transformer.py` tests only linear transforms with finite bounds. It has no log transform, no mixed log/linear case and no infinite bounds, which are the cases where `transvars.m`'s logic matters. `test_transform_inverse_largeN` builds `np.ones((10 ^ 6, D))`, where `^` is XOR, so it tests 12 rows, not a million.
- No test exercises `_bounds_check_`'s moves or refusals: effective bounds, plb = lb, x0 on a bound, scalar bounds with D > 1, half-bounded or mixed variables. `test_random_x0_is_uniform_in_the_transformed_box` checks the draw against the implementation's own transformed box, and its bounds are chosen outside the 1e-3 margin, so it would not see F1. None of `test_bads_optimization.py` uses plb = lb or a log-scaled variable.
- No test compares the defaults with MATLAB's `defopts`. The `tol_noise` and `search_factor_min` differences (F3, F4) pass every test, and the only noise-test run uses noise far above either threshold.
- `test_bads_seed.py` checks the seed kinds against the implementation's own reported values. That fits a Python-only option; it does not cover negative ints or `bool`.
- Nothing checks the fields of `OptimizeResult` against its whitelist (the missing `status`), or builds one from a bound-method target (F6).
- `test_noisy_runs.py` mirrors `setupoptions.m`'s noise combinations faithfully; no test covers the checks that were not ported (F10) or a user `None` (F8).
