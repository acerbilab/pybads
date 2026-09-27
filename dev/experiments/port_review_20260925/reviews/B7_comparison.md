<!-- Report of the B7 comparison reviewer (function logger, initial design, utilities, MATLAB-comparison track), wave 4 of the port review, reading PyBADS at 0d866e8 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session, with the complete history; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave4/B7_comparison/. Nothing in it is verified. -->

# B7 comparison review: function logger, initial design, utilities

## 1. Coverage

**Read in full (Python, at `0d866e8`):**
- `pybads/function_logger/function_logger.py` and `pybads/init_functions/init_sobol.py`
- `pybads/utils/period_check.py` and its three call sites
- `pybads/function_logger/constraints_check.py`, read as a reader of the log
- In `bads.py`:
  - the construction of the logger (`BADS.__init__`, 357-370)
  - `_init_optim_state_` (600-775)
  - `_init_mesh_` (1035-1177)
  - `_init_optimization_` (1179-1290)
  - the final samples (1640-1720)
  - the search call site (1820-1935) and the poll call site (2155-2400)
  - the search-step condition (1400-1404)
- The readers of the log in `gaussian_process_train.py` (60-130, 1095-1270)
- `force_to_grid`, `VariableTransformer.inverse_transf` and `utils/timer`
- The slice's options in both `.ini` files
- `test_function_logger.py`, the CHANGELOG entries on the logger and the design, and `docsrc/source/api/classes/function_logger.rst`. `advanced_docs.rst` holds only a toctree.

**Read in full (MATLAB, at `74919c0`):**
- `private/funlogger.m`, `private/evalinitmesh.m`, `init/initSobol.m` and `init/private/i4_sobol_generate.m`
- `utils/periodCheck.m`, `utils/uCheck.m`, `utils/force2grid.m` and `utils/origunits.m`
- `private/setupvars.m` (1-175)
- `bads.m` (395-470, 516-517, 800-812, 1125-1200, 1440-1475)

**Skimmed:**
- `i4_sobol.m`: its dimension check and seed handling only, not the direction-number tables
- `transvars.m`, `searchES.m:120-130` and `gpupdate.m:115-125`

**Not reached:**
- `i4_bit_hi1.m` and `i4_bit_lo0.m`, which the substitution settled by KD-B7-1 replaces
- `initLHS.m`, `initRand.m`, `lhs.m` and `tau_sobol.m` (KD-B7-2)
- The rendered options page

**Scripts run** (in the scratchpad `wave4/B7_comparison`, one BLAS thread, all seeded):

| Script | What it checks |
|---|---|
| `check_seed.py` | The PyBADS Sobol seed, design sizes, platform notes |
| `check_matlab_seed.py` | A transcription of `initSobol.m:9-15`, under both readings of `prod` |
| `check_record_paths.py` | Counts the paths `_record` takes in nine 200-evaluation runs, levels 0, 1 and 2 |
| `check_design_size.py` | Design sizes by `fun_eval_start`, and affine ranks |
| `check_design_map.py` | The design's map onto the plausible box, and the shape of `x` |
| `check_call_checks.py` | The value and SD checks |
| `check_periodic.py` | `periodic_vars` handling and the stub |
| `check_ring.py` | A transcription of the MATLAB ring's indexing |

Every script printed `pybads.__file__` as `/home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__` as `/home/user/gpyreg-v1.3.3/gpyreg/__init__.py`, with NumPy 2.4.6 and SciPy 1.17.1 on x86_64 Linux.

## 2. Answers to the first questions

### 1. The evaluation

**The point the target receives.** `__call__` maps `u` back through `VariableTransformer.inverse_transf` (`function_logger.py:111-116`). The target receives a 1-D float array of shape `(D,)`; `check_design_map.py` saw only `(2,)`. MATLAB passes a 1×D row vector from `origunits` (`funlogger.m:89`). The Python inverse also clips to the original bounds; MATLAB's `transvars 'inv'` does not. Points reaching the target already lie within the bounds, so the clip only absorbs rounding.

**Outputs at levels 0 and 1.**
- The target returns one value. A size-1 array or list is unwrapped (lines 138-159).
- MATLAB's `fval = fun(x)` keeps a function's first output. A MATLAB target returning `[f, sd]` without `SpecifyTargetNoise` therefore runs, with `sd` dropped.
- In PyBADS such a tuple is refused with `InvalidFuncValue`. This is a difference between the languages, and the safer behaviour.

**Outputs at level 2.** The target must return exactly a tuple of two (lines 123-133). A list or an array of two is refused with the `specify_target_noise` message; MATLAB calls `[fval,fsd] = fun(x)`.

**Value check.** Lines 162-170 require a finite real scalar, which matches `funlogger.m:103` for numeric outputs.

**SD check.** Lines 173-179 require finite, real and > 0, as `funlogger.m:108` does, but without its `isscalar`. `None`, a list, or an array with several elements raises `TypeError`, or NumPy's own `ValueError`, instead of the documented `InvalidNoiseValue` (F7).

**Errors raised by the target.** They are re-raised with a note appended to `args` (lines 144-155). MATLAB warns `bads:funError` and rethrows. Both stop the run. The Python value checks sit outside the `try`, so their errors carry no note; MATLAB's warning precedes those too. The difference is cosmetic.

**The count.** `func_count += 1` after the checks, on every path of `_record` (line 192), and not in `add`. MATLAB counts `'iter'` and `'single'` after its checks, plus an explicit +1 for the noise test (`evalinitmesh.m:42`). The totals agree: in every run of `check_record_paths.py`, `func_count` equals the new rows plus the unrecorded evaluations.

**The docs.** The docstrings say the same, apart from F7. `function_logger.rst` is only an autoclass directive.

### 2. The record

**Rows and growth.** Each recorded evaluation gets a new row (lines 438-459). The arrays are preallocated at `cache_size` (500) rows and grow by 50% (`_expand_arrays`). MATLAB keeps a fixed ring of `CacheSize` rows (1e4) that overwrites its oldest rows (F4).

**Merge at level 2 (KD-B7-3).** The code matches the entry:
- it merges into the row that matches in every coordinate;
- it takes precision-weighted `Y` and `S`, adds 1 to `n_evals` and averages the time;
- it returns the merged value with the new observation's SD.

The entry leaves some details out: `Y_orig` keeps the first observation, and `Y_max` is not updated. Since W3-1 no BADS run reaches the merge: 0 merges in 3 level-2 runs (F3).

**Paths that do not record.**
- **The noise test.** It runs at levels 0 and 1 whenever `uncertainty_handling` is left empty, as it is by default. It sets `n_evals[0]` to 2 and `fun_eval_time[0]` to the mean of the two times. The caller restores the total, which leaves the test out of it as MATLAB does (`bads.py:1061-1065`). MATLAB calls the target directly, without the checks (F6).
- **The final samples.** They take the chosen iterate's row: `n_evals` +1 and an averaged time per sample, with the total including them, as for MATLAB's `'single'`.
- A point with no row would return `idx=None`. That is not reached, because the chosen iterate is always in the log.

**Other fields.**
- `X_flag` marks the filled rows.
- `Xn` is the 0-based last row and always equals `X_max_idx`.
- `Y_max` is updated on new rows only, and nothing reads it.
- The per-row `fun_eval_time` is read into `t_train`, which is unused.
- `total_fun_eval_time` feeds `overhead` and counts the same evaluations as MATLAB.

**`finalize` against `'done'`.**
- `finalize` trims `X_orig`, `Y_orig`, `X`, `Y`, `S`, `X_flag` and `fun_eval_time` to `Xn+1`. It does not trim `n_evals`, and it keeps `X`, which is MATLAB's `U`, the array that `'done'` removes.
- BADS never calls `finalize`. MATLAB calls `'done'` only when `optimState` is returned (`bads.m:1191-1193`).
- After a run, the object's logger therefore holds padded arrays.
- `reset_fun_eval_time` is never called. If it were called after the arrays grew, it would give back only `cache_size` rows.

**`add`, and MATLAB's import of earlier evaluations.**
- `add` records a point given in the transformed space. It checks the value and the SD, and puts in an SD of 1 when the logger holds `S`. It does not count in `func_count`, as MATLAB's import does not, and records a time only when one is given.
- MATLAB's `FunValues.X` is in the original space (`setupvars.m:127-167`, `funlogger.m:53-83`). MATLAB keeps repeated imported points as separate rows, where PyBADS would merge them at level 2.
- MATLAB picks the start after the design among `[y0; design]` only (`evalinitmesh.m:121`). `_init_mesh_` takes the argmin over every row, so imported rows would compete for the start.
- None of this is reachable: `add` has no caller in `pybads/`, and a non-empty `fun_values` is refused (`bads.py:757-762`), as KD-B1-4 says.

**What the readers of the log get:**
- **GP training set** (`get_grid_search_neighbors`) and **`contraints_check`:** `X[:X_max_idx+1]`, `Y`, and `S` only at level 2, as W0-9 intends. Correct at every level.
- **Start choice after the design:** `Y[:Xn+1]`. Correct.
- **Level-2 `fsd`:** read from `S` at the argmin row, as `bads.m:448-450`.
- **Loop counts and the LCB:** `func_count`, the same count as MATLAB's `funccount`.
- **`n_eff`** (`gaussian_process_train.py:1133`): the sum of `n_evals`, so it includes the noise test (F6).
- **Search-step condition** (`bads.py:1402`): `len(Y[X_flag]) > D`, where MATLAB tests `size(gpstruct.y,1) > nvars` (`bads.m:516-517`). The two are equivalent at default: `n_train_min = 50` exceeds D, so the GP keeps every row up to 50.

### 3. The initial design

**The seed.** Both sides seed from `u0` when it is finite, but the derivations differ (F1, the open candidate of KD-B7-1). The branch that draws from the generator is unreachable on both sides: `u0` is always finite, because `BADS.__init__` (and `setupvars.m:79-85`) draws `x0` first.

**The number of points.** Both sides take `min(fun_eval_start, max_fun_evals - 1)`, raised in a noisy run to `min(max(20, ·), max_fun_evals)`, as `evalinitmesh.m:93-101` does. PyBADS then:
- rounds up to `2**ceil(log2 n)` (KD-B7-1);
- raises the exponent once more when `2**m == D` (F2);
- cuts the result to the evaluations left (KD-B2-6).

**The map.** `plb + r*(pub - plb)` in the transformed space, as `initSobol.m:17`. The check confirmed that the transformed plausible box is `[-1, 1]` per variable, with a log-transformed variable log-uniform in the original space.

**The return value.** `(u_init, m)`, where `m` is the exponent, not the number of samples the docstring names (F7). The caller uses only `u_init`.

**`_init_mesh_` against `evalinitmesh.m`.** The order is the same on both sides:
1. the design (`InitFcn`);
2. `period_check`;
3. `force_to_grid` at `search_mesh_size`;
4. `contraints_check` with projection onto `[lb_search, ub_search]` and `tol_mesh/2` bins against the whole log;
5. evaluation in the sorted order the check returns, as MATLAB's `setdiff` does;
6. the argmin over `[y0; design]`, taking the first index on ties as MATLAB's `min` does.

It matches, apart from the settled cut.

**Internal-track notes.**
- The design is a balanced scrambled set of the plausible box until the cut, the grid and the constraint check thin it.
- It depends on D only, not on the start or `random_seed`, for every start inside the plausible box (F1).
- It becomes platform-dependent for a start coordinate at or below -1 (F1).
- Starts on a 2-D `u0` are not reached, because `self.u` is 1-D (`bads.py:712`).

### 4. The utilities

**`period_check` against `periodCheck.m`.** The stub returns its input (confirmed). `periodCheck.m:4` returns `x` unchanged when there are no periodic variables. So the two agree on every input the code can receive.

**KD-B1-6.** It is what the code does: any non-`None` `periodic_vars` raises in `_init_optim_state_` (`bads.py:643-647`). That includes `[]`, which MATLAB treats as "no periodic variables" (`setupvars.m:107-108`); this is a small interface difference.

**The call sites** are inconsistent, a latent problem (F5).

## 3. Findings

### F1. The Sobol seed is not MATLAB's derivation: it depends only on D for every start inside the plausible box, and is 1 for every D ≥ 8
- **Location:** `pybads/init_functions/init_sobol.py:52-62`, `69`; MATLAB: `init/initSobol.m:9-16`
- **Category:** random draws
- **Proposed classification:** port discrepancy. It confirms the open candidate of KD-B7-1 with specifics.
- **Confidence:** high for the Python behaviour; unsure for the MATLAB seed values, which only MATLAB can settle.
- **Reached at default options:** yes, in every run at every level.
- **History:** the MATLAB lines have not changed since `12f7ff8` (2017-03-18). The Python lines were written in `c7c88ab` (2022-06-02). `1d075ab` moved only the branch for a non-finite `u0` onto the generator, and `e62a9e1` changed only the docstring. The two never agreed.

**What MATLAB does.** `num2str(u0(1:min(10,end)))` prints the values, for example `'0.25         -0.5'`. It takes the product of the character codes, mod 997, plus 1, and uses the result as a start index into the unscrambled sequence.

**What PyBADS does.**
1. It takes `u0[0:min(11, D)]`, 11 coordinates where MATLAB's `1:10` is `[0:10]`.
2. It casts them with `.astype(np.uint64)`, which truncates each value. Every coordinate in (-1, 1) becomes 0.
3. It prints `'0 0 … 0'` and multiplies the character codes in int64, which wraps around.

**The consequence of the wrap.** The product is `48^k·32^(k-1) = 2^(9k-5)·3^k` for `k = min(11, D)`. For k ≥ 8 it wraps to exactly 0, so the seed is 1.

**Measured seeds** (`check_seed.py`), for any start with all first-11 coordinates in (-1, 1), including the random `x0`:

| D | 1 | 2 | 3 | 5 | 6 | ≥ 8 |
|---|---|---|---|---|---|---|
| Seed | 49 | 948 | 967 | 748 | 843 | 1 |

**Starts on or outside the plausible box.**
- A coordinate ≥ 1 changes the seed, in the same way on every platform.
- A coordinate ≤ -1 is a negative float cast to uint64, whose result the hardware decides:
  - x86-64 gives `2^64 - n`; at D = 2 the seed becomes 1.
  - AArch64 saturates the conversion to 0, which gives the interior seed. This is inferred from the conversion's semantics; I did not run it on ARM.
- So for such starts the design differs between platforms.

**MATLAB's own outcome** (transcription, `check_matlab_seed.py`) depends on the class in which `prod` accumulates a uint64 array:
- If it accumulates in double, the seed follows the digits of `u0`. At D = 2, `[0.25, -0.5]` gives 966 and `[0.2998, 0.6992]` gives 656; two random 10-D starts gave 274 and 952.
- If it saturates in uint64, the seed is `mod(2^64-1, 997)+1 = 961` for any multi-D, non-integer `u0`.
- Either way, MATLAB's design depends only on `u0`, never on the random stream.

**Consequence if real.** Every PyBADS run from a start inside the plausible box uses one fixed design per D, whatever the start and `random_seed`. A population of seeded runs, or a user's restarts from several `x0`, therefore share their initial points apart from `x0`. The effect on results was not measured.

**Suggested reproduction.** `check_seed.py`: `init_sobol` gives the same design for starts `[0.1, 0.2, -0.3]` and `[-0.7, 0.9, 0.0]` with different generators, and a different one when a coordinate is -1. Whether MATLAB's `prod` saturates needs MATLAB.

**Test adequacy.** No test covers `init_sobol`. `test_bads_seed.py` checks that a seed reproduces a run, not that designs vary.

### F2. The design is doubled when its power of two equals D; MATLAB has no counterpart (W0-18)
- **Location:** `pybads/init_functions/init_sobol.py:73-76`; MATLAB: `private/evalinitmesh.m:101`, `104`, `init/initSobol.m:16`
- **Category:** defaults
- **Proposed classification:** unsure. It may be intentional, but no record gives a reason.
- **Confidence:** high, from the code.
- **Reached at default options:** yes, at level 0 for D ∈ {1, 2, 4, 8, 16}, where the design has 2, 4, 8, 16 and 32 points against D in MATLAB. In a noisy run, where `fun_eval_start` becomes 20 and rounds to 32, only at D = 32. A user's `fun_eval_start` also reaches it.
- **History:** the Python lines date from `c7c88ab` (2022-06-02) and are unchanged. MATLAB's `i4_sobol_generate(nvars, Ninit, seed)` has not changed since 2017. The two never agreed.

**What the code does.** The condition compares `2**m` with D, not with `fun_eval_start`. So the size jumps (`check_design_size.py`):
- at D = 4, a `fun_eval_start` of 2 gives 2 points, and 3 gives 8;
- at D = 8, 4 gives 4 points, and 5 gives 16.

With the doubling, the design has the smallest power of two strictly above D.

**No geometric reason was found.** The first D points of the scrambled set have affine rank D−1, as any D points in general position do; the doubled set has rank D (D = 2, 4, 8, 16, three seeds each). With `x0`, the undoubled design already has D+1 points, which is MATLAB's count of D design points plus the start.

**Consequence if real.** D extra evaluations at initialization for those D. At D = 2 that is 5 initial evaluations instead of 3 (MATLAB also 3). At D = 16 it is 33, against 17 in MATLAB or with the rounding alone. That is a negligible share of `500*D`; the effect on results was not measured.

**Suggested reproduction.** `check_design_size.py`.

**Test adequacy.** No test checks the design's size.

### F3. The level-2 merge of KD-B7-3 is unreachable in a BADS run since W3-1
- **Location:** `pybads/function_logger/function_logger.py:406-436`; MATLAB: `private/funlogger.m:117-129`
- **Category:** state/caching
- **Proposed classification:** possibly intentional. The code matches the entry, but the entry's clauses about runs describe a regime that W3-1 removed.
- **Confidence:** high
- **Reached at default options:** no. Only direct users of `FunctionLogger` reach it.
- **History:** the code dates from `199d787` (2022-11-20); `ab4dded` (2026-09-25) fixed which row it merges into. W3-1 (`149d528`, squash-merged in `0d866e8`) makes `contraints_check` remove points already evaluated. MATLAB has no merge.

**Why no run reaches it.**
- Every recorded evaluation after the start passes `contraints_check` first. That check removes any candidate in the `tol_mesh/2` bin of a logged point (`constraints_check.py:37-49`), and an exact repeat always falls in the same bin.
- The poll's set is filtered when it is built, and its points are distinct.
- The noise test and the final samples take the unrecorded path.

**Evidence.** `check_record_paths.py`, 3 seeds at each level, 200 evaluations each:

| Level | New rows | Merged | Unrecorded |
|---|---|---|---|
| 0 | 94-114 | 0 | 1 |
| 1 | 189 | 0 | 11 |
| 2 | 190 | 0 | 10 |

The CHANGELOG's "Points evaluated again" entry says the same: "neither repeats any now".

**Consequence if real.** No effect on results. But the entry's "add_and_update_gp then adds that value beside the point's earlier row", and its population test of returning the observation, which the CHANGELOG ties to runs that did merge, no longer describe any run.

**Suggested reproduction.** `check_record_paths.py`.

**Test adequacy.** `test_record_duplicate_with_user_noise_*` call `_record` directly. No test asserts that a run never merges.

### F4. The log grows without bound; MATLAB's is a ring of `CacheSize` (1e4) that overwrites its oldest rows and never fills its last one
- **Location:** `pybads/function_logger/function_logger.py:303-340`, `439-441`; `advanced_bads_options.ini:20-21`. MATLAB: `private/funlogger.m:36`, `120-121`; `bads.m:202`, `412`.
- **Category:** state/caching
- **Proposed classification:** possibly intentional (port discrepancy). The MATLAB side has an off-by-one of its own.
- **Confidence:** high
- **Reached at default options:** no for D ≤ 20, since at most 500D−1 evaluations are stored. Yes in a run of D ≥ 21 that stores 1e4 evaluations. Also reached through a user's `cache_size`: in MATLAB a `CacheSize` below the budget truncates the log, while in PyBADS `cache_size` is only the initial allocation, although the `.ini` calls it the "Size of cache".
- **History:** MATLAB lines 120-121 date from `e185ad2` (2017-03-14), `CacheSize` from `aed408c` (2017). The growth dates from the first port, `c7c88ab`. The two never agreed.

**What MATLAB does.** `Xn = max(1, mod(Xn+1, nmax))` sends the evaluation after row nmax−1 to row 1. Row nmax stays NaN, and `Xmax` reaches nmax. The transcription (`check_ring.py`, nmax = 5) writes rows `[1,2,3,4,1,2,3,4,1]`. After the wrap, `uCheck` and `gpupdate` read only the most recent nmax−1 evaluations.

**Consequence if real.** The two sides are identical up to 9999 stored evaluations. After that, the GP's neighbours and the removal of evaluated points differ, and MATLAB can evaluate an old point again. PyBADS's growth is the saner choice.

**Suggested reproduction.** `check_ring.py`.

**Test adequacy.** `test_call_expand_cache` tests the growth, not MATLAB's semantics.

### F5. `period_check`'s call sites are inconsistent: the poll discards its return value, and the design passes the option rather than the mask
- **Location:** `pybads/bads/bads.py:2220-2225` (poll, result not assigned), `1130-1135` (design, passes `self.options["periodic_vars"]`, which is `None`), `1837-1842` (search, passes the boolean mask `optim_state["periodic_vars"]`); `pybads/utils/period_check.py:4-6`. MATLAB: `bads.m:807` (`upollnew = periodCheck(...)`), `private/evalinitmesh.m:107`, `utils/periodCheck.m:4`.
- **Category:** cross-module
- **Proposed classification:** suspected defect (latent)
- **Confidence:** high
- **Reached at default options:** no. Periodic variables are refused (KD-B1-6).
- **History:** the Python dates from `c7c88ab` (2022-06-02), reformatted in `157bd09`. MATLAB is unchanged since 2017.

**Consequence if real.** None today. A port of periodic variables that implemented only `period_check` would not wrap the poll's points, and would receive an index list, not a mask, at the design. The ES search's periodic check (`searchES.m:128`) is also missing (`es_search.py:138`, a TODO KD-B1-6 lists).

**Suggested reproduction.** Read the three call sites.

**Test adequacy.** None cover it.

### F6. The noise test goes through the logger's checks and leaves a trace in row 0; MATLAB calls the target directly
- **Location:** `pybads/bads/bads.py:1055-1065`; `pybads/function_logger/function_logger.py:390-404`; `pybads/bads/gaussian_process_train.py:1133`. MATLAB: `private/evalinitmesh.m:38-47`.
- **Category:** control flow
- **Proposed classification:** possibly intentional
- **Confidence:** high
- **Reached at default options:** yes, at levels 0 and 1, whenever `uncertainty_handling` is left empty.
- **History:** MATLAB has not changed since `fd3f7a2` and `9246d16` (2017). `c7c88ab` called `function_logger.fun(self.x0)` directly, as MATLAB does, though at the user's `x0` rather than at the start on the grid. `9037851` (2022-09-22) moved the test into the logger as a recorded evaluation, and `8ff10f5` (2022-11-04) made it unrecorded. `8aecb6a` restores the total time, and `0c56d86` (W0-17) set the condition.

**Two effects.**
- **(a) Validation.** A second value at `x0` that is NaN, infinite or not a scalar raises `ValueError` in PyBADS. MATLAB reads NaN as no noise (`abs(y - NaN) > TolNoise` is false), reads Inf as noise, and continues.
- **(b) Row 0's counts.** `n_evals[0]` becomes 2, so `n_eff` runs one evaluation ahead of `eff_starting_points`. The PyBADS-only `init_N` schedule is then evaluated at `x = 1/n_budget` instead of 0 from the first refit on. At D = 3 by default, that gives 123 prior draws instead of 128. `fun_eval_time[0]` also averages the two times, though nothing reads it.

**Consequence if real.** Negligible on results. Effect (a) changes behaviour only for a malformed target.

**Suggested reproduction.** In `check_record_paths.py` at level 0, `n_evals[0]` is 2.0 and `sum(n_evals)` = `func_count` = rows + 1.

**Test adequacy.** `test_target_time_counts_every_evaluation_but_the_noise_test` covers the total time only.

### F7. The SD check lacks MATLAB's `isscalar`, and some docstrings do not match the code
- **Location:** `pybads/function_logger/function_logger.py:173-179` (with `__call__`'s "Raises" section, 95-101), `211-218` (`add`); `pybads/init_functions/init_sobol.py:46-49`, `80`. MATLAB: `private/funlogger.m:108`.
- **Category:** control flow
- **Proposed classification:** port discrepancy (minor; messages and docs)
- **Confidence:** high
- **Reached at default options:** no. Only a malformed level-2 target reaches the SD check.
- **History:** the SD check dates from `c7c88ab` and `9037851`; MATLAB line 108 from `a3b6ebd` (2021-01-08). The second return value of `init_sobol` dates from `4e6a001` (2022-11-15).

**What differs** (`check_call_checks.py`):
- An SD of `None`, `[0.5]` or `[0.5, 0.6]` raises `TypeError` or NumPy's ambiguous-truth `ValueError`. A 2-element array raises through `.item()`, with a "FuncError" note. None of them gives the documented `InvalidNoiseValue`, and a size-1 list is refused as an SD but accepted as a value.
- `init_sobol` returns the exponent `m` where its docstring says "Number of samples".
- `add`'s docstring does not say that `x` is in the transformed space, which MATLAB's `FunValues.X` is not, or that a missing SD becomes 1.

**Consequence if real.** Error messages and documentation only.

**Suggested reproduction.** `check_call_checks.py`.

**Test adequacy.** `test_call_invalid_sd_value` covers only an infinite SD.

## 4. Test adequacy notes
- **`test_function_logger.py`** tests the implementation's own choices through direct `_record` calls: the level-2 merge, and averaging a repeat's time into its row. Neither has a MATLAB counterpart, and since W3-1 no run reaches the merge. No test checks MATLAB's behaviour, one row per evaluation, at the level of a run.
- **`test_finalize`** does not check that `n_evals` is trimmed, and nothing checks that a run ever calls `finalize`; BADS never does.
- **Nothing tests `init_sobol`:** not its size (the rounding and the doubling), not its map onto the plausible box, and not how it depends on `u0` or `random_seed`. The seed tests check that a fixed seed reproduces a run, which a design that never changes also passes.
- **`test_get_gp_training_options_small_budget`** asserts `eff_starting_points == max_fun_evals - 1`. That mirrors the cut and the counting of the noise test (KD-B2-6), not MATLAB, whose design exceeds such budgets.
