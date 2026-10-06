<!-- Report of the M reviewer (the MATLAB changes since the port began, comparison track), wave 0 of the port review, reading PyBADS at ab4dded in ../pybads-review and MATLAB BADS at 74919c0; saved verbatim from its final message on 2026-09-26. Its check scripts are kept on the orchestrator's machine only (dev/scripts/runs/LOCAL.md). Nothing in it is verified. -->

# M comparison review: MATLAB changes since the port began

I read all eight MATLAB commits hunk by hunk and matched each one to PyBADS at `ab4dded`. PyBADS carries most of the changes. There are ten findings (F1 to F10). Four matter at default options or close to them:
- **F1 and F2 (noisy runs):** at the end of each iteration, PyBADS swaps its working GP for a GP stored in the history. The stored GPs then drift away from the hyperparameters recorded for their iteration.
- **F3 (noisy runs):** the final `fsd` uses `np.std` with ddof=0, where MATLAB's `std` divides by N−1.
- **F4 (random `x0`):** the random start is drawn uniformly in the original space, not in the transformed box.

The other six are small, unreachable or inert.

Paths below: Python is relative to `C:\Users\luigi\Documents\GitHub\pybads-review` (commit `ab4dded`). MATLAB is relative to `C:\Users\luigi\Documents\GitHub\bads` (commit `74919c0`) unless a commit is named. "KD-…" and "claim C…" refer to `known_differences.md`.

## 1. Coverage

**Read completely**
- **The eight diffs:** `git show` of each commit. For `d4fead5` this includes the rename diff `utils/gpTrainingSet.m` → `private/gpupdate.m` and the full `gpTrainingSet.m` at `d4fead5^` (lines 1-120 and 335-470; the unchanged middle comes from the rename diff).
- **MATLAB at `74919c0`:**
  - `private/gpupdate.m`, `private/setupoptions.m`, `private/bads_output.m`: whole files.
  - `utils/gpHyperOptimize.m` 1-235.
  - `private/evalinitmesh.m` 1-100; `private/setupvars.m` 1-100.
  - `private/funlogger.m` 24-150.
  - `gpdef/gpdefBads.m` 1-60 and 150-315.
  - `bads.m`: `defopts` 146-295, search stage 505-740, poll loop 800-935, end of iteration, final estimate and output 1040-1200, and the subfunctions 1257-1478 (`EvalImprovement`, `UpdateIncumbent`, `reevaluateIterList`, `FinalEstimate`).
- **PyBADS at `ab4dded`:**
  - `pybads/bads/gaussian_process_train.py`: all of it.
  - `pybads/bads/bads.py`: lines 150-2280 and 2581-2692.
  - `pybads/bads/optimize_result.py` 1-165; `pybads/function_logger/function_logger.py` 72-195, plus greps; `pybads/utils/iteration_history.py` 17-140.
  - `pybads/variable_transformer/variables_transformer.py` 150-175; `pybads/stats/get_hpd.py`.
  - The two `.ini` files, grepped for every option these commits touch.
- **gpyreg v1.3.3:** `GP.update` (docstring and data handling), the `GaussianNoise` docstring, `ConstantMean.get_bounds_info` and `_bounds_info_helper`.
- **PyBADS history:** `git log -L`/`-S` of every line cited in a finding.

**Skimmed or not reached**
- Skimmed: `bads.py` 1-150 (docstring). I did not read `_is_gp_refit_time_` or `_get_target_from_gp_`.
- Not reached: the rest of `gpdefBads.m` (60-150, likelihood definitions, slice B6), gpyreg `fit` internals, and the README, example and test-script changes (out of scope).

**Checks run** (scripts and logs in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pybads\10994841-8958-4975-b288-1fd2bcd6013b\scratchpad\review\M_comparison`). All ran on 2-D noisy spheres, noise SD 1, at most 200 evaluations, one BLAS thread, with the review worktree and gpyreg v1.3.3 printed:
- `check_noisy_history.py`: final `fsd` formula; stored GPs against recorded hyperparameters.
- `check_alias.py`: instrumented `_re_evaluate_history_`, 3 seeds.
- `check_switch.py` and `check_switch_idx.py`: GP swapped in at each switch, 4 seeds.
- `check_random_x0.py`: random `x0`, 200 constructions, no optimization.
- An inline NumPy check of `argsort` tie order.

## 2. Per commit

Status is one of: carried, before, neither, n/a (not applicable).

### `8515191` (`~isempty` check on `gpstruct.s`)
| MATLAB | Python | Status | Note |
|---|---|---|---|
| `utils/gpHyperOptimize.m:141-143` | `gaussian_process_train.py:649-651` | carried | `if s2 is not None and s2.size > 0`. Present since `c7c88ab` (2022-06-02); works on local copies since `b6a1cf8` (2025-10-16). |

### `bfe8e22` (remove `s` with the removed points)
| MATLAB | Python | Status | Note |
|---|---|---|---|
| `gpHyperOptimize.m:139-143` | `gaussian_process_train.py:627-651` | carried | X, Y and s2 are dropped together on local copies. The GP keeps its full data, as MATLAB's by-value `gpstruct` does. Retry index is equivalent (`i_try > remove_points_after_tries-1`, 0-based). |
| `bads.m` version strings; `install.m` | — | n/a | |

### `75ec49f` (uncertainty handling, specified noise, printing)
| MATLAB | Python | Status | Note |
|---|---|---|---|
| `bads.m:147,178-182` (`bads_version`, `'version'` call) | `optimize_result.py:148-156` | n/a | Version comes from package metadata. |
| `bads.m` display `'Train'` line; help text | — | n/a | Cosmetic. |
| `bads.m:1097` (poll step, noisy, `iter > 1`) | `bads.py:1407-1412` | carried | 0-based `poll_iteration > 0`. |
| `bads.m:1101-1104` (yval, fval, fsd, fhyp from `iterList(iter)`) | `bads.py:1413-1419` | carried / **neither** | `best_gp_hyp` stands for `fhyp`. Line 1419 also replaces the working GP (**F1**, **F2**). |
| `bads.m:1107-1109` (skip first iterate) | `bads.py:1421-1431` | carried | Since `8e59038` (2022-06-03); `c7c88ab` did not skip. |
| `bads.m:1112-1118` (switch to a better iterate) | `bads.py:1434-1445` | carried / **neither** | yval, fval, fsd, u and hyp carried. `self.best_u` (1439) is a dead assignment, harmless because MATLAB does not move `ubest` either. Lines 1443-1445 swap in the stored GP of that iterate (**F1**, **F2**). |
| `bads.m:1144-1145` (final choice skips first) | `bads.py:1482-1483` | carried | Since `8e59038`. |
| `bads.m:1185-1188` → `private/bads_output.m` | `bads.py:1542-1568`, `optimize_result.py` | carried | Structure only. |
| `bads_output.m:5-13` targettype | `optimize_result.py:100-106` | carried | |
| `bads_output.m:14-20` problemtype | `optimize_result.py:108-117` | carried | Names differ (KD-B1-8). |
| `bads_output.m:21` iterations | `optimize_result.py:119` | **neither** | One less than MATLAB (**F7**). |
| `bads_output.m:22-24,26-27,34,43-44,47-48` | `optimize_result.py:120-123,142-143,156,161`; `bads.py:1542-1552` | carried | |
| `bads_output.m:25,29-33` rngstate, maxconstraint | — | n/a | On the sheet (KD-B1-1, KD-B1-8). |
| `bads_output.m:37` yval | `optimize_result.py:124-130` | **neither** | None for deterministic runs and for `noise_final_samples = 0` (**F7**). |
| `private/evalinitmesh.m:52-62` (message after the test, "(specified noise)") | `bads.py:976-988` | carried | The test condition above it (`evalinitmesh.m:38` vs `bads.py:964`) is claim C1 and was not changed by this commit. |
| `private/setupoptions.m:86-88` | `bads.py:833-834` | carried | |
| `setupoptions.m:90-92` | `bads.py:836-840` | carried | Since `068e57f` (2026-09-25). Before that PyBADS set `False`, with no effect: the level comes from `specify_target_noise` (`bads.py:867-868`). |
| `setupoptions.m:94-97` | `bads.py:842-852` | carried | |

### `c4d2b9a`
| MATLAB | Python | Status | Note |
|---|---|---|---|
| URLs; "±" encoding in the final message | `bads.py:1557-1563` | n/a | Cosmetic. |

### `d4fead5` (v1.1.0): `bads.m` and other files
| MATLAB | Python | Status | Note |
|---|---|---|---|
| `HessianUpdate`, `HessianMethod`, `HessianAlternate` removed from `defopts` | `advanced_bads_options.ini:252-255` | before | Options kept, no effect (KD-B1-4, KD-B1-5(e)). |
| NONBCON retro-compatibility removed | — | n/a | Keyword arguments. |
| rebuild calls renamed (`bads.m:527-533,830-836`) | `bads.py:1603-1619,2074-2090` (`local_gp_fitting`) | carried | |
| `bads.m:624`, `899` (`funlogger` returns SD) | `bads.py:1709,2155` | carried | |
| `bads.m:633-641` search `'add'` with `[ysearch,ysearch_sd]` | `bads.py:1717-1731` → `gaussian_process_train.py:1250-1253` | carried | SD used only with `specify_target_noise`: since `f9e9326` (2022-11-02), squared since `1c8c71d`. The condition on the line above is not from this commit (see incidental). |
| `bads.m:643-658` (`gpstructnew`) | `bads.py:1739-1763` | carried | |
| `bads.m:908-916` poll `'add'` with `[ypoll,ypoll_sd]` | `bads.py:2165-2175` | carried | |
| `covmatadapt` and curvature-decay blocks removed | `bads.py:1788-1793` (no-op) | before | No effect (KD-B1-4). |
| `updatehess` call removed from `UpdateIncumbent` | `bads.py:2595` (comment only) | carried | |
| `reevaluateIterList` call renamed (`bads.m:1392-1398`) | `bads.py:2664-2673` | carried | The call is carried; the GP it is given differs (**F2**). |
| `gpdef/gpdefBads.m` pp0-pp3 and lin kernels removed | — | n/a | Kernel hard-wired (KD-B6-1). |
| `gpdefBads.m:164-165` initial `hyp.mean` = median of the lowest `ceil(0.8N)` values (was 0) | `gaussian_process_train.py:868,885,911,116-117`; `stats/get_hpd.py:35` | carried up to rounding | PyBADS uses `round(0.8N)` (**F8**). Present since `c7c88ab`, from PyVBMC's code. |
| `funlogger.m:1,27` (`fsd` output) | `function_logger.py:193` | carried | |
| `setupvars.m` Hessian setup removed | `bads.py:744` (comment only) | carried | |
| `updatehess.m`, `multibayes.m` deleted | none | carried | |
| `acqPortfolio.m`, `hetsphere.m`, `runtest.m`, whitespace, `setupoptions.m` evalfields | — | n/a | |

### `d4fead5`: `gpTrainingSet.m` (before) → `gpupdate.m` (after), piece by piece
| Piece | Before (`d4fead5^`) | After (`74919c0`) | Python | Follows |
|---|---|---|---|---|
| Data U, Y, S | `U,Y` = `(1:Xmax)`; S if `optimState.S` exists | same (30-33) | `gaussian_process_train.py:1100-1105` | Both (same). S is also taken at level 1 when `uncertainty_handling=True` (NaN, **F10**). |
| `'add'`: y*, SD | SD read inside `try`; `bads.m` passed only y, so the read failed and the point was never added under `SpecifyTargetNoise` | 43-49 | `:1250-1253` | After |
| `'add'`: update | rank-1 update, append inside `try`; on failure not added | rank-1 only without `SpecifyTargetNoise`; always append; full recomputation otherwise or on failure (51-83) | `:1257-1269`: full `gp.update`; on failure GP restored and `needs_rebuild` set | After with specified noise. Otherwise neither: KD-B5-1. |
| `'add'`: non-finite y* penalty | 53-62 | 69-78 | none (`:1273` TODO) | Neither; unreachable (**F5**). |
| `'add'`: `refit_flag = false` | implicit (early return) | 41 | never refits | Both |
| `'nearest'` | 70-96 | 85-111 (identical) | `:1107-1141` | Both. Ported in `c7c88ab`; the cap `min(ntrain, Xmax)` fixed in `1d075ab` (2026-09-25). Tie order differs (**F9**). |
| `'grid'` Hessian branch; `'covgrid'`, `'neighborhood'` | present | removed | none | n/a (`gp_method` fixed to `'nearest'`, KD-B1-5(b)) |
| Fitness shaping (`&& ~add_flag`) | 337-341 | 252-256 | `:275-278` no-op | n/a (KD-B5-5) |
| Non-finite substitution | 343-350, with the `error_index` typo | 258-265, fixed in `74919c0` | `:280-287` | Neither (**F5**) |
| Rotation | 358-380 | removed | `:289` TODO | After (off at default in both) |
| Prior update (`gpdefBads`) | 382-385 | 273-276 | `:293-364` | Both (content is B6's) |
| Refit, lenscale, pollscale, radius | 388-447 | 279-338 (same) | `:366-516` | Both |
| Posterior, failure | 449-460 | 340-354 (Debug print added) | `:522-552` | KD-B5-2 |

### `a21f2ee` (v1.1.1)
| MATLAB | Python | Status | Note |
|---|---|---|---|
| `bads.m:198` `Ninit = 10 + nvars` | `advanced_bads_options.ini:23` | carried, then superseded | PyBADS took it in `8ff10f5` (2022-11-04) and reverted in `4e6a001` (2022-11-15). |
| `evalinitmesh.m:92` comment | `bads.py:995-1000` | n/a | The rule itself is carried. |

### `019f0b4` (v1.1.2)
| MATLAB | Python | Status | Note |
|---|---|---|---|
| `bads.m:198` back to `nvars` | `advanced_bads_options.ini:23` `fun_eval_start = D` | carried | |
| `bads.m:1137` `ysd_vec = []` | `bads.py:1467` | carried | |
| `bads.m:1157,1188`; `bads_output.m:38-40` | `bads.py:1500-1518`; `optimize_result.py:132-138` | carried | |
| `FinalEstimate` `bads.m:1449-1460` | `bads.py:1500-1507` | carried | `ysd_vec` is NaN at level 1 and not reported. |
| `bads.m:1464-1466` one-sample rule | `bads.py:1511-1515` | carried | Since `068e57f`. |
| `bads.m:1469-1471` mean and `std/sqrt(n)` | `bads.py:1528-1533` | **neither** | ddof=0 (**F3**). |
| `bads.m:1472-1476` precision-weighted estimate | `bads.py:1520-1527` | carried | Since `068e57f`. |
| `bads.m:1158-1159` (context lines) | `bads.py:1534-1537` | **neither** | Recorded in the wrong slot (**F6**). |
| `setupvars.m:79-85` random u0 in the transformed box | `bads.py:254-264` | **neither** | Differs for log-transformed variables (**F4**). |

### `74919c0` (v1.1.3)
| MATLAB | Python | Status | Note |
|---|---|---|---|
| `gpupdate.m:262` `err_index` | `gaussian_process_train.py:280-287` | **neither** | **F5** |
| version strings, `CLAUDE.md`, script | — | n/a | |

## 3. Findings

### F1. The end-of-iteration re-evaluation replaces the working GP with a stored GP; MATLAB changes only `fhyp`
- Location: `pybads/bads/bads.py:1419`, `:1443-1445`; MATLAB: `bads.m:1101-1104`, `:1112-1118` (and `:1059`, `:769`)
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high on the mechanism, medium on the size of the effect
- Reached at default options: yes, for any noisy target (level ≥ 1, detected at default), from the second iteration on.
- History: MATLAB `75ec49f` (2022-05-09) rewrote this block. The swap to the chosen iterate's GP dates from `c7c88ab` (2022-06-02, comment "overwrite best gp"); the swap at the current iteration from `9037851` (2022-09-22).
- What the code does:
  - After `_re_evaluate_history_`, PyBADS sets `gp = iteration_history["gp"][poll_iteration]`.
  - When an earlier iterate beats the current one by more than `tol_fun`, it sets `gp = iteration_history["gp"][idx_impr]` instead. That GP carries iteration `idx_impr`'s hyperparameters and its `poll_scale`, `len_scale` and `effective_radius`.
- What it should do: MATLAB assigns only `fhyp = iterList.hyp{index}`, which is used for `UpdateTarget` (`bads.m:539`) and `fpollhyp` (`bads.m:777`). It leaves `gpstruct` alone, so the next rebuild at `searchcount == 0` uses the latest hyperparameters. PyBADS's `best_gp_hyp` (lines 1416, 1440) already plays `fhyp`'s part; the GP swap is extra.
- Consequence if real: after each switch, the next iteration rebuilds and ranks LCB candidates with an older GP. Its training-set radius, length scale and poll scale are old too, unless a refit happens.
  - Frequency: 8, 9, 6 and 4 switches in runs of 12, 14, 14 and 11 iterations (seeds 3-6).
  - Size: the swapped-in GP's constant mean was 8-12.5 higher. Log length scales, log output scale and log RQ shape differed by about 0.1-0.2; log `poll_scale` by at most 0.005.
- Suggested reproduction: `check_switch.py` / `check_switch_idx.py` (output as above). To settle the effect: drop the `gp = …` assignments at 1419 and 1443-1445 and run the noisy population gate.
- Test adequacy: none would catch it. The noisy tests in `test_bads_optimization.py` check final error against seed-swept tolerances.

### F2. The swapped-in GP is shared with its history slot, so that slot takes on later hyperparameters and later re-evaluations use them
- Location: `pybads/bads/bads.py:1419`, `:1443`, `:2659-2673`; `pybads/utils/iteration_history.py:46`, `:95-101`; MATLAB: `bads.m:1388` (`gpstruct.hyp = optimState.iterList.hyp{index}`)
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, for noisy targets.
- History: follows from F1 (`c7c88ab`, `9037851`). `IterationHistory.get` returns stored objects by reference.
- What the code does:
  - After the swap, the working GP is the same object as slot k. Rebuilds and refits in the next iteration change that slot in place.
  - The drift stops when the next `record` grows the array: `_expand_array` reassigns it through `__setitem__`, which deep-copies it. The drift also stops if a noisy search replaces `gp` with its copy (`bads.py:1860`).
  - Result: slot k keeps hyperparameters from a later point of the run. Later calls to `_re_evaluate_history_` re-evaluate iterate k with them.
- What it should do: MATLAB re-evaluates every iterate with the hyperparameters recorded at the end of that iteration.
- Consequence if real: some iterates' re-evaluated `fval`/`fsd` differ from MATLAB's. These values feed the switch test (F1) and the final choice (`q_beta`, `bads.py:1479-1492`).
  - `check_noisy_history.py`: 4 of 14 slots have hyperparameters different from `gp_hyp_full` beside them.
  - `check_alias.py`: at run end, slots [2, 4], [1, 4, 7] and [3, 4, 10] (seeds 3-5).
- Suggested reproduction: `check_alias.py`. Its per-call listing shows the mismatches building up.
- Test adequacy: none.

### F3. The final estimate's SD uses `np.std` with ddof=0; MATLAB's `std` divides by N−1
- Location: `pybads/bads/bads.py:1531-1533`; MATLAB: `bads.m:1471`
- Category: formula
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, for a noisy target without `specify_target_noise` that runs more than one iteration (`noise_final_samples = 10`).
- History: MATLAB's `std(yval_vec)/sqrt(numel(yval_vec))` predates the port. `019f0b4` moved it into the `~SpecifyTargetNoise` branch. PyBADS `np.std` dates from `8e59038` (2022-06-03), replacing `np.sd`; the lines were last touched in `068e57f`.
- What the code does: `np.std(yval_vec) / np.sqrt(n)`. It should be `np.std(yval_vec, ddof=1) / np.sqrt(n)`.
- Consequence if real: the reported `fsd` is too small by a factor √((N−1)/N): 0.949 at N = 10, and 0.707 with one final sample (two values). `x` and `fval` are unaffected.
- Suggested reproduction: `check_noisy_history.py`. Reported `fsd` 0.22567 = ddof 0; the MATLAB formula gives 0.23788.
- Test adequacy: `test_noisy_runs.py` tests only the level-2 formula (`test_final_estimate_weights_samples_by_precision`, `test_final_estimate_from_one_sample`). No test covers level 1.

### F4. The random `x0` is uniform in the original plausible box; MATLAB draws uniformly in the transformed box
- Location: `pybads/bads/bads.py:254-264`; MATLAB: `private/setupvars.m:79-85`
- Category: random draws
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: no. It needs `x0` missing or non-finite, plus a log-transformed variable: all four bounds positive, `pub/plb ≥ 10`, and `nonlinear_scaling` on (the default). For linearly transformed variables the two draws have the same distribution.
- History: MATLAB `019f0b4` (2022-11-14; before it, the midpoint of the transformed box). PyBADS added its draw in `d466948` (2022-11-10) and moved it to `self.rng` in `1d075ab`. KD-B1-1 settles where the draws come from, not their distribution.
- What the code does: `rng.uniform(plb, pub)` in x. MATLAB's `rand(1,nvars).*(PUB-PLB)+PLB` is in u, which is log-uniform in x for a log variable.
- Consequence if real: a different starting-point distribution. With `plb = 1`, `pub = 100`, 8.5% of starts fall below 10 against 50% in MATLAB (median 57.9 against 10).
- Suggested reproduction: `check_random_x0.py` (output as above, 200 seeds).
- Test adequacy: `test_bads_seed.py` checks that the random `x0` reproduces, not its distribution.

### F5. Non-finite targets: wrong substitution index, a dead `s2` branch, and no penalty in `add`; unreachable
- Location: `pybads/bads/gaussian_process_train.py:280-287`, `:1271-1273`; MATLAB: `private/gpupdate.m:258-265`, `:69-78`
- Category: indexing/shape
- Proposed classification: suspected defect in both (MATLAB had the `error_index` typo from before the port until `74919c0`)
- Confidence: high on the code, and high that it is unreachable
- Reached at default options: no. `FunctionLogger` raises on non-finite values (`function_logger.py:160-170`), as MATLAB's `funlogger.m:103-105` does.
- History: MATLAB `74919c0` (2025-12-05). PyBADS since `c7c88ab`.
- What the code does:
  - `gp.y[y_idx_penalty]` uses an index into the finite subset as if it indexed the whole array. MATLAB maps it through `idx_values`.
  - The `s2` branch is guarded by `"S" in optim_state`, which is never true.
  - `add_and_update_gp` has no counterpart to the penalty at `gpupdate.m:69-78`.
- Consequence if real: none today. It would matter if non-finite values were ever admitted.
- Suggested reproduction: a direct `local_gp_fitting` call with `y = [inf, 1, 5]`. Not run, because it is unreachable.
- Test adequacy: none.

### F6. The final estimate is written to the last iteration's history slot, not the chosen iterate's
- Location: `pybads/bads/bads.py:1534-1537`; MATLAB: `bads.m:1158-1159`
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- Reached at default options: yes, for noisy runs with more than one iteration and final samples.
- History: context lines of `019f0b4`. PyBADS since `c7c88ab`.
- What the code does: records at `poll_iteration`. It should record at `min_q_beta_idx`.
- Consequence if real: only `bads.iteration_history["fval"/"fsd"]` after the run is affected. The result's `fval`/`fsd` are not.
- Suggested reproduction: after a noisy run, compare `iteration_history["fsd"][-1]` with `res["fsd"]` when the chosen iterate is not the last.
- Test adequacy: none.

### F7. `OptimizeResult` reports `iterations` one lower than MATLAB, and `yval_vec` None where MATLAB returns `yval`
- Location: `pybads/bads/optimize_result.py:119`, `:124-130`; MATLAB: `private/bads_output.m:21`, `:37`
- Category: cross-module
- Proposed classification: possibly intentional (AGENTS.md says the count starts at 0). But the docstring (`optimize_result.py:42-43`) says "Number of iterations performed", and KD-B1-8 leaves "the count in iterations" not settled.
- Confidence: high
- Reached at default options: yes, every run.
- History: MATLAB `75ec49f`. PyBADS since `c7c88ab`.
- What the code does:
  - `iterations` is the 0-based index of the last iteration. In the measured run the history holds 14 iterations and the result reports 13.
  - `yval_vec` is None for deterministic runs and for `noise_final_samples = 0`. MATLAB returns the incumbent's observation in both cases.
- Consequence if real: interface only.
- Test adequacy: none checks `iterations` against the history length.

### F8. Initial GP mean: median of the lowest `round(0.8N)` values instead of `ceil(0.8N)`
- Location: `pybads/bads/gaussian_process_train.py:868`, `:885`; `pybads/stats/get_hpd.py:35`; MATLAB: `gpdef/gpdefBads.m:164-165`
- Category: formula
- Proposed classification: port discrepancy
- Confidence: high on the code; the effect is small
- Reached at default options: yes, when N mod 5 is 3 or 4. At initialization N = 1 + the Sobol count, so this covers D = 1, 4-7 and 16-31 for deterministic targets, and every noisy run (N = 33).
- History: MATLAB `d4fead5` (before it, `hyp.mean = 0`). PyBADS since `c7c88ab`, from PyVBMC's code.
- What the code does: the median of a set one value smaller than MATLAB's. The same set also gives the mean prior's centre and SD (`:970-971`); that prior line is B6's.
- Consequence if real: a slightly different starting value for one hyperparameter of the initial fit.
- Test adequacy: none.

### F9. The `'nearest'` training set breaks distance ties differently: `np.argsort` is not stable, MATLAB's `sort` is
- Location: `pybads/bads/gaussian_process_train.py:1118`; MATLAB: `private/gpupdate.m:94`
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high on the mechanism, low on a material effect
- Reached at default options: yes, once there are more evaluations than `ntrain` and equally distant mesh points straddle the cutoff.
- History: `c7c88ab`. The piece is identical in both MATLAB versions.
- Consequence if real: a tied point can be swapped for its twin in the training set.
- Suggested reproduction: 120 distances on a 0.25 grid with ties. The default `argsort` gives a different first-50 set than `kind="stable"` (NumPy 2.5.3).
- Test adequacy: none.

### F10. With `uncertainty_handling=True` and no specified noise, `gp.s2` holds NaN; MATLAB holds no `s`
- Location: `pybads/function_logger/function_logger.py:59-60`; `pybads/bads/bads.py:279-288`; `pybads/bads/gaussian_process_train.py:1104-1105`, `:1138-1140`, `:272-273`; MATLAB: `private/funlogger.m:48`, `gpupdate.m:111`
- Category: state/caching
- Proposed classification: possibly intentional (inert)
- Confidence: medium
- Reached at default options: no. At default the noise test raises the level after construction, so `noise_flag` stays False and `s2` is None.
- History: since `8ff10f5`/`1c8c71d`. `test_gp_update_failures.py:70` notes it.
- What the code does: stores NaN variances, and `add` appends zeros. At level 1, `gp_noisefun = [1, 2, 0]` sets `user_provided_add=False`, and gpyreg ignores `scale_user_provided` without it, so no effect was found.
- Test adequacy: level-1 runs pass with it.

## 4. Test adequacy notes
- Nothing tests the end-of-iteration re-evaluation's effect on the working GP or the stored GPs (F1, F2). Two cheap tests would: stored GP hyperparameters against `gp_hyp_full` after a noisy run, and working-GP hyperparameters after a switch.
- `test_noisy_runs.py` covers the level-2 final estimate and option handling, but not the level-1 `fsd` formula (F3).
- No test covers `_robust_gp_fit_`'s point removal, with or without `s2`: the path of `8515191` and `bfe8e22`.
- `test_bads_seed.py` checks that the random `x0` reproduces, not its distribution (F4).
- No test reads `iteration_history` after the final estimate (F6), or compares `iterations` with the number of iterations (F7).

**Observed outside these commits' hunks** (not reviewed in depth; for the other slices):
- `advanced_bads_options.ini:13`: `tol_noise = eps·tol_fun`, where MATLAB (`bads.m:195`) uses `sqrt(eps)·TolFun`.
- `gaussian_process_train.py:677-685`: each retry raises the lower bound of the log noise by the running nudge (+1, +3, +6, …), which passes the upper bound 5 after about 5 retries. MATLAB (`gpHyperOptimize.m:164`) adds `nudge(2) = 0` at default. Also `:645` uses `np.percentile` with its default method, where MATLAB uses `prctile1` (hazen).
- `bads.py:2141`: the product runs over D+1 terms, where MATLAB (`bads.m:869`) uses `nvars`.
- `bads.py:793`: `range(len())` raises `TypeError`, so the `fun_values` option cannot work.
- `bads.py:1718-1722`: the search-add condition is always true (the `&` slip AGENTS.md names). MATLAB (`bads.m:633`) skips the last search of a round.
- `bads.py:2589`: `_update_incumbent_` also moves `self.u`. In MATLAB, `u` (the centre of later searches in the round, `bads.m:529,544`) changes only at `bads.m:769`. Unverified.
- `setupoptions.m:80-82`: MATLAB's error on `NoiseSize ≤ 0` has no PyBADS counterpart.
- `bads.py:2659-2673`: the re-evaluation uses each stored GP's `len_scale` and `effective_radius`. MATLAB uses the current `gpstruct` with the old hyperparameters swapped in (`bads.m:1388`).
