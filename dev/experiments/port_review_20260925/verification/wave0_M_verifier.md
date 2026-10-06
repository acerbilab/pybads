<!-- Report of the verifier of wave 0, slice M and the preparatory agent's claims C1-C8, reading PyBADS at 95da7f1 in ../pybads-review (and ab4dded for comparison) and MATLAB BADS at 74919c0; saved verbatim from its final message on 2026-09-26. Its check scripts are kept on the orchestrator's machine only (dev/scripts/runs/LOCAL.md). -->

# Wave 0 verification: M and the preparatory claims

**Setup.** PyBADS at `95da7f1` is `C:\Users\luigi\Documents\GitHub\pybads-review`. PyBADS at `ab4dded` is `...\pybads\dev\scripts\runs\worktrees\winref_ab4dded`, used only for before/after comparisons. MATLAB is `C:\Users\luigi\Documents\GitHub\bads` at `74919c0`. Every run used gpyreg v1.3.3 from `...\dev\scripts\runs\gpyreg\v1.3.3`, and every log prints `pybads.__file__` and `gpyreg.__file__`. All runs had one BLAS thread and at most 200 evaluations, and were run one at a time. Tracked files are unchanged: `git status` is clean in both worktrees.

My scripts and logs are all in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pybads\10994841-8958-4975-b288-1fd2bcd6013b\scratchpad\review\verify_M\`:
- `v_f1f2_measure.py`, with logs `f1f2_case0_1790398433.log` and `f1f2_cases123_1790398463.log`;
- `v_f1f2_ab.py`, with log `f1f2_ab_1790398522.log`;
- `v_small_checks.py`, with logs `small_a_*`, `small_b_*`, `small_c_*` and `small_ab4dded_*` (the last one ran at `ab4dded`).

Unless marked otherwise, line numbers are at `95da7f1`.

## 1. Summary

| Item | Classification | Reached at default | At `95da7f1` | Dating | Confidence |
|---|---|---|---|---|---|
| M-F1 | confirmed port discrepancy | yes: noisy targets, every iteration from the 2nd; effect at switches | yes | never agreed: PyBADS since `c7c88ab` (switch) and `9037851` (every iteration); MATLAB has changed only `fhyp` since 2017 | high |
| M-F2 | confirmed port discrepancy | yes (noisy targets) | yes | never agreed: `c7c88ab`/`9037851`; MATLAB uses `iterList.hyp{index}` since `6c93629` (2017) | high |
| M-F3 | confirmed port discrepancy | yes (level 1, ≥2 iterations) | fixed by #71 | never agreed (`8e59038` until #71) | high |
| M-F4 | confirmed port discrepancy | no (needs a missing x0 and a log-transformed variable) | yes | never agreed: midpoint in x vs midpoint in u, then random in x (`d466948`, 2022-11-10) vs random in u (`019f0b4`, 2022-11-14) | high |
| M-F5 | confirmed port discrepancy (unreachable) | no | yes | PyBADS since `c7c88ab`; MATLAB errored on its own typo until `74919c0` (2025-12-05) | high |
| M-F6 | confirmed port discrepancy | yes (noisy runs with final samples) | fixed by #71 | never agreed (`c7c88ab` until #71) | high |
| M-F7 | `iterations`: confirmed port discrepancy. `yval_vec`: intentional difference (docstring), missing from the sheet | yes | `iterations` fixed by #71; `yval_vec`=None remains | `iterations` never agreed until #71 | high |
| M-F8 | confirmed port discrepancy (small) | yes: N mod 5 ∈ {3,4}, i.e. D = 1, 4–7, 16–31, and every noisy run | yes | MATLAB changed after the port (`d4fead5`, 2022-10-31) | high |
| M-F9 | confirmed port discrepancy (the mechanism) | row order yes; set membership not seen | yes | never agreed (`c7c88ab`) | high on mechanism, low on effect |
| M-F10 | confirmed port discrepancy, inert | no | yes | `c7c88ab`/`8ff10f5` | high |
| C1 | record wrong (and the code differs from MATLAB) | no (the default `None` behaves as recorded) | yes | code since `c7c88ab`/`8e59038`; record `1d075ab`; MATLAB since `fd3f7a2` (2017) | high |
| C2 | record wrong | yes: D ∈ {1, 2, 4, 8, 16, 32} | yes | `c7c88ab`; record `1d075ab` | high |
| C3 | record wrong (code side confirmed; the plan text was not opened) | yes (every run with a finite x0) | yes | `c7c88ab` | high on code, medium on the record text |
| C4 | record wrong | n/a (comment) | yes | `c7c88ab` | high |
| C5 | record wrong | no | yes | `9037851` | high |
| C6 | record wrong (claim confirmed) | n/a | yes | `c7c88ab` | high |
| C7 | record wrong | n/a | yes | two pointers stale since MATLAB `d4fead5`; the third never matched | high |
| C8 | record right as written, but ambiguous | n/a | yes | `685da15` | medium |

## 2. Per item

### M-F1: the end-of-iteration swap of the working GP

**What MATLAB does** (`bads.m:1059`, `1091-1120`, `1378-1415`):
- At the end of each iteration it sets `fhyp = gpstruct.hyp` (1059). It stores `iterList.u/yval/fval/fsd(iter)` and `iterList.hyp{iter} = gpstruct.hyp` (1091-1097).
- `reevaluateIterList(optimState,gpstruct,options)` gets `gpstruct` by value. For `index = 1:iter` it sets `gpstruct.hyp = iterList.hyp{index}` (1388). It then calls `gpupdate(...,'nearest',ui,[],...,0)`, which rebuilds around `ui` with this local copy's `lenscale` and `effectiveradius` (`gpupdate.m:91`, `97`). A call with refit 0 never changes those two (`gpupdate.m:279-338` run only when refitting). It re-centres the priors (`gpdefBads.m:198-305`), recomputes `post`, predicts, and writes `iterList.fval/fsd(index)`. It returns only `optimState`.
- In the caller, `yval`, `fval`, `fsd` and `fhyp` come from `iterList(iter)`. `fhyp` keeps its value, because `hyp{iter}` is `gpstruct.hyp`. On a switch, `yval`, `fval`, `fsd`, `u` and `fhyp` come from `iterList(index)`.
- `ubest` and `gpstruct` never change. `gpstruct` stays the working GP, with `post = []` if the poll moved (1049).
- `fhyp` is read only by `UpdateTarget` in the search (539) and by `fpollhyp` (777, then 841).
- So the next iteration rebuilds the latest GP, with its latest hyperparameters and geometry, around `u` (523-533). The old hyperparameters serve only the target.
- History: MATLAB has never assigned `gpstruct` in this block. `git log -L` goes back to 2017, and `75ec49f` (2022-05-09, before the port) kept `fhyp` only.

**What PyBADS does** (`bads.py:1436-1507`, `2724-2752`):
- It records `gp_hyp_full` and a deep copy of `gp` (1458-1461).
- `_re_evaluate_history_` ignores its `gp` argument. It runs `local_gp_fitting(gps[i], u_i, ..., False)` in place on each stored GP, with that GP's own hyperparameters and `temporary_data`.
- It then sets `gp = self.iteration_history.get("gp")[poll_iteration]` (1481). On a switch it sets `gp = ...[idx_impr]` (1505-1507). `best_gp_hyp` (1478, 1502) already plays the role of `fhyp`. `self.best_u` (1501) is a dead assignment, harmless because `u_best` is untouched, as `ubest` is in MATLAB.
- The swap at the current iteration does nothing numerically: that slot holds the working GP's hyperparameters, and the next iteration always rebuilds at `search_count == 0`. Its only effect is the sharing in F2.
- A switch hands the next iteration an old GP: its hyperparameters (possibly drifted, see F2), `len_scale`, `poll_scale` and `effective_radius`.

**Measurement** (`v_f1f2_measure.py`). Noisy 2-D spheres (seeds 0-2) and a noisy 3-D Rosenbrock (seed 3), noise SD 1, 200 evaluations:

| Run | Re-evaluations with a switch |
|---|---|
| seed 0, sphere | 7 of 11 |
| seed 1, sphere | 1 of 12 |
| seed 2, sphere | 9 of 13 |
| seed 3, Rosenbrock | 5 of 10 |

How the swapped-in GP differs from the working GP at a switch:
- On the spheres, typically: `mean_const` +3.2 to +13.6, log length scales −0.06 to −0.20, log output scale −0.11 to −0.38. Two switches were almost exact (under 0.005): seed 0 at iteration 2 and seed 1 at iteration 7.
- Log shape and log noise stay within ±0.06.
- Max |Δ log `poll_scale`| is at most 0.010. Max |Δ `len_scale`| is at most 0.95, and Δ `effective_radius` at most 0.007.
- On the Rosenbrock every difference is at most 0.03.

**A/B of the fix** (`v_f1f2_ab.py`). The optimizer source is rewritten in memory, touching no file.
- Removing both swaps changed all 6 seeded noisy runs; the first differing evaluation came at 57 to 125.
- Taking a deep copy at the swap instead (fixing F2 alone) changed 5 of 6.
- The final true f(x) moved both ways: for example 0.0016 to 0.0119, and 0.0185 to 0.0185. Six runs settle nothing about which is better.

**Where I agree or disagree with M.** I agree with the mechanism and the dating. One correction: the swapped-in GP carries whatever that slot now holds, which by F2 can be a later iteration's hyperparameters, not necessarily iteration `idx_impr`'s.

**Disposition: fix.**
- Delete `bads.py:1481` and `1505-1507`, keeping the `best_gp_hyp` lines. `1554` is dead too.
- This changes default noisy runs. Deterministic runs never reach the block.
- Gate: the fingerprint of deterministic runs must stay the same, plus a population comparison on a noisy configuration.

### M-F2: the swapped-in GP is shared with its history slot

**What I checked.**
- `IterationHistory.get` is `MutableMapping.get`, which goes to `dict.__getitem__` (`iteration_history.py:48-49`). It returns the stored object array, so `[i]` is the stored GP object itself.
- `record` deep-copies the value (93).
- `_expand_array` (95-101) reassigns the whole array through `__setitem__`, which deep-copies it (46). So every stored GP is copied whenever a new slot is added.
- After the swap, the working GP is slot k. Everything the next iteration does changes slot k in place: the rebuilds and refits of the search and the poll (`_robust_gp_fit_` sets the fitted hyperparameters on its input GP, `gaussian_process_train.py:687`) and `add_and_update_gp`.

The sharing ends at either of two points:
1. the next `record("gp", ...)` that grows the array, at the end of the next poll iteration or at termination; slot k then becomes a copy of the working GP of that moment;
2. a successful noisy search, which sets `gp = new_gp` (1931), a deep copy made at 1811.

**Measurement** (`v_f1f2_measure.py`). I checked identity (`is`) at every search and poll entry, and compared each slot's hyperparameters with `gp_hyp_full` at every re-evaluation.

| Run | Step entries whose working GP is a slot | Slots differing from `gp_hyp_full` at the end |
|---|---|---|
| seed 0 | 35 of 93 | 3 of 13 |
| seed 1 | 28 of 117 | 3 of 14 |
| seed 2 | 22 of 99 | 3 of 15 |
| Rosenbrock | 11 of 107 | 0 of 11 |

Size of the drift:
- Seed 0, slot 1: `[0.191, 0.208, 0.384, 0.006, -0.004, -7.489]` (log ℓ₁, log ℓ₂, log σ_f, log shape, log noise, mean).
- Seed 2, slot 1: `mean_const` −14.2. Seed 1: log σ_f +0.18.

Effect on decisions:
- I compared the switch choice from PyBADS's re-evaluation with the choice from the recorded hyperparameters (slot copy with `gp_hyp_full[i]`). They differ in 20 of 46 end-of-iteration re-evaluations: 5 of 11, 8 of 12, 7 of 13 and 0 of 10.
- For example, seed 1 at iterations 2-9: `choice PY None F2 1`.
- The re-evaluated `fval` differs by up to 0.02.
- The final choice (`q_beta`) was the same in the 3 runs that made a final re-evaluation.

**Where I agree or disagree with M.** I agree. M's slot lists match what I see. The shared slot starts at every iteration, not only at switches, because the swap at the current iteration also shares.

**Disposition: fix, with F1.** Once the swaps are gone, no slot is ever the working GP. For MATLAB's semantics in full, `_re_evaluate_history_` should rebuild a copy of the working GP (its unused `gp` argument) with `gp_hyp_full[i]` (`bads.m:1388`). That also settles M's "observed outside" point about per-slot geometry, and removes the need to store GPs at all. Same gates as F1.

### M-F3: final `fsd` computed with ddof=0 (fixed by #71)

- Code: `bads.py:1590-1596` now uses `np.std(yval_vec, ddof=1)/sqrt(n)`, as MATLAB's `bads.m:1469-1471`.
- Check (`final` section), seed 0: `res fsd 0.329102; std ddof0/sqrt(n) 0.312214; std ddof1/sqrt(n) 0.329102`. At `ab4dded` the same run gives `res fsd 0.312214`.
- Dating: `np.std` came in `8e59038` (2022-06-03). MATLAB's `std` dates from 2017.
- Disposition: none.

### M-F4: random x0 drawn in the original space

- PyBADS `bads.py:253-264` draws `rng.uniform(plb, pub)` in x, before `_init_optim_state_` builds the transform. MATLAB `setupvars.m:79-85` draws in u.
- Check (`f4` section), 200 seeds with plb 1 and pub 100 (a log-transformed variable; flags `[[ True False]]`): `fraction < 10: 0.085 (uniform in u: 0.5); median 57.87 (uniform in u: 10)`. The linear variable is unaffected.
- Dating: at the port, PyBADS used the midpoint in x and MATLAB the midpoint in u, so they never agreed for log-transformed variables.
- I agree with M.
- Disposition: fix, by drawing uniformly in the transformed plausible box and mapping back.
  - Runs with a finite x0 are unchanged (fingerprint).
  - With linear variables only, the numbers stay the same up to rounding.
  - Add a unit test of the distribution.

### M-F5: substitution of non-finite targets (unreachable)

- `gaussian_process_train.py:280-287` indexes the full array with an index into the finite subset.
- Check (`f5` section): I injected `inf` at the incumbent's row. The code substituted `57.47690201`, where MATLAB's rule (`gpupdate.m:258-265`) gives the maximum of the finite targets, `200.0000`.
- The `s2` guard `"S" in optim_state` (284) is never true, because nothing writes that key. MATLAB's `isfield(optimState,'S')` is true under `SpecifyTargetNoise`.
- `add_and_update_gp` has no counterpart of MATLAB's penalty (`gpupdate.m:69-78`).
- It is unreachable: a target that returns `inf` raises `ValueError: FunctionLogger:InvalidFuncValue` (`function_logger.py:160-170`, like `funlogger.m:103-105`). The `fun_values` path fails earlier, at `range(len())` (`bads.py:793`).
- I disagree with M's "defect in both". MATLAB at `74919c0` is right; before it, MATLAB would have errored.
- Disposition: fix cheaply, or add a comment. Nothing can move, so the fingerprint must stay the same.

### M-F6: final estimate recorded in the last slot (fixed by #71)

- `bads.py:1597-1601` now records at `min_q_beta_idx`, as `bads.m:1158-1159` does.
- Check, seed 0: the chosen iterate 2 holds `(0.147278, 0.329102)`, which equals the result. The last slot keeps its own `(-0.080560, 0.084867)`. At `ab4dded` the estimate sat in the last slot.
- Disposition: none.

### M-F7: `iterations` and `yval_vec`

- **`iterations`** is fixed by #71 (`optimize_result.py:119-120`, now `iter + 1`). Check: `res iterations 13; history length 13`. At `ab4dded` it was 12. It now equals MATLAB's `output.iterations = iter` (`bads_output.m:21`).
- **`yval_vec`** is still `None` for deterministic runs and for `noise_final_samples=0` (`optimize_result.py:124-130`). MATLAB returns `yval_vec = yval` (`bads.m:1134`, `bads_output.m:37`).
  - PyBADS's `optim_state["yval_vec"]` holds the same stale value MATLAB would return: the incumbent at the end of the loop, not the chosen iterate. In my check that is `-0.166`, while `yval` is `-0.815`.
  - The docstring (`optimize_result.py:28-32`) documents `None`, so I treat it as intentional. It is not on the sheet.
- Disposition: keep and document, by adding it to KD-B1-8. It is interface only, with no numerical gate. KD-B1-8's "not settled: the count in iterations" is now settled.

### M-F8: initial GP mean uses round(0.8N) instead of ceil(0.8N)

- `stats/get_hpd.py:35` uses `round(hpd_frac*N)`, feeding `gaussian_process_train.py:858` and `875`. MATLAB `gpdefBads.m:164-165` uses `ceil`.
- Check (`f8` section):

| Case | N | PyBADS | MATLAB |
|---|---|---|---|
| deterministic D=2 | 5 | 11.0681 | 11.0681 |
| deterministic D=4 | 9 | 28.1892 | 29.1212 |
| deterministic D=5 | 9 | 35.5929 | 37.0301 |
| noisy D=2 | 33 | 13.6082 | 13.6468 |

- It is only the starting value of the first fit.
- I agree with M.
- Disposition: fix narrowly, by computing only the starting mean MATLAB's way. Changing `get_hpd` would also move the high-density set used for the prior and the bounds. Either way it changes default runs, so it needs a population comparison.

### M-F9: unstable tie order in the nearest-neighbour training set

- `gaussian_process_train.py:1108` uses `np.argsort`, which is not stable. MATLAB's `sort` is stable (`gpupdate.m:94`).
- Synthetic check, 120 distances on a 0.25 grid: the first-50 sets differ (NumPy 2.5.3).
- In real runs no tie ever fell across the cutoff, and the training-set membership never changed. Only the row order changed:
  - sphere D=2: 1 of 41 selections;
  - Rosenbrock D=3: 13 of 124;
  - noisy sphere: 82 of 271 (in noisy runs, `n_train_max` is at least 200, so the cutoff never binds at 200 evaluations).
- I agree on the mechanism and see a smaller effect than M implies.
- Disposition: fix with `kind="stable"`. Run the fingerprint first; if the rounding changes move it, a population comparison, expected to show no flag.

### M-F10: NaN noise variances with `uncertainty_handling=True`

- Check (`f10` section): `s2` is all NaN as the code stands. Forcing `noise_flag` off gives an `identical evaluation sequence: True` over 120 evaluations.
- gpyreg ignores `s2` when `user_provided_add` is off (`noise_functions.py:43-46`, `265-272`).
- It is reached only with `uncertainty_handling=True` (not the default) and no specified noise.
- Disposition: keep, or tidy by holding `S` only at level 2, as MATLAB holds `S` only with `SpecifyTargetNoise`. The fingerprint must stay the same.

### C1: the noise test also runs with `uncertainty_handling=False` (record wrong)

- Check (`c1` section): with a noisy target and `uh=False`, the level is 0 at construction and 1 after initialization, with 34 target calls. That is identical to `uh=None`.
- With a deterministic target and `False`, initialization makes 6 calls: x0, the test and 4 design points. MATLAB would not run the test.
- Code: `bads.py:894-903` sets `False` to level 0, and `997` tests at level < 1. MATLAB `evalinitmesh.m:17` and `38-50` test only when the option is empty.
- AGENTS.md (lines 135-136) states MATLAB's condition, which the code does not implement.
- Disposition: fix the code to test only when the option is `None`. The option's description implies `False` means off. This changes only runs with `False`, so the fingerprint of default runs must stay the same.

### C2: extra doubling of the Sobol design (record wrong)

- `init_sobol.py:72-76` doubles the design when `2**ceil(log2(n))` equals D.
- Check: D=1 gives 2 points, D=2 gives 4, D=4 gives 8, D=8 gives 16, D=16 gives 32 and D=32 gives 64; a noisy D=32 run also gives 64. In real runs `eff_starting_points` is 3, 5 and 9 for D = 1, 2 and 4.
- MATLAB draws exactly `Ninit` points (`evalinitmesh.m:94-104`, `initSobol.m:16`).
- No comment or record explains the doubling. The Owen comment covers only the power of two, and KD-B7-1 leaves the doubling open.
- Disposition: correct the record: AGENTS.md line 137, and the docstring's "fun_eval_start: Number of initial function evaluations". Removing the doubling instead would need a population comparison at those D.

### C3: the Sobol seed does not follow MATLAB (record wrong)

- Check: every u0 inside the open plausible box (5 of 5 for D=2 and D=3) gives the same design, because the integer parts are all `[0 0]`. u0 = ±1 or 1.5 give other designs, and −1 casts to `18446744073709551615`.
- MATLAB (`initSobol.m:10-16`) takes the character codes of `num2str` of the first 10 values, and uses the result as a skip index (`i4_sobol_generate.m`: "SKIP, the number of initial points to skip"). PyBADS takes the integer parts of the first 11 values, and the result seeds scipy's scrambling.
- I did not open the plan, which the rules put out of bounds; I rely on the sheet's quotation of it. AGENTS.md's own description is right.
- Disposition: correct the plan's sentence. No gate.

### C4: kernel comment (record wrong)

- The comment at `bads.py:916` says "squared exponential", but the code selects `gp_cov_fun = 1 -> RationalQuadraticARD`.
- Disposition: correct the comment.

### C5: zero-start comment in the initial fit (record wrong)

- I injected failures into the first three `GP.fit` calls. Attempts 1-3 do not start from zeros; `attempt 4: hyp0 all zeros: True`.
- The comment at `gaussian_process_train.py:176-177` says this happens after the second failure.
- MATLAB fits no GP at initialization (`bads.m:465-469`). Its zeros are starting values for the covariance only (`gpdefBads.m:48`). PyBADS zeroes the noise and the mean too.
- Disposition: correct the comment.

### C6: "Missing port" comments (record wrong, claim confirmed)

- `gaussian_process_train.py:955`, `986`, `1163` and `1261` name PyVBMC features. A grep of the MATLAB tree finds no noise shaping, no integrated mean and no mean function 14; `gpdefBads.m:167` has only `@meanConst`.
- The output-warping comments (868, 903, 952, 988) are acceptable, since they name MATLAB's unsupported warped likelihood (KD-B6-4).
- Disposition: correct the comments.

### C7: stale MATLAB pointers (record wrong)

- `:267` "gpTrainingSet": that file became `private/gpupdate.m` (`'nearest'`, 85-111) in `d4fead5`.
- `:359` "line-code 302": the line was right at `d4fead5^` (301-302). It is 287-291 at `74919c0`.
- `function_logger.py:289`: MATLAB's `'done'` (`funlogger.m:132-147`, the same at the port) trims X, Y, S and `funevaltime`, and removes U, which is PyBADS's `X`. That comment never matched, so it is wrong from the start rather than stale.
- Disposition: correct the pointers.

### C8: changelog wording on failed GP updates (record right, ambiguous)

- In CHANGELOG.md (lines 135-144), "as in MATLAB BADS" introduces three items, and all three hold for MATLAB:
  - the point is appended and the next step rebuilds;
  - `UpdateTarget` always uses the current posterior (`bads.m:1296-1312`);
  - a NaN prediction counts as no improvement.
- The refit sentence is separate, and does not attribute the refit to MATLAB. The prep's point on the substance is right: MATLAB refits only through `IsRefitTime` (`bads.m:1242-1244`).
- "(in the poll, only with `poll_training` on)" is inexact: `bads.py:2133` forces the refit at iteration 0 even with `poll_training` off.
- Disposition: a light edit: "unlike MATLAB BADS, which refits only when its prediction check allows", and "after the first iteration".

## 3. New, outside the brief

1. **Reproduced.** A noisy run whose budget is smaller than the power-of-two design overruns it. With `max_fun_evals=30`, D=2 and `uncertainty_handling=None`, the run makes 34 evaluations, `max_fun_evals` rises to 34 and `noise_final_samples` falls to −4. With `True` it raises `ValueError: cannot convert float NaN to integer` (0/0 at `gaussian_process_train.py:1052-1058`). The cause is the 32-point design (`bads.py:1033-1054`) and `bads.py:1138-1146`. MATLAB caps the design at `MaxFunEvals−1` points (`evalinitmesh.m:101`).
2. **Unverified.** After a failed rebuild in a noisy run, the swap at 1481 hands on a slot whose `needs_rebuild`/`needs_refit` markers the re-evaluation removed, so the forced refit is lost.
3. **Performance.** `_expand_array` deep-copies every stored GP each time an iteration is added.
4. **Incidental data** on M's per-slot-geometry point. With the working GP's geometry, the Rosenbrock re-evaluations differ by up to 0.12 in `fval`, and the switch choice changed once (4 against 6). On the spheres the geometry made no difference.
5. **Small slips.** `init_sobol` returns the exponent where its docstring says "number of samples". `_get_gp_training_options:1056` mixes `x_` and `x` in its cubic (harmless as called). Unstable `argsort` also appears at `es_search.py:190`, `240` and `get_hpd.py:34`; I did not check those.
6. **Needs MATLAB.** Whether `prod(uint64(strseed))` saturates, which would make MATLAB's Sobol seed near-constant too.
