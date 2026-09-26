<!-- Report of the verifier of wave 2, slice B2 (the two B2 reports and the items kept from the reviewers, given as B2-K1 to B2-K9), reading PyBADS at fef6c14 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), with the complete history, in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave2/B2_verifier/. -->

# Wave 2 verification: B2

I checked every finding against PyBADS at `fef6c14` (`/home/user/pybads-review`), MATLAB BADS at `74919c0` (`/home/user/bads`) and gpyreg v1.3.3 (`/home/user/gpyreg-v1.3.3`).

- **Scripts and outputs:** all are in `/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/review/B2_verifier/`. Each script has a saved `.out` file from a second, sequential run through `run_all.sh`; the two runs gave the same results.
- **Imports:** every script printed `pybads: /home/user/pybads-review/pybads/__init__.py` and `gpyreg: /home/user/gpyreg-v1.3.3/gpyreg/__init__.py`.
- **Commit dates:** I dated lines with `git log -L` / `-S` on both histories. PyBADS commits: `c7c88ab` (2022-06-02, first port), `8e59038` (2022-06-03), `e004c79` (W0-1), `95da7f1` (#71) and `fef6c14` (#74, which squash-merges W1-19 `e041de0`, W1-26 `cd1831f` and W1-35), the last three all 2026-09-26. MATLAB: its first commit is `6c93629` (2017-03-14).
- **Report IDs:** I-Fn is the internal report, C-Fn the comparison report.

## 1. Summary

| Finding | Classification | Reached at default | Dating | Confidence |
|---|---|---|---|---|
| I-F1 = C-F4 = B2-K1: the move after the re-estimation writes `self.best_u`; `u` moves, `u_best` does not | The `best_u` line: **confirmed, inert** (dead attribute). The behaviour: **confirmed shared defect**, a design decision because fixing it departs from MATLAB | yes, levels 1 and 2 | Python has matched MATLAB's behaviour since `c7c88ab`, only because `best_u` is a dead name; MATLAB unchanged since 2017-03 | high (mechanism), medium ("defect") |
| I-F2: the moved target hyperparameters are overwritten at every pass | **confirmed, inert** (shared with MATLAB, `bads.m:1059`) | the code runs; it has no effect at default | the same since both first versions | high |
| I-F3 = C-F2 = B2-K5: the initial design exceeds a small `max_fun_evals`; the reserve for final samples goes negative | **confirmed port discrepancy** (rounding after the cap) plus **confirmed shared defect** (the noise test is not counted: +1 evaluation on both sides) | no | never agreed (`c7c88ab` against 2017) | high |
| I-F4 = C-F6: `sloppy_improvement=False` crashes | **confirmed port discrepancy** | no | `c7c88ab` | high |
| I-F5 = C-F8: `f_vals` crashes the first display | **confirmed defect** (the option exists only in PyBADS) | no | `c7c88ab` | high |
| I-F6: `overhead` leaves the final samples out of the target's time | **confirmed port discrepancy** for the final samples; the untimed noise test is shared | yes, levels 1 and 2 | never agreed | high |
| I-F7 = C-F7: `display="notify"`/`"final"` behave as `"iter"` | **confirmed port discrepancy** | no | `c7c88ab` | high |
| I-F8: the final message with 0 or 1 final samples | **not a defect** as a port matter: MATLAB has the same message logic. The (2,1) shape is C-F13 | no | shared | high |
| I-F9: termination is checked at every pass | **not a defect**: faithful to `bads.m:1062-1085` | yes | shared since 2017 / `8e59038` | high |
| I-F10: noisy incumbent keeps the raw design minimum through two iterations | **design question** (shared: MATLAB's `iter > 1`) | yes, levels 1 and 2 | shared since both first versions | high |
| I-F11: both choices skip the first iterate | **not a defect**: faithful to MATLAB's deliberate `75ec49f` | yes, levels 1 and 2 | MATLAB 2022-05-09; Python `8e59038` | high |
| I-F12 = B2-K2 (termination part): `min_iter`, `min_fun_evals` are unread | **confirmed, inert** (PyVBMC leftovers; MATLAB has no counterpart) | not applicable | `c7c88ab` | high |
| I-F13 = C-F3: the accelerated mesh reduction is tested one iteration late | **confirmed port discrepancy** | yes, all levels | never agreed (`c7c88ab`; `8e59038` fixed the sibling guards only) | high |
| C-F1: `tol_noise` is eps·tol_fun, not sqrt(eps)·TolFun | **confirmed port discrepancy** | the test runs at default; its outcome differs only when the repeat differs by (2.2e-19, 1.5e-11] | never agreed | high |
| C-F5: the current iterate keeps its estimate after a failed rebuild | **intentional difference, missing from the sheet** | only when that rebuild fails, levels 1 and 2 | `fef6c14` (W1-35) | high |
| C-F9: the Actions column shows a stale or merged action | **confirmed port discrepancy** (display only) | yes (`display="iter"`), all levels | never agreed | high |
| C-F10: a run that ends in initialization reports `iterations=0`; MATLAB reports 1 | **design question**; contradicts KD-B1-8 | no | `95da7f1` | high |
| C-F11: `output_fcn` timing, message and stop semantics | The message and stop semantics: **intentional difference, missing from the sheet** (#71). The `'init'` timing: **design question** | no | `'init'` timing since `c7c88ab`; the rest `95da7f1` | high |
| C-F12: `fun_values` crashes | **confirmed port discrepancy** | no | `c7c88ab` | high |
| C-F13: `yval_vec` has shape (2,1) with one final sample at level 1 | **confirmed port discrepancy** | no | `2650aef` (2022-11-04) | high |
| B2-K3 | Clause 1 and the marker clause of wave 0: **no longer holds** (`e004c79`). Clause 2: **no longer holds** for past iterates (NaN, `fef6c14`); for the current iterate it is C-F5. W1-35's clauses: **hold** | only on failure, levels 1 and 2 | — | high |
| B2-K4 | **no longer holds** (`e004c79`): the working GP's geometry is used, as in MATLAB | yes, levels 1 and 2 | — | high |
| B2-K6 | The `ValueError`: **no longer holds** (W1-26, `cd1831f`). The course of such a run: **design question**, shared; the GP on one point **needs MATLAB** | no (needs `non_box_cons`) | — | high (course), low (MATLAB detail) |
| B2-K7: `IterationHistory` deep-copies every stored GP when it grows | **confirmed, inert** (performance only) | yes | `c7c88ab` | high |
| B2-K8: the loop discards the GP that `_poll_step_` returns | **confirmed, inert** | yes | — | high |
| B2-K9: NaN estimates can stay in `iteration_history` after a run | **confirmed, inert** (as ruled; the same as MATLAB's `iterList`) | only on failure, levels 1 and 2 | `fef6c14` | high |

## 2. Per finding

### I-F1 = C-F4 = B2-K1: the move after the re-estimation, `self.best_u`

The two reports describe the same behaviour, and the survey's row (B2-K1) is the same line; I verified it once.

**Lines**
- **PyBADS:** `bads.py:1531-1541` (1536 `self.best_u = self.u.copy()`), `1406` (`self.u = self.u_best` at every pass), `1737-1739` (the search target at `u_best`), `1757` (the ES search at `self.u`). `best_u` is never read; `u_best` is read at 1406, 1738, 1884 and 2408.
- **MATLAB:** `bads.m:1111-1118` sets `u`, not `ubest`. `539` `UpdateTarget(ubest,fhyp,...)`; `543-549` search at `u`; `665` `udist(ubest,...)`; `769` `u = ubest` at every pass; `777` `fpollhyp = fhyp`.

**Check:** `v_move2.py` (noisy sphere, D=2, SD 0.5, 200 evaluations, 6 seeds).
- It reads each move from a snapshot of the history taken after the re-estimation.
- Default options:
  ```
  total {} {'moves': 25, 'to_other_location': 23, 'search_centred_at_idx': 23, 'search_target_at_old_ubest': 23, 'search_target_hyp_moved': 23, 'polled': 22, 'poll_at_idx': 0, 'poll_at_old_with_moved_fval': 9, 'poll_hyp_moved': 0}
  ```
- After every move to another location, the next search is centred at the chosen iterate, and its target is predicted at the old `u_best` with the chosen iterate's hyperparameters. No poll ran at the chosen iterate. In 9 of 22 polls the poll ran around the old `u_best` with the moved `fval`; in the rest a search had moved the incumbent first.
- This is MATLAB's flow line for line, as the table of lines shows.

**Dating**
- The Python line was written in `c7c88ab` as `self.best_u = self.u.copy() # TODO in Matlab is not done`. The final choice in the same file wrote `self.u_best = ... # TODO in Matlab is not done`, so `best_u` was most likely meant to be `u_best`. `8e59038` dropped the comment.
- MATLAB's block has set only `u` since the first commit (`6c93629`, `7dcdc2b` 2017-03-16; `75ec49f` only added "skip first").
- So the Python behaviour has matched MATLAB's since the port, only because the attribute is dead.

**Agreement with the reports**
- I agree with the comparison reviewer (behaviour shared).
- I disagree with the internal reviewer's "port discrepancy": the port does exactly what MATLAB does.
- The shared behaviour is inconsistent: the incumbent's (location, value) pair belongs to two different points until the next improvement or re-estimate. The poll's improvements are judged against another point's lower value, and the round records the old location with that value. `optim_state["u"/"fval"]` are not updated either, as MATLAB's `optimState` is not.

**Consequence:** `v_move_ab.py` compares the port with a variant that also moves `u_best` (10 seeds, 150 evaluations): `changed runs 4 of 10 | median err port 0.0078 variant 0.0077 | variant better in 2`. The other two changed runs were worse, the largest change being seed 1, 0.0184 → 0.0293. No direction is established.

**Recommended disposition: decide the design.**
- Option (a): keep MATLAB's behaviour and delete the dead line. Inert; the fingerprint must be unchanged.
- Option (b): move `u_best` and `optim_state["u"...]` with the value, i.e. call `_update_incumbent_`. This departs from MATLAB, changes noisy runs at default, and needs a population comparison whose configuration includes the benchmark's noisy targets, on both references.

### I-F2: the moved target hyperparameters

**Lines**
- **PyBADS:** `bads.py:1425` (`best_gp_hyp` reset at every pass), `1537-1541`, `1737-1744`, `2103`.
- **MATLAB:** `bads.m:1059` (`fhyp = gpstruct.hyp` at every pass), `539`, `777`.

**Check:** `v_move2.py`.
- At default options the moved vector reached the next search's target in 23 of 23 moves and the poll in 0 of 22.
- With `search_n_try=0` (`v_move2_nosearch.out`) it reached the poll in 25 of 25, as it would in MATLAB.
- The search's target (`optim_state["f_target*"]`) is read by nothing in `pybads/search`, `acquisition_functions` or `poll` (grep). MATLAB reads its target in `searchES.m:151-163` only with `acqNegEIMin`/`acqNegPIMin`, which are not the default.

**Agreement:** the mechanism is as the internal reviewer says, but it is MATLAB's. The comment at 1537-1538 accurately describes the assignment. There is no consequence at default options.

**Dating:** the same on both sides since both first versions (`c7c88ab` / `6c93629`).

**Recommended disposition: keep** (no action).

### I-F3 = C-F2 = B2-K5: the initial design against a small budget

**Lines**
- **PyBADS:** `bads.py:1066-1071`, `1073-1078` (the cap `min(fun_eval_start, max_fun_evals - 1)`), `init_sobol.py:71-76` (the rounding up to 2^k, doubled when equal to D), `bads.py:1175-1184` (the reserve).
- **MATLAB:** `private/evalinitmesh.m:38-50` (the noise test calls `funwrapper` directly, then `funccount+1`), `93-104`, `bads.m:441-442`.

**Check:** `v_budget.py`, with a transcription of MATLAB's count, `1 + test + min(Ninit', MFE-1)`:

| Run | PyBADS | MATLAB |
|---|---|---|
| D=2 det, MFE 3 / 4 / 5 / 6 | 6 evaluations each | 4 |
| D=3 det, MFE 4 | 6 | 5 |
| D=5 det, MFE 7 | 10 | 7 |
| D=2 noisy, MFE 10 | 18, nfs −8, MFE raised to 18 | 11, nfs −1, MaxFunEvals 11 |
| D=2 noisy, MFE 25 | 34, nfs −9, MFE raised to 34 | 22, nfs 3 |
| D=2 noisy, MFE 38 | 34, ends in iteration 1 with `fsd 1.0`, no final samples | 22, nfs 10 |

- MATLAB also overshoots by one whenever the cap binds and the noise test runs (D=2, MFE 3 → 4; noisy MFE 10 → 11, reserve −1).
- No run raised: the division by zero of wave 0's record no longer happens (W1-19, `e041de0`, in `fef6c14`).
- Wave 1's "3 or 4 at D=2 makes 6 evaluations" holds.
- A negative or zero reserve always coincides with a run that ends at its first pass, so no final estimate is ever computed from it.

**Dating:** the cap and the rounding are both in `c7c88ab` (2022-06-02), so they never agreed with MATLAB. The internal report dates the rounding to `cdc2e0f`; that is wrong. MATLAB's lines date from 2017, the noise test from `fd3f7a2`.

**Sheet:** KD-B7-1 settles the power-of-two size, not the cap. This is not a contradiction, but the entry's justification does not cover exceeding the budget.

**Recommended disposition: fix.**
- Cap after rounding, with the cap computed as `max_fun_evals - func_count` so that it counts the noise test: round down to a power of two, or truncate the Sobol set.
- Floor the reserve at 0.
- Touches `bads.py:1073-1088`, `init_sobol.py` and `1175-1184`.
- Default budgets (500·D) are unaffected: gate with an unchanged fingerprint and a budget test.

### I-F4 = C-F6: `sloppy_improvement=False`

**Check:** `v_options.py` gives `AttributeError 'float' object has no attribute 'copy' | at pybads/bads/bads.py 1349`. `2.0**0` is a Python float, and only `np.maximum` made it a NumPy scalar.

**Lines:** PyBADS `bads.py:1339-1349`; MATLAB `bads.m:504-507` supports the option.

**Dating:** the `.copy()` dates from `c7c88ab`.

**Recommended disposition: fix** (`np.asarray(...)` or drop `.copy()`). Not reached at default: unchanged fingerprint and a test.

### I-F5 = C-F8: `f_vals`

**Check:** `v_options.py` gives `ValueError Unknown format code 'f' for object of type 'str' | at pybads/bads/bads.py 2886`, with `display="off"` too.

**Lines**
- **PyBADS:** `bads.py:581-604` (a cache only, `cache_active`), `2840-2869` (an 8-field format), `2871-2894` (6 or 7 values passed).
- **MATLAB:** there is no `f_vals` (KD-B1-4); its import is `FunValues` (C-F12).

**Dating:** broken since `c7c88ab`.

**Recommended disposition: fix or remove.** If removed, the change needs an "Upgrading" line, since unknown option names raise. Unchanged fingerprint.

### I-F6: `overhead`

**Lines**
- **PyBADS:** `bads.py:1648-1658`; `function_logger.py:384-398` (a call with `record_duplicate_data=False` returns without adding to `total_fun_eval_time`); `402-429` (a merged level-2 repeat does not add either); `440` (only a new row adds).
- **MATLAB:** `private/funlogger.m:130` adds the time of every `'iter'` and `'single'` call, final samples included; its noise test calls `funwrapper` directly (`evalinitmesh.m:41`) and is untimed. `bads_output.m:48` computes the overhead.

**Check:** `v_overhead.py` (50 ms target, 60 evaluations):
```
det  : ... reported overhead 0.245, overhead from target time 0.223, MATLAB-style (all but the noise test) 0.245
noisy: ... reported overhead 0.330, overhead from target time 0.089, MATLAB-style (all but the noise test) 0.107
```
- Deterministic runs agree with MATLAB's convention.
- Noisy runs leave the 10 final samples untimed, where MATLAB times them.

**Dating:** never agreed (Python `c7c88ab`; `9915bbf` 2022-11-13 timed the untimed calls per row but not in the total; MATLAB `603da99` 2017-04-24).

**Recommended disposition: fix.** Add these times to the total, and optionally the noise test (a departure). The change touches only an output: unchanged fingerprint, as long as the fingerprint excludes timings.

### I-F7 = C-F7: display levels

**Check:** `v_options.py`:
```
display='notify': logger level INFO, records emitted 17
display='final': logger level INFO, records emitted 17
display='iter': logger level INFO, records emitted 17
```

**Lines**
- **PyBADS:** `bads.py:224-232`; `basic_bads_options.ini:2-3` lists the values.
- **MATLAB:** `bads.m:317-328` sets `prnt` (notify 1, final 2), which controls `evalinitmesh.m:52`, `65`, `77` and `bads.m:1172`.

**Dating:** `c7c88ab`. KD-B2-3 leaves the content of the display open, so the sheet does not settle this.

**Recommended disposition: fix** (map notify to the opening line only, final to that plus the final message, e.g. through two logger levels or a filter). Unchanged fingerprint.

### I-F8: the final message with 0 or 1 final samples (C-F13 separately)

**Check:** `v_misc.py`.
- With `noise_final_samples=0` the message prints the last incumbent's observation (−0.1508) beside the chosen iterate's GP estimate (−0.0518).
- At level 2 with one sample it prints "(1 sample) … (GP mean ± SEM)".
- At level 1 with one sample it prints "from 2 samples".

**MATLAB:** identical. `bads.m:1136` sets `yval_vec = yval` before the choice at `1138-1158`; the message is at `1172-1181`; `FinalEstimate` at `1464-1465` appends `yval` for one sample.

**Agreement:** I disagree with the internal "port discrepancy": this is shared, and the only Python-only parts are the shape (C-F13) and an array printed as `[0.34511739]` at level 2.

**Recommended disposition: keep** (optionally print the chosen iterate's observation; display only).

### I-F9: termination at every pass

**Check:** `v_misc.py`: `max_iter=1: polls 1, searches 0`; `max_iter=2: polls 1, searches 1, iterations 2`; `max_iter=3: polls 2, searches 5`.

**MATLAB:** `bads.m:1062-1085` runs at every pass with `iter >= MaxIter` and `iter > TolStallIters`. Python's `poll_iteration >= max_iter-1` and `> tol_stall_iters-1` with base index `poll_iteration - T` are its exact 0-based transcription.

**Agreement:** the mechanism is as described, and it is MATLAB's; AGENTS.md and the CHANGELOG of #71 state the counting.

**Recommended disposition: keep.** At most, correct the record: the descriptions of `max_iter` and `tol_stall_iters` could say that a round counts once begun.

### I-F10: the noisy incumbent's value in the first two iterations

**Check:** `v_misc.py`: `design raw minimum -0.7563; incumbent (fval, fsd) entering polls 1-3: [(-0.7563, 1.0), (-0.7563, 1.0), (-0.0692, 0.1269)]`.

**Lines:** PyBADS `bads.py:1503-1507` (`poll_iteration > 0`); MATLAB `bads.m:1097` (`iter > 1`), unchanged since `6c93629`.

**Recommended disposition: decide the design** (low priority).
- Option (a): keep MATLAB's behaviour.
- Option (b): re-estimate from the first poll. Changes noisy runs at default and needs a population comparison with noisy configurations.

### I-F11: skipping the first iterate

**Lines**
- **PyBADS:** `bads.py:1523` and `1580`, added in `8e59038` (2022-06-03). `c7c88ab` did not skip.
- **MATLAB:** `bads.m:1108` "Skip first" and `1146`, both added in `75ec49f` (2022-05-09) by MATLAB's author.

**Recommended disposition: keep.** It is a faithful port of a deliberate MATLAB change.

### I-F12 = B2-K2 (termination part): `min_iter`, `min_fun_evals`

B2-K2's part is covered by I-F12.

**Check:**
- A grep of `pybads/` finds the two names only in `advanced_bads_options.ini:281-284`; the `min_iter` in `test_gp_update_failures.py` is a local parameter.
- A grep of the whole MATLAB tree finds no `MinIter`/`MinFunEvals`.
- Their defaults (5·D, D) are PyVBMC's options of the same names (from memory; PyVBMC is not in the environment). They appeared in `c7c88ab` as `minfunevals`/`miniter` beside other PyVBMC options and were renamed in `cdc2e0f`.
- Their descriptions reach the user documentation through the options page, which includes the `.ini` files verbatim.

**Recommended disposition: correct the record.** Add them to KD-B1-5 (d), or remove them with an "Upgrading" line. Unchanged fingerprint.

### I-F13 = C-F3: the accelerated mesh reduction

**Lines**
- **PyBADS:** `bads.py:2421-2446` (`optim_state["iter"] > accelerate_mesh_steps`, 0-based, base `iter - steps`).
- **MATLAB:** `bads.m:976-982` (`iter > AccelerateMeshSteps`, 1-based). The base index is the same iterate; only the guard is one iteration late.

**Check:** `v_accel.py` evaluates MATLAB's guard beside the port's at each failed poll. At 0-based iteration 3, where only MATLAB tests, MATLAB would halve the mesh again in 4 of 8 default runs:
```
sphere D=2 x0=0 seed 0: ... iteration where only MATLAB tests: [(3, 'MATLAB would halve again')]
rosen D=2 seed 0: ... [(3, 'MATLAB would halve again')]
ellipsoid D=4 seed 0: ... [(3, 'no extra halving')]
```

**Dating:** `c7c88ab`. `8e59038` made the same off-by-one correction for `max_iter` (`>= maxiter - 1`) and the stall criterion (`> tolstalliters - 1`), but not here. MATLAB is unchanged since 2017.

**Recommended disposition: fix** (`iter >= steps`). This changes default runs at every level: gate with a population comparison at default options on both references.

### C-F1: the noise test's threshold

**Lines:** PyBADS `advanced_bads_options.ini:13` (`np.spacing(1.0) * tol_fun`), read at `bads.py:1037`; MATLAB `bads.m:195` (`sqrt(eps)*options.TolFun`), used at `evalinitmesh.m:43`.

**Check:** `v_tolnoise.py` uses a deterministic target whose value depends on the order of a summation:
```
max |diff| 1.11e-16
PyBADS tol_noise 2.22e-19 ; MATLAB's would be 1.49e-11
target_type: stochastic | level 1 | design points 32 | tol_stall_iters 10 | n_train_min 100 | func_count 100 | fval 0.251329 fsd 5.85e-18
```
PyBADS runs this deterministic target as a noisy one; MATLAB's threshold would call it deterministic. No record justifies the value; the option's description is MATLAB's.

**Dating:** never agreed (MATLAB `fd3f7a2` 2017-03-29; Python `c7c88ab`).

**Recommended disposition: fix** (the `.ini` expression). The benchmark's targets either repeat exactly or have noise far above 1.5e-11, so the fingerprint should be unchanged; add a test with a one-ulp target.

### C-F5, with B2-K3: a failed rebuild of the current iterate

**Lines:** PyBADS `bads.py:2811-2815`; MATLAB `bads.m:1378-1412`, `private/gpupdate.m:340-354` (`post = []`), `utils/gppred.m:39-55` (with an empty `post` it redoes the same inference inside `try`, which leaves NaN).

**Check:** `v_reeval_fail.py` injects `LinAlgError` into the posterior updates of the re-estimation:
```
re-eval 2 (n=4) failed at [3]: before [    nan -0.2115 -0.2141 -0.233 ]
      after [ 0.1594 -0.0832 -0.0864 -0.233 ] | working GP same object/data/hyp: True, markers (None, None)
```
The current iterate keeps its recorded estimate, where MATLAB would give NaN.

**Why intentional:** the PI's ruling on W1-35 (2026-09-26), the CHANGELOG `[Unreleased]` entry ("when that iterate is the current one, it keeps its estimate"), the docstring, and `test_noisy_re_estimate_after_failed_rebuild`, which pins it. It is not on the sheet.

**Recommended disposition: keep and document** (a KD entry under B2).

### C-F9: the Actions column

**Lines:** PyBADS `bads.py:2473-2484` and `2892` (`logging_action[-1]`); MATLAB `bads.m:1016-1028` rebuilds `action` at every poll ("Train", " (failed)", ", skip").

**Check:**
- `v_misc.py` with `search_n_try=0`: 15 of 29 polls show a stale "Train".
- `v_actions.py` at default options: 1 of 8 polls (Rosenbrock D=2, seed 0), and 3 of 11 and 2 of 12 in two noisy sphere runs; 0 in Rosenbrock seed 1 and Ackley D=3.

**Dating:** never agreed (`c7c88ab`/`8e59038`).

**Recommended disposition: fix** (build the action string per poll). Display only: unchanged fingerprint.

### C-F10: `iterations` for a run that ends in initialization

**Check:** `v_misc.py`: `max_fun_evals=1 ... iterations 0` for `uncertainty_handling` None and True.

**Lines:** PyBADS `optimize_result.py:120` with `bads.py:820` (`iter = -1`); MATLAB `bads.m:482` (`iter = 1` before the loop) and `bads_output.m:21`.

**Sheet:** KD-B1-8 says `iterations` "counts as MATLAB's `output.iterations` does, from 1". That is not true for this case, and the tests `test_one_function_evaluation` and `test_output_fcn_stops_run_at_init` pin 0.

**Dating:** never agreed (`95da7f1`).

**Recommended disposition: decide the design.** Either report 1 as MATLAB does (and change the two tests), or keep 0 and correct KD-B1-8 and the CHANGELOG wording. Unchanged fingerprint.

### C-F11: the output function

**Check (by reading):**
- The `'init'` call at `bads.py:1287-1300` comes after `_init_optimization_`, i.e. after the noisy option changes and the first GP fit. MATLAB calls it at `bads.m:427`, before `431-457`.
- A stop gets its own message in PyBADS. MATLAB's `msg` stays "…after initialization." (`bads.m:425`).
- MATLAB assigns the return value to `isFinished_flag`, so a false return can reopen a run that `evalinitmesh` ended. PyBADS applies `stop and not is_finished`.

**Dating:** the `'init'` placement since `c7c88ab`; the rest since `95da7f1` (#71).

**Recommended disposition: keep and document.** The message and stop semantics improve on MATLAB and should get a KD entry. Decide whether the `'init'` call should move before the option changes. Unchanged fingerprint.

### C-F12: `fun_values`

**Check:** `v_funvalues.py`: `ValueError The truth value of an array ... | at pybads/bads/bads.py 755` for 1 and 2 points.
- Other defects follow behind it: `range(len())` at `787`, and `self.function_logger` is used before `__init__` creates it (`286` against `290`).
- MATLAB (`setupvars.m:127-165`, `funlogger.m` 'init') keeps the imported points in the log and takes its incumbent from x0 and the design only (`evalinitmesh.m:121-123`). PyBADS's argmin over the whole log (`bads.py:1114-1119`) would include them, but that cannot be reached today.

**Dating:** `c7c88ab`.

**Recommended disposition: fix** (with B1), and restrict the argmin to x0 and the design. Unchanged fingerprint.

### C-F13: the shape of `yval_vec`

**Check:** `v_misc.py`: `level 1, nfs=1: yval_vec shape (2, 1)`; every other case is `(n,)`.

**Lines:** PyBADS `bads.py:1606-1612` (`np.vstack`); MATLAB `bads.m:1465` gives a 1×2 row.

**Dating:** `2650aef` (`c7c88ab`'s `np.vstack(yval_vec, self.yval)` could not run).

**Recommended disposition: fix** (`np.append`). Unchanged fingerprint and a test.

### B2-K3: the re-estimation rebuilds the stored GPs; failed rebuilds

Covered in part by C-F5 and by both reports' answers to Q3; I verified it as an item of its own with `v_reeval_fail.py`.

| Clause | At `fef6c14` | Evidence |
|---|---|---|
| The stored GPs are rebuilt in place | **No longer holds**: `e004c79` (W0-1) works on `tmp_gp = copy.deepcopy(gp)` and reads only `u` and `gp_hyp_full` from the history | `stored GPs changed after their record: [] of 12` |
| A failed rebuild records the restored GP's value | **No longer holds** for past iterates, which get NaN as in MATLAB (`fef6c14`); for the current iterate it is C-F5 | `failed at [0]: ... after [nan -0.2115 -0.2141]` |
| Wave 0's markers clause (the swap hands on a slot whose markers the re-estimation removed) | **No longer holds**: `e004c79` removed the swap | `working GP same object/data/hyp: True, markers (None, None)` in every injected case |
| W1-35: past iterate NaN; both choices skip NaN; the current iterate keeps its estimate; the incumbent is never NaN | **Holds**: `bads.py:1525-1527` (`nanargmax` over `[1:]`), `1581-1582` (`nanargmin`) | the run with 11 of 12 iterates NaN returned `fval 0.2377` at the current iterate |

**Recommended disposition: correct the record** (the survey's row).

### B2-K4: whose geometry selects the neighbours in the re-estimation

Covered by both reports' Q3 answers, not as findings.

**Check:** `v_neighbors.py` (D=3 noisy, 3 seeds, 271 rebuilds) intercepts `get_grid_search_neighbors` during `_re_evaluate_history_`:
```
seed 0: ... iterate rebuilds 77, geometry = working GP's 77, = stored GP's 12, stored geometry differs from working 65, of which neighbour set would differ 0
```
- `local_gp_fitting` without a refit leaves `len_scale` and `effective_radius` alone (`gaussian_process_train.py:530-594` run only when `refit_flag` is set). The copy of the working GP therefore keeps its geometry, as MATLAB's `gpupdate` does without a refit (`gpupdate.m:85-97`, `279`).
- At these sizes the stored GPs' geometry would change no neighbour set either: the noisy floor of 100 points and the radius cover the whole log.

**Verdict:** **no longer holds** since `e004c79`. **Correct the record.**

### B2-K6: a thin feasible band as `non_box_cons`

The reports do not cover it.

**Check:** `v_thin.py` and `v_thin2.py`, with `|x1 − x2| ≤ 0.005`, x0 = (0.5, 0.5), 200 evaluations.

**D=2, deterministic, seeds 0 and 1:**
- None of the 4 design points is feasible (`4 4 0 0.1309`).
- The GP is trained on x0 alone. The `ValueError` of wave 0 is gone (W1-26, `cd1831f`), but gpyreg's `get_bounds_info` emits RuntimeWarnings (log of zero spread).
- No search runs: there are no more than D points.
- Every poll has 4 axis-aligned candidates, all infeasible (the smallest |x1−x2| goes 2.0, 1.0, …, 0.031).
- The mesh shrinks 0.5 → 0.0039, with the acceleration from iteration 4.
- The stall criterion ends the run at iteration 6, after 2 evaluations, at x0: `func_count 2, iterations 6, x [0.5 0.5], ... change in the function value less than options['tol_fun']`.

**Other cases:**
- At D=3 the polls along x3 are feasible, and the run converges to (0.297, 0.302, 0.300) in 45 evaluations.
- Noisy at D=2, the doubled stall window lets the mesh reach the band: 8 polls go by without an evaluation, then the run continues to 108 evaluations.

**MATLAB:** its LTMADS basis is also axis-aligned at these meshes (`poll/pollMADS2N.m:7-14`: `nmax = 1`), with the same stall criterion, so the course is shared. How MATLAB's `gpdefBads`/`gpfit` behaves on one point (`log(std(y)) = -Inf`) **needs MATLAB**.

**Recommended disposition: decide the design.**
- Option (a): document that a band thinner than the mesh can resolve ends on `tol_fun`.
- Option (b): do not let the stall criterion fire over iterations without an evaluation. This departs from MATLAB; it is not reached at default, so the gate is an unchanged fingerprint plus a configuration with `non_box_cons`.

### B2-K7: `IterationHistory` deep copies

This is the comparison reviewer's "observation, not a finding".

**Check:** `v_ih_copy.py`: `200 records: 20100 GP deep copies ..., 1.66 s` (n(n+1)/2). `_expand_array` (`iteration_history.py:95-101`) reassigns the grown array through `__setitem__` (`46`), which deep-copies it.

**Dating:** `c7c88ab`; MATLAB stores no GPs per iteration.

**Recommended disposition: fix** (use `dict.__setitem__` in `_expand_array`). Performance only: unchanged fingerprint.

### B2-K8: the GP that `_poll_step_` returns

Covered by the comparison's "checked, no finding" and the internal Q2.

**Check:** `v_misc.py`: `poll returned the same GP object in 10 of 10` and `29 of 29` polls. `local_gp_fitting` restores in place; `_robust_gp_fit_` (`gaussian_process_train.py:790`) and `add_and_update_gp` (`1370`) return the object they were given.

**Recommended disposition: keep**, or assign the return value for robustness (unchanged fingerprint).

### B2-K9: NaN left in `iteration_history`

The reports do not cover it.

**Check:** `v_reeval_fail.py` makes the last re-estimation fail at every past iterate: `NaN left in history after run at [0, …, 10]`, while the result's `fval` is finite.
- This is W1-35 as ruled, and MATLAB's `iterList` would hold NaN too.
- The CHANGELOG documents "NaN in `iteration_history`".
- Only consumers of the history (`bads_dump.py`, out of scope) see it.

**Recommended disposition: keep and document** (for example in the `IterationHistory` documentation page).

## 3. Met while verifying (unverified unless stated)

1. `bads.py:799` sets `optim_state["lastreeval"]`, a key nothing reads, beside the `last_re_eval` of line 599 that is read (grep).
2. On a one-point training set, gpyreg's `get_bounds_info` (called from `_gp_hyp`, `gaussian_process_train.py:954`) emits "divide by zero in log", "Degrees of freedom <= 0" and "invalid value in divide" RuntimeWarnings (seen in `v_thin_warn.py`; not analysed; B6).
3. `constraints_check.py`: `u1_idx = idx_sort[idx_sort < len(u1)]` is unsorted (the sorted variant is commented out), so candidates come out in `np.unique`'s lexicographic order, not their original order. Not compared with MATLAB's `uCheck`; B7.
4. At level 2 with `noise_final_samples=1`, the final message prints the array (`[0.34511739]`) where MATLAB prints a number (seen).
5. A merged level-2 repeat is not added to `total_fun_eval_time` (by reading; part of I-F6).
