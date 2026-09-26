<!-- Report of the verifier of wave 3, slice B4 (the two B4 reports and the items kept from the reviewers, given as B4-K1 onwards, briefs/wave3_kept_B4.md), reading PyBADS at 8aecb6a in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), with the complete history, in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave3/B4_verifier/. -->

# Wave 3 verification: B4

I checked PyBADS at `8aecb6a` (`/home/user/pybads-review`), MATLAB BADS at `74919c0` and gpyreg v1.3.3. Every script printed `/home/user/pybads-review/pybads/__init__.py` and `/home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. The scripts and their outputs (`*.out`) are in `/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/review/B4_verifier/`. The emulation of MATLAB's target is in `vhybrid.py`; under the GP's own hyperparameters it reproduces `gp.predict` exactly, in both of gpyreg's representations. I made no level-2 run: reach at level 2 comes from reading the code.

**Items the two reports share, verified once each:**
- I-F3 = C-F1 = K6
- I-F4 = C-F3
- I-F6 = C-F2
- I-F7 = K3 (also in C §2.4)
- I-F10 = K5
- C-F4 = K2
- I-F1 and I-F2 also appear as observations in C §2.1.

## 1. Summary

| Finding | Classification | Reached at default | Dating | Confidence |
|---|---|---|---|---|
| I-F3 = C-F1 = K6 (p_less: no sort, D+1 terms) | confirmed port discrepancy | yes, levels 0/1/2, at every poll step; it decides only after a good poll | never matched (Python `c7c88ab`; MATLAB unchanged since 2017) | high |
| I-F6 = C-F2 (`contraints_check` keeps logged points; slice B3's module) | confirmed port discrepancy | yes, all levels (measured at level 0) | matched at `c7c88ab`; broken by `8e59038` (2022-06-03) | high |
| I-F4 = C-F3 (`uncertain_incumbent=False` crashes) | confirmed port discrepancy | no (level 0 with a non-default option) | the branch never worked (`c7c88ab`, `9037851`) | high |
| C-F4 = K2 (target recomputed under `hyp_best`; MATLAB uses a hybrid) | design question | yes, all levels; hyperparameters differed in 5 of 34 measured stop decisions, all at level 1 | never matched (`c7c88ab` used the current GP, `9037851` recomputes) | high on the mechanism, medium on MATLAB (emulated) |
| I-F10 = K5 (`np.seterr` changed for the whole process) | confirmed defect (Python only) | yes, all levels | `f9e9326` (2022-11-02) | high |
| I-F7 = K3 (fallback keeps the non-finite variance) | confirmed shared defect, inert in practice | no (needs a non-finite prediction) | MATLAB 2017; Python `c7c88ab`, reshaped in `685da15` | high |
| I-F1 (the poll basis is always the ± coordinate directions, n_max = 1) | design question | yes, all levels | identical to MATLAB since its first commit (2017) | high |
| I-F2 (`poll_scale` cancels in the poll) | not a defect | yes | identical to MATLAB, which does it on purpose | high |
| I-F5 (level-0 poll target predicted where the GP has no data) | design question (shared with MATLAB) | yes, level 0 | shared since 2017 / `c7c88ab` | high |
| I-F8 (`argmin` picks a NaN; fallback is dead) | confirmed, inert | no | never matched, `c7c88ab` | medium-high |
| I-F9 (an unreliable GP, e.g. a zero SD, stops a good poll) | not a defect (MATLAB's rule) | rule yes, all levels; no zero SD after a good poll in 15 runs | same since 2017 / `c7c88ab` | medium |
| C-F5 (MATLAB's `pollmoved_flag` persists; PyBADS rebuilds once) | confirmed port discrepancy (unintended; contradicts a changelog claim) | yes, all levels; no effect within 200 evaluations | matched for searches after a poll move until `fef6c14` (2026-09-26) | high |
| C-F6 (`poll_training=False`: no refit recorded, unreliability flag kept) | intentional difference, missing from the sheet | no | `fef6c14` | high |
| C-F7 (`improvement_quantile` outside (0, 1) accepted) | confirmed port discrepancy | no | never matched (MATLAB check since `d04640a`, 2017) | high |
| K1 (bound method appended to `u_success`) | no longer holds | – | fixed in `0c56d86` | high |
| K4 (stale `optim_state` yval/fval/fsd after a re-estimate) | confirmed, inert (shared) | yes, levels 1/2; no consequence | Python `c7c88ab`; MATLAB 2017, where it is staler | high |
| K7 (`np.vstack(u_poll, u_poll_new)`) | confirmed, inert | no (unreachable) | `c7c88ab` | high |
| K8 (return of `period_check` discarded) | confirmed, inert | executed, no effect | `c7c88ab` | high |
| K9 (`u_base` never used) | confirmed, inert | executed, no effect | `9037851` | high |
| K10 (accelerated mesh reduction, off by one) | no longer holds; the fixed code matches `bads.m` | – | fixed in `c9a2cde` (squashed into `8aecb6a`) | high |
| K11 (Sto-BADS treats a NaN as uncertain) | no longer holds | – | fixed in `0c56d86` | high |

## 2. Per finding

### I-F3 = C-F1 = K6: p_less
**Code.**
- Python `bads.py:2273-2287`:
  - `f_pi = np.sort(f_pi)[::-1]`
  - `p_less = np.prod(1 - f_pi[0 : np.minimum(self.D + 1, len(f_pi))])`
- MATLAB `bads.m:868-869`:
  - `fpi = sort(fpi,'descend'); pless = prod(1-fpi(1:min(nvars,end)))`

**Check.** `v02_pless_unit.py`, on a gpyreg GP through `acq_fcn_lcb`:
- Output: `shapes: z (6, 1) f_mu (6, 1) fs (6, 1)` and `np.sort on (n,1) changes nothing: True`.
- The reviewers' toy (D = 3, the first point with PoI 0.69, five near 0): port `p_less=0.9999999960`, stops; MATLAB `0.3099999994`, does not stop.
- The two causes separated:
  - Sorting only: 0.31.
  - D against D+1: with 4 points left at PoI 1e-7, the port gives 0.9999996 (no stop) and MATLAB 0.9999997 (stops). The threshold is 0.99999967.

**Measured.** In `v06_instrument.py` and `v12_target_level1.py`, 34 stop decisions after a good poll with a reliable GP, at levels 0 and 1:
- No decision flipped.
- In 4 of the 6 decisions with more than D+1 points left, the largest PoI was left out of the product.

This agrees with both reports: the rule is wrong, and its effect is rare because the threshold 1 − 1e-6/D is extreme.

**Dating.** The Python lines are unchanged since `c7c88ab` apart from formatting. The MATLAB lines date from 2017 (`fbfd4a0`, `e43650b`). The two never matched.

**Disposition: fix.**
- The change: `np.sort(np.ravel(f_pi))[::-1]` and `min(D, n)`.
- It is reached at default and can move a run, so it needs a population comparison at default options with D ≥ 3 problems. The fingerprint may stay the same.
- Add a unit test of p_less.

### I-F6 = C-F2: `contraints_check` removes no logged point
**Code.**
- Python `function_logger/constraints_check.py:33-43`: `np.unique` over `vstack(u1, u2)` keeps first occurrences. The rows of `u1` come first, so none is dropped.
- MATLAB `utils/uCheck.m:17-27`: `setdiff(u1,u2,'rows')`.

**Check.** `v03_ucheck.py`, a 2-D poll set with two points already logged:
- Port keeps `[[-0.5 0] [0 -0.5] [0 0.5] [0.5 0]]`.
- The MATLAB transcription keeps `[[-0.5 0] [0 0.5]]`.

**Measured.** `v13_reevals.py`:
- One poll evaluation of a logged design point in 60 poll evaluations, in each of two level-0 Rosenbrock D=3 runs. It is the same point in both, because the Sobol design is seeded from `u0`.
- None in three level-1 runs, and none in the search.
- `v06`: 0 of 56 poll sets held a logged point.

**Dating.**
- `c7c88ab` had a correct set difference (L1 distance of the rounded rows equal to zero).
- `8e59038` (2022-06-03) replaced it with the `np.unique` form.
- `uCheck.m` is unchanged since 2017.
- `test_search.py::test_incumbent_constraint_check` (under `testing/bads/search/`) asserts the defective behavior and cites the survey.

**Disposition: fix, in slice B3's module.**
- Restore a set difference; it affects the search too.
- It changes default runs: population comparison at default.
- Update the test.

### I-F4 = C-F3: `uncertain_incumbent=False` crashes
**Code.**
- `bads.py:2696-2699` returns Python floats.
- The callers call `.item()` on them at `2245`/`2249` (poll) and `1748`/`1752` (search).
- MATLAB `bads.m:1328-1332` works.

**Check.** `v04_uncertain_incumbent_off.py`:
- `AttributeError 'float' object has no attribute 'item' | at ['optimize:1408', '_poll_step_:2245']`, both for a float target and for an `np.float64` one, and also with `uncertainty_handling=False`.
- Level 1 is unaffected.

**Dating.** `optim_state["fval"]` has been a Python float since `c7c88ab`, whose `_init_mesh_` already called `.item()`. The branch has never run.

**Disposition: fix.**
- Return arrays in the else branch, as the fallback does.
- No default run changes: fingerprint gate, plus a test.

### C-F4 = K2: the target under `hyp_best`
**Code.**
- Python `bads.py:2659-2670`: a deep copy, `set_hyperparameters(hyp_best)` (which recomputes the posterior), then `predict`.
- MATLAB, `bads.m:1301` then `gppred.m:39-47` then `mygp.m:122-123, 146-187`:
  - With `post` not empty, `post = y` is reused.
  - `Ks`, `kss` and the mean are computed under the new `hyp`.
  - So the prediction is the current `alpha`/`L`/`sW` combined with `hyp_best`'s kernel and mean.
  - I read this path myself; the reviewer's description is correct.

**Check.** `v05_target_hybrid_unit.py`:
- The emulation equals `gp.predict` under the GP's own hyperparameters (`True True`).
- With hyperparameters from 1 to 3 iterations earlier, the hybrid's mean is 1.2e3 to 2.8e7, where the port's recomputation gives about 1e-4 and the observed values are at most 5e-3.

**Measured.**
- `v06`: 0 of 21 relevant decisions had differing hyperparameters.
- `v12`, level 1, wide box: 5 of 13 did.
  - Targets: port `143.053`, MATLAB hybrid `1257608`, current GP's own prediction `143.0541`.
  - 1 decision would flip under the hybrid; 0 under the current GP.
- The hybrid is not a GP prediction under any single set of hyperparameters. I agree with the comparison reviewer that MATLAB's form looks unintended.
- This does not contradict the sheet: KD-B4-2 states the question as open.

**Dating.** `c7c88ab` predicted from the current GP. `9037851` (2022-09-22) recomputes under `hyp_best`. `685da15` added the fallback. MATLAB's lines date from 2017. The port never matched MATLAB.

**Disposition: decide the design.** The options:
- (a) Keep the recomputation and add it to KD-B4-2 as deliberate.
- (b) Predict from the current GP. This removes the `LinAlgError` path of KD-B4-2 and equals MATLAB whenever `hyp_best` equals the current hyperparameters. The measured change is tiny (143.053 against 143.054), but it moves runs: population comparison at level 1.
- (c) Emulate the hybrid. Not recommended.

### I-F10 = K5: `np.seterr`
**Code.** `bads.py:2271-2272`: `if logging.getLogger().level > logging.DEBUG: np.seterr(divide="ignore")`. It is never restored, and it tests the root logger, not the BADS logger.

**Check.** `v07_seterr.py`:
- `geterr before: {'divide': 'warn', ...}`, `geterr after: {'divide': 'ignore', ...}`.
- `user 1/0 after the run warns: False`.

**Dating.** `f9e9326` (2022-11-02). MATLAB has no counterpart.

**Disposition: fix.**
- Use `with np.errstate(divide="ignore", invalid="ignore"):` around the computation of `gamma_z`.
- Numerics unchanged: fingerprint gate.
- Extend `test_seeded_run_leaves_global_state_untouched` to cover `np.geterr()`.

### I-F7 = K3: the fallback uses the non-finite variance
**Code.**
- Python `bads.py:2672-2695`: the fallback replaces μ and σ, but the formula at `2693-2695` still uses the raw `fs2`.
- MATLAB does the same (`bads.m:1310-1311`, `1321`).

**Check.** `v09_target_fallback.py`:
- μ NaN with a finite s² gives a target of 0.49, as intended.
- s² = NaN gives `f_target=[nan]`; s² = inf gives `[-inf]`.
- The intended value, with `fsd` in the formula, is 0.48999950.
- `test_target_fallback_to_incumbent` only calls `f_target.item()`, which accepts NaN.

**Reach.**
- In MATLAB it follows a failed rebuild, where every poll prediction is NaN anyway.
- In the port a failed rebuild restores a consistent GP, and gpyreg's `predict` gives non-finite values only on overflow. I saw none.

**Dating.** MATLAB 2017; Python `c7c88ab`, reshaped in `685da15`.

**Disposition: fix (small).**
- Use `f_target_s**2` in the fallback's formula, and assert a finite target in the test.
- Fingerprint gate.

### I-F1: n_max is always 1
**Code.**
- `poll_mads_2n.py:22` is identical to `pollMADS2N.m:7`.
- With `search_size_integer = min(0, 2k − 10)` and k ≤ 0, the search mesh over the poll mesh is at most 2^-10, so n_max = 1.

**Check.** `v01_poll_basis.py`:
- `n_max over msi in [-39, 0]: {1}`.
- The port's basis equals the MATLAB transcription in entries, row maxima and |det| for n_max = 1, 3 and 8.
- Default runs on Rosenbrock D=3 and ellipsoid D=4: all polls are coordinate polls, with ptp(log `poll_scale`) up to 2.55.
- Both user documents (`README.md:118`, `docsrc/source/index.rst:46`, and MATLAB's README) describe "steps in one direction at a time".
- The docstring of `poll_mads_2n` overclaims ("dense refining directions", convergence guarantees) and cites the Sto-MADS paper for LTMADS.

**Disposition: decide the design; recommend keep and correct the docstring.**
- The port equals MATLAB and the documented behavior.
- Real LTMADS directions would depart from MATLAB and need a population comparison at default.

### I-F2: `poll_scale` cancels
**Code.** `pollMADS2N.m:23-24`, commented "Counteract subsequent multiplication by pollscale", against `bads.m:803`. The port copies both (`poll_mads_2n.py:36-37`, `bads.py:2146-2150`).

**Check.** `v01`: `vv/mesh` is a signed permutation matrix. In MATLAB, `pollscale` shapes the poll only through the non-default `pollGPS2N`, and otherwise acts through ES-ell.

**Disposition: correct the record.** `AGENTS.md` says `poll_scale` "drive[s] the poll basis". No gate.

### I-F5: level-0 target from a GP without the polled point
**Code.** `bads.py:2309-2332` adds the point to the GP only at level > 0. MATLAB does the same (`bads.m:908`). The target is then predicted at `u_poll_best` (`2242-2244`; MATLAB `UpdateTarget(upollbest,…)`).

**Check.** `v11_level0_poll.py`:
- In all 7 stop decisions after a good poll, `u_poll_best` was not in the GP.
- Rosenbrock: observed 4.155, predicted 86.41 (target 51.79).
- Ellipsoid: observed 26.35, predicted 11.44 (target 9.147).

**Disposition: decide the design; recommend keep, as MATLAB does.** Adding the point, or using the observation at level 0, changes default runs: population comparison at level 0.

### I-F8: `np.argmin` on a NaN
**Check.** `v09`: `argmin with NaN -> 1 | nanargmin -> 2 | fallback condition fires: False`.
- MATLAB's `min` skips NaN.
- The fallback (`bads.py:2257-2270`) is dead on both sides (`bads.m:857`).

**Reach.** NaN predictions need an inconsistent or overflowing GP; I saw none.

**Disposition: optional fix** (`nanargmin` with an all-NaN guard). Fingerprint gate.

### I-F9: an unreliable GP stops a good poll
**Code.** `_is_poll_stop_` (`2588-2615`) and `2279-2287` are MATLAB's `bads.m:862-895`, including "GP is unreliable, just stop polling". A zero SD gives γ = ±inf in both, hence p_less = 0 and an unreliable GP.

**Measured.**
- Zero SDs are frequent: 16 of 60, 21 of 60 and 38 of 72 poll steps at level 0 (`v11`).
- None fell after a good poll: 0 in the 6 runs of `v06`, 0 in `v11`.
- The description of `tol_poi` ("set to 0 to always complete polling") is MATLAB's text and ignores this stop.
- The `consecutive_skipping` aside (`last_skipped = -1`) matches MATLAB's `lastskipped = 0` with 1-based `iter`.

**Disposition: keep.** Optionally make the description of `tol_poi` more precise.

### C-F5: `pollmoved_flag` persists
**Code.**
- MATLAB: `pollmoved_flag` is assigned only inside the poll (`bads.m:956`, `958`), and `bads.m:1049` clears `post` at the end of every pass until the next poll. Every search of the rounds after a moving poll therefore rebuilds (without a refit; `pollscale` is recomputed only on a refit, `gpupdate.m:279-308`).
- PyBADS: `bads.py:2498` sets `reset_gp`, and `1734-1737` clears it after one rebuild.

**Dating.**
- Before `fef6c14`, `reset_gp` was never cleared. That matched MATLAB for searches after a poll move, but over-rebuilt at every poll step and after search moves.
- `fef6c14` made the rebuild once-only. The CHANGELOG `[Unreleased]` entry "Rebuilds of the local GP" says "where MATLAB BADS rebuilds it once … as MATLAB BADS does", and so does the comment at `1734-1736`. Both are true after a search move and false after a poll move.

**Measured.** `v06` and `v06b_sticky_debug.py`: 24 such searches.
- The rebuilt training set was always the same multiset, with the same hyperparameters and `sn2_mult`.
- Predictions agreed to about 1e-7 relative, i.e. rounding.
- Effect within 200 evaluations: none. A difference needs the nearest-neighbour set to change (the `n_train_max` cap or the radius), so only longer runs can show one.

**Disposition: decide the design, and correct the changelog and comment either way.**
- Either restore MATLAB's persistence after a poll move only. That needs a population comparison with long runs, and the fingerprint may move by rounding.
- Or keep rebuilding once and add a sheet entry.

### C-F6: `poll_training=False`
**Code.** Python `bads.py:2190-2200` against MATLAB `IsRefitTime` (`bads.m:1246-1252`, which sets `unrelgp_flag = 0`) and `bads.m:823`. The difference is as the reviewer describes.

**Why it is intentional.** The CHANGELOG `[Unreleased]` entry "Refits without poll training" and the code comment, both from `fef6c14`.

**Disposition: record it on the sheet.** KD-B5-2 covers only the forced refit.

### C-F7: `improvement_quantile` outside (0, 1)
**Code.** MATLAB `bads.m:1269-1271` raises an error. Python `bads.py:2018-2037` accepts any value. Both only warn for q > 0.5 at setup (`bads.py:789`, `setupoptions.m:76`).

**Check.**
- `v09`: q = 0 and q = 1 give NaN at level 0.
- `v10_quantile_run.py`: q ∈ {0, 1, 1.5} runs all 100 evaluations and ends at fval 1.212, the best initial point. At q = 0.5 the run reaches 1.96e-6 in 55 evaluations.

**Disposition: fix.**
- Refuse such values with `ValueError` when `BADS` is created, and add a changelog line.
- Fingerprint gate.

### K1: bound method appended to `u_success`
At `8aecb6a`, `bads.py:2416` reads `self.u_best.copy()`. `git log -L` shows it was changed in `0c56d86`; the item was true up to `95da7f1`. The survey's status needs correcting.

### K4: stale `optim_state` values after a re-estimate
**Code.** `bads.py:1507-1509` updates `self.yval`, `self.fval` and `self.fsd` but not `optim_state`.

**Measured.** `v06`: stale in 6, 20 and 12 of about 158 target computations per level-1 run; 0 at level 0.

**Reach.** The values are read only by the fallback at `2679-2682` (never reached) and by the copy `output_fcn` receives. MATLAB's `optimState.fval` is staler: its move after a re-estimate never updates it (`bads.m:1111-1118`).

**Disposition: keep, or sync the values in the re-estimate.** Fingerprint gate.

### K7: `np.vstack(u_poll, u_poll_new)`
**Code.** `bads.py:2180`. `v08_vstack.py` shows it would raise `TypeError vstack() takes 1 positional argument but 2 were given`.

**Reach.** `B` is filled once with 2D rows and never emptied, so the refill branch cannot run. `v06`: `poll_mads_2n` was called once in every one of 56 polls. MATLAB's `pollMADS2N` likewise returns `[]` once `B` is non-empty.

**Disposition: remove the branch or correct it.** Fingerprint gate.

### K8: return of `period_check` discarded
`bads.py:2154-2159`. It is inert: the stub returns its input, and periodic variables are refused (KD-B1-6). **Disposition:** assign the result when periodic variables are ported.

### K9: `u_base` never used
`bads.py:2442-2444`, from `9037851`, next to a commented-out condition. MATLAB has no such variable. **Disposition:** remove it; fingerprint gate.

### K10: accelerated mesh reduction
The fixed code (`bads.py:2433-2456`, `>=`, from `c9a2cde`, squashed into `8aecb6a`) matches `bads.m:976-982`:
- The condition: `iter0 + 1 > A` is equivalent to `iter0 >= A`.
- The index: MATLAB's 1-based `iter_m − A` is the 0-based `iter0 − A`.

`v11` confirms that at each poll with `iter ≥ 3` the history holds exactly `iter` entries (`all entries == iter: True`), so entry `iter − 3` is MATLAB's `iterList(iter_m − 3)`.

### K11: Sto-BADS and a NaN estimate
At `8aecb6a`, `bads.py:2065-2066` returns −1 for a non-finite estimate. `v09`: `sto rule, NaN f_new -> -1`. Changed in `0c56d86`.

## 3. Met while verifying (unverified)
- `_get_target_from_gp_` deep-copies the GP and recomputes its posterior at every search and poll step, although at default nothing reads the search's target. This costs time and adds a path that can raise `LinAlgError` (KD-B4-2). I did not measure the cost.
- The frequent zero predictive SDs at level 0 (I-F9) look like rounding of the latent variance clamped at 0. I did not establish the cause, nor whether MATLAB's `mygp` produces them as often.
