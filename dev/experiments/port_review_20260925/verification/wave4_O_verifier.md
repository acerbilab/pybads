<!-- Report of the verifier of wave 4, slice O (the third reader's report and the items kept from the reviewer, given as O-K1 onwards, briefs/wave4_kept_O.md), reading PyBADS at 0d866e8 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), with the complete history, in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave4/O_verifier/. -->

# Wave 4 verification: O

Everything below was checked at `0d866e8`, in `/home/user/pybads-review`, against MATLAB `74919c0` and gpyreg v1.3.3. Each script printed `/home/user/pybads-review/pybads/__init__.py` and `/home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. Scripts and logs are in `/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad/wave4/O_verifier/`.

## 1. Summary

| Item | Classification | Reached at default | Dating | Confidence |
|---|---|---|---|---|
| O_third F1: an uncertain Sto-BADS poll "moves" to the incumbent and is marked as moved | confirmed defect (internal track, Sto-BADS); no measured numerical effect | no: needs `stobads=True` at level 1 or 2 | the self-move has existed since `9037851` (2022-09-22); its effect (extra rebuilds) exists since `0d866e8`; no MATLAB counterpart | high |
| O-K1: with several samples, `poll_scale` is unweighted and `effective_radius` is one per sample (covered by O_third §2 Q4, "Differences") | confirmed, inert (no path gives more than one sample) | no, at any option or level | never agreed with MATLAB for more than one sample (Python `c7c88ab`, MATLAB `e43650b` 2017); one sample since `8ff10f5` (2022-11-04) | high |
| O-K2: `acq_fcn_lcb`'s summary line, its unused `n`, and `update_hedge`'s docstring | confirmed defect (documentation, internal track) | text only; the code runs at levels 0–2 | `c7c88ab` / `de1ee08` (LCB), `9037851` (hedge) | high |
| O-K3: `hedge_gamma` is not checked | confirmed shared defect (missing validation; MATLAB is identical) | no (0.125 is in range) | the two sides always agreed; neither ever checked (MATLAB 2017, Python `c7c88ab`) | high |
| O-K4: `sqrt_beta` is checked at the first search, and a callable's value is not checked | confirmed defect (internal track: the refusal is PyBADS's own, W3-10) | no (default `None`) | check written by W3-10 (`599115b`, in `0d866e8`); callable branch unchecked since `c7c88ab` | high |
| O-K5: Fig. 1 draws the poll's steps anisotropic | confirmed, inert (documentation shared with MATLAB's README) | not applicable | MATLAB's figure since 2017-05-10 (`3332e3e`); in PyBADS since `8bb3d59` (2023-02-06); the poll has cancelled `poll_scale` since MATLAB's first commit | high on the facts, medium on whether it misleads |

## 2. Per item

### F1 (O_third): an uncertain poll "moves" the incumbent to itself and is marked as moved

**Code at `0d866e8`, `pybads/bads/bads.py`:**
- `u_poll_best = self.u.copy()`, with `poll_best_improvement = 0` (2172–2176).
- `u_poll_best` changes only when `poll_improvement > poll_best_improvement` (2401–2407).
- In the Sto-BADS branch, `elif self.options["opp_stobads"] and sto_poll == 0:` calls `_update_incumbent_(u_poll_best, y_poll_best, f_poll_best, f_sd_poll_best)` and sets `is_poll_moved = True` (2455–2459). So when every uncertain point has an improvement ≤ 0, the "move" goes to the incumbent's own `u`, `yval`, `fval` and `fsd`.
- `self.poll_moved = is_poll_moved` (2551). The loop then sets `reset_gp = True` at the end of every pass (1467–1471), so every search rebuilds the local GP (1778–1800) until a poll that does not move.
- The self-move changes nothing else. `certain_good_poll` is False, so the mesh contracts exactly as it would with no move.

**MATLAB:** no counterpart (KD-S-1). MATLAB's poll moves only when `PollBestImprovement > 0` (`bads.m:951-958`), so `upollbest` is always a polled point there. The comment at 2450–2451 and the CHANGELOG ("moves to its best point") both say the poll moves to a polled point.

**Reproduction.** `v_f1_selfmove.py` wraps the functions and reads the cause of each rebuild from the caller's frame of `local_gp_fitting`. It counts a search rebuild as extra only if all of these hold: there is no refit, it is not the first search of the round, there is no `needs_rebuild`, and no search move has happened since the last rebuild. Log: `run_f1_1790511498.log`, 200 evaluations per run, seeds 3–5.
- Flat target `0.01·Σx²` plus N(0, 1), D = 3, level 1: uncertain polls 1/3/1, of which self-moves 1/1/1 (the other two in seed 4 were real moves). Each self-move caused exactly 3 extra rebuilds, that is `search_n_try − 1` (= 4 − 1).
- Same target at level 2 (`specify_target_noise`, SD 1): the same counts.
- Anisotropic quadratic, D = 2, noise SD 0.5: no poll was uncertain.

**Consequence.** `v_f1_bitwise.py` (`run_f1_bitwise2_1790511828.log`) reruns 3 of those runs with a self-moving poll left unmarked. Every evaluated point and value was identical bit for bit, in all 3:
- `level 1 seed 3: evaluated points identical: True; max |dx| of result 0, |dfval| 0; time as is 5.8s, unmarked 5.5s`
- The other two runs match; the time differences are within noise.
- The earlier `run_f1_bitwise_1790511743.log` reported "False" only because its comparison did not treat NaN as equal; it is superseded.

So the extra rebuilds, which have no refit, reproduced the same GP in these runs. They could differ only when the rebuild re-selects a different training set.

**Where I agree or disagree with the report.** I agree with the mechanism, the location, the history and the test note. `test_stobads.py:103-113` asserts `bool(poll["moves"])`, which only says that `_update_incumbent_` was called, so the test passes with a self-move.

I did not reproduce the report's counts of 3, 5 and 14 rebuilds. Its counter keeps its flag for the whole round and does not exclude rebuilds that a search move already requires (2052). From reading, it can over-count; I did not measure this. The report's "small numerical effect, cost in time" is, in my runs, no numerical effect and negligible time.

**Dating.**
- The fallback to `u_poll_best` has been there since `9037851` (2022-09-22). There the rule was `opp_stobads and sto_success > -1`, with the last point's outcome. `0c56d86` (W0-10) switched to the best outcome and kept the fallback.
- The effect of the false flag depends on the reset rule, which changed twice:
  - `9037851` to `e004c79`: `reset_gp` was never cleared by a rebuild, so every later search and poll step of the next round rebuilt.
  - `fef6c14` and `8aecb6a`: a one-shot reset, which the round's first search, a rebuild anyway, cleared. The false flag had no effect.
  - Since `0d866e8` (wave 3, "Rebuilds of the local GP"): extra rebuilds again.

**Also, by reading and not run:** with `improvement_quantile < 0.5`, `poll_improvement` is the quantile. An uncertain point with μ > 0 then still leaves `u_poll_best` at the incumbent.

**Recommended disposition: decide the design, then fix.** The options:
- (a) An uncertain poll moves only if some uncertain point has a positive improvement. Otherwise it neither moves nor is marked. The change is `and poll_best_improvement > 0`, or a test that `u_poll_best` is not the incumbent, at 2455.
- (b) It moves to the best uncertain point even when μ ≤ 0, as the search does (2010, 2046).

Either fix touches `bads.py:2449-2461` and `test_stobads.py` (assert that the incumbent changed). Neither changes a run at default, so the fingerprint of a default run is the gate. Under a Sto-BADS configuration, (a) left every point identical in my 3 runs, so a fingerprint under `stobads=True` should show whether it moves anything. (b) changes Sto-BADS runs and needs a population comparison with `stobads=True` on noisy targets.

### O-K1: several hyperparameter samples in the geometry (covered by O_third §2 Q4; verified once, here)

**Code.**
- `gaussian_process_train.py:548-556`: `ll - np.mean(ll)` followed by `np.exp(np.sum(ll, axis=0))`, an unweighted sum over samples.
- `gaussian_process_train.py:578-593`: `effective_radius` is computed per sample, as an array.
- MATLAB `gpupdate.m:294-299` uses `exp(sum(hypweight .* (ll - mean(ll(:))), 2))`, and `313-317` takes α as the `hypweight`-weighted mean, then computes a scalar radius.
- `len_scale` (534–545) uses weights of 1/N, which equals MATLAB's for equal `hypweight`.

**Check.** `v_ok1_samples.py`, part B (`run_ok1B_1790511932.log`), forces two samples through `_robust_gp_fit_` and compares with a transcription of MATLAB:
- `poll_scale PyBADS: [1.880 0.229 2.321]`, `MATLAB: [1.371 0.479 1.523]`. PyBADS's values are exactly MATLAB's squared.
- `effective_radius PyBADS: [1.0577 1.0206]`, `MATLAB: 1.0304`.
- `len_scale` is equal on both sides.

**Reachability.** `v_ok1_samples.py`, part A (`run_ok1_1790511910.log`), covers level 0 at default, level 1 at default, and level 1 with `double_refit=True` and `gp_samples=5`. Every `local_gp_fitting` call (28, 147 and 152) left one sample, and `effective_radius` had size 1.
- With `double_refit`, `_robust_gp_fit_` receives 2 rows: the second fit's starting points (`gaussian_process_train.py:452-466`).
- It always returns 1 row: gpyreg's `fit` with `n_samples = 0` returns `hyp_start` reshaped to one row (`gaussian_process.py:1994-1999`), and the fallback path returns `starts[[argmin]]`.
- `gp_s_N` is hard-coded to 0 at both call sites of `_get_gp_training_options` (402–478, 1087), and `gp_samples` is unread (KD-B5-4).

No option or path reaches the item.

**Dating.** Python since `c7c88ab` (2022-06-02); MATLAB's weighted form since `e43650b` (2017-03-30). The two never agreed for N > 1 and agree for N = 1. One sample has been guaranteed since `8ff10f5` (2022-11-04). Before that, `_get_numb_gp_samples`, a PyVBMC leftover, set the count; I did not check whether it was ever above 0.

**Recommended disposition: fix or document.** The fix is weights of 1/N in `poll_scale` and the weighted mean of α before the radius, in `gaussian_process_train.py:548-593`. At minimum, a comment that the code assumes one sample. A default run does not change; gate: fingerprint.

### O-K2: `acq_fcn_lcb`'s summary and unused `n`; `update_hedge`'s docstring (not covered by the report)

**Code.**
- `acq_fcn_lcb.py:8`: "It retrieves the point at the lower confidence bound…". The function returns the LCB values, the mean and the SD at every row of `xi`; it retrieves no point. The page is published through `docsrc/source/api/functions/acquisition_functions.rst` (automodule).
- `acq_fcn_lcb.py:41`: `n = xi.shape[0]` is never read. MATLAB's `n` is used in its `catch` branch (`acqLCB.m:181`), which the port dropped.
- `search_hedge.py:123`: "Update the probability of improvement…". The method updates only the gains `g`, each with the expected improvement, weighted by 1/`phat` and divided by the mesh size (172–175). The probability of improvement is an intermediate.

**Check.** `v_ok2_unused.py`: `assigned, never read: ['n']`; `update_hedge changed attributes: ['g'] g -> [ 5. 16.]`, which equals the hand-computed values.

**Dating.** The LCB summary has been there since `c7c88ab` and was reworded in `de1ee08` (2023-01-04). `n` has been there since `c7c88ab`. The `update_hedge` docstring has been there since `9037851`.

**Recommended disposition: fix** (the docstrings, and remove `n`). Nothing moves; the fingerprint is optional.

### O-K3: `hedge_gamma` is not checked (not covered by the report)

**Code.** `search_hedge.py:52` reads the value, and `65-69` computes `p = softmax(β·g)·(1 − nγ) + γ`. No check exists in `bads.py:795-845` or elsewhere. MATLAB `searchHedge.m:36,45-48` is the same formula, and `setupoptions.m` only evaluates `HedgeGamma`. Both sides accept any value.

**Check.** `v_ok3_hedge_gamma.py` (`run_ok3_1790512014.log`) calls `ESSearchHedge.__call__` with the ES searches stubbed, against a transcription of `searchHedge.m`. The probabilities and the choices over 4000 draws were identical on both sides for every γ. With g = [10, 0] and β = 1:

| γ | p | chosen (of 4000) |
|---|---|---|
| 0.125 | [0.875, 0.125] | [3490, 510] |
| 0.75 | [0.25, 0.75] (inverted) | favours the lower gain |
| 1.25 | [−0.25, 1.25] | [0, 4000], always the lower-gain search |
| −0.1 | [1.1, −0.1] | [4000, 0], never explores |

A strategy with negative p is never chosen, because `rand < cumsum` then never picks it, so `phat` stays positive.

`v_ok3_run.py` (`run_ok3b_1790512036.log`) shows that BADS is created with such values and runs without any warning:
- `hedge_gamma=1.25: … ES-wcm chosen 11, ES-ell 14; searches with a negative probability 6`, with the first searches choosing the strategy whose gain is 0 over the one whose gain is 10.
- `hedge_gamma=-0.5: … 8` searches with a negative probability.

The ledger's thresholds hold: the probabilities invert above 1/n and turn negative above 1/(n − 1). The option's description in `advanced_bads_options.ini:265`, "Minimum probability of each search…", holds only for 0 ≤ γ ≤ 1/n.

**Dating.** Identical on both sides throughout: MATLAB since `6c93629` (2017), Python since `c7c88ab`. Neither side ever validated the value.

**Recommended disposition: fix.** Refuse a `hedge_gamma` outside [0, 1/n], with n = `len(search_method)`, when `BADS` is created, beside the checks at `bads.py:812-845`. Record the item as shared with MATLAB. A default run does not change; gate: fingerprint.

### O-K4: `sqrt_beta` is checked late, and a callable's value is not checked (not covered by the report; its note on the set of refused values is adjacent)

**Code.** The check is in `acq_fcn_lcb.py:51-64`. Its only caller with a user `sqrt_beta` is `es_search.py:154-157`, at the first search. `bads.py` does not look at `search_acq_fcn`. The callable branch (49–50) uses the returned value unchecked.

**Check.**
- `v_ok4_sqrt_beta.py` (`run_ok4_1790512065.log`, D = 3, deterministic): for `-1.0`, `0.0` and `'ucb'`, `BADS() created, 0 evaluations so far`, then `optimize() raised ValueError after 12 evaluations`.
- `v_ok4_where.py` locates the error: the steps were `[('poll', 6), ('search', 12)]`, so the refusal came after the initial design and the first poll.
- Callables:
  - returning `-1.0`: runs to the end silently;
  - returning `nan`: runs silently to fval 0.142, where the default schedule reaches 4e-6;
  - returning an array of 2: `IndexError … index 4065 is out of bounds`;
  - returning `'x'`: `UFuncTypeError`.

**MATLAB.** `acqLCB.m:16-21` also checks at each call, but accepts any numeric scalar. Its callers catch the error (`searchES.m:136-150`, `bads.m:566-585`), so MATLAB never stops; it degrades silently. It does not check a callable's value either. This touches KD-B5-3's note that these `try`/`catch` sites have no recorded decision.

**Dating.** The check was written by W3-10 (`599115b`, merged in `0d866e8`, 2026-09-27). The callable branch has been unchecked since `c7c88ab`.

**Recommended disposition: fix.**
- Apply the same check to `search_acq_fcn[1]` when `BADS` is created, beside `bads.py:812-845`.
- Check a callable's return value (a positive finite real scalar) at each call in `acq_fcn_lcb`.
- A default run does not change; gate: fingerprint.

### O-K5: Fig. 1 (not covered by the report)

**What the figure shows.** `docsrc/source/_static/bads-cartoon.png` is byte-identical to MATLAB's `docs/bads-cartoon.png` (same md5). `v_ok5_cartoon.py` measures the left panel: it is square, 720 × 720 px, with no ticks. The poll cross is centred at (238, 595); its horizontal arms span 115 px each way and its vertical arms 64/63 px, markers included. That puts the marker centres at about ±108 and ±55 px, so the x1 steps are about twice the x2 steps.

**What the code does.** `poll_mads_2n.py:75` divides by `poll_scale`, and `bads.py:2211-2216` multiplies it back. MATLAB `pollMADS2N.m:24` and `bads.m:803` do the same. At default, n_max = 1, so every poll step is ±Δ along one coordinate of `u` space, whatever the GP. In the original coordinates, the step along x_d is Δ·(pub_d − plb_d)/2 for a linear variable, so the steps are anisotropic only through the plausible box (or the log transform), never through `poll_scale`.

The figure does not show the plausible box. It matches the code only if x1's plausible range was twice x2's. The README and `index.rst` text ("steps in one direction at a time") is correct.

**Dating.** The cancellation has been in MATLAB since its first commit (`6c93629`, 2017-03-14), before the figure was committed (2017-05-10). The figure has been in PyBADS since `8bb3d59` (2023-02-06).

**Recommended disposition: keep and document.** Add a caption clause saying that the poll's steps are equal in the normalized coordinates and scale with the plausible box in the original ones, or redraw the figure. It is documentation only, so no gate. The same applies to MATLAB's README.

## 3. Found while verifying (marked where unverified)

- **The report's three differences "recorded outside the sheet"** are absent from `known_differences.md` (a grep for `ncovlen`, one-dimensional, `len_scale`, `sqrt_beta`, W3-10 and random index found nothing).
  - D = 1 `lenscale`: checked by reading.
  - The `sqrt_beta` refusal of 0 and negative values: checked by my O-K4 run.
  - The random index when every poll LCB value is NaN, where MATLAB's `min` gives index 1 (`bads.m:853-857`): checked by reading.

  They belong on the sheet.
- **Unverified:** `hedge_beta` and `hedge_decay` are unchecked in the same way (`search_hedge.py:53-54`). For example, a `hedge_decay` above 1 would make the gains grow without bound.
- **Unverified, context for F1's design question:** the report says a Sto-BADS search moves on every uncertain outcome, μ < 0 included, and records the move as "incremental", which multiplies `search_factor` by `search_scale_incremental` = 2 (`bads.py:2010`, `2046`, `_update_search_stats_`). I confirmed this by reading only; I did not measure it.
