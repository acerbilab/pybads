<!-- Report of the verifier of wave 3, slice B3 (the two B3 reports and the items kept from the reviewers, given as B3-K1 onwards, briefs/wave3_kept_B3.md), reading PyBADS at 8aecb6a in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), with the complete history, in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave3/B3_verifier/. -->

# Wave 3 verification: B3

What I read: PyBADS at `8aecb6a` (`/home/user/pybads-review`), MATLAB BADS at `74919c0`, and gpyreg v1.3.3.

Where things are: my scripts and their outputs are in `/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/review/B3_verifier/`. Every script printed `pybads.__file__ = /home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__ = /home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. "Int" means the internal report and "Cmp" the comparison report.

## 1. Summary

| Finding | Classification | Reached at default | Dating | Confidence |
|---|---|---|---|---|
| Int F1 = Cmp F1 = **B3-K3**: `contraints_check` keeps points already evaluated | confirmed port discrepancy | yes. Level 0: the search, when the optimum is on a bound or the grid is coarse. Levels 1-2: the poll | matched MATLAB only in `c7c88ab` (2022-06-02); lost in `8e59038` (2022-06-03); MATLAB unchanged since 2017 | high |
| **B3-K4** (and the "reorders" half of Int F1): the output is in `np.unique` order | not a defect: this is MATLAB's `setdiff` order | yes, all levels | the order has matched MATLAB since `8e59038` | high |
| Int F2 = Cmp F4: `ucov` ignores its weights | confirmed shared defect | yes, all levels (every ES-wcm search) | always agreed with MATLAB (`ucov.m` 2017; `c7c88ab`, rewritten in `e404fcc`) | high that the weights do nothing; medium that weighting was intended |
| Int F3 = Cmp F3: ES-wcm uses ⌊μ⌋+1 best points | confirmed port discrepancy | yes, all levels | never agreed (`c7c88ab`; `searchES.m` unchanged since 2017) | high |
| Int F4 = Cmp F2: selection mask shifted by one rank | confirmed port discrepancy | yes, all levels, every search | never agreed (`c7c88ab`; `ESupdate.m` unchanged since 2017) | high |
| Int F5 = Cmp F5 = **B3-K1**: the hedge reward's normal density is mis-parenthesized | confirmed port discrepancy | yes, levels 1 and 2 (not 0) | never agreed (`c7c88ab`; MATLAB's line 64 is correct since `5bb226d`, 2017) | high |
| Int F6 = Cmp F8: `hedge_gamma = 0` crashes | confirmed shared defect (both crash, at different places) | no | Python since `c7c88ab`; MATLAB since 2017 | high |
| Int F7 = Cmp F7: the fraction of new points is miscounted | confirmed port discrepancy | no (needs `n_search_iter` ≥ 4 to matter) | never agreed (`c7c88ab`) | high |
| Int F8 = Cmp F9 (+ **B3-K11**): a plain numeric `sqrt_beta` is refused; the docstring calls the SD a variance | confirmed port discrepancy (K11: confirmed defect, inert) | no | never agreed (`c7c88ab`); docstring from `de1ee08` | high |
| Int F9 = Cmp F6 (+ **B3-K10**): an emptied later ES generation discards every candidate, with a misleading warning | confirmed port discrepancy | only when `non_box_cons` empties a later generation (any level) | never agreed (`c7c88ab`); an `IndexError` before `0c56d86`, an empty set since | high |
| Cmp F10: empty search set: forced failure, no hedge update | design question (partly an intentional difference missing from the sheet) | the hedge part yes (all levels) when the set is empty after an earlier search; the move part no | forced failure `0c56d86`; hedge guard `c7c88ab` | medium-high |
| **B3-K2**: after a failed rebuild, the search ranks by the restored GP's LCB | design question | only after a failed rebuild (rare), any level | since `685da15`; before, the run crashed; never agreed | high |
| **B3-K5**: `ESSearchCMA` is broken | confirmed, inert (unreachable) | no | `c7c88ab` | high |
| **B3-K6**: an empty search set crashes the run | no longer holds (`0c56d86`); the fix matches MATLAB's status at default but not its hedge update (see Cmp F10) | yes | fixed 2026-09-26 | high |
| **B3-K7**: `search_factor_min` unread | no longer holds (`3272bdd`, in `8aecb6a`); matches `UpdateSearch` | yes | fixed 2026-09-26 | high |
| **B3-K8**: `force_to_grid` rounds halves to even | confirmed port discrepancy | yes, all levels, but only on exact halves (in practice the start point) | never agreed (`c7c88ab`) | high |
| **B3-K9**: unstable `np.argsort` at `es_search.py:190` and `:246` | confirmed port discrepancy | yes, all levels, every search | never agreed (`c7c88ab`) | high |
| **B3-K12**: `search_n_try` is a float | no longer holds (`2d4304c`, in `8aecb6a`) | yes | fixed 2026-09-26 | high |
| **B3-K13**: a deterministic `fsd` is the int 0 | confirmed, inert (the empty branch's `f_sd_search = 0` reaches nothing) | yes | `c7c88ab` | high |

## 2. Per finding

### F1: evaluated points are not removed (Int F1, Cmp F1, B3-K3)

**Code.**
- Python: `pybads/function_logger/constraints_check.py:33-43`. It stacks the candidates' bins over the logged points' bins, calls `np.unique(return_index)` and keeps indices `< len(u1)`. A candidate's first occurrence is always among the candidates, so every distinct candidate bin survives.
- MATLAB: `utils/uCheck.m:25`, `[~,idx] = setdiff(u1,u2,'rows')`, which drops the evaluated bins.

**Check: `v_ucheck.py`, which compares the function with a transcription of `uCheck.m`.** The input has 8 rows, 3 of them in evaluated bins:
- PyBADS keeps 6 rows, and `PyBADS rows whose bin was evaluated: 3`.
- The transcription keeps `[[0.25,-0.25],[0.75,0],[1,0]]`, and `MATLAB rows whose bin was evaluated: 0`.

**Check: repeats in seeded runs, by stage (`v_repeats*.py`).**

| Configuration | Repeats with the port | Repeats with MATLAB's removal patched in |
|---|---|---|
| Optimum on a bound, Σ(x+1)² on [0,5]^D, D = 1 | search 9/16 (seeds 0 and 1) | 0/16; same `fval` and same count (25) |
| Same, D = 2 | search 20/29 and 21/29 | 0/29; same `fval` and same count (49) |
| Interior sphere, D = 1, x0 = 3, bounds ±20 | search 1-2 of 16 (4 seeds; reproduces Cmp) | not run |
| Interior sphere, D = 3 | 0 | not run |
| Level 2, heteroscedastic ellipsoid D = 3, 4 seeds | search 0; **poll 13/77 and 6/77** in two of the four runs | not run |
| Level 1, noisy sphere D = 3, 2 seeds | search 0; **poll 6/61** in one of the two runs | not run |

So in noisy runs the repeats come through the poll (`bads.py:2166`, `proj=False`), which neither reviewer found:
- Int said none came from the poll; that holds for its deterministic runs.
- Cmp saw one poll repeat in three runs.

At levels 0 and 1 a repeat becomes a new row of the function log, so it is a duplicate GP training input (`function_logger.py:405-447`). At level 2 it is merged into its row by precision weighting.

**Dating.**
- `c7c88ab` removed the matches with an L1 test, which agrees with MATLAB.
- `8e59038` replaced that with the `np.unique` code (and an early `return` that skipped `non_box_cons`, which `f9e9326` fixed).
- `uCheck.m` is unchanged since `a8817f9` (2017).

**Where I agree or disagree.** I agree with both reports on the discrepancy. The poll's reach at levels 1 and 2 is wider than either says. I did not re-check the survey's analysis of the crashes, the `ellipsoid_D10` repeat, or the TODO's 0.46→0.33 median error, which it measured together with other changes.

**Tests.** `test_incumbent_constraint_check` asserts the current behavior.

**Disposition: fix.**
- The fix goes in `constraints_check.py`: drop the candidates whose bin is in `u2`, as `setdiff` does. The test needs updating.
- It changes runs at default: the bound-optimum searches, the noisy polls, and the ES's random stream whenever a generation loses an evaluated bin, since `ll` then shrinks.
- Gate: a population comparison with configurations that reach it: an optimum on a bound (level 0), and level 1 and level 2 targets.

### B3-K4: the order of the output

True as stated: the output is in `np.unique` order, and the sorting variant at line 41 is commented out. But MATLAB's `setdiff` also returns the rows sorted.
- `v_ucheck.py`: `order: PyBADS == MATLAB on 200/200 random sets; input-order variant == MATLAB on 0/200`.
- The initial design goes through the same check on both sides (`evalinitmesh.m:113`, `bads.py:1091`).
- The poll's `B` is never indexed alongside `u_poll`.

One residual difference: within one bin of 2^-20, the port keeps the first row in input order, while MATLAB keeps the lexicographically smallest exact row (`same-bin representative: PyBADS [[0.10000017881393433, 0.2]] MATLAB [[0.10000005960464478, 0.2]]`). This matters only once the search grid is finer than the bin (mesh_size ≤ 2^-6), and it moves the point by less than 2^-20 in u space. It is negligible.

The Int F1 statement "it should keep the input order" is wrong against MATLAB.

**Disposition: correct the record.** No code change. If anything, the comment at line 29 ("preserve the initial order") could say that the final order is the sorted one, as in MATLAB.

### F2: `ucov` is unweighted (Int F2, Cmp F4)

**Code.** Python `es_search.py:324-329`; MATLAB `utils/ucov.m:19`. Both compute Σ_k w_k·SᵀS = SᵀS, since the weights sum to 1.

**Check: `v_ucov.py`.** `ucov(PyBADS) == S^T S: True`, `ucov(MATLAB transcription) == S^T S: True`, `weighted sum_i w_i s_i s_i^T == S^T S: False`. Over 20 captured GP states (Rosenbrock, D = 3), applying the weights would change the normalized search covariance by 13-46% (median 33%).

**Dating.** The port transcribed MATLAB literally in `c7c88ab`; `e404fcc` (2023) changed the broadcast and kept it unweighted. The two have always agreed.

**Where I agree or disagree.** I agree with Cmp that this is shared. Int's "port discrepancy (the error is in NumPy broadcasting)" is wrong: MATLAB has the same arithmetic.

**Disposition: decide the design.**
- Option (a): keep it and document it as a shared defect, since MATLAB's published behavior is unweighted.
- Option (b): weight both sides. That changes every ES-wcm search at default and needs a population comparison; it also requires F3's fix, because a weighted form raises on 9 rows against 8 weights.

### F3: ⌊μ⌋+1 rows (Int F3, Cmp F3)

**Code.** Python `es_search.py:247`, `y_idx[0 : floor(mu+1)]`; MATLAB `searchES.m:62`, `index(1:floor(mu))`.

**Check: `v_ucov.py`.** The rows taken are `(n_train, floor(mu), floor(mu+1))` = `(11, 5, 6), (12, 6, 7)...`. The normalized search covariance differs from MATLAB's by `min 0.021 median 0.109 max 0.395` (relative Frobenius), and `port == transcription with floor(mu)+1 rows: True`.

**Dating.** `c7c88ab`; never agreed.

**Disposition: fix.** Take `floor(mu)` rows. This changes runs at default, so it needs a population comparison; the default benchmark reaches it.

### F4: the selection mask (Int F4, Cmp F2)

**Code.** Python `es_search.py:70-74` uses MATLAB's 1-based `cw` as 0-based indices; MATLAB `ESupdate.m:19-21`.

**Check: `v_mask.py`, against a transcription of `ESupdate.m`.**
- For every (μ, λ) tried, `port == [0] + MATLAB(1-based)` holds.
- At μ = λ = 2048, the offspring of parents 0..5 are `MATLAB [17, 12, 10, 9, 8, 7] port [1, 17, 12, 10, 9, 8]`.
- The port's sum equals MATLAB's 1-based sum, so `test_search_selection_mask`'s golden 885072 locks the shift in. A correct 0-based mask sums to 883024.
- Paired ES runs on 20 captured states × 5 seeds: `ES-wcm: fixed mask lower best LCB in 0.79, port lower in 0.17`; `ES-ell: 0.73 vs 0.19`. The median gain is small (-1.7 on a median |LCB| of 29.8).

**Dating.** `c7c88ab`; never agreed.

**Disposition: fix.** Use `np.repeat(np.arange(len(w)), w)`, or `M - 1`, and update the test. This changes every search at default, so it needs a population comparison.

### F5 / B3-K1: the hedge reward

**Code.** Python `search_hedge.py:152-155`, `exp(-0.5*g**2/sqrt(2π))`; MATLAB `acqPortfolio.m:64`, `exp(-0.5*(gammaz.^2))/sqrt(2*pi)`.

**Check: `v_reward.py`.**
- The port's reward is `424.01` times MATLAB's at γ = -3, `2.51` at γ = 0 and `1.20` at γ = +2.
- In a level-1 run, all 60 hedge updates went through the branch with SD > 0, and the port/MATLAB reward ratio had quartiles `[2.36, 2.59, 2.94]`.
- Level 0 takes the `fs == 0` branch, where the two agree.

**Dating.** The port's formula has been wrong since `c7c88ab`. MATLAB's line has been correct since `5bb226d` (2017-03-16); `d4fead5` removed only comments. The survey's "MATLAB side not checked" is now checked.

**Sheet.** KD-B3-3 calls this update "ported"; it is ported with the wrong formula.

**Disposition: fix** (the parenthesis). It changes only noisy runs, so it needs a population comparison with level 1 and level 2 configurations. The fingerprint changes if its runs include noisy targets.

### `hedge_gamma = 0` (Int F6, Cmp F8)

**Check: `v_gamma0.py`.** `update_hedge #1: chosen=0 prob=[0.999955, 4.5e-05] len(u_search)=3`, then `ValueError The dimension of input data 2 doesn't match GP's input dimension 3`. It crashes at the **first** search; Cmp's "second search" is wrong.

**The cause** is `search_hedge.py:127`, which slices coordinates.

**MATLAB** (`acqPortfolio.m:40`) takes `u(min(iHedge,end),:)`, which is the same single row for every strategy, and then fails at line 47 on `gpstructnew`, which is undefined there (the assignment was commented out before `d4fead5`). So Int's "scored at the chosen point" is MATLAB's intended behavior, not a defect.

**Disposition: fix** (`np.atleast_2d(u_search)[min(i, n-1)]`). This changes nothing at default, so the gate is an unchanged fingerprint plus one γ = 0 run.

### F7: the fraction of new points (Int F7, Cmp F7)

**Code.** Python `es_search.py:177-192`; MATLAB `searchES.m:170-193`.

**Check: `v_frac.py`, same z values under both counting rules.**
- MATLAB: `[nan, 0.56, 0.427, 0.392, 0.371]`.
- Port: `[0.0, 0.56, 0.769, 0.868, 0.934]`.
- True share of new points: `[1.0, 0.56, 0.427, ...]`.

The fraction enters the scale only for 1 < i < `n_search_iter` (1-based), so never at the default of 2. At `n_search_iter = 3` the only difference is the negligible off-by-one; from 4 on the count is badly off, because the pool is untrimmed.

Int's (d), a NaN scale from 0/0, is unreachable today: an emptied generation breaks the loop first (F9). It would become reachable after F9's fix, which must therefore guard `ntest == 0`. MATLAB itself gets 0/0 there.

**Disposition: fix.** The fingerprint stays unchanged; test with a configuration at `n_search_iter` ≥ 4.

### F8 / F9 / B3-K11: `acq_fcn_lcb` (Int F8, Cmp F9)

**Check: `v_lcb.py`.**
- `2.0` and `2` raise `AttributeError: ... has no attribute 'size'`.
- `np.float64(2.0)`, `array(2.)` and `array([2.])` work, so Int's "an array raises" holds only for arrays of more than one element.
- `inf` raises `ValueError`, where MATLAB accepts any numeric scalar.
- `'my_schedule'` raises `TypeError`, where MATLAB `feval`s the name.
- `third output == sqrt(latent variance): True`, but the docstring says `'f_s: GP variance', 'GP variance at z.'`.

**Code.** Python `acq_fcn_lcb.py:42` (and docstring `:27-28`); MATLAB `acqLCB.m:16-18`.

**Dating.** The check is from `c7c88ab`; the docstring from `de1ee08`.

**Disposition: fix.** Use `np.size`/`np.ndim` on `np.asarray(sqrt_beta)`, decide whether to accept non-finite values and names, and correct the docstring. The fingerprint stays unchanged.

### F9 / B3-K10: an emptied later generation (Int F9, Cmp F6)

**Code.** Python `es_search.py:170-175` sets `z_candidates = rng.random(0)`, discarding the earlier values; lines 196-197 and 219-221 then return an empty set. MATLAB `searchES.m:168-182` keeps `zold` and returns the best earlier candidate.

**Check: `v_emptygen.py`.** `generation sizes seen by LCB: [2048, 0]`; `PyBADS ES returns shape (0, 3)`, where MATLAB returns the best of generation 1. The log shows `'...Something went wrong with the acquisition function, random search is performed'`, although no random search runs.

**B3-K10: `v_thinband.py`.** The band is |x1−x2| ≤ 0.01, at D = 3, deterministic, seed 0: `ES warnings=3 ES calls returning an empty set=3`, and every emptied generation is the second, `[(1,), (1,), (1,)]`. The warning therefore marks this path, a generation emptied by the constraint, and not a failed acquisition; the GP having few points is incidental. The second generation's size equals the number of first-generation survivors (e.g. `(2048, 12), (12, 0)`), in MATLAB as well, which makes a thin band empty it easily.

**Dating.** The reset dates from `c7c88ab`; `0c56d86` turned the crash into an empty set.

**Disposition: fix.** Skip the empty generation, keep `z_candidates`, guard `frac`, and reword or remove the warning. Nothing changes without `non_box_cons`, so the fingerprint stays unchanged; test with a thin-band configuration. I saw no emptied generation with MATLAB-like removal and no constraint (`v_f1f9.py`, 8 bound-optimum runs), but F1's fix makes one possible.

### Cmp F10 and B3-K6: an empty search set

**B3-K6's two survey rows no longer hold.** `0c56d86` changed the ES to return `us, z` when empty (`es_search.py:219-221`) and the step to set `u_search = None` (`bads.py:1904`), with a forced failure at `:1956-1958`.

**Check: `v_empty_search.py`**, with the ES check emptied at searches 2 and 3.
- The run completes.
- The statuses are `['failure', 'failure']`.
- The search factor goes `1.4142->1.0000` and `1.0000->0.7071`.
- `hedge g unchanged=True`.
- At level 1 with `improvement_quantile=0.75`, the improvement computed is `0.2411`, yet the search is still a failure.

**Against MATLAB** (`bads.m:667-725`, with `EvalImprovement` at `1257-1279`):
- (a) At q ≤ 0.5, or at level 0, MATLAB also fails. This part matches.
- (b) At q > 0.5 with fsd > 0, MATLAB counts an incremental search and moves the incumbent to the *previous* search's `usearch`, taking `fval` and SD 0. This is a MATLAB defect that the port avoids, off default.
- (c) MATLAB decays all hedge gains on an empty set (`~isempty(usearch)` with a stale `usearch`, er = 0). The port skips the update. This is reached at default whenever the set is empty after an earlier search.
- (d) If the run's first search is empty, MATLAB errors on the undefined `usearch` at line 722.

So the fix does not do what MATLAB does "on every path" (the wording of the commit and of the comment at `bads.py:1956`): it matches MATLAB's status at default and diverges on (b) to (d).

**Disposition: decide the design and correct the record.**
- Whether to decay the gains as MATLAB does. That changes runs only when a set is empty: fingerprint, plus a `non_box_cons` configuration.
- Record (b) and (d) on the sheet as intentional.
- Mark the survey rows fixed by `0c56d86`.

### B3-K2: the search after a failed rebuild

**Check: `v_failed_rebuild.py`**, a real LinAlgError injected into `local_gp_fitting`'s posterior step and into its retry, at search 5.
- `exit flag -2 markers {'needs_rebuild': True, 'needs_refit': True}`.
- `GP handed to the ES is the restored one (marked): True`, and its predictions are finite.
- The port evaluates the restored GP's LCB minimizer, offset `[-142, 38, 93]` grid units, LCB `0.0087`.
- MATLAB's choice would be the lexicographically first candidate, offset `[-1891, 1696, 291]`, LCB `1.5514` under the same GP.

**Why MATLAB picks that candidate.** Its `gpupdate.m:340-354` sets `post = []`; `gppred.m:39-60` recomputes, fails in the same way, and leaves `hypw` NaN; `acqLCB.m:35` then sums over no samples, so z ≡ 0. Its stable sort then keeps `uCheck`'s lexicographic order.

**What the sheet covers.** KD-B5-2 settles the restore itself. It does not say how the search ranks afterwards, while the poll treats such a GP as unreliable (`bads.py:2236-2239`).

**Dating.** `685da15`; before it, a failed rebuild crashed the run.

**Disposition: decide the design.**
- Keep the ranking by the restored GP, which I recommend: MATLAB's choice is an arbitrary far point.
- Or mimic MATLAB.
- Or pick a random candidate.

Document whichever is chosen under KD-B5-2. Keeping it changes nothing.

### B3-K5: `ESSearchCMA`

**Check: `v_small.py`.** Calling it raises `TypeError: slice indices must be integers ... (at line 261: U_worst = U[y_idx[-1 : -1 : (len(y_idx) - np.floor(mu) + 1)]])`. That slice is also empty and in the wrong order, and line 262 calls `ucov` with 4 of its 7 arguments. The hedge given `'ES-cma+'` raises `not implemented yet`.

The class is unreachable, as KD-B3-1 says.

**Disposition: keep and document, or delete the class.** No run changes.

### B3-K7: `search_factor_min`

Fixed in `3272bdd`, squashed into `8aecb6a`. `bads.py:2770-2776` against `UpdateSearch` at `bads.m:1342-1375`: in `v_small.py`, a transcription driven by 400 random statuses gave `0/400 mismatches` with `adaptive_incumbent_shift` both on and off, equal statistics lists, and a minimum factor of 0.5000. The default is 0.5 on both sides (`bads.m:238`).

**Disposition: correct the record** (fixed, matches MATLAB).

### B3-K8: rounding halves

**Code.** Python `grid_functions.py:12`, `np.round`; MATLAB `force2grid.m:5`, `round`, which rounds halves away from zero.

**Check: `v_grid.py`.**
- Halves at `[0.5, 1.5, 2.5, -0.5, -1.5]` give `port [0, 2, 2, -0, -2]` and `MATLAB [1, 2, 3, -1, -2]`.
- x0 = [1, 3] in a plausible box of [-2048, 2048] starts the port at x = `[0.0, 4.0]`; MATLAB would start at u0/tol `[1, 2]`, i.e. x = [2, 4].
- `exact halves among 96000 Sobol design coordinates: 0`.

ES draws are continuous, so exact halves there have probability zero. The poll is not gridded at default (`force_poll_mesh = False`). The search bounds do not depend on the rounding mode, because of the correction step at `bads.py:646-661` and `2705-2715`.

**Dating.** `c7c88ab`; never agreed.

**Disposition: fix.** Use `sign(q)·floor(|q|+0.5)`. The fingerprint should stay unchanged unless one of its start points sits on a half.

### B3-K9: unstable `argsort`

**Check: `v_argsort*.py`**, with a stable sort swapped in through a proxy.
- Line 190 gave a different permutation in 31-35 of 58 calls on the sphere and 18-26 of 80-88 on Rosenbrock.
- Line 246 gave a different permutation in 9/9 and 10/12 calls on a quantized sphere.
- Seeded runs differed from the stable-sort runs in **5 of 6** cases.
- At line 190, every differing call held ties between distinct candidates. These are far candidates whose predictions equal the prior exactly, and they rank high (`median first-tie rank fraction 0.006`).
- The returned best point never changed (0 of 57 differing calls). The selected set changed in 11 of 57 calls and the parents' order in all 57, and that changes the second generation.

MATLAB's `sort` is stable. With a stable sort, the port's order among ties would match MATLAB's at both lines: old candidates before new, and `uCheck`'s order within each.

**Disposition: fix** (`kind="stable"` at both lines). This changes runs at default, so it needs a population comparison.

### B3-K12 and B3-K13

**K12.** `search_n_try` is an int at D = 1, 2, 3, 6, 7 and 20, with values equal to MATLAB's `max(nvars, floor(3+nvars/2))` (`v_small.py`). It was a float at `8aecb6a^` and was fixed in `2d4304c`. Disposition: correct the record.

**K13.** The empty branch at `bads.py:1907` still returns the int `0` (`returned f_sd=0 (int)` in `v_empty_search.py`). It reaches only:
- `_eval_improvement_` and `_sto_success_improvement_`, whose `sigma` becomes a float;
- the step's return value, which the loop never uses (`bads.py:1356-1363`).

The incumbent and the hedge are not reached, and `search_dist = 0` reaches only `search_stats`, which nothing reads. Disposition: inert; a cosmetic `0.0` at most, with the fingerprint unchanged.

## 3. New, met while verifying

1. **Verified.** The noisy runs' repeated evaluations come from the poll (F1 above), which neither report nor B3-K3's record says.
2. **By reading, unverified.** Once F9 is fixed, the port's `frac = n_new/ntest` can hit 0/0. MATLAB's own 0/0 at `n_search_iter` ≥ 3 makes the scale NaN, and `uCheck`'s `min`/`max` projection, which ignores NaN, then sends every candidate to the corner `UBsearch`: a MATLAB-side defect.
3. **Unverified.** NumPy's default `argsort` may use SIMD sorting whose order among ties depends on the CPU (NumPy here is 2.4.6). If so, B3-K9 would also make seeded runs depend on the CPU.
4. **Verified.** Constructing `BADS()` adds a handler to the root logger (`root handlers before: 0`, after: 1), and `ESSearch.__init__` calls `logging.basicConfig` each time (`es_search.py:49`). This is outside the numerics.
5. **By reading.** `hedge_gamma`'s option description is the section header above it (`advanced_bads_options.ini:264-265`).
6. **By reading.** KD-B3-3's "the search hedge's reward update, which is ported" should say that it is ported with F5's formula error.
