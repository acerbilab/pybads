# Wave 3 ledger: B3 and B4

The verified findings of wave 3 of the port review
([plan](../../../plans/port-correctness-review.md), "Wave 3 pickup"): slice
B3, the search, and slice B4, the poll, the mesh, the incumbent and the
target, each read on both tracks. The reports are `reviews/B3_internal.md`,
`reviews/B3_comparison.md`, `reviews/B4_internal.md` and
`reviews/B4_comparison.md`. Each slice was verified by a fresh Opus agent
that had not written either report, with checks of its own:
`wave3_B3_verifier.md` and `wave3_B4_verifier.md`. The verifiers were also
given the items of the review's records that belong to their slice and that
the reviewers did not receive (the plan's "Wave 3 pickup", step 4), as
B3-K1 to B3-K13 and B4-K1 to B4-K11, quoted in `../briefs/wave3_kept_B3.md`
and `../briefs/wave3_kept_B4.md`. Every agent read PyBADS at `8aecb6a` (PI,
2026-09-26: `dev-next` after wave 2's fix pass, #76), MATLAB BADS at
`74919c0` and gpyreg v1.3.3 (`98ab5a4`), with the complete history of both,
in a cloud session (Linux, Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, where
the fingerprint of `dev/scripts/fingerprint.py` at `8aecb6a` is
`dc11118754b18b47`, the Linux reference's). The scripts and their outputs
are under `scripts/wave3/<slice>_<track>/` and
`scripts/wave3/<slice>_verifier/`, formatted by the pre-commit hooks after
they ran. The reports cite them at the sandbox's scratch paths.

Lines are at `8aecb6a`, MATLAB lines at `74919c0`. "B3-I F1" is finding F1
of the B3 internal report, "B3-C" the B3 comparison report, and likewise
"B4-I" and "B4-C"; "B3-K1" is a kept item. "Survey" names the row of the
candidate table of `dev/results/2026-09-23-codebase-survey.md` that
describes the same behavior. The dispositions are proposals until the PI
rules: the verifier's recommendation unless the row says why not. The gate
is the one of `AGENTS.md`, "Numerical gates", that a fix would need.
"Default run" says whether a run at default options reaches the code, and
at which uncertainty level (0 deterministic, 1 noise inferred, 2
`specify_target_noise`). Wave 0's fix pass reached `dev-next` squash-merged
as `0c56d86`, wave 1's as `fef6c14` and wave 2's as `8aecb6a`; wave 2's
commits cited here are those of its branch, `dev-port-review-w2`.

Two findings are shared by the slices: `contraints_check`, B3's function,
is called by the poll too (W3-1), and the target and the improvement
function, B4's, are called by the search. Each is verified once, in the
slice of its code, with the other slice's reach added.

## B3: the search

| Id | Source | What | Classification | Default run | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|
| W3-1 | B3-I F1, B3-C F1, B4-I F6, B4-C F2, B3-K3 | `contraints_check` removes no point already evaluated (`function_logger/constraints_check.py:33-43`): `np.unique` over the candidates' bins stacked above the log's returns each bin's first occurrence, which is always a candidate, so every distinct candidate bin survives; MATLAB's `setdiff(u1,u2,'rows')` drops the evaluated bins (`utils/uCheck.m:25`). Repeats in seeded runs: the search, at level 0, when the optimum is on a bound or the grid is coarse (9 of 16 and 20-21 of 29 search evaluations on Σ(x+1)² on [0,5]^D, D = 1, 2; 1-2 of 16 on an interior 1-D sphere; none on an interior 3-D sphere); the poll, at levels 1 and 2 (13 and 6 of 77 poll evaluations in two of four level-2 runs, 6 of 61 in one of two level-1 runs), which neither report measured. At levels 0 and 1 a repeat is a new row of the log and a duplicate training input of the GP; at level 2 it is merged into its row. With MATLAB's removal patched in, the bound-optimum runs had no repeats, the same `fval` and the same count | confirmed port discrepancy | yes, all levels (search at level 0, poll at levels 1 and 2) | matched MATLAB only in `c7c88ab` (an L1 test of the rounded rows); lost in `8e59038` (2022-06-03); `uCheck.m` unchanged since `a8817f9` (2017) | the `contraints_check` row; `dev/TODO.md`, "Previously evaluated points evaluated again" | fix: drop the candidates whose bin is in the log, as `setdiff`; `test_incumbent_constraint_check`, which asserts the current behavior, corrected; with W3-9 first, since the removal can empty a later ES generation | population comparison at default, with configurations that reach it: an optimum on a bound (level 0) and the noisy ones; the seeded tests checked over their seeds (`dev/TODO.md`) |
| W3-2 | B3-K4, B3-I F1 (the order) | `contraints_check` returns its candidates in `np.unique`'s order (the sorting variant at `constraints_check.py:41` is commented out) | not a defect: MATLAB's `setdiff` returns the rows sorted too (200 of 200 random sets in the same order; an input-order variant 0 of 200). Within one bin of 2^-20 the port keeps the first row, MATLAB the smallest, a difference below 2^-20 in `u` | yes | the order matches MATLAB since `8e59038` | — | correct the record: B3-I's "keep the input order" is not MATLAB's; the comment at `constraints_check.py:29` ("preserve the initial order") can say that the result is sorted, as MATLAB's, with W3-1 | none |
| W3-3 | B3-I F2, B3-C F4 | `ucov` ignores its weights (`search/es_search.py:324-329`): Σ_k w_k·SᵀS = SᵀS, since the weights sum to 1; `utils/ucov.m:19` computes the same. The ES-wcm covariance is the unweighted scatter of the best points about the incumbent, where the comments ("weighted covariance matrix"), the log weights and the name point to Σᵢ wᵢ sᵢsᵢᵀ; weighting would change the normalized search covariance by 13-46% (median 33%) over 20 GP states, and final values over 10 seeds on two problems did not move beyond the spread | confirmed shared defect | yes, all levels (every ES-wcm search) | the two have always agreed (`ucov.m` 2017; `c7c88ab`, rewritten in `e404fcc`, 2023) | — | PI: (a) keep MATLAB's arithmetic, correct the comments, and put it in `matlab_side_defects.md` as a shared observation; (b) weight both sides (with W3-4, without which a weighted form raises). Proposed: (a); B3-I's "port discrepancy" does not hold | none for (a); population comparison at default for (b) |
| W3-4 | B3-I F3, B3-C F3 | ES-wcm takes the ⌊μ⌋+1 best training points for ⌊μ⌋ weights (`es_search.py:247`, `y_idx[0 : floor(mu+1)]`); MATLAB takes `index(1:floor(mu))` (`search/searchES.m:62`). The normalized search covariance differs from MATLAB's by 2-40% (median 11%, relative Frobenius) | confirmed port discrepancy | yes, all levels (every ES-wcm search) | never agreed (`c7c88ab`; `searchES.m` unchanged since 2017) | — | fix: `floor(mu)` rows | population comparison at default |
| W3-5 | B3-I F4, B3-C F2 | the ES's selection mask is shifted by one parent rank (`es_search.py:70-74`): MATLAB's 1-based start positions `cw` are used as 0-based indices, so the mask is `[0] + M[:-1]` of MATLAB's `M` (`utils/ESupdate.m:19-21`). At μ = λ = 2048 the offspring of parents 0 to 5 are 1, 17, 12, 10, 9, 8 against MATLAB's 17, 12, 10, 9, 8, 7. On 20 captured states × 5 seeds the correct mask found a lower best LCB in 79% (ES-wcm) and 73% (ES-ell) of pairs, the port's in 17% and 19%, by a small median margin | confirmed port discrepancy | yes, all levels (the one reproduction of every search) | never agreed (`c7c88ab`; `ESupdate.m` unchanged since 2017) | — | fix (`np.repeat(np.arange(len(w)), w)`); `test_search_selection_mask`, whose golden sum 885072 is MATLAB's 1-based sum and locks the shift in, corrected (a correct 0-based mask sums to 883024) | population comparison at default |
| W3-6 | B3-I F5, B3-C F5, B3-K1 | the hedge's expected reward uses `exp(-0.5*g**2/sqrt(2*pi))` (`search/search_hedge.py:152-155`) where `acq/acqPortfolio.m:64` has `exp(-0.5*(gammaz.^2))/sqrt(2*pi)`, φ(γ): the port's reward is 2.51 times MATLAB's at γ = 0 and 424 times at γ = -3, so a strategy whose point was worse than the incumbent is still rewarded; in a level-1 run all 60 updates took this branch (ratio quartiles 2.36, 2.59, 2.94). Level 0 takes the `fs == 0` branch, where the two agree | confirmed port discrepancy | yes, levels 1 and 2 | never agreed (`c7c88ab`; MATLAB's line correct since `5bb226d`, 2017) | `search/search_hedge.py:141` | fix (the parenthesis); KD-B3-3's "ported" gains the formula's history | population comparison of the noisy configurations |
| W3-7 | B3-I F6, B3-C F8 | `hedge_gamma = 0` stops the run at its first search: `update_hedge` scores every strategy and slices the 1-D point's coordinates (`search_hedge.py:127`), so `gp.predict` receives D−1 values. MATLAB takes the same row for every strategy (`acqPortfolio.m:40`), which is its intent, and then fails on `gpstructnew`, undefined there (`acqPortfolio.m:47`) | confirmed shared defect (both fail, at different places) | no (`hedge_gamma = 0`) | Python `c7c88ab`; MATLAB since 2017 (the assignment of `gpstructnew` commented out before `d4fead5`) | — | fix: score each strategy at the search point as a row (MATLAB's intent); an entry in `matlab_side_defects.md`. B3-C's "second search" is the first | fingerprint; a test with `hedge_gamma = 0` |
| W3-8 | B3-I F7, B3-C F7 | the fraction of new candidates behind the ES's scale update is miscounted (`es_search.py:177-192`): `z_idx[0:ntest+1] > nold` looks at one entry too many and misses index `nold`, and from the third generation the untrimmed pool counts older rows as new (0.56, 0.77, 0.87, 0.93 against MATLAB's 0.56, 0.43, 0.39, 0.37, `searchES.m:170-193`). The update runs only for 1 < i < `n_search_iter` (1-based), never at the default 2; from 4 on the scale grows where MATLAB's shrinks [the doublecheck: from 3 on it grows faster than MATLAB's, and where MATLAB's shrinks from about the fifth generation, so from `n_search_iter` 6; `wave3_doublecheck_docs.md`, F3] | confirmed port discrepancy | no (`n_search_iter` ≥ 3; material from 4) | never agreed (`c7c88ab`) | — | fix, with a guard for `ntest == 0`, which W3-9's fix makes reachable (MATLAB gets 0/0 there) | fingerprint; a test with `n_search_iter = 4` |
| W3-9 | B3-I F9, B3-C F6, B3-K10 | when every candidate of a later ES generation is removed, the fallback for a failed acquisition sets `z_candidates = rng.random(0)` (`es_search.py:170-175`), which discards the earlier candidates' values, and the ES returns an empty set with the warning "random search is performed", although none is; MATLAB keeps `zold` and returns the best earlier candidate (`searchES.m:168-182`). B3-K10's warning at D = 3 on a thin band is this path: every emptied generation was the second, whose size is the number of first-generation survivors (12 of 2048 in one call), on both sides; the GP's few points are incidental | confirmed port discrepancy | no (a `non_box_cons` that empties a later generation; W3-1's fix can too) | never agreed (`c7c88ab`); an `IndexError` until `0c56d86`, an empty set since | — (B3-K10) | fix: skip an empty generation and keep the candidates; reword or remove the warning (a NaN acquisition, the real failure, is never caught); before W3-1 | fingerprint; a test with a thin band |
| W3-10 | B3-I F8, B3-C F9, B3-K11 | `acq_fcn_lcb` refuses a plain number as `sqrt_beta` (`acquisition_functions/acq_fcn_lcb.py:42`: `(2.0).size` raises `AttributeError`), and a non-finite value and a schedule's name, which `acqLCB.m:16-18` accepts; a NumPy scalar works. The docstring calls the SD output a variance (`acq_fcn_lcb.py:27-28`, B3-K11) | confirmed port discrepancy; the docstring: confirmed, inert | no (a number as the second element of `search_acq_fcn`) | the check `c7c88ab`; the docstring `de1ee08` | — | fix: test `np.size`/`np.ndim` of `np.asarray(sqrt_beta)`; PI: whether non-finite values and names are accepted (proposed: a positive finite number or a callable, refused otherwise with a message); the docstring corrected | fingerprint |
| W3-11 | B3-C F10, B3-K6 | an empty search set: the survey's two rows (the ES's `IndexError`, the step's `UnboundLocalError`) no longer hold since `0c56d86` (W0-15), which counts it as a failed search. B3-C F10 finds that the fix does what MATLAB does only in the status at default: MATLAB (a) fails as PyBADS at `improvement_quantile` ≤ 0.5 or level 0; (b) at q > 0.5 with `fsd` > 0 counts an incremental search and moves the incumbent to the previous search's stale `usearch` with `fval` and SD 0; (c) runs the hedge update with that stale point, er = 0, so every gain decays, where PyBADS skips the update (`bads.py:1996`); (d) errors on an undefined `usearch` when the run's first search is empty (`bads.m:667-725`, `1257-1279`). The comment at `bads.py:1956` and the commit say "as in MATLAB BADS … on every path" | the rows: no longer hold; the rest: design question ((b) and (d) are MATLAB defects that PyBADS avoids; (c) a difference) | the hedge's (c): when a set is empty after an earlier search (only through `non_box_cons`), all levels; (b): no | forced failure `0c56d86`; the hedge's guard `c7c88ab`; MATLAB since 2017 | the two rows on the empty search set | PI: decay the gains on an empty set as MATLAB, a failed search on the hedge's path too; or keep the skip and put it on the sheet. Proposed: decay them (the verifier leaves it open), since a failed search with a point decays them as well and W0-15's ruling counts an empty set as a failed search; (b) and (d) on the sheet as deliberate, the comment corrected; the survey's rows corrected | fingerprint; a test with a `non_box_cons` that empties a set |
| W3-12 | B3-K2 | after a failed rebuild, the search ranks its candidates by the LCB of the GP restored by `local_gp_fitting` (a consistent GP with finite predictions; with an injected failure in 3-D it chose a point at an offset of (-142, 38, 93) grid units, LCB 0.0087); MATLAB's `post = []` makes `gppred` fail again, `acqLCB` sums over no samples, z ≡ 0, and the stable sort keeps `uCheck`'s first, lexicographic candidate (at an offset of (-1891, 1696, 291) units, LCB 1.55 under the same GP). The poll treats such a GP as unreliable on both sides | design question | only after a failed rebuild (rare; none in the default suite), all levels | since `685da15` (before it the run stopped); never agreed | the `_search_step_` row "(at `a83bd51`)" | keep the ranking by the restored GP (MATLAB's choice is an arbitrary far point), and extend KD-B5-2 with it | none |
| W3-13 | B3-K5 | `ESSearchCMA` cannot run: `U[y_idx[-1 : -1 : ...]]` takes a float slice index (`TypeError`), the slice would be empty and reversed, and `ucov` is called with 4 of its 7 arguments (`es_search.py:261-262`); the hedge refuses `'ES-cma+'` | confirmed, inert (unreachable, KD-B3-1) | no | `c7c88ab` | `search/es_search.py:239-253` | PI: remove the class, or leave it with a comment that it is broken. Proposed: remove it (KD-B3-1 and `AGENTS.md` updated), since it cannot be reached and cannot run | fingerprint |
| W3-14 | B3-K8 | `force_to_grid` rounds halves to even (`search/grid_functions.py:12`, `np.round`), MATLAB's `force2grid.m:5` away from zero: `x0 = [1, 3]` in a plausible box `[-2048, 2048]` starts at `[0, 4]` in PyBADS and `[2, 4]` in MATLAB. No exact half among 96,000 Sobol design coordinates; ES draws are continuous, the poll is not put on the grid at default, and the search bounds do not depend on the rounding (their correction step) | confirmed port discrepancy | yes, all levels, but only on exact halves: in practice a start point (an integer `x0` in a box symmetric about 0) | never agreed (`c7c88ab`) | — | fix: `sign(q)·floor(|q| + 0.5)` | fingerprint (unchanged unless a start lies on a half); a test of the rounding |
| W3-15 | B3-K9 | the ES orders candidates and training points with the unstable `np.argsort` (`es_search.py:190`, `246`), MATLAB with a stable `sort`. Ties are common: far candidates whose predictions equal the GP's prior mean exactly. A stable sort gave another permutation in 31-35 of 58 calls on a sphere and 18-26 of 80-88 on Rosenbrock (line 190), 9 of 9 and 10 of 12 on a quantized sphere (line 246), and 5 of 6 seeded runs differed; the best point returned never changed, the selected set in 11 of 57 calls and the parents' order in all 57. With a stable sort, the order among ties is MATLAB's | confirmed port discrepancy | yes, all levels (every search) | never agreed (`c7c88ab`) | — (wave 0's "Found while verifying") | fix (`kind="stable"` at both lines); whether NumPy's unstable sort orders ties differently on different CPUs (the verifier's unverified note) would make seeded runs depend on the machine, which the fix removes | population comparison at default |
| W3-16 | B3-K7 | `search_factor_min` unread | no longer holds: `3272bdd` (W2-16) floors the factor after a failed search; against a transcription of `UpdateSearch`, 0 of 400 random statuses differ, with `adaptive_incumbent_shift` on and off | yes | fixed 2026-09-26 | — (the preparatory report's (a)) | none | — |
| W3-17 | B3-K12 | `search_n_try` a float | no longer holds: an `int` since `2d4304c` (W2-24), equal to MATLAB's at D = 1, 2, 3, 6, 7, 20 | yes | fixed 2026-09-26 | — | none | — |
| W3-18 | B3-K13 | the empty branch's `f_sd_search = 0` (`bads.py:1907`) is an `int`; it reaches `_eval_improvement_` (as a float after the arithmetic) and the step's return value, which the loop does not use; the result's `fsd` is a float since `d964576` (W2-12) | confirmed, inert | only with an empty set | `c7c88ab` | — | `0.0`, with W3-11 | fingerprint |

## B4: the poll, the mesh, the incumbent and the target

| Id | Source | What | Classification | Default run | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|
| W3-19 | B4-I F3, B4-C F1, B4-K6 | `p_less`, the probability that no remaining poll point improves, is taken over unsorted probabilities and D+1 of them (`bads.py:2281-2284`): `f_pi` is (n, 1), so `np.sort` sorts along the axis of length 1 and `[::-1]` reverses the rows; the product runs over the last D+1 points of the lexicographic poll set. MATLAB sorts descending and takes the D largest (`bads.m:868-869`). With a first point at PoI 0.69 and five near 0 at D = 3: the port 0.9999999960, stops; MATLAB 0.31, goes on. In 34 stop decisions after a good poll (verifier) and 80 more (B4-C), none flipped; in 4 of 6 with more than D+1 points left the largest PoI was left out. The threshold, 1 − 1e-6/D, makes the effect rare | confirmed port discrepancy | yes, all levels (it decides only after a good poll with a reliable GP) | never agreed (`c7c88ab`; MATLAB 2017) | — (the preparatory report's (e)) | fix: sort the raveled probabilities, take `min(D, n)` | population comparison at default (D ≥ 3 configurations); a unit test of `p_less` |
| W3-20 | B4-I F4, B4-C F3 | `uncertain_incumbent=False` at level 0 stops the run at the first poll: `_get_target_from_gp_` returns Python floats there (`bads.py:2696-2699`), and the callers call `.item()` on them (`2245`, `2249`; the search `1748`, `1752`): `AttributeError`. MATLAB's branch works (`bads.m:1328-1332`) | confirmed port discrepancy | no (`uncertain_incumbent=False`, level 0) | the branch never worked (`c7c88ab`, `9037851`) | — | fix: return arrays, as the fallback does | fingerprint; a test |
| W3-21 | B4-C F4, B4-K2 | the target under `hyp_best`: PyBADS sets the hyperparameters on a copy and recomputes its posterior (`bads.py:2659-2670`); MATLAB's `UpdateTarget` keeps `post` and evaluates the kernel and mean under `hyp` (`bads.m:1301`, `utils/gppred.m:39-47`, `utils/mygp.m:122-123`, `146-187`), a hybrid that is no GP prediction under one set of hyperparameters (emulated: equal to `gp.predict` under the GP's own hyperparameters; under those of 1 to 3 iterations earlier, means of 1.2e3 to 2.8e7 where the observed values are at most 5e-3). Hyperparameters differed in none of 21 decisions of three runs at level 0 and three at level 1 (10 and 11 decisions), and in 5 of 13 of five other runs at level 1; the hybrid would flip 1, the current GP's own prediction none (143.053 against 143.054) | design question (KD-B4-2 leaves it open) | yes, all levels (it matters when `hyp_best` differs from the current hyperparameters: a refit in the poll after its best point, a move after the re-estimate) | never agreed (`c7c88ab` predicted from the current GP; `9037851` recomputes; `685da15` the fallback) | the `_get_target_from_gp_` row "(at `676083d`)" | PI: (a) keep the recomputation and settle it in KD-B4-2; (b) predict from the current GP, which removes a copy per step and the `LinAlgError` path of KD-B4-2, and moves runs by little; (c) MATLAB's hybrid, not recommended. Proposed: (a), which realizes MATLAB's evident intent (the target under the best iteration's hyperparameters) with a valid prediction | none for (a); population comparison at level 1 for (b) |
| W3-22 | B4-I F10, B4-K5 | the poll calls `np.seterr(divide="ignore")` when the root logger is above DEBUG (`bads.py:2271-2272`) and never restores it: after a run, `np.geterr()["divide"]` is `'ignore'` and 1/0 in the user's code no longer warns | confirmed defect (Python only) | yes, all levels | `f9e9326` (2022-11-02) | the `_poll_step_` row "(at `4bde5e9`)" | fix: `np.errstate(divide="ignore", invalid="ignore")` around `gamma_z`; `test_seeded_run_leaves_global_state_untouched` extended to `np.geterr()` | fingerprint |
| W3-23 | B4-I F7, B4-K3 | when the target's prediction is not finite, `_get_target_from_gp_` falls back to the incumbent's `fval` and `fsd`, but the target's formula keeps the raw `fs2` (`bads.py:2672-2695`): `fs2 = NaN` gives a NaN target, `inf` gives `-inf`, and the poll then treats the GP as unreliable. MATLAB does the same (`bads.m:1310-1311`, `1321`), where it follows a failed rebuild; in PyBADS a restored GP is consistent, and gpyreg's predictions are non-finite only on overflow (none seen). `test_target_fallback_to_incumbent` accepts a NaN target | confirmed shared defect, inert in practice | no (a non-finite prediction) | MATLAB 2017; Python `c7c88ab`, reshaped in `685da15` | the `_get_target_from_gp_` row on a non-finite target | fix: `f_target_s**2` in the fallback's formula; the test asserts a finite target; an entry in `matlab_side_defects.md` | fingerprint |
| W3-24 | B4-I F1 | the poll's basis is always the ± coordinate directions: `n_max = max(1, round(search_mesh_size/mesh_size))` (`poll/poll_mads_2n.py:22`) is 1 at every default state, since the search mesh is at least 2^10 times finer than the poll mesh, so the basis is a signed permutation of the identity: a coordinate poll, not LTMADS's dense directions. `pollMADS2N.m:7` is identical, and both user documents (`README.md`, `docsrc/source/index.rst`, MATLAB's README) describe steps in one direction at a time; the docstring of `poll_mads_2n` claims "dense refining directions" and convergence guarantees and cites the Sto-MADS paper for LTMADS | design question, shared with MATLAB | yes, all levels | the two agree (MATLAB 2017, `c7c88ab`) | — | keep MATLAB's poll, correct the docstring of `poll_mads_2n`, and put it in `matlab_side_defects.md` as a shared observation; real LTMADS directions would depart from MATLAB | none if kept |
| W3-25 | B4-I F2 | the GP's `poll_scale` never shapes the poll vectors: `poll_mads_2n` divides by it and `_poll_step_` multiplies it back (`poll_mads_2n.py:36-37`, `bads.py:2146-2150`) | not a defect: MATLAB does the same on purpose ("Counteract subsequent multiplication by pollscale", `pollMADS2N.m:23-24`, `bads.m:803`); `poll_scale` shapes only the ES-ell search (and MATLAB's non-default `pollGPS2N`) | yes | the two agree | — | correct the record: `AGENTS.md` ("drive the poll basis and the ES-ell search") and the description of `gp_rescale_poll` ("scaling factor of poll vectors"), which scales only the ES-ell search | none |
| W3-26 | B4-I F5 | at level 0 the poll's GP never takes the poll's own evaluations (only levels 1 and 2 add them, `bads.py:2309-2332`), so after an improving point the target is predicted at `u_poll_best`, where the GP has no data (observed 4.155, predicted 86.41; observed 26.35, predicted 11.44), and the remaining points' LCB and PoI ignore the poll's observations. MATLAB does the same (`bads.m:908`, `UpdateTarget(upollbest, …)`); using a GP that holds the point changed 1 of 9 runs | design question, shared with MATLAB | yes, level 0 | the two agree (2017, `c7c88ab`) | — | keep MATLAB's behavior, as a shared observation in `matlab_side_defects.md` | none if kept; population comparison at level 0 if changed |
| W3-27 | B4-I F8 | `np.argmin` returns the first NaN of the acquisition (`bads.py:2257`, the search's `1805`), where MATLAB's `min` skips NaN; the fallback "randomly choose index" can never fire (`argmin` always returns a finite index), on both sides | confirmed, inert | no (a NaN prediction; none seen) | never agreed (`c7c88ab`) | — | fix cheaply: `nanargmin` with a guard for an all-NaN set, which makes the fallback live, at both sites | fingerprint |
| W3-28 | B4-I F9 | a good poll stops whenever the GP is unreliable, and a zero predictive SD at any remaining point makes γ infinite and the GP unreliable, whatever `tol_poi` says, although its description says 0 always completes polling. Zero SDs were frequent at level 0 (16 to 38 of 60 to 72 poll steps), none after a good poll in 15 runs | not a defect: MATLAB's rule (`bads.m:862-895`) | yes (the rule), all levels | the two agree | — | keep; the description of `tol_poi` says that an unreliable GP stops a good poll | none |
| W3-29 | B4-C F5 | MATLAB's `pollmoved_flag` is set only by the poll (`bads.m:956`, `958`) and read at the end of every pass (`1049`: `gpstruct.post = []`), so after a poll that moved the incumbent every search of the next round rebuilds the local GP, until a poll that does not move; PyBADS rebuilds once, which the first search of the round does anyway (`bads.py:1734-1737`, `2498`). After a search move MATLAB rebuilds once, as PyBADS. W1-2's premise, "MATLAB's `post = []` asks for one rebuild" (`verification/wave1.md`), missed line 1049; the changelog's "Rebuilds of the local GP" and the comment at `bads.py:1734-1736` say "as MATLAB BADS does". The rebuilds keep the hyperparameters, and in 24 such searches (95 in B4-C's runs) the training set and the predictions were the same: no effect within 200 evaluations; longer runs can differ once the nearest-neighbour set changes [the doublecheck: runs of fewer than 200 evaluations change too, once the local GP reaches `n_train_max` (51 points at D = 3), whose rebuild recomputes the posterior; its gate changed 277 of 540 runs, `rastrigin_D3`'s among them, of 65 to 184 evaluations; `wave3_doublecheck_B4.md`, F1] | confirmed port discrepancy (MATLAB's persistence looks unintended to the reviewer) | yes, all levels | matched after a poll move until `fef6c14` (W1-2), which made the rebuild once only; MATLAB 2017 | — | PI: (a) persist after a poll move only, as MATLAB; (b) keep one rebuild and put it on the sheet. Either way the changelog entry, the comment and wave 1's row are corrected. Proposed: (a), since W1-2 ruled toward MATLAB on a premise that missed this line | population comparison at default for (a) (long runs reach it) |
| W3-30 | B4-C F6 | with `poll_training=False` the poll neither records a refit it does not make nor clears the unreliability flag (`bads.py:2195-2200`), where MATLAB's `IsRefitTime` sets `lastfitgp`, resets the statistics and clears `unrelgp_flag` before the refit is cancelled (`bads.m:822-823`, `1242-1252`) | intentional difference, missing from the sheet (W1-8's ruling; the changelog, "Refits without poll training"; `matlab_side_defects.md`) | no (`poll_training=False`) | `fef6c14` | — | a sheet entry (with KD-B5-2) | none |
| W3-31 | B4-C F7 | `_eval_improvement_` accepts an `improvement_quantile` outside (0, 1) (`bads.py:2018-2037`), where MATLAB refuses it (`bads.m:1269-1271`): at 0 or 1 `erfcinv` is infinite, the improvement at level 0 is 0·∞ = NaN, the incumbent never moves, and a 2-D sphere spends its 100 evaluations to end at its best initial point, without an error | confirmed port discrepancy | no (`improvement_quantile` ≤ 0 or ≥ 1) | never agreed (MATLAB's check since `d04640a`, 2017) | — | fix: refuse such a value with `ValueError` when `BADS` is created (a stricter interface: a changelog entry and an "Upgrading from" line) | fingerprint |
| W3-32 | B4-K1 | a successful poll appends the bound method `self.u_best.copy` | no longer holds: `self.u_best.copy()` since `0c56d86` (W0-16) | yes | fixed in wave 0's fix pass | `bads.py:2183` | correct the survey's row | — |
| W3-33 | B4-K4 | after a re-estimate that moves nothing, `optim_state`'s `yval`, `fval` and `fsd` keep older values (`bads.py:1507-1509` update only the object's), read only by the target's fallback (never reached) and by the copy that `output_fcn` receives; stale in 6 to 20 of about 158 target computations per level-1 run. MATLAB's `optimState.fval` is staler (its move never updates it, `bads.m:1111-1118`) | confirmed, inert (shared) | yes, levels 1 and 2, without consequence | Python `c7c88ab`; MATLAB 2017 | — | keep, or keep `optim_state` in step in the re-estimate. Proposed: keep them in step, since `output_fcn` receives them (the verifier leaves both open) | fingerprint |
| W3-34 | B4-K7 | `np.vstack(u_poll, u_poll_new)` with two arguments (`bads.py:2180`) would raise `TypeError`; the branch cannot run, since the basis is filled once with 2D rows (56 of 56 polls called `poll_mads_2n` once) | confirmed, inert (unreachable) | no | `c7c88ab` | — (wave 0's "Found while verifying") | remove the branch | fingerprint |
| W3-35 | B4-K8 | the poll discards the return of `period_check` (`bads.py:2154-2159`) | confirmed, inert (the stub returns its input; periodic variables are refused, KD-B1-6) | executed, without effect | `c7c88ab` | — | keep; assign the result when periodic variables are ported (KD-B1-6) | none |
| W3-36 | B4-K9 | `u_base` computed and never used in the accelerated mesh reduction (`bads.py:2442-2444`), beside a commented-out condition | confirmed, inert | executed, without effect | `9037851` | — | remove it | fingerprint |
| W3-37 | B4-K10 | the accelerated mesh reduction tested from the wrong iteration (W2-29) | no longer holds: `c9a2cde` tests `iter >= accelerate_mesh_steps`, MATLAB's `iter > steps` with its count from 1, and reads the same stored iteration (`iter - steps`, MATLAB's `iter - steps` counted from 1); at each poll with `iter` ≥ 3 the history holds exactly `iter` entries | yes | fixed 2026-09-26 | — | none | — |
| W3-38 | B4-K11 | under `stobads`, a NaN estimate after a failed add counts as uncertain | no longer holds: `_sto_success_improvement_` returns −1 for a non-finite estimate since `0c56d86` (W0-11) | no (`stobads`) | fixed in wave 0's fix pass | the `_poll_step_` row with `stobads` "(at `a83bd51`)" | correct the survey's row | — |
| W3-39 | wave 2's doublecheck (`wave2.md`, "Doublecheck", "Left"), added after the triage | `accelerate_mesh_steps` below 1 stops the run with `TypeError` at its first failed poll: the accelerated mesh reduction reads `iteration_history`'s `fval` and `fsd` at `iter - accelerate_mesh_steps` (`bads.py:2434-2443`), an iteration not recorded yet (the current one at 0; at the first failed poll `iteration_history.get("fval")` is still `None`). Reproduced by the orchestrator at `4f50376` on a 2-D sphere, seeds 0 and 1 (`scripts/wave3/orchestrator/w3_acc0.py`). MATLAB fails too: `iterList` starts empty (`setupvars.m:179-182`) and `bads.m:976-979` read `iterList.fval(iter - AccelerateMeshSteps)` with `iter > 0` | confirmed shared defect | no (default 3) | older than the review (`157bd09`, 2022; MATLAB 2017) | — | PI, after the gates: refuse a value that is not a positive integer when `BADS` is created (fixed in `5d711bf`), as W3-31 does for `improvement_quantile` (a stricter interface: a changelog entry and an "Upgrading from" line), and an entry in `matlab_side_defects.md` | fingerprint; a test with `accelerate_mesh_steps=0` |
| W3-40 | W3-24's gate (`geometry_w3-24_crashes.txt`), added after the gates | a rebuild of the local GP on two distinct points stops the run with gpyreg's `ValueError` for a prior with a sigma of 0: the empirical prior of the log length scales takes its centre and width from the spread of the training set's pairwise distances (`gaussian_process_train.py:360-382`), and two points have one distance. Reached by W3-24's tilted poll on the thin bands of the `geometry` suite (three runs), and, with MATLAB's coordinate poll, by a band along a coordinate, in four of four seeds (`scripts/wave3/doublecheck/b_B4/w3_40_band.out`). MATLAB computes the same zero width (`gpdef/gpdefBads.m:240-251`), where GPML's `priorGauss` gives NaN, which enters only the fit's objective | confirmed shared defect (B6's code) | no (a feasible region that leaves two points in the training set; no run of the `default` or `geometry` suite at MATLAB's poll) | never agreed on a guard: Python `c7c88ab`; MATLAB `31a39f3` (2017) | — (`dev/TODO.md`, "The GP on a one-point training set") | PI, after the gates: keep the previous prior, as a rebuild on targets without spread keeps its own (KD-B6-2); fixed in `a14524d`, with an entry in `matlab_side_defects.md` | fingerprint; a test of a rebuild on two points |

## Notes on the reports

- **Corrections by the verifiers.** B3: I-F1's "keep the input order" is not
  MATLAB's, whose `setdiff` sorts (W3-2); I-F2 is shared with MATLAB, not a
  port discrepancy (W3-3); I-F5, "unsure", is a port discrepancy, since
  MATLAB's line is right (W3-6); I-F6's "scored at the chosen point" is
  MATLAB's intent, and C-F8's crash comes at the first search, not the
  second (W3-7); I-F7's NaN scale is unreachable today (W3-8); I-F8's "an
  array raises" holds only for arrays of more than one element (W3-10);
  neither report saw that in noisy runs the repeated evaluations come from
  the poll (W3-1). B4: I-F1 and I-F2 are MATLAB's behavior (W3-24, W3-25),
  I-F5 and I-F9 too (W3-26, W3-28); C-F5 matched MATLAB after a poll move
  until `fef6c14` (W3-29); C-F6 is intentional (W3-30).
- **The items kept from the reviewers.** Found by the reviewers: the hedge's
  reward (B3-K1, both B3 reports), the evaluated points kept (B3-K3, all
  four reports), the order of `contraints_check` (B3-K4, B3-I), the emptied
  generation behind the warning (B3-K10, both B3 reports, not tied to the
  warning's record), the docstring of `acq_fcn_lcb` (B3-K11, B3-I), `p_less`
  (B4-K6, the preparatory report's (e), both B4 reports), the target under
  `hyp_best` (B4-K2, B4-C), the target's fallback (B4-K3, B4-I, and B4-C in
  passing), `np.seterr` (B4-K5, B4-I), and in passing the unreachable
  `np.vstack` (B4-K7, both B4 reports) and the stale `optim_state` values
  (B4-K4, B4-C). Named by the reviewers as immaterial, and verified as
  findings: the rounding of halves (B3-K8, both B3 reports: it matters at
  the start point, W3-14) and the unstable sort (B3-K9, both B3 reports:
  ties are common and 5 of 6 seeded runs change, W3-15). Not found: the
  search after a failed rebuild (B3-K2), `ESSearchCMA` (B3-K5), the `int`
  SD of the empty branch (B3-K13), `period_check`'s return (B4-K8) and
  `u_base` (B4-K9). Wave 2's fixes in
  these slices, W2-16 and W2-29, hold as MATLAB's (W3-16, W3-37); the kept
  items that wave 0's fixes had closed no longer hold (W3-11, W3-32, W3-38),
  and neither does `search_n_try`'s type (W3-17).
- **The sheet.** KD-B3-3 calls the search hedge's reward update "ported"
  (W3-6: with the wrong formula); KD-B4-2 leaves the target's recomputation
  open (W3-21). Entries to add: the poll without training (W3-30), and, as
  ruled, the empty search set (W3-11), the search after a failed rebuild
  (W3-12, with KD-B5-2) and the rebuilds after a poll move (W3-29 (b)). No
  finding contradicts an entry.
- **Records.** `AGENTS.md` says that `poll_scale` drives the poll basis
  (W3-25); the changelog's "Rebuilds of the local GP", the comment at
  `bads.py:1734-1736` and W1-2's row of `verification/wave1.md` say that
  MATLAB rebuilds once (W3-29); the comment at `bads.py:1956` says that an
  empty search set is handled as in MATLAB on every path (W3-11); the brief
  of B3 said that the search runs from the first iteration, where on both
  sides the first iteration only polls (`search_count` starts at
  `search_n_try`, `setupvars.m:173`), which misled no reviewer.
- **Tests.** `test_incumbent_constraint_check` asserts W3-1's behavior;
  `test_search_selection_mask`'s golden sum locks in W3-5; `test_u_cov`,
  `test_search` and `test_search_hedge` check shapes only;
  `test_target_fallback_to_incumbent` accepts a NaN target (W3-23);
  `test_poll_mads.py` asserts `n_max = 1` at the default mesh sizes and
  multiplies back by `poll_scale` before its checks (W3-24, W3-25); nothing
  tests `update_hedge`'s values, `p_less`, `uncertain_incumbent=False` or
  `np.geterr()`.
- **A proposal for grouping the fixes.** Moving results at default, each
  under a population comparison on Linux against
  `population_linux_wave2_20260926` (this box computes as its environment:
  the fingerprint at `8aecb6a` is its `dc11118754b18b47`): the ES search,
  W3-4, W3-5 and W3-15 (one batch, its steps compared if it is flagged);
  W3-9 then W3-1, alone, with configurations that reach it (an optimum on a
  bound, the noisy ones); W3-6, on the noisy configurations; W3-19; and
  W3-29 if (a) is ruled. W3-14 moves the fingerprint only if one of its
  starts lies on a half. The rest must move nothing, each under the
  fingerprint: W3-7, W3-8, W3-10, W3-11, W3-13, W3-18, W3-20, W3-22,
  W3-23, W3-27, W3-31, W3-33, W3-34, W3-36, and the records (W3-2, W3-3,
  W3-12, W3-21, W3-24 to W3-26, W3-28, W3-30).

## Survey rows of B3 and B4

Every open row of the candidate table that belongs to these slices is
closed by a row above: `search/search_hedge.py:141` (W3-6); `bads.py:2183`
(W3-32, fixed in wave 0); `_get_target_from_gp_` at `676083d` (W3-21) and on
a non-finite target (W3-23); `_poll_step_` with `stobads` at `a83bd51`
(W3-38, fixed in wave 0, slice S); `_search_step_` at `a83bd51`, after a
failed rebuild (W3-12); `contraints_check` (W3-1, W3-2);
`search/es_search.py:239-253` (W3-13); the two rows on the empty search set
at `8afbe16` (W3-11, fixed in wave 0); and `_poll_step_` at `4bde5e9`
(W3-22). `dev/TODO.md`'s "Previously evaluated points evaluated again" is
W3-1, and three of its "Follow-ups of the GP-update guards" are settled
here or earlier: the target's posterior (W3-21), the search after a failed
rebuild (W3-12) and the NaN under `stobads` (W0-11).

## Found while verifying

Met by a verifier, outside the reports' findings and the kept items, and
not verified beyond the check named; each goes to the wave of its slice, or
is proposed here.

- `BADS()` adds a handler to the root logger (none before, one after), and
  `ESSearch.__init__` calls `logging.basicConfig(stream=sys.stdout, ...)` at
  every construction (`es_search.py:48-49`), which configures the user's
  root logger when it has no handler (B3 verifier, verified; B3-C noted the
  call). Proposed: remove the call, under the fingerprint, with the
  `BADS` logger left as KD-B2-3 describes it.
- Once W3-9 is fixed, `frac = n_new/ntest` can be 0/0 (W3-8's guard); in
  MATLAB the same 0/0 at `n_search_iter` ≥ 3 makes the scale NaN, and
  `uCheck`'s `min`/`max` projection, which ignores NaN, sends every
  candidate to the corner `UBsearch` (B3 verifier, by reading): a
  MATLAB-side defect for `matlab_side_defects.md`, off MATLAB's defaults.
- NumPy's default `argsort` may use SIMD sorting whose order among ties
  depends on the CPU (B3 verifier, unverified); if so, W3-15 also makes
  seeded runs machine-dependent, and its fix removes that. With W3-15.
- `_get_target_from_gp_` deep-copies the GP and recomputes its posterior at
  every search and poll step, and at default nothing reads the search's
  target (B4 verifier; B3-I and B4-C note the same): time, and a path that
  can raise `LinAlgError` (KD-B4-2). With W3-21: under (b), or by computing
  the search's target only when an option reads it.
- Zero predictive SDs are frequent at level 0 (W3-28); their cause, perhaps
  the latent variance clamped at 0 after rounding, and whether MATLAB's
  `mygp` gives them as often, are not established (B4 verifier).
- `hedge_gamma`'s description is the header of its section
  (`advanced_bads_options.ini:264-265`; B3-I, B3 verifier): with W3-7.
- From the reports' answers, not findings: `sloppy_improvement` also
  floors the sufficient improvement at `tol_fun`, which its description
  does not say (B3-I, B4-I); ES-ell ignores the sum-rule flag of
  `search_method`, non-default (B3-I); `udist`'s periodic branch indexes
  the distance matrix's rows by variable and is unreachable (B3-I,
  KD-B1-6); the docstring of `_get_target_from_gp_` says that the target is
  shifted only for a stochastic function and calls `f_target_s` a variance
  (B4-I); the search step counts the points of the log, MATLAB the GP's
  training set, equal in practice (B3-C, a B2 item): for the docstrings
  and descriptions of the fix pass.

## Rulings (PI, 2026-09-27)

The orchestrator proposed a disposition for every row, following its
verifier's recommendation unless the row says why not; where the verifier
left the design open (W3-11, W3-33), the row gives the proposal and its
reason. The PI ruled W3-24 (b), after a clarification of what the row
means, and accepted every other proposal as written. As in waves 0 to 2, a
fix is one commit per row on the wave's branch, with a test that fails at
`8aecb6a` and passes at the commit, and a changelog line in every commit a
user can notice; a stricter interface also has an "Upgrading from" line.
The fix pass follows `wave2.md`, "Fix pass", with the whole fast suite run
after every cherry-pick, since the fix agents run only their own test
files, and CI checked after every push that touches `pybads/`.

**Fix, moving nothing** (each under the fingerprint):

- W3-7: `update_hedge` scores each strategy at the search point taken as a
  row, MATLAB's intent; an entry in `matlab_side_defects.md` (MATLAB's
  undefined `gpstructnew`); a test with `hedge_gamma = 0`. With it the
  description of `hedge_gamma`, which is its section's header.
- W3-8: the fraction of new candidates counted as MATLAB counts it, with a
  guard for `ntest == 0`; a test with `n_search_iter = 4`.
- W3-9: an emptied ES generation is skipped and the earlier candidates
  kept; the warning reworded to say what happened (a NaN acquisition is
  not caught today); a test with a thin band. Before W3-1.
- W3-10: `sqrt_beta` is a positive finite number or a callable, and
  anything else is refused with a message; the docstring's SD. An
  "Upgrading from" line if a value that ran before is refused.
- W3-11: an empty search set updates the hedge as a failed search (every
  gain decays, as in MATLAB); MATLAB's move to a stale point at
  `improvement_quantile` > 0.5 and its error on a first empty search go on
  the sheet as differences PyBADS keeps; the comment at `bads.py:1956`
  corrected; W3-18's `0.0` in the same commit.
- W3-13: `ESSearchCMA` removed; KD-B3-1 and `AGENTS.md` updated.
- W3-14: `force_to_grid` rounds halves away from zero, as MATLAB; if the
  fingerprint moves, the row takes a population comparison instead.
- W3-20: `_get_target_from_gp_` returns arrays with
  `uncertain_incumbent=False`; a test.
- W3-22: `np.errstate` around `gamma_z`, and no global `np.seterr`;
  `test_seeded_run_leaves_global_state_untouched` extended to
  `np.geterr()`.
- W3-23: the fallback's target from `f_target_s**2`; the test asserts a
  finite target; an entry in `matlab_side_defects.md`.
- W3-27: `nanargmin` with a guard for an all-NaN set, at the search and the
  poll.
- W3-31: an `improvement_quantile` outside (0, 1) refused when `BADS` is
  created (a stricter interface: a changelog entry and an "Upgrading from"
  line).
- W3-33: the re-estimate keeps `optim_state`'s `yval`, `fval` and `fsd` in
  step with the incumbent's.
- W3-34: the unreachable refill of the poll basis removed; W3-36: `u_base`
  removed.
- `ESSearch.__init__` no longer calls `logging.basicConfig` ("Found while
  verifying").
- The records and descriptions: the comment of `contraints_check` on its
  order (W3-2); the comments of `ucov` and ES-wcm, which call the scatter
  weighted (W3-3); `AGENTS.md` on `poll_scale` and the description of
  `gp_rescale_poll` (W3-25); the description of `tol_poi` (W3-28); the
  description of `sloppy_improvement` (its floor at `tol_fun`) and the
  docstring of `_get_target_from_gp_` ("Found while verifying").

**Fix, moving results**, in this order, each ending in a population
comparison on Linux against the end of the step before, the first against
`population_linux_wave2_20260926`, whose environment this sandbox has (the
fingerprint at `8aecb6a` is its `dc11118754b18b47`); the fingerprint
recorded at every commit:

1. The ES search: W3-4 (`floor(mu)` best points), W3-5 (the selection mask,
   and `test_search_selection_mask`'s golden sum), W3-15 (stable sorts),
   one comparison of the batch, its steps compared one by one if it is
   flagged.
2. W3-1: `contraints_check` removes the points already evaluated, as
   `uCheck.m`; `test_incumbent_constraint_check` corrected; after W3-9. The
   default suite, and a suite that reaches the search's repeats, with an
   optimum on a bound; the seeded tests checked over their seeds with
   `dev/scripts/tolerance_sweep.py` if one fails (`dev/TODO.md`'s item
   closes with it).
3. W3-6: the hedge's expected reward with φ(γ); KD-B3-3 corrected. The
   default suite, whose five noisy configurations it reaches.
4. W3-19: `p_less` over the D largest probabilities, sorted; a unit test.
5. W3-29 (a): after a poll that moves the incumbent, the search rebuilds
   the local GP at each pass until a poll that does not move, as MATLAB;
   the changelog's "Rebuilds of the local GP", the comment at
   `bads.py:1734-1736` and W1-2's row in `verification/wave1.md` corrected.
   Long runs reach it; if the default suite does not, a note says so.
6. W3-14, if its fingerprint moves (above).
7. W3-24 (PI: (b)). The poll draws LTMADS directions, a departure from
   MATLAB. BADS's meshes already follow LTMADS's relation between the poll
   size and the mesh size (the locked search mesh is `2^(2k-10)` at the
   poll mesh `2^k`), and `pollMADS2N.m:7` inverts the ratio that bounds the
   basis. So `poll_mads_2n` takes the bound `n_max = max(1,
   round(mesh_size / search_mesh_size))`, `2^(10-k)` at default, and the
   poll vectors are the basis times `mesh_size / n_max` (the search mesh
   size at default): the diagonal step keeps the length `mesh_size`, the
   lower-triangular entries tilt the directions, and the poll points lie on
   the search mesh. This is the variant that B4-I measured
   (`scripts/wave3/B4_internal/check12_ltmads_variant.py`). The division by
   `poll_scale` and the multiplication back stay (W3-25), and a new basis is
   drawn at each poll as now (LTMADS itself keeps one direction per mesh
   index; not adopted). With it: the docstring of `poll_mads_2n`, the
   poll's description in `README.md` and `docsrc/source/index.rst` ("steps
   in one direction at a time"), `test_poll_mads.py`, a sheet entry, an
   entry in `matlab_side_defects.md` (the inverted ratio), and a changelog
   line under "Changed". Gate: a population comparison at default, as the
   last step of the pass so that it measures this change alone, with a
   check on nonsmooth targets whose descent direction is diagonal (B4-I's
   ridges, over more seeds) and on W2-37's thin band; if the comparison
   flags a worsening, the row comes back to the PI, as W2-25's rule was.

**Keep, and record:**

- W3-3 (a): MATLAB's unweighted scatter stays, as a shared observation in
  `matlab_side_defects.md`.
- W3-12: the search ranks by the restored GP after a failed rebuild;
  KD-B5-2 extended.
- W3-21 (a): the target's posterior recomputed under `hyp_best` stays;
  KD-B4-2 settles it (MATLAB's hybrid is no prediction under one set of
  hyperparameters).
- W3-26: the level-0 poll's GP without the poll's points, as MATLAB, a
  shared observation in `matlab_side_defects.md`.
- W3-28: MATLAB's stop on an unreliable GP (its description above).
- W3-30: on the sheet, beside KD-B5-2.
- W3-35: `period_check`'s return, until periodic variables are ported
  (KD-B1-6).
- W3-16, W3-17, W3-37: no longer hold; W3-11, W3-32 and W3-38 close the
  survey's rows that wave 0 fixed.
- The survey's 11 rows of these slices are closed by this ledger; MATLAB's
  0/0 in the ES scale at `n_search_iter` ≥ 3 ("Found while verifying")
  goes in `matlab_side_defects.md`.

**Out of this pass:** the copy of the GP and the recomputed posterior of
`_get_target_from_gp_` at every search step, whose target nothing reads at
default (time only), and the cause of the frequent zero predictive SDs at
level 0: `dev/TODO.md` lines. The first-iteration note of the B3 brief
needs no change.

**After the gates (PI, 2026-09-27)**, on the orchestrator's report of
W3-24's flagged gate ("Fix pass"):

- W3-24: revert the LTMADS directions and keep MATLAB BADS's coordinate
  poll, with a docstring of `poll_mads_2n` that says what its basis is; the
  inverted ratio goes into `matlab_side_defects.md` as a shared
  observation, with the gate's evidence, and KD-B4-3 goes.
- W3-39: refuse an `accelerate_mesh_steps` that is not a positive integer
  when `BADS` is created (the orchestrator's proposal).
- W3-40, the degenerate prior of the length scales on two points that
  W3-24's gate exposed (PI: "can we still fix this?"): fixed in this pass,
  by the orchestrator's proposal of keeping the previous prior, as a
  rebuild on targets without spread keeps its own (KD-B6-2).

## Fix pass

Done (2026-09-27). As in waves 1 and 2: the fixes go on
`dev-port-review-w3`, from the brief's commit `326aefe`; each is made by a
fix agent, a fresh Opus agent with a git worktree of its own and the brief
`../briefs/wave3_fix_common.md`, one commit per row with its regression
test; the orchestrator reviews each diff, cherry-picks it, adds the
changelog lines and runs the whole fast suite and the fingerprint after
every pick. Four agents: A, the search step, the hedge and the LCB (W3-11
with W3-18, W3-7, W3-10, W3-6); B, the target, the options and the poll's
dead code (W3-20, W3-23, W3-31, W3-33, W3-34, W3-36, the records of W3-25
and W3-28, W3-14); C, the ES search and `contraints_check` (W3-9, W3-8,
W3-13, the root logger, the comments of W3-2 and W3-3, W3-4, W3-5, W3-15,
W3-1); D, the poll (W3-22, W3-27, W3-19, W3-29, W3-24). The orchestrator
made the three commits that the PI's rulings after the gates asked for
(the revert of W3-24, W3-39, W3-40). The agents' reports are
in `../fixes/`, their scripts in `scripts/wave3/fix_<agent>/`; the hashes
they cite are those of their branches, and the fingerprints in their
commit messages are those at their branches' commits, from `326aefe`.
`dev-next` gained the doublecheck of wave 2 (`68d4516`) during the pass,
merged into the branch at `4f50376`.

The fingerprint is that of `dev/scripts/fingerprint.py` at the commit on
the branch (Linux, gpyreg 1.3.3 from the clone at `98ab5a4`, one BLAS
thread), computed again by the orchestrator at every commit of the pass
that changes the package (`scripts/wave3/orchestrator/fp_all.out`, where
W3-14's `1f7c8ee` comes last, computed after the others, and `388d879`, a
commit of records with W3-14's package, stands in its place). The populations are the
`default` suite × seeds 0-29, each run from a worktree at its commit and
compared with the one before, and for W3-1 and W3-24 the `geometry` suite
(`8824c9e`: a sphere with its minimum on a lower bound, nonsmooth ridges
along the diagonal, a thin feasible band) × seeds 0-29 before and after;
the comparisons are in `wave3_fixpass/`, the orchestrator's scripts in
`scripts/wave3/orchestrator/`. Every population reads gpyreg from the
clone at the tag `v1.3.3`; its records give gpyreg's version as
`1.3.4.dev10+gd96d0d9f7`, the metadata of the venv's editable install of
`../gpyreg`, with the module path of the clone.

| Row | Commit | Fingerprint | Gate and outcome |
|---|---|---|---|
| W3-11, W3-18 | `4388e6d` | `dc11118754b18b47` | fingerprint unchanged; an empty search set decays the hedge's gains |
| W3-7 | `4d357e4` | `dc11118754b18b47` | fingerprint unchanged; a run with `hedge_gamma=0` completes |
| W3-10 | `599115b` | `dc11118754b18b47` | fingerprint unchanged; CI's smoke run passed |
| W3-20 | `5f31837` | `dc11118754b18b47` | fingerprint unchanged |
| W3-23 | `dac062e` | `dc11118754b18b47` | fingerprint unchanged |
| W3-31 | `ec1b2d0` | `dc11118754b18b47` | fingerprint unchanged |
| W3-33 | `43ee8ed` | `dc11118754b18b47` | fingerprint unchanged |
| W3-34 | `01ee524` | `dc11118754b18b47` | fingerprint unchanged |
| W3-36 | `e4b3bca` | `dc11118754b18b47` | fingerprint unchanged |
| W3-25, W3-28 and the descriptions | `fd8641d` | `dc11118754b18b47` | none (`AGENTS.md`, the descriptions, a docstring) |
| W3-14 | `1f7c8ee` | `dc11118754b18b47` | fingerprint unchanged, but it moves the default suite (batch 1, below) |
| W3-9 | `a77d95d` | `dc11118754b18b47` | fingerprint unchanged |
| W3-8 | `c788617` | `dc11118754b18b47` | fingerprint unchanged; CI's smoke run passed |
| W3-13 | `4865fad` | `dc11118754b18b47` | fingerprint unchanged |
| the root logger | `d0c7178` | `dc11118754b18b47` | fingerprint unchanged |
| W3-2, W3-3 | `7e09887` | `dc11118754b18b47` | none (comments) |
| W3-22 | `f595f1b` | `dc11118754b18b47` | fingerprint unchanged; the suite shows 12 more warnings, gpyreg's log of a zero width in three tests of degenerate training sets, which an earlier test's process-wide `np.seterr` had hidden |
| W3-27 | `a1bf658` | `dc11118754b18b47` | fingerprint unchanged; CI's smoke run passed |
| *batch 1* | `a1bf658` | `dc11118754b18b47` | the default suite against `population_linux_wave2_20260926`: no flag in 54 tests, but 31 of the 540 runs change, 29 of them ending at other points, all 6-D, 10-D or noisy (`sphere_D10` 7, `ellipsoid_D10` 7, `ellipsoid_D6` 6, `rosenbrock_D6` 6, `multisensory_s1_D6_homo` 3, `ackley_D6` 1, `ellipsoid_D3_homo` 1). W3-14 alone moves them: each of the 31, run again, equals the reference at W3-14's parent `fd8641d` and batch 1 at `1f7c8ee` (`wave3_fixpass/w3-14_attribution.txt`). The ES search's candidates fall on halves of the search grid once its mesh is fine (at `2^-42`, `x / tol` is of the order of `1e12`, up to `4.4e12` at `|x| = 1`, where the fraction of a double comes in steps of `2^-13` to `2^-11`), which W3-14's agent had excluded by reading and the fingerprint's small runs do not reach. So this gate is W3-14's population comparison, which its ruling asks for when it moves: no flag (`wave3_fixpass/batch1_vs_reference.md`) |
| W3-4 | `81c6a15` | `fd21e5d0c558f2f0` | with the batch |
| W3-5 | `115a922` | `3a6c31fe4f430b0b` | with the batch |
| W3-15 | `c276d79` | `3a6c31fe4f430b0b` | the ES batch against batch 1: no flag in 54 tests, every run changed (`wave3_fixpass/es_vs_batch1.md`); so its steps were not compared one by one |
| W3-1 | `149d528` | `3a6c31fe4f430b0b` | the default suite against the ES batch: no flag in 54 tests, 325 runs changed (`w3-1_vs_es.md`); the geometry suite before and after: no flag in 21 tests, 124 of 210 runs changed (`geometry_w3-1_vs_es.md`). It removes every repeated evaluation: over seeds 0-29, 137 of the 1411 evaluations of `edgesphere_D2`, 94 of 6840 of `edgesphere_D3_homo`, 13 of `sphere_band_D3`, 3 and 6 of the ridges and 100 of the 8800 of `ellipsoid_D3_hetero` repeated an earlier one before it, none after (`w3-1_repeats.txt`). `ellipsoid_D3_hetero`'s median error moves from 0.43 to 0.36, unflagged. No seeded test failed |
| W3-6 | `d79ab75` | `360971bf1f0ba6cb` | the default suite against W3-1: no flag in 54 tests; 128 runs changed, those of the five noisy configurations; the fraction solved falls on `ellipsoid_D3_homo` (0.63 → 0.50) and holds or moves by one run elsewhere (`w3-6_vs_w3-1.md`) |
| W3-19 | `8e28124` | `360971bf1f0ba6cb` | the default suite against W3-6: no flag in 54 tests; 5 runs changed (`w3-19_vs_w3-6.md`) |
| W3-29 | `0b7add3` | `360971bf1f0ba6cb` | the default suite against W3-19: no flag in 54 tests; 277 runs changed, so the default suite reaches it (`w3-29_vs_w3-19.md`) |
| W3-24 | `869a033` | `f5af904cfcb6b73c` | flagged, and reverted by the PI's ruling (below): the default suite against W3-29 flags `ellipsoid_D6` (a higher error) and `sphere_nonbox_D3` (fewer evaluations), and the geometry suite flags `edgesphere_D2` and `edgesphere_D4` (fewer evaluations) and `sphere_band_D2` and `sphere_band_D3`, whose runs crash for the first time (`w3-24_vs_w3-29.md`, `geometry_w3-24_vs_w3-29.md`) |
| merge of `dev-next` | `4f50376` | `f5af904cfcb6b73c` | fingerprint unchanged; the doublecheck of wave 2, whose one change of behaviour, an `f_vals` without a finite value, no run of the benchmark reaches; the `default` and `geometry` suites unchanged |
| revert of W3-24 | `b03a320` | `360971bf1f0ba6cb` | W3-29's fingerprint again; `poll_mads_2n`'s docstring says what its basis is, a coordinate poll at default |
| W3-39 | `5d711bf` | `360971bf1f0ba6cb` | fingerprint unchanged |
| W3-40 | `a14524d` | `360971bf1f0ba6cb` | fingerprint unchanged; on W3-24's code with the change, the three runs of the thin bands that crashed end at errors of 1e-7, 3e-10 and 1.2e-5 (`w3-40_crashed_runs.txt`) |
| *head* | `a14524d` | `360971bf1f0ba6cb` | the default and geometry suites against W3-29: every run identical (every field but the wall time; `head_vs_w3-29.md`, `geometry_head_vs_w3-29.md`), so the merge, the revert, W3-39 and W3-40 reach no run of either suite; this population is the new Linux reference, `population_linux_wave3_20260927`, whose comparison with `population_linux_wave2_20260926`, the net change of the pass, flags nothing in 54 tests, and whose null check flags nothing in 36 |

- **Choices within the rulings**, made by the orchestrator on the agents'
  reports:
  - W3-14 rounds exactly: `np.modf` splits `x / tol` into its integer and
    fractional parts, and the integer part moves away from zero when the
    fractional part is at least one half in magnitude. The agent's
    `sign(q) * floor(|q| + 0.5)` took 0.49999999999999994 to 1, and a first
    exact variant, `q - trunc(q)`, warned on an infinite bound (`inf -
    inf`), which eight tests of the suite reach; changed when
    cherry-picking, with a test of the largest double below one half.
  - W3-24 keeps `poll_mads_2n`'s return type: the function returns the
    basis in units of the poll size (the LTMADS matrix divided by `n_max`,
    exact at default, where `n_max` is a power of two), and the poll's
    vectors stay `B_new * mesh_size * poll_scale`, so that `AGENTS.md`'s
    description holds and no "Upgrading from" line is needed; the agent's
    first version returned `(B_new, n_max)` (its report's rework, the
    fingerprint unchanged by it). The comment on the permutation says that
    it permutes the rows only, the same set of directions as MATLAB's
    permutation of rows and columns.
  - W3-9 keeps reproducing after an emptied generation, from the kept
    candidates at the unchanged scale, rather than ending the ES search;
    the two agree at the default `n_search_iter` of 2. Its warning stays at
    WARNING, as the ruling's "reworded" reads; a thin band can log it a few
    times per run, which the PI may prefer at DEBUG.
  - W3-13 also removes `ESSearchWM`'s `active_flag` and its branch, which
    only `ESSearchCMA` switched on (the broken lines that the row cites);
    `pybads.search.ESSearchCMA` has an "Upgrading from" line, since a
    script that imports it stops.
  - W3-10 refuses zero, negative, boolean and complex values of
    `sqrt_beta`, which 1.1.0 accepted as a NumPy scalar or a one-element
    array: an "Upgrading from" line.
  - W3-27 makes the fallback live when every acquisition value is NaN (a
    random choice with a warning), where MATLAB's `min` returns the first
    index and its fallback cannot fire either; no such case was observed.
  - W3-29 sets the flag only at the end of the poll, moved or not, so a
    poll that makes no rebuild of its own (an empty poll set, or the
    budget spent) no longer cancels a pending rebuild of the search, as
    MATLAB's emptied posterior does not.
  - W3-1 keeps, within one bin of `tol_mesh`, the first candidate in input
    order, where MATLAB's `setdiff` keeps the smallest; the difference is
    below `tol_mesh / 2`.
  - The tests of W3-9 and W3-11 reach the changed code by other means
    than the configurations their rows name, a thin band and a
    `non_box_cons` that empties a set: W3-9's with a constraint that
    refuses every candidate at its second call, W3-11's with a hedge
    patched to return an empty set (recorded by the doublecheck).
  - W3-14 moved results although its fingerprint did not, so its
    population comparison is batch 1's, the first gate of the pass, where
    the ruling put it sixth; each gate still measures one step.
  - Conflicts when cherry-picking, all of tests appended at the end of the
    same file, resolved by keeping both (W3-14, W3-9, W3-8, W3-27 and W3-6,
    over tests of A and C). A union merge driver, tried first on the test
    files, dropped common lines of the two sides; the picks it made were
    reset before any gate ran, and the conflicts were resolved by keeping
    both sides in full wherever their common base was empty
    (`scripts/wave3/orchestrator/resolve_appends.py`).

- **W3-24's flagged worsening, and its revert.** Against W3-29
  (`w3-24_vs_w3-29.md`, `geometry_w3-24_vs_w3-29.md`, the medians of
  every step in `medians_default.md` and `medians_geometry.md`):
  - the default suite flags `ellipsoid_D6`, a higher error (median
    1.1e-7 → 4.3e-7; paired log10 ratio +0.67 [+0.15, +1.12]; KS 0.53, p
    Holm 0.016), every run still solved, and `sphere_nonbox_D3`, fewer
    evaluations (97 → 88; KS 0.53, p Holm 0.016) with an error of 6.1e-6 →
    1.0e-5, solved throughout. Unflagged, the deterministic errors rise on
    most configurations, far below their tolerances (paired ratios +0.10
    to +0.40 on `ackley_D6`, `sphere_D10`, `multisensory_s1_D6` and
    `timing_D5`, every run solved); among the noisy ones `sphere_D3_hetero`
    improves (solved 0.40 → 0.60, ratio -0.19 [-0.35, +0.01]),
    `ellipsoid_D3_homo` falls (solved 0.73 → 0.57) and
    `multisensory_s1_D6_homo` loses two runs (1.00 → 0.93);
  - the geometry suite flags `edgesphere_D2` and `edgesphere_D4`, fewer
    evaluations (47 → 45, 96 → 79) at lower median errors, and the two thin bands:
    `sphere_band_D2` gains 2 solved runs of 30 (W2-37's stall at `x0` ends
    the rest, as before) and 2 crashes, and `sphere_band_D3` falls from
    30 solved runs to 23, with 1 crash and 6 runs that end far off (the
    worst at an error of 74.7), in fewer evaluations (57.5 → 48 over the
    runs that did not crash). The
    ridges, the valley that W3-24 aims at, do not improve over these
    starts: `ridge_D2` 0.83 → 0.73 solved (the worst error 0.023 → 3.9),
    `ridge_D4` 0.87 → 0.83, the median errors unchanged by the tests;
  - the three crashes (`geometry_w3-24_crashes.txt`) are gpyreg's
    `ValueError` for a prior with a sigma of 0: when the tilted poll
    reaches the thin band, the local GP holds two distinct points, whose
    one pairwise distance makes the empirical prior of the length scales
    degenerate (`gaussian_process_train.py:360-382` at `869a033`, as
    MATLAB's `gpdefBads.m:240-251`): a latent defect of the GP layer that
    W3-24 exposed, fixed as W3-40;
  - the net change of the pass against `population_linux_wave2_20260926`
    flags nothing in 54 tests at W3-29 (`0b7add3`, `w3-29_vs_reference.md`)
    and four configurations at W3-24 (`869a033`, `w3-24_vs_reference.md`):
    `ellipsoid_D6` and `multisensory_s1_D6`, higher errors, every run
    solved, and `ellipsoid_D3_unbounded` (133 → 149) and
    `sphere_nonbox_D3` (101 → 88), the number of evaluations.

  The orchestrator reported it to the PI, who ruled to revert it and keep
  MATLAB BADS's poll ("After the gates"): `b03a320`, whose fingerprint is
  W3-29's.
- **Changelog.** Every row that a user can notice has a line under
  `Unreleased`, written by the orchestrator when cherry-picking, from the
  agents' proposals: "Changed" for W3-10, W3-31 and W3-39 (stricter
  interfaces) and W3-13 (a removal); "Fixed" for the rest. A line in
  "Upgrading from 1.1.0" for W3-10, W3-31, W3-39 and W3-13. W3-11 and W3-9
  extend "Search without a candidate", W3-25 and W3-28 "Descriptions of the
  options", W3-31 and W3-39 "Checks of `max_fun_evals`,
  `improvement_quantile` and `accelerate_mesh_steps`", and W3-29 replaces
  "Rebuilds of the local GP", all unreleased. W3-24's entry went with its
  revert. W3-2, W3-3, W3-23, W3-33, W3-34 and W3-36 have no line. The
  entries of W3-14 and W3-1 give what their gates measured (`bd110f0`,
  `6281663`).
- **The suite** passes at every commit of the pass, 457 tests at the head
  (`a14524d`), with the fingerprint `360971bf1f0ba6cb`; the pre-commit
  hooks pass on the whole tree. CI's smoke run passed at every push that
  touched the package (`599115b`, `c788617`, `a1bf658`, `869a033`,
  `dd78136`, `a14524d`).

**Found while fixing** (2026-09-27), reported by the fix agents outside
their rows (the letter names the agent) or met by the orchestrator, and not
fixed in this pass; each is left to the PI or to the wave of the slice that
owns its code.

- **W3-14's premise** (the orchestrator): halves are frequent among the
  search's candidates at a fine mesh (batch 1's row); the commit message
  of `1f7c8ee` says otherwise.
- **Warnings that `np.seterr` hid** (the orchestrator, W3-22): gpyreg's
  `get_bounds_info` takes the log of a zero width on a degenerate training
  set (`covariance_functions.py:476-479` at v1.3.3), the warnings that
  `dev/TODO.md`'s "The GP on a one-point training set" names.
- **The search and the LCB** (A): the empty-set branch of `_search_step_`
  sets `search_dist = 0`, an `int`, read only by `_update_search_stats_`;
  `acq_fcn_lcb`'s summary line says that it retrieves a point, and it
  computes an unused `n`; `update_hedge`'s docstring speaks of a
  probability of improvement; `hedge_gamma` is not checked (above `1/n`
  the hedge's probabilities invert, above `1/(n-1)` they turn negative);
  `sqrt_beta` is checked at the first search, not when `BADS` is created,
  and the value that a callable returns is not checked.
- **The target and the poll** (B): the final estimate of a noisy run sets
  `u`, `yval`, `fval` and `fsd` on the object but not in `optim_state`, so
  `output_fcn`'s `"done"` call receives the last iteration's values (its
  `x` is right); `_poll_step_`'s docstring calls an SD a variance, and the
  main loop discards the values it returns [the doublecheck: its first
  wording, which report B repeats, said that the step does not return
  them]; `grid_functions.py` imports matplotlib's `axis` unused.
- **The ES search and `contraints_check`** (C): `contraints_check`'s
  docstring says it returns an incumbent, and its module imports an unused
  `Value`; `ESSearch.__call__`'s docstring has placeholders; with
  `n_search_iter = 0` a run stops with `ZeroDivisionError` at its first
  search (`search_hedge.py:58`), an `ESSearch` built directly returns its
  `np.empty` placeholders, and no check refuses a value below 1; W3-9's
  warning is at WARNING, which a thin band can log a few times per run
  (DEBUG, if the PI prefers) [the doublecheck: report C's other tests of
  `test_search.py` that draw from NumPy's global stream do not exist; the
  one C corrected was the only one].
- **The poll** (D): a 4-D ridge started on its valley stalls at `x0`
  before and after W3-24, since the only descent direction is the exact
  diagonal; Fig. 1 of the documentation (`docsrc/source/_static/bads-cartoon.png`,
  in `README.md` and `index.rst`) draws the poll's steps anisotropic, which
  `poll_scale` does not make them (W3-25) [the doublecheck: the steps are
  equal in `u` and scale with the plausible box in `x`, so a figure drawn in
  `x` can show unequal steps; no defect is established]; the main loop discards the GP that `_poll_step_`
  returns, which works because the GP functions change it in place;
  `poll_mads_2n.py` imports `GP` unused.

## Doublecheck

Done (2026-09-27), after #77 was squash-merged into `dev-next` as
`0d866e8`, as for waves 1 and 2 (PI): four fresh read-only Opus reviewers,
of (a) the fixes of B3 (W3-1 to W3-15 and the root logger, with the gates
of batch 1, the ES batch and W3-1), (b) the fixes of B4 (W3-19 to W3-36,
W3-39 and W3-40, W3-29's gate, and W3-24 with its gate and its revert), (c)
the user-facing documentation, and (d) the records, gates and tooling. Each
checked that every row implements its ruling, that the comparisons with
MATLAB BADS that the rulings rest on hold, and that every statement is true
of the code at `0d866e8`. Their briefs are in
`../briefs/wave3_doublecheck.md`, their reports, saved verbatim, in
`wave3_doublecheck_B3.md`, `wave3_doublecheck_B4.md`,
`wave3_doublecheck_docs.md` and `wave3_doublecheck_records.md`, and their
scripts and outputs under `scripts/wave3/doublecheck/`. The clone they read
was shallow, from `d76fc6d` (2023-02-06); none of their checks needed older
history, which was fetched afterwards to date the row of W3-40. The
orchestrator ran the suite and the fingerprints on Linux (the cloud session
of the pass: Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, the gpyreg 1.3.3
clone), the PI the fingerprints on Windows, and the orchestrator checked
each finding it took against the code, MATLAB BADS at `74919c0` or 1.1.0.

**What holds.** Every row implements its ruling, and the comparisons with
MATLAB BADS that the rulings rest on hold, except the rounding of
`contraints_check`'s bins (below, "Left"). Transcriptions of MATLAB's
functions gave the port's results on the same inputs: `uCheck.m` on grids
no finer than its bins (3000 of 3000 random sets, 409 of 409 calls in eight
runs), `searchES.m` and `ESupdate.m` (64 of 64 searches, and every mask of
the sizes tried), the update of `acqPortfolio.m` (600 states; the decay of
an empty set over 200), `force2grid.m` (479,936 doubles and the special
cases), `p_less` of `bads.m:862-872` (16,800 random sets and 169 poll
steps), and the rebuild flags of `bads.m` (`pollmoved_flag` and line 1049)
over the events of 26 runs, which the step before W3-29 fails. The revert
of W3-24 leaves the code, the tests and the user documents as they were at
W3-29, except the docstring that the PI's ruling asks for. The counts and
numbers of this ledger and of `wave3_fixpass/`, recomputed from the
committed records, match, except those corrected below; `population.py
compare` and `summary` on the two committed Linux references reproduce the
new reference's `comparison.md`, `null_check.md` and `summary.md` byte for
byte; the commit of the `geometry` suite leaves the `default` suite
unchanged; and the orchestrator's scripts do what the records say. The
suite passes at `0d866e8` (457 tests). The fingerprints of `fp_all.out`
recompute at the pass's key commits, and on Windows (Python 3.12.6, NumPy
2.5.3, SciPy 1.18.1, the gpyreg 1.3.3 clone) they follow Linux's pattern
except at W3-15:

| Commit | Step | Linux, one BLAS thread | Linux, default | Windows, default | Windows, one BLAS thread |
|---|---|---|---|---|---|
| `8aecb6a` | the revision of wave 3 | `dc11118754b18b47` | `dc11118754b18b47` | `6825faa249798851` | `8d8552d1f5bee1e6` |
| `a1bf658` | batch 1 | `dc11118754b18b47` | — | `6825faa249798851` | `8d8552d1f5bee1e6` |
| `81c6a15` | W3-4 | `fd21e5d0c558f2f0` | — | `9dd07dc8abf18dbe` | `a3005bc9db4f522b` |
| `115a922` | W3-5 | `3a6c31fe4f430b0b` | — | `b88911b5c7d77783` | `76041bc200918bb9` |
| `c276d79` | W3-15 | `3a6c31fe4f430b0b` | — | `fbb6d990d263425a` | `60db94c1c79bd844` |
| `149d528` | W3-1 | `3a6c31fe4f430b0b` | — | `fbb6d990d263425a` | `60db94c1c79bd844` |
| `d79ab75` | W3-6 | `360971bf1f0ba6cb` | — | `7779b81cecfb120a` | `ac49960f71e37c97` |
| `8e28124` | W3-19 | `360971bf1f0ba6cb` | — | `7779b81cecfb120a` | `ac49960f71e37c97` |
| `0b7add3` | W3-29 | `360971bf1f0ba6cb` | — | `7779b81cecfb120a` | `ac49960f71e37c97` |
| `869a033` | W3-24 | `f5af904cfcb6b73c` | — | `9cbe1a2060b82d1b` | `44ffa02e820f5de7` |
| `b03a320` | the revert of W3-24 | `360971bf1f0ba6cb` | — | `7779b81cecfb120a` | `ac49960f71e37c97` |
| `0d866e8` | `dev-next` after #77 | `360971bf1f0ba6cb` | `360971bf1f0ba6cb` | `7779b81cecfb120a` | `ac49960f71e37c97` |

On Linux, with four cores, the default number of BLAS threads gives the
hash of one thread at both commits measured; on Windows the two settings
differ at every commit. On Windows W3-15 moves the hash at both settings,
where Linux's stays: its one change, stable sorts in the ES search,
reorders tied candidates, which leave the histories of the six runs
unchanged on Linux (its commit message) and change them on Windows.
NumPy's default sort orders ties unlike the stable sort on both machines
(300 of 300 random arrays of three distinct values on Linux, whose NumPy
finds AVX-512; the same on Windows, AVX2); whether it orders them
differently on the two CPUs ("Found while verifying") is not established,
and since W3-15 no sort of the search depends on it. The fingerprint that
W3-15 leaves unchanged is Linux's: so the table of "Fix pass", its commit
message, and fix agent C's report ("At W3-15 and W3-1 it stays
`3a6c31fe4f430b0b`").

**Fixed in the commit that adds this section**, whose fingerprint is
`360971bf1f0ba6cb` (Linux, one BLAS thread) and whose suite passes (461
tests):

- The code, where a statement required it: an `improvement_quantile` that
  is a string, a complex number or an array of several values raises
  W3-31's `ValueError`, as the Raises section of `BADS` says, where it
  raised `TypeError` or NumPy's error on an ambiguous truth value;
  `test_improvement_quantile_outside_zero_one_is_refused` takes the three.
- The changelog: 1.1.0's failures with an `accelerate_mesh_steps` below 1
  or not an integer (`IndexError` for 0, from the second iteration on;
  `inf` ran without the accelerated reduction) and with an
  `improvement_quantile` of 1 in a noisy run (the incumbent moved at most
  searches); the scale of the ES search (faster than MATLAB's from the
  third generation, growing where MATLAB's shrinks from about the fifth);
  the hedge's excess reward (2.5 times or more, 424 times at three
  standard deviations); the rounding's example (within wider hard bounds);
  batch 1's 31 changed runs, 29 of them ending at other points; ES-wcm's
  ⌊μ⌋ (the best half of the training points); the counts of W3-1, measured
  with the release's other changes; W3-9's kept candidates; W3-40's prior,
  kept whole; and the check of `improvement_quantile` above.
- Docstrings and descriptions: `acq_fcn_lcb`, rendered on the site (its
  summary, and the numpydoc form of its sections); `_poll_step_` (the
  coordinate poll, an SD); `_get_target_from_gp_` (the branch that makes
  no prediction); `poll_mads_2n` (the shape of `poll_scale`, the path of
  the record it cites); `contraints_check` (what it returns, and a comment
  on the rounding of its bins); `update_hedge`; the example of
  `ESSearchHedge`; `ESSearch.__call__`. The descriptions of
  `improvement_quantile` (its range), `search_acq_fcn` (`sqrt_beta`), and
  of `poll_method`, `skip_poll`, `search_improve_frac`, `search_optimize`
  and `poll_acq_fcn`, which do nothing, and `acq_hedge`, which stops a run.
- The sheet: wave 3's deliberate differences, which it lacked and which
  slice O reads, KD-B3-7 (`sqrt_beta`, W3-10), KD-B3-8 (`hedge_gamma = 0`,
  W3-7), KD-B4-4 (the target's fallback, W3-23), KD-B4-5 (acquisition
  values that are all NaN, W3-27) and KD-B4-6 (the checks of W3-31 and
  W3-39, with `inf`, which MATLAB runs with); KD-B3-3 (`acq_hedge=True`
  stops a run), KD-B3-5 (`EvalImprovement`), the title of KD-B4-1 (MADS
  2N), and the numbers of KD-B4-2 and KD-B5-2. `matlab_side_defects.md`:
  `Nsearchiter`, and `inf` in W3-39's entry. The survey's row of
  `contraints_check`: the rounding of its bins.
- This ledger: the rows W3-8 (the scale), W3-12 (the offsets), W3-21 (the
  decisions of levels 0 and 1), W3-29 (runs of fewer than 200 evaluations
  change too), and a row for W3-40; B3-K13 in the notes; batch 1's row (29
  of the 31 end at other points; the spacing of doubles); the range of
  `fp_all.out`; the evaluations of `sphere_band_D3` and the errors of
  `edgesphere_D2` in W3-24's paragraph; the tests of W3-9 and W3-11 among
  the choices; the items of agents B and C under "Found while fixing" that
  do not hold (`_poll_step_` returns its values; no other test of
  `test_search.py` draws from the global stream; `n_search_iter = 0` stops
  a run with `ZeroDivisionError`); a note on Fig. 1; a stray bullet.
- `dev/TODO.md`: the two items of B1 and B2 that the pass fixed
  (`hedge_gamma`'s description, W3-7; `optim_state` after a re-estimate,
  W3-33) are removed, and a line holds the minor items of B3 and B4.
- The other records: `dev/README.md` (the `geometry` suite); the new
  reference's README (W2-45 to W2-47 in its provenance; the setting of its
  fingerprint); the port review's README (the fix brief of wave 3, the
  sandbox's paths in the orchestrator's scripts, this doublecheck's
  records); the plan ("Wave 4 pickup", which named only W3-6 of what wave 3
  changed in slice O's code; the worklog's count of the survey rows that
  wave 0 fixed, four; the Close item, whose Windows reference predates
  waves 0 to 3); and the pointer of `population_ellipsoid_hetero_linux_20260925`
  to the `TODO.md` item that W3-1 closed.

**Left**, and where each goes:

- For the PI, a change that moves results (`wave3_doublecheck_B3.md`, F1):
  `contraints_check` bins the candidates and the evaluated points with
  `np.round`, which takes a half to the even integer, where `uCheck.m`'s
  `round` takes it away from zero. From the poll mesh `2^-6`, the search
  grid (`2^-22` and finer) is finer than the bin, `tol_mesh / 2 = 2^-20`,
  and a quarter of its coordinates fall on halves of a bin, so that other
  candidates are merged into one bin, or removed as evaluated, than in
  MATLAB BADS, within `tol_mesh / 2`. In seeded runs of at most 200
  evaluations, the output of the ES search's check changes in 2 to 116 of
  32 to 202 calls, and the point it returns in 0 to 3 of 25 to 101
  searches; whole runs evaluate other points in 5 of 9, with the same
  final states. The checks of W3-1, on grids never finer than the bin,
  missed it. A fix
  bins with `force_to_grid`'s exact rule, gated by the `default` and
  `geometry` suites, seeds 0-29, against `population_linux_wave3_20260927`
  and the `geometry` population of `a14524d`. The ES search splits its
  first population with the same rounding (`es_search.py:33-35`), which
  differs from MATLAB's only when `n_search / n_search_iter` is odd, not at
  default. The comment in `contraints_check` and the survey's row say so;
  the sheet does not, until the PI rules.
- For the PI, changes that move no results: `acq_hedge=True` stops a run
  with `UnboundLocalError` at its first improving search, as in 1.1.0
  (KD-B3-3), which a refusal when `BADS` is created would make a clear
  message; `accelerate_mesh_steps=inf`, which MATLAB BADS and 1.1.0 run
  without the accelerated reduction, is refused since W3-39 (KD-B4-6),
  where accepting it, as the check of `max_fun_evals` accepts `inf`, is the
  alternative. Both are in `dev/TODO.md` until the PI rules.
- For wave 4, slice O, whose branch was cut from `ed82ec0`, before this
  commit: the corrected "Wave 4 pickup", the new entries of the sheet
  (KD-B3-7, KD-B3-8, KD-B4-4 to KD-B4-6), and O's items of "Found while
  fixing" (A): `hedge_gamma` is not checked (above `1/n` the hedge's
  probabilities invert, above `1/(n-1)` they turn negative), `sqrt_beta` is
  checked at the first search, not when `BADS` is created, and the value
  that a callable returns is not checked, and `acq_fcn_lcb` computes an
  unused `n`. Nothing of this doublecheck belongs to B7.
- For `dev/TODO.md` ("Minor items of slices B3 and B4"): the other items
  of "Found while fixing", `n_search_iter` below 1, `force_to_grid`'s
  missing docstring, and the items for the PI above.
