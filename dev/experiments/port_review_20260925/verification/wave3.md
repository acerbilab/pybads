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
| W3-8 | B3-I F7, B3-C F7 | the fraction of new candidates behind the ES's scale update is miscounted (`es_search.py:177-192`): `z_idx[0:ntest+1] > nold` looks at one entry too many and misses index `nold`, and from the third generation the untrimmed pool counts older rows as new (0.56, 0.77, 0.87, 0.93 against MATLAB's 0.56, 0.43, 0.39, 0.37, `searchES.m:170-193`). The update runs only for 1 < i < `n_search_iter` (1-based), never at the default 2; from 4 on the scale grows where MATLAB's shrinks | confirmed port discrepancy | no (`n_search_iter` ≥ 3; material from 4) | never agreed (`c7c88ab`) | — | fix, with a guard for `ntest == 0`, which W3-9's fix makes reachable (MATLAB gets 0/0 there) | fingerprint; a test with `n_search_iter = 4` |
| W3-9 | B3-I F9, B3-C F6, B3-K10 | when every candidate of a later ES generation is removed, the fallback for a failed acquisition sets `z_candidates = rng.random(0)` (`es_search.py:170-175`), which discards the earlier candidates' values, and the ES returns an empty set with the warning "random search is performed", although none is; MATLAB keeps `zold` and returns the best earlier candidate (`searchES.m:168-182`). B3-K10's warning at D = 3 on a thin band is this path: every emptied generation was the second, whose size is the number of first-generation survivors (12 of 2048 in one call), on both sides; the GP's few points are incidental | confirmed port discrepancy | no (a `non_box_cons` that empties a later generation; W3-1's fix can too) | never agreed (`c7c88ab`); an `IndexError` until `0c56d86`, an empty set since | — (B3-K10) | fix: skip an empty generation and keep the candidates; reword or remove the warning (a NaN acquisition, the real failure, is never caught); before W3-1 | fingerprint; a test with a thin band |
| W3-10 | B3-I F8, B3-C F9, B3-K11 | `acq_fcn_lcb` refuses a plain number as `sqrt_beta` (`acquisition_functions/acq_fcn_lcb.py:42`: `(2.0).size` raises `AttributeError`), and a non-finite value and a schedule's name, which `acqLCB.m:16-18` accepts; a NumPy scalar works. The docstring calls the SD output a variance (`acq_fcn_lcb.py:27-28`, B3-K11) | confirmed port discrepancy; the docstring: confirmed, inert | no (a number as the second element of `search_acq_fcn`) | the check `c7c88ab`; the docstring `de1ee08` | — | fix: test `np.size`/`np.ndim` of `np.asarray(sqrt_beta)`; PI: whether non-finite values and names are accepted (proposed: a positive finite number or a callable, refused otherwise with a message); the docstring corrected | fingerprint |
| W3-11 | B3-C F10, B3-K6 | an empty search set: the survey's two rows (the ES's `IndexError`, the step's `UnboundLocalError`) no longer hold since `0c56d86` (W0-15), which counts it as a failed search. B3-C F10 finds that the fix does what MATLAB does only in the status at default: MATLAB (a) fails as PyBADS at `improvement_quantile` ≤ 0.5 or level 0; (b) at q > 0.5 with `fsd` > 0 counts an incremental search and moves the incumbent to the previous search's stale `usearch` with `fval` and SD 0; (c) runs the hedge update with that stale point, er = 0, so every gain decays, where PyBADS skips the update (`bads.py:1996`); (d) errors on an undefined `usearch` when the run's first search is empty (`bads.m:667-725`, `1257-1279`). The comment at `bads.py:1956` and the commit say "as in MATLAB BADS … on every path" | the rows: no longer hold; the rest: design question ((b) and (d) are MATLAB defects that PyBADS avoids; (c) a difference) | the hedge's (c): when a set is empty after an earlier search (only through `non_box_cons`), all levels; (b): no | forced failure `0c56d86`; the hedge's guard `c7c88ab`; MATLAB since 2017 | the two rows on the empty search set | PI: decay the gains on an empty set as MATLAB, a failed search on the hedge's path too; or keep the skip and put it on the sheet. Proposed: decay them (the verifier leaves it open), since a failed search with a point decays them as well and W0-15's ruling counts an empty set as a failed search; (b) and (d) on the sheet as deliberate, the comment corrected; the survey's rows corrected | fingerprint; a test with a `non_box_cons` that empties a set |
| W3-12 | B3-K2 | after a failed rebuild, the search ranks its candidates by the LCB of the GP restored by `local_gp_fitting` (a consistent GP with finite predictions; with an injected failure it chose a point 142, 38, 93 grid units away, LCB 0.0087); MATLAB's `post = []` makes `gppred` fail again, `acqLCB` sums over no samples, z ≡ 0, and the stable sort keeps `uCheck`'s first, lexicographic candidate (-1891, 1696, 291 units away, LCB 1.55 under the same GP). The poll treats such a GP as unreliable on both sides | design question | only after a failed rebuild (rare; none in the default suite), all levels | since `685da15` (before it the run stopped); never agreed | the `_search_step_` row "(at `a83bd51`)" | keep the ranking by the restored GP (MATLAB's choice is an arbitrary far point), and extend KD-B5-2 with it | none |
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
| W3-21 | B4-C F4, B4-K2 | the target under `hyp_best`: PyBADS sets the hyperparameters on a copy and recomputes its posterior (`bads.py:2659-2670`); MATLAB's `UpdateTarget` keeps `post` and evaluates the kernel and mean under `hyp` (`bads.m:1301`, `utils/gppred.m:39-47`, `utils/mygp.m:122-123`, `146-187`), a hybrid that is no GP prediction under one set of hyperparameters (emulated: equal to `gp.predict` under the GP's own hyperparameters; under those of 1 to 3 iterations earlier, means of 1.2e3 to 2.8e7 where the observed values are at most 5e-3). Hyperparameters differed in 0 of 21 decisions at level 0 and 5 of 13 at level 1; the hybrid would flip 1, the current GP's own prediction none (143.053 against 143.054) | design question (KD-B4-2 leaves it open) | yes, all levels (it matters when `hyp_best` differs from the current hyperparameters: a refit in the poll after its best point, a move after the re-estimate) | never agreed (`c7c88ab` predicted from the current GP; `9037851` recomputes; `685da15` the fallback) | the `_get_target_from_gp_` row "(at `676083d`)" | PI: (a) keep the recomputation and settle it in KD-B4-2; (b) predict from the current GP, which removes a copy per step and the `LinAlgError` path of KD-B4-2, and moves runs by little; (c) MATLAB's hybrid, not recommended. Proposed: (a), which realizes MATLAB's evident intent (the target under the best iteration's hyperparameters) with a valid prediction | none for (a); population comparison at level 1 for (b) |
| W3-22 | B4-I F10, B4-K5 | the poll calls `np.seterr(divide="ignore")` when the root logger is above DEBUG (`bads.py:2271-2272`) and never restores it: after a run, `np.geterr()["divide"]` is `'ignore'` and 1/0 in the user's code no longer warns | confirmed defect (Python only) | yes, all levels | `f9e9326` (2022-11-02) | the `_poll_step_` row "(at `4bde5e9`)" | fix: `np.errstate(divide="ignore", invalid="ignore")` around `gamma_z`; `test_seeded_run_leaves_global_state_untouched` extended to `np.geterr()` | fingerprint |
| W3-23 | B4-I F7, B4-K3 | when the target's prediction is not finite, `_get_target_from_gp_` falls back to the incumbent's `fval` and `fsd`, but the target's formula keeps the raw `fs2` (`bads.py:2672-2695`): `fs2 = NaN` gives a NaN target, `inf` gives `-inf`, and the poll then treats the GP as unreliable. MATLAB does the same (`bads.m:1310-1311`, `1321`), where it follows a failed rebuild; in PyBADS a restored GP is consistent, and gpyreg's predictions are non-finite only on overflow (none seen). `test_target_fallback_to_incumbent` accepts a NaN target | confirmed shared defect, inert in practice | no (a non-finite prediction) | MATLAB 2017; Python `c7c88ab`, reshaped in `685da15` | the `_get_target_from_gp_` row on a non-finite target | fix: `f_target_s**2` in the fallback's formula; the test asserts a finite target; an entry in `matlab_side_defects.md` | fingerprint |
| W3-24 | B4-I F1 | the poll's basis is always the ± coordinate directions: `n_max = max(1, round(search_mesh_size/mesh_size))` (`poll/poll_mads_2n.py:22`) is 1 at every default state, since the search mesh is at least 2^10 times finer than the poll mesh, so the basis is a signed permutation of the identity: a coordinate poll, not LTMADS's dense directions. `pollMADS2N.m:7` is identical, and both user documents (`README.md`, `docsrc/source/index.rst`, MATLAB's README) describe steps in one direction at a time; the docstring of `poll_mads_2n` claims "dense refining directions" and convergence guarantees and cites the Sto-MADS paper for LTMADS | design question, shared with MATLAB | yes, all levels | the two agree (MATLAB 2017, `c7c88ab`) | — | keep MATLAB's poll, correct the docstring of `poll_mads_2n`, and put it in `matlab_side_defects.md` as a shared observation; real LTMADS directions would depart from MATLAB | none if kept |
| W3-25 | B4-I F2 | the GP's `poll_scale` never shapes the poll vectors: `poll_mads_2n` divides by it and `_poll_step_` multiplies it back (`poll_mads_2n.py:36-37`, `bads.py:2146-2150`) | not a defect: MATLAB does the same on purpose ("Counteract subsequent multiplication by pollscale", `pollMADS2N.m:23-24`, `bads.m:803`); `poll_scale` shapes only the ES-ell search (and MATLAB's non-default `pollGPS2N`) | yes | the two agree | — | correct the record: `AGENTS.md` ("drive the poll basis and the ES-ell search") and the description of `gp_rescale_poll` ("scaling factor of poll vectors"), which scales only the ES-ell search | none |
| W3-26 | B4-I F5 | at level 0 the poll's GP never takes the poll's own evaluations (only levels 1 and 2 add them, `bads.py:2309-2332`), so after an improving point the target is predicted at `u_poll_best`, where the GP has no data (observed 4.155, predicted 86.41; observed 26.35, predicted 11.44), and the remaining points' LCB and PoI ignore the poll's observations. MATLAB does the same (`bads.m:908`, `UpdateTarget(upollbest, …)`); using a GP that holds the point changed 1 of 9 runs | design question, shared with MATLAB | yes, level 0 | the two agree (2017, `c7c88ab`) | — | keep MATLAB's behavior, as a shared observation in `matlab_side_defects.md` | none if kept; population comparison at level 0 if changed |
| W3-27 | B4-I F8 | `np.argmin` returns the first NaN of the acquisition (`bads.py:2257`, the search's `1805`), where MATLAB's `min` skips NaN; the fallback "randomly choose index" can never fire (`argmin` always returns a finite index), on both sides | confirmed, inert | no (a NaN prediction; none seen) | never agreed (`c7c88ab`) | — | fix cheaply: `nanargmin` with a guard for an all-NaN set, which makes the fallback live, at both sites | fingerprint |
| W3-28 | B4-I F9 | a good poll stops whenever the GP is unreliable, and a zero predictive SD at any remaining point makes γ infinite and the GP unreliable, whatever `tol_poi` says, although its description says 0 always completes polling. Zero SDs were frequent at level 0 (16 to 38 of 60 to 72 poll steps), none after a good poll in 15 runs | not a defect: MATLAB's rule (`bads.m:862-895`) | yes (the rule), all levels | the two agree | — | keep; the description of `tol_poi` says that an unreliable GP stops a good poll | none |
| W3-29 | B4-C F5 | MATLAB's `pollmoved_flag` is set only by the poll (`bads.m:956`, `958`) and read at the end of every pass (`1049`: `gpstruct.post = []`), so after a poll that moved the incumbent every search of the next round rebuilds the local GP, until a poll that does not move; PyBADS rebuilds once, which the first search of the round does anyway (`bads.py:1734-1737`, `2498`). After a search move MATLAB rebuilds once, as PyBADS. W1-2's premise, "MATLAB's `post = []` asks for one rebuild" (`verification/wave1.md`), missed line 1049; the changelog's "Rebuilds of the local GP" and the comment at `bads.py:1734-1736` say "as MATLAB BADS does". The rebuilds keep the hyperparameters, and in 24 such searches (95 in B4-C's runs) the training set and the predictions were the same: no effect within 200 evaluations; longer runs can differ once the nearest-neighbour set changes | confirmed port discrepancy (MATLAB's persistence looks unintended to the reviewer) | yes, all levels, without effect in runs of 200 evaluations | matched after a poll move until `fef6c14` (W1-2), which made the rebuild once only; MATLAB 2017 | — | PI: (a) persist after a poll move only, as MATLAB; (b) keep one rebuild and put it on the sheet. Either way the changelog entry, the comment and wave 1's row are corrected. Proposed: (a), since W1-2 ruled toward MATLAB on a premise that missed this line | population comparison at default for (a) (long runs reach it) |
| W3-30 | B4-C F6 | with `poll_training=False` the poll neither records a refit it does not make nor clears the unreliability flag (`bads.py:2195-2200`), where MATLAB's `IsRefitTime` sets `lastfitgp`, resets the statistics and clears `unrelgp_flag` before the refit is cancelled (`bads.m:822-823`, `1242-1252`) | intentional difference, missing from the sheet (W1-8's ruling; the changelog, "Refits without poll training"; `matlab_side_defects.md`) | no (`poll_training=False`) | `fef6c14` | — | a sheet entry (with KD-B5-2) | none |
| W3-31 | B4-C F7 | `_eval_improvement_` accepts an `improvement_quantile` outside (0, 1) (`bads.py:2018-2037`), where MATLAB refuses it (`bads.m:1269-1271`): at 0 or 1 `erfcinv` is infinite, the improvement at level 0 is 0·∞ = NaN, the incumbent never moves, and a 2-D sphere spends its 100 evaluations to end at its best initial point, without an error | confirmed port discrepancy | no (`improvement_quantile` ≤ 0 or ≥ 1) | never agreed (MATLAB's check since `d04640a`, 2017) | — | fix: refuse such a value with `ValueError` when `BADS` is created (a stricter interface: a changelog entry and an "Upgrading from" line) | fingerprint |
| W3-32 | B4-K1 | a successful poll appends the bound method `self.u_best.copy` | no longer holds: `self.u_best.copy()` since `0c56d86` (W0-16) | yes | fixed in wave 0's fix pass | `bads.py:2183` | correct the survey's row | — |
| W3-33 | B4-K4 | after a re-estimate that moves nothing, `optim_state`'s `yval`, `fval` and `fsd` keep older values (`bads.py:1507-1509` update only the object's), read only by the target's fallback (never reached) and by the copy that `output_fcn` receives; stale in 6 to 20 of about 158 target computations per level-1 run. MATLAB's `optimState.fval` is staler (its move never updates it, `bads.m:1111-1118`) | confirmed, inert (shared) | yes, levels 1 and 2, without consequence | Python `c7c88ab`; MATLAB 2017 | — | keep, or keep `optim_state` in step in the re-estimate. Proposed: keep them in step, since `output_fcn` receives them (the verifier leaves both open) | fingerprint |
| W3-34 | B4-K7 | `np.vstack(u_poll, u_poll_new)` with two arguments (`bads.py:2180`) would raise `TypeError`; the branch cannot run, since the basis is filled once with 2D rows (56 of 56 polls called `poll_mads_2n` once) | confirmed, inert (unreachable) | no | `c7c88ab` | — (wave 0's "Found while verifying") | remove the branch | fingerprint |
| W3-35 | B4-K8 | the poll discards the return of `period_check` (`bads.py:2154-2159`) | confirmed, inert (the stub returns its input; periodic variables are refused, KD-B1-6) | executed, without effect | `c7c88ab` | — | keep; assign the result when periodic variables are ported (KD-B1-6) | none |
| W3-36 | B4-K9 | `u_base` computed and never used in the accelerated mesh reduction (`bads.py:2442-2444`), beside a commented-out condition | confirmed, inert | executed, without effect | `9037851` | — | remove it | fingerprint |
| W3-37 | B4-K10 | the accelerated mesh reduction tested from the wrong iteration (W2-29) | no longer holds: `c9a2cde` tests `iter >= accelerate_mesh_steps`, MATLAB's `iter > steps` with its count from 1, and reads the same stored iteration (`iter - steps`, MATLAB's `iter - steps` counted from 1); at each poll with `iter` ≥ 3 the history holds exactly `iter` entries | yes | fixed 2026-09-26 | — | none | — |
| W3-38 | B4-K11 | under `stobads`, a NaN estimate after a failed add counts as uncertain | no longer holds: `_sto_success_improvement_` returns −1 for a non-finite estimate since `0c56d86` (W0-11) | no (`stobads`) | fixed in wave 0's fix pass | the `_poll_step_` row with `stobads` "(at `a83bd51`)" | correct the survey's row | — |

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
  search after a failed rebuild (B3-K2), `ESSearchCMA` (B3-K5),
  `period_check`'s return (B4-K8) and `u_base` (B4-K9). Wave 2's fixes in
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
