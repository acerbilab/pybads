# Wave 4 ledger: B7 and O

The verified findings of wave 4 of the port review
([plan](../../../plans/port-correctness-review.md), "Wave 4 pickup"): slice
B7, the function logger, the initial design and the utilities, read on both
tracks, and slice O, the formulas of the improvement, the acquisition and
the geometry, read by a third reader who re-derived each formula and then
compared it with MATLAB BADS. The reports are `reviews/B7_internal.md`,
`reviews/B7_comparison.md` and `reviews/O_third.md`. Each slice was
verified by a fresh Opus agent that had not written its reports, with
checks of its own: `wave4_B7_verifier.md` and `wave4_O_verifier.md`. The
verifiers were also given the items of the review's records that belong to
their slice and that the reviewers did not receive (the plan's "Wave 4
pickup", step 4), as B7-K1 to B7-K9 and O-K1 to O-K5, quoted in
`../briefs/wave4_kept_B7.md` and `../briefs/wave4_kept_O.md`. Every agent
read PyBADS at `0d866e8` (PI, 2026-09-27: `dev-next` after wave 3's fix
pass, #77), MATLAB BADS at `74919c0` and gpyreg v1.3.3 (`98ab5a4`), with
the complete history of both, in a cloud session (Linux, Python 3.11.15,
NumPy 2.4.6, SciPy 1.17.1, where the fingerprint of
`dev/scripts/fingerprint.py` at `0d866e8` with one BLAS thread is
`360971bf1f0ba6cb`, the Linux reference's). The scripts and their outputs
are under `scripts/wave4/<slice>_<track>/` and
`scripts/wave4/<slice>_verifier/`, formatted by the pre-commit hooks after
they ran; the B7 verifier's copy of v1.1.0's package, made with `git
archive` for its dating, is left out, since the tag holds it. The reports
cite the scripts at the sandbox's scratch paths.

Lines are at `0d866e8`, MATLAB lines at `74919c0`. "B7-I F1" is finding F1
of the B7 internal report, "B7-C" the B7 comparison report and "O F1" the
third reader's; "B7-K1" is a kept item, and a letter (A to N) names the
B7 verifier's grouping of the findings that describe the same behavior.
"Survey" names the row of the candidate table of
`dev/results/2026-09-23-codebase-survey.md` that describes the same
behavior. The dispositions are proposals until the PI rules: the verifier's
recommendation unless the row says why not. The gate is the one of
`AGENTS.md`, "Numerical gates", that a fix would need. "Default run" says
whether a run at default options reaches the code, and at which
uncertainty level (0 deterministic, 1 noise inferred, 2
`specify_target_noise`). Wave 0's fix pass reached `dev-next`
squash-merged as `0c56d86`, wave 1's as `fef6c14`, wave 2's as `8aecb6a`
and wave 3's as `0d866e8`; the commits of the fix passes cited here are
those of their branches and pull requests.

## B7: the function logger, the initial design and the utilities

| Id | Source | What | Classification | Default run | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|
| W4-1 | B7-I F1, B7-C F1 (the interior starts), B7-K1, B7-K2 (the seed); verifier A | the initial design depends on neither the start nor `random_seed`: `init_sobol` seeds scipy's scrambled Sobol set from the characters of `array2string(u0[:11].astype(uint64))` (`init_functions/init_sobol.py:52-62`), and every coordinate inside the plausible box, `(-1, 1)` in `u`, has the integer part 0, so each D has one seed (49, 948, 967, 241, 748, 843 and 431 at D = 1 to 7, the int64 product wrapping from D = 7; 1 from D = 8, where it is 0). The branch that draws the seed from `bads.rng` (`:64`) is unreachable, since `BADS.__init__` replaces a non-finite `x0` first. With a random `x0` at D = 3 over 20 seeds, the design was the same in all 20 runs and 16 of them started from the same design point. The scrambling draws from scipy's own generator, which contradicts KD-B1-1's headline and the statements that every random draw of a run comes from one generator (`bads.py:142`, `docsrc/source/index.rst:23`, the changelog's 1.1.0 entry). MATLAB's `initSobol.m:9-15` seeds a skip index into the unscrambled sequence from `prod(uint64(num2str(u0(1:10))))`: MATLAB documents `prod` of an integer array as returning double, so it neither keeps `uint64` nor saturates; the product of a non-integer start exceeds 2^53 at D ≥ 2, and the seed then depends on MATLAB's `mod` beyond `flintmax`: an exact remainder gives 378 to 395 seeds over 500 random starts, MATLAB's documented formula or its round-off compensation gives 1 for all of them (one design per D, as PyBADS). Only MATLAB can say which | needs MATLAB (MATLAB's own effect); the PyBADS facts confirmed | yes, all levels | never agreed (Python `c7c88ab`; `initSobol.m` unchanged since `6c93629`, 2017) | the `init_sobol` row; W0-19 | PI: (a) seed the scrambling from `bads.rng`, so that `random_seed` decides the design, as KD-B1-1 says of every draw (a departure from MATLAB, whose design depends on the start alone); (b) a well-defined seed from the start's values, MATLAB's intent ("Seed depends on U0"), independent of `random_seed`; (c) keep, settle KD-B7-1 and correct the statements on the generator. Proposed: (a). It does not depend on what MATLAB computes, so no MATLAB run is written up for it; the one call that settles MATLAB's side, `mod(prod(uint64(num2str([0.25 -0.5]))),997)+1` (966 for an exact `mod`, 1 otherwise), goes into `matlab_side_defects.md` as an open question. The verifier leaves the design to MATLAB's answer; the row departs from it because PyBADS's own contract decides the case | population comparison at default (every run moves from its first design point), and `tolerance_sweep.py` for the seeded optimization tests |
| W4-2 | B7-I F2, B7-C F1 (the platform), B7-K2 (the cast), B7-K3; verifier B | a start coordinate at or below -1 in `u` (an `x0` on or below its plausible lower bound: `x0 = lb` with the plausible bounds omitted, `x0 = plb` given, `test_small_noisy_func`'s `x0 = -3` at `u0 = -1.5`) reaches the cast of a negative float to `uint64` (`init_sobol.py:55`), undefined in C: x86-64 gives `2**64 - 1` without a warning (seed 1 at D = 2), AArch64 saturates to 0 (seed 948, the interior seed; by reading and by the survey's macOS warning), so such a seeded run has another design on Apple silicon than on x86. The kept note that W2-4 made the cast reachable does not hold: at v1.1.0 `x0 = lb` with the plausible bounds omitted also gave `u0 = -1` exactly, since 1.1.0 moved `x0` and `plb` together (`scripts/wave4/B7_verifier/v2_bound_start_v110.out`); any `x0 ≤ plb` has reached it since `c7c88ab` | confirmed defect | yes, all levels, from such starts | `c7c88ab`; MATLAB has no counterpart | the `init_sobol` row (its clause "since W2-4" corrected) | fix, unless W4-1 (a) or (b) removes the cast: `u0[:11].astype(np.int64).astype(np.uint64)`, well defined and equal to x86's values, so that x86 runs do not move and arm64 runs from such starts do (a changelog line); the survey's row and wave 2's doublecheck note corrected | fingerprint (x86 unchanged); the `bounds` suite's start on a lower bound reaches the cast, and runs as before on x86 |
| W4-3 | B7-I F3, B7-C F2, B7-K4 (W0-18); verifier C | the design is raised to the next power of two when its size equals D (`init_sobol.py:73-76`): at level 0 it has 2, 4, 8, 16 and 32 points at D = 1, 2, 4, 8 and 16, where the rounding alone gives D points, as MATLAB draws; at levels 1 and 2 only at D = 32 (the design is 32 at every D ≤ 20). The rule compares with D, not `fun_eval_start`, so user sizes jump (D = 4: 2 gives 2 points, 3 gives 8; D = 16: 8 gives 8, 9 gives 32). It came with the whole file in `c7c88ab`, with no message or comment; the commented-out lines above it show an earlier form of `fun_eval_start` points, and the Owen comment argues only for a power of two. The search needs more than D rows, which `x0` and the rounded design already give | design question | yes, level 0 at D = 1, 2, 4, 8, 16 | never agreed (`c7c88ab`, unchanged) | — (W0-18, whose ruling leaves the decision to this wave) | PI: keep, with a stated reason (none is recorded), or remove it, so that the design is `2**ceil(log2(fun_eval_start))`. Proposed: remove (the verifier's recommendation): D extra evaluations at those D (4 of 61 in a 2-D run) and the jumps, with no reason found; the description of `fun_eval_start`, `AGENTS.md`, KD-B7-1, KD-B2-6 and claim C2 follow | population comparison with level-0 runs at a power-of-two D (the default suite's 2-D configurations, the `oned` suite, and the 4-D ones of `geometry`); the fingerprint (D = 3) unchanged |
| W4-4 | B7-I F4, B7-C F7 (in part), B7-K6; verifier D | `init_sobol` returns the exponent where its docstring says "Number of samples" (2 for 4 points; the caller discards it, `bads.py:1112`); its parameter defaults are types, `lb` and `ub` are unused, `plb`/`pub` are called "bounds for the parameters", and `fun_eval_start` "Number of initial function evaluations" | confirmed, inert | the call yes; the value is discarded | `4e6a001` (2022-11-15) | — | fix: return the number of points, or document the exponent; the docstring corrected (with W4-3 if ruled) | fingerprint |
| W4-5 | B7-I F5, B7-C F3, B7-K9; verifier E | the level-2 merge of a repeated point, KD-B7-3, matches the code (`function_logger.py:406-436`; `Y_orig` and `Y_max` not updated) but no run reaches it at any option since W3-1 (`149d528`): every recorded evaluation after `x0` passes `contraints_check`, which removes a candidate whose bin holds a logged point, `specify_target_noise` leaves no noise test, the final samples take the path that records nothing, and `add` has no caller (level 2: 190 new rows and 10 unrecorded repeats, no merge, in each of three runs). KD-B7-3's "Until the next rebuild, `add_and_update_gp` then adds that value…" and its population evidence, the changelog's "Repeated points with user-specified noise" (`CHANGELOG.md:158`), which reads against "Points evaluated again … neither repeats any now" (`:586-590`), and `AGENTS.md`'s `FunctionLogger` bullet describe runs before W3-1 | confirmed, inert | no, at no option | the merge `c7c88ab`, its row fixed in `032dfcb`; unreachable since `149d528` (in `0d866e8`); MATLAB never merges | the `function_logger.py`, `__call__` row (at `1a21844`) | correct the records: KD-B7-3 says that only a direct use of `FunctionLogger` reaches the merge, which closes the survey's row; the changelog's two entries reconciled; `AGENTS.md`. The code stays (a documented class) | none |
| W4-6 | B7-I F6, B7-C F6 (b), B7-K7; verifier F | the noise test takes the path that records nothing and still adds 1 to the start's `n_evals` and averages its time into that row's `fun_eval_time` (`function_logger.py:390-404`): `n_eff`, the sum of `n_evals` (`gaussian_process_train.py:1133`), counts it, where `eff_starting_points` (`bads.py:1175`) does not, so the budget fraction behind `init_N` ("the fraction of the budget used after the initial design") runs one evaluation ahead: at D = 2, 123 prior draws instead of 128 at the first refit, 77 instead of 81 ten evaluations later. The row's time has no reader (`t_train` is unused). MATLAB calls the target directly for the test (`evalinitmesh.m:41`) and writes `funevaltime` only for `'iter'` calls | `n_eff`: confirmed defect (minor; `init_N` is PyBADS's own schedule); the time: confirmed, inert | yes, levels 0 and 1 (`uncertainty_handling` left empty) | matched MATLAB at `c7c88ab` (a direct call); recorded as a row in `9037851`, unrecorded with the `n_evals` increment kept in `f9e9326`/`8ff10f5` (2022-11) | — (B7-K7 is also in `dev/TODO.md`, the minor items of B1 and B2) | fix: the noise test leaves the start's row as it was (`n_evals` and time), as MATLAB's direct call; low priority | population comparison at default (`init_N` moves in every level-0 and level-1 run), batched with W4-1 or W4-3 |
| W4-7 | B7-K8, B7-I F7; verifier G | `overhead`: the final samples and a merged repeat add their time to the target's since `17e65ee` (W2-20), and the noise test stays out, as MATLAB's `funlogger.m:130` and `evalinitmesh.m:41` do (with one second per call: level 1, 100 calls, 89 rows, 99 s); the run's time spans the untimed noise test, so it counts as the optimizer's, on both sides (`bads.m:1186`) | B7-K8: no longer holds (`17e65ee`, in `8aecb6a`); B7-I F7: confirmed shared defect, negligible | yes, levels 0 and 1 (F7) | MATLAB's accounting since `603da99` (2017); the Python fix 2026-09-26 | — | keep, as W2-20's ruling did; optionally a sentence in the `overhead` description | none |
| W4-8 | B7-I F8, B7-C F7 (the SD); verifier H | malformed target outputs do not raise the documented `ValueError`: the SD check (`function_logger.py:173-179`) lacks MATLAB's `isscalar` (`funlogger.m:108`), which `add` has (`268-277`), so an SD of `[0.5]` or `None` raises `TypeError` and one of two elements NumPy's ambiguous-truth `ValueError`, while a value of `[1.0]` is accepted; an array of size above 1 raises with the note "Error in executing the logged function"; a Python complex with a zero imaginary part passes the check and fails later with `TypeError`, after the row's index has advanced and `X` has been written | confirmed port discrepancy (minor) | no (a malformed target) | Python `c7c88ab`/`9037851`; MATLAB's `isscalar` since `a3b6ebd` (2021); never agreed | — | fix: convert and check the SD as the value is, validate before writing the row, and drop the note on the logger's own conversion | fingerprint |
| W4-9 | B7-I F9; verifier I | `finalize` trims every array but `n_evals`, so a later `n_evals[X_flag]` raises `IndexError` and a later call grows the arrays to unequal lengths; `reset_fun_eval_time` rebuilds `fun_eval_time` at `cache_size` rows, shorter than the others once they grew. No code calls either, and the documentation page presents both | confirmed, inert | no | `c7c88ab` | — | fix both (they are on the class's documentation page), with `test_finalize` checking `n_evals` | fingerprint |
| W4-10 | B7-I F10, B7-C F7 (`add`); verifier J | the docstrings of `__call__` and `add` do not say that `x` is in `u` space (`function_logger.py:79-81`, `208-210`); `add` refuses a one-element array that `__call__` accepts, sets a missing SD to 1 when the logger holds SDs and drops a given one when it does not; `add` has no caller (`fun_values` is refused, KD-B1-4), and `test_add_parameter_transform` runs at `x = 0`, where the transform is the identity | confirmed, inert | no | `c7c88ab` | — | correct the docstrings now; `add`'s semantics with the port of `fun_values` (`dev/TODO.md`) | none |
| W4-11 | B7-I F11, B7-C F5; verifier K | `period_check`'s call sites: the poll discards the result (`bads.py:2220-2225`), which W3-35 ruled to keep until periodic variables are ported; also, the design passes the option, an index list or `None` (`1130-1135`), where the search and the poll pass `optim_state`'s boolean mask; MATLAB assigns the result at every site | confirmed, inert (latent; also W3-35) | executed, without effect | `c7c88ab`; MATLAB since 2017 | — | keep, as W3-35: with the port of periodic variables (KD-B1-6), whose `dev/TODO.md` line gains the argument's form | none |
| W4-12 | B7-C F4; verifier L | the log grows by half when full (`function_logger.py:303-340`), where MATLAB's is a ring of `CacheSize` rows (1e4) that overwrites its oldest rows and never writes its last one (`funlogger.m:120-121`: rows 1, 2, 3, 4, 1, … for 5); the two agree up to 9999 logged evaluations. `cache_size`'s docstring calls it the initial size, its description in `advanced_bads_options.ini:20` the "Size of cache" | intentional difference, missing from the sheet | no at D ≤ 20 | never agreed | — | a sheet entry; the description of `cache_size` corrected; MATLAB's unwritten row in `matlab_side_defects.md` | none |
| W4-13 | B7-C F6 (a); verifier M | the noise test's second value goes through the logger's checks, so a NaN or infinite value there raises `ValueError`; MATLAB calls the target directly and reads NaN as deterministic (the comparison is false) and infinity as noisy (`evalinitmesh.m:41-47`) | confirmed port discrepancy (benign) | no (a non-finite second value) | matched MATLAB at `c7c88ab`; diverged in `9037851` | — | keep (a non-finite value is refused at every other evaluation too), and a sheet entry | none |
| W4-14 | B7-K5; verifier N | a noisy run with a small budget never takes the final samples it reserved: they need `poll_iteration > 0` (`bads.py:1631`), as MATLAB's `iter > 1` (`bads.m:1138`), and PyBADS's design of 32 points (MATLAB's 20) ends the run in its first iteration for budgets up to 44 at D = 2, up to 48 with the first poll (MATLAB up to 32): at 38, 4 samples reserved and none taken, `yval_vec` the incumbent's observation and `fsd` the default `noise_size`. `dev/TODO.md`'s wording ("fewer evaluations than `noise_final_samples`") misses budgets 45 to 48 | confirmed shared defect, widened by the design's rounding | no (a small `max_fun_evals`), levels 1 and 2 | MATLAB's `iter > 1` since `3f9522b` (2017); the widening since `c7c88ab` | — (`dev/TODO.md`, minor items of B1 and B2) | PI: (a) take the reserved samples at the incumbent when the run ends in its first iteration, a departure from MATLAB (an entry in `matlab_side_defects.md`); (b) keep MATLAB's rule and correct the `TODO.md` wording. Proposed: (a), since the budget the user gave is spent on nothing | fingerprint (default runs do not reach it); a test with such a budget |

## O: the improvement, the acquisition and the geometry

Every formula of the slice matches MATLAB BADS line by line and the third
reader's own derivation: the improvement at every quantile and each of its
call sites, the sufficient improvement, the final quantile choice, the LCB
and its schedule, the Hedge's probabilities and update, and `len_scale`,
`poll_scale` and `effective_radius` with their uses (`scripts/wave4/O_third/`:
transcriptions compared over random inputs and over 33 refits and 166
Hedge updates of four runs, with differences of at most 9e-16). The
findings are one defect of Sto-BADS, which has no MATLAB counterpart, and
the kept items.

| Id | Source | What | Classification | Default run | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|
| W4-15 | O F1 | with `stobads` and `opp_stobads`, an uncertain poll whose uncertain points all have an improvement of at most 0 "moves" the incumbent to itself: `u_poll_best` starts at the incumbent and changes only for an improvement above 0 (`bads.py:2172-2176`, `2401-2407`), and the branch `opp_stobads and sto_poll == 0` calls `_update_incumbent_(u_poll_best, …)` and sets `is_poll_moved` (`2455-2459`), so every search of the next round rebuilds the local GP (`1467-1471`) until a poll that does not move. The comment above it and the changelog say that an uncertain poll moves to its best polled point. On a flat noisy target at D = 3, one self-move in each of three seeded runs at level 1 and at level 2, each costing `search_n_try - 1` extra rebuilds without a refit; with the move left unmarked, every evaluated point of three runs was identical, bit for bit. By reading, with `improvement_quantile` < 0.5 an uncertain point whose mean improves can leave `u_poll_best` at the incumbent too | confirmed defect (Sto-BADS; no numerical effect measured) | no (`stobads=True`, levels 1 and 2) | the fallback since `9037851`; W0-10 (`0c56d86`) kept it; its effect depends on the rebuild rule: every step until `e004c79`, none at `fef6c14` and `8aecb6a`, `search_n_try - 1` rebuilds since W3-29 (`0d866e8`); no MATLAB counterpart | — | fix: an uncertain poll moves, and is marked as moved, only to a polled point (`poll_best_improvement > 0`); whether an uncertain outcome that does not improve should move at all is W0-13's question, left to its measurement (`dev/TODO.md`), whose item gains the report's note that the search moves on such outcomes and scales its factor as for an incremental search | fingerprint; the same under `stobads=True` (the verifier's runs did not move) |
| W4-16 | O-K1 (O's answer to Q4); O verifier | with several hyperparameter samples, `poll_scale` would sum the centred log length scales over the samples unweighted (`gaussian_process_train.py:548-556`), which gives MATLAB's value squared for two samples, and `effective_radius` would be one value per sample (`578-593`), where MATLAB weights by `hypweight` and takes the weighted α (`gpupdate.m:294-299`, `313-317`); `len_scale` agrees. No option or path gives the GP more than one sample: `_robust_gp_fit_` returns one row, `gp_s_N` is 0 at both call sites of `_get_gp_training_options`, and `gp_samples` is unread (KD-B5-4) | confirmed, inert (unreachable) | no, at no option | Python `c7c88ab`, MATLAB's weights since `e43650b` (2017); one sample guaranteed since `8ff10f5` | — | fix cheaply, as MATLAB's (weights of 1/N and the weighted α), with the verifier's forced two-sample check as its test; or a comment that the code assumes one sample. Proposed: the fix | fingerprint |
| W4-17 | O-K2; O verifier | `acq_fcn_lcb`'s summary line says that it retrieves a point (it returns the LCB, mean and SD of every row), and it assigns an `n` that nothing reads (MATLAB's `n` serves a `catch` branch the port dropped); `update_hedge`'s docstring speaks of updating a probability of improvement, where it updates the gains `g` | confirmed defect (documentation) | the code, yes; the text only | `c7c88ab` and `de1ee08` (the LCB), `9037851` (the hedge) | — | fix the docstrings, and remove `n` | none |
| W4-18 | O-K3; O verifier | `hedge_gamma` is not checked: the probabilities `(1 - n γ) softmax(β g) + γ` invert above `1/n` and turn negative above `1/(n - 1)` (`search_hedge.py:65-69`, as `searchHedge.m:45-48`); `BADS` runs with 1.25 or -0.5 without a warning, a strategy of negative probability is never chosen, and at 1.25 the search takes the strategy of lower gain every time. Its description ("Minimum probability of each search") holds only for 0 ≤ γ ≤ 1/n | confirmed shared defect (a missing check; MATLAB identical) | no (0.125 is in range) | the two always agreed (MATLAB `6c93629`, 2017; Python `c7c88ab`) | — | fix: refuse a `hedge_gamma` outside `[0, 1/n]`, `n` the number of search methods, when `BADS` is created (a stricter interface: a changelog entry and an "Upgrading from" line); an entry in `matlab_side_defects.md`; `hedge_beta` and `hedge_decay` looked at with it ("Found while verifying") | fingerprint |
| W4-19 | O-K4; O verifier | a `sqrt_beta` that `acq_fcn_lcb` refuses (W3-10) is refused only at the first search, after the initial design and the first poll (12 evaluations at D = 3), since `BADS` does not look at `search_acq_fcn`; a callable's value is not checked: one returning -1 or NaN runs silently (NaN: `fval` 0.142 where the default schedule reaches 4e-6), one returning an array of 2 or a string stops with `IndexError` or `UFuncTypeError`. MATLAB checks at each call too, accepts any numeric scalar, catches the error at its callers and degrades silently | confirmed defect (the refusal is PyBADS's own) | no (`sqrt_beta` left to its default) | the check `599115b` (W3-10, in `0d866e8`); the callable's branch `c7c88ab` | — | fix: the same check on `search_acq_fcn[1]` when `BADS` is created, and a check of a callable's value at each call (a stricter interface for a callable that returns a non-positive or non-finite value: a changelog entry and an "Upgrading from" line) | fingerprint |
| W4-20 | O-K5; O verifier | Fig. 1 (`docsrc/source/_static/bads-cartoon.png`, in `README.md` and `index.rst`), byte-identical to MATLAB's `docs/bads-cartoon.png`, draws the poll's steps about twice as long along x1 as along x2; the poll cancels `poll_scale` (W3-25), and its steps are ±Δ along one coordinate of `u` at default, anisotropic in the original coordinates only through the plausible box or the log transform, which the figure does not show; the text beside it ("steps in one direction at a time") is right | confirmed, inert (documentation, shared with MATLAB's README) | — | MATLAB's figure since `3332e3e` (2017-05-10), in PyBADS since `8bb3d59` (2023-02-06); the cancellation since MATLAB's first commit | — | keep the figure, with a clause in its caption: the steps are equal in the normalized coordinates and scale with the plausible box in the original ones | none |

## Notes on the reports

- **Corrections by the verifiers.** B7: both reports' seeds hold, and the
  int64 product already wraps at D = 7 (seed 431); B7-C F1's "if it
  accumulates in double, the seed follows the digits" holds only if
  MATLAB's `mod` is exact beyond `flintmax`, and its saturating reading of
  `prod` is contrary to MATLAB's documentation (W4-1); B7-K3's dating does
  not hold (W4-2); B7-K2's clause on NumPy 1.x on Windows no longer applies,
  since PyBADS requires NumPy 2. O: the verifier did not reproduce O F1's
  counts of 3, 5 and 14 rebuilds, whose counter can over-count, and found no
  numerical effect where the report says "small" (W4-15).
- **A correction by the orchestrator.** The B7 verifier found no commit
  `c044fea`, which the survey names for the requirement of NumPy 2: it is
  a commit of #65's head (`build: require NumPy 2, and run each test once in
  CI`), squash-merged as `1c8c71d`, whose ref the clone lacked while the
  verifier ran and holds since; the survey's citation stands.
- **The items kept from the reviewers.** Found by the reviewers: the seed
  from the integer parts (B7-K1, B7-K2, both B7 reports), the undefined cast
  (B7-K2, B7-K3, both, without the corrected dating), the doubling (B7-K4,
  both), `init_sobol`'s return (B7-K6, both), the noise test's trace in the
  start's row (B7-K7, both, with its effect on `n_eff` that no record had),
  the merge and KD-B7-3 (B7-K9, both: unreachable since W3-1), and the
  several hyperparameter samples (O-K1, in O's answer to Q4). Not found: the
  reserved final samples of a small noisy budget (B7-K5), the fixed
  `overhead` (B7-K8, which B7-C's answer describes as MATLAB's accounting
  without naming the fix), and O-K2 to O-K5 (O's note on the values that
  W3-10 refuses is adjacent to O-K4). W2-20's fix holds as MATLAB's (W4-7).
  None of the preparatory report's differences seen in passing belongs to
  these slices, and the survey has no open row in O's code.
- **The sheet.** KD-B1-1's headline, "Every random draw comes from one
  `numpy.random.Generator`", does not hold for the design, whose scrambling
  draws from scipy's generator seeded from `u0`, and it cites
  `init_sobol.py:64`, which no run reaches, as a draw site (W4-1); KD-B7-3's
  account of what the GP receives after a merge, and its population
  evidence, describe runs before W3-1 (W4-5); KD-B1-6's "unreachable" is
  loose for `_variable_transformer_`'s periodic branch, which runs with a
  random `x0` ("Found while verifying"). KD-B7-1 and KD-B2-6 follow the
  rulings on W4-1 and W4-3. Entries to add: the growing log (W4-12), the
  noise test through the logger's checks (W4-13), and three differences
  that O found recorded outside the sheet: `len_scale` at D = 1, which MATLAB
  sets to 1 (W1-17, the changelog's "One-dimensional problems"), W3-10's
  refusal of a `sqrt_beta` that is zero, negative, non-finite or a name,
  which MATLAB accepts, and W3-27's random choice when every acquisition
  value is NaN, where MATLAB's `min` returns the first. No finding
  contradicts any other entry.
- **Records.** The statements that every random draw of a run comes from
  one generator (`bads.py:142`, `index.rst:23`; the changelog's released
  1.1.0 entry stays as it was) (W4-1); the survey's row of `init_sobol` and
  wave 2's doublecheck note, "since W2-4" (W4-2); the description of
  `fun_eval_start`, `AGENTS.md` and claim C2 if W4-3 is ruled; the
  changelog's two entries on repeats and `AGENTS.md`'s `FunctionLogger`
  bullet (W4-5); the description of `cache_size` (W4-12); `dev/TODO.md`'s
  wording of the small budget (W4-14); the comment above the uncertain
  poll's move (W4-15); the descriptions of `hedge_gamma` (W4-18) and, if
  ruled, `overhead` (W4-7).
- **Tests.** Nothing tests `init_sobol` (its size, its dependence on the
  start and the seed, its map, its platform independence), and the seed
  tests pass with a design that ignores the seed (W4-1 to W4-4);
  `test_function_logger.py`'s merge tests call `_record` directly (W4-5);
  `test_finalize` omits `n_evals` (W4-9); `test_add_parameter_transform`
  runs where the transform is the identity (W4-10);
  `test_call_invalid_sd_value` covers only an infinite SD (W4-8);
  `test_stobads.py::test_uncertain_poll_moves_only_with_opp_stobads` asserts
  only that `_update_incumbent_` was called (W4-15); the Hedge's reward
  test runs at `g = 0`, `phat = 1` and a mesh size of 1; nothing compares
  the geometry after a refit with `gpupdate.m`'s formulas, tests
  `_eval_improvement_` at a quantile other than 0.5, or the final quantile's
  choice.
- **A proposal for grouping the fixes.** Moving results at default, each
  under a population comparison on Linux against
  `population_linux_wave3_20260927` (this box computes as its environment:
  the fingerprint at `0d866e8` is its `360971bf1f0ba6cb`), as steps of
  their own: W4-3 (with level-0 runs at a power-of-two D), W4-1 (every
  run), and W4-6, which is small enough to ride with either. The rest must
  move nothing, each under the fingerprint: W4-2 (if W4-1 keeps the cast),
  W4-4, W4-8, W4-9, W4-14, W4-15 (also under `stobads=True`), W4-16, W4-18,
  W4-19, and the records (W4-5, W4-7, W4-10 to W4-13, W4-17, W4-20).

## Survey rows of B7 and O

The two open rows of the candidate table in these slices are closed by the
rows above: `init_functions/init_sobol.py`, `init_sobol` (W4-1, W4-2, its
clause on W2-4 corrected), and `function_logger/function_logger.py`,
`__call__` at `1a21844` (W4-5: KD-B7-3's merge, unreachable since W3-1).
Slice O has no open row. `dev/TODO.md`'s "Bug hunt and verification against
MATLAB BADS" names the Sobol seed as a starting point of the review (W4-1).

## Found while verifying

Met by a verifier, outside the reports' findings and the kept items, and
not verified beyond the check named; each is proposed here, since this is
the review's last wave.

- `hedge_beta` and `hedge_decay` are not checked either
  (`search_hedge.py:53-54`); a `hedge_decay` above 1 would make the gains
  grow without bound (O verifier, unverified). With W4-18.
- With a random `x0`, `_variable_transformer_` runs before the refusal of
  periodic variables (`bads.py:326`, `643-647`), so an index out of range
  in `periodic_vars` raises `IndexError` there instead of the `ValueError`,
  and an empty `periodic_vars`, which MATLAB treats as no periodic
  variable (`setupvars.m:107-108`), is refused (B7 verifier, reproduced;
  from B7-I's answers, and B7-C's). Proposed: refuse `periodic_vars` before
  the transform, and take an empty one as `None`, under the fingerprint.
- A level-1 run that ends in its first iteration reports `fsd` equal to
  `noise_size`, a default rather than an estimate; MATLAB appears to do the
  same (`bads.m:448-452`; B7 verifier, seen in `v4`). With W4-14.
- After MATLAB's ring of evaluations wraps, `U(1:Xmax)` and `Y(1:Xmax)`
  include the row it never writes (`uCheck.m:23`, `gpupdate.m:30`; B7
  verifier, unverified). With W4-12, in `matlab_side_defects.md`.
- Under the literal formula of MATLAB's `mod`, one transcribed seed came
  out negative, which `i4_sobol.m:249-250` clamps to 0, the Sobol set's
  origin at the plausible box's lower corner (B7 verifier, unverified; only
  MATLAB can say). With W4-1's open question.
- From the reports' answers, not findings: `VariableTransformer`'s inverse
  clips to the original bounds, where MATLAB's `transvars` does not, which
  only absorbs rounding (B7-C); a MATLAB target returning `[f, sd]` without
  `SpecifyTargetNoise` runs with `sd` dropped, where PyBADS refuses the
  tuple (B7-C); the logger's `uncertainty_handling_level` keeps its
  construction value after the noise test raises the run's level, and
  `y_max` and `cache_count` are never read (B7-I); `Y_max` is not updated
  by a merge (both). For the docstrings of the fix pass, or none.

## Rulings (PI, 2026-09-27)

The orchestrator proposed a disposition for every row, following its
verifier's recommendation unless the row says why not. The PI ruled W4-1
(a), the design seeded from the run's generator; W4-3 keep, the design's
doubling kept at every D, where the proposal was to remove it; and W4-14
(a), the reserved final samples taken at the incumbent; and accepted every
other proposal as written. As in waves 0 to 3, a fix is one commit per row
on the wave's branch, with a test that fails at `0d866e8` and passes at the
commit, and a changelog line in every commit a user can notice; a stricter
interface also has an "Upgrading from" line. The fix pass follows
`wave3.md`, "Fix pass", with the whole fast suite and the fingerprint after
every cherry-pick, CI checked after every push that touches `pybads/`, and
the head's population as the new Linux reference. The sheet,
`matlab_side_defects.md`, the survey and `dev/TODO.md` are updated in the
pass, each entry with the fix it describes, as in wave 3.

**Fix, moving results**, in this order, each ending in a population
comparison on Linux against the end of the step before, the first against
`population_linux_wave3_20260927`, whose environment this sandbox has (the
fingerprint at `0d866e8` is its `360971bf1f0ba6cb`):

1. W4-1 (a): `init_sobol` seeds the scrambling of the design from the
   run's generator (`bads.rng`, which it already receives), and no longer
   from the integer parts of `u0`, so that `random_seed` decides the design,
   a departure from MATLAB BADS, whose design depends on the start alone.
   W4-2's undefined cast goes with the seed from `u0`. With it: KD-B1-1 and
   KD-B7-1, `AGENTS.md`'s sentence on the Sobol design's seed, the survey's
   row of `init_sobol` closed (its clause on W2-4 and wave 2's doublecheck
   note corrected, W4-2), and in `matlab_side_defects.md` the open question
   of MATLAB's own seed, the one call that settles it
   (`mod(prod(uint64(num2str([0.25 -0.5]))),997)+1`: 966 for an exact
   `mod`, 1 otherwise), with the negative seed that `i4_sobol.m` would clamp
   ("Found while verifying"). A test that two seeds give two designs and one
   seed the same design, whatever the start. The default suite; the seeded
   optimization tests checked over their seeds with
   `dev/scripts/tolerance_sweep.py` if one fails.
2. W4-6: the noise test leaves the start's row of the log as it was
   (`n_evals` and its time), as MATLAB BADS's direct call does, so that
   `n_eff` counts what `eff_starting_points` counts; B7-K7's line of
   `dev/TODO.md` closes with it. The default suite against W4-1's step.

**Fix, moving nothing** (each under the fingerprint):

- W4-4: `init_sobol` returns the number of points, as its docstring says,
  and its docstring states the rounding to a power of two and the doubling
  (W4-3), with its parameters described as what they are.
- W4-8: the target's SD is converted and checked as its value is, both
  before the row is written, and the logger's own conversion errors carry
  no note that blames the target.
- W4-9: `finalize` trims `n_evals` too, and `reset_fun_eval_time` keeps the
  arrays' current length; `test_finalize` checks `n_evals`.
- W4-14 (a): when a noisy run ends within its first iteration, the final
  samples it reserved are taken at the incumbent, the only candidate, and
  give the reported `fval` and `fsd`, as the final estimate does after a
  later iteration; a departure from MATLAB BADS, which shares the gap, in
  `matlab_side_defects.md` and on the sheet; `dev/TODO.md`'s item of the
  small noisy budget closes with it, and so does the default `fsd` of such
  a run ("Found while verifying"). A test at a small noisy budget.
- W4-15: an uncertain poll moves the incumbent, and is marked as moved,
  only to a polled point that improves on it; `test_stobads.py` asserts that
  the incumbent changed. Also under the fingerprint with `stobads=True`.
  W0-13's item of `dev/TODO.md` gains the report's note that the search
  moves on every uncertain outcome and scales its factor as for an
  incremental search.
- W4-16: `poll_scale` and `effective_radius` over several hyperparameter
  samples as MATLAB BADS computes them (weights of 1/N, the weighted α),
  with a test that forces two samples.
- W4-18: `hedge_gamma` outside `[0, 1/n]`, `n` the number of search
  methods, is refused when `BADS` is created (a stricter interface), its
  description saying so; `hedge_beta` and `hedge_decay` looked at with it
  ("Found while verifying"); an entry in `matlab_side_defects.md`.
- W4-19: the `sqrt_beta` of `search_acq_fcn` is checked when `BADS` is
  created, and a callable's value at each call (a stricter interface for a
  callable that returns a value that is not positive and finite).
- `periodic_vars` ("Found while verifying"): refused before the variables
  are transformed, and an empty one taken as `None`, as MATLAB BADS takes
  it; KD-B1-6's "unreachable" corrected.
- The records and docstrings: `init_sobol`'s with W4-4; the statements that
  every random draw of a run comes from one generator hold after W4-1
  (`bads.py:142`, `index.rst:23`); KD-B7-3, the changelog's two entries on
  repeats and `AGENTS.md`'s `FunctionLogger` bullet (W4-5); the
  description of `overhead`, which counts the noise test as the
  optimizer's time, on both sides (W4-7); the docstrings of `__call__` and
  `add`, which take `x` in `u` space (W4-10); the description of
  `cache_size` (W4-12); `acq_fcn_lcb`'s summary and its unused `n`, and
  `update_hedge`'s docstring (W4-17); the caption of Fig. 1 in `README.md`
  and `docsrc/source/index.rst` (W4-20); and the docstrings that the pass
  touches, from the reports' answers.

**Keep, and record:**

- W4-3 (PI): the design keeps its doubling when its power of two equals D,
  at every D. KD-B7-1 records the design's size as deliberate, by this
  ruling; KD-B2-6's "Not settled" clause, claim C2 and W0-18 close. The
  description of `fun_eval_start` and `AGENTS.md` already state it.
- W4-5: the merge stays, a documented behavior of `FunctionLogger` that no
  run reaches since W3-1; the survey's row of `function_logger.py`,
  `__call__`, closes by KD-B7-3 as corrected.
- W4-7: the accounting of `overhead`, MATLAB BADS's, as W2-20 ruled; a
  shared observation in `matlab_side_defects.md`.
- W4-10: `add`'s semantics wait for the port of `fun_values`
  (`dev/TODO.md`).
- W4-11: `period_check`'s call sites, as W3-35 ruled, until periodic
  variables are ported; that line of `dev/TODO.md` gains the form of the
  design's argument.
- W4-12: the growing log, a sheet entry; MATLAB's unwritten row of its ring,
  and the row it reads after the ring wraps, in `matlab_side_defects.md`.
- W4-13: the noise test through the logger's checks, a sheet entry.
- W4-20: the figure, which is MATLAB BADS's, with the caption above.
- The sheet gains entries for the three differences that O found recorded
  elsewhere: `len_scale` at D = 1 (W1-17), W3-10's refusal of a `sqrt_beta`
  that is not positive and finite, and W3-27's random choice when every
  acquisition value is NaN.
- The survey's two rows of B7 are closed by this ledger; slice O had none.

**Out of this pass:** nothing of this ledger. The plan's "Close" follows
wave 4 in a session of its own.
