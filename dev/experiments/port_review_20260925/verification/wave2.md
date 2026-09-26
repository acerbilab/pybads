# Wave 2 ledger: B1 and B2

The verified findings of wave 2 of the port review
([plan](../../../plans/port-correctness-review.md), "Wave 2 pickup"): slice
B1, the setup of a run and its result, and slice B2, the course of a run,
each read on both tracks. The reports are `reviews/B1_internal.md`,
`reviews/B1_comparison.md` (with `reviews/B1_comparison_history.md`, its
reviewer's re-dating on the complete history), `reviews/B2_internal.md` and
`reviews/B2_comparison.md`. Each slice was verified by a fresh Opus agent
that had not written either report, with checks of its own:
`wave2_B1_verifier.md` and `wave2_B2_verifier.md`. The verifiers were also
given the items of the review's records that belong to their slice and that
the reviewers did not receive (the plan's "Wave 2 pickup", step 4), as
B1-K1 to B1-K8 and B2-K1 to B2-K9. Every agent read PyBADS at `fef6c14`
(PI, 2026-09-26: `dev-next` after the fix passes of waves 0 and 1), MATLAB
BADS at `74919c0` and gpyreg v1.3.3 (`98ab5a4`), in a cloud session (Linux,
Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, where the fingerprint of
`dev/scripts/fingerprint.py` at `fef6c14` is `91f947f78e1087c2`, the Linux
reference's). The scripts and their outputs are under
`scripts/wave2/<slice>_<track>/` and `scripts/wave2/<slice>_verifier/`,
formatted by the pre-commit hooks after they ran. The reports cite them at
the sandbox's scratch paths.

The sandbox's clone was shallow (oldest commit `ce3a0b3`, 2022-11-22) when
the reviewers began; the complete history was fetched while they ran, and
the verifiers dated every row on it. A reviewer's date of "`ce3a0b3`" means
"by that date".

Lines are at `fef6c14`, MATLAB lines at `74919c0`. "B1-I F1" is finding F1
of the B1 internal report, "B1-C" the B1 comparison report, and likewise
"B2-I" and "B2-C"; "B1-K1" is a kept item. "Survey" names the row of the
candidate table of `dev/results/2026-09-23-codebase-survey.md` that
describes the same behavior. The dispositions are proposals until the PI
rules: the verifier's recommendation unless the row says why not. The gate
is the one of `AGENTS.md`, "Numerical gates", that a fix would need. "Default
run" says whether a run at default options reaches the code, and at which
uncertainty level (0 deterministic, 1 noise inferred, 2
`specify_target_noise`); "inputs" means that no option is needed, only
arguments of that form.

None of the configurations of the benchmark's `default` and `oned` suites
reaches the effective bounds of W2-4: over seeds 0-29 no start point and no
plausible bound lies within `1e-3 (ub - lb)` of a finite hard bound
(orchestrator, `scripts/wave2/orchestrator/reach_eff.py`), and the
fingerprint's runs keep clear of it too. A gate that must reach it needs a
suite of its own.

## B1: setup, options, defaults, bounds, transform, result

| Id | Source | What | Classification | Default run | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|
| W2-1 | B1-I F1, B1-C F2 (mixed case), B1-K1 | the half-bounds check takes `any` over all variables (`bads.py:533-544`), so any problem that mixes fully bounded and fully unbounded variables is refused, against the docstring (`bads.py:64-66`) and the error message, which mean a test per variable; MATLAB has no such check and accepts both. With the test per variable the mixed problem runs, and W1-1's code already replaces infinite bounds variable by variable | confirmed port discrepancy (the `and`/`or` slip of `AGENTS.md`, "MATLAB logicals") | inputs (mixed bounds), all levels | never agreed (`c7c88ab`, unchanged; MATLAB never had it) | `bads.py:500-507` | fix: `np.any(np.isfinite(lb) != np.isfinite(ub))`; the example of `AGENTS.md` gains a second instance | fingerprint; a test with mixed bounds |
| W2-2 | B1-C F2 (half-bounded case), B1-K1 | a variable bounded on one side only is refused (`bads.py:533-544`), as the message says; MATLAB accepts it (`setupvars.m:27-38` only cautions), and no record decides the difference. Lifting it needs W2-4 first: the effective bounds put a range of 1e3 in place of the infinite one and move `x0` and `plb` by a unit | design question | inputs (half bounds) | never agreed (`c7c88ab`) | the same row | PI: support half bounds as MATLAB (after W2-4), or keep the refusal and put it on the sheet. Proposed: support them, since the transform and W1-1's geometry already handle infinite bounds variable by variable, and only runs refused today change | fingerprint and a test either way |
| W2-3 | B1-I F2, B1-C F5 | scalar bounds are not replicated when D > 1 (`bads.py:181-184`, `380-400`), as the docstring (`bads.py:63`) and `boundscheck.m:7-10` do: `ValueError` at construction | confirmed port discrepancy | inputs (scalar bounds, D > 1) | never agreed (MATLAB `42ae029`, 2017; Python `c7c88ab`) | — | fix: broadcast scalars to (1, D) before the shape test | fingerprint |
| W2-4 | B1-C F1, B1-I F4, B1-I F3 | `_bounds_check_` computes effective bounds 1e-3 of the linear range inside each hard bound (`bads.py:450-531`), clamps `x0` into them, moves `plb`/`pub` inside them and refuses a plausible box that the margin leaves empty; MATLAB never moves them (`boundscheck.m:12-16`, `setupvars.m:79-99`). On `[1e-3, 1e3]` with `plb` omitted the plausible box loses three decades; `[1, 10]` loses its log transform; `x0` at the optimum 0.01 starts at 1.001; `[0, 1]` with `plb = 1e-4`, `pub = 5e-4` is refused. I-F3's test (the "x0 inside the plausible box" check compares with the effective bounds) holds as a description, but its proposed direction is not MATLAB's: outside the margin PyBADS already equals MATLAB | confirmed port discrepancy | yes, for bounded problems with `plb`/`pub` omitted (they default to `lb`/`ub`, `bads.py:181-184`) or near a hard bound, or `x0` near a bound; all levels. Not reached by the benchmark or the fingerprint | never agreed (`c7c88ab`; the `plb` default from `9037851`; MATLAB never; the identifiers suggest VBMC's `boundscheck`, not checked) | — | fix: remove the block and keep MATLAB's checks (the order, `x0` within the hard bounds); do not adopt I-F3's direction | fingerprint (unchanged: its runs keep clear); a population comparison on a suite that reaches it (bounded problems with `plb`/`pub` omitted, a log-scaled variable, `x0` on a bound) |
| W2-5 | B1-I F5 | the transform's self-test (`variables_transformer.py:207-231`) uses an absolute tolerance, 1e-6, and refuses valid bounds of large magnitude (\|b\| ≳ 1e10 linear, `ub` ≳ 1e9 log); `transvars.m:30`, `169-178` has the same test, and both sides refuse the same draws | confirmed shared defect | inputs (bounds of large magnitude) | the two agree (MATLAB `6c93629`, 2017; Python `c7c88ab`) | — | keep MATLAB's test, and put it in `matlab_side_defects.md` as a shared observation; a relative tolerance only if the PI prefers it | none if kept; fingerprint if changed |
| W2-6 | B1-I F6, B1-C F9, B2-C F12, B1-K2 | a non-empty `fun_values` stops `BADS()` with `ValueError`: `not np.isreal(X)` on an array (`bads.py:755-761`), `range(len())` (`787`), and `self.function_logger`, which is created at `290`, after `_init_optim_state_` (`286`). MATLAB imports the evaluations into the logger (`setupvars.m:126-167`, `funlogger.m`). B2-C adds that, once repaired, `_init_mesh_` would take its incumbent over the imported points, where MATLAB takes it over `x0` and the design | confirmed port discrepancy (never worked) | no (the default `{}` skips it) | never agreed (`c7c88ab`; the loop `8e59038`) | `bads.py:722-770` | refuse a non-empty `fun_values` with a message now (a stricter interface, which costs no working script), and keep the port as a `TODO.md` item, whose incumbent is then taken over `x0` and the design only, as MATLAB (B2-C F12); the verifiers also offer the port now | fingerprint; a test of the refusal |
| W2-7 | B1-I F6, B1-C F9, B2-I F5, B2-C F8 | `f_vals` (PyBADS-only, KD-B1-4) sets `cache_active`, which selects a display format of 8 fields that `_display_function_log_` fills with 6 or 7 values: `ValueError` at the first display line, with `display="off"` too (the string is formatted before the logger). Its values reach only `optim_state["cache"]`, which nothing reads | confirmed defect (a PyBADS-only option that cannot work) | no (`f_vals` given) | `c7c88ab`, `8ff10f5` | — | remove the option or refuse it with a message (a stricter interface), and correct KD-B1-4 | fingerprint |
| W2-8 | B1-I F7, B1-C F12 | a multi-row `x0` passes the checks (without bounds, `plb`/`pub` are estimated from the set, `bads.py:340-378`, a branch MATLAB BADS lacks), and `optimize()` then fails with a broadcasting error (`u0` flattened, `bads.py:696`); MATLAB refuses it (`boundscheck.m:18-27`) | confirmed port discrepancy | inputs (N0 > 1) | never agreed (`c7c88ab`) | — | fix: refuse N0 > 1, and drop the estimation branch | fingerprint |
| W2-9 | B1-C F12 | `x0=None` with `lb`/`ub` and no plausible bounds is accepted, and the start drawn in the (moved) hard box; MATLAB (`bads.m:331-342`) and PyBADS's own "Raises" section refuse it | confirmed port discrepancy | inputs | matched at `c7c88ab`; diverged in `9037851`, which set the plausible-bound default before the test | — | fix: test `x0 is None` before the default (a stricter interface: a changelog entry and an "Upgrading from" line) | fingerprint |
| W2-10 | B1-C F12, B1-I F10 (the docs) | the check of `non_box_cons`'s output (`bads.py:546-556`) accepts (N, k) and gives `IndexError` or `AttributeError` on a scalar; MATLAB requires N×1 with a clear error (`setupvars.m:11-25`). The docstring's example `lambda x: np.sum(x.^2,1)>1` (`bads.py:81`) is MATLAB syntax, and the N×D-in, (N,)-out contract is unstated | confirmed port discrepancy (the check); confirmed defect (the docs) | no | the check matched at `c7c88ab`, weakened in `8e59038`; the docs `c7c88ab` | — | fix: accept (N,) or (N, 1) and refuse anything else with a message; correct the docstring | fingerprint |
| W2-11 | B1-I F10 | with `x0=None` and `non_box_cons`, a random start that violates the constraint stops `BADS()` (4 of 20 seeds on a disc in a square); MATLAB draws the start in `setupvars.m:83-85` and refuses it in `evalinitmesh.m:22-26` just the same | confirmed shared defect | no (`x0=None` with `non_box_cons`) | both since November 2022 (Python `d466948`, MATLAB `019f0b4`) | — | PI (the verifier leaves the design open): redraw, start from the best feasible point of the design, or keep MATLAB's error and document it. Proposed: redraw in the plausible box until the constraint holds, up to a cap, then the current error; an entry in `matlab_side_defects.md` | fingerprint; a test |
| W2-12 | B1-I F8, B1-C F7, B1-K3 (`status`, `exit_flag`) | `status` is in the result's key list (`optimize_result.py:62-84`) and on KD-B1-8's list, but never set: `result["status"]` raises `KeyError`. `bads.py:1436-1464` computes `exit_flag` only in comments and one unused assignment; MATLAB returns `exitflag` (0 budget, iterations or output function; 1 mesh; 2 stall; `bads.m:423`, `1062-1083`) | confirmed defect; contradicts KD-B1-8 | yes, every run | `8ff10f5` (2022-11-04) created `OptimizeResult` with `status` commented out; MATLAB's exit flag never ported | `optimize_result.py` | fix: set `status` to MATLAB's exit flag, and correct KD-B1-8 | fingerprint (it hashes `x`, `fval`, `func_count`, `yval_vec`) |
| W2-13 | B1-I F8, B1-C F7, B1-K3 (`success`) | `success` is `True` in every run, at the budget or the iteration limit too (`optimize_result.py:159-162`, a TODO); scipy's `OptimizeResult`, which the class names as its model, has `success` False at an iteration limit | design question (KD-B1-8 leaves it open) | yes, every run | `8ff10f5` | `optimize_result.py` | PI: `success = status > 0`, MATLAB's and scipy's convention (a change a script can see: a changelog entry and an "Upgrading from" line); or keep `True` and document it. Proposed: `status > 0`, with W2-12 | fingerprint |
| W2-14 | B1-I F9, B1-C F6 | `OptimizeResult.__setitem__` deep-copies every value (`optimize_result.py:182-186`), `fun` and `non_box_cons` included: a bound method or callable object is copied with its instance, and one that holds a lock makes `optimize()` raise after all its evaluations, losing the result. MATLAB stores `func2str(fun)` (`bads_output.m:4`) | confirmed defect (Python-only) | the copy every run; the failure only with such a callable | `8ff10f5` | — | fix: store `fun` and `non_box_cons` by reference | fingerprint |
| W2-15 | B1-I F11, B1-C F11, B2-I F7, B2-C F7 | `display` is compared as an exact string (`bads.py:224-232`): `"notify"` and `"final"`, which the option's description lists, and `"none"`, `"OFF"` or `"Iter"` all leave the logger at INFO, the full display; `"full"` is not described. MATLAB lower-cases the first three letters: notify 1, off/none 0, iter/all 3, final 2 (`bads.m:312-328`) | confirmed port discrepancy (KD-B2-3 settles the mechanism, not the levels) | no (the default is `"iter"`) | never agreed (`c7c88ab`; MATLAB 2017) | — | fix: MATLAB's mapping, with the opening and final messages at a level above the iteration lines, and `"full"` in the description | none (not numerical) |
| W2-16 | B1-C F4, B1-I F12 (part) | `search_factor_min` is read by nothing: after each failed search `_update_search_stats_` (`bads.py:2759-2767`, slice B3's code) multiplies the search factor by `sqrt(0.5)` with no floor, where MATLAB takes `max(SearchFactorMin, searchfactor*SearchScaleFailure)` (`bads.m:1366`). The factor scales the ES covariance on both sides; at D = 6 a round's late searches run at 0.177 against MATLAB's 0.5 | confirmed port discrepancy | yes, D ≥ 2, all levels | never agreed (MATLAB `6c93629`, 2017; Python `c7c88ab`) | — (difference (a) the preparatory agent saw in passing; wave 0's "Found while verifying" left it to B3, wave 3) | fix: the floor, in this wave's fix pass (verified here; wave 3's reviewers then read the fixed code); not on KD-B1-5, since MATLAB reads it | population comparison at default |
| W2-17 | B1-C F3, B1-K4, B2-C F1 | `tol_noise` is `eps · tol_fun`, 2.2e-19 (`advanced_bads_options.ini:13`, read at `bads.py:1037`), where MATLAB has `sqrt(eps) · TolFun`, 1.5e-11 (`bads.m:195`, `evalinitmesh.m:43`): a target whose repeat at `x0` differs in its last bits (a summation in varying order, a jitter of 1e-12) runs as noisy, with 100 evaluations where MATLAB's threshold gives 55 and a smaller error | confirmed port discrepancy (a slip of `sqrt(eps)`) | yes (`uncertainty_handling=None`), level 0 → 1 for nearly deterministic targets | never agreed (MATLAB `fd3f7a2`, 2017; Python `c7c88ab`) | — (difference (b) the preparatory agent saw in passing) | fix: `np.sqrt(np.spacing(1.0)) * tol_fun`, and the description's "variabitility" | fingerprint (its targets repeat exactly or are declared noisy); a test with a nearly deterministic target; the benchmark's targets are unaffected |
| W2-18 | B1-C F10 | MATLAB's checks of the options are not ported: `MaxFunEvals` a positive integer, and a warning for `ImprovementQuantile > 0.5` (`setupoptions.m:71-78`). `max_fun_evals=0` fails with "cannot convert float NaN to integer", `30.5` runs 31 evaluations | confirmed port discrepancy | no | never ported (MATLAB 2017-2018) | — | fix: refuse a `max_fun_evals` that is not a positive integer (a stricter interface) and warn for `improvement_quantile > 0.5` | fingerprint |
| W2-19 | B1-I F13, B1-C F8 | a user value of `None` replaces the default (`options.py:48-51`) where MATLAB's empty value stands for it (`setupoptions.m:5-9`): numeric options then fail with `TypeError`, some in `optimize()`, and `nonlinear_scaling=None` turns the log transform off; MATLAB-style strings are truthy, so `uncertainty_handling='off'` gives a noisy run | design question (KD-B1-3 leaves `None` open and settles verbatim values) | no (a user value) | `c7c88ab` | — | PI: (i) `None` stands for the default, except for the options where `None` is a value (`uncertainty_handling`, `noise_size`, `random_seed`, ...); (ii) the boolean options refuse a non-boolean (a stricter interface); or both. Proposed: both, with KD-B1-3 updated, a changelog entry and an "Upgrading from" line for (ii) | fingerprint |
| W2-20 | B1-C F13, B2-I F6 | `overhead` divides by `total_fun_eval_time`, to which neither the evaluations with `record_duplicate_data=False` (the final samples) nor a merged level-2 repeat add (`function_logger.py:384-398`, `424-431`); MATLAB's `funlogger` adds the time of every evaluation but the noise test (`funlogger.m:130`) | confirmed port discrepancy (informational) | yes, levels 1 and 2 | never agreed (`c7c88ab`; the duplicate paths `8ff10f5`, `9915bbf`) | — | fix: add the evaluation's time on both paths; the noise test stays out, as in MATLAB | fingerprint |
| W2-21 | B1-K5 | the random `x0` drawn in the original plausible box | no longer holds: `0c56d86` (W0-5) draws in the transformed box, and the draw is MATLAB's (200 of 200 starts identical on the same uniform numbers), except where W2-4 moves `plb` first | — | — | — (difference (c) the preparatory agent saw in passing) | none beyond W2-4 | — |
| W2-22 | B1-K6 (and both reports' answers to Q1) | `_read_config_file` splits a comment line at its first `=` or `:`, so `Options.descriptions` and `str(options)` cut 8 descriptions (`noise_size`, `periodic_vars`, `stobads_frame_size_scaling_power`, `gp_samples`, `gp_sample_widths`, `stable_gp_samples`, `upper_gp_length_factor`, `temperature`); the docs include the `.ini` files verbatim | confirmed, inert (cosmetic) | — | `c7c88ab` | — | fix (`delimiters=("=",)` and the comment lines rejoined) | none |
| W2-23 | B1-K7 (and B1-I's answer to Q1) | nine options have no description line (`n_basis`, `search_factor_min`, `gp_cov_fun`, `use_effective_radius`, `gp_fixed_mean`, `hessian_method`, `hessian_alternate`, `hedge_beta`, `hedge_decay`; MATLAB's `defopts` lacks one for several of them), and 36 descriptions end in MATLAB's closing `'` | confirmed, inert (cosmetic; the docs show these lines) | — | `c7c88ab` | — | fix: a description for each, and the quotes removed | none |
| W2-24 | B1-K8 (both reports' defaults tables) | `search_n_try` is a NumPy float (`np.maximum(D, np.floor(3 + D/2))`), and `search_count` starts as it; every read compares it with an integer count, exactly | confirmed, inert | — | `c7c88ab` | — | cast it to `int` when the file is next touched | none |

## B2: main loop, termination, noisy re-evaluation, final estimate

B2's findings that repeat one of B1 are in the rows above: `tol_noise`
(W2-17, B2-C F1), `f_vals` (W2-7, B2-I F5, B2-C F8), the display levels
(W2-15, B2-I F7, B2-C F7), `overhead` (W2-20, B2-I F6) and `fun_values`
(W2-6, B2-C F12).

| Id | Source | What | Classification | Default run | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|
| W2-25 | B2-I F1, B2-C F4, B2-K1 | after the re-estimation, a better earlier iterate gives the incumbent its `yval`, `fval`, `fsd` and target hyperparameters, and `self.u`, but its location goes to `self.best_u` (`bads.py:1536`), a name nothing reads; `u_best` stays, and the next pass sets `self.u = self.u_best` (`1406`). MATLAB does the same (`bads.m:1111-1118` sets `u`, not `ubest`; `769`): the next search is centred at the chosen iterate with its target predicted at the old incumbent, and a poll that no search precedes with an improvement runs around the old point, judged against the chosen iterate's value (9 of 22 polls in 6 noisy runs; 25 moves). The line was written in `c7c88ab` with "# TODO in Matlab is not done" | confirmed shared defect (the line itself: confirmed, inert) | yes, levels 1 and 2 | the behaviour has matched MATLAB since `c7c88ab`, only because `best_u` is dead; MATLAB since 2017 | `bads.py:1405` | PI: (a) keep MATLAB's behaviour and delete the dead line; or (b) move the incumbent with its value (`_update_incumbent_`), a departure from MATLAB that changes noisy runs (4 of 10 changed, no direction). Proposed: (b), gated, with an entry in `matlab_side_defects.md`, since the internal track finds the pair (location, value) of two different points and the port's own TODO saw it | (a) fingerprint; (b) population comparison of the noisy configurations |
| W2-26 | B2-I F2 | the target hyperparameters that the move sets are overwritten at the end of every pass (`bads.py:1425`), so they reach the next search's target, which nothing reads, and not the poll (0 of 22; 25 of 25 with `search_n_try=0`); MATLAB's `fhyp = gpstruct.hyp` (`bads.m:1059`) does the same | confirmed, inert (shared) | runs at levels 1 and 2, without effect | the two agree (both first versions) | — | keep; revisit with W2-25 (b) | none |
| W2-27 | B2-I F3, B2-C F2, B2-K5 | the initial design is capped at `max_fun_evals - 1` (`bads.py:1073-1078`) before `init_sobol` rounds it up to a power of two and doubles it when that equals D (`init_sobol.py:71-76`), so a small budget is exceeded, and in a noisy run the reserve for the final samples goes negative and raises `max_fun_evals` (`bads.py:1175-1184`); both sides also leave the noise test's evaluation out of the cap. D = 2, `max_fun_evals` 3 to 6: 6 evaluations, MATLAB 4; noisy D = 2 at 25: 34, MATLAB 22; at 38: the run ends after its design, with no final sample. The division by zero of wave 0's record no longer happens (W1-19) | confirmed port discrepancy (the rounding after the cap); confirmed shared defect (the noise test not counted) | no (a budget below the design: about `2 + 2^ceil(log2 D)`, or 34 plus the final samples in a noisy run) | never agreed (both in `c7c88ab`, not `cdc2e0f` as B2-I says; MATLAB 2017) | — (difference (g) the preparatory agent saw in passing; waves 0 and 1, "Found while verifying") | fix: cap the design after its rounding, at the evaluations left (the noise test counted), and floor the reserve at 0; whether the doubling stays is W0-18's question, wave 4. An entry in `matlab_side_defects.md` for the noise test | fingerprint; a test of small budgets |
| W2-28 | B2-I F4, B2-C F6 | `sloppy_improvement=False` stops every run at its first pass: without `np.maximum` the sufficient improvement is a Python float, and `.copy()` raises `AttributeError` (`bads.py:1339-1349`); MATLAB supports the option (`bads.m:504-507`) | confirmed port discrepancy | no | `c7c88ab` | — | fix | fingerprint; a test |
| W2-29 | B2-I F13, B2-C F3 | the accelerated mesh reduction (in the poll, slice B4's code) is tested when `optim_state["iter"] > accelerate_mesh_steps` with the 0-based count (`bads.py:2421-2446`), MATLAB's `iter > steps + 1`; `8e59038` corrected the same shift for `max_iter` and the stall criterion only. At the iteration where only MATLAB tests, MATLAB halves the mesh again in 4 of 8 default runs | confirmed port discrepancy | yes, all levels | never agreed (`c7c88ab`; MATLAB 2017) | — | fix (`>=`), in this wave's fix pass (verified here; wave 3's reviewers read the fixed code) | population comparison at default |
| W2-30 | B2-C F5, B2-K3 (the current iterate) | a failed rebuild of the current iterate in the re-estimate keeps its recorded estimate (`bads.py:2811-2815`), where MATLAB's `gppred` gives NaN | intentional difference, missing from the sheet (the PI's ruling on W1-35; the changelog, the docstring, `test_noisy_re_estimate_after_failed_rebuild`) | only when that rebuild fails, levels 1 and 2 | `fef6c14` | — | keep, and a sheet entry (KD-B2-4) | none |
| W2-31 | B2-C F9 | the display's Actions column shows the last action appended (`bads.py:2473-2484`, `2892`), a stale "Train", or "Skip" where the poll trained and skipped; MATLAB rebuilds `action` at each poll (`bads.m:1016-1028`). A stale entry in 1 of 8, 3 of 11 and 2 of 12 polls of default runs | confirmed port discrepancy (display) | yes, `display="iter"` | never agreed (`c7c88ab`, `8e59038`) | — | fix: the action built per poll | none |
| W2-32 | B2-C F10 | a run that ends in its initialization (`max_fun_evals=1`, or `output_fcn` stopping at `"init"`) reports `iterations = 0`, MATLAB 1 (`bads.m:482`, `bads_output.m:21`); KD-B1-8 says the count is MATLAB's, from 1, and `test_one_function_evaluation` and `test_output_fcn_stops_run_at_init` pin 0 | design question; contradicts KD-B1-8 for this case | no | `95da7f1` (#71) | — | PI: report 1 as MATLAB (the two tests change), or keep 0 and correct KD-B1-8. Proposed: keep 0, which says that no iteration ran, and correct the entry | none |
| W2-33 | B2-C F11 | the output function: a stop has a message of its own, and a false return at `"init"` cannot reopen a run that ended (#71), where MATLAB keeps the stale message and assigns the return value; the `"init"` call comes after the noisy option changes and the first GP fit, MATLAB's before them (`bads.m:427`, `431-457`) | intentional difference, missing from the sheet (the message and the stop, #71); design question (the timing of `"init"`) | no (`output_fcn` given) | the `"init"` placement `c7c88ab`, the rest `95da7f1` | — | keep, with a sheet entry that includes the timing; moving the call means splitting `_init_optimization_`, for nothing a run computes | none |
| W2-34 | B2-C F13 | with one final sample at level 1, `yval_vec` is `np.vstack` of the sample and `yval`, shape (2, 1) (`bads.py:1606-1612`), where every other case is (n,) and MATLAB gives a 1×2 row (`bads.m:1465`) | confirmed port discrepancy | no (`noise_final_samples=1`, level 1) | `2650aef` (2022-11-04) | — | fix (`np.append`) | fingerprint (its noisy runs take 10 samples); a test |
| W2-35 | B2-I F12, B1-I F12 (part), B2-K2 | `min_iter` and `min_fun_evals` are read by nothing, and MATLAB has no such options; they came with PyVBMC's options in `c7c88ab`. Their descriptions reach the documentation | confirmed, inert (leftovers) | no effect | `c7c88ab` | `optimize_result.py` (its part) | correct the record: add them to KD-B1-5 (d); removing them would need an "Upgrading from" line, since an unknown name raises | none |
| W2-36 | B2-I F10 | a noisy run's incumbent keeps the raw minimum of its initial design, a biased order statistic, with `fsd = noise_size`, through two iterations, since the re-estimate starts at `poll_iteration > 0` (`bads.py:1503-1507`); MATLAB's `iter > 1` (`bads.m:1097`) is the same | design question, shared with MATLAB | yes, levels 1 and 2 | the two agree (both first versions) | — | keep MATLAB's behaviour, as a shared observation in `matlab_side_defects.md` | none if kept; population comparison of the noisy configurations if changed |
| W2-37 | B2-K6 | a thin feasible band (`\|x1 - x2\| <= 0.005`) at D = 2: no point of the design is feasible, the GP is trained on `x0` alone (gpyreg's `get_bounds_info` warns; the `ValueError` of wave 0's record is gone since W1-26), no search runs (no more than D points), every axis-aligned poll point is infeasible, and the stall criterion ends the run at iteration 6 after 2 evaluations, at `x0`. At D = 3 the run converges; a noisy run at D = 2 goes on after 8 polls without an evaluation. MATLAB's poll basis is axis-aligned at these meshes too, with the same criterion | design question, shared; needs MATLAB (its GP on one point, `log(std(y)) = -Inf`) | no (`non_box_cons`) | — | — | PI: document that a feasible region thinner than the mesh can resolve ends the run on `tol_fun`, or keep the stall criterion from counting iterations without an evaluation, a departure from MATLAB. Proposed: document it | none if documented |
| W2-38 | B2-K7 | `IterationHistory._expand_array` reassigns the grown array through `__setitem__`, which deep-copies it with every stored GP: n(n+1)/2 copies (20100 for 200 records, 1.7 s) | confirmed, inert (time only) | yes | `c7c88ab` | — | fix: grow the array without the copy | fingerprint |
| W2-39 | B2-K9 | the NaN estimates of past iterates whose last re-estimates failed stay in `iteration_history` after the run (W1-35 as ruled; MATLAB's `iterList` too), while the result is finite; the changelog says so | confirmed, inert | only on failure, levels 1 and 2 | `fef6c14` | — | keep, and say it on the `IterationHistory` documentation page | none |
| W2-40 | B2-K3 (the other clauses), B2-K4 | the stored GPs rebuilt in place; a failed rebuild recording the restored GP's value for a past iterate; the swap handing on a slot without its markers; the stored GP's `len_scale` and `effective_radius` choosing the neighbours | no longer hold: `e004c79` (W0-1) re-estimates from a copy of the working GP, whose geometry a rebuild without a refit leaves as it was, as MATLAB's `gpupdate`; `fef6c14` (W1-35) records NaN for a past iterate. W1-35's clauses hold (`bads.py:1525-1527`, `1581-1582`); at these sizes the stored geometry would have changed no neighbour set either | — | — | the `_re_evaluate_history_` row (at `676083d`); the second clause of the `_save_gp_stats_` row | correct the survey's rows | — |
| W2-41 | B2-I F8 | with 0 or 1 final samples the final message prints the last incumbent's observation, "GP mean ± SEM" for a sample at level 2, "from 2 samples" at level 1; MATLAB's messages are the same (`bads.m:1136`, `1172-1181`, `1464-1465`) | not a defect as a port matter (shared); the shape is W2-34 | no | shared | — | keep | none |
| W2-42 | B2-I F9 | termination is checked at every pass, so `max_iter` counts a round once begun and the stall criterion can end a run on a search pass: the exact 0-based transcription of `bads.m:1062-1085` | not a defect | yes | shared | — | keep; the descriptions of `max_iter` and `tol_stall_iters` may say that a round counts once begun | none |
| W2-43 | B2-I F11 | the move and the final choice skip the first iterate: MATLAB's own change `75ec49f` (2022-05-09, "Skip first"), ported in `8e59038` | not a defect | yes, levels 1 and 2 | MATLAB `75ec49f`; Python `8e59038` | — | keep | none |
| W2-44 | B2-K8 | the loop discards the GP that `_poll_step_` returns; the poll returned the same object in every case checked (39 of 39) | confirmed, inert | yes | — | — | keep, or assign the return value | fingerprint if changed |

## Notes on the reports

- **Corrections by the verifiers.** B1: I-F3's proposed direction (test the
  start against the plausible box) is not MATLAB's (W2-4); I-F5 is shared
  with MATLAB (W2-5); the refusal of a random start of I-F10 is shared, not
  a discrepancy (W2-11); the display format of `f_vals` has 8 fields, not 7
  (W2-7); 8 descriptions are cut, not 3 (W2-22); the docstring sentences
  that B1-I dates to `ce3a0b3` are from `c7c88ab`. B2: I-F1 is MATLAB's
  behaviour, not a port discrepancy (W2-25); I-F2, I-F8, I-F9 and I-F11 are
  MATLAB's too (W2-26, W2-41 to W2-43); B2-I dates the rounding of the design
  to `cdc2e0f`, where it is `c7c88ab` (W2-27); C-F5 and C-F11 are
  intentional (W2-30, W2-33).
- **The differences the preparatory agent saw in passing**, kept from the
  reviewers: (a) `search_factor_min` was found by B1-C F4 from the defaults
  table, though its code is B3's (W2-16); (b) `tol_noise` by B1-C F3 and
  B2-C F1, and noted by B1-I (W2-17); (c) the random `x0` no longer holds
  (W2-21); (g) the budget below the design by both B2 reports (W2-27). Of
  the differences it left off the sheet because no record decides them, the
  three of these slices were found: `cache_size` (B1-C: harmless, a log
  that grows against MATLAB's circular buffer of 1e4), `_bounds_check_`
  doing more than `boundscheck.m` (W2-4, W2-8) and a user value of `None`
  (W2-19).
- **Code of later slices, found here and verified:** W2-16 (B3's
  `_update_search_stats_`) and W2-29 (B4's poll). Both are proposed for this
  wave's fix pass; wave 3, which has its pass ahead, then reviews the fixed
  code.
- **The sheet.** KD-B1-8 is contradicted twice: `status` is never set
  (W2-12), and a run that ends in its initialization reports 0 iterations
  (W2-32). KD-B1-4 lists `f_vals` as read by code (W2-7). Entries to add:
  the current iterate's estimate after a failed rebuild (W2-30), the output
  function's message and stop (W2-33), and the refusal of a start that
  `non_box_cons` rejects once put on the grid (`bads.py:684-693`, `1bee482`,
  "Check gridizied non-box-cons": deliberate by its commit, no MATLAB
  counterpart; B1 verifier, V6); `min_iter` and `min_fun_evals` go on
  KD-B1-5 (d) (W2-35); `search_factor_min` must not, since MATLAB reads it.
  No finding contradicts another entry.
- **Minor records, without a row,** for the docstrings of the fix pass
  (both B1 reports, confirmed by the B1 verifier where it checked): a
  negative seed raises `ValueError` where the docstring says `TypeError`,
  and `True` is accepted as seed 1; the warning "Estimatingplausible"; the
  order message says `<` where the test is `<=`; `result["x0"]` is the
  start before it is put on the grid, not the point evaluated;
  `total_time` leaves out the setup; the doubling of
  `mesh_overflow_warning` and the level-0 default of `noise_size`,
  `sqrt(tol_fun)`, are in no description.
- **Tests.** `test_transform_inverse_largeN` builds `np.ones((10 ^ 6, D))`,
  12 rows (`^` is XOR), not a million (both B1 reports);
  `test_variable_transformer.py` has no log transform, infinite bound or
  large magnitude; no test reaches `_bounds_check_`'s moves and refusals;
  the final-estimate tests recompute the implementation's formulas, and
  nothing checks the incumbent after a move (W2-25).
- **A proposal for grouping the fixes.** Moving results at default, each
  under its own population comparison on Linux against
  `population_linux_wave1_20260926` (this box computes as its environment:
  the fingerprint at `fef6c14` is its `91f947f78e1087c2`): W2-16, W2-29, and
  W2-25 if (b) is ruled. W2-4 moves no run of the benchmark and needs a
  suite that reaches it. The rest must move nothing, each under the
  fingerprint: W2-1, W2-3, W2-6 to W2-15, W2-17 to W2-20, W2-22 to W2-24,
  W2-27, W2-28, W2-31, W2-34, W2-38, and the records (W2-30, W2-32, W2-33,
  W2-35, W2-39, W2-40).

## Survey rows of B1 and B2

Every open row of the candidate table that belongs to these slices is
closed by a row above: `bads.py:500-507` (W2-1, W2-2); `bads.py:722-770`
(W2-6); `optimize_result.py` (W2-12, W2-13, W2-35); `bads.py:1405`
(W2-25); `_re_evaluate_history_` at `676083d` (W2-40, W2-30); and the
second clause of the `_save_gp_stats_` row, left to this wave by wave 1
(W2-40).

## Found while verifying

Met by a verifier, outside the reports' findings and the kept items, and
not verified beyond the small check named; each goes to the wave of its
slice, or is proposed here.

- `force_to_grid` rounds halves to even (`np.round`,
  `search/grid_functions.py:8-12`), MATLAB's `force2grid.m` away from zero:
  `x0 = 1` in a plausible box `[-2048, 2048]` starts at 0 in PyBADS and at 2
  in MATLAB (B1 verifier, N1, `v_round.py`); how often an exact half occurs
  inside a run is not measured. Slice B3, wave 3.
- A deterministic result's `fsd` is the `int` 0 (B1 verifier, N2; not
  traced, perhaps the search's `f_sd_search = 0`); with the result's
  records of the fix pass (W2-12).
- A start of `±inf` with a finite bound is refused by the bounds check,
  where MATLAB replaces a non-finite start by a random point
  (`setupvars.m:79`; B1 verifier, N4, by reading). Slice B1: proposed with
  W2-4, since both touch the checks of the start.
- `optim_state["lastreeval"]` (`bads.py:799`) is written and never read,
  beside the `last_re_eval` that is (B2 verifier). With W2-38.
- On a one-point training set gpyreg's `get_bounds_info`, called from
  `_gp_hyp`, emits RuntimeWarnings (B2 verifier, `v_thin_warn.py`, with
  W2-37; wave 1's fix agents saw the same helper replace the targets). Slice
  B6, whose wave has passed: a `TODO.md` line, or with W2-37.
- `contraints_check` returns its candidates in `np.unique`'s order, not
  their own (the sorted variant is commented out; B2 verifier, by reading).
  Slice B3, wave 3.
- At level 2 with one final sample the final message prints an array,
  `[0.345]`, where MATLAB prints a number (B2 verifier). With W2-41.

## Rulings (PI, 2026-09-26)

The orchestrator proposed a disposition for every row, following its
verifier's recommendation unless the row says why not. Before the ruling it
set out the ten rows left to the PI with their context and a
recommendation each (W2-2, W2-5, W2-11, W2-13, W2-19, W2-25, W2-32, W2-33,
W2-36, W2-37), and revised two of the rows' proposals: W2-5 from "keep" to a
relative tolerance, since the absolute test fails from rounding alone on
bounds where the transform is exact to machine precision, and W2-9 from
"refuse" to "keep accepting it", since the docstring advises `plb = lb`
where in doubt and the code defaults to exactly that. The PI accepted every
recommendation, the two revisions included. As in waves 0 and 1, a fix is
one commit per row on the wave's branch, with a test that fails at
`fef6c14` and passes at the commit, and a changelog line in every commit a
user can notice; a stricter interface also has an "Upgrading from" line.
The fix pass is not started.

**Fix, moving nothing** (each under the fingerprint):

- W2-1: the half-bounds test per variable; the example of `AGENTS.md`,
  "MATLAB logicals", gains this instance.
- W2-3: scalar bounds broadcast to (1, D).
- W2-5 (revised): the transform's self-test with a relative tolerance,
  `1e-6 · max(1, |b|)`; it only accepts bounds refused today. An entry in
  `matlab_side_defects.md`, as a shared defect that PyBADS fixes.
- W2-6: a non-empty `fun_values` refused with a message; the port is a
  `TODO.md` item.
- W2-7: `f_vals` refused with a message, and KD-B1-4 corrected.
- W2-8: a multi-row `x0` refused, and the estimation of the plausible bounds
  from a starting set removed.
- W2-9 (revised): `x0=None` with only `lb` and `ub` stays accepted (with
  W2-4, the start is then drawn in the transformed hard box); the "Raises"
  section of `BADS` is corrected.
- W2-10: `non_box_cons`'s output accepted as (N,) or (N, 1) and refused
  otherwise with a message; the docstring's example written in Python, and
  the contract (N×D in, one violation per row out) stated.
- W2-11: a random start that violates `non_box_cons` is redrawn in the
  plausible box, up to 1000 draws, then today's error; runs whose first draw
  is feasible use the same draws. An entry in `matlab_side_defects.md`.
- W2-12 and W2-13: `status` is MATLAB's exit flag (0 at the budget, the
  iteration limit or a stop by the output function; 1 at `tol_mesh`; 2 on
  the stall criterion), KD-B1-8 corrected, and `success = status > 0`. All
  540 runs of the Linux reference end on the stall criterion (407) or
  `tol_mesh` (133), so no default run changes `success`; a run that ends on
  a limit the user set reports False (an "Upgrading from" line). A
  deterministic result's `fsd`, the `int` 0 ("Found while verifying"), is
  traced and made a float in the same commit.
- W2-14: `fun` and `non_box_cons` kept by reference in the result.
- W2-15: MATLAB's display levels (the first three letters, lower case), with
  the opening and final messages above the iteration lines; `"full"` in the
  option's description.
- W2-17: `tol_noise = sqrt(eps) · tol_fun`, and the description's typo; a
  test with a nearly deterministic target.
- W2-18: a `max_fun_evals` that is not a positive integer refused, and a
  warning for `improvement_quantile > 0.5`.
- W2-19: a user value of `None` stands for the default (for the options
  whose default is `None`, nothing changes), and the boolean options refuse
  a non-boolean with a message; MATLAB's `'on'`/`'off'` are not converted,
  since KD-B1-3 settles that values are used verbatim. KD-B1-3 updated.
- W2-20: `overhead` counts the final samples and the merged level-2
  repeats; the noise test stays out, as in MATLAB.
- W2-22 and W2-23: the descriptions of the option files (whole, present,
  without MATLAB's quotes); W2-24: `search_n_try` cast to `int` in the same
  commit.
- W2-27: the initial design capped after its rounding, at the evaluations
  left with the noise test counted, and the reserve for the final samples
  floored at 0; an entry in `matlab_side_defects.md` for the uncounted noise
  test. Whether the doubling stays is W0-18's question, wave 4.
- W2-28: `sloppy_improvement=False`.
- W2-31: the display's actions built per poll.
- W2-34: `yval_vec` of shape (2,) with one final sample at level 1; the
  number, not an array, in the final message at level 2 ("Found while
  verifying").
- W2-38: `IterationHistory` grows without the deep copy; the unread
  `optim_state["lastreeval"]` goes in the same commit.
- W2-2: half-bounded variables supported as MATLAB, after W2-4 and W2-1;
  tests with and without the log transform, and a check of a few seeds on a
  half-bounded problem before the commit (not a gate), for code paths that
  have not met a one-sided infinity.

**Fix, moving results**, in this order, each ending in a population
comparison on Linux against `population_linux_wave1_20260926`, whose
environment this sandbox has (the fingerprint at `fef6c14` is its
`91f947f78e1087c2`), against the end of the step before; the fingerprint
recorded at every commit:

1. W2-16: the floor `search_factor_min` on the search factor after a failed
   search (B3's code), at default.
2. W2-29: the accelerated mesh reduction tested from MATLAB's iteration
   (B4's code), at default.
3. W2-25 (PI: (b)): the incumbent moves with its value
   (`_update_incumbent_`), and the dead `best_u` line goes; a comparison of
   the noisy configurations. If it flags a worsening, the row comes back to
   the PI, and MATLAB's behaviour goes on the sheet. An entry in
   `matlab_side_defects.md`.
4. W2-4: the effective bounds removed, MATLAB's checks kept, I-F3's
   direction not adopted; with it the start of `±inf` that MATLAB replaces
   by a random point ("Found while verifying"), after a check that it holds.
   The benchmark does not reach the change: its gate is a comparison on a
   suite of its own (bounded problems with `plb`/`pub` omitted, a
   log-scaled variable, a start on a bound), with the fingerprint
   unchanged. W2-2 follows it.

**Keep, and record:**

- W2-26: kept, as MATLAB; revisited with W2-25 if its comparison says so.
- W2-30: kept, on the sheet (KD-B2-4, the current iterate's estimate after
  a failed rebuild, by the ruling on W1-35).
- W2-32 (PI: keep 0): a run that ends in its initialization reports 0
  iterations; KD-B1-8 corrected to say so.
- W2-33: kept, on the sheet (the stop's message, a false return at
  `"init"`, the timing of the `"init"` call).
- W2-35: `min_iter` and `min_fun_evals` on KD-B1-5 (d).
- W2-36 (PI: keep MATLAB's): a shared design observation in
  `matlab_side_defects.md`; if W2-25's comparison moves the noisy runs, it
  can be measured as a separate step.
- W2-37 (PI: document): the documentation of `non_box_cons` says that a
  feasible region thinner than the mesh can resolve ends the run early, and
  suggests a reparametrization. MATLAB's GP on one point is not needed for
  this disposition, so no MATLAB run is written up.
- W2-39: the NaN estimates on the `IterationHistory` documentation page.
- W2-21 and W2-40: the survey's rows corrected.
- W2-41, W2-42 (the descriptions of `max_iter` and `tol_stall_iters` say
  that a round counts once begun), W2-43, W2-44: kept.
- The sheet also gains the refusal of a start that `non_box_cons` rejects
  once on the grid (`1bee482`), and the minor records of "Notes on the
  reports" are corrected in the docstrings of the pass.

**Out of this pass:** `force_to_grid`'s rounding of halves and the order of
`contraints_check`'s candidates go to slice B3, wave 3, which has its pass
ahead. gpyreg's `get_bounds_info` on a one-point training set belongs to
B6, whose wave has passed: a `TODO.md` line, with wave 1's note on the same
helper. The `fun_values` port is a `TODO.md` line (W2-6).
