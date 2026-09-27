<!-- Report of reviewer (c) of the doublecheck of wave 4, the user-facing documentation (briefs/wave4_doublecheck.md), reading PyBADS at 81385ac in /home/user/pybads-review and release 1.1.0 in /home/user/pybads-v1.1.0, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave4/doublecheck/c_docs/. -->


# Wave 4 doublecheck: the user-facing documentation

## 1. Coverage

**Read completely:**
- `CHANGELOG.md` at 81385ac: the whole `Unreleased` section (lines 6–708) and its diff from 6ceed6f. A word-level diff of every entry against 6ceed6f shows no text lost or duplicated, and the released part is unchanged.
- The ledger `verification/wave4.md`, in full.
- `git diff 6ceed6f 81385ac` over `pybads/`, `README.md` and `docsrc/`.
- The texts the pass touched, together with the code they describe: `init_sobol.py`, `acq_fcn_lcb.py` (with `check_sqrt_beta`), `rounding.py`, `force_to_grid`, `constraints_check.py`, `optimize_result.py`, `function_logger.py` (the class, `__call__`, `add`, `finalize`, `reset_fun_eval_time`), `search_hedge.py` (`__call__`, `update_hedge`), and in `bads.py` the docstring, the checks at creation, `_init_mesh_` and the final estimate.
- The changed option descriptions, plus those of `fun_eval_start`, `periodic_vars` and `n_search_iter`.
- The Fig. 1 captions in `README.md` and `index.rst`.
- The 1.1.0 counterparts of all of the above.
- MATLAB lines: `initSobol.m`, `searchHedge.m`, `searchES.m:100-130`, `uCheck.m`, `force2grid.m`, `setupvars.m:98-115`, `evalinitmesh.m:35-50`, `funlogger.m:120-130`, `bads.m:440-455` and `1125-1190`.

**Built:** the API pages for `acq_fcn_lcb`, `FunctionLogger`, `OptimizeResult` and `BADS`, from a minimal copy of `conf.py` in `docbuild/` (Sphinx 9.0.4, numpydoc 1.11, text builder).

**Skimmed:** fix report B (its note on `init_sobol`'s defaults), and the commit messages of 8daf7ad, 36e8b70 and fba29cd.

**Not reached:**
- The wave-3 claims inside the entries wave 4 extended, beyond spot checks. I spot-checked at 1.1.0: 31 evaluations for `max_fun_evals=30.5`; `accelerate_mesh_steps` values 0, -1, 2.5 and `True`; and an empty search stopping with `IndexError`. I did not check the Sto-BADS poll's first sentence, the ES scale's first sentence, the numbers under "Points evaluated again" or 1.1.0's `improvement_quantile` behaviour.
- `acqPortfolio.m`, the sheet, `AGENTS.md`, and the tests.

**Scripts and outputs** are in `/home/user/dc4/c_docs`:
- `fl_outputs`, `fl_finalize`, `opts_runs` (groups `hedge`, `nsi`, `sqrt_beta`, `periodic`), `hedge_decay_overflow`, `noisy_first_iter`, `design_start`, `fsd_not_estimate`, `init_sobol_sig`, `noise_test_neval`, `checks_v110` and `gp_empty_predict`, each with its `.out` where it has one.
- These ran at 1.1.0 and at 81385ac, one BLAS thread, every run of 200 evaluations or fewer.
- `checks_v110.out` comes from the script's first version, whose flaky constraint allowed 2 calls; `checks_v110_flaky.out` comes from the current version.

## 2. What holds

**Structure and the upgrade list**
- **Structure of `Unreleased`.** Headings are Upgrading, Changed, Fixed, in that order, with blank lines around each; the blank line before "### Fixed" is restored. No entry is duplicated or orphaned (89 titled entries, no repeats). Each wave-4 entry sits under the right heading.
- **"Upgrading from 1.1.0".** There are four new lines (`init_sobol`, complex values, hedge parameters, `n_search_iter`) and one changed line (`sqrt_beta`). Each has its entry and says what the entry says, with the exceptions of F4 and F7.

**Entries (verified at 1.1.0 and at 81385ac)**
- **W4-4.** At 1.1.0, `init_sobol` returned 2 for a 4-point design; at 81385ac it returns 4 (`design_start.out`, `init_sobol_sig.out`). Signature: see F7.
- **W4-8.**
  - 1.1.0 raised `TypeError` for an SD of `None` or `[0.5]`, and attached the "Error in executing the logged function" note to an array of several elements.
  - At 81385ac every malformed value or SD raises the documented `ValueError` with `Xn=-1` and `func_count=0` (nothing recorded), and an SD of one element is accepted (`fl_outputs.out`). Complex values: see F4.
- **W4-9.** In 1.1.0, `n_evals` stayed at 500 or 9 rows after `finalize`, `reset` made `fun_eval_time` `cache_size` rows long, and `n_evals[X_flag]` raised `IndexError`. At 81385ac every array has the same length (`fl_finalize.out`).
- **Empty `periodic_vars`.**
  - 1.1.0 refused `[]`, with a given or a random `x0`, and never raised the `IndexError` the ledger mentions, so no line for that part is needed.
  - At 81385ac, `[]` and `np.array([])` run; `[0]` and `[5]` raise `ValueError`, with a random `x0` too.
  - This matches MATLAB's `setupvars.m:107-108`.
- **W4-1.** At 81385ac every start gives one design per seed and two seeds give two designs (`design_start.out`), matching `initSobol.m:9-15`. The part about 1.1.0 is F2.
- **W4-6.**
  - 1.1.0's start row held `n_evals` 2 after the noise test; at 81385ac it holds 1 (`noise_test_neval.out`).
  - 1.1.0's budget was `min(max_fun_evals, n_train_max)`. At the default budget, where `n_train_max` binds, 1.1.0 was exactly one evaluation ahead.
  - "Still counts in `max_fun_evals` and `func_count`" holds.
- **W4-14.**
  - 1.1.0 raised `KeyError: 'yval_vec'` at `max_iter=1` (levels 1 and 2) and at `max_fun_evals=45`. At 38 it raised the NaN `ValueError` that "Small budgets" already covers.
  - At 81385ac these runs take 10 (or 5) final samples at the incumbent.
  - "MATLAB BADS takes none then" holds (`bads.m:1138`, `iter > 1`).
- **W4-15.** The text matches `bads.py:2583-2596` (`sto_poll == 0 and poll_best_improvement > 0`).
- **W4-19.** 1.1.0 behaved as the entry says:
  - `AttributeError` for `2.0`;
  - NumPy 0, -1, `[-1]`, `np.True_` and `np.complex128(2)` ran;
  - callables returning -1 or NaN ran;
  - an array or a string gave `IndexError` or `UFuncTypeError`.
  At 81385ac all of these are refused at creation, and callables at the search.
- **W4-21's rounding clause.** Holds: `uCheck.m:22,24`, `constraints_check.py:44-50`.
- **W4-25.** The text holds, except NaN (F5).
- **W4-27.**
  - 1.1.0 reached the "random search is performed" `warn` for an empty generation: `acq_fcn_lcb` returns `z` of size 0 on zero rows (`gp_empty_predict.py`).
  - At 81385ac the message is at DEBUG on the `BADS` logger, which `display="full"` shows.
- **W4-18 and W4-29 at 81385ac.** Every value tested is refused at creation, with the messages the text gives (`opts_runs_81385ac.out`). The part about 1.1.0 is F1.
- **W4-5.** The merge text holds, apart from F6.

**Rows with no changelog line: none is noticeable to a user of 1.1.0**
- W4-10, W4-12, W4-24 and W4-30 change only docstrings or descriptions.
- W4-16: one hyperparameter sample in every run.
- W4-17: removes an unused variable.
- W4-20: the caption only.
- W4-22: `import pybads` still imports `matplotlib.pyplot`, at 1.1.0 and at 81385ac.
- W4-23: the value reaches only `search_stats`.
- W4-26: the three-state `output_fcn` is unreleased, and the "Output function" entry stays true.
- W4-28: no effect.
- The records-only rows W4-2, W4-3, W4-7, W4-11 and W4-13 change no code. W4-2's platform effect is in "Initial design".

**Docstrings and descriptions**
- **Docstrings that hold:**
  - `init_sobol`: the power of two, the doubling, the plausible box, "only the size is read", the return values.
  - `acq_fcn_lcb`: "search and poll" (call sites `bads.py:1993` and `2443`, `es_search.py:178`); `check_sqrt_beta`.
  - `force_to_grid`, against `force2grid.m`.
  - `round_half_away`: 0.49999999999999994 goes to 0, where `floor(x + 0.5)` gives 1.
  - The comment in `contraints_check`, and `update_hedge`.
  - `FunctionLogger.add`, `finalize` and `reset_fun_eval_time`.
  - `OptimizeResult.overhead` (`evalinitmesh.m:41` calls `funwrapper`, the raw target; `bads.m:1186`) and `yval_vec`.
  - `BADS`'s Raises section, which covers `periodic_vars`; the new checks fall under its "for instance".
- **Option descriptions that hold:**
  - `cache_size`: the log grows by `ceil(Xn/2)`.
  - `hedge_gamma`, `hedge_beta`, `hedge_decay`.
  - `fun_eval_start`, against `_init_mesh_`.
  - `Options.descriptions` reads each of them whole.
- **Fig. 1 caption.** Identical in `README.md` and `index.rst`, and true:
  - `VariableTransformer` maps `plb` and `pub` to -1 and 1, for a log-scaled variable too;
  - at default, `n_max = 1`, so the poll basis is a signed permutation of the identity, and it is divided by `poll_scale`.
  - The caption says nothing of the log transform, as the ruling's clause doesn't either.
- **API page of `acq_fcn_lcb`.** It documents `acq_fcn_lcb` and `check_sqrt_beta` (the private helper is excluded), and both render cleanly.

## 3. Findings

### F1. "Search hedge parameters" misstates what 1.1.0 did with an infinite or NaN `hedge_beta` and with overflowing gains
- Where: `CHANGELOG.md:184-191` at 81385ac.
- Kind: false statement
- Severity: minor
- **What the entry says:** "1.1.0, like MATLAB BADS, ran with any value: … an infinite or NaN one made every choice random; a `hedge_decay` above 1 made the searches' gains grow until they overflowed, after which every choice was random."
- **What 1.1.0 did:** it has no fallback. `search_hedge.py:72` does `np.argwhere(...)[0]`; the fallback came with 06badd3, which is not in v1.1.0. So NaN probabilities stop the run with `IndexError` at the first search.
  - `hedge_beta` values `inf`, `nan` and `-1000`, and `hedge_decay=nan`, all stop at the first search (`opts_runs_v110_hedge.out`).
  - `hedge_decay` values 1e10, 1e30 and 1e100 stop at the search after the gains overflow (updates 30, 10 and 3; `hedge_decay_overflow_v110.out`).
  - Strings stop with `TypeError` or `UFuncTypeError`.
- The random choice is MATLAB's (`searchHedge.m`: `randi(nh)` when `find` is empty) and the unreleased code's; it was never 1.1.0's.
- Would the correction move results: no.
- Proposed correction: "1.1.0 ran with any number, as MATLAB BADS does: a `hedge_gamma` above 1/n made the search hedge favor the search of lower gain, and above 1/(n − 1) gave some searches a negative probability, so that they were never chosen; a negative `hedge_beta` inverted the hedge too, and a `hedge_decay` above 1 made the searches' gains grow, a negative one alternate in sign. An infinite or NaN `hedge_beta`, one far below 0, a NaN `hedge_decay`, or gains grown until they overflowed stopped the run with `IndexError` at the next search, where MATLAB BADS chooses a search at random; a string stopped it with `TypeError`."

### F2. "Initial design": 1.1.0's design did depend on the start, at or beyond the plausible box
- Where: `CHANGELOG.md:695-699` at 81385ac.
- Kind: false statement
- Severity: minor
- **What the entry says:** "In 1.1.0 the design depended on neither the seed nor the start: every start inside the plausible box gave one design…"
- **What 1.1.0 did:** it accepted starts at or outside the plausible box, and seeded the design from the integer parts of `u0`. At D = 2 with `plb=-3`, `pub=3` (`design_start.out`):
  - interior starts give one design (`3d5cc87930`);
  - a start on or below `plb` gives `116790c277`;
  - a start at `x0 = 3` or `4` (`u0` 1.0 and 1.333) gives `3772f598bf`;
  - `x0 = 9` (`u0 = 3`) gives `d5de8dd646`;
  - every one of these is the same for seeds 0 and 1.
- Would the correction move results: no.
- Proposed correction: "In 1.1.0 the design did not depend on the seed, and depended on the start only through the integer parts of its normalized coordinates: every start inside the plausible box gave one design for each number of variables, a start on or beyond a plausible upper bound gave another, and a start on or below a plausible lower bound gave one that could differ between x86 and arm64 machines."

### F3. "Scale of the evolution-strategy search": the first split's claims about 1.1.0 and about `n_search_iter` 2 do not hold
- Where: `CHANGELOG.md:620-624` at 81385ac.
- Kind: false statement
- Severity: minor
- **What the entry says:** that 1.1.0 drew the extra candidate "at the larger" of the two scales, and "The default `n_search_iter`, 2, is affected by neither."
- **What is true:**
  - `np.round` differs from MATLAB's `round` (`searchES.m:111`) only when `mu/2 = k + 0.5` with `k` even, that is `mu ≡ 1 (mod 4)`. For `mu` 1365 and 2045, 1.1.0 gives `[682, 683]` and `[1022, 1023]`; MATLAB and the port give `[683, 682]` and `[1023, 1022]`. For `mu` 1367, 2047 and 683 all three agree.
  - `mu = int(n_search / n_search_iter)`, so `n_search_iter = 2` with `n_search = 4090` gives `mu = 2045` and is affected. Only the defaults, 4096 and 2, are unaffected.
- Would the correction move results: no.
- Proposed correction: "… as MATLAB BADS does; PyBADS drew it at the larger when that number is one more than a multiple of 4 (1365 at `n_search_iter = 3`, for instance). The defaults, `n_search = 4096` and `n_search_iter = 2`, are affected by neither."

### F4. "1.1.0 accepted a NumPy complex value": only one whose imaginary part is zero, and the noise SD too
- Where: `CHANGELOG.md:68-70` and `683-685` at 81385ac.
- Kind: false statement
- Severity: minor
- **At 1.1.0** (`fl_outputs.out`):
  - `np.complex128(1+1j)` raised the documented `ValueError`;
  - `np.complex128(1+0j)` was accepted, with `ComplexWarning`;
  - with `specify_target_noise`, an SD of `np.complex128(0.5)` was also accepted;
  - a Python `complex(1, 0)` raised `TypeError` after writing the row.
- **At 81385ac** all of these raise `ValueError`.
- The Upgrading line names only the value, but a complex SD that 1.1.0 accepted is now refused too.
- Would the correction move results: no.
- Proposed correction for the Upgrading line: "A target that returns a value, or with `specify_target_noise=True` a noise SD, of a complex type raises `ValueError`, even when its imaginary part is zero, which 1.1.0 accepted for a NumPy complex number." For the entry: "…and accepted a NumPy complex value or SD whose imaginary part is zero."

### F5. `n_search_iter`: in 1.1.0 a NaN raised `ValueError`, not `TypeError`
- Where: `CHANGELOG.md:135-138` at 81385ac.
- Kind: false statement
- Severity: minor
- **What the entry says:** "`TypeError` for any float, whole numbers included."
- **At 1.1.0** (`opts_runs_v110_nsi.out`):
  - `nan` raised `ValueError: cannot convert float NaN to integer` (`search_hedge.py:58`);
  - 0.5, 2.0, 2.5, `inf` and 1e-300 raised `TypeError` or its subclass `UFuncTypeError`;
  - -2.5 raised `ValueError`, which the entry covers as a negative value.
- Would the correction move results: no.
- Proposed correction: "…`TypeError` for any other float, whole numbers included (`ValueError` for NaN)…"

### F6. "Repeated points…": "a run no longer evaluates a point again" is not true of the final samples
- Where: `CHANGELOG.md:211-213` at 81385ac.
- Kind: false statement
- Severity: minor
- **What is true:** with `specify_target_noise=True`, the case this entry is about, every noisy run evaluates its returned point `noise_final_samples` more times (`bads.py:1790-1795`). No run merges because those samples are not recorded (`record_duplicate_data=False`); "Points evaluated again" covers only the design, the search and the poll.
- Would the correction move results: no.
- Proposed correction: "`FunctionLogger` now merges it with its own earlier evaluation; the initial design, the search and the poll no longer evaluate a point again (see "Points evaluated again") and the final samples are not recorded, so that no run of `BADS` merges one."

### F7. `init_sobol`'s parameters became required; a 1.1.0 call that left out `lb` and `ub` now fails, and nothing in the changelog says so
- Where: `pybads/init_functions/init_sobol.py:7-15`, `CHANGELOG.md:66-67` at 81385ac; `fixes/B_W4-4_…md:28` and 8daf7ad's message ("no call could have used the defaults").
- Kind: defect of a fix (a stricter interface with no changelog line)
- Severity: minor
- **What happened:** 1.1.0 gave every parameter a type as its default, and `lb` and `ub` are unused. So `init_sobol(u0=…, plb=…, pub=…, fun_eval_start=10)` ran at 1.1.0 and returned `(16, 3)`. At 81385ac the same call raises `TypeError: missing 2 required positional arguments: 'lb' and 'ub'` (`init_sobol_sig.out`). The record's "no call could have used the defaults" does not hold for `lb` and `ub`.
- Would the correction move results: no.
- Proposed correction: give `lb` and `ub` a default of `None` (they are unused); otherwise add to the Upgrading line "…and takes `lb` and `ub`, which it does not read, as required arguments", and flag the record.

### F8. `OptimizeResult.fsd` names only a stop by `output_fcn` as a case where `fsd` is not an estimate
- Where: `pybads/bads/optimize_result.py:32-36` at 81385ac.
- Kind: false statement (incomplete)
- Severity: minor
- **What is true:** a noisy run that takes no final samples because of its budget also returns `noise_size` at level 1 (`fsd_not_estimate.out`):
  - `max_fun_evals=1`: `fsd` 1.0 or 2.5;
  - a budget that the start and the design use up (33 at D = 2): `fsd` 1.0 or 2.5, `yval_vec` None.
- The description of `yval_vec` already names that budget case.
- Would the correction move results: no.
- Proposed correction: "For a noisy run that takes no final samples, one that `output_fcn` stops in its initialization or whose `max_fun_evals` leaves no evaluation for them, it is not an estimate: …"

### F9. `FunctionLogger.__call__`'s Returns section: `SD` is `None` unless the logger takes SDs from the target
- Where: `pybads/function_logger/function_logger.py:93-96` (and the class's `fun`, `:14-17`) at 81385ac.
- Kind: false statement
- Severity: minor
- **What is true:**
  - At levels 0 and 1, `__call__` returns SD `None` (`fl_outputs.out`), and `idx` is `None` for an unrecorded evaluation of a point not in the log. W4-10 made `add`'s entry "float or None"; `__call__` still says `float` and `int`.
  - The class's `fun` returns an SD "optionally … if the function fun is stochastic", but it must return exactly `(f, sd)` at level 2 only.
- Would the correction move results: no.
- Proposed correction: "SD : float or None — the SD that the target returned, None when the logger takes none (`uncertainty_handling_level` below 2)"; "idx : int or None"; for `fun`: "and, when `uncertainty_handling_level` is 2, a tuple of the value and its (estimated) SD."

### F10. The statements on randomness: exact in effect, inexact in letter, and credited to 1.1
- Where: `pybads/bads/bads.py:142-143` and `136-137`, `docsrc/source/index.rst:23-26`, `README.md:23` at 81385ac.
- Kind: false statement / other
- Severity: minor
- **The letter.** After W4-1, `random_seed` decides every draw and nothing touches the global stream. But the design's scrambling draws come from the generator scipy creates from one integer drawn from `bads.rng`, not from `bads.rng` itself. `AGENTS.md` states this precisely; `rng`'s docstring says "The generator of every random draw of the run".
- **The 1.1 framing.** `README.md` and `index.rst` put "Every random draw of a run comes from one NumPy random generator, created from the `random_seed` option" under "What's new in PyBADS 1.1". The new "Initial design" entry says the reverse of 1.1.0 (its design ignored the seed).
- **A misplaced sentence.** "To obtain reproducible results …, set `options['random_seed']`" sits inside the description of `gamma_uncertain_interval`. It has been there since 0c56d86 and renders as part of that parameter.
- Would the correction move results: no.
- Proposed correction:
  - `rng`: "The generator of every random draw of the run, including the random `x0` and the one draw that seeds the scrambling of the initial design."
  - "What's new": qualify it for 1.1.0's design, or rewrite it at the next release.
  - Move the reproducibility sentence into the description of `options`.

### F11. Option descriptions: `search_acq_fcn` says "positive number" where the check requires a finite one; `periodic_vars` describes an option that `BADS` refuses
- Where: `pybads/bads/option_configs/advanced_bads_options.ini:183`, `26` and `87` at 81385ac.
- Kind: false statement
- Severity: minor
- **`search_acq_fcn`:** an infinite value, or a callable returning `inf`, is refused (`opts_runs_81385ac.out`), so "positive number" should read "positive finite number".
- **`periodic_vars`:** it reads "Array with indices of periodic variables, like periodic_vars = [1, 2]", but any non-empty value raises `ValueError`. This is recorded in the ledger's "Found while fixing" (C) and in `dev/TODO.md:258`, and it stays in the options page.
- **`n_search_iter`:** it does not state its new range, where the sibling `accelerate_mesh_steps` says "(a positive integer)".
- Would the correction move results: no.
- Proposed correction:
  - `search_acq_fcn`: "…or a positive finite number, checked when BADS is created; a callable returns a positive finite number, checked at each call."
  - `periodic_vars`: "Indices of periodic variables (not supported yet: BADS refuses any but None or an empty list)", with the text that `test_options.py` asserts updated.
  - `n_search_iter`: "Number of optimization iterations for search (a positive integer)."

### F12. `check_sqrt_beta` is a new public function with no changelog mention
- Where: `pybads/acquisition_functions/acq_fcn_lcb.py:74-100` and `__init__.py:1` at 81385ac; `CHANGELOG.md`, where no line mentions it.
- Kind: other
- Severity: minor
- **What happened:** the pass chose to make it public so the API page documents it, and the page renders it. The changelog lists the removal of the public `ESSearchCMA`, but not this addition.
- Would the correction move results: no.
- Proposed correction: a sentence in "LCB parameter of the search": "`pybads.acquisition_functions.check_sqrt_beta` makes the same check." Alternatively, an "### Added" line.

## 4. Outside my scope
- `README.md:93` ("drawn uniformly at random within the plausible bounds") and `docsrc/source/quickstart.rst:32` ("randomly drawn from the problems bounds") describe the random `x0` loosely. It is uniform in the transformed plausible box, as the `BADS` docstring says. Both predate wave 4.
- `BADS.optimize`'s docstring renders with numpydoc's underline warning and a docutils "Unexpected indentation" error. This predates wave 4.
- The References block of `ESSearchHedge`'s class docstring is malformed. This predates wave 4.
- Fix report B, line 28, and 8daf7ad's message: "no call could have used the defaults" does not hold (see F7). This is reviewer (d)'s record.
