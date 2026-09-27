<!-- Report of reviewer (c) of the doublecheck of wave 3, the user-facing documentation (briefs/wave3_doublecheck.md), reading PyBADS at 0d866e8 in /home/user/pybads-review and release 1.1.0 in /home/user/pybads-v1.1.0, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave3/doublecheck/c_docs/. -->


# Wave 3 doublecheck: the user-facing documentation

## 1. Coverage

**Read in full:**
- The CHANGELOG's `Unreleased` section. Wave 3's diff `68d4516..0d866e8` I read line by line; the rest I read to catch entries wave 3 may have made false.
- `verification/wave3.md`: rows W3-1 to W3-40, Notes, "Found while verifying", Rulings, "After the gates", and "Fix pass" (including "Choices within the rulings", "Changelog" and "Found while fixing").
- The non-test diff of `pybads/`, and all of `pybads/bads/option_configs/*.ini`.
- At 0d866e8: `acq_fcn_lcb.py`, `search_hedge.py`, `es_search.py`, `constraints_check.py`, `grid_functions.py` (`force_to_grid`), `poll_mads_2n.py`, the `BADS` class docstring (its Parameters and Raises), `_get_target_from_gp_`, most of `_poll_step_` and `_search_step_`, and `_is_poll_stop_`.
- `README.md` and `docsrc/source/index.rst` on the search and the poll, plus Fig. 1. There is no diff to README or docsrc over the pass.
- `docsrc/source/api/**`, including which modules the pages render.

**MATLAB at 74919c0, read:** `acqPortfolio.m`, `searchES.m`, `uCheck.m` and its callers (including `evalinitmesh.m:113`), `force2grid.m`, `ucov.m`, `pollMADS2N.m`, `acqLCB.m`, `setupvars.m:179-182`, and `bads.m` at 523-735, 826, 845-905, 956-990, 1040-1055, 1089-1092 and 1255-1340.

**1.1.0, compared by running it** (`/home/user/pybads-v1.1.0`) against 0d866e8.

**Gate records checked:** `batch1_vs_reference.md`, `w3-14_attribution.txt`, `w3-1_repeats.txt`; KD-B1-5 and KD-B3-3 of the sheet.

**Skimmed:** new test names in the diff; the verifier and reviewer text on W3-8.

**Not reached:**
- The fix agents' reports in full.
- A Sphinx build. I parsed the docstrings with numpydoc instead.
- Long runs where the search mesh reaches about 2^-42. My runs of up to 200 evaluations stop on `tol_fun` first, so the claim about halves at a fine grid rests on the attribution record.

**Scripts and outputs** are in `/tmp/claude-0/-home-user-pybads/cec8ef42-ff6e-5789-9637-a276f362470b/scratchpad/dc/c_docs/` (29 `.out` files; `*_v110.out` and `*_head.out`). All ran with one BLAS thread and gpyreg from the v1.3.3 clone, and each prints `pybads.__file__` and `gpyreg.__file__`.

## 2. What holds

- **Upgrading line on `sqrt_beta`.** Holds. At 1.1.0, `np.float64(0.0)`, `np.float64(-1.0)`, `np.True_`, `np.complex128(1+0j)` and `array([0.])` ran; at head each raises `ValueError` at iteration 2, the first search (`opts_check_py_beta_*.out`).
- **"LCB parameter of the search".** Holds. At 1.1.0, `2.0` and `2` raised `AttributeError` at the first search; at head they run. `inf`, `True` and `'x'` raise `ValueError`. The docstring says the third return value is an SD and describes the callable and the `None` schedule correctly (`acqLCB.m:10-20`).
- **Upgrading lines on `improvement_quantile` and `accelerate_mesh_steps`.** Hold: 0, 1, -0.5, 1.5, 0, -1, 2.5, `inf` and `True` are refused when `BADS` is created; 3.0 is converted. What the entry says 1.1.0 did does not hold in full (F1, F2).
- **`ESSearchCMA`.** Holds. At 1.1.0 it failed with `TypeError` (a slice index) in `_initialize_`, and `search_method` `'ES-cma+'` was refused by the hedge (`cma_110_v110.out`). Nothing under `docsrc/`, `README.md` or `examples/` names it.
- **"Search without a candidate" extension.**
  - The gains decay: the code does `self.g *= self.decay`. MATLAB (`bads.m:720-723`, `acqPortfolio.m:40-68`) gives the chosen search er = 0 and the others er/Inf.
  - An emptied second generation now runs, with the new warning 25 times. At 1.1.0 the same run stopped with `IndexError` (`empty_gen_py_*.out`).
  - MATLAB `searchES.m:162-175` keeps `zold`.
- **"Rebuilds of the local GP" (replaced).** Holds.
  - 1.1.0 kept `reset_gp` set until the end of the poll and rebuilt at every poll step (`bads.py:1584`, `1828`, `2033`, `2303` at v1.1.0).
  - Head sets it at every pass while `poll_moved` is true (`bads.py:1470-1471`, `2551`).
  - MATLAB matches: `bads.m:523`, `707`, `826`, `956-958`, `1049`.
- **"Descriptions of the options" extension.** Holds.
  - `tol_poi`: at 0, `p_less > 1` never holds; an unreliable GP stops a good poll; `complete_poll` bypasses the stop (`_is_poll_stop_`, `bads.py:2641-2667`; `bads.m:877-895`).
  - `sloppy_improvement` floors the improvement at `tol_fun` (`bads.py:1391-1394`).
  - `poll_scale` is read only by the poll (divided out and multiplied back) and by ES-ell (`es_search.py:283`).
  - Options parses all the new descriptions whole (`descs_head.out`).
- **`hedge_gamma=0`.** Holds. 1.1.0 raised `ValueError` at the first search (`update_hedge:130`); head completes. MATLAB's `u(min(iHedge,end),:)` is one row.
- **`uncertain_incumbent=False`.** Holds. 1.1.0 raised `AttributeError` at the first poll; head runs. The target is `fval - tol_fun`, as `bads.m:1328-1332`.
- **"Rounding to the search grid".** Halves go away from zero, as `force2grid.m:5` does. With wider or infinite hard bounds, 1.1.0 started at `[0, 4]` and head at `[2, 4]` (see F5 for the equal-bounds case). 31 of 540 runs, all 6-D, 10-D or noisy, and no flag in any of the 54 tests all match `batch1_vs_reference.md` and `w3-14_attribution.txt` (31 of 31 attributed).
- **"Scale of the evolution-strategy search".** "3 or more" holds: at 3 the off-by-one alone changes the count in 43 of 87 updates. "The default, 2, is not affected" holds. The explanation does not fully hold (F3).
- **"Root logger".** Holds. At 1.1.0, `ESSearchWM()` added a stdout handler to the root logger; at head it adds none, and `BADS()` still calls `basicConfig` (`bads.py:221`) (`geterr_logger_py_*.out`).
- **"NumPy's error handling".** Holds. After a 1.1.0 run `divide` is `'ignore'`; after a head run it is `'warn'`. Nothing else in `pybads/` or gpyreg calls `np.seterr`.
- **"NaN acquisition values".** Holds: `nanargmin` with a guard, where 1.1.0 used `argmin`; MATLAB's `min` skips NaN.
- **"Covariance of ES-wcm", "Offspring", "Ties".** Hold against the 1.1.0 code (`floor(mu + 1)`, the `cw` mask, unstable `argsort`) and against MATLAB `searchES.m:57-62`, `ESupdate.m` and its stable `sort`.
- **"Points evaluated again".**
  - Holds for the initial design too: with `x0` on a design point, 1.1.0 evaluated it a third time and head does not (`init_dups2_py_*.out`). MATLAB checks the design as well (`evalinitmesh.m:113`).
  - 137/1411 and 100/8800, with 0 after, match `w3-1_repeats.txt`. Those counts were taken at the pass's `c276d79`, not at 1.1.0.
  - 1.1.0 also repeats on a comparable edge sphere (78 of 1430 evaluations; head 0 of 1395, `edge_repeats_py_*.out`).
- **"Search hedge in noisy runs".** The formula holds. At level 0, `fs == 0` gives the `max(0, ·)` branch. The range stated is not accurate (F4).
- **"Early stop of the poll".** Holds. 1.1.0's code is `np.sort(f_pi)[::-1][:D+1]` on an (n, 1) column; MATLAB is `bads.m:868-869`. "Rare" matches 5 of 540 runs.
- **"Length scales on two points".** Holds. 1.1.0 has the same code (`gaussian_process_train.py:323` at v1.1.0). Head skips the update when max == min, so the previous prior stays (centre and width).
- **Rows without a line** (W3-2, W3-3, W3-23, W3-33, W3-34, W3-36). None is noticeable by a 1.1.0 user:
  - W3-2 and W3-3 are comments.
  - W3-23 acts only after a non-finite prediction, which no run has shown.
  - W3-33 changes `optim_state` fields that only the new `output_fcn` copy reads, and its entry promises only "a copy".
  - W3-34 and W3-36 remove dead code.
- **`_get_target_from_gp_` docstring.** Now true: it covers noisy runs and `uncertain_incumbent`, calls the value an SD, gives the shapes and describes the fallback.
- **`poll_mads_2n` docstring.** Its basis description, `n_max`, the `pollMADS2N.m:7` citation, the note that the permutation only reorders directions, and the MADS reference are all true.
- **README and index.rst.** The poll text ("one direction at a time") still matches the coordinate poll at default; W3-24's text went with the revert.
- **`BADS` Raises section.** Its "for instance" covers the new option checks.

## 3. Findings

### F1. The changelog misstates how 1.1.0 failed with an invalid `accelerate_mesh_steps`
- Where: `CHANGELOG.md:101-105` at 0d866e8.
- Kind: false statement.
- Severity: minor.
- **Stated:** "1.1.0 stopped the run with `TypeError` at its first failed poll for 0 or a negative value, as MATLAB BADS stops, and with `IndexError` for a float."
- **What 1.1.0 did** (`acc_seeds_v110.out`, `opts_check_py_acc_v110.out`). 1.1.0's condition is `iter > accelerate_mesh_steps` (`bads.py:2241` at v1.1.0).
  - 0: `IndexError` at a failed poll of the second iteration, in 12 of 12 runs (2 targets × 6 seeds).
  - Negative values: `TypeError` at the first poll, 12 of 12.
  - 3.0 and 2.5: `IndexError` once the iteration exceeds them.
  - `inf` and `1000.0`: the run completed with no acceleration. `True` ran as 1. All of these are now refused, except 1000.0, which is converted.
- The upgrading line covers these cases, but the entry's account of 1.1.0 is wrong.
- **MATLAB accepts `Inf`:** `iter > Inf` is never true (`bads.m:976`). The ledger does not record that PyBADS now refuses a value MATLAB accepts (see Outside my scope).
- Would the correction move results: no.
- Proposed correction: "1.1.0 stopped the run at a failed poll: with `IndexError` for 0 (from the second iteration on), with `TypeError` for a negative value at a failed first poll, and with `IndexError` for a float once the iterations exceeded it; a larger float, `inf` included, ran without the accelerated reduction, and `True` ran as 1. MATLAB BADS stops for 0 and negative values too."

### F2. "At 0 or 1 never moved the incumbent" is false for a noisy run at 1
- Where: `CHANGELOG.md:99-101`.
- Kind: false statement.
- Severity: minor.
- **Stated:** "1.1.0 ran with it, and at 0 or 1 never moved the incumbent."
- **Evidence** (`iq_moves_v110.out`, which counts `_update_incumbent_` calls after initialization, 2-D sphere, seed 0):
  - Deterministic at 0 and at 1: 0 moves. The run ends at the best point of the initial design, `[-0.94, 1.99]`, not at `x0`.
  - Noisy at 0: 0 moves.
  - Noisy at 1: 107 moves. `-sqrt(2)*erfcinv(2)` is +inf, so every search succeeds, and the run stays in its first iteration until the budget is spent.
- Would the correction move results: no.
- Proposed correction: "1.1.0 ran with it: at 0, and at 1 without noise, the incumbent never left the best point of the initial design; a noisy run at 1 moved it at every search."

### F3. The scale did not "grow where MATLAB BADS's shrinks" until the fifth generation
- Where: `CHANGELOG.md:560-561`.
- Kind: false statement.
- Severity: minor.
- **Stated:** "PyBADS counted older candidates as new from the third generation on, so that the scale grew where that of MATLAB BADS shrinks."
- **The update rule:** the scale is multiplied by `exp(es_beta*(frac-0.2))`, with `es_beta = 1` (`searchES.m:185`). It grows whenever `frac > 0.2`.
- **Check:** `frac_signs*.py`, which counts MATLAB's fraction and 1.1.0's on the same pools during real runs.
  - Generations 3 and 4: MATLAB's median fraction is 0.43 and 0.26, never below 0.2 in 71 to 91 updates. Both scales grow; 1.1.0's grows faster.
  - Generations 5 to 7 (`n_search_iter=8`): MATLAB's fraction is below 0.2 in 38, 62 and 65 of 71 updates, while 1.1.0's is about 0.87 to 0.91.
- So the claim holds only from `n_search_iter` 6 on. The review's own reproduction also contradicts it: MATLAB's fractions there are 0.56, 0.43, 0.39 and 0.37, all above 0.2.
- Would the correction move results: no.
- Proposed correction: "…counted older candidates as new from the third generation on, so that the scale grew faster than MATLAB BADS's, and from about the fifth generation grew where MATLAB BADS's shrinks."

### F4. "2.5 to hundreds of times" understates the hedge's excess reward
- Where: `CHANGELOG.md:594-596`.
- Kind: number does not recompute.
- Severity: minor.
- **Stated:** a search whose point was worse than the incumbent was rewarded "2.5 to hundreds of times more than MATLAB BADS does".
- **Ratio of 1.1.0's reward to MATLAB's**, with γ = (fval_old − f)/fs (`reward_ratio_head.out`):

  | γ | 2 | 1 | 0 | −1 | −2 | −3 | −4 | −5 |
  |---|---|---|---|---|---|---|---|---|
  | ratio | 1.2 | 1.5 | 2.51 | 7.9 | 48 | 424 | 5736 | 1.3e5 |

- The factor has no upper bound for worse points.
- Would the correction move results: no.
- Proposed correction: "rewarded a search whose point was worse than the incumbent 2.5 times or more than MATLAB BADS does, the more the worse the point (424 times at three standard deviations)."

### F5. The rounding example does not hold at 1.1.0 when the hard bounds equal the plausible box
- Where: `CHANGELOG.md:549-552`.
- Kind: false statement.
- Severity: minor.
- **Evidence** (`half_start_py_*.out`):
  - With hard bounds ±4096 or ±inf, 1.1.0 started at `[0, 4]` and head starts at `[2, 4]`.
  - With hard bounds ±2048, equal to the plausible box, 1.1.0 moved the plausible bounds inward (`bads:TooCloseBounds`) and started at `[1.996, 3.992]`.
  - The test uses ±4096.
- Would the correction move results: no.
- Proposed correction: "`x0 = [1, 3]` in a plausible box `[-2048, 2048]` within wider hard bounds starts at `[2, 4]`, where 1.1.0 started at `[0, 4]`."

### F6. `acq_fcn_lcb`'s docstring, which the docs site renders, is not in numpydoc form, and its summary is false
- Where: `pybads/acquisition_functions/acq_fcn_lcb.py:7-37`. `docsrc/source/api/functions/acquisition_functions.rst` renders it with `automodule`.
- Kind: false statement / other.
- Severity: minor.
- **Summary line:** "It retrieves the point at the lower confidence bound". The function returns the LCB values at `xi`, as its rewritten Returns section now says. This is recorded under "Found while fixing" (A) but not fixed.
- **Form** (`npdoc_head.out`):
  - The parameters are written `name: type`. numpydoc parses them as names with no type, for example `'sqrt_beta: None, callable or float'`, and the Returns entries as types with no names.
  - The `==========` underlines give numpydoc "wrong underline length" warnings for Returns and for the Raises section the pass added.
  - The reference style, `optimize_result.py`, uses `name : type` and `-` underlines.
- Would the correction move results: no.
- Proposed correction: summary "Lower confidence bound of the GP at the points `xi`."; `Parameters` / `Returns` / `Raises` with `-` underlines of the header's length, and entries such as `xi : np.ndarray`, `sqrt_beta : None, callable or float`, `z : np.ndarray`.

### F7. `_poll_step_` still says "the LTMADS poll direction method"
- Where: `pybads/bads/bads.py:2156` (also `:2168`).
- Kind: false statement.
- Severity: minor.
- **Stated vs true:** the reverted `poll_mads_2n` docstring (`poll_mads_2n.py:17-24`) says the default poll is a signed permutation of the identity, stepping along one coordinate at a time, and that PyBADS keeps MATLAB's poll rather than LTMADS's directions. `_poll_step_`'s summary contradicts that.
- `:2168` also calls `f_sd_poll_best` an "Estimated GP variance"; it is an SD. That part is recorded under "Found while fixing" (B).
- Would the correction move results: no.
- Proposed correction: "…performs the poll step, along the directions of `poll_mads_2n` (the coordinate directions at default)…"; ":2168" to "Estimated GP standard deviation at the best poll point."

### F8. Leftovers in `poll_mads_2n`'s docstring
- Where: `pybads/poll/poll_mads_2n.py:23-24` and `:37-38`.
- Kind: other / false statement.
- Severity: minor.
- **Pointer the reader can't follow:** "(the port review's wave 3, W3-24)" names a record the docstring gives no path to. It is there because of the review's history.
- **Wrong shape:** `poll_scale` is documented as "of shape `(1, D)`". The poll passes `gp.temporary_data["poll_scale"]`, which is `ll.flatten()` or `np.ones(D)` (`gaussian_process_train.py:573`, `:1080`), shape `(D,)`. Broadcasting accepts both.
- Would the correction move results: no.
- Proposed correction: drop the parenthesis, or cite `dev/experiments/port_review_20260925/verification/wave3.md`, W3-24. Write "of shape `(D,)` or `(1, D)`".

### F9. Other docstrings of functions the pass touched are still false
- Where:
  - `constraints_check.py:18-20`: "Return a new incumbent that satisfies the boundaries". It returns the candidate set, with evaluated, duplicate and infeasible points removed. Recorded (C).
  - `search_hedge.py:123`: "Update the probability of improvement". It updates the gains with the expected reward. Recorded (A).
  - `search_hedge.py:23`: `lambda x: np.sum(x.^2, 1) > 1`, MATLAB syntax that is not Python. Not recorded; the `BADS` docstring had the same fix in wave 2.
  - `es_search.py:91-107`: `ESSearch.__call__` has placeholders. Recorded (C).
  - `bads.py:2708-2710`: `f_target_s` is documented as the GP's SD. With `uncertain_incumbent=False` at level 0 it is `np.zeros(1)`. Not recorded.
  - `force_to_grid` (`grid_functions.py:8`) has no docstring.
- Kind: false statement.
- Severity: minor. None of these is rendered on the docs site.
- Would the correction move results: no.
- Proposed correction: rewrite each from the code, for example for `contraints_check`: "Project or drop the candidates outside the bounds, remove duplicates and points already evaluated (within `tol_mesh`) and those violating `non_box_cons`; return the remaining candidates, sorted as MATLAB's `setdiff`." Use `np.sum(x**2, axis=1) > 1` in the hedge docstring.

### F10. Option descriptions of the search and the poll describe effects that no code produces
- Where: `advanced_bads_options.ini`, the comment lines above `poll_method` (31), `skip_poll` (49), `search_improve_frac` (94), `search_optimize` (118), `poll_acq_fcn` (182) and `acq_hedge` (186). The options page includes these verbatim.
- Kind: false statement. The descriptions predate wave 3 but belong to the search and the poll.
- Severity: minor.
- **Evidence** (`misc2_head.out`):
  - `skip_poll=False`, `search_optimize=True`, `poll_method='nonexistent'`, `poll_acq_fcn=('acq_LCB', 50.0)` and `search_improve_frac=0.5` each give the default run's result exactly.
  - `acq_hedge=True` stops the run with `UnboundLocalError` (`method`, `bads.py:2027-2036`) at the first improved search.
- KD-B1-5 records the first five as having no effect; the options page does not say so, where other such options carry "(unused)".
- Would the correction move results: no.
- Proposed correction: append "(unused, in MATLAB BADS too)" to `skip_poll` and `search_improve_frac`, "(unused: the poll always uses poll_mads_2n)" to `poll_method`, "(unused: the poll always ranks by LCB with the default schedule)" to `poll_acq_fcn`, "(not ported: no effect)" to `search_optimize`, and "(not supported)" to `acq_hedge`.

### F11. Two option descriptions are silent on what the pass now accepts or refuses
- Where: `advanced_bads_options.ini:79` (`improvement_quantile`) and `:183` (`search_acq_fcn`).
- Kind: other (an omission).
- Severity: minor.
- **`accelerate_mesh_steps`** gained "(a positive integer)".
- **`improvement_quantile`** did not gain its range, which `BADS` now enforces.
- **`search_acq_fcn`** does not say that its second element is `sqrt_beta`. The changelog tells users they can pass a number there, and the error message names only `sqrt_beta`.
- Would the correction move results: no.
- Proposed correction:
  - `# Quantile when computing improvement, greater than 0 and less than 1 (<0.5 for conservative improvement)`
  - `# Acquisition fcn for search stage, ('acq_LCB', sqrt_beta): sqrt_beta None (the default schedule), a callable sqrt_beta(t, D) or a positive number`

### F12. Two entries are worded for a reader who has the code
- Where: `CHANGELOG.md:265-267` and `:575-577`.
- Kind: other.
- Severity: minor.
- **"The search evaluates the best candidate of the earlier generations"** is true when the emptied generation is the last, as at the default `n_search_iter` of 2. With 3 or more, later generations reproduce from the kept candidates, and the best of all generations is evaluated (MATLAB does the same).
- **"The ⌊μ⌋ best points"** uses a μ that the changelog never defines: half the GP's training points.
- Would the correction move results: no.
- Proposed correction:
  - "…the search keeps the candidates of the earlier generations, as in MATLAB BADS, and evaluates the best one…"
  - "…from the best half of the Gaussian process's training points (⌊n/2⌋ of n), one per weight…"

## 4. Outside my scope

- **(b) `accelerate_mesh_steps=inf`.** MATLAB treats `Inf` as no acceleration (`iter > Inf`, `bads.m:976`), and 1.1.0 did the same. PyBADS now refuses it. The ruling's "not a positive integer" covers it, but the ledger does not record the departure from MATLAB.
- **(b) `improvement_quantile='0.3'`.** It raises `TypeError` (`'<' not supported`), not the `ValueError` that the `BADS` Raises section and the other option checks give (`bads.py:813-818`).
- **(d) KD-B3-3 says "`acq_hedge=True` does nothing".** It stops the run with `UnboundLocalError` at the first improved search (see F10).
- **(a)/(d) The W3-8 ledger row and B3-C F7.** Both say "from 4 on the scale grows where MATLAB's shrinks". MATLAB's fraction stays above 0.2 through generation 4 (see F3).
- **(d) Fig. 1 note in "Found while fixing" (D).** It says Fig. 1 is wrong to draw anisotropic poll steps. The poll's steps are isotropic in u but scale with the plausible box in x, so the figure's axes can show unequal steps; the note's premise does not establish a defect.
- **(d) Changelog counts in "Points evaluated again".** They are those of the pass's `c276d79`, not 1.1.0. The magnitude is similar at 1.1.0 (78 of 1430 evaluations on a comparable problem), so the entry stands.
