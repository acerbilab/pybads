<!-- Report of reviewer (b) of the doublecheck of wave 3, the fixes of B4 (briefs/wave3_doublecheck.md), reading PyBADS at 0d866e8 in /home/user/pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave3/doublecheck/b_B4/. -->


# Wave 3 doublecheck: the fixes of B4

## 1. Coverage

**Read completely**
- `verification/wave3.md`: the B4 rows, "Notes", "Rulings" with "After the gates", "Fix pass" with its choices, and "Found while fixing".
- The diffs of every commit in my scope: `5f31837`, `dac062e`, `ec1b2d0`, `43ee8ed`, `01ee524`, `e4b3bca`, `fd8641d`, `f595f1b`, `a1bf658`, `8e28124`, `0b7add3`, `869a033`, `b03a320`, `5d711bf`, `a14524d`. I read `5d43aa8` and `1af364f` through their results at `0d866e8`.
- Fix reports B (without its W3-14 part) and D.
- The gate files: `w3-19_vs_w3-6`, `w3-29_vs_w3-19`, `w3-29_vs_reference`, `w3-24_vs_w3-29`, `geometry_w3-24_vs_w3-29`, `geometry_w3-24_crashes.txt`, `w3-24_vs_reference`, `w3-40_crashed_runs.txt`, `geometry_w3-29_summary`, `medians_default`, `medians_geometry`, `head_vs_w3-29` and `geometry_head_vs_w3-29`, with their `_fields.txt`. Also `fp_all.out`.
- The records my rows cite:
  - `matlab_side_defects.md`: W3-23, W3-24, W3-26, W3-39, W3-40.
  - `known_differences.md`: KD-B4-1, KD-B4-2, KD-B5-9, KD-B6-2.
  - The CHANGELOG lines of the rows, the survey rows for W3-22, W3-23, W3-32 and W3-38, the TODO item on the one-point GP, and W1-2's row in `wave1.md`.
- The code at `0d866e8`: `_poll_step_`, `_get_target_from_gp_`, the re-estimate, the checks in `_init_optim_state_`, the search's call sites, the main loop's rebuild flag, `poll_mads_2n.py`, the prior block of `gaussian_process_train.py` and the `.ini` descriptions.
- MATLAB: `bads.m` 440-1135 and 1230-1345, `pollMADS2N.m`, `setupvars.m` 170-185, `setupoptions.m` 1-80, `gpdefBads.m` 200-300, `gppred.m` 1-80 and gpml's `priorGauss.m`.

**Skimmed:** `wave3_B4_verifier.md` (only its lines on W3-29), the plan's wave-3 worklog, and gpyreg's `predict` (the clamp at `gaussian_process.py:2810`).

**Not reached:** the reviewers' reports `reviews/B4_*.md`, the agents' and orchestrator's scripts (other than `fp_all.out`), and the suite and populations, which I did not run.

**Checks run.** All scripts and outputs are in `/tmp/claude-0/-home-user-pybads/cec8ef42-ff6e-5789-9637-a276f362470b/scratchpad/dc/b_B4/`:
- `checks_options.py`, `v110_runs.py`, `v110_inf.py`
- `w3_19_pless.py`, `w3_29_rebuilds.py`, `w3_29_what_changes.py`, `bench_run.py`
- `w3_22_27.py`, `w3_40_runs.py`, `ridge4.py`

Each prints `pybads.__file__` and `gpyreg.__file__`. The trees for other commits were extracted with `git archive` into `at_<rev>` and deleted afterwards.

## 2. What holds

- **W3-19 (`8e28124`).** `p_less` equals a transcription of `bads.m:862-872` exactly:
  - on 16,800 random sets (D = 1..7, n = 1..2D);
  - on 169 real poll steps (n = 1..8, D = 2..4), where `f_mu` and `fs` are (n, 1), so `gamma_z` is a column;
  - the old lines differ from MATLAB in 5,699 of the random sets.

  The test's value is right. The gate compares `d79ab75` with `8e28124`: 5 runs changed, no flag. The ruling asked for "a unit test"; the test is a patched run that checks the exact value, and report D says why.
- **W3-20 (`5f31837`).** The branch now returns arrays of shape (1, 1), `zeros(1)` and (1, 1), with MATLAB's values (`bads.m:1328-1332`).
- **W3-21 (keep).** KD-B4-2 says "Settled".
- **W3-22 (`f595f1b`).** No `np.seterr` is left in pybads or gpyreg.
  - Default runs at `0d866e8` (sphere D2, Rosenbrock D3, noisy sphere D2) leave `np.geterr()` unchanged and emit no RuntimeWarning. At `8aecb6a` the same runs leave `divide='ignore'`.
  - `np.errstate` covers only the `gamma_z` expression. Its added `invalid` concerns a 0/0 there, which the poll already treats as an unreliable GP.
- **W3-23 (`dac062e`).** The fallback now takes `fs2 = fsd**2`, and the MATLAB lines 1309-1311 and 1321 are as the row states.
  - MATLAB's `max(NaN,0)` is 0, so a NaN variance with a finite mean skips MATLAB's fallback, but its target is still NaN, as the entry says.
  - `np.max` versus `max(·,0)` makes no difference, because gpyreg clamps `s2` at 0.
- **W3-24, its gate and its revert.**
  - The gate files compare the commits and suites they name: `0b7add3` against `869a033`; all 540 default runs and 182 of 210 geometry runs changed.
  - The flags, the crashes and the four flags against the reference are as the ledger reads them. One nuance: "at lower errors" for `edgesphere_D2` rests on the medians, while the paired ratio is +0.12 [−0.79, +0.86].
  - After `b03a320`:
    - the code of `poll_mads_2n` and `_poll_step_`, `README.md`, `index.rst` and the tests equal `0b7add3`'s; only the docstring and the permutation comment differ;
    - the changelog entry is gone, KD-B4-3 is gone, and the `matlab_side_defects` entry is present;
    - both head gates are identical in every field but the wall time.
  - The docstring's statements hold, except F6:
    - the bound is `pollMADS2N.m:7`;
    - the integer ranges equal `randi(nmax*2-1)-nmax`;
    - a row-only permutation gives MATLAB's set of directions;
    - `n_max` is 1 at default, since the locked search mesh is 2^(2k−10) and k ≤ 0.
- **W3-25 (`fd8641d`).** `AGENTS.md` and `gp_rescale_poll`'s description match the code. `poll_scale` is read only by `es_search.py:283` and by the poll, which cancels it. `len_scale` and `effective_radius` are used where stated.
- **W3-26 (keep).** The entry is present, and MATLAB lines 841 and 908-916 are right.
- **W3-27 (`a1bf658`).** Both sites use `nanargmin` with a guard for an all-NaN set, and draw from `self.rng`.
  - With every acquisition value NaN, the same seed gives the same evaluations and a different seed different ones; NumPy's global state is untouched.
  - The test covers both sites.
- **W3-28.** `tol_poi`'s description matches `_is_poll_stop_` and `complete_poll`.
- **W3-29 (`0b7add3`).** I rebuilt MATLAB's `post` and `pollmoved_flag` logic (523, 707, 826, 956/958, 1049) as a state machine over the recorded events of 26 runs:
  - the runs: Rosenbrock D = 2-4, a noisy sphere, and forced empty poll sets at polls 3-9;
  - at `0d866e8`: 0 mismatches, including search moves after a poll move and polls that make no rebuild;
  - at `8e28124`, the positive control: the missing rebuilds show up (2, 6, 6 and 1 in the runs with a poll move).

  The comments, the test docstring, the changelog entry and W1-2's corrected row are true. The gate changed 277 runs, with no flag.
- **W3-30.** KD-B5-9 holds.
- **W3-31 (`ec1b2d0`).** The check refuses 0, 1, −0.2, 1.5, nan, inf and booleans, and accepts floats in (0, 1) and `None` (the default). MATLAB's check is `bads.m:1269-1271`, and NaN passes it, as the commit says. At 1.1.0, 0 and 1 end at the best initial point.
- **W3-33 (`43ee8ed`).** Both paths keep `optim_state` in step: without a move at `bads.py:1567-1570`, and with a move through `_update_incumbent_`. MATLAB (`bads.m:1096-1118`) updates only its local copies, as the commit says.
- **W3-34, W3-35, W3-36.** They match MATLAB: `pollMADS2N` returns `[]` once B is non-empty, and `bads.m:976-982` has no `u_base`.
- **W3-39 (`5d711bf`).** The check refuses 0, −1, 2.5, nan, inf, booleans, `'3'`, arrays and complex values. It accepts 3, 3.0, `np.int64`, `np.int32`, `np.uint8` and `np.float64(3.0)`, stored as int, and `None`. At the parent, 0 and −1 raise TypeError at the first failed poll, as the row and the test say.
- **W3-40 (`a14524d`).** The previous prior is kept whole (centre and sigma) whenever the distances have no spread: two distinct points, duplicates included.
  - I reproduced `w3-40_crashed_runs.txt` exactly: 96, 148 and 37 evaluations.
  - A band |x2| ≤ 1e-3 along x1 reaches the two-point rebuild with the coordinate poll too: 4 seeds crash at `5d711bf` and complete at `a14524d`. This supports the changelog's "thin feasible region".
  - In MATLAB, `covsigma` is 0 and gpml's `priorGauss` gives NaN for every value, which enters only the fit's objective.
  - The fix is recorded in KD-B6-2, in `matlab_side_defects.md` and in the changelog.
- **W3-32, W3-37, W3-38.** The survey rows are closed, and W3-37's index arithmetic checks out.
- **Report D.** Its statements hold, and its 4-D ridge stalls at x0 (f = 6, 84 evaluations) at both commits.
- **Report B.** It holds except for F4.

## 3. Findings

### F1. W3-29's row says the persistent rebuilds have no effect within 200 evaluations; its gate and a reproduction show otherwise
- Where:
  - `dev/experiments/port_review_20260925/verification/wave3.md:79` at 0d866e8: in "What", "no effect within 200 evaluations; longer runs can differ once the nearest-neighbour set changes"; in "Default run", "without effect in runs of 200 evaluations";
  - the same claim in `wave3_B4_verifier.md:31` and `:241`.
- Kind: false statement
- Severity: minor. The fix-pass row (`:426`) reports 277 changed runs, but nothing corrects the row.
- What is stated, what is true, and the evidence:
  - **The gate** (`w3-29_vs_w3-19.md`): `rastrigin_D3` goes from 0.07 to 0.03 solved, and all its runs take 65-184 evaluations. `ellipsoid_D3` (113-161 evaluations) moves its median error from 1.50e-6 to 2.34e-6.
  - **A direct reproduction** (`bench_run.py`, `rastrigin_D3`, seeds 0-9, capped at 200) at `8e28124` and `0b7add3`, which differ only by W3-29:
    - seed 3 goes from 174 to 165 evaluations and ends at another x;
    - seed 5 goes from 171 to 180;
    - seed 1 goes from 200 to 183.
  - **Why** (`w3_29_what_changes.py`): at D = 3 the local GP reaches 51 points within 200 evaluations, and a rebuild trims it to 50. Even on the same training set, the rebuilt posterior differs from the incrementally updated one by about 1e-13.
- Would the correction move results: no
- Proposed correction: a bracketed note, as on W1-2's row. "[fix pass: the gate changed 277 of 540 runs, `rastrigin_D3`'s at most 184 evaluations among them: the local GP reaches `n_train_max` within 200 evaluations, and a rebuild recomputes the posterior]". In "Default run": "yes, all levels".

### F2. W3-39's changelog entry misstates what 1.1.0 did, and `inf`, which MATLAB and 1.1.0 run with, is now refused without a record
- Where: `CHANGELOG.md:101-104`, and `pybads/bads/bads.py:830`
- Kind: false statement (and a stricter interface than MATLAB's, not recorded)
- Severity: minor
- What is stated, what is true, and the evidence. The entry says "1.1.0 stopped the run with `TypeError` at its first failed poll for 0 or a negative value … and with `IndexError` for a float." My runs at 1.1.0 (`v110_runs.py`, `v110_inf.py`):
  - 0 raises `IndexError: index 1 is out of bounds` at the second poll. 1.1.0 tests `iter > steps`, so its first failed poll, at iteration 0, reads nothing.
  - −1 raises TypeError at the first poll, as stated.
  - 2.5 and 3.0 raise IndexError at the 4th or 5th poll, as stated.
  - `inf` and `1e9` complete.

  The "TypeError at the first failed poll" for 0 is the behaviour at `8aecb6a` and `b03a320`, which the ledger's row states correctly. MATLAB runs with `Inf`: `iter > Inf` (`bads.m:976`) is never true, and `setupoptions.m` has no check. PyBADS now raises ValueError for `inf`.
- Would the correction move results: no
- Proposed correction: "… 1.1.0 stopped the run, with `IndexError` or `TypeError`, for 0, a negative value or a float once the run reached it, as MATLAB BADS stops for a value below 1 or not an integer; `inf`, with which 1.1.0 and MATLAB BADS never accelerate, now raises `ValueError` too." Then either accept `inf`, as the check of `max_fun_evals` does, or record the refusal, for instance in W3-39's `matlab_side_defects` entry.

### F3. An `improvement_quantile` given as a string or a complex number raises TypeError, not the promised ValueError
- Where: `pybads/bads/bads.py:814-819`, and `CHANGELOG.md:55-56` and `:98-100`
- Kind: false statement
- Severity: minor
- What is stated, what is true, and the evidence (`checks_options.py`):
  - `"0.3"` and `"3"` raise `TypeError: '<' not supported between instances of 'int' and 'str'`;
  - `0.3+0j` raises TypeError too;
  - `array([0.3, 0.4])` raises NumPy's error about an ambiguous truth value.

  The two sibling checks refuse `'3'` and `"200*D"` with their own ValueError, by an `isinstance` test (`bads.py:795-809`, `825-837`).
- Would the correction move results: no
- Proposed correction: in the check, refuse a value that is not a real number with the same ValueError, using the `isinstance` test of the check of `max_fun_evals`.

### F4. `_poll_step_` does return what its docstring lists; the ledger and report B say otherwise, and B misreads the re-estimate
- Where:
  - `verification/wave3.md:566-567`: "lists return values it does not return";
  - `fixes/B_W3-20_…_W3-14.md:101`: "the method returns nothing";
  - `fixes/B_W3-20_…_W3-14.md:99`: "if the current iterate's re-estimate fails, `self.fval` and `self.fsd` are NaN".
- Kind: false statement
- Severity: minor
- What is stated, what is true, and the evidence:
  - `bads.py:2553` returns the docstring's five values, at `8aecb6a` and `326aefe` too. The main loop discards them (`:1459`).
  - `_re_evaluate_history_` keeps the current iterate's recorded estimate when its rebuild fails (`bads.py:2890-2892`, `if i == n_iter - 1: continue`, since `fef6c14`).
- Would the correction move results: no
- Proposed correction: "`_poll_step_`'s docstring calls an SD a variance, and the main loop discards the values it returns"; add a bracketed note in report B.

### F5. The changelog says the prior of the length scales keeps "its previous width, as the prior of the GP mean does"; the whole prior is kept
- Where: `CHANGELOG.md:602-606`
- Kind: false statement (a misleading analogy)
- Severity: minor
- What is stated, what is true, and the evidence:
  - `gaussian_process_train.py:372-383` skip the assignment, so the centre and the sigma both stay; the test asserts both.
  - The mean's prior with a zero range is re-centred at `y_mean` with its previous width (`:345-356`).
- Would the correction move results: no
- Proposed correction: "… the prior of the length scales stays as it was, its centre and its width."

### F6. Docstrings and the sheet misstate the poll's basis and its inputs
- Where: `pybads/poll/poll_mads_2n.py:37-38`, `pybads/bads/bads.py:2156`, and `known_differences.md:223`
- Kind: false statement
- Severity: minor
- What is stated, what is true, and the evidence:
  - The docstring gives `poll_scale` the shape `(1, D)`. Runs pass `(D,)`: `ll.flatten()` at `gaussian_process_train.py:573` and `np.ones(D)` at `:1080`. Only the unit tests pass `(1, D)`.
  - `_poll_step_`'s docstring says the poll uses "the LTMADS poll direction method", and KD-B4-1's title says "The poll always uses LTMADS". The default poll is MATLAB's coordinate poll.
- Would the correction move results: no
- Proposed correction:
  - "of shape `(D,)`";
  - `_poll_step_`: "… along the directions of `poll_mads_2n`, the signed coordinate directions at default";
  - KD-B4-1: "The poll always uses MADS 2N (`poll_mads_2n`) …".

### F7. `_get_target_from_gp_`'s Returns section describes only the branch that predicts
- Where: `pybads/bads/bads.py:2704-2712`
- Kind: false statement
- Severity: minor
- What is stated, what is true, and the evidence: in W3-20's branch (`:2759-2765`), `f_target_mu` is the incumbent's `fval`, not "the GP's mean prediction", and `f_target_s` is `np.zeros(1)`, not the GP's SD.
- Would the correction move results: no
- Proposed correction: add to each entry "the incumbent's `fval`" and "0 (`np.zeros(1)`)" for the case where no prediction is made.

## 4. Outside my scope

- `wave3.md:504` gives `sphere_band_D3` as "58 → 48" evaluations, where `medians_geometry.md` gives 47 at W3-24. This is a number for reviewer (d).
- The gates against the reference name `population_linux_wave2_20260926`'s package as `8510ca8`, whose `pybads/` differs from `8aecb6a`'s in 8 files. That is provenance, for reviewer (d).
