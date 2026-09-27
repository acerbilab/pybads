<!-- Report of reviewer (d) of the doublecheck of wave 3, the records, gates and tooling (briefs/wave3_doublecheck.md), reading PyBADS at 0d866e8 in /home/user/pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave3/doublecheck/d_records/. -->


# Wave 3 doublecheck: the records, gates and tooling

## 1. Coverage

**Read completely, at 0d866e8 unless noted:**
- `verification/wave3.md`.
- Every file of `verification/wave3_fixpass/` (the comparisons, `_fields.txt`, the summaries, the medians, `w3-1_repeats.txt`, `w3-14_attribution.txt`, `geometry_w3-24_crashes.txt`, `w3-40_crashed_runs.txt`).
- All 14 files of `verification/scripts/wave3/orchestrator/`.
- `known_differences.md` and `matlab_side_defects.md`.
- The port review's `README.md`.
- The survey diff `8aecb6a..0d866e8`, and every row of its candidate table.
- `dev/TODO.md`: the diff and the whole open list.
- `AGENTS.md`: the diff and the bullets that wave 3 touches.
- `dev/README.md`.
- The geometry commit `8824c9e` and the module docstring of `benchmark_targets.py`.
- `population_linux_wave3_20260927`: its README, and the `meta` of all 540 records. The `meta` of `population_linux_wave2_20260926` too.
- The plan's "Wave 3 pickup" and its worklog.

**"Wave 4 pickup" is not at 0d866e8.** It and the worklog line for wave 3's merge are in `ed82ec0` (#78, on `dev-next` after `0d866e8`). I read them with `git show ed82ec0`.

**MATLAB lines read at 74919c0:**
- `uCheck.m`, `searchES.m:39-72` and `130-211`, `ucov.m`, `pollMADS2N.m`.
- `acqPortfolio.m:30-70` and its history, `acqLCB.m:1-30`, `gpdefBads.m:232-256`.
- `bads.m`: 154-288 (the defaults), 510-760, 800-990, 1044-1049, 1220-1335.
- `setupvars.m:170-185`, and the relevant parts of `gppred.m` and `mygp.m`.

**Checks run** (all from my scratch directory, one BLAS thread, each printing `pybads.__file__` and `gpyreg.__file__`):
- `population.py compare` of wave 2 against wave 3, `compare --split` and `summary`, from the review worktree's `dev/scripts/`. All three are byte-identical to the committed `comparison.md`, `null_check.md` and `summary.md`. The comparison also equals `w3-29_vs_reference.md` apart from the header.
- `medians.py` on the two committed references. It reproduces the reference and W3-29 columns of `medians_default.md` (18 of 18 rows).
- A `meta` scan of both references.
- The `x0` and final state of each run, wave 2 against wave 3.
- `benchmark_targets.py --check --suite geometry`: all ok, in 1.1 s.
- The `default` suite of `benchmark_targets.py` before (`6010815`) and after, for 18 configurations × 30 seeds: `x0`, options, bounds, `f_min`, `x_min`, `f_vec`, the noise draws and `non_box_cons`. Nothing differs.
- `w3_acc0.py` at `4f50376` (archived): `TypeError` at `bads.py:2478` for seeds 0 and 1. At 0d866e8 it gives `ValueError` at construction.
- A Python transcription of `acqPortfolio.m` 'upd' on an empty set (stale `usearch`, `fsearch = fval`, SD 0), against `update_hedge(None)`: the largest difference over 200 random states is 0.0.
- The GitHub Actions runs of `dev-port-review-w3`: the `tests` push runs at `599115b`, `c788617`, `a1bf658`, `869a033`, `dd78136` and `a14524d` all succeeded, as did `merge-tests` on PR #77's head `1af364f`.

**Skimmed:** the four reviewers' reports (finding counts and the items the Notes attribute to them), the two verifiers' reports and the B3/B4 verifier outputs (only the numbers the sheet repeats), and the kept-item briefs.

**Not reached:**
- The fix agents' reports in full.
- The 457 tests at the head, and the "12 more warnings" of W3-22: these need the suite.
- The worst errors that W3-24's paragraph quotes (3.9, 74.7) and the full sets of changed runs per step: these come from populations that are not committed.
- The CHANGELOG, which is scope (c).

## 2. What holds

- **Fingerprint table against `fp_all.out` and the branch.** All 30 commits and hashes match, in git order. The shared hashes are right: `dc111…` from `4388e6d` to `a1bf658`, `3a6c…` at W3-5, W3-15 and W3-1, `3609…` at W3-6, W3-19 and W3-29 and from the revert on, `f5af…` at W3-24 and the merge. The branch's tree equals 0d866e8's. No commit after `a14524d` touches `pybads/` or `dev/scripts/`, so 0d866e8's fingerprint is `360971bf1f0ba6cb`, as the Wave 4 pickup says.
- **Batch 1.** No flag in 54 tests. The attribution lists 31 runs, whose per-configuration counts (7, 7, 6, 6, 3, 1, 1) and "all 6-D, 10-D or noisy" hold, and "31/31, 31/31, not explained: []" holds.
- **The ES batch.** `final.x` changed in 540 of 540 runs; no flag.
- **W3-1.** 325 runs changed on the default suite (by `fval`) and 124 of 210 on the geometry suite, with no flag in 54 and in 21 tests. The repeats of `w3-1_repeats.txt` match the table and `TODO.md`: 137 of 1411, 94 of 6840, 13, 3 and 6, and 100 of 8800, none after. The median error 0.43 → 0.36 matches `medians_default.md`.
- **W3-6, W3-19, W3-29.** 128, 5 and 277 runs changed. The solved fractions 0.63 → 0.50, and "one run elsewhere", hold.
- **W3-24's paragraph.** Every flag, KS statistic, Holm p, ratio, CI, solved fraction and median holds (1.1e-7 → 4.3e-7; 97 → 88; 47 → 45 and 96 → 79; 23 of 30 solved; 1 and 2 crashes; ridges 0.83 → 0.73 and 0.87 → 0.83). So do the net change of four flags against the wave 2 reference at W3-24 and none at W3-29. The crash files match the three seeds (sphere_band_D2 13 and 23, sphere_band_D3 15), and so do the errors after W3-40 (1e-7, 3e-10, 1.2e-5). The two exceptions are F6 and F14.
- **The head.** The only field that differs from W3-29 is `wall_s`, on both suites.
- **The rows and the reports.** Rows W3-1 to W3-39 exist. The findings per report are 9, 10, 10 and 7, and the kept items 13 and 11. Every report finding and every kept item maps to a row.
- **The survey.** There are exactly 11 B3/B4 rows. Each is closed with its row and a correct verdict and commit. No B3/B4 row is left open (the two open rows left are B7's). The `test_incumbent_constraint_check` statement holds.
- **The sheet.**
  - The citations are at `8aecb6a` (checked: `bads.py:1956-2004`, `1801-1805`, `2195-2200`, `2137-2143`, and `es_search.py:225-303`). That matches the README and the path conventions.
  - KD-B3-1 (without `ESSearchCMA`) and KD-B3-3 (the formula's history) hold.
  - KD-B3-5 holds against `bads.m:667-725`, where the error on a first empty search is at line 722 and comes at any quantile, and its gains hold (the transcription above). The label is F9's.
  - KD-B3-6 holds against `searchES.m:170-193` and `uCheck.m:8` (`max(min(NaN,UB),LB)` gives `UBsearch`), and against the code's `ntest > 0` guard.
  - KD-B4-2 holds against `bads.m:1296-1312`, except for F5's number.
  - KD-B4-3 is gone, with no dangling reference. KD-B5-9 holds (`bads.m:822-823`, `1242-1252`), and so does the W3-40 part of KD-B6-2 (`gpdefBads.m:240-251`).
- **`matlab_side_defects.md`.** Every wave 3 entry holds against its MATLAB lines:
  - W3-7: `acqPortfolio.m:40`, `:47`; `gpstructnew` is read only there, and was commented out in `12f7ff8` and `91f39f5`.
  - W3-23: `1310-1311`, `1321`.
  - W3-39: `976-979`, `1091`, `setupvars.m:179-182`.
  - W3-40, W3-3 (`ucov.m:19`, `searchES.m:55-58`), W3-26 (`841`, `908-916`), W3-11.
  - W3-24: `pollMADS2N.m:7`, `10`, `14`, with `SearchGridMultiplier` 2, `SearchGridNumber` 10 and `MaxPollGridNumber` 0. Its gate evidence holds.
  - The 0/0 entry holds, but for the option name (F8).
- **`dev/TODO.md`.** The two lines of "Out of this pass" are present. "Previously evaluated points evaluated again" and "Follow-ups of the GP-update guards" are removed; the fourth follow-up is KD-B5-2's "by design".
- **`AGENTS.md`.** Every statement that wave 3 touches holds at 0d866e8: the poll's basis and bound, `poll_scale`, `len_scale`, `effective_radius`, the extension points without `ESSearchCMA`, the draw sites (all through `self.rng`, W3-27's fallback included), and the markers.
- **The geometry suite.** It matches its description: the edgesphere minimum is `ceil(D/2)` on the lower bound; along the ridge valley any coordinate step raises f by at least 9|h|; the band is `|x1 - x2| <= 0.005` with `c1 = c2`. `--check` passes. The default suite is unchanged, and the merge changed only docstrings and `--check` logic.
- **The orchestrator's scripts.** Each does what the records say:
  - `fp.sh` checks out a worktree, sets one BLAS thread and puts the worktree and the clone on `PYTHONPATH`.
  - `gate.sh` and `gates_chain.sh` follow the ledger's chain and naming.
  - `pick.sh` and `picks_run.sh` resolve only test-file conflicts, then run the suite with `-x` and `fp.sh`.
  - `attrib.sh` and `one_run.sh` compare `final` without `wall_s`.
  - `count_repeats.py` counts exact duplicate rows, or level-2 merges with the final samples subtracted; its method is right for the logger (`record_duplicate_data=False` adds no row).
  - `resolve_appends.py`, `medians.py` (crashed runs included) and `w3_acc0.py` (reproduced) hold too.
- **The new reference.** Its `meta` holds: `a14524d` clean, `pybads_source` the worktree `h_a14524d`, gpyreg the clone at `98ab5a4` (string `1.3.4.dev10+gd96d0d9f7`), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, one thread, 09:29 to 09:48. The README's numbers recompute (270 → 220, 372 → 332, 0.32 → 0.43, +0.13 [-0.04, +0.28], 0.13 → 0.10, -0.29 [-0.48, +0.07], no crash, 0.03 to 1.00), and the `x0` of all 540 runs equal the wave 2 reference's.
- **The plan's worklog.** The kickoff, run, triage and fix-pass lines hold (the counts 9, 10, 10, 7; the prep report's (a) and (e); "every other proposal accepted"), except F13. At `ed82ec0`, the merge line's CI claim holds.

## 3. Findings

### F1. `dev/TODO.md` lists as open two items that the pass fixed
- Where: `dev/TODO.md:198` and `:216-217` at 0d866e8 ("Minor items of slices B1 and B2 … Not fixed").
- Kind: false statement.
- Severity: substantial.
- **What is stated:** "`hedge_gamma`'s description is its section's header" and "after a re-estimate that moves nothing, `optim_state`'s `yval`, `fval` and `fsd` keep older values", both as not fixed.
- **What is true:** both were fixed in the pass.
  - W3-7's commit `4d357e4` gave `hedge_gamma` its own line (`advanced_bads_options.ini:265`, "Minimum probability of each search in the hedge's choice of a search").
  - W3-33's commit `43ee8ed` sets `optim_state["yval"]`, `["fval"]` and `["fsd"]` in the re-estimate (`bads.py:1518-1521`).
  - The lines came with `68d4516` (wave 2's doublecheck), merged at `4f50376` after both fixes, and were not reconciled.
- Would the correction move results: no.
- Proposed correction: delete both clauses from the item.

### F2. The sheet has no entry for wave 3's own deliberate differences, which wave 4's slice O reads
- Where: `known_differences.md` at 0d866e8: B3 (after KD-B3-6) and B4.
- Kind: other.
- Severity: substantial.
- **What is ruled and done:**
  - W3-10 refuses a schedule's name and any non-positive or non-finite `sqrt_beta`. MATLAB's `acqLCB.m:16-21` accepts a name, a handle or any numeric scalar.
  - W3-27 picks at random when every acquisition value is NaN, where MATLAB's `min` takes the first index (Choices).
  - W3-7 (scoring at the search point), W3-23 (the target from the incumbent's SD) and W3-39 (a check at creation) fix shared defects. Wave 2's equivalents each got a sheet entry marked "A shared defect that PyBADS fixes" (KD-B1-10, B1-11, B2-6, B2-7).
- **Why it matters:** the sheet's own rule is that a difference not on it is reported as new. W3-30 was exactly this omission for W1-8. Slice O reads `acq_fcn_lcb` and the hedge's update in wave 4.
- Would the correction move results: no.
- Proposed correction: add entries for W3-10 and W3-27 at least (and W3-7, W3-23, W3-39 in the style of KD-B1-10), each citing its ruling and commit.

### F3. "Wave 4 pickup" under-lists what wave 3 changed in slice O's code
- Where: `dev/plans/port-correctness-review.md`, "Wave 4 pickup", steps 1 and 3, at `ed82ec0` (#78). It is not present at 0d866e8.
- Kind: false statement.
- Severity: substantial.
- **What is stated:** wave 3 "changed code that slice O reads (the hedge's reward, W3-6; `poll_mads_2n`'s docstring after W3-24's revert)", and O's brief names "wave 3's rulings (W3-24 reverted, W3-6, W3-25)".
- **What is true:** O's code (the slice table) also includes:
  - `acq_fcn_lcb`, changed by W3-10 (`599115b`);
  - the hedge's `update_hedge`, changed by W3-7 (`4d357e4`) and W3-11 (`4388e6d`);
  - `_eval_improvement_`'s quantile, refused outside (0, 1) by W3-31 (`ec1b2d0`);
  - the historic improvement of the accelerated mesh reduction, changed by W3-36 (`e4b3bca`) and W3-39 (`5d711bf`);
  - the ES code in which `ESSearchELL` uses `poll_scale`, changed by W3-5, W3-8, W3-9 and W3-15.
- Would the correction move results: no.
- Proposed correction: in step 1, "(the hedge's reward and update, W3-6, W3-7, W3-11; `acq_fcn_lcb`'s `sqrt_beta`, W3-10; the quantile check, W3-31; the accelerated mesh reduction, W3-36, W3-39; `poll_mads_2n`'s docstring after W3-24's revert)". In step 3, name W3-7, W3-10, W3-11 and W3-31 among the rulings.

### F4. Wave 3's "Found while fixing" items in B3 and B4 code have no open-work record
- Where: `verification/wave3.md:543-546` ("each is left to the PI or to the wave of the slice that owns its code") and `dev/TODO.md` at 0d866e8.
- Kind: other.
- Severity: minor.
- **What is true:** the wave of B3 and B4 is wave 3 itself, and none of A-D's items reached `TODO.md`. Among them:
  - `hedge_gamma` unchecked (the hedge's probabilities turn negative above 1/(n−1));
  - `n_search_iter` below 1 unchecked;
  - `sqrt_beta` checked only at the first search;
  - `output_fcn`'s `"done"` call receiving stale `optim_state` (B2);
  - docstrings and unused imports.
- Wave 2's leftovers went to `TODO.md` ("Minor items of slices B1 and B2"). O's items are carried by Wave 4 pickup step 4; the B3/B4-only ones are not.
- Would the correction move results: no.
- Proposed correction: a `TODO.md` item "Minor items of slices B3 and B4 of the port review", listing the "Found while fixing" items that are not O's.

### F5. "0 of 21 decisions at level 0" counts level-1 decisions
- Where: `verification/wave3.md:71` (W3-21) and `known_differences.md:234` (KD-B4-2).
- Kind: number does not recompute.
- Severity: minor.
- **What is stated:** "Hyperparameters differed in 0 of 21 decisions at level 0 and 5 of 13 at level 1."
- **What is true:** `B4_verifier/v06_instrument.out` has 21 relevant decisions, 10 in its three level-0 runs and 11 in its three level-1 runs, and `B_hypdiff` 0 in all six. The 5 of 13 are `v12`'s level-1 runs, in a wide box.
- Would the correction move results: no.
- Proposed correction: "in 0 of 21 decisions at levels 0 and 1 (10 and 11), and in 5 of 13 at level 1 in a wide box".

### F6. Batch 1: 29 of the 31 changed runs end at other points, not 31
- Where: `verification/wave3.md:419` (batch 1 row). The same sentence is at `CHANGELOG.md:554`.
- Kind: number does not recompute.
- Severity: minor.
- **What is stated:** "31 of the 540 runs end at other points".
- **What is true:** `batch1_vs_reference_fields.txt` gives `final.x` changed in 29 runs and `final.fval` in 30. `w3-14_attribution.txt` lists 31 changed runs. So at least two of the 31 end at the same point, differing only in `fsd` (e.g. `ellipsoid_D3_homo_seed4`, by one ulp) or `min_noise_var` (`multisensory_s1_D6_homo_seed0`).
- Would the correction move results: no.
- Proposed correction: "31 of the 540 runs change, 29 of them ending at other points".

### F7. W3-40 has no row in the ledger
- Where: `verification/wave3.md`: the B3 and B4 tables end at W3-39. W3-40 appears only at `:358-361`, `:431` and `:514`.
- Kind: other.
- Severity: minor.
- **What is true:** W3-40 is a fix of the pass, with its own commit (`a14524d`), a test, a changelog line and a sheet edit. W3-39, also added after the triage, got a row. The brief's "rows W3-1 to W3-40" does not hold. Nothing records W3-40's classification, default reach (none in the benchmark at MATLAB's poll) or dating (`gpdefBads.m:240-251`, 2017).
- Would the correction move results: no.
- Proposed correction: add a W3-40 row after W3-39 in the same form: source "W3-24's gate", the GP layer (B6), and disposition "PI, after the gates: fixed in `a14524d`".

### F8. `matlab_side_defects.md` names a MATLAB option that does not exist
- Where: `matlab_side_defects.md:187`.
- Kind: false statement.
- Severity: minor.
- **What is stated:** "from `SearchNiter` 3 (the default is 2)".
- **What is true:** the option is `Nsearchiter` (`bads.m:231`; `searchES.m:125`, `190`; KD-B1-2's `Nsearchiter`→`n_search_iter`).
- Would the correction move results: no.
- Proposed correction: "from `Nsearchiter` 3".

### F9. KD-B3-5 mislabels `bads.m:1257-1279`
- Where: `known_differences.md:207`.
- Kind: false statement.
- Severity: minor.
- **What is stated:** "`1257-1279` (`SearchStep`'s improvement tests)".
- **What is true:** `bads.m` has no `SearchStep`. Lines 1257-1282 are the subfunction `EvalImprovement`. The search's improvement test is at 676-681, inside the cited 667-725.
- Would the correction move results: no.
- Proposed correction: "`1257-1282` (`EvalImprovement`)".

### F10. The records do not say that the orchestrator's scripts depend on the sandbox
- Where: the port review's `README.md:80-84` and `verification/wave3.md:392-394`.
- Kind: other.
- Severity: minor.
- **The paths the scripts hard-code:**
  - `/tmp/claude-0/-home-user-pybads/67ddf7b2-…/scratchpad` and its `orch/`, `gates/`, `bisect/` (`gate.sh`, `gates_chain.sh`, `pick.sh`, `picks_run.sh`, `attrib.sh`, `one_run.sh`);
  - `/home/user/pybads`, `/home/user/pybads-fp` and `/home/user/gpyreg-v1.3.3` (`fp.sh`, `cl.py`'s `/home/user/pybads/CHANGELOG.md`).
- **The inputs that are not committed:** `$S/orch/picks_CD.log`, `$S/gates/g0.log`, `$S/bisect/moved.txt`.
- **The consequence:** the scripts call each other at `$S/orch/`, so they cannot run from the committed directory. No record says any of this. The rerun behind `w3-40_crashed_runs.txt` has no script at all.
- **Two self-descriptions are also wrong:**
  - `same_fields.py` says "timings left out", but it compares `final.wall_s`, which every `_fields.txt` lists.
  - `cl.py`'s docstring and `pick.sh`'s usage omit the `replace` mode that W3-29's changelog entry used.
- Would the correction move results: no.
- Proposed correction: one sentence in the README's wave 3 paragraph: "The orchestrator's shell scripts and `cl.py` name the sandbox's paths (the scratch directory, `/home/user/pybads`, `/home/user/pybads-fp`, `/home/user/gpyreg-v1.3.3`), call each other from the scratch directory's `orch/`, and read logs that are not kept (`picks_CD.log`, `g0.log`, `moved.txt`); the rerun of `w3-40_crashed_runs.txt` has no script." Also correct the docstrings of `same_fields.py` and `cl.py`.

### F11. `fp_all.out` does not hold "every commit of the pass/branch"
- Where: `verification/wave3.md:387` and the README's `:82`.
- Kind: false statement.
- Severity: minor.
- **What is stated:** the fingerprint was "computed again by the orchestrator at every commit of the pass" (the ledger), and "at every commit of the branch" (the README).
- **What is true:**
  - `fp_all.out` holds the 30 commits that change the package, plus `388d879`.
  - Absent: the docs commits `a4da1fe`, `aecd1ba`, `97bfc99`, `dd78136`, `bd110f0`, `6281663`, `391373e`, `5d43aa8` and `1af364f`, the pre-pass commits `eee0e4e`…`326aefe`, and `68d4516`. None of them changes the package, so nothing is lost.
  - `1f7c8ee` is listed after `869a033`, out of order. At its pick, the suite and fingerprint ran at `388d879`, which has the same package code.
- Would the correction move results: no.
- Proposed correction: "at every commit of the pass that changes the package (`fp_all.out`; `1f7c8ee`'s code first at `388d879`, which changes only records)".

### F12. KD-B5-2 and W3-12 present one offset vector as three distances
- Where: `known_differences.md:252` and `verification/wave3.md:57`.
- Kind: false statement.
- Severity: minor.
- **What is stated:** KD-B5-2 has "142, 38 and 93 grid units from the incumbent in PyBADS, 1891, 1696 and 291 in MATLAB's rule". The row has "142, 38, 93 grid units away" against "-1891, 1696, 291 units away".
- **What is true:** `B3_verifier/v_failed_rebuild_out.txt` has one injected failure, with offsets `[-142, 38, 93]` (port) and `[-1891, 1696, 291]` (MATLAB), per coordinate, in 3-D. The row drops the port's sign and keeps MATLAB's.
- Would the correction move results: no.
- Proposed correction: "with an injected failure (3-D), at an offset of (-142, 38, 93) grid units from the incumbent in PyBADS and (-1891, 1696, 291) under MATLAB's rule".

### F13. The plan's worklog counts three survey rows fixed by wave 0; there are four
- Where: `dev/plans/port-correctness-review.md:724-725`.
- Kind: number does not recompute.
- Severity: minor.
- **What is stated:** "closes the 11 open survey rows of B3 and B4 (three of them fixed by wave 0 without the survey saying so)".
- **What is true:** four rows were fixed in `0c56d86` with status "seen": `bads.py:2183` (W3-32), `_poll_step_` with `stobads` (W3-38), and the two empty-set rows at `8afbe16` (W3-11). They are three ledger rows.
- Would the correction move results: no.
- Proposed correction: "four of them (three ledger rows) fixed by wave 0 without the survey saying so".

### F14. W3-24's paragraph mixes two medians for sphere_band_D3
- Where: `verification/wave3.md:504`.
- Kind: number does not recompute.
- Severity: minor.
- **What is stated:** "in fewer evaluations (58 → 48)", citing `medians_geometry.md`.
- **What is true:** `medians_geometry.md` gives 58 → 47. The 47 counts the crashed run, seed 15 at 3 evaluations. The 48 is the median of the 29 runs that did not crash; the 58 is the rounded 57.5 of all 30 at W3-29 (`geometry_w3-29_summary.md`).
- Would the correction move results: no.
- Proposed correction: "(57.5 → 48 over the runs that did not crash)".

### F15. The batch 1 row's double-precision arithmetic is inconsistent
- Where: `verification/wave3.md:419`.
- Kind: number does not recompute.
- Severity: minor.
- **What is stated:** "at `2^-42`, `x / tol` is about `1e12`, where the fraction of a double comes in steps of `2^-11`".
- **What is true:** `np.spacing(1e12)` is 2^-13. The spacing is 2^-11 only for `x / tol` in [2^41, 2^42), that is |u| ≥ 0.5 at a mesh of 2^-42.
- Would the correction move results: no.
- Proposed correction: "at `2^-42`, `x / tol` is up to about `4e12`, where the fraction of a double comes in steps of `2^-13` to `2^-10`".

### F16. KD-B4-1's title still says the poll uses LTMADS
- Where: `known_differences.md:223`.
- Kind: false statement.
- Severity: minor.
- **What is stated:** "The poll always uses LTMADS (`poll_mads_2n`)". The body, edited by wave 3, and W3-24 say that the basis is the signed coordinate directions at every default state, "not LTMADS's dense directions".
- Would the correction move results: no.
- Proposed correction: "The poll always uses MADS 2N (`poll_mads_2n`, MATLAB's `pollMADS2N`); the other poll methods are not ported".

### F17. The new reference's fingerprint does not name its thread setting
- Where: `dev/experiments/population_linux_wave3_20260927/README.md:20`.
- Kind: other.
- Severity: minor.
- **What is stated:** "the fingerprint of `dev/scripts/fingerprint.py` is `360971bf1f0ba6cb` at all of them", with no setting.
- **Why it matters:** `AGENTS.md` ("Numerical gates") says that one thread and the default can give different hashes, and that "a recorded hash names its setting". The ledger and the Wave 4 pickup name it as one BLAS thread.
- Would the correction move results: no.
- Proposed correction: "… is `360971bf1f0ba6cb` (Linux, gpyreg 1.3.3 from the clone, one BLAS thread) at all of them".

### F18. "Notes on the reports" leaves out one kept item
- Where: `verification/wave3.md:104-121`.
- Kind: other.
- Severity: minor.
- **What is true:** the note sorts every kept item except B3-K13 (the `int` `f_sd_search`, W3-18) into found, immaterial, not found or no longer holding. No report mentions `f_sd_search`.
- Would the correction move results: no.
- Proposed correction: "Not found: the search after a failed rebuild (B3-K2), `ESSearchCMA` (B3-K5), the `int` SD of the empty branch (B3-K13), `period_check`'s return (B4-K8) and `u_base` (B4-K9)."

### F19. The port review's README omits wave 3's fix brief, and the ledger has a stray bullet
- Where: the port review's `README.md:32-34`, and `verification/wave3.md:525`.
- Kind: other.
- Severity: minor.
- **What is true:**
  - The README lists only `briefs/wave1_fix_common.md` and `briefs/wave2_fix_common.md`. `briefs/wave3_fix_common.md` exists, and the ledger cites it.
  - The ledger's line 525 reads "- - **Changelog.**" and renders as a nested empty bullet.
- Would the correction move results: no.
- Proposed correction: "`briefs/wave1_fix_common.md` to `briefs/wave3_fix_common.md`: the briefs of the fix agents of waves 1 to 3", and "- **Changelog.**".

### F20. `dev/README.md` does not describe the geometry suite
- Where: `dev/README.md:45-59` at 0d866e8.
- Kind: false statement.
- Severity: minor.
- **What is stated:** `benchmark_targets.py` defines "the suites `smoke`, `default`, `oned` … and `bounds`". The list of problems names neither `edgesphere`, `ridge` nor `sphere_band`.
- **What is true:** `8824c9e` added `geometry` (`SUITES["geometry"]`, 7 configurations) and the three targets. The ledger's gates of W3-1 and W3-24 used them.
- Would the correction move results: no.
- Proposed correction: add "a sphere whose minimum lies on a hard bound, `edgesphere`, a nonsmooth valley along the diagonal, `ridge`, and a sphere in a thin feasible band, `sphere_band`" to the problems, and "`geometry` (those three: the gates of W3-1 and W3-24 of the port review)" to the suites.

## 4. Outside my scope

- `CHANGELOG.md:554` repeats "31 of 540 runs end at other points" (F6's number); this is scope (c).
- W3-33 has no changelog line, although it changes the `optim_state` that `output_fcn` receives, which a user can notice (scopes (b) and (c)).
- `dev/experiments/population_ellipsoid_hetero_linux_20260925/README.md:106` points to the removed `TODO.md` item "Previously evaluated points evaluated again" (a dated record).
- The plan's "Close" item says the Windows reference "predates waves 0 to 2"; it predates wave 3 too.
