<!-- Report of reviewer (d) of the doublecheck of wave 4, the records, gates and tooling (briefs/wave4_doublecheck.md), reading PyBADS at 81385ac in /home/user/pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave4/doublecheck/d_records/. -->


# Wave 4 doublecheck: the records, gates and tooling

## 1. Coverage

**Read completely:**
- The brief: the common part and scope (d).
- `verification/wave4.md`.
- All 18 files of `verification/wave4_fixpass/`.
- Every file of `verification/scripts/wave4/orchestrator/`: `fp.sh`, `gate.sh`, `gates_w421.sh`, `pick.sh`, `picks_run.sh`, `chain_rest.sh`, `chain_rest.log`, the `picks_*.log`, the `*.list`, `fp_all.out`, `changed_runs.py`, `same_fields.py`, `resolve_appends.py`, `cl.py`, `w430_edit.py`, `w430_init_stop.py` with its `.out`, and `w41_fixed_designs.py` with its `.out`. Of `changelog_entries/`, the file names and `w45_repeats.txt`.
- `population_linux_wave4_20260927/README.md`, `comparison.md`, `null_check.md` and `summary.md`, plus the `meta` of all 540 records of both Linux references.
- The diffs `6ceed6f..81385ac` of:
  - `AGENTS.md` and `dev/README.md`;
  - `dev/TODO.md` and `dev/results/2026-09-23-codebase-survey.md`;
  - the note in `verification/wave2.md`;
  - `matlab_side_defects.md` and the sheet (`known_differences.md`);
  - the plan and the review's `README.md`.
- The sheet's header, path conventions and the entries KD-B1-1, KD-B1-6, KD-B2-6 to KD-B2-9, KD-B3-7, KD-B3-8, KD-B4-5, KD-B4-6, KD-B5-10 and KD-B7-1 to KD-B7-5, plus the claims' preamble and C2.
- `briefs/wave4_kept_B7.md` and `briefs/wave4_kept_O.md`.
- The "outside my rows" and "Uncertain" parts of the fix reports A to E.
- The outputs of fix E, and fix C's W4-15 logs.

**Skimmed:**
- The headings of the three review reports, to count their findings.
- The B7 verifier's outputs `v1`, `v1b`, `v2_v110`, `v4`, `v5`, `v6` and `v7`, which settle the numbers of rows W4-1 to W4-14 that I spot-checked.
- The other parts of the fix reports.

**Not reached:**
- The verifiers' reports in full.
- The numbers of rows W4-15 to W4-20 beyond the pass table.
- The structure of `CHANGELOG.md`, which is reviewer (c)'s.

**Checks run** (all from `/home/user/dc4/d_records`, one BLAS thread):
- `population.py compare` of the two references, `compare --split` and `summary`, on scratch copies of the references. All three outputs are byte-identical to the committed `comparison.md`, `null_check.md` and `summary.md`.
- `pop_meta.py`: the metadata, times, medians and fractions solved of both references.
- `changed_runs.py` and `same_fields.py` on the net change between the references.
- `replay_cl.sh`: the committed `cl.py` replayed over the parent's `CHANGELOG.md` of all 16 commits that edit it.
- `refresh_citations_to.py`: `refresh_citations.py` adapted to compare two revisions, applied to the kickoff's carry of the sheet.
- `fp_outputs.py` at `86512c9`.
- `w428_identity.py` at `ffaf424`.
- fix C's `w415_compare.py` on the committed `.npz` files.
- `w430_init_stop.py` at `81385ac`.
- `seed966.py`.
- The GitHub API for `c044fea`.

## 2. What holds

**Counts:**
- There are 30 rows: W4-1 to W4-14, W4-15 to W4-20, W4-21, W4-22 to W4-28 (seven), W4-29 and W4-30.
- The reports hold 11 findings (B7 internal), 7 (B7 comparison) and 1 (O). Each is mapped to a row, and the worklog repeats the counts correctly.
- There are 9 + 5 kept items and 2 survey rows.
- The "Notes on the reports" hold as far as the reports and the verifier outputs settle them:
  - The seeds 49, 948, 967, 241, 748, 843 and 431, with the wrap at D = 7.
  - 378 to 395 seeds over 500 starts.
  - 16 of 20 runs from the same design point.
  - 123 against 128 prior draws, and 77 against 81.
  - 99 s.
  - The budgets 44 and 48.
  - `c044fea` exists on GitHub with the cited subject.

**`wave4_fixpass/` against the ledger:**
- W4-21: 34 runs changed, 25 at other points, fraction solved 0.73 → 0.63. Geometry: 30 runs changed, `ridge_D2` 1.5e-4 → 2.2e-4, p = 0.196, 0.83 → 0.77.
- W4-1: 540 runs changed; +0.27 [-0.22, +0.86]; 0.40 → 0.60 and 0.10 → 0.23; "one run or none" elsewhere.
- W4-6: 390 runs changed (13 configurations × 30), and the 5 noisy configurations identical.
- Geometry: KS 0.6, p Holm 4.97e-4, error KS p 0.393; the steps are 46-49 at mean 47.0 and 46-55 at mean 48.5.
- The fixed designs: 47.6 to 49.3; 30 of 30 runs equal W4-21's.

**The chain composes:**
- Every fraction solved chains from the wave-3 reference through W4-21 and W4-1 to W4-6 into the net comparison, for example `ellipsoid_D3_homo` 0.73 → 0.63 → 0.43 → 0.43.
- The medians of the `*_changed.txt` files chain as well. The wave-3 reference's medians equal the "before" values of `w4-21_changed.txt`. The wave-4 reference's medians equal the "after" values of `w4-6_changed.txt` for the 13 deterministic configurations and of `w4-1_changed.txt` for the 5 noisy ones.
- The net change: 540 runs changed; +0.39 [-0.10, +0.86]; -0.23 [-0.54, -0.07].

**Numbers I reproduced by running the scripts:**
- W4-21: 8 of the 104 `contraints_check` calls differ in each deterministic fingerprint run, and 0 in the noisy runs.
- W4-28: the poll returns the GP it was given in 32 of 32 polls.
- W4-15: all 9 runs identical.
- W4-6's completion: the 8 failing cases, the fingerprints `c91725823bc62b29` and `4146a986863602cb`, and the per-run logs (three runs move at 80, none at 81), all from fix E's files.

**The fix-pass table against `fp_all.out` and `git log 8c8d6f8..origin/dev-port-review-w4`:**
- All 28 fingerprints match, and the order matches the branch.
- No two rows share a hash; W4-6 has two commits.
- On the branch, `2dc5807`, `5442a6c`, `fba29cd` and the `docs(dev)` commits change no package file.
- `fp_all.out` covers every commit that changes the package, with `b78f782` under its picked hash `39b647a`, plus `2dc5807` and `5442a6c`.
- That `39b647a` and `b78f782` hold the same tree is supported by `chain_rest.log`: the retag's amend ran the hooks with "(no files to check)". `39b647a` itself is not in the clone.
- The branch's tree equals `81385ac`'s.

**The sheet:**
- The kickoff carry holds: 96 citations moved and 17 were left to a reading by hand, of which 5 were rewritten lines, 2 gpyreg citations and 10 claim citations.
- The labelled citations are right: KD-B1-1 and KD-B7-1 at `efe5e95`, KD-B1-6 at `b78f782`, and KD-B7-5 at `46af65a`.
- KD-B7-3's `406-436` is right at `0d866e8`.
- Every MATLAB line checked holds:
  - `initSobol.m:9-16`
  - `evalinitmesh.m:41-47`, `98-104`
  - `funlogger.m:117-129`, `120-121`, `130`
  - `bads.m:448-452`, `1111-1118`, `1138`, `1150-1165`
  - `acqLCB.m:10-21`
  - `acqPortfolio.m:40`, `47`, `69`
  - `searchHedge.m:45-46`
  - `setupoptions.m:26`
  - `searchES.m:125`
  - `gpupdate.m:285-292`
  - `gpdefBads.m:51`
  - `i4_sobol.m:249-250`
  - `uCheck.m:23`
  - `gpupdate.m:30`
- The texts of KD-B2-8, KD-B2-9, KD-B3-7, KD-B3-8, KD-B5-10 (43 of 180 runs, matching `wave1.md`), KD-B7-3, KD-B7-5 and the claims' preamble hold at `81385ac`.
- KD-B1-1's headline reads as "`random_seed` decides every draw", the ruling's sense. The design's scrambling itself draws from scipy's generator, seeded by one draw.

**`matlab_side_defects.md`:**
- The five new entries are present, and every MATLAB line cited holds.
- The call `mod(prod(uint64(num2str([0.25 -0.5]))),997)+1` gives 966 with exact integers and 1 with the documented formula (`seed966.py`).

**The survey:**
- The three rows are closed with true verdicts.
- W4-2's correction (`3a8096b`) is in the survey's row and in `wave2.md`'s note. The cast is present at `c7c88ab`, and the v1.1.0 output supports the date.

**`dev/TODO.md`:**
- The B3/B4 minor-items list, B7-K7's line and the small noisy budget are gone.
- W4-11's line and W0-13's note are there; for W0-13 the item already said that the search widens after an incremental improvement.
- The items spot-checked are still open: "thecurrent", "Re-evalate", the MATLAB-outputs comment, the module-level test call, and the two commented-out lines.

**`AGENTS.md`:**
- The `FunctionLogger` bullet is true: every recorded evaluation after `x0` passes `contraints_check` (`bads.py:1256`, `1979`, `2362`, `es_search.py:167`).
- The list of what draws in the randomness bullet is true.
- The architecture sentence on the design's size and its cut to the budget is true.

**The new reference:**
- Its command, commit, `pybads_source` (clean), version strings, environment, times (17:11:48–17:35:37), 0 crashed runs, range of fraction solved (0.07 to 1.00), null check (36 tests) and fingerprint all check.
- No package file changes after `46af65a`.
- `dev/README.md` names the reference. It does not name its fingerprint, as for the earlier references; the reference's README gives it.

**`cl.py`:**
- As committed, it no longer drops the blank line before "### Fixed".
- Replayed over the 16 picks, it reproduces 14 of them exactly. The two exceptions are `bd793f2`, where it keeps the blank line that the pick lost, and `fba29cd`, whose blank line was restored by hand.
- No other edit lost or duplicated text.

**The plan and the review's README:**
- The worklog lines of the kickoff, the review, the triage and the fix pass hold, apart from the points under findings F3 and F9.

## 3. Findings

### F1. W4-21's population gate: per-configuration counts under the wrong total, and "median errors held or lower" is false for `ellipsoid_D3`
- Where: `dev/experiments/port_review_20260925/verification/wave4.md:480` at 81385ac (the fix-pass table, row W4-21)
- Kind: number does not recompute
- Severity: minor
- **What is stated:** "34 of 540 runs changed, 25 of them ending at other points (`ellipsoid_D3_homo` 13, `ellipsoid_D3_hetero` 10, `rosenbrock_D2` 4, …), the median errors held or lower".
- **What `w4-21_changed.txt` says:**
  - The per-configuration numbers are runs *changed*; they sum to 34. The runs at other points are 9, 7, 4, 2, 2, 0, 0 and 1.
  - `ellipsoid_D3`'s median error rose from 2.34e-06 to 3.15e-06. `pop_meta.py` confirms 2.34e-06 in the wave-3 reference, and `w4-1_changed.txt` confirms 3.15e-06 at W4-21.
- **It already misled a later reader:** brief (b) says "13 runs ending at other points" for `ellipsoid_D3_homo`.
- Would the correction move results: no.
- **Proposed correction:** "34 of 540 runs changed (`ellipsoid_D3_homo` 13, … `sphere_D3_homo` 1), 25 of them ending at other points (9, 7, 4, 2, 2, 0, 0, 1), the median errors held or lower but `ellipsoid_D3`'s (2.3e-6 → 3.2e-6; fraction solved unchanged at 0.97)".

### F2. "Four agents:" is followed by five
- Where: `verification/wave4.md:449` at 81385ac
- Kind: false statement
- Severity: minor
- **What is stated:** "Four agents: A, …; B, …; C, …; D, …; and E, …". The plan's worklog says "Five fresh Opus fix agents".
- Would the correction move results: no.
- **Proposed correction:** "Five agents:".

### F3. The fix-pass table says W4-12 changed a docstring, and it has no row for the records-only rows
- Where: `verification/wave4.md:478` (W4-12) and `472-501` at 81385ac
- Kind: false statement
- Severity: minor
- **W4-12:** the gate column reads "none (a docstring and a description)". `f1247d0` changes only `advanced_bads_options.ini:20`. `cache_size`'s docstring, which already said "initial size", is unchanged (`function_logger.py:26-27`).
- **The missing rows:** the table lists commits only. W4-2, W4-3, W4-11 and W4-13, whose records went into `a84a3dd` and `3a8096b`, have no row. The plan's worklog cites "Fix pass" for "every row of the rulings is fixed or recorded".
- Would the correction move results: no.
- **Proposed correction:**
  - W4-12: "none (a description)".
  - Add a line under the table: "Recorded without a commit of their own: W4-2 (`3a8096b`), W4-3, W4-11 and W4-13 (`a84a3dd`: KD-B7-1, KD-B2-6 and C2; `dev/TODO.md`; KD-B7-5)".

### F4. "Each in a narrow band of its own" does not hold for two of the six fixed designs
- Where: `verification/wave4.md:501` at 81385ac
- Kind: false statement
- Severity: minor
- **What `w41_fixed_designs.out` shows:**
  - `fixed2` spans 46-53 evaluations with 7 distinct counts, and `fixed3` spans 46-53. These are about as wide as W4-1's own population (46-55, 7 distinct counts).
  - The means (47.6 to 49.3) and "948 cheapest" (47.0) hold.
  - The mean of the six other designs is 48.5, W4-1's population mean, which supports the attribution better than the "narrow band" does.
- Would the correction move results: no.
- **Proposed correction:** "six other fixed designs cost 47.6 to 49.3 evaluations on average (ranges from 48-49 to 46-53), a mean of 48.5 over the six, that of W4-1's population: 948 was the cheapest of the seven".

### F5. Sheet line citations that are not at the revision the sheet and the README state
- Where: `known_differences.md:156` (KD-B2-6), `:164` (KD-B2-7), `:106` (KD-B1-11), `:470` (KD-B7-4), `:288` (KD-B4-6), `:69` (KD-B1-6); `dev/experiments/port_review_20260925/README.md:11-16` at 81385ac
- Kind: false statement
- Severity: minor
- **What is stated:** the sheet's path conventions and the README say that the Python citations are at `0d866e8`.
- **Bare "lines N-M" citations that the kickoff's carry did not move:** `refresh_citations.py` does not parse such citations, so they are still at `8aecb6a`.
  - KD-B2-6 "lines 1071-1078, 1165-1178" is 1121-1128, 1216-1229 at `0d866e8`.
  - KD-B2-7 "1528-1545" is 1589-1606.
  - KD-B1-11 "308-329" is 324-345. At `0d866e8`, 308 is inside the `VariableTransformer` call.
  - Each text is identical at the mapped lines (a diff of the two revisions).
- **Wave 4's own entries:**
  - KD-B7-4's `function_logger.py:309-321` has no label but is at `f1247d0`/`81385ac`. At `0d866e8` the method is 303-340, and 309 is inside its docstring.
  - KD-B4-6's "812-839 … and that of W4-25 (`36e8b70`)": W4-25's check is not in that range at any revision. It is at 847-865 at `36e8b70`.
  - KD-B1-6 cites `setupvars.m:107-108` for MATLAB's `isempty`, which is at line 109 (and at 52).
- **The README omits:** the entries of wave 4's rulings cite the code the pass wrote at labelled pass commits (`efe5e95`, `b78f782`, `46af65a`). The sheet's header comment says so; the README does not.
- Would the correction move results: no.
- **Proposed correction:**
  - Carry the three bare citations: 1121-1128 and 1216-1229; 1589-1606; 324-345.
  - KD-B7-4: "`function_logger.py:303-340`".
  - KD-B4-6: "`812-839` (W3-31, W3-39), and at `36e8b70` `847-865` (W4-25)".
  - KD-B1-6: "`private/setupvars.m:107-110`".
  - README: add "except where an entry of wave 4's rulings names a commit of the pass beside its lines".

### F6. "MATLAB BADS's design depends on the start alone" overstates what is known
- Where: `AGENTS.md:250-252` and `known_differences.md:447` (KD-B7-1) at 81385ac
- Kind: false statement
- Severity: minor
- **What is stated:** both say that MATLAB BADS's design depends on the start alone.
- **What is known:**
  - It holds as "no random draw": `initSobol.m:10-12`; the `randi` of line 14 is reached only for a non-finite `u0`.
  - Whether MATLAB's design varies with the start at all is the open question of `matlab_side_defects.md`, "Questions that need MATLAB". Under the documented `mod` formula, every non-integer start gives seed 1: one design per D, as PyBADS had.
- Would the correction move results: no.
- **Proposed correction:** "MATLAB BADS derives its design from the start, with no random draw (whether it varies with the start is a question for MATLAB, `matlab_side_defects.md`)".

### F7. W4-14's MATLAB comparison sets PyBADS with its first poll against MATLAB's design alone
- Where: `matlab_side_defects.md:103-104` and `verification/wave4.md:61` at 81385ac
- Kind: number does not recompute
- Severity: minor
- **What is stated:** "at D = 2, up to 48 evaluations against MATLAB's 32". The ledger row says "up to 48 with the first poll (MATLAB up to 32)".
- **What `v4_noisy_budget.out` shows:** PyBADS's design alone ends the run in its first iteration up to 44. MATLAB's 32 is its design alone; MATLAB with the first poll was not computed.
- Would the correction move results: no.
- **Proposed correction:** "(at D = 2, up to 44 evaluations with the design alone against MATLAB's 32, and up to 48 with the first poll)".

### F8. Two statements of the new reference's README
- Where: `dev/experiments/population_linux_wave4_20260927/README.md:60-62` at 81385ac
- Kind: false statement
- Severity: minor
- **"The other steps reach no run of it":**
  - Several of those steps do run in the suite's runs. W4-28's assignment runs in every iteration. W4-16's weighted `poll_scale` runs at every refit. W4-14's rewritten final estimate and W4-26's `optim_state` block run at the end of every noisy run.
  - They change no result as far as the fingerprint shows. But the steps from W4-17 to W4-30 have no population of their own: they lie inside W4-1's comparison, where every run changes.
- **"Unflagged, the largest shifts are on the noisy 3-D ellipsoids":**
  - `comparison.md` shows `sphere_D3_hetero` moving 0.40 → 0.60 (+0.20). That exceeds `ellipsoid_D3_hetero`'s +0.13.
  - By the paired ratio, `ellipsoid_D3` (-0.37) and `ellipsoid_D3_unbounded` (-0.34) exceed `ellipsoid_D3_hetero` (-0.23).
- Would the correction move results: no.
- **Proposed correction:**
  - "the other steps change no run of it by the fingerprint (those between W4-21 and W4-1 have no population of their own)".
  - "Unflagged, the fraction solved moves most on the noisy 3-D configurations: `ellipsoid_D3_homo` …, `sphere_D3_hetero` 0.40 → 0.60 (-0.21 [-0.49, +0.00]), and `ellipsoid_D3_hetero` … (the one interval that excludes 0)".

### F9. The records of the orchestrator's tooling are incomplete or inexact
- Where: `README.md:111-133` (the wave-4 paragraph), `verification/wave4.md:466-468`, `orchestrator/same_fields.py:1` at 81385ac
- Kind: false statement
- Severity: minor
- **`cl.py` is not the version that ran.** The README says the scripts "record what ran". The committed `cl.py` is not the one that made W4-29's changelog edit. Replayed over the parent, it gives `bd793f2`'s `CHANGELOG.md` except that it keeps the blank line before "### Fixed", which the pick lost. No record says that `cl.py` was changed after `bd793f2`; only `fba29cd`'s commit message mentions the lost line.
- **Python scripts also name sandbox paths.** The README says "the shell scripts and the lists name the sandbox's paths". `cl.py` (`/home/user/pybads/CHANGELOG.md`), `w430_edit.py` and `w41_fixed_designs.py` (`…/runs/population/geo_w421_86512c9`) do too; wave 3's paragraph named `cl.py`.
- **`same_fields.py`'s docstring is false.** It says "(timings left out)", but every `_fields.txt` lists `final.wall_s`. Wave 3's paragraph records this for its copy of the script; wave 4's does not.
- **Hand runs that the records do not name.** `chain_rest.sh` runs no geometry gate at W4-1. `geo_w41_efe5e95` (`geometry_w4-1_vs_w4-21.md` and its `_fields.txt`), `geometry_w4-6_vs_w4-1.md` and `geometry_edgesphere_D2_steps.txt` were all made by hand, and no script is committed for the last. Yet the README says only "W4-6's completion and gates were run by hand", and the ledger says the geometry suite ran "for W4-21 and at the end".
- **`w430_init_stop.out` predates W4-1.** It ran in fix agent C's worktree, before W4-1. Rerun at `81385ac`, `fsd` is still 1.0, 2.5, 0.7 and 0.7, but `fval` is 3.0939 rather than 0.8829.
- Would the correction move results: no.
- **Proposed correction:** extend the README's wave-4 paragraph with these five points.
  - `cl.py` was corrected after W4-29's pick, whose lost blank line `fba29cd` restored.
  - `cl.py`, `w430_edit.py` and `w41_fixed_designs.py` name sandbox paths too.
  - `same_fields.py` lists `wall_s`, as in wave 3.
  - W4-1's geometry gate, the W4-6 against W4-1 geometry comparison and the steps summary (no script) were run by hand.
  - `w430_init_stop.out` is from agent C's branch, before W4-1.

  In `wave4.md:467`, write "for W4-21, W4-1 and at the end".

### F10. The chain ran a population gate alongside the suite, against `dev/README.md`, and nothing records it
- Where: `orchestrator/chain_rest.sh:22-27` at 81385ac; `dev/README.md:31-33`
- Kind: other
- Severity: minor
- **What the rule says:** `dev/README.md` says "a population run, the test suite and the example scripts never run concurrently".
- **What the script did:**
  - `chain_rest.sh` starts W4-1's default gate in the background (`&`, four workers) and then picks W4-6 and runs its suite and fingerprint ("# 5. W4-6, picked while W4-1's gate runs").
  - `chain_rest.log` shows the gate from 16:45 to 17:11. `e7bd01d` was committed at 16:45:48 and its suite ran for 68 s inside that window.
  - `46af65a` (committed 17:00:24, with a 149 s suite) also falls inside the window.
- No record mentions it.
- Would the correction move results: no. Seeded runs, one BLAS thread each; only the `wall_s` of W4-1's population is affected.
- **Proposed correction:** a sentence under "Fix pass": W4-1's default gate ran alongside the picks and suites of `e7bd01d` and `46af65a`, a departure from `dev/README.md`'s rule, which moves only its wall times.

### F11. Fix reports: an item left out, a false claim not flagged, and W4-10's `TODO.md` line
- Where: `verification/wave4.md:526-573`; `dev/TODO.md:230-265` and `:173-175`; `fixes/D_W4-29.md:52`; `fixes/E_W4-6_completion.md:50` at 81385ac
- Kind: false statement
- Severity: minor
- **An item left out.** "Found while fixing" says it holds the fix agents' items, the minor ones in `dev/TODO.md`. It omits E's item: `test_get_gp_training_options_samplers` and `_opts_N` assign `hyp_dict_none` and never use it. This is still true at `test_gaussian_process_train.py:185` and `:219`.
- **A false claim not flagged.**
  - D's report says `6e24519` "edited `CHANGELOG.md` itself, although fix agents are told not to". Those lines are the orchestrator's: `batch2.list` gives `9189f9f|changed …w418_changed.txt;;upgrading …w418_up.txt`.
  - Replaying `cl.py` gives `6e24519`'s `CHANGELOG.md` from its parent exactly, so the agent's commit touched no changelog.
  - No record flags the claim.
- **W4-10's `TODO.md` line.** It names "what it records for a repeated point" but not what W4-10 actually left to the port of `fun_values`: that `add` records a missing SD as 1 when the logger holds SDs and drops a given one when it does not.
- Would the correction move results: no.
- **Proposed correction:**
  - Add E's item to "Found while fixing" and to the `TODO.md` item.
  - Add a note: "D's report is wrong that `6e24519` edited `CHANGELOG.md` itself: its lines are the orchestrator's (`changelog_entries/w418_*.txt`)".
  - W4-10's `TODO.md` line: "…keeps checks of its own on the value and its SD, and records a missing SD as 1 (or drops a given one when the logger holds none) …".

## 4. Outside my scope
- The same "design depends on the start alone" wording (F6) is in `init_sobol.py:56-60`'s comment and in the changelog's W4-1 entry: reviewers (a) and (c).
- W4-6's pass suite ran with `-x` and stopped at `test_get_gp_training_options_small_budget[2-2]` ("1 failed, 268 passed"). The "8 of 8 cases" failure comes from fix E's rerun at `e7bd01d`. The rest of the suite never ran at `e7bd01d`; it passed in full at `46af65a`. Reviewer (a), for how the ledger records the failure.
- Every row of the kickoff carry and the verifier numbers checked hold. I did not check the code behavior of the checks (complex scalars, one-element arrays): reviewer (b).

Scratch outputs are in `/home/user/dc4/d_records`: `cmp_wave3_wave4.md`, `null_wave4.md`, `summary_wave4_recomputed.md`, `pop_meta.out`, `replay_cl.out`, `rc_ed82ec0_8aecb6a_to_0d866e8.txt`, `fp_outputs_86512c9.out`, `w428_identity_ffaf424.out`, `w430_init_stop_81385ac.out`, and the scripts that made them. The scratch copies of the populations and trees were removed.
