# Wave 3 doublecheck: the reviewers' briefs

The four prompts of the doublecheck of wave 3 (PI, 2026-09-27), after its
fix pass was squash-merged into `dev-next` as `0d866e8` (#77): the common
part below, from "You are one of four reviewers" to the end of "What to
return", then the scope part of the reviewer, with the placeholders
replaced. Each reviewer is a fresh, read-only general-purpose Opus agent;
the four ran at once. Each agent's prompt named the file that held its
brief, to be read in full first, and repeated its main rules (read-only, no
`git stash`, checks run from the scratch directory, the report's title).

- `{SCOPE}`: the scope's name, as in the title of its part;
- `{SCRATCH}`: a scratch directory of the reviewer's own, outside every
  repository; its contents are copied to
  `verification/scripts/wave3/doublecheck/<letter>_<scope>/`.

The paths are those of the cloud session that ran it:
`/home/user/pybads-review`, a detached worktree of this repository at
`0d866e8`; `/home/user/bads`, MATLAB BADS at `74919c0`;
`/home/user/gpyreg-v1.3.3`, gpyreg at `v1.3.3`; `/home/user/pybads-v1.1.0`,
a detached worktree at the tag `v1.1.0`; `/home/user/pybads/.venv/bin/python`,
the venv of `AGENTS.md` (Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1).

---

You are one of four reviewers in a doublecheck of wave 3 of a correctness review of PyBADS, the Python port of the MATLAB optimizer BADS. Wave 3 reviewed slice B3 (the search) and slice B4 (the poll, the mesh, the incumbent and the target) on two tracks (internal correctness, and comparison with MATLAB BADS); verifiers checked every finding and wrote the ledger; the PI ruled on each row; and a fix pass implemented the rulings, one commit per row, gated by the fingerprint and by population comparisons. The pass was squash-merged into `dev-next` as `0d866e8` (#77). Your scope is **{SCOPE}** (below). You are read-only: you report what does not hold; you do not edit, commit or fix anything.

For every item of your scope you check three things:

1. **The ruling.** Each row implements its ruling: the ledger's "Rulings", with "After the gates" where it applies, and the choices the orchestrator recorded under "Fix pass" ("Choices within the rulings"). A choice that departs from the ruling's words is a finding only if the ledger does not record it or its reason does not hold.
2. **MATLAB.** The comparisons with MATLAB BADS that the rulings rest on hold: read the MATLAB lines yourself, at `74919c0`, and where it settles the question, transcribe the MATLAB function into Python and compare it with the port on the same inputs.
3. **The statements.** Every statement in your scope is true of the code at `0d866e8`: the ledger's text on the row, the comments and docstrings of the code the row changed, the test names and docstrings, and the other records your scope names. Commit messages cannot be changed; report a false one only when a record relies on it.

## Where things are

- `/home/user/pybads-review`: a detached git worktree of PyBADS at `0d866e8`, the revision under check. Cite lines at this revision. The repository's whole history is there: every commit of the fix pass is on `origin/dev-port-review-w3` (`git log --oneline 8aecb6a..origin/dev-port-review-w3`), whose tree equals `0d866e8`'s; `8aecb6a` is the revision that wave 3 reviewed, `68d4516` is `0d866e8`'s parent on `dev-next` (the doublecheck of wave 2, merged into the pass's branch at `4f50376`), and the tag `v1.1.0` is the last release. Read the repository only through this worktree and git: `/home/user/pybads` is the orchestrator's checkout, which changes while you work.
- The records of the review, in that worktree, under `dev/experiments/port_review_20260925/`: the ledger `verification/wave3.md` (rows W3-1 to W3-40, "Notes on the reports", "Found while verifying", "Rulings" with "After the gates", "Fix pass" with "Found while fixing"); the verifiers' reports `verification/wave3_B3_verifier.md` and `wave3_B4_verifier.md`; the reviewers' reports `reviews/B3_*.md` and `reviews/B4_*.md`; the fix agents' reports `fixes/*_W3-*.md` (their hashes are those of their own branches, not of `origin/dev-port-review-w3`); the gates' records `verification/wave3_fixpass/`; the scripts `verification/scripts/wave3/` (the orchestrator's under `orchestrator/`, among them `fp_all.out`, the fingerprint at every commit of the pass); the known-differences sheet `known_differences.md`; `matlab_side_defects.md`; the plan `dev/plans/port-correctness-review.md` (its worklog and "Wave 4 pickup"); and the ledgers of waves 1 and 2 (`verification/wave2.md`, "Doublecheck", is the previous doublecheck). The new Linux reference population is `dev/experiments/population_linux_wave3_20260927/`.
- `/home/user/bads`: MATLAB BADS at `74919c0` (v1.1.3), with its history (`git log -L<start>,<end>:<path>`).
- `/home/user/gpyreg-v1.3.3`: gpyreg at the tag `v1.3.3`, the version every gate ran.
- `/home/user/pybads-v1.1.0`: a detached worktree at the tag `v1.1.0`; it runs with the same gpyreg.
- `AGENTS.md` and `CLAUDE.md` may be loaded into your context automatically; `AGENTS.md` is itself a record under check (reviewer (d)'s), not the specification.

## Checks you may run

Small checks only: short scripts, single function calls, and `BADS.optimize()` runs of at most 200 evaluations, one at a time, with one BLAS thread. Do not run the test suite, `population.py run`, sweeps, `fingerprint.py` or installs: the orchestrator runs the whole suite at `0d866e8` and the fingerprints at the pass's key commits. If a statement can only be settled by a heavier run, say so in your report with the command, and the orchestrator decides. There is no MATLAB. Run scripts as:

```
cd {SCRATCH}
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="/home/user/pybads-review:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -u your_script.py
```

Every script prints `pybads.__file__` and `gpyreg.__file__` once. Run every script and every import check from `{SCRATCH}`, never from the root of a repository or worktree: Python puts the current directory (for `python -c` and `python -m`) or the script's own directory first on `sys.path`, so a check run from a repository's root imports that root's `pybads/` whatever `PYTHONPATH` says. The scripts under `dev/scripts/` that import `benchmark_targets.py` (`population.py`, `gp_update_failures.py` and others) select PyBADS by their own checkout, put first on `sys.path`, whatever `PYTHONPATH` says; `fingerprint.py` follows `PYTHONPATH`. To run PyBADS at another commit, extract it into your scratch directory, `git -C /home/user/pybads-review archive <rev> pybads | tar -x -C {SCRATCH}/at_<rev>`, and put `{SCRATCH}/at_<rev>` first on `PYTHONPATH`; for 1.1.0 use `/home/user/pybads-v1.1.0`. Seed every run (`random_seed`; a noisy target draws from its own `np.random.default_rng(seed)`).

## Rules

- Tracked files in every repository and worktree are read-only. Scripts and outputs go only into `{SCRATCH}`.
- Never use `git stash`, and no other git command that changes a working tree, the index or the refs (`checkout`, `switch`, `reset`, `worktree`, `commit`, `fetch` and the like), in any repository or worktree: the stash stack and the refs are shared by all the worktrees of a repository and by the other agents. Read-only commands (`git log`, `git show`, `git diff`, `git blame`, `git grep`, `git archive`) are fine; `git show <rev>:<path>` reads a file at another revision.
- Stay within your scope; the other three reviewers cover the rest (the scopes are listed below). Something you meet outside your scope goes, in a line, under "Outside my scope".
- Use the plain vocabulary of code review and debugging: review, reviewer, finding, discrepancy, defect, reproduction.

The four scopes: (a) the fixes of B3; (b) the fixes of B4; (c) the user-facing documentation; (d) the records, gates and tooling.

## What to return

Your final message is the report (report files are not accepted), titled `# Wave 3 doublecheck: {SCOPE}`, in four parts:

1. **Coverage**: what you read completely, what you skimmed, what you did not reach.
2. **What holds**: one line per item of your scope (a row, a record, a gate), with the evidence you checked.
3. **Findings**, each in this format:

```
### F<n>. <one-line title>
- Where: <path>:<line> at 0d866e8 (the record's or the code's)
- Kind: ruling not implemented | MATLAB comparison does not hold | false statement | number does not recompute | defect of a fix | other
- Severity: substantial (it misleads a user or a later wave, or the code does not do what its ruling says) | minor (a wording or a number that misleads little)
- What is ruled or stated, what is true, and the evidence (code lines, MATLAB lines, a script and its output).
- Would the correction move results: no | yes (which runs) | unsure. Say whether a fix of the code would change any run at default options or behind which option.
- Proposed correction: the replacement text for a statement; for code, a description.
```

4. **Outside my scope** (if any): one line each.

A report with no findings says so; the coverage and "What holds" are then the deliverable.

---

## Scope (a): the fixes of B3

Rows W3-1 to W3-15 of the ledger (W3-18 rides with W3-11), and the root logger (the first item of "Found while verifying", ruled under "Fix, moving nothing"), with the gates of the ES batch (W3-4, W3-5, W3-15) and of W3-1, and batch 1, which is W3-14's gate.

- The commits on `origin/dev-port-review-w3`: W3-11 and W3-18 `4388e6d`, W3-7 `4d357e4`, W3-10 `599115b`, W3-14 `1f7c8ee`, W3-9 `a77d95d`, W3-8 `c788617`, W3-13 `4865fad`, the root logger `d0c7178`, W3-2 and W3-3 `7e09887`, W3-4 `81c6a15`, W3-5 `115a922`, W3-15 `c276d79`, W3-1 `149d528`, W3-6 `d79ab75`; `bd110f0` and `6281663` changed the records of W3-14 and W3-1 after their gates. W3-12 was ruled "keep, and record" (KD-B5-2 extended); W3-16 and W3-17 no longer hold.
- The code: `pybads/bads/bads.py` (`_search_step_`, `_update_search_bounds_`, `_update_search_stats_`, and wherever the rows' commits touched it), `pybads/search/es_search.py`, `search/search_hedge.py`, `search/grid_functions.py`, `search/__init__.py`, `acquisition_functions/acq_fcn_lcb.py`, `function_logger/constraints_check.py`, and the tests the commits added or changed (`pybads/testing/bads/search/test_search.py`, `test_empty_search.py`, and others). The MATLAB counterparts: `utils/uCheck.m`, `search/searchES.m`, `utils/ESupdate.m`, `utils/ucov.m`, `search/searchHedge.m`, `acq/acqPortfolio.m`, `acq/acqLCB.m`, `utils/force2grid.m`, and `bads.m` (`UpdateSearch`, the search stage).
- The fix agents' reports: `fixes/A_W3-11_W3-7_W3-10_W3-6.md`, `fixes/C_W3-9_W3-8_W3-13_logging_W3-2_W3-3_W3-4_W3-5_W3-15_W3-1.md`, and the W3-14 part of `fixes/B_W3-20_W3-23_W3-31_W3-33_W3-34_W3-36_W3-25_W3-28_W3-14.md`.
- The gates: `verification/wave3_fixpass/batch1_vs_reference.md` and `w3-14_attribution.txt`; `es_vs_batch1.md`; `w3-1_vs_es.md`, `geometry_w3-1_vs_es.md`, `geometry_es_summary.md` and `w3-1_repeats.txt` (with `scripts/wave3/orchestrator/count_repeats.py`), each with its `_fields.txt`. Check that each compares the commits and the suite its row says, that it reached the changed code, and that the ledger's reading of it matches the file. Reviewer (d) recomputes every number of the records; you check what the gates show about the rows.
- Among the questions: does `contraints_check` now remove exactly what `setdiff(u1, u2, 'rows')` removes, on the same bins (the rounding of `uCheck.m`), for the search, the poll and the initial design, and is W3-2's sorted order still MATLAB's? Do the ES's selection mask, its ⌊μ⌋ rows, its stable sorts and its count of new candidates equal `searchES.m` and `ESupdate.m` on the same inputs (and does any other sort of the search remain unstable where MATLAB's is stable)? Does W3-9 keep the earlier candidates as `searchES.m` keeps `zold`, and what does the warning say? Does the empty set's decay of the hedge's gains (W3-11) equal MATLAB's `acqPortfolio.m` update at er = 0? Is φ(γ) (W3-6) now exact? Does `hedge_gamma = 0` (W3-7) score each search at the point as MATLAB's line 40 intends? Is W3-10's validation what the ruling says (a positive finite number, a callable or `None`)? Does W3-14's rounding equal `force2grid.m` on every double (halves, the largest double below one half, negatives, infinities, NaN)? Is anything of `ESSearchCMA` and `active_flag` left (W3-13)? Does creating an ES search leave the root logger alone, and does `BADS` still configure it as KD-B2-3 says?

## Scope (b): the fixes of B4

Rows W3-19 to W3-36, W3-39 and W3-40, W3-29's gate, and W3-24 with its gate and its revert.

- The commits on `origin/dev-port-review-w3`: W3-20 `5f31837`, W3-23 `dac062e`, W3-31 `ec1b2d0`, W3-33 `43ee8ed`, W3-34 `01ee524`, W3-36 `e4b3bca`, W3-25 and W3-28 with the descriptions `fd8641d`, W3-22 `f595f1b`, W3-27 `a1bf658`, W3-19 `8e28124`, W3-29 `0b7add3`, W3-24 `869a033` and its revert `b03a320`, W3-39 `5d711bf`, W3-40 `a14524d`; `391373e`, `5d43aa8` and `1af364f` wrote the records after the gates. W3-21, W3-26, W3-28, W3-30 and W3-35 were ruled "keep, and record"; W3-32, W3-37 and W3-38 no longer hold.
- The code: `pybads/bads/bads.py` (`_poll_step_`, `_eval_improvement_`, `_is_poll_stop_`, `_get_target_from_gp_`, `_update_incumbent_`, the re-estimate of W3-33, the checks of `__init__` for W3-31 and W3-39, the search's call sites of W3-20 and W3-27, the rebuild flag of W3-29 in the main loop and the search), `pybads/poll/poll_mads_2n.py`, `pybads/bads/gaussian_process_train.py` (W3-40), the descriptions in `pybads/bads/option_configs/*.ini` that the rows changed, and the tests the commits added or changed (`test_run_control.py`, `test_bads_seed.py`, `poll/test_poll_mads.py`, `test_bads_inputs.py`, `test_noisy_runs.py`, `test_gaussian_process_train.py`, `test_gp_update_failures.py`, and others). The MATLAB counterparts: `bads.m` (the poll stage, `EvalImprovement`, `UpdateIncumbent`, `UpdateTarget`, the flags `pollmoved_flag` and the end of the pass, lines around 800-1060 and 1240-1340), `poll/pollMADS2N.m`, `private/setupvars.m`, `gpdef/gpdefBads.m` (the empirical priors, around lines 230-260).
- The fix agents' reports: `fixes/B_W3-20_W3-23_W3-31_W3-33_W3-34_W3-36_W3-25_W3-28_W3-14.md` (not its W3-14 part) and `fixes/D_W3-22_W3-27_W3-19_W3-29_W3-24.md`.
- The gates: `verification/wave3_fixpass/w3-19_vs_w3-6.md`, `w3-29_vs_w3-19.md`, `w3-29_vs_reference.md`, `w3-24_vs_w3-29.md`, `geometry_w3-24_vs_w3-29.md`, `geometry_w3-24_crashes.txt`, `w3-24_vs_reference.md`, `w3-40_crashed_runs.txt`, `geometry_w3-29_summary.md`, `medians_default.md`, `medians_geometry.md`, `head_vs_w3-29.md`, `geometry_head_vs_w3-29.md`, each with its `_fields.txt`; the ledger's paragraph "W3-24's flagged worsening, and its revert". Check that each compares the commits and suites it says, that it reached the changed code, and that the ledger's reading of it, and the rulings made on it, rest on what the file shows. Reviewer (d) recomputes every number of the records; you check what the gates show about the rows.
- Among the questions: does `p_less` (W3-19) equal `bads.m:868-869` on the same probabilities, for n below and above D? Does W3-29 rebuild at every search after a poll that moved the incumbent until a poll that does not, and once after a search move, exactly as `pollmoved_flag` and `bads.m:1049` do, including a search move in between and a poll that makes no rebuild of its own? Does W3-27's random choice draw from the run's generator, and are both sites covered? Does `np.errstate` (W3-22) cover what the global `np.seterr` covered in the poll, and nothing that should warn? Do W3-31's and W3-39's checks refuse and accept what the ledger and the changelog say (floats that are whole numbers, NumPy integers, booleans, `None`, strings such as `"3"`), and compare with MATLAB's checks? Does W3-33 keep `optim_state`'s `yval`, `fval` and `fsd` in step at every path of the re-estimate? Does `b03a320` leave no trace of W3-24 in code, tests, docstrings or user documents beyond what the ruling after the gates asks for (the docstring of `poll_mads_2n` that says what its basis is), and is that docstring true? Does W3-40 keep the previous prior exactly when a rebuild has two distinct points, as the ruling says, and what does MATLAB's `gpdefBads.m` do there; is the difference recorded where it should be (the sheet, the changelog)?

## Scope (c): the user-facing documentation

Wave 3's entries of `CHANGELOG.md` and its lines of the "Upgrading from 1.1.0" list (`git -C /home/user/pybads-review diff 68d4516 0d866e8 -- CHANGELOG.md`; entries that wave 3 extended or replaced are yours in full), against the code at `0d866e8` and against release 1.1.0; then the docstrings, the option descriptions, `README.md` and `docsrc/source/` as they stand at `0d866e8`, with priority on what wave 3 changed (`git diff 68d4516 0d866e8 -- pybads README.md docsrc`).

- The rules the changelog follows (`AGENTS.md`, "Conventions"): a change that a user can notice is listed under `Unreleased`, in a sentence written for users and relative to the last release, 1.1.0 (a fix to a feature that no release has shipped belongs to that feature's entry); a change that can stop a script written for 1.1.0, or change what it returns, also has one line in "Upgrading from 1.1.0", kept in step with its entry. The ledger's "Fix pass", "Changelog", says which rows have lines and which have none (W3-2, W3-3, W3-23, W3-33, W3-34, W3-36): check that none of the rows without a line is noticeable by a user of 1.1.0, and that what an entry says 1.1.0 did, 1.1.0 did (run it at `/home/user/pybads-v1.1.0`).
- Check every claim of an entry: what the code does now, what 1.1.0 did, "as in MATLAB BADS" (read the MATLAB lines), and every number (for instance the benchmark's 31 of 540 runs, 137 of 1411 evaluations, 100 of 8800, "2.5 to hundreds of times"), against the ledger's gates. "The initial design, the search and the poll no longer evaluate again a point already evaluated" is one such claim.
- Docstrings: of every function and class the pass touched (`acq_fcn_lcb`, `update_hedge` and the hedge, the ES classes, `contraints_check`, `force_to_grid`, `poll_mads_2n`, `_poll_step_`, `_get_target_from_gp_`, `BADS`'s parameters and Raises section, and the others the diff shows), in the numpydoc style of `pybads/bads/optimize_result.py`. The option descriptions: the comment line above each option in `pybads/bads/option_configs/basic_bads_options.ini` and `advanced_bads_options.ini`, which the options page includes verbatim; first those the pass changed or its rulings name (`hedge_gamma`, `gp_rescale_poll`, `tol_poi`, `sloppy_improvement`, `improvement_quantile`, `accelerate_mesh_steps`, `search_acq_fcn`, `n_search_iter`, `search_method`, `poll_training`, `uncertain_incumbent`), then the rest of those of the search and the poll. `README.md` and `docsrc/source/index.rst` on the search and the poll (W3-24's text should have gone with its revert), and the API pages under `docsrc/source/api/` of what the pass removed or changed (`ESSearchCMA`).
- These texts are read by a user who has only the final text: flag a sentence that exists only because of the review's history (a qualifier pointing at context the text never gives, a "now" that describes a change where a state is meant outside the changelog, a reassurance about a concern the text never raised).

## Scope (d): the records, gates and tooling

- **The numbers.** Every count and number of `verification/wave3.md` and of `verification/wave3_fixpass/`, recomputed from the committed records: the comparison files and their `_fields.txt`, `w3-1_repeats.txt`, `w3-14_attribution.txt`, the crash files, the medians, the reference populations `population_linux_wave2_20260926` and `population_linux_wave3_20260927` (whole), and `fp_all.out`. You may run `population.py compare` and `population.py summary` (not `run`) on committed populations, from the review worktree's `dev/scripts/`; a comparison of the pass's intermediate steps can be checked only against its committed file. The fingerprint table of "Fix pass" against `fp_all.out` and the commits of `origin/dev-port-review-w3` (hashes, order, which rows share a hash). The row counts (W3-1 to W3-40), the counts of findings per report, of kept items and of survey rows, and the statements of "Notes on the reports" and "Survey rows" that the reports and the survey can settle.
- **The sheet** (`known_differences.md`): the entries that wave 3's rulings add, correct or remove (KD-B3-1 with W3-13, KD-B3-3 with W3-6, KD-B4-2 with W3-21, KD-B5-2 with W3-12 and W3-30, the empty search set of W3-11, the rebuilds of W3-29 if any, KD-B4-3 gone with W3-24's revert, and any other), against the code at `0d866e8` and MATLAB at `74919c0`, and the revision of their line citations against what the port review's `README.md` says of the sheet's citations.
- **`matlab_side_defects.md`**: the entries wave 3 added (W3-3, W3-7, W3-23, W3-24's inverted ratio with its gate's evidence, W3-26, W3-39, MATLAB's 0/0 in the ES scale, and any other), each against the MATLAB lines it cites.
- **The survey** (`dev/results/2026-09-23-codebase-survey.md`): the 11 rows of B3 and B4 that the ledger closes, each closed with its row and a true verdict.
- **`dev/TODO.md`**: the lines the rulings ask for ("Out of this pass": the copy of the GP and the recomputed target at every search step; the zero predictive SDs at level 0), the items of "Found while fixing" that went there, "Previously evaluated points evaluated again", which W3-1 closes, and nothing listed as open that the pass did.
- **`AGENTS.md`**: every statement that wave 3's code or rulings touch (`poll_scale`, `poll_mads_2n` and its bound, the extension points without `ESSearchCMA`, the randomness bullet's list of what draws, the GP update and its markers, W3-40's prior if mentioned), true of the code at `0d866e8`.
- **The geometry suite** of `dev/scripts/benchmark_targets.py` (`8824c9e`): its targets do what the records say (a sphere with its minimum on a lower bound, nonsmooth ridges along the diagonal, a thin feasible band), their optima and tolerances hold (`benchmark_targets.py --check`, if it is light; say what you ran), the `default` suite is unchanged by the commit (so that the references still pair by seed), and `dev/README.md` describes the suite.
- **The orchestrator's scripts** (`verification/scripts/wave3/orchestrator/`): each does what the records say it did (`fp.sh` and `fp_all.out`, `gate.sh`, `gates_chain.sh`, `pick.sh`, `picks_run.sh`, `attrib.sh`, `one_run.sh`, `count_repeats.py`, `resolve_appends.py`, `same_fields.py`, `medians.py`, `cl.py`, `w3_acc0.py`), and the records say where they depend on the sandbox's paths.
- **The new reference** `population_linux_wave3_20260927`: its `README.md` (the command, the provenance, the commit, the environment, the null check, the comparison with the previous reference, what "no flag" can detect), its records' `meta` (the commit, `pybads_source`, the versions), and `dev/README.md` and the plan's "Wave 4 pickup", which name it and its fingerprint.
- **The plan and the review's README**: the worklog lines of wave 3 (kickoff, run, triage, fix pass, merge), "Wave 4 pickup" (what it says wave 3 changed that B7 and O read), and the port review's `README.md` paragraph on wave 3.
