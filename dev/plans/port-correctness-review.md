# Plan: independent correctness review of PyBADS and its MATLAB port

Started 2026-09-25. Owner of the item: `TODO.md`, "Bug hunt and
verification against MATLAB BADS". This file holds the design, the reviewer
brief, the working rules and the worklog. The records of the review (the
known-differences sheet, the counterpart map, the reviewers' reports, the
verification scripts and the per-wave ledgers) are kept under
`experiments/port_review_20260925/`; the consolidated ledger goes to
`results/<date>-port-correctness-review.md` at the close.

The design follows PyVBMC's review of its own port
(`../pyvbmc/dev/plans/port-correctness-review.md`, 2026-09-19 to 09-24),
scaled to a smaller package: about 5,400 lines of numerical code, 60% of
them in `pybads/bads/bads.py` and `pybads/bads/gaussian_process_train.py`.

## Purpose

Several independent reviewers read the PyBADS code for errors and latent
defects, half of them checking internal correctness and half comparing the
code with MATLAB BADS, with a third reader on the formulas of the
improvement, the acquisition and the GP geometry. The survey
(`results/2026-09-23-codebase-survey.md`) found defects by one reading and
checked few of them against MATLAB; its candidate table is the evidence that
a systematic pass is needed, not the pass itself. Passing tests and the
fingerprint pin the current behavior; they do not establish that the port
is right.

Coverage follows the categories of PyVBMC's review: formulas, indexing and
array shapes, defaults, control flow, random draws, state and caching, and
cross-module behavior.

## Decisions (PI, 2026-09-25)

- gpyreg's internals are out of scope: PyVBMC's review read them (its waves
  6 and 7, gpyreg 1.2.1 to 1.3.3). How PyBADS uses gpyreg is in scope,
  compared with how MATLAB BADS uses GPML. That includes gpyreg's
  `RationalQuadraticARD`, the only kernel PyBADS uses: it has no `gplite`
  counterpart, PyVBMC never reaches it, and PyVBMC's review read it on the
  internal track alone. Its reference is GPML's `covRQard` as BADS calls it
  (`gpml_fast/covRQard_fast.m`).
- The MATLAB comparison target is the latest `master` of `acerbilab/bads`,
  v1.1.3 (below), so that the review also finds what moved in MATLAB after
  the port and was never carried over.
- Reviewers are Opus agents. Fable is used only on request.
- At most four agents run at a time, as a guide against runaway fan-outs.
- Agents may run small checks: short scripts, single function calls, and
  `BADS.optimize()` runs of at most 200 evaluations, one at a time, with
  one BLAS thread. They do not run the test suite, populations, sweeps,
  installs, or anything else that competes for the one heavy-compute slot,
  which the orchestrator holds.
- No MATLAB run is planned. A run is written up, for a developer with
  MATLAB to perform, only if a finding is classified **needs MATLAB** and
  its disposition depends on what MATLAB computes.
- Confirmed findings are brought to the PI for triage before any fix.
- A fix may make an interface stricter: a value that was accepted and now
  fails with a clear message costs a user one correction in one place. A
  fix must not change behavior silently.
- Five waves (below).

**Terminology.** The records and the agent prompts use the vocabulary of
code review and debugging: review, reviewer, finding, discrepancy, defect,
reproduction, disposition. They avoid the vocabulary of computer security,
which automated content classifiers have misread in earlier debugging
sessions.

## Reference revisions

| Code | Location | Revision |
| --- | --- | --- |
| PyBADS under review | this repository, `dev-next` | `ab4dded` for wave 0; `95da7f1` for waves 1 to 4 (the freeze: `dev-next` after #70 and #71) |
| gpyreg | the clone `dev/scripts/runs/gpyreg/v1.3.3` | `98ab5a4` (v1.3.3) |
| MATLAB BADS, comparison target | `../bads`, `master` | `74919c0` (v1.1.3, 2025-12-05; equal to the remote `master` on 2026-09-25) |

Reviewers read PyBADS in a detached worktree at the revision of their wave,
outside this repository (`../pybads-review`), so that the orchestrator's
edits in the main checkout do not reach them.

The port has no single MATLAB source revision. PyBADS's first commit,
`9e63b0f`, is dated 2022-02-11; the first ported algorithm is `c7c88ab`
(2022-06-02), the noisy version `8e59038` (2022-06-03), the full version
with Sto-BADS `9037851` (2022-09-22), and v1.0.0 was tagged on 2023-06-10.
MATLAB has eight commits touching code since 2022-02-11, which slice M
checks one by one:

| MATLAB commit | Date | Change |
| --- | --- | --- |
| `8515191` | 2022-05-06 | `~isempty` check on `gpstruct.s` (`utils/gpHyperOptimize.m`) |
| `bfe8e22` | 2022-05-06 | removal of points for stability with user-specified noise (`bads.m`, `utils/gpHyperOptimize.m`) |
| `75ec49f` | 2022-05-09 | uncertainty handling and user-specified noise, printing (`bads.m`, `private/bads_output.m`, `private/evalinitmesh.m`, `private/setupoptions.m`) |
| `c4d2b9a` | 2022-10-30 | `bads.m` update |
| `d4fead5` | 2022-10-31 | v1.1.0: `utils/gpTrainingSet.m` rewritten as `private/gpupdate.m`; `setupoptions.m`, `setupvars.m`, `gpHyperOptimize.m`; `updatehess.m` and `multibayes.m` removed |
| `a21f2ee` | 2022-10-31 | v1.1.1, updated defaults (`bads.m`, `private/evalinitmesh.m`), among them `Ninit` from `nvars` to `10 + nvars`, which `019f0b4` set back |
| `019f0b4` | 2022-11-14 | v1.1.2, fixes with user-specified noise (`bads.m`, `private/bads_output.m`, `private/setupvars.m`); `Ninit` back to `nvars` |
| `74919c0` | 2025-12-05 | v1.1.3, `error_index` to `err_index` in `private/gpupdate.m` |

A discrepancy is dated during verification: the verifier reads the history
of the MATLAB lines (`git log -L` in `../bads`) and of the Python lines, and
records whether the Python ever matched MATLAB, whether MATLAB changed after
the module was ported, or whether the two never agreed.

Two facts about the GP layer shape slices B5 and B6:

- MATLAB BADS's GP is GPML 3.6 (`gpml-matlab-v3.6-2015-07-07/`, third-party
  and out of scope as a library) with BADS's own fast replacements
  (`gpml_fast/`) and its training and prediction code (`gpdef/gpdefBads.m`,
  `private/gpupdate.m`, `utils/gpHyperOptimize.m`, `utils/minimizebnd.m`,
  `utils/mygp.m`, `utils/gppred.m`, `utils/update_posterior.m`,
  `utils/likGaussHe.m`). The port maps these onto gpyreg objects, whose
  parameterizations need not match GPML's: every hyperparameter, bound and
  prior is compared in the units both sides use.
- `gaussian_process_train.py` calls gpyreg's name-mangled private
  `gp._GP__gp_obj_fun`.

## Slices

Each slice gets two reviewers, one per track, except M and S (one each) and
O (a third reader that combines both tracks: re-derive, then compare with
MATLAB). MATLAB files marked *unported* have no Python counterpart; the
comparison reviewer confirms that and spends no more time on them.

| Slice | Python | MATLAB counterparts |
| --- | --- | --- |
| **B1** Setup, options, defaults, bounds, transform, result | `bads.py`: `__init__`, `_bounds_check_`, `_init_optim_state_`, `_init_rng_`; `options.py`, `option_configs/` (the two option files, `options_confs.py`; `test_options*.ini` are test fixtures shipped in the package), `variable_transformer/`, `optimize_result.py`, `rng.py`, `search/grid_functions.py:grid_units`, the package's `__init__.py` files (exports) | `bads.m` up to the main loop (`defopts`, the setup), `private/setupvars.m`, `private/setupoptions.m`, `private/boundscheck.m`, `utils/transvars.m`, `utils/origunits.m`, `utils/gridunits.m`, `utils/maskindex.m`, `utils/evalbool.m` (no counterpart), `private/bads_output.m`. The comparison report holds the full defaults table: every `defopts` entry against the `.ini` files, at several `D` |
| **B2** Main loop, termination, noisy re-evaluation, final estimate | `bads.py`: `_init_mesh_`, `_init_optimization_`, `optimize`, `_re_evaluate_history_`, `_check_mesh_overflow_`, the display; `utils/iteration_history.py` | `bads.m` from the main loop to the end, its subfunctions `reevaluateIterList`, `FinalEstimate`, `meshOverflowCheck`; `private/evalinitmesh.m`. Unported: `private/fixedbads.m` and `expandvars` (fixed variables, which PyBADS refuses), `private/scatterplot.m` (plotting) |
| **B3** Search | `bads.py`: `_search_step_`, `_update_search_bounds_`, `_update_search_stats_`; `search/es_search.py`, `search/search_hedge.py`, `search/grid_functions.py` (`force_to_grid`, `udist`), `acquisition_functions/acq_fcn_lcb.py`, `function_logger/constraints_check.py` (`ESSearchCMA` is unreachable) | the subfunctions `UpdateSearch` and `updateSearchBounds` of `bads.m`; `search/searchHedge.m`, `search/searchES.m`, `acq/acqLCB.m`, `acq/acqPortfolio.m` (its `'upd'` branch, ported as `update_hedge`), `utils/ESupdate.m`, `utils/uCheck.m`, `utils/force2grid.m`, `utils/udist.m`, `utils/ucov.m`. Unported and unused by MATLAB's defaults: `search/searchWCM.m` (not the counterpart of `ESSearchWM`, which is `searchES` method 1), `search/private/*`, `utils/xCheck.m`, `acq/acqHedge.m` and the other search and acquisition functions |
| **B4** Poll, mesh, incumbent, target | `bads.py`: `_poll_step_`, `_eval_improvement_`, `_is_poll_stop_`, `_get_target_from_gp_`, `_update_incumbent_`; `poll/poll_mads_2n.py`. Sto-BADS belongs to S | `poll/pollMADS2N.m`, the poll stage of `bads.m` and its subfunctions `EvalImprovement`, `UpdateIncumbent`, `UpdateTarget`. Unused by MATLAB's defaults: `poll/private/*`, `private/covmatadapt.m` |
| **B5** GP training set and refit policy | `gaussian_process_train.py`: `local_gp_fitting`, `get_grid_search_neighbors`, `add_and_update_gp`, `_robust_gp_fit_`, `_get_gp_training_options`, `_get_fevals_data`, `_estimate_noise_` (from PyVBMC; its result is unused); `bads.py`: `_is_gp_refit_time_` (which inlines `gppredcheck`), `_save_gp_stats_`, `_record_gp_refit_` | the subfunctions `IsRefitTime` and `savegpstats` of `bads.m`; `private/gpupdate.m`, `utils/gppredcheck.m`, `utils/swtest.m` (substituted by `scipy.stats.shapiro`), `utils/gpHyperOptimize.m` (the policy: starting points, nudges, refit conditions); `utils/update_posterior.m`, compared against its absence (the rank-1 path is not taken). Unported and unused by MATLAB's defaults: `utils/gpHyperSVGD.m`, `utils/gpHyperSample.m` (not the counterpart of `_get_samples_from_slice_sampler_`) |
| **B6** GP model and its gpyreg objects | `gaussian_process_train.py`: `init_and_train_gp`, `_gp_hyp`, the prior and bound updates of `local_gp_fitting`, `_meanfun_name_to_mean_function`, `_cov_identifier_to_covariance_function`, the samplers of the priors; `stats/get_hpd.py` (from PyVBMC, read only by `_gp_hyp`); gpyreg's `RationalQuadraticARD`, `ConstantMean`, `NegativeQuadratic`, `GaussianNoise` and the `fit`/`update`/`predict` calls as PyBADS makes them | `gpdef/gpdefBads.m`, `gpml_fast/covRQard_fast.m`, `gpml_fast/infPrior_fast.m`, `gpml_fast/infExact_fastrobust.m`, `gpml_fast/sq_dist_fast.m`, `utils/likGaussHe.m`, `utils/minimizebnd.m`, `utils/mygp.m`, `utils/gppred.m`, `utils/gppriorrnd.m`, `utils/gpset.m`, `utils/prctile1.m`; GPML's `priorGauss`, `meanConst`, `sq_dist` and `solve_chol` as the reference for the priors, the mean and the factorization (out of scope as library code). Unused by MATLAB's defaults: `gpdef/private/gpdefStationaryNew.m`, `gpml_fast/exact_inference_*.m`, `gpml_fast/infExact_fast.m`, `utils/private/fminbayes.m`, the other kernels of `gpml_fast/` |
| **B7** Function logger, initial design, utilities | `function_logger/function_logger.py`, `init_functions/init_sobol.py`, `utils/period_check.py` | `private/funlogger.m`, `init/initSobol.m`, `init/private/i4_sobol*.m`, `init/private/i4_bit_hi1.m`, `init/private/i4_bit_lo0.m`, `utils/periodCheck.m`. Unported: `init/initLHS.m` (also `initSobol`'s fallback), `init/initRand.m` |
| **M** MATLAB changes since the port began (comparison track) | whichever PyBADS code corresponds | the eight commits above, one by one |
| **S** Sto-BADS (internal track only) | `bads.py`: `_sto_success_improvement_` and the `stobads` and `opp_stobads` branches of `_search_step_` and `_poll_step_` | none: Sto-BADS is PyBADS's own. The specification is the options' descriptions, the docstrings and the Sto-MADS rule it adapts (Audet, Dzahini, Kokkolaras and Le Digabel, 2021) |
| **O** Third reader: improvement, acquisition and geometry | `_eval_improvement_`, `_sto_success_improvement_`, the final quantile selection and the historic improvement of `optimize`, `acq_fcn_lcb`, the Hedge reward of `search_hedge.py`, `poll_scale`, `len_scale` and `effective_radius` in `gaussian_process_train.py` and their use in `poll_mads_2n.py` and `es_search.py` | `bads.m` (`EvalImprovement`, the final selection), `acq/acqLCB.m`, `acq/acqPortfolio.m`, `private/gpupdate.m`, `poll/pollMADS2N.m`, `search/searchES.m` |

Out of scope as unused or non-numerical: `stats/kde1d.py` and
`stats/kldiv_mvn.py` (PyVBMC leftovers that no PyBADS code calls: candidates
for removal, not review), `bads/bads_dump.py`, `function_examples.py`,
`utils/timer/`, `__main__.py`, `decorators/` (imported, never applied),
the plotting code, `warp/` (unsupported on both sides); on the MATLAB side
`bads_examples.m`, `hetsphere.m`, `rosenbrocks.m`, `install.m`,
`private/checklist.m`, `private/runtest.m` (the source of the tests'
problems) and `gpml_fast/test_*.m`. The slice table follows the
corrections of the counterpart map
(`experiments/port_review_20260925/prep_report.md`).

That is 17 reviewer runs: seven two-track slices (14), M, S and O, plus the
preparatory agent, and verification agents as findings accumulate.

## Waves

| Wave | Agents | Slices |
| --- | --- | --- |
| 0 | 3 | the preparatory agent; S; then M, once the sheet exists |
| 1 | 4 | B5 and B6, both tracks: the GP, where the survey's defects cluster |
| 2 | 4 | B2 and B1, both tracks |
| 3 | 4 | B3 and B4, both tracks |
| 4 | 3 | B7, both tracks; O |

Each wave stops for the PI's triage. Fix passes may be batched over two
waves into one pull request.

## Preparatory agent: the known-differences sheet and the counterpart map

One agent, run before any comparison-track reviewer, writes the sheet and
the map. Its sources are `AGENTS.md`, `CHANGELOG.md`,
`pybads/bads/README.md`, `dev/plans/gp-update-guards.md`,
`dev/plans/tooling-and-rng.md`, the entries of the survey's candidate table
marked fixed, "by design" or "tested, not adopted", the READMEs under
`dev/experiments/`, the "Missing port", "TODO" and "Matlab" comments in the
package, and the documentation under `docsrc/`. An entry is a *settled*
deliberate difference from MATLAB, in this form: Python location; MATLAB
location; what differs; why, with the record that decided it; kind
(deliberate change, unported feature, removed feature, substituted
library). Open findings and undecided questions, the survey's open rows
among them, do not go on the sheet. Every entry is a claim a reviewer may
challenge.

The same agent builds the counterpart map mechanically: every `*.m` file of
`../bads` outside `gpml-matlab-v3.6-2015-07-07/` searched for across
`pybads/` and the documents above, reconciled against the slice table.
Mismatches (a MATLAB file with a counterpart in another slice, a Python
file with no slice, a counterpart the table misses) go into its report and
are fixed in the table before the comparison-track waves start.

## The survey and the reviewers

Reviewers do not see the survey, so that their findings are independent of
it. Verification matches every finding to the survey's rows ("also in the
survey" in the ledger), and the ledger closes every open row of the
candidate table: a row that no reviewer reached is verified on its own in
the wave of its slice.

## Reviewer brief

Every reviewer is a fresh general-purpose agent, never a fork. It does not
open any file under `dev/` except the sheet. `CLAUDE.md` and `AGENTS.md` are
loaded into every agent by the harness, so the brief says that `AGENTS.md`
describes intended behavior, is not the specification, and that its
pointers into `dev/` are not to be followed. Each reviewer receives:

- the slice: the Python files and, on the comparison track, the MATLAB
  files, with the comparison revision and how to read a file's history
  (`git log -L<start>,<end>:<path>` in `../bads`);
- the track: *internal correctness* (does the code do what its docstrings,
  the option descriptions, the BADS paper (Acerbi and Ma, 2017, NeurIPS) and
  the mathematics require) or *MATLAB comparison* (line by line, does the
  Python do what the MATLAB does, and where not, is the difference on the
  sheet);
- how PyBADS uses the slice's code at default options, so that every
  finding says whether a default run reaches it;
- the sheet (comparison track and O);
- the categories and the finding format below;
- the terminology rule;
- the working rules: tracked files in every repository are read-only;
  scripts and outputs go only into the scratchpad directory given in the
  prompt; small checks as in "Decisions"; tests may be read to judge
  whether they would catch an error, but a test is not the specification;
- the interpreter: `.venv/Scripts/python.exe` of this repository, with
  `PYTHONPATH` naming the review worktree and the gpyreg clone, every
  script printing `pybads.__file__` and `gpyreg.__file__` once, and
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1.

The reviewer returns its report as its final message; the orchestrator
saves that text verbatim. The report has three parts: **coverage** (what was
read completely, what was skimmed, what was not reached), **findings** in
the format below, and **test adequacy notes**. A reviewer with no findings
says so; the coverage section is then the deliverable.

Finding format:

```
### F<n>. <one-line title>
- Location: <python path>:<line>; MATLAB: <path>:<line> or "no counterpart"
- Category: formula | indexing/shape | defaults | control flow |
  random draws | state/caching | cross-module
- Proposed classification: port discrepancy | suspected defect in both |
  possibly intentional | unsure
- Confidence: high | medium | low
- Reached at default options: yes | no (which option or input reaches it)
- History (comparison track): did the MATLAB lines change after
  2022-02-11? Cite the commit if so.
- What the code does, what it should do, and why (derivation, paper
  equation, or the MATLAB lines).
- Consequence if real: effect on results, when it triggers, how large.
- Suggested reproduction: the smallest check that would settle it.
- Test adequacy: would an existing test have caught it? Which one?
```

Reviewers do not propose or make fixes.

## Verification and the ledger

Every finding is verified before it enters the ledger, by a small script or
a second reading of the MATLAB source by a different agent, kept under
`experiments/port_review_20260925/verification/`. The classifications:

- **confirmed port discrepancy** (PyBADS differs from MATLAB and MATLAB is
  right, or the difference is unintended);
- **confirmed shared defect** (both are wrong);
- **intentional difference** (missing from the sheet, which is then
  updated);
- **listed as intentional, but the justification does not hold**;
- **needs MATLAB**;
- **not a defect**.

The per-wave ledger (`verification/wave<N>.md`) records, per finding: an
identifier (`W<N>-<k>`), the slice, both locations, the category, the
classification, the dating from both histories, the evidence, the survey
row if any, the PI's disposition and the fix commit. Durable entries of the
sheet are consolidated at the close into `pybads/bads/README.md`, which
becomes the catalogue of deliberate differences. Defects found on the
MATLAB side are collected in
`experiments/port_review_20260925/matlab_side_defects.md`.

## Fixes and gates

Fixes wait for the PI's triage. They go on the branch of the wave,
`dev-port-review-w<N>`, cut from `dev-next`, one commit per finding, each
with a regression test that would have caught it, and reach `dev-next`
through a pull request. Changes to gpyreg go to gpyreg on its own branch
and pull request.

The gates are those of `AGENTS.md`, "Numerical gates":

- A fix that must move nothing shows the same hash of
  `dev/scripts/fingerprint.py` before and after, on one machine, with the
  same gpyreg clone.
- A fix that moves results is gated by the population comparison against
  the reference of its platform (`dev/README.md`), with a configuration that
  reaches the changed code when it lies behind a non-default option.
- A fix to the guarded GP updates also runs
  `dev/scripts/gp_update_failures.py --inject`.
- A fix that changes a seeded test's tolerance or seed measures the errors
  over the seeds with `dev/scripts/tolerance_sweep.py` first.
- The suite, and a changelog line for every change a user of the last
  release can notice, checked against the tag of that release (a defect that
  an earlier commit of the pass brought in never reached a user).
- After a fix pass, the full CI matrix on the pull request into `dev-next`.

## Working rules

- At most four agents at a time; the orchestrator holds the heavy-compute
  slot.
- Reviewers read the review worktree, which the orchestrator moves only
  between waves.
- An agent that changes code works in its own git worktree, commits one
  finding at a time and does not push; the orchestrator reviews each diff
  and cherry-picks it onto the wave's branch. Scripts under `dev/scripts/`
  that import the benchmark put their own checkout first on `sys.path`
  (`AGENTS.md`); the others import PyBADS from `PYTHONPATH`, so every
  command in an agent's brief names the worktree there and prints
  `pybads.__file__`.
- The review proceeds one wave at a time. After every wave the orchestrator
  reports the wave's findings to the PI, who decides what follows.
- After every wave: `git status --porcelain --ignored` in this repository,
  the review worktree, `../bads` and the gpyreg clone, ignoring `.venv/` and
  `docs/`; anything a reviewer left is removed. Reports are saved as
  `experiments/port_review_20260925/reviews/<slice>_<track>.md`, the
  reports of fix agents under `fixes/`.
- An agent's report exists only as its final message. A long one is kept
  by the harness in the session's `tool-results` directory; a short one is
  taken from the agent's transcript (`subagents/agent-<id>.jsonl`), as the
  last string of the agent's own output that holds the report's title.
  PyVBMC's `dev/experiments/port_review_20260919/extract_report.py` does
  that.
- A ruling that leaves an item to a later slice says whether that slice
  still has a pass ahead; an item left to a slice already taken gets its own
  line in `TODO.md`, or is fixed.
- The heavy gates run one after the other, unbuffered and straight into a
  log file, and each leaves a record that is checked before the next
  statement about it is written.

## Wave 1 pickup

For a session that runs wave 1 away from the orchestrator's machine, in a
cloud sandbox (PI, 2026-09-26), while W0-1 is investigated there.
Everything it needs is in this repository and in two public ones; nothing
of `dev/scripts/runs/` (machine-local) is needed. It runs the reviews and
their verification of slices B5 and B6, writes the ledger, and stops for
the PI's triage: no fix, no population, no change to the package, no pull
request.

1. **Setup** (Linux; the paths are examples):

   ```console
   git clone https://github.com/acerbilab/pybads && cd pybads
   git switch dev-port-review && git switch -c dev-port-review-w1
   git worktree add --detach ../pybads-review 95da7f1
   git clone https://github.com/acerbilab/bads ../bads
   git -C ../bads checkout 74919c0
   git clone https://github.com/acerbilab/gpyreg ../gpyreg-v1.3.3
   git -C ../gpyreg-v1.3.3 checkout v1.3.3
   python -m venv .venv && .venv/bin/pip install -e ".[dev]"
   PYTHONPATH=../pybads-review:../gpyreg-v1.3.3 .venv/bin/python -c "import pybads, gpyreg; print(pybads.__file__, gpyreg.__file__)"
   ```

   The last line prints the review worktree's `pybads` and the clone's
   `gpyreg`; the reviewers' scripts select both the same way.
2. **The reviewers.** Four fresh general-purpose Opus agents at once, never
   forks: B5 internal, B5 comparison, B6 internal, B6 comparison. Each
   prompt is the slice part (`experiments/port_review_20260925/briefs/wave1_B5.md`
   or `wave1_B6.md`), then `briefs/wave1_common.md` from "You are a
   reviewer" to its first rule, then the part of the track, with the
   placeholders of `wave1_common.md` replaced; each reviewer has a scratch
   directory of its own outside the repositories.
3. **Saving.** Each report is saved verbatim as
   `experiments/port_review_20260925/reviews/<slice>_<track>.md`, under a
   header comment that says what it read and when, with
   `experiments/port_review_20260925/extract_report.py` from the agent's
   transcript (see "Working rules"). The sandbox is not kept, so each
   reviewer's scratch directory is copied into
   `experiments/port_review_20260925/verification/scripts/wave1/<slice>_<track>/`.
4. **Verification.** Once both reports of a slice are saved, one fresh Opus
   verifier for the slice (`briefs/wave1_verifier.md`), which did not write
   either report; its report saved as `verification/wave1_<slice>_verifier.md`,
   its scripts beside the reviewers'. The verifier of a slice also receives,
   quoted in its prompt, the open rows (status "seen", "not looked at" or
   "seen (MATLAB side read)") of the candidate table of
   `results/2026-09-23-codebase-survey.md` that belong to the slice and that
   neither report covers, and verifies them as findings.
5. **The ledger.** `verification/wave1.md`, rows W1-1 onwards, in the form
   of `verification/wave0.md`: source, what, classification, default run,
   dating, survey row, proposed disposition, gate; then "Found while
   verifying". Two sources of the orchestrator are not given to reviewers or
   verifiers and are only checked against the reports: the list "Seen in
   passing" of `prep_report.md` (whether a reviewer found an item of B5 or
   B6 by itself), and the rows of `verification/wave0.md` that wave 1's fix
   pass takes, W0-7 (the starting GP mean, B6) and W0-8 (the stable sort of
   the training set, B5): a finding that repeats one is marked "also W0-7"
   or "also W0-8".
6. **Close.** `git status --porcelain --ignored` in every checkout, and
   nothing a reviewer left; a line in the worklog below ("wave 1 run and
   verified, cloud session"); commit on `dev-port-review-w1`, push, and
   report to the PI. The orchestrator merges the branch into
   `dev-port-review` after the PI's triage.

## Wave 2 pickup

For the session that starts wave 2, slices B1 and B2 on both tracks, on
the orchestrator's machine or in a cloud sandbox as wave 1 ran. Everything
it needs is in this repository at `dev-next` and in the public repositories
of MATLAB BADS and gpyreg. The one record of the review that only the
orchestrator's machine holds is the check scripts of wave 0's agents
(`dev/scripts/runs/LOCAL.md`); no brief hands them to an agent, and wave
0's reports and ledger are tracked.

1. **The revision under review**, the PI's decision at the kickoff: the
   freeze `95da7f1`, or `fef6c14`, `dev-next` after the fix passes of waves
   0 and 1, which add 425 lines to the package and remove 240, most of
   them in `bads.py` and `gaussian_process_train.py` (B2's
   `_re_evaluate_history_` among them, W0-1). A new revision moves the
   review worktree, the table "Reference revisions" and the briefs; the
   sheet's citations are carried to it with `refresh_citations.py --base
   95da7f1`, and the entries that wave 1's rulings added are read against
   it.
2. **Setup**: step 1 of "Wave 1 pickup", with
   `git switch dev-next && git switch -c dev-port-review-w2` and the review
   worktree at the revision of step 1.
3. **The briefs**, `briefs/wave2_*.md`, made from wave 1's:
   `wave1_common.md` and `wave1_verifier.md` with the revision, and a slice
   part for B1 and one for B2 in the form of `wave1_B5.md` (the files of the
   slice table, how a default run reaches them, the first questions).
4. **Kept from the reviewers**, for the verifiers or to check the reports
   against: the open rows of the survey's candidate table in B1 and B2, the
   items of `prep_report.md`'s "Seen in passing" that belong to them, and
   what waves 0 and 1 left to these slices (the sections "Found while
   verifying" of `verification/wave0.md` and `verification/wave1.md`).
5. **Then** steps 2 to 6 of "Wave 1 pickup", with wave 2's names: the
   reports `reviews/B1_<track>.md` and `reviews/B2_<track>.md`, the scripts
   under `verification/scripts/wave2/`, the verifiers' reports
   `verification/wave2_<slice>_verifier.md`, the ledger
   `verification/wave2.md` from W2-1, and the branch `dev-port-review-w2`,
   pushed at the close for the PI's triage.
6. **The gates of the fix pass.** On Linux the reference is
   `experiments/population_linux_wave1_20260926/`, which pairs by seed only
   in its environment (Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1), where
   `dev/scripts/fingerprint.py` at `fef6c14` with gpyreg 1.3.3 prints
   `91f947f78e1087c2`; a box that prints another hash computes differently
   and needs those versions or a reference of its own. On Windows the
   reference, `population_gpfixes_20260925` at `ab4dded`, predates waves 0
   and 1, and a gate there first needs a new reference at `fef6c14`; the
   fingerprint at `fef6c14` on the orchestrator's machine is
   `3411ef0625d24b22`.

## Worklog

- [x] 2026-09-25: design discussed and decided with the PI (decisions
  above): the slices and waves, the target v1.1.3 (`../bads` fast-forwarded
  from v1.1.2 to `74919c0`), short capped `optimize()` runs as small checks,
  the freeze of `dev-next` after the session's two pull requests, wave 0
  allowed to start before it.
- [x] 2026-09-25 and 09-26: wave 0 run. The preparatory agent: the sheet
  (`known_differences.md`, 30 entries and 8 claims of the records that did
  not check out), the counterpart map (`counterpart_map.md`, the 114 MATLAB
  files outside GPML) and its report (`prep_report.md`, the corrections of
  the slice table, applied above, and seven differences seen in passing,
  kept from the reviewers as a check of the later waves). S internal (7
  findings) and M comparison (10 findings), fresh Opus reviewers by the
  brief, reading `ab4dded` in `../pybads-review`; M after the sheet. The
  reports are saved verbatim under `reviews/`, the check scripts on the
  orchestrator's machine (`dev/scripts/runs/LOCAL.md`). The sweep after the
  wave was clean. Three of M's findings (the final `fsd`, the history slot
  of the final estimate, the iteration count) are fixed in `7b50a3a` (#71), made
  before the reports from the survey's rows. Nothing is verified yet.
- [x] 2026-09-26: wave 0 verified (PI: go). Two fresh read-only Opus
  verifiers, one for S and one for M and the claims C1 to C8, each checking
  at `95da7f1` and `ab4dded`; their reports under `verification/`
  (`wave0_S_verifier.md`, `wave0_M_verifier.md`), the ledger
  `verification/wave0.md`, rows W0-1 to W0-21. Every finding holds as
  reported but for corrections of detail (the NaN state of S-F2 is shorter
  than reported; S-F6 and S-F7 have MATLAB counterparts; M-F5 is not a
  defect of MATLAB at `74919c0`). Three rows were fixed by #71 before the
  wave; two are design questions of Sto-BADS; the rest await the PI's
  triage. The verifiers found two of the differences kept from the
  reviewers (`search_factor_min`; a budget below the initial design),
  left to their slices' waves.
- [x] 2026-09-26: wave 0 triaged (PI; the rulings in
  `verification/wave0.md`) and fixed: the fix pass on `dev-port-review`,
  one commit per row, the fingerprint unchanged at each, the suite passing,
  and the benchmark unchanged (`verification/wave0.md`, "Fixes"); W0-7 and
  W0-8 go to the GP fix pass of wave 1; the design of Sto-BADS's rule
  (W0-12, W0-13) is a `TODO.md` item. W0-1, a commit of its own
  (`d6e3f61`, local branch `w0-1-investigation`), flagged the number of
  evaluations of three noisy configurations and raised the median error
  of `ellipsoid_D3_homo`, not significantly; under investigation on the
  orchestrator's machine (PI): which of its parts moves the runs, and the
  error at an equal number of evaluations. Wave 1 starts meanwhile in a
  cloud session (PI), from "Wave 1 pickup" below.
- [x] 2026-09-26: W0-1 investigated and ruled (`w01_investigation/`): on
  `ellipsoid_D3_homo` over 90 seeds, no significant change of the error,
  with or without the stall criterion, and 14 to 18% fewer evaluations,
  most of them from the removal of the drift; the rise over seeds 0-29 was
  not in seeds 30-89. PI: W0-1 stays, in a pull request of its own.
- [x] 2026-09-26: wave 1 run and verified, cloud session (from "Wave 1
  pickup", on `dev-port-review-w1`). Four fresh Opus reviewers, B5 and B6 on
  both tracks, reading `95da7f1` in `../pybads-review` (B5 internal 14
  findings, B5 comparison 12, B6 internal 10, B6 comparison 7), then one
  fresh Opus verifier per slice, which also verified the survey's open rows
  of its slice that neither report covered (B5-R1 to B5-R3, B6-R1). The
  reports and the verifications are saved verbatim with `extract_report.py`,
  the scripts of all six agents under `verification/scripts/wave1/`,
  formatted by the pre-commit hooks. The ledger `verification/wave1.md`,
  rows W1-1 to W1-34, closes the 13 open survey rows of B5 and B6. The
  reviewers found both differences seen in passing that belong to these
  slices, (d) and (f); W0-7 and W0-8 recur, not as findings. The sweep after
  the wave was clean (only `__pycache__`, removed). The import check of step
  1 must run outside the repository's root, whose `pybads/` comes first on
  `sys.path`; the reviewers' scripts ran from their scratch directories.
- [x] 2026-09-26: wave 1 triaged (PI; the rulings in
  `verification/wave1.md`). The orchestrator's proposals accepted, with the
  PI's amendments: W1-25 is measured behind a switch in gpyreg that is off
  by default, W1-27 (the GP fit at initialization) is kept, and W1-8 and
  W1-17, which MATLAB shares, are fixed; W1-6, W1-28 and W1-34 as proposed.
  The fix pass is not started.
- [x] 2026-09-26: the records of wave 1 rebased onto `dev-next` at
  `e004c79` (wave 0 and W0-1; `dev-port-review` superseded), and step 0 of
  wave 1's fix pass, its baseline on Linux: the reference
  `experiments/population_linux_wave0_20260926/`, which flags W0-1's
  evaluations as on Windows, and the fingerprint `bfbc6d6737e99d88` at
  `ac3dfed` (`verification/wave1.md`, "Fix pass").
- [x] 2026-09-26: wave 1's fix pass (`verification/wave1.md`, "Fix pass").
  Every row ruled for a fix is committed on `dev-port-review-w1`, by fix
  agents in worktrees of their own, reviewed and cherry-picked, with W0-7
  and W0-8, and W1-35, found while fixing: W0-1's re-estimate crashed
  every noisy run whose rebuild failed (PI: a failed iterate drops out of
  the choices, as MATLAB, and the incumbent keeps its estimate). The rows
  that move nothing kept the fingerprint; the three batches, W1-2 and W1-1
  compare without a worsening flag, the two flags, lower errors, traced to
  W1-23 and W1-4 by their steps, and W1-17 on a new 1-D suite. The net
  change against the baseline flags only `ackley_D6`, a lower error; the
  pass ends in the Linux reference
  `experiments/population_linux_wave1_20260926/`. The gpyreg side is
  merged (acerbilab/gpyreg#56, the switch, and #57, W1-24), with no
  release for now (PI). W1-25's switch, measured, stays off in PyBADS, to
  be revisited after all the fixes (PI; `TODO.md`). One pull request into
  `dev-next` carries the records and the fixes.
- [x] 2026-09-26: wave 1 doublechecked after its merge (PI): four fresh
  read-only Opus reviewers, of the fixes in `bads.py`, those in
  `gaussian_process_train.py`, the records, and gpyreg's side, reporting
  only substantial mistakes. Every fix implements its ruling. Found: with
  gpyreg's switch on, a fit's second optimization could start from a design
  point whose factorization had failed, and end the fit (acerbilab/gpyreg#58,
  bit-identical with the switch off; W1-25's measurement predates it); the
  changelog's W1-17 entry named the search's steps, which the length scale
  never set; `AGENTS.md` still had the re-estimate read every stored GP,
  which W0-1 changed; and `TODO.md` still listed follow-ups of the
  GP-update guards that W0-1, W1-12, W1-13 and W1-20 fixed. The survey's
  `_re_evaluate_history_` row, fixed by W0-1 and W1-35 and still open, is
  wave 2's W2-40.
- [x] 2026-09-26: the freeze. `dev-next` at `95da7f1`, after #70 (the
  Windows reference, `2210046`) and #71 (the small defects of the noise
  options, the final estimate, the iteration count, `output_fcn` and
  `max_fun_evals=1`); `dev-port-review` cut from it; the review worktree
  `../pybads-review` moved to it. The sheet's Python line citations are at
  `ab4dded`; before a comparison reviewer of waves 1 to 4 receives the
  sheet, its citations, entry KD-B1-8 and the claims C1 and C2 are checked
  against `95da7f1`, which #71 changed in `bads.py`,
  `gaussian_process_train.py` and `optimize_result.py`.
- [x] 2026-09-26: the sheet carried to `95da7f1`. `refresh_citations.py`
  (PyVBMC's, with a fixed base commit and the bare backticked line numbers
  that follow a path on its line) moved 60 citations and found none whose
  line #71 changed; the 15 citations of a file without its `pybads/` path
  were mapped by the same diff (8 moved), and a sample of the moves read
  against both revisions. KD-B1-8 no longer leaves the count in
  `iterations` open. C1 and C2 still hold at `95da7f1`; their verification
  is under way.
- [x] 2026-09-26: wave 1 merged: #74 squash-merged into `dev-next` as
  `fef6c14`, on W0-1 (#73, `e004c79`); gpyreg's side (acerbilab/gpyreg#56
  and #57) merged into gpyreg's `main` (`33e3165`) without a release, both
  bit-identical by default, with gpyreg 1.3.3 still PyBADS's minimum and
  CI pin. On the orchestrator's machine at `fef6c14`, the suite passes (273
  tests) and the fingerprint with the gpyreg 1.3.3 clone is
  `3411ef0625d24b22`. The review worktree stays at `95da7f1` until wave 2's
  kickoff ("Wave 2 pickup").
- [ ] Waves 2 to 4.
- [ ] Close: the consolidated ledger, the catalogue in
  `pybads/bads/README.md`, the survey's rows closed, `TODO.md`.
