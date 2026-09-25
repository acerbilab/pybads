# Plan: guards on the GP updates that can raise

Created: 2026-09-25
Status: DRAFT, awaiting the user's review of the Open Questions

## Summary

Three GP calls inside a run are not guarded against gpyreg's `LinAlgError`
("Singular matrix for L Cholesky decomposition"). Each one stopped a run
of a benchmark population (`dev/TODO.md`, first item; the survey,
"Crashes on unguarded GP updates"). This plan guards them the way MATLAB
BADS (`acerbilab/bads` at `74919c0`, v1.1.3) guards its own calls. It keeps
one invariant: the GP handed back to the loop always has posteriors that
match its training data. When a local GP could not be built, the next
search or poll step rebuilds it. Runs that do not reach a failed update
must not change, bit for bit. The fingerprint and a population comparison
on one machine check that.

## The three calls (at `09996b5`)

1. **`_get_target_from_gp_`** (`bads.py:2419`, the `set_hyperparameters`
   at `:2445`). It is reached from the search (`:1604`) and the poll
   (`:2049`) when `uncertainty_handling_level > 0` or
   `uncertain_incumbent` is set. It sets the best iteration's
   hyperparameters on a deep copy of the GP and predicts at the incumbent.
   A failed `set_hyperparameters` stopped `ellipsoid_D3` seed 20 (baseline
   population, gpyreg 1.3.1).
2. **`add_and_update_gp`** (`gaussian_process_train.py:1185`). It assigns
   the grown `gp.X` and `gp.y` directly (`:1207-1210`), then calls
   `gp.update(compute_posterior=True)` (`:1212`), which recomputes every
   posterior in full with the hyperparameters the GP already holds. It is
   reached after each search evaluation (`bads.py:1699`) and, in noisy
   runs, after each poll evaluation (`:2131`). It stopped `ellipsoid_D10`
   seed 7 (baseline population).
3. **The recovery in `local_gp_fitting`** (`gaussian_process_train.py:232`).
   The function replaces `gp.X`, `gp.y` and `gp.s2` with the nearest
   neighbours of the centre (`:253-258`), sets new priors, refits when
   `refit_flag` is set, and calls `gp.update(hyp=hyp_gp)` (`:506`). If that
   raises, it restores the old priors and calls
   `gp.set_hyperparameters(old_hyp_gp)` on the new data (`:512-513`), with
   no guard. Without a refit, `hyp_gp` is `old_hyp_gp`, so the recovery
   repeats the computation that just failed. It is deterministic, so it
   fails again, and every failure on that path stops the run. It stopped
   `ellipsoid_D10` seeds 13 and 26 (generator population, gpyreg 1.3.1).
   It is reached from the search (`bads.py:1587`, and `:1715` in noisy
   runs), from the poll (`:2034`), and from `_re_evaluate_history_`
   (`:2573`) in noisy runs.

## What MATLAB BADS does

Read from `bads.m`, `private/gpupdate.m` and `utils/gppred.m` at
`74919c0`:

- `gpupdate` computes the posterior inside a `try`. If it fails, the GP
  keeps its new data and hyperparameters, `gpstruct.post = []`, and
  `exitflag = -2`. For `'add'`, it first tries a rank-1 update
  (`update_posterior`), then the full computation, each inside a `try`. The
  point is added to `x` and `y` whatever happens. The callers of `'add'`
  ignore the exit flag.
- The search (`bads.m:522-536`) and the poll (`:825-838`) call `gpupdate`
  again whenever `isempty(gpstruct.post)`. A GP whose posterior failed is
  therefore rebuilt at the next search or poll step, and at each step after
  that until a rebuild succeeds.
- `gppred` computes each hyperparameter sample's prediction inside a
  `try`, and a failed sample stays NaN. With an empty `post`, it
  recomputes from the data.
- `UpdateTarget` (`bads.m:1296`) replaces a non-finite prediction by the
  incumbent's `fval` and `fsd`. It still computes
  `ftarget = ftargetmu - sdlevel*sqrt(ftargets2 + TolFun^2)` from the
  failed `ftargets2`, so the target is NaN. The poll then finds a
  non-finite `gammaz`, sets `pless = 0` and marks the GP as unreliable
  (`unrelgp_flag = 1`).
- The recovery of `local_gp_fitting` (old hyperparameters on the new data)
  has no MATLAB counterpart.

gpyreg's side, from 1.3.3 on, the minimum that PyBADS requires: a `fit`,
`update` or `set_hyperparameters` that raises puts the GP back as it was
when the call started (data, bounds, priors and posteriors). `predict` on
a posterior computed in the same process does not raise `LinAlgError`
(its docstring names only posteriors pickled by gpyreg 1.3.1 or earlier).
If `alpha` is `None` (posteriors made with `compute_posterior=False`),
`predict` raises `TypeError`. If the posteriors come from other training
data, `predict` raises `ValueError` when the sizes differ, and **silently
predicts wrong values when they are equal**. That is common, because the
training set is often capped at `n_train_max`. A GP left with new data
beside old posteriors is therefore the one state that must never reach the
loop.

## Design

PyBADS cannot hold an "empty posterior" as MATLAB does without changing
every prediction site (`acq_fcn_lcb`, the hedge, the ES searches, the
noisy predictions, `_re_evaluate_history_`). Instead, each guarded call
leaves a consistent GP, and a marker on the GP stands in for MATLAB's
empty `post`:

- **Marker.** `gp.temporary_data["needs_rebuild"] = True` means the GP
  handed back is not the one last asked for (Open Question 3). The search
  and the poll add it to their rebuild conditions (`bads.py:1580-1585`,
  `:2033`). `local_gp_fitting` removes it whenever it leaves a posterior on
  the new training set. The marker is a bool, so it can be deep-copied
  with the GP (`AGENTS.md`, randomness).
- **Target** (call 1). Put `set_hyperparameters` and `predict` in one
  `try`. On `LinAlgError`, set the prediction and its variance to NaN,
  which triggers the existing fallback to the incumbent's `fval` and
  `fsd`. The target stays NaN, as in MATLAB (Open Question 2). The GP is a
  discarded deep copy, so no marker is needed.
- **Add** (call 2). Pass the point through
  `gp.update(X_new=..., y_new=..., s2_new=..., hyp=<held hyperparameters>)`
  instead of assigning `gp.X` and `gp.y` first. Passing `hyp` keeps the
  full recomputation that runs today; without it, gpyreg would take its
  rank-1 path and move results (Open Question 5). `s2_new` gets exactly
  what `:1209-1210` appends today (see "Out of scope"). On `LinAlgError`,
  gpyreg leaves the GP without the point, with the posteriors it had, and
  the GP is marked. The point stays in the function logger, so the next
  rebuild takes it (Open Question 1).
- **Local fit** (call 3). Snapshot the GP at entry: `X`, `y`, `s2`,
  `get_priors()`, the `posteriors` array and a shallow copy of
  `temporary_data`. Nothing in the function mutates these objects in
  place. The data are replaced, not modified. `set_hyperparameters` and
  `update` with `hyp` build a new `posteriors` array. `_robust_gp_fit_`
  fits a deep copy and only sets hyperparameters on `gp`. The failure path
  then becomes:
  - The first `update` fails, and the recovery succeeds: unchanged from
    today (exit flag -2, new data, old hyperparameters), and the marker is
    removed. Such runs complete today and must not move (Open
    Question 4).
  - The recovery also fails: restore the snapshot in place (the same
    object, so `_re_evaluate_history_`'s stored GPs stay consistent),
    return exit flag -2, and set the marker. The loop continues with the
    previous local GP. The display already reports "Train (failed)" when
    a refit fails.
- Only `np.linalg.LinAlgError` is caught, as at the existing guards. A
  `ValueError` from gpyreg's input checks is a bug and should stop the
  run.

The noisy search branch and `_re_evaluate_history_` need no change of
their own. They call `local_gp_fitting` and then predict from what it
returns, which is now always consistent.

## Scope

- **In scope**: the three guards, the marker and the two rebuild
  conditions; tests that inject failures; a count of how often each site
  fails over the benchmark suite; the fingerprint and population gates on
  this machine; the records (changelog, `AGENTS.md`, survey, TODO, this
  plan's worklog).
- **Out of scope** (each one recorded in the survey, not fixed here):
  - the rank-1 update of `'add'` (Open Question 5);
  - `S`, a standard deviation, stored in `gp.s2`, a variance: by
    `add_and_update_gp` (the survey's existing row) and also by
    `local_gp_fitting`, through `get_grid_search_neighbors` (`:1101-1103`),
    which is a new row. It only affects `specify_target_noise` and would
    change level-2 results;
  - `_robust_gp_fit_`'s bound inversion after five failed fits (survey,
    "Failed fits"), which is not reached at default options;
  - `_re_evaluate_history_` refitting the GPs stored in `IterationHistory`
    in place;
  - the `&` slip at `bads.py:1693-1697`.

## Conventions

- Branch `claude/cloud-box-next-steps-s7ithi` (this session's), from
  `dev-next` at `09996b5`. Conventional commits, ending with the
  `Co-Authored-By:` line and never a `Claude-Session:` trailer
  (`AGENTS.md`). The pre-commit hooks run on every commit.
- The interpreter is `.venv/bin/python`. Every evidence run sets
  `PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3` (a clone at the tag,
  `98ab5a4`), one BLAS thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` = 1) and prints `gpyreg.__file__`. The clones are
  listed in `dev/scripts/runs/LOCAL.md`.
- One heavy process at a time. Long runs log unbuffered to
  `dev/scripts/runs/<name>_<epoch>.log` and are read from the log.
- Package code is committed before any run whose records name the commit.
- If a check contradicts an assumption a step rests on, stop and report
  the mismatch before improvising.

## Phases

### Phase 0: environment and baselines

**Status**: [~] in progress (2026-09-25)

1. [x] Set up the venv (Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1). Install
   gpyreg editable from `../gpyreg` at `v1.3.3`, and install PyBADS
   editable with `[dev]`. Clone `dev/scripts/runs/gpyreg/v1.3.3`.
2. [x] Fingerprint at `09996b5` with the v1.3.3 clone: `e5f46bce200bfaa7`,
   the same in two processes. This is the Linux hash of this machine; the
   Windows hash, `57241c985a68c78b`, is not comparable.
3. [x] Suite with reruns off: 109 passed in 28 s.
4. [ ] Write `dev/scripts/runs/LOCAL.md`, listing the clones and their
   commands.
5. [ ] Pre-change population: `population.py run --suite default --seeds
   0-29 --workers 4 --out dev/scripts/runs/population/population_linux_pre_20260925`
   at the commit of this plan, which leaves the package code as at
   `09996b5`. Then run `summary`, the null check
   (`compare <it> --split`), and `compare
   dev/experiments/population_gpyreg133_20260924 <it>` to measure the
   Linux-to-Windows difference, which matters only for Open Question 6.

**Verification**: the population has 540 records, with any crash listed
by configuration, seed and traceback; there is a null-check verdict; the
cross-platform comparison has been read.

### Phase 1: failure-injection tests, failing first

**Status**: [ ] not started

New file `pybads/testing/bads/test_gp_update_failures.py`. The tests make
`gpyreg.GP.update` or `GP.set_hyperparameters` raise `LinAlgError` through
`monkeypatch`, with a counter so that only the chosen calls fail:

1. `add_and_update_gp` with a failing `update`: the call returns without
   raising. `gp.X`, `gp.y` and `gp.s2` equal their values before the call,
   `gp.posteriors` is the same object, and the marker is set. Without
   failure: the GP's data are the old data plus the point, and the
   posteriors equal those of today's assign-then-update code, which pins
   that the change is bit for bit (`np.array_equal` on `alpha` and `L`).
2. `local_gp_fitting` with both `update`s failing, with and without
   `refit_flag`: the exit flag is -2. The GP (compared against a deep copy
   taken at entry: data, priors, hyperparameters, posteriors'
   `alpha`/`L`, `temporary_data` apart from the marker) is as it was at
   entry, and the marker is set. Its predictions at the centre equal the
   copy's.
3. `local_gp_fitting` with only the first `update` failing: exit flag -2,
   new data, old hyperparameters, no marker (today's behaviour).
4. `_get_target_from_gp_` with a failing `set_hyperparameters`, at
   uncertainty levels 1 and 2: `f_target_mu` and `f_target_s` are the
   incumbent's `fval` and `fsd`, `f_target` is NaN, and the three values go
   through `.item()` at the call sites. `optim_state["fval"]` may be a
   Python float, so the fallback's types are part of the test.
5. End to end: seeded runs (a sphere, `D = 3`, 150 evaluations): a
   deterministic run where every 5th `add_and_update_gp` update fails; a
   noisy run (`uncertainty_handling=True`) where the recovery in
   `local_gp_fitting` fails at its first occurrence, which the test forces
   by also failing the first `update`; and a noisy run where the target
   prediction fails once. Each run finishes with a finite `fval`. After
   each injected failure, the next search or poll step calls
   `local_gp_fitting` (spied).
6. Run the new tests at `09996b5`. Every test with an injected failure
   raises `LinAlgError`, which is the positive control that the tests
   reach the three calls.

**Verification**: the tests fail at `09996b5` for the stated reason.
Tests 1 (no-failure half) and 3 pass there already.

### Phase 2: the guards

**Status**: [ ] not started

1. Call 1, call 2, call 3 and the marker, as in Design. The docstrings of
   `add_and_update_gp` and `local_gp_fitting` state what a failure leaves.
2. The two rebuild conditions in `bads.py`.
3. `gp.temporary_data.pop("needs_rebuild", None)` on each path of
   `local_gp_fitting` that leaves a posterior on its new training set.

**Verification**: the Phase 1 tests pass, and the whole suite passes with
reruns off. The fingerprint is `e5f46bce200bfaa7`, in two processes. If it
moves, stop: either one of the six runs hits a failure (the Phase 3 count
tells which), or the change is not bit for bit.

### Phase 3: reach and evidence

**Status**: [ ] not started

1. `dev/scripts/gp_update_failures.py`, after `gpyreg_issue_checks.py`.
   It runs a suite in-process, wraps `GP.update`, `GP.set_hyperparameters`
   and `GP.predict`, and records for each run the `LinAlgError`s by PyBADS
   call site (the innermost `pybads` frame), how each guard ended
   (recovered, restored, fallback) and the run's outcome, to one JSON
   file. It includes an `--inject P` option that makes a fraction `P` of
   the guarded calls fail, drawn from the script's own
   `numpy.random.Generator`, never from `bads.rng` or the global stream.
   One line in `dev/README.md`, Scripts.
2. Run it over the default suite, seeds 0-29, without injection, to learn
   how often each site fails under gpyreg 1.3.3 on this machine. That
   tells whether the population below reaches the change at all.
3. Post-change population, with the same command as in Phase 0 and the
   out name `population_linux_post_20260925`. Then run `compare pre post`.
   Expected: no flag, and every run the count shows without a failure
   identical to its pre-change record (x, fval, func_count). Any other
   difference is a finding.
4. Stress: `--inject 0.02` over the default suite, seeds 0-9. Every run
   finishes. Report the median log10 error beside the pre-change
   population's, as a description of how runs degrade under failures, not
   as a test.

**Verification**: counts per site, the comparison's verdict, identical
records where no failure occurred, and all injected runs finished.

### Phase 4: records

**Status**: [ ] not started

1. `CHANGELOG.md`, Unreleased, Fixed. **Failed GP updates.** A run no
   longer stops with `LinAlgError` ("Singular matrix for L Cholesky
   decomposition") when a Gaussian-process update fails while adding a
   point, predicting the optimization target, or rebuilding the local GP.
   Like MATLAB BADS, the run carries on with the previous GP and rebuilds
   it at the next step. Runs without such a failure give the same results.
   No Upgrading line: nothing that ran before stops or returns something
   different.
2. `AGENTS.md`, "What spans files": the invariant (never assign `gp.X` or
   `gp.y` before an update that can fail, and pass new data through
   `gp.update`; the one place that replaces the training set,
   `local_gp_fitting`, snapshots it), and `temporary_data["needs_rebuild"]`,
   set in `gaussian_process_train.py` and read in `bads.py`.
3. Survey: the "Crashes on unguarded GP updates" section gets the fix
   commit and the counts. New candidate rows: `S` in `gp.s2` in
   `local_gp_fitting`; the NaN target of `UpdateTarget` (MATLAB and port
   alike, Open Question 2); the recovery without a MATLAB counterpart; the
   dropped point versus MATLAB's kept point; full recomputation versus
   MATLAB's rank-1 update; `_re_evaluate_history_` mutating stored GPs.
4. `dev/TODO.md`: close the item, or cut it down to what Phase 3 leaves
   open.
5. The `dev/README.md` index; the Linux reference if Open Question 6
   stands; the Worklog; Status.

## Open questions (the default stands unless the user says otherwise)

1. **Failed add: drop the point, or keep it as MATLAB does?** Default:
   drop it. gpyreg's restore keeps the GP consistent, and the marker has
   the next step rebuild the GP from the function logger, which holds the
   point. MATLAB keeps the point beside an empty posterior, and its
   predictions go NaN until the rebuild. Keeping the point in PyBADS would
   mean NaN-tolerant predictions at every call site.
2. **Target after a failed prediction: NaN, as in MATLAB, or finite?**
   Default: NaN. The poll then treats the GP as unreliable (`p_less = 0`
   and `do_gp_calibration = True`), as MATLAB does. The alternative, a
   target from the incumbent's `fsd`, keeps the poll's probability of
   improvement usable, but departs from the reference. It is noted in the
   survey either way.
3. **Where the rebuild marker lives.** Default: on the GP,
   `temporary_data["needs_rebuild"]`. It travels with the GP through deep
   copies and the noisy branch's `new_gp`, as MATLAB's empty `post` does,
   and no signature changes. The alternative is a `BADS` attribute, plus
   an exit flag returned by `add_and_update_gp`. That function is exported
   from `pybads.bads`, so the second return value would change its
   signature.
4. **Keep the recovery (old hyperparameters on the new data)?** Default:
   keep it. Runs where it succeeds complete today and must not move. If it
   were dropped to follow MATLAB, results would change; that belongs to
   the bug hunt, with its own gate.
5. **Rank-1 update for `'add'`, as MATLAB does it first.** Default: out of
   scope. It would move results at default options and needs its own
   population comparison.
6. **Commit the Phase 0 population as a Linux reference**
   (`dev/experiments/population_linux_20260925/`, about 2 MB, with the
   README the convention asks for)? Default: yes. Later cloud sessions can
   then gate changes on this platform, where the Windows reference only
   compares across platforms.

## Worklog

- 2026-09-25: plan drafted; MATLAB BADS read at `74919c0`; gpyreg 1.3.3's
  `update`, `set_hyperparameters` and `predict` read at `98ab5a4`. Phase 0
  steps 1-3 done (above).
