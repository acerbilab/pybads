# Plan: guards on the GP updates that can raise

Created: 2026-09-25
Status: APPROVED (2026-09-25). Revised the same day after an independent
review; the user re-settled Open Question 2 and settled the new Open
Question 7. Phases 0-2 done.

## Summary

Three GP calls inside a run are not guarded against gpyreg's `LinAlgError`
("Singular matrix for L Cholesky decomposition"). Each one stopped a run
of a benchmark population (`dev/TODO.md`, first item; the survey,
"Crashes on unguarded GP updates"). This plan guards them after MATLAB
BADS (`acerbilab/bads` at `74919c0`, v1.1.3). It keeps one invariant: the
GP handed back to the loop always has posteriors that match its training
data. A local GP that could not be built is rebuilt at the next search or
poll step, and a failed rebuild forces a refit at the next one. Runs that
do not reach a failed update must not change, bit for bit. The
fingerprint and a population comparison on one machine check that.

## The three calls (at `09996b5`)

1. **`_get_target_from_gp_`** (`bads.py:2419`, the `set_hyperparameters`
   at `:2445`). It is reached from the search (`:1604`) and the poll
   (`:2049`) when `uncertainty_handling_level > 0` or
   `uncertain_incumbent` is set. `uncertain_incumbent` is `True` by
   default (`advanced_bads_options.ini:121`), so the call runs in
   deterministic runs too. It sets the best iteration's hyperparameters on
   a deep copy of the GP, which recomputes the posterior, and predicts at
   the incumbent. A failed `set_hyperparameters` stopped `ellipsoid_D3`
   seed 20, a deterministic run (baseline population, gpyreg 1.3.1). The
   fallback behind it (`:2449-2455`) has never run to completion. It
   assigns `optim_state["fval"]`, a Python float (stored with `.item()`,
   `:1047`, `:1728`, `:2141`), and both call sites then call
   `f_target_mu.item()` (`:1607`, `:2052`), which raises `AttributeError`.
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
   `gp.set_hyperparameters(old_hyp_gp)` on the new data (`:512-513`),
   without a guard. Without a refit, `hyp_gp` is `old_hyp_gp`, and priors
   do not enter the posterior computation. The recovery then repeats the
   computation that just failed, fails again deterministically, and stops
   the run. This call stopped `ellipsoid_D10` seeds 13 and 26 (generator
   population, gpyreg 1.3.1). It is reached from the search
   (`bads.py:1587`, and `:1715` in noisy runs), from the poll (`:2034`),
   and from `_re_evaluate_history_` (`:2573`) in noisy runs.

## What MATLAB BADS does

Read from `bads.m`, `private/gpupdate.m`, `utils/gppred.m` and
`utils/mygp.m` at `74919c0`:

- **`gpupdate`.** It computes the posterior inside a `try`. If that fails,
  the GP keeps its new data and hyperparameters, `gpstruct.post = []`, and
  `exitflag = -2`.
- **`'add'`.** It first tries a rank-1 update (`update_posterior`), except
  under `SpecifyTargetNoise`, then the full computation, each inside a
  `try`. The point is added to `x` and `y` whatever happens. The callers
  (`bads.m:634`, `:910`) ignore the exit flag.
- **Rebuild.** The search (`bads.m:522-536`) and the poll (`:825-838`) call
  `gpupdate` again whenever `isempty(gpstruct.post)`. A GP whose posterior
  failed is therefore rebuilt at the next step, and at each step after
  that until a rebuild succeeds. In the meantime, its NaN predictions go
  into the GP statistics (`savegpstats`) that the refit check reads
  (`gppredcheck`, `bads.m:1229`).
- **`gppred`.** It predicts for each hyperparameter sample inside a `try`,
  and a failed sample stays NaN. With an empty `post`, it recomputes from
  the data. With a `post`, it passes that posterior on, and `mygp` reuses
  it (`mygp.m:123`, `post = y`).
- **`UpdateTarget`** (`bads.m:1296`). It sets `gptemp.hyp = hyp` (the best
  iteration's) but keeps `gptemp.post`, so it predicts from the current
  posterior with `hyp` in the mean and covariance functions, without
  refactorizing. It cannot hit a Cholesky failure. A NaN prediction is
  possible only when `post` is empty after a failed rebuild. In that case
  the incumbent's `fval` and `fsd` replace the prediction, while `ftarget`
  is still computed from the failed `ftargets2`, so the target is NaN. The
  poll then sets `pless = 0` and `unrelgp_flag = 1`, and `acqLCB` ignores
  the target.
- **The recovery** of `local_gp_fitting` (old hyperparameters on the new
  data) has no MATLAB counterpart.

gpyreg's side, from 1.3.3 on, the minimum that PyBADS requires:

- **Restore.** A `fit`, `update` or `set_hyperparameters` that raises puts
  the GP back as it was when the call started: a shallow copy of its
  attributes plus its posterior entries. `temporary_data` comes back as
  the same object, and only `clean()` resets it, which neither PyBADS nor
  gpyreg's `fit` and `update` call.
- **`predict`** on a posterior computed in the same process does not raise
  `LinAlgError`. Its docstring names only posteriors pickled by gpyreg
  1.3.1 or earlier. It does fail on two states:
  - With `alpha` `None` (posteriors made with `compute_posterior=False`,
    as in the intermediate state of `local_gp_fitting`'s refit path), it
    raises `TypeError`.
  - With posteriors from other training data, it raises `ValueError` when
    the sizes differ. When they are equal it silently predicts wrong
    values, which is common because the training set is often capped at
    `n_train_max`. A snippet, 10 points: the stale posterior predicts 1.90
    where the consistent one gives 5.28.

  A GP left with new data beside old or missing posteriors is the state
  that must never reach the loop.

## Design

PyBADS cannot hold an "empty posterior" as MATLAB does without changing
every prediction site (`acq_fcn_lcb`, the hedge, the ES searches, the
noisy predictions, `_re_evaluate_history_`). Instead, each guarded call
leaves a consistent GP, and markers on the GP stand in for MATLAB's empty
`post`.

- **Markers.**
  - `gp.temporary_data["needs_rebuild"] = True`: the GP handed back is not
    the one last asked for.
  - `gp.temporary_data["needs_refit"] = True`: a rebuild of the local GP
    failed, so the next rebuild refits the hyperparameters (Open
    Question 7).

  Both are plain bools, safe to deep-copy with the GP (`AGENTS.md`,
  randomness), and they travel with it through the noisy search branch's
  `new_gp`.
  - **Search** (`bads.py:1581-1585`): adds `needs_rebuild` to its rebuild
    condition and ORs `needs_refit` into `refit_flag`.
  - **Poll** (`:2021-2033`): does the same. `needs_refit` does not
    override `poll_training`: with `poll_training` off, after the first
    iteration, the poll rebuilds without refitting.
  - **`local_gp_fitting`**: removes both whenever it leaves a posterior on
    its new training set.

  When `_is_gp_refit_time_` decides on a refit, it also records it: it
  sets `optim_state["lastfitgp"]` to the evaluation count and resets the
  GP statistics. That bookkeeping moves into a method of its own,
  `_record_gp_refit_`, which a forced refit calls as well.
- **Target** (call 1). Put `set_hyperparameters` and `predict` on the deep
  copy in one `try`. On `LinAlgError`, predict at the incumbent from `gp`
  itself: its own hyperparameters and consistent posterior, with no
  refactorization (Open Question 2). That keeps what MATLAB effectively
  does, a prediction from the current posterior. A prediction that is
  still not finite goes to the existing fallback. The fallback's
  `f_target_mu` becomes
  `np.atleast_2d(np.asarray(self.optim_state["fval"], dtype=float))`, so
  that the call sites' `.item()` works. That code runs only after a
  failure, so the no-failure path stays bit for bit. `f_target_s` does not
  go through `.item()`. No marker: the copy is discarded and `gp` is
  untouched.
- **Add** (call 2).
  - **The call.** Pass the point through
    `gp.update(X_new=np.atleast_2d(x_new), y_new=np.atleast_2d(y_new), s2_new=..., hyp=<held hyperparameters>)`
    instead of assigning `gp.X` and `gp.y` first. `y_new` has to be an
    array: a Python float raises `AttributeError` in gpyreg's
    `_convert_shapes`. Passing `hyp` keeps today's full recomputation; the
    rank-1 path is a TODO item.
  - **`s2_new`.** It gets what `:1209-1210` appends today (see "Out of
    scope"). Where `s2_new` is `None` and the GP holds an `s2` array,
    gpyreg appends zeros. At level 1 with an explicit
    `uncertainty_handling=True`, `noise_flag` is set and `gp.s2` holds
    NaN, since `S` is never filled there. Today that array is not
    extended; with the change it grows by a zero per add. The noise
    function ignores it (`gp_noisefun = [1, 2, 0]`): the review's snippet
    found `alpha`, `L`, `sW`, `sl` and the predictions bit-equal at levels
    0, 1 and 2, but the GP's `s2` attribute differs.
  - **On failure.** On `LinAlgError`, gpyreg leaves the GP without the
    point, with the posteriors it had, and `needs_rebuild` is set. The
    point stays in the function logger, so the next rebuild takes it (Open
    Question 1).
- **Noisy poll after a failed add** (`bads.py:2131-2141`). The GP does not
  contain `u_new`, so `f_poll` and `f_sd_poll` are set to NaN instead of
  predicted. That follows MATLAB, where `gppred` recomputes the failed
  posterior, fails again and returns NaN. The poll improvement is then NaN
  and the point counts as no improvement (`poll_improvement >
  poll_best_improvement` is false; `stobads` is off by default). The poll
  detects the failure by the GP's number of training points not having
  grown, not by the marker, which a failed rebuild earlier in the same
  step may already have set. The noisy search branch needs no change: it
  rebuilds `new_gp` from the function logger, point included.
- **Local fit** (call 3).
  - **Snapshot.** At entry, take `snap = vars(gp).copy()` and
    `td = dict(gp.temporary_data)`, as gpyreg's own restore does. A
    hand-picked list of attributes would depend on how `set_priors`
    resets gpyreg's prior caches, and on the slice sampler's call of
    `gp._GP__gp_obj_fun` on the live GP. Nothing in the function mutates
    an attribute's object in place, apart from `temporary_data` and the
    freshly assigned `gp.y` and `gp.s2`, whose earlier objects the
    snapshot holds. `set_hyperparameters`, and `update` with `hyp`, build
    a new `posteriors` array. `_robust_gp_fit_` fits a deep copy and only
    sets hyperparameters on `gp`.
  - **First `update` fails, recovery succeeds.** Unchanged from today
    (exit flag -2, new data, old hyperparameters), and the markers are
    removed. Such runs complete today and must not move (Open
    Question 4).
  - **Recovery fails too.** Restore in place with
    `vars(gp).clear(); vars(gp).update(snap)` and then
    `gp.temporary_data.clear(); gp.temporary_data.update(td)`. Using the
    same object keeps `_re_evaluate_history_`'s stored GPs, and
    `optimize()`, which ignores `_poll_step_`'s returned GP, consistent.
    Then return exit flag -2 and set both markers.
  - **Not restored, and benign:** `optim_state["ntrain"]` and
    `optim_state["second_fit"]`, the records the call writes into the
    iteration history, and the generator draws a failed refit consumed.
- **Scope of the catch.** Only `np.linalg.LinAlgError` is caught, as at
  the existing guards; `scipy.linalg.LinAlgError` is the same class. A
  `ValueError` from gpyreg's input checks is a bug and should stop the
  run.
- **Logging.** Each guard logs one `logging.debug` line. The display
  reports "Train (failed)" only after a refit (`bads.py:2289-2292`), so
  otherwise a failure would be silent.

## Scope

- **In scope**: the three guards, the markers and the rebuild and refit
  conditions; the noisy poll's NaN after a failed add; the fallback's
  type; tests that inject failures, one of them a real Cholesky failure;
  a count of how often each site fails over the benchmark suite; the
  fingerprint and population gates on this machine; the records
  (changelog, `AGENTS.md`, survey, TODO, this plan's worklog).
- **Out of scope** (each one recorded in the survey, not fixed here):
  - the rank-1 update of `'add'` (Open Question 5; a `dev/TODO.md` item);
  - the port refactorizing the target's posterior under the best
    iteration's hyperparameters, where MATLAB reuses the current posterior
    (`UpdateTarget`). This is why call 1 can fail at all, and it changes
    targets at default options;
  - `S`, a standard deviation, stored in `gp.s2`, a variance: by
    `add_and_update_gp` (the survey's existing row) and also by
    `local_gp_fitting`, through `get_grid_search_neighbors` (`:1101-1103`),
    a new row. Only `specify_target_noise` is affected, and fixing it would
    change level-2 results;
  - `_robust_gp_fit_`'s bound inversion after five failed fits (survey,
    "Failed fits"), which is not reached at default options;
  - `_re_evaluate_history_` refitting the GPs stored in `IterationHistory`
    in place. After a failed rebuild it also records the restored GP's
    `fval` and `fsd`, where MATLAB would record NaN;
  - the `&` slip at `bads.py:1693-1697`.

## Conventions

- Branch `claude/cloud-box-next-steps-s7ithi` (this session's), from
  `dev-next` at `09996b5`. Conventional commits, ending with the
  `Co-Authored-By:` line and never a `Claude-Session:` trailer
  (`AGENTS.md`). The pre-commit hooks run on every commit.
- The interpreter is `.venv/bin/python`. Every evidence run sets
  `PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3` (a clone at the tag,
  `98ab5a4`), one BLAS thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` = 1), and prints `gpyreg.__file__`. The clones are
  listed in `dev/scripts/runs/LOCAL.md`.
- One heavy process at a time. Long runs log unbuffered to
  `dev/scripts/runs/<name>_<epoch>.log` and are read from the log.
- Package code is committed before any run whose records name the commit,
  and nothing is committed while a population runs: each record names the
  commit checked out when its run started.
- If a check contradicts an assumption a step rests on, stop and report
  the mismatch rather than improvise.

## Phases

### Phase 0: environment and baselines

**Status**: [x] done (2026-09-25)

1. [x] Set up the venv (Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1). Install
   gpyreg editable from `../gpyreg` at `v1.3.3`, and install PyBADS
   editable with `[dev]`. Clone `dev/scripts/runs/gpyreg/v1.3.3`.
2. [x] Fingerprint at `09996b5` with the v1.3.3 clone: `e5f46bce200bfaa7`,
   the same in two processes. This is this machine's Linux hash; the
   Windows hash, `57241c985a68c78b`, is not comparable.
3. [x] Suite with reruns off: 109 passed in 28 s.
4. [x] `dev/scripts/runs/LOCAL.md` lists the clones and their commands.
5. [x] Pre-change population:
   `population.py run --suite default --seeds 0-29 --workers 4 --out dev/scripts/runs/population/population_linux_pre_20260925`
   at `500ff1b`, whose package code is that of `09996b5`. Then run
   `summary` and the null check (`compare <it> --split`). As information
   only, also run
   `compare dev/experiments/population_gpyreg133_20260924 <it>`. That
   comparison mixes the platform with the versions (Python 3.11.15, NumPy
   2.4.6, SciPy 1.17.1 here, against 3.12.6, 2.5.3 and 1.18.1), and
   pairing by seed means nothing across platforms.

**Verification**: 540 records, with any crash listed by configuration,
seed and traceback, and the null-check verdict.

### Phase 1: failure-injection tests, failing first

**Status**: [x] done (2026-09-25)

New file `pybads/testing/bads/test_gp_update_failures.py`.

**The injection.** A fixture wraps `gpyreg.GP.update` and
`GP.set_hyperparameters`. It raises `LinAlgError` only for calls whose
innermost `pybads` frame is one of `add_and_update_gp`,
`local_gp_fitting` or `_get_target_from_gp_`, that compute a posterior
(`compute_posterior=True`), and that the test selects with a per-site
counter. Several other callers go through `update`: `set_hyperparameters`
itself, `fit`'s internal updates, the initial and robust fits, and
`local_gp_fitting`'s `compute_posterior=False` calls. Those are never
hit. Only the outermost gpyreg call is counted, so that
`set_hyperparameters` → `update` counts once.

1. **`add_and_update_gp`, at levels 0, 1 and 2** (level 1 with an explicit
   `uncertainty_handling=True`, where `gp.s2` holds NaN).
   - **With a failure:** the call returns without raising. `gp.X`, `gp.y`
     and `gp.s2` equal their values before the call, `gp.posteriors` is
     the same object, and `needs_rebuild` is set.
   - **Without a failure:** `alpha`, `L`, `sW` and the predictions equal,
     with `np.array_equal`, those of today's assign-then-update code run
     on a deep copy. `gp.X` and `gp.y` are the old data plus the point.
     `gp.s2` is `None` at level 0, the old array plus `0` at level 1, and
     the old array plus the value `:1209-1210` appends today at level 2.
2. **Real Cholesky failure.** A GP with duplicated inputs and a noise
   log-scale of -60, which fails reliably in the review's snippet, goes
   through `add_and_update_gp`. The GP comes back as it was, which tests
   gpyreg's restore itself; the scheduled run against gpyreg `main` then
   catches a regression in it.
3. **`local_gp_fitting` with both `update`s failing**, with and without
   `refit_flag`.
   - The exit flag is -2.
   - The GP matches a deep copy taken at entry: data, priors,
     hyperparameters, the posteriors' `alpha` and `L`, and
     `temporary_data` apart from the markers. It is the same object.
   - Its predictions at the centre equal the copy's.
   - Both markers are set.
4. **`local_gp_fitting` with only the first `update` failing**: exit
   flag -2, new data, old hyperparameters, no markers (today's
   behaviour).
5. **`_get_target_from_gp_`, at levels 0 (`uncertain_incumbent`), 1 and
   2.**
   - With a failing `set_hyperparameters`, `f_target_mu` and `f_target`
     come from `gp.predict` at the incumbent under `gp`'s own
     hyperparameters.
   - With that prediction also made non-finite (patched), `f_target_mu`
     is the incumbent's `fval` as a `(1, 1)` array and `f_target_s` its
     `fsd`.
   - Both go through the call sites' `.item()`.
6. **The markers cause the rebuild.** Spy on
   `pybads.bads.bads.local_gp_fitting`, the name `bads.py` imports.
   - In a search step with `search_count > 0`, `reset_gp` false and no
     refit due, the call happens with `needs_rebuild` set and does not
     happen without it.
   - With `needs_refit` set, it happens with `refit_flag` true.
   - The same two checks in a poll step with `poll_count > 0`.
7. **Noisy poll after a failed add**: the poll improvement passed on is
   NaN, and the incumbent does not move to `u_new`.
8. **End to end** (a sphere, `D = 3`, 150 evaluations, seeded): a
   deterministic run where every 5th add fails; a noisy run
   (`uncertainty_handling=True`) with one double failure of
   `local_gp_fitting`; a noisy run where the target's
   `set_hyperparameters` fails once. Each run finishes with a finite
   `fval`.
9. **Positive control.** Run the tests at `09996b5`. Every test with an
   injected failure fails there: most by raising `LinAlgError`, and test
   5's fallback case with `AttributeError`.

**Verification**: the tests fail at `09996b5` for the stated reasons.
Test 1 without a failure (apart from its `s2` at level 1) and test 4 pass
there already.

### Phase 2: the guards

**Status**: [x] done (2026-09-25)

1. Call 1, call 2, call 3, the markers and the noisy poll's NaN, as in
   Design. The docstrings of `add_and_update_gp` and `local_gp_fitting`
   state what a failure leaves.
2. The rebuild and refit conditions in the search and the poll. Check
   what `_is_gp_refit_time_` records when it decides on a refit, and make
   the forced refit record the same.
3. The `CHANGELOG.md` entry, in this phase's commit (`AGENTS.md`),
   Unreleased, Fixed. **Failed GP updates.** A run no longer stops with
   `LinAlgError` ("Singular matrix for L Cholesky decomposition") when a
   Gaussian-process update fails while adding a point, predicting the
   optimization target or rebuilding the local GP. The run carries on, as
   in MATLAB BADS, and the GP is rebuilt at the next step. Runs without
   such a failure give the same results. No Upgrading line: nothing that
   ran before stops or returns something different.

**Verification**: the Phase 1 tests pass, and the whole suite passes with
reruns off. The fingerprint is `e5f46bce200bfaa7`, in two processes. Any
failure in the six runs other than a recovered one would have stopped
them at `09996b5`, which produced a hash, so a moved fingerprint means the
change is not bit for bit: stop and find where. The fingerprint does not
reach level 2, which test 1 covers.

### Phase 3: reach and evidence

**Status**: [ ] not started

1. **`dev/scripts/gp_update_failures.py`**, after `gpyreg_issue_checks.py`.
   - It builds each run as `population.py` does (targets, options,
     seeds), in-process, with the Phase 1 wrappers.
   - It records for each run, to one JSON file:
     - the `LinAlgError`s by site;
     - how each guard ended: recovered, restored, or the target's
       fallback;
     - the longest streak of consecutive failed rebuilds;
     - the evaluations spent with `needs_rebuild` set;
     - the final `x`, `fval` and `func_count`.
   - It checks those final values against the post-change population's
     records. That validates the mapping of runs, and shows that the
     wrappers change nothing.
   - Its `--inject P` option makes a fraction `P` of the guarded
     computations fail. The decision is drawn from the script's own
     `numpy.random.Generator`, never from `bads.rng` or the global stream,
     and memoized on the site and a hash of `X`, `y` and the
     hyperparameters. A retry of the same computation then fails again,
     as a real failure does, so the restore path is reached; independent
     draws would reach it only with probability `P**2`.
   - One line in `dev/README.md`, Scripts.
2. **Post-change population**: the Phase 0 command with the output name
   `population_linux_post_20260925`, at the Phase 2 commit.
   - **Identity**: a record diff, since `compare` tests distributions and
     not identity. Every run without a failure, or with only recovered
     ones (the Open Question 4 path), is identical in every `final` field
     except `wall_s`. A run with an unrecovered failure would have crashed
     before the change.
   - **Verdict**: `compare pre post` gives it.
3. **The failure count** over the default suite, seeds 0-29, without
   injection: how often each site fails under gpyreg 1.3.3 on this
   machine, and so how much of the change the population reaches.
4. **Stress**: `--inject 0.02` over the default suite, seeds 0-9. Every
   run finishes. Report the streaks, the evaluations spent stale, and the
   median log10 error beside the post-change population's, as a
   description of how runs degrade under failures, not as a test.

**Verification**: counts per site; identical records where the change is
not reached, and the comparison's verdict; the count's final values equal
to the population's; all injected runs finished.

### Phase 4: records

**Status**: [ ] not started

1. `AGENTS.md`, "What spans files":
   - the invariant: never assign `gp.X` or `gp.y` before an update that
     can fail; pass new data through `gp.update`; the one place that
     replaces the training set, `local_gp_fitting`, snapshots it;
   - the `temporary_data` markers, set in `gaussian_process_train.py` and
     read in `bads.py`.
2. **Survey.** The "Crashes on unguarded GP updates" section gets the fix
   commit and the counts. The `:1164` row gets current line numbers. New
   candidate rows:
   - the target's refactorization under the best iteration's
     hyperparameters, against MATLAB's reuse of the current posterior;
   - `S` in `gp.s2` in `local_gp_fitting`;
   - MATLAB's NaN target after a failed rebuild;
   - the recovery without a MATLAB counterpart;
   - the dropped point, against MATLAB's kept point;
   - full recomputation, against MATLAB's rank-1 update, which MATLAB
     skips under `SpecifyTargetNoise`;
   - `_re_evaluate_history_` mutating stored GPs, and its stale values
     after a failed rebuild.
3. `dev/TODO.md`: close the item, or cut it down to what Phase 3 leaves
   open.
4. **Linux reference.** Copy the post-change population to
   `dev/experiments/population_linux_20260925/`, with the README the
   convention asks for (Open Question 6).
5. The `dev/README.md` index, the Worklog, and Status.

## Open questions (all settled 2026-09-25)

1. **Failed add: drop the point, or keep it as MATLAB does?** Settled at
   the default: drop it. gpyreg's restore keeps the GP consistent, and the
   marker has the next step rebuild the GP from the function logger, which
   holds the point. MATLAB keeps the point beside an empty posterior, and
   its predictions go NaN until the rebuild. Keeping the point in PyBADS
   would mean NaN-tolerant predictions at every call site.
2. **Target after a failed prediction.** Re-settled after the review: use
   the current GP's prediction. The first version said MATLAB gives a NaN
   target here. In fact MATLAB never refactorizes at this point, since it
   reuses the current posterior, and reaches a NaN target only after a
   failed rebuild. The options were: a NaN target (the earlier default),
   the incumbent's `fval` and `fsd` with a finite target, or, chosen, a
   prediction from the GP as it stands (its own hyperparameters and
   posterior).
3. **Where the markers live.** Settled at the default: on the GP, in
   `temporary_data`. They travel with the GP through deep copies and the
   noisy branch's `new_gp`, as MATLAB's empty `post` does, and no
   signature changes. The alternative was a `BADS` attribute plus an exit
   flag returned by `add_and_update_gp`, which is exported from
   `pybads.bads`.
4. **Keep the recovery (old hyperparameters on the new data)?** Settled
   at the default: keep it. Runs where it succeeds complete today and must
   not move. Dropping it to follow MATLAB would change results; that
   belongs to the bug hunt, with its own gate.
5. **Rank-1 update for `'add'`, as MATLAB does it first.** Settled: out of
   scope, and listed in `dev/TODO.md`. It would move results at default
   options and needs its own population comparison.
6. **Commit a Linux reference population?** Settled at the default: yes.
   After the review, it is the post-change population, since the
   pre-change one belongs to code that crashes on failures. Later cloud
   sessions can then gate changes on this platform, where the Windows
   reference only compares across platforms.
7. **Force a refit after a failed rebuild?** New after the review;
   settled: yes. Without one, the same hyperparameters on a similar
   neighbour set will likely keep failing. The restored GP's finite
   predictions also never trip the reliability check that sends MATLAB to
   a refit, so the GP could stay stale for up to a refit period. A failed
   add only marks for a rebuild, as in MATLAB. Only runs with failures are
   affected.

## Worklog

- 2026-09-25: plan drafted; MATLAB BADS read at `74919c0`; gpyreg 1.3.3's
  `update`, `set_hyperparameters` and `predict` read at `98ab5a4`. Phase 0
  steps 1-4 done (above). The user settled Open Question 5 (out of
  scope, a TODO item) and the others at their defaults.
- 2026-09-25: independent review of the plan by a fresh-context agent,
  which confirmed the line references, the MATLAB and gpyreg claims, and
  the bit-equality of the add change at levels 0-2, by snippet. It found a
  blocking defect: the target's fallback raises `AttributeError`. Its
  should-fix findings:
  - MATLAB's `UpdateTarget` reuses the current posterior;
  - the noisy poll after a failed add;
  - the `s2` growth at level 1;
  - `y_new` as a float;
  - the injection keyed on caller frames, and a real Cholesky failure;
  - the stress injection's `P**2` reach;
  - identity checked by record diff;
  - the stale GP after a failed rebuild;
  - the changelog's commit and wording.

  The two MATLAB claims behind the blocker and Open Question 2 were
  checked again by hand (`bads.py:1047`, `:2454`, `:1607`; `bads.m:1301`,
  `gppred.m:40-47`, `mygp.m:123`). The plan was revised; the user chose
  the current GP's prediction for Open Question 2 and a forced refit for
  the new Open Question 7.
- 2026-09-25: Phase 0 done. Pre-change population
  `population_linux_pre_20260925` (default suite, seeds 0-29, 4 workers,
  24.4 minutes): all 540 runs finished, none crashed. Its records name
  `500ff1b`, `88abbf8` and `517f058`, some "dirty": documentation
  commits were made under dev/ while it ran.
  `git diff 09996b5 517f058 -- pybads/ pyproject.toml setup.py` is empty,
  so every run used the package code of `09996b5`. The Conventions now
  forbid commits during a population. Null check (`--split`): no flag in
  36 tests. Against the Windows reference (information only): no flag in
  54 tests, every median log10 error ratio within [-0.28, +0.08], every
  interval containing zero.
- 2026-09-25: Phase 1. `test_gp_update_failures.py` has 39 tests
  (parametrized over levels 0-2). At the unfixed code (package code of
  `09996b5`), 35 fail, each for the stated reason:
  - 20 by the injected `LinAlgError`;
  - 1 by a real Cholesky `LinAlgError` (test 2);
  - 3 by `AttributeError` (the target's fallback);
  - 6 by markers that are neither set nor cleared;
  - 4 by a missing rebuild or refit (tests 6);
  - 1 by the level-1 `s2`, which the old code does not extend.

  The 4 that pass are the no-failure baselines: test 1 at levels 0 and 2,
  and the two unmarked probes. Tests 3 and 4 (recovered failure and
  success) also check that the markers are cleared, so they fail at the
  old code, unlike the plan's first estimate for test 4. 11 s.
- 2026-09-25: Phase 2.
  - The 39 new tests pass (20 s), and the whole suite passes with reruns
    off (148 tests, 46 s).
  - The fingerprint with the v1.3.3 clone is `e5f46bce200bfaa7` in two
    processes, as before.
  - A forced refit records itself through `_record_gp_refit_`, the
    bookkeeping moved out of `_is_gp_refit_time_`.
  - In the poll, a forced refit gives way to `poll_training`.
  - Under `stobads` (off by default), a NaN `f_poll` goes into
    `_sto_success_improvement_`; this is not examined.
