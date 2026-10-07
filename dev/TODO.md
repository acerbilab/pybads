# PyBADS: open work

Updated 2026-10-07. PyBADS 1.5.1 was released on 2026-10-06: the tag
`v1.5.1` on `6e5133a3`, a GitHub release, the upload to PyPI, and the
conda-forge recipe (`conda-forge/pybads-feedstock` #13); `AGENTS.md`
("Setup and commands") gives a release's steps. Other records
name the items by their titles, the port review's ledger
([results/2026-09-28-port-correctness-review.md](results/2026-09-28-port-correctness-review.md))
among them, so a title stays as it is while its item is open.

## Later releases

- [ ] **Sto-BADS: whether to remove it.** Sto-BADS (`stobads`, with
  `opp_stobads`, `stobads_frame_size_scaling_power` and the keyword-only
  argument `gamma_uncertain_interval` of `BADS`) is PyBADS's own,
  experimental, and brings no measured gain
  ([results/2026-09-28-stobads-rule.md](results/2026-09-28-stobads-rule.md)).
  The PI (2026-09-30) keeps it where users find it, on the options page
  since 1.1.0, but deprecated from 1.5.0, so that no user takes
  `stobads=True` for the setting of a noisy target (KD-S-1 of
  `pybads/bads/README.md`). Open: whether a later release removes it. Its
  options would then raise `ValueError` as unknown ones, as the 66 removed
  in 1.5.0 do, and `gamma_uncertain_interval` a `TypeError`, a break that
  needs a line under "Upgrading from"; only `stobads=True` warns now, so a
  script that sets another of these names meets the removal unwarned. The
  `improvement` oracle computes its `sto_flags` through
  `BADS._sto_success_improvement_` (`pybads/testing/oracles/_oracles.py`),
  and `dev/scripts/make_oracle_fixtures.py` reads
  `stobads_frame_size_scaling_power`: the removal changes that oracle's
  recipe, which takes `--write --reason` (`AGENTS.md`, "Numerical
  gates").
- [ ] **Tips: PyVBMC's review.** PyBADS's runtime tips copy PyVBMC's,
  whose tips the PI is to review before PyVBMC's release (PyVBMC's
  `dev/TODO.md`, "Review of the tips."). When that review concludes, its
  outcome for the policy or the wording is applied to PyBADS's copy
  (`pybads/bads/_runtime_tips.py`, `_tip_catalog.py`;
  `dev/plans/runtime-tips.md`). The review covers the old-release reminder
  as well, which shares the tips' slot and switch: its outcome reaches
  `pybads/bads/_release_reminder.py`, `pybads/_update_check.py`,
  `dev/plans/version-check.md`, the FAQ's "How do I know whether a newer
  version of PyBADS exists?", the API page of `check_for_updates` and the
  changelog's "Update reminders".
- [ ] **Periodic variables evaluated at their upper bound.** KD-B1-6 of
  `pybads/bads/README.md` says that the candidates of a periodic variable
  are wrapped into `[lb, ub)`, but the target can receive one exactly at
  `ub`. When `pub == ub` and rounding maps `ub` slightly above 1 in `u`
  space, the grid point `u = 1` lies below it and is not wrapped, and
  `inverse_transf` returns `ub` for it. With `lb = [-1,
  0.38201463519268053]`, `ub = [1, 0.42393798440894204]`, `plb = [-1,
  0.39039930503593284]`, `pub = ub`, `periodic_vars=[1]`, `x0=None` and
  `random_seed=0`, `ub` maps to `u = 1.0000000000000016`, and 3 of the
  run's 50 evaluations have `x[1] == ub`; a fuzz of about 550 runs with
  periodic variables met it in 4 (2026-10-06). The point lies within the
  hard bounds, and a periodic target takes the same value at `lb` and
  `ub`, but a target that indexes a table by the variable can fail there,
  and one point can be evaluated at both bounds. The usual bounds of an
  angle (0 to 2π, ±π, 0 to 360, ±180) map exactly, and so do those of the
  periodic configurations of `dev/scripts/benchmark_targets.py`: a fix
  moves results only where the bounds do not, so its gate needs a
  configuration whose bounds do not map exactly
  (`test_every_source_of_candidates_wraps_them` checks the `u` points
  alone).
- [ ] **The transform's self-test refuses a small bound beside a very
  large one.** `VariableTransformer` checks that the inverse of its
  transform returns each bound within `1e-6 * max(1, |b|)`
  (`pybads/variable_transformer/variables_transformer.py`). With `lb =
  -1.3` and `ub = 1e11`, and the plausible bounds equal to them or
  omitted, the linear transform's offset and scale are about 5e10, whose
  rounding, about 1e-5 at the bound -1.3, exceeds that bound's tolerance,
  and `BADS` raises "Cannot invert the transform to obtain the identity at
  the provided boundaries."; a narrower plausible box is accepted. 1.1.0
  refused these bounds too. The changelog's "bounds from about 1e10 in
  magnitude are no longer refused" (KD-B1-10) holds except in this case.
  A tolerance scaled by the size of the transform, its offset and scale,
  accepts them and moves no run; KD-B1-10 then says so.
- [ ] **A target flat over the initial design.** A target can take one
  value at `x0` and at nearly every point of the initial design: a
  constant target; a start inside a region where `fun` returns a large
  penalty, with no point of the design outside it, which the FAQ's "How
  do I prevent PyBADS from evaluating certain inputs or regions of input
  space?" tells users to avoid; a floor, such as a loss that is exactly 0,
  over the plausible box. Its only sign to the user is gpyreg's
  `UserWarning` "The training targets are all equal, so they have no
  scale for the recommended bounds to take: a range of one is assumed
  instead.", in gpyreg's terms, which a user cannot act on. It comes from
  gpyreg's `get_bounds_info`, which `_gp_hyp`
  (`pybads/bads/gaussian_process_train.py`) calls at the first GP, on the
  lowest `hpd_frac` of the initial targets, and gpyreg's `GP.fit` at every
  fit; Python's default filter prints it once per session, whatever
  `display` is. The run carries on and, if no evaluation leaves the
  plateau, ends on `tol_fun` at the plateau's value, with `success=True`:
  at D = 3, with `fun` returning 1e10 outside the ball of radius 2 around
  1.5 in each variable, the plausible box `[-2, 2]` and `x0` at -1.5,
  seeds 0, 1 and 5 of 0-5 ended so after 53 evaluations (2026-10-06).
  1.1.0 raised gpyreg's `ValueError` on such a design. To do: a message of
  PyBADS's own, in the user's terms (the target took the same value at
  every point of the initial design: check `x0`, and whether `fun`
  returns a constant penalty), linking that answer of the FAQ, with
  gpyreg's warning silenced where PyBADS gives its own; and whether such a
  run reports `success`. `test_plateau_initial_design_runs` and
  `test_rebuild_with_equal_targets_keeps_output_scale_prior`
  (`pybads/testing/bads/test_gaussian_process_train.py`) emit the warning.
- [ ] **Update reminders and tips: loose ends.** Small, and in the files
  that the outcome of "Tips: PyVBMC's review." (above) reaches:
  - Under uv, `check_for_updates()` suggests `python -m pip install
    --upgrade pybads, or with conda: ...`, which fails in a uv venv, which
    has no pip: the distribution's `INSTALLER` reads `uv`, which
    `_update_command` (`pybads/_update_check.py`) does not know. A `uv pip
    install --upgrade pybads` for it goes on the API page of
    `check_for_updates` too.
  - Two sessions that start a run within about 2 ms of each other can both
    show the reminder, and record one showing:
    `consider_release_reminder` (`pybads/bads/_release_reminder.py`)
    reads the state file and writes it with no lock between processes.
    Process pools of 8 showed it twice in one of 12 configurations.
  - A showing dated in the future in the state file, written under a
    wrong clock, silences the reminder of that version for good.
  - `check_for_updates()` ignores a release's `requires_python`: once a
    release drops a version of Python, users of that version are told
    that a release they cannot install is available.
  - `FAILURE_MESSAGE` (`pybads/_update_check.py`) says "Could not reach
    PyPI" for the reasons "no release found" and "unreadable reply" too,
    where PyPI did reply.
  - The tip `multiple_starts` (`pybads/bads/_tip_catalog.py`) advises "at
    least 10 different starting points, ideally dozens", as the FAQ's
    answer on `x0` does, while the answer that the tip links, "How do I
    run PyBADS from several starting points?", gives only `n_runs = 10` in
    its example; `AGENTS.md` wants a tip to restate the advice of the
    answer that it links, with its quantities.
- [ ] **Rough edges of the interface.** Each is harmless or as in 1.1.0:
  - The levels of the opening and the final messages, 25 and 22
    (`_LOG_NOTIFY` and `_LOG_FINAL` in `pybads/bads/bads.py`), print as
    `Level 25` and `Level 22` under a user's logging format that shows
    `%(levelname)s`. `logging.addLevelName` names them, at the cost of a
    change to the logging module's names, which are global.
  - `poll_mesh_multiplier=2`, an integer, fails with NumPy's "Integers to
    negative integer powers are not allowed" at the mesh's update in
    `_optimize_`, and `cache_size=1000.0` with a `TypeError` from
    `np.full` in `FunctionLogger`: the checks of the options do not cover
    them.
  - Without `x0`, a scalar plausible bound makes D equal to 1 even when
    the hard bounds are arrays, and `BADS` then raises a `ValueError` that
    the hard bounds do not have D = 1 elements; so does a scalar
    `lb` beside an array `ub` without plausible bounds. The changelog's "A
    scalar bound stands for the same bound in every variable" holds when
    `x0` or a plausible bound gives D.
  - Target values of magnitude 1e154 or more stop the run, as the variance
    of the targets overflows (gpyreg's `ValueError` "The prior of
    mean_const has an infinite sigma", or an `OverflowError`), and values
    of magnitude 1e-200 or less give `RuntimeWarning`s from gpyreg.
  - A target that returns `np.float32` gives a result whose `fval` is an
    `np.float32`.

## Needing no release

- [ ] **The oracles' rebaseline test and the Linux reference under newer
  versions.** Remaining: the Linux reference,
  `population_linux_gpyreg140_20260930`, ran
  under Python 3.11, NumPy 2.4.6 and SciPy 1.17.1. Runs under Python 3.12,
  NumPy 2.5.3 and SciPy 1.18.1 do not pair with it seed by seed: the same
  code at 10 seeds drew a flag on `ellipsoid_D6` (KS test, p = 0.032 after
  Holm), which 30 seeds did not. A Linux gate under the newer versions
  takes a new reference, or selects the reference's versions.

## Waiting on MATLAB BADS

Each needs MATLAB and the BADS toolbox, and one session with them can serve
all three.

- [ ] **`ellipsoid_D3_hetero` after `020d6a8`.** Squaring the target's noise
  standard deviations, as MATLAB does, made the runs of this benchmark
  configuration worse: over 90 seeds the median error rose from 0.21 to
  0.54 on Windows and from 0.18 to 0.58 on Linux, mostly along the flat
  axis of the ellipsoid
  ([Windows](experiments/population_ellipsoid_hetero_20260925/README.md),
  [Linux](experiments/population_ellipsoid_hetero_linux_20260925/README.md)),
  while the spheres with target noise improved. The fix stays. Three
  differences from MATLAB, fixed on Linux, bring the median to 0.25 and
  the flat axis back to its error before `020d6a8`: the bound of the GP log
  length scales (`97b2c66`, the largest effect), a repeated point merged
  into another point's row of the function log (`032dfcb`), and the GP
  mean prior, re-centred at each rebuild (`8afbe16`). Returning the
  observation of a repeated point, as MATLAB's `funlogger` does, makes the
  runs worse, and the lower bound of the noise hyperparameter never moves,
  since no fit fails. The runs remain worse than before `020d6a8` (p =
  0.0008), now along the two steep axes. The port review fixed two more
  differences:
  - the evaluated points that `contraints_check` kept: W3-1 (`149d528`,
    wave 3) removes them, as MATLAB does; over seeds 0-29 of this
    configuration its 100 repeats (of 8800 evaluations, in 17 runs) are
    gone, and the median error moved from 0.43 to 0.36, unflagged
    (`experiments/port_review_20260925/verification/wave3_fixpass/`);
  - the bounds of the GP mean, which the port set from the initial design
    and MATLAB leaves infinite; once `8afbe16` re-centred the prior of the
    mean, it could fall outside them, which made the log prior NaN in the
    fits of 67 runs of the Windows population at `ab4dded`
    ([experiments/population_gpfixes_20260925/](experiments/population_gpfixes_20260925/README.md)).
    W1-23 (`172df00`, wave 1) leaves them infinite, as MATLAB does.

  Still open: a run of MATLAB BADS on this problem, which would show
  whether correct noise handling alone gives such runs. Measured on
  2026-09-28: its GP runs at an output variance near 1e15 times its noise
  variance, and most of its fits keep a noise that gpyreg multiplied
  (KD-B6-6); with gpyreg's switch to MATLAB's rule its errors are not
  measurably smaller, nor with any of the four Sto-BADS arms measured
  ([results/2026-09-28-gp-health.md](results/2026-09-28-gp-health.md),
  [results/2026-09-28-stobads-rule.md](results/2026-09-28-stobads-rule.md)).
- [ ] **Zero predictive SDs: how often MATLAB gives them.** The predictive
  SD of the GP is exactly 0 at about a tenth of the poll's acquisitions
  over the four suites, up to 40% on some configurations, noisy ones
  included, and each makes the poll's GP unreliable (W3-28). Counted and
  traced on 2026-09-28
  ([results/2026-09-28-gp-health.md](results/2026-09-28-gp-health.md)):
  rounding, the latent variance `kss - v'v` cancelled below the rounding
  of `kss` near the training inputs and clamped at 0, where the output
  variance exceeds the noise by 1e12 or more, the same cause as KD-B6-6;
  MATLAB's `mygp.m:187` clamps the same way. Open only: how often MATLAB's
  own fits reach them, which needs MATLAB.
- [ ] **Numerical oracles computed by MATLAB BADS.** The oracles of
  `pybads/testing/oracles/` are PyBADS's own numbers on stored states: they
  pin the numerics against change, not the port against MATLAB BADS.
  Oracles computed by MATLAB BADS on the same states would check the pure
  pieces that are not deliberate differences (`pybads/bads/README.md`):
  `transvars.m`, `udist.m`, `force2grid.m`, `ucov.m`, the priors of
  `gpdefBads.m` but the cases of KD-B6-2 and the centre of the mean's
  prior on one point (KD-B6-5), `acqLCB.m` with `gppred.m` at
  fixed hyperparameters, the ES search's weights, `searchHedge.m`'s update
  and `pollMADS2N.m` with injected draws. The fixtures are the inputs such
  a harness would take: plain arrays and JSON, with prescribed draws
  (`ScriptedGenerator` in `_oracles.py`), which can be handed to MATLAB as
  arrays. Generating the references needs MATLAB and the BADS toolbox.

## Not adopted

- [ ] **Rank-1 GP update when adding a point: not adopted, to revisit if
  its terms change.** MATLAB BADS adds a point to the GP by a rank-1
  update of the posterior (`private/gpupdate.m`, `utils/update_posterior.m`);
  PyBADS's `add_and_update_gp` recomputes every posterior in full, and says
  why at its call of `gp.update`. The measurement
  ([results/2026-09-28-where-pybads-spends-its-time.md](results/2026-09-28-where-pybads-spends-its-time.md))
  settled the questions: gpyreg's rank-1 path agrees with the full
  recomputation to rounding, target noise included (so there is no reason
  to skip it under `specify_target_noise`, as MATLAB does); it would save
  at most 2.5 % of PyBADS's own time, and is slower below about 60
  training points; but on the ellipsoids 40 to 44 % of the additions meet a
  GP whose factorization needed gpyreg's noise multiplier, where it
  differs from the recomputation by up to 1e-2 of the targets' spread, and
  by 0.21 when the multiplier it carries over differs from the one the
  recomputation picks. The PI did not adopt it (2026-09-28). Revisit if the
  training sets grow well beyond 200 points, if gpyreg's handling of the
  multiplier changes (KD-B6-6 of `pybads/bads/README.md`, kept as it is
  by the PI on 2026-09-28), or if a profile shows the update after
  a new point taking a larger share. Adopting it moves the ellipsoids'
  runs, so it needs the population comparison, and the target's reuse of
  the GP's own posterior (`_get_target_from_gp_`), which relies on
  posteriors computed in full, needs revisiting with it.

## Porting work

- [ ] **Benchmarking on neurobench**, open porting work listed in
  `pybads/bads/README.md`: PyBADS on cognitive and neural science models
  ([neurobench](https://github.com/lacerbi/neurobench)).
