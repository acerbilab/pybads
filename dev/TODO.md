# PyBADS: open work

Updated 2026-09-28. The list describes scope, not priority or execution
order. The next release is 1.5.0 (tag `v1.5.0`), the version the PI has
decided on; "the next release" below means it.

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
- [ ] **The old `LinAlgError` crashes and the bound of the GP length
  scales.** `_gp_hyp` bounded each log length scale by `cov_range = min(100,
  10 * (ub - lb) / scale)`, where MATLAB's `gpdefBads.m` bounds it by
  `log(covrange)`: 80 against 4.38 on the targets of the benchmark with its
  shifted box; `97b2c66` takes MATLAB's bound. At the failing calls of the
  four `LinAlgError` crashes of the survey's section "Crashes on unguarded
  GP updates", most log length scales exceed 4.38, up to 59.7, so that many
  distinct inputs coincide numerically, and the output scale, at its upper
  bound (MATLAB's too), sets an output variance 2e22 to 2e24 times the
  noise variance on them. Open:
  - whether MATLAB's bound would have kept those GPs factorizable: the
    GP of each failing call refitted under it. The inputs, targets and
    hyperparameters saved at those calls (machine-local,
    `dev/scripts/runs/LOCAL.md`) lack the priors and bounds that
    `_gp_hyp` sets; a capture that keeps a deep copy of the GP before
    each call of `gpyreg.GP.update` (a wrapper loaded through a
    `sitecustomize.py` first on `PYTHONPATH`) has them all. A rerun with
    the bound changed follows another trajectory and cannot show it. A
    rerun of the crashes needs Windows with the environment of their
    records (Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1, one BLAS thread),
    their commits `2226883` and `c85cddb` (reachable from
    `refs/pull/59/head`) and a clone of gpyreg at v1.3.1, since no run of
    the suite fails under gpyreg 1.3.3.
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
- [ ] **conda-forge recipe.** The test command of `conda-forge/pybads-feedstock`
  (`recipe/meta.yaml`) passes `--reruns=5` and requires
  pytest-rerunfailures. The tests of 1.1.0, which it runs, are not all
  seeded, so both stay until the first release after 1.1.0, whose tests
  are: drop them in the version-update PR that the feedstock's bot opens
  for that release, before it is merged.
- [ ] **"What's new" at the next release.** `README.md` and
  `docsrc/source/index.rst` list under "What's new in PyBADS 1.1" that
  every random draw of a run comes from one generator created from
  `random_seed`, which 1.1.0's initial design did not follow (its
  scrambling was seeded from the start). The list "What's new in PyBADS
  1.5" replaces it, and says that `random_seed` now decides the initial
  design (the doublecheck of wave 4 of the port review,
  `experiments/port_review_20260925/verification/wave4.md`,
  "Doublecheck"), and that periodic variables (`periodic_vars`, Example
  6) are supported. At the same release, `skills/pybads/SKILL.md` and the
  FAQ (`docsrc/source/faq.md`), which name no release, name 1.5, as
  PyVBMC's skill and FAQ do. The published documentation follows `main`,
  so until that release the FAQ describes code that no release has, among
  it `output_fcn(x, optim_state, state)`, which 1.1.0 calls as
  `output_fcn(x, "init")`.
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
- [ ] **Exact step-by-step replay and numerical oracles**, after PyVBMC's
  (`dev/scripts/golden_replay.py`, `pyvbmc/testing/oracles/`). They were
  to follow the bug hunt, so as not to pin its defects, and the hunt is
  done ([the port correctness review](results/2026-09-28-port-correctness-review.md)):
  nothing holds them back. The random draws go through one generator per
  run (`bads.rng`), which replay needs. An oracle computed by MATLAB BADS
  needs MATLAB's own numbers, which PyVBMC's MATLAB-comparison helpers
  (`pyvbmc/testing/_compare_matlab.py`: `randn2` and the draws that
  reproduce MATLAB's random stream) give. The population comparison of
  `dev/scripts/population.py` checks distributions, not trajectories,
  until then. A replay that compares trajectories bit for bit holds on
  Linux and Windows but not on macOS arm64
  ([results/2026-09-28-macos-arm64-repeatability.md](results/2026-09-28-macos-arm64-repeatability.md)).
- [ ] **Profiler**, after PyVBMC's (`dev/scripts/profile_run.py` and kin),
  once PyBADS times its search, poll and GP-training stages separately:
  today its timer covers only the whole run and the target's evaluations.
  One stage is measured: the failed tries of the refits take 9% of the
  deterministic runs' time over four suites, 42% of `ellipsoid_D3`'s
  ([results/2026-09-28-gp-health.md](results/2026-09-28-gp-health.md)).
- [ ] **Benchmarking on neurobench**, the open porting work listed in
  `pybads/bads/README.md`: PyBADS on cognitive and neural science models
  ([neurobench](https://github.com/lacerbi/neurobench)).
- [ ] **Prior evaluations (`fun_values`).** MATLAB BADS imports
  evaluations made before the run into its log and its GP
  (`private/setupvars.m:126-167`, `private/funlogger.m`) and takes its
  first incumbent from `x0` and the initial design only
  (`private/evalinitmesh.m:120-123`). PyBADS's `fun_values` never worked,
  and by the ruling on row W2-6 of the port review
  (`experiments/port_review_20260925/verification/wave2.md`) a non-empty
  value is refused with a message. A port imports them after the function
  logger exists, keeps them out of the choice of the first incumbent, and
  needs a test that its GP holds them. `FunctionLogger.add`, which such a
  port would call, keeps checks of its own on the value and its SD, records
  a missing SD as 1 when the logger holds SDs and drops a given one when it
  does not, and what it records for a repeated point is settled with it
  (row W4-10); so is the bookkeeping of the final samples, which still add
  to the incumbent's `n_evals` in the log and average their times into its
  row, after the run's last decision (PI, 2026-09-28).
- [ ] **The GP on a one-point training set.** When `non_box_cons` leaves
  only `x0` feasible (the thin band of row W2-37 of the port review), the
  GP is fitted on one point: gpyreg's bounds helper replaces the targets by
  `[0, 1]`, so the mean's prior at the initial fit is centred at 0.5
  whatever the target (wave 1's fix pass, `verification/wave1.md`, "Found
  while fixing"), and `get_bounds_info`, called from `_gp_hyp`, warns of a
  log of zero and a variance with no degrees of freedom (wave 2's
  verifiers, `verification/wave2.md`, "Found while verifying"). On two
  distinct points, the empirical prior of the length scales had a sigma of
  0, which gpyreg refuses; since W3-40 (wave 3's fix pass,
  `verification/wave3.md`) a rebuild keeps the previous prior there.
  Slice B6, whose wave has passed: decide the priors and bounds of such a
  GP, in PyBADS or in gpyreg, with a test on the thin band. Measured on
  2026-09-28 over the four suites
  ([results/2026-09-28-gp-health.md](results/2026-09-28-gp-health.md)):
  only non-box constraints reach it, every run of the thin bands on one
  point and 2 of 30 runs of `sphere_nonbox_D3` on two, and the runs at
  D = 3 go on to converge.
- [ ] **The example notebooks' saved outputs.** Nothing runs the notebooks
  of `examples/`, and the saved outputs of the first five predate the port
  review, whose fix passes change their numbers, and some of their
  messages: `pybads_example_2_nonbox_constraints.ipynb` shows the warning
  `bads:TooCloseBounds`, which W2-4 removed;
  `pybads_example_4_user_provided_noise.ipynb` a termination message on
  `tol_mesh` that speaks of the change in the function value, which wave 0
  corrected; `pybads_example_5_extended_usage.ipynb` a result with
  `'fsd': 0` and without `status` (W2-12, W2-13), from version
  `0.8.3.dev21`. The rerun of all five waited for the review's fix passes,
  so that the outputs would not be regenerated at every pass
  (`experiments/port_review_20260925/verification/wave2.md`, "Fix pass");
  the passes have all landed (the review closed on 2026-09-28), so the
  rerun goes with the headless run of the examples before the release.
  Example 6 (periodic variables) was run with gpyreg's development branch
  (`3f1a732`), before any gpyreg release had `periods`, and is rerun with
  the others, with the released gpyreg.
- [ ] **Loose ends of the port review.** Observations that the reports and
  the fix agents made outside their findings, which no ruling took up and
  which change no default run, are listed in the consolidated ledger
  ([results/2026-09-28-port-correctness-review.md](results/2026-09-28-port-correctness-review.md),
  "Open ends"). Each is fixed, documented, or dropped.
- [ ] **Checks of option values when `BADS` is created.** `BADS` refuses
  a bad value of some options when it is created, with a `ValueError`
  that names the option (among them `max_fun_evals`, the options whose
  default is a boolean, `tol_fun` and `random_seed`), but not of others.
  Measured on 2026-09-28: a string for `max_iter`, `search_n_try` or
  `tol_stall_iters` (`"200*D"`, `"D"`, `"5"`) raises a `TypeError` inside
  `optimize()`, after the initial design has spent its evaluations; a
  string for `tol_mesh` raises NumPy's `TypeError` at creation, and one
  for `noise_size` too, at the comparison of its check, neither naming the
  option; `max_iter=2.5` is taken as it is; and `tol_mesh=-1.0` runs, with
  a `RuntimeWarning` from its logarithm, and the mesh criterion never ends
  the run. A check of each when `BADS` is created, with a `ValueError`
  that names the option, as for `max_fun_evals`, changes what a script
  written for 1.1.0 gets, so it takes a line of the changelog's "Upgrading
  from" list, and the FAQ's answer on the differences from MATLAB BADS,
  which describes the present behaviour, changes with it.
- [ ] **User documentation written out twice.** Some advice is written
  out in several places, which can drift apart: the list of the problems
  that PyBADS suits, in `README.md` ("When should I use PyBADS?"),
  `docsrc/source/index.rst` ("Should I use PyBADS?"), section 0 of
  Example 1 and the FAQ ("Which kind of problems is PyBADS suited for?");
  the passing of additional data to the objective, in the getting-started
  page (`docsrc/source/quickstart.rst`) and the FAQ; and the amount of
  noise that PyBADS handles, in the "Remarks" of Example 3 and the FAQ.
  Each is kept in one place and linked from the others, or the copies are
  kept in step.
- [ ] **gpyreg releases after 1.3.3.** PyBADS's minimum gpyreg
  (`pyproject.toml`) and its CI pin (`GPYREG_PIN`) name one release, 1.3.3
  as of 2026-09-25 ([assessment](results/2026-09-25-gpyreg-1.3.3.md)).
  Each new release moves both, after the population comparison
  (`dev/scripts/population.py compare`) against the current reference
  shows that it has no effect on PyBADS, or explains the one it has. A
  move is a change for users: an entry in `CHANGELOG.md` and a line in its
  "Upgrading from" list. PyBADS's next release waits for gpyreg's next one
  (PI, 2026-09-28), which is to hold:
  - `periods` on the ARD kernels (`SquaredExponential`, `Matern`,
    `RationalQuadraticARD`), on gpyreg's branch
    `claude/todo-discussion-q8c9im`, which PyBADS's `periodic_vars` needs:
    `_gp_periods` passes them to the kernel of a run with periodic
    variables, and a run without them builds its kernel as before, so the
    comparison of this release is expected to flag nothing. The move is
    therefore required for 1.5.0, not only allowed: under gpyreg 1.3.3 a
    run with periodic variables stops with `TypeError` when its GP is
    built, and the tests of periodic variables fail under the CI pin. The
    changelog's entry "Requirements" and the line of "Upgrading from
    1.1.0" name the new minimum. The `default` suite holds two periodic
    configurations, `periodic_D4` and `periodic_D3_homo` (PI, 2026-09-29),
    which its references, the Windows one at 100 seeds and the Linux one
    at 30, lack: `population.py compare` tests only the configurations
    that both populations hold, so until then it skips them without a
    word. The release's comparison runs the whole `default` suite with the
    release's clone on both platforms, at the references' numbers of
    seeds, and its populations become the new references, the two
    periodic configurations included (PI, 2026-09-29): on Linux these two
    are compared with the "on" arm of
    [experiments/population_periodic_linux_20260928/](experiments/population_periodic_linux_20260928/README.md);
    on Windows, which has no run of them, the whole `periodic` suite runs
    in both arms (with `--options '{"periodic_vars": null}'` for "off"),
    as on Linux. gpyreg's release notes list `periods` under "1.3.4 (unreleased)"; a new
    feature may call for 1.4.0, the maintainers' choice;
  - the fix of the port review's W1-24 (the log prior of a prior far
    outside its bounds, acerbilab/gpyreg#57) and W1-25's switch
    (acerbilab/gpyreg#56), which stays off in PyBADS (KD-B6-6), both on
    gpyreg's `main` at `1893eff`;
  - the gradients of the rational-quadratic ARD and Matern kernels
    computing the factor common to all dimensions once, to the same bits,
    on gpyreg's `main` since acerbilab/gpyreg#60 (merge commit `280d8c0`).
    The kernel takes 31 to 45 % of a PyBADS run in its own code, in the
    predictions at the ES search's candidates and in the hyperparameter
    fits; only the fits compute its gradient, and on `ellipsoid_D10` their
    kernel takes 6.0 s of a 23-s run (Windows, gpyreg 1.3.3,
    [results/2026-09-28-where-pybads-spends-its-time.md](results/2026-09-28-where-pybads-spends-its-time.md)).
    On Linux with one BLAS thread, the kernel with its gradient takes a
    third less time at D = 10 and 150 training points and a quarter less
    at D = 6 and 110, and PyBADS's runs of `ellipsoid_D10`, seeds 0-5,
    take 0.85 to 0.95 of their wall time with gpyreg at `1893eff` (median
    0.90, paired by seed, the two arms side by side on four cores), with
    records otherwise equal;
  - pytest, pytest-rerunfailures and numdifftools, which only gpyreg's tests
    use, in a `test` extra instead of gpyreg's runtime dependencies, also
    since acerbilab/gpyreg#60, so that installing PyBADS does not install
    them. Moving PyBADS's minimum to that release makes false the
    "(gpyreg 1.3.3 still installs it)" of the entry "Requirements" of
    `CHANGELOG.md`'s `Unreleased`, which changes with it.

  At `1893eff`, with the switch off, `main` gives gpyreg 1.3.3's records in
  all 1,080 runs of the `default`, `geometry`, `oned` and `bounds` suites
  on Linux
  ([experiments/gp_switch_linux_20260928/](experiments/gp_switch_linux_20260928/README.md)).
  With gpyreg at `280d8c0` (the tree of the last commit of
  acerbilab/gpyreg#60, `1095742`) and PyBADS at `bec8a57a`,
  `dev/scripts/fingerprint.py` gives 1.3.3's hash, `4146a986863602cb`
  (Linux, NumPy 2.4.6, SciPy 1.17.1, one BLAS thread), as the kernel
  change's bit identity implies. The release's own gate, the comparison
  run with its clone, is therefore expected to flag nothing on Linux, and
  takes the Windows comparison too. conda-forge's `gpyreg-feedstock` lists
  the three test packages among its run requirements (`recipe/meta.yaml`),
  and its bot merges its version-update PR once a CI that only imports
  gpyreg passes: that PR has them dropped before it merges.
