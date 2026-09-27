# PyBADS: open work

Updated 2026-09-27. The list describes scope, not priority or execution
order.

- [ ] **gpyreg's inflation of the GP noise (W1-25), after wave 1's fixes.**
  gpyreg multiplies the noise by ten per failed Cholesky factorization, up
  to 1e9, and the posterior keeps the multiplier, which the hyperparameters
  do not show; MATLAB BADS treats the failure as an error. gpyreg's
  `raise_on_cholesky_failure` (acerbilab/gpyreg#56, off by default) gives
  MATLAB's behaviour; turned on at the end of wave 1's first batch it made
  the deterministic ellipsoids take more evaluations and end with larger
  errors (`dev/experiments/port_review_20260925/verification/wave1.md`,
  "W1-25's measurement"). That measurement predates acerbilab/gpyreg#58:
  with the switch on, a fit whose low-noise design points all failed
  started its second optimization from one of them and raised, discarding
  its first. Revisit once all of wave 1's fixes have landed
  (PI, 2026-09-26): measure again at that head, count the failed
  factorizations and fits per run and which retries leave a worse GP, and
  weigh a jitter scaled to the signal that the fit and the predictions
  share. Turning the switch on in PyBADS needs a gpyreg release and moves
  its minimum and CI pin.
- [ ] **Rank-1 GP update when adding a point.** MATLAB BADS adds a point
  to the GP (`gpupdate(..., 'add', ...)`, `private/gpupdate.m`) by a rank-1
  update of the posterior (`utils/update_posterior.m`), falls back to the
  full recomputation when that fails, and skips the rank-1 update under
  `SpecifyTargetNoise`. The port's `add_and_update_gp` recomputes every
  posterior in full. gpyreg's `update` has a rank-1 path of its own (one
  new point, no new hyperparameters, posteriors that hold their factors),
  which PyBADS does not take, and which accepts a noise variance for the
  new point. The guards of
  [plans/gp-update-guards.md](plans/gp-update-guards.md) keep the full
  recomputation (its Open Question 5) so that runs without a failure do
  not move. Taking the rank-1 path would move results at default options,
  so it needs the population comparison. To settle: whether gpyreg's path
  follows MATLAB's, whether PyBADS should skip it with target noise as
  MATLAB does, and what it saves in time.
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
  0.0008), now along the two steep axes. Still open:
  - the evaluated points that `contraints_check` kept: W3-1 (`149d528`, the
    port review's wave 3) removes them, as MATLAB does; over seeds 0-29 of
    this configuration its 100 repeats (of 8800 evaluations, in 17 runs)
    are gone, and the median error moved from 0.43 to 0.36, unflagged
    (`experiments/port_review_20260925/verification/wave3_fixpass/`);
  - a run of MATLAB BADS on this problem, which would show whether correct
    noise handling alone gives such runs;
  - the bounds of the GP mean, which the port fixes by the initial design
    and MATLAB leaves infinite (a row of the survey's candidate table);
    since `8afbe16` the prior of the mean can fall outside them, which
    makes the log prior NaN in the fits of 67 runs of the Windows reference
    ([experiments/population_gpfixes_20260925/](experiments/population_gpfixes_20260925/README.md)).
- [ ] **The uncertainty interval of Sto-BADS.** Rows W0-12 and W0-13 of the
  port review's ledger (`experiments/port_review_20260925/verification/wave0.md`):
  the success rule of `stobads=True` compares the estimated improvement
  with `gamma * epsilon * mesh_size**2`, epsilon the GP's standard
  deviations, which do not shrink with the mesh as the accuracy that
  Sto-MADS requires of its estimates does, so that its "certain" outcomes
  are nearly coin flips at small meshes; and `opp_stobads` moves the search
  incumbent on any uncertain outcome, to worse estimates too, and widens
  the search as after an incremental improvement. To decide (PI,
  2026-09-26) after a population with `stobads=True` on the noisy
  configurations of the default suite that compares the current rule, the
  rule without the mesh factor (`stobads_frame_size_scaling_power = 0`)
  and `opp_stobads` moves limited to a positive estimated improvement.
  Since wave 4 (W4-15) an uncertain poll moves the incumbent only to a
  point that improves on it; the search's move on an uncertain outcome is
  not limited so.
- [ ] **conda-forge recipe.** The test command of `conda-forge/pybads-feedstock`
  (`recipe/meta.yaml`) passes `--reruns=5` and requires
  pytest-rerunfailures. The tests of 1.1.0, which it runs, are not all
  seeded, so both stay until the first release after 1.1.0, whose tests
  are: drop them in the version-update PR that the feedstock's bot opens
  for that release, before it is merged.
- [ ] **The target's copy of the GP at every step.**
  `_get_target_from_gp_` deep-copies the GP and recomputes its posterior
  under the best iteration's hyperparameters at every search and poll step
  (the port review's sheet, KD-B4-2), and at default options nothing reads
  the search's target (the port review's wave 3, "Found while verifying").
  It costs time, and it is the path on which the call can raise
  `LinAlgError`. To settle: compute the search's target only when an
  option reads it, and reuse the posterior when the best iteration's
  hyperparameters are the current ones; a change must leave the
  fingerprint of `dev/scripts/fingerprint.py` unchanged, or take the
  population comparison.
- [ ] **Zero predictive SDs at uncertainty level 0.** In deterministic runs
  the predictive SD of the GP is often exactly 0, and the poll's check of
  an unreliable GP reads it (the port review's wave 3, W3-28 and "Found
  while verifying"). Its cause, perhaps the latent variance clamped at 0
  after rounding in gpyreg's `predict`, and whether MATLAB's `mygp` gives
  it as often, are not established. To settle: count the zero SDs over the
  default suite, trace them in gpyreg, and compare with MATLAB's prediction
  of the same GP.
- [ ] **Bug hunt and verification against MATLAB BADS.** In progress:
  [plans/port-correctness-review.md](plans/port-correctness-review.md), one
  branch per wave (`dev-port-review-w<N>`), each merged into `dev-next`. A systematic check of the port against the MATLAB reference (`acerbilab/bads`),
  settling the reach and effect of each candidate defect. The starting point
  is the [survey](results/2026-09-23-codebase-survey.md): its candidate
  table (only partly looked at, never compared with MATLAB), and a finding
  of its section on the tests, the seed of the initial Sobol design, fixed
  by wave 4 (W4-1); what MATLAB's own seed is, a question for MATLAB, is in
  `experiments/port_review_20260925/matlab_side_defects.md`.
  PyVBMC's MATLAB-comparison helpers (`pyvbmc/testing/_compare_matlab.py`:
  `randn2` and the draws that reproduce MATLAB's random stream) come with
  it, for the comparisons that need MATLAB's own numbers.
- [ ] **Exact step-by-step replay and numerical oracles**, after PyVBMC's
  (`dev/scripts/golden_replay.py`, `pyvbmc/testing/oracles/`), after the
  bug hunt, so that they do not pin today's defects; the random draws go
  through one generator per run (`bads.rng`), which replay needs. The
  population comparison of `dev/scripts/population.py` checks
  distributions, not trajectories, until then.
- [ ] **Profiler**, after PyVBMC's (`dev/scripts/profile_run.py` and kin),
  once PyBADS times its search, poll and GP-training stages separately:
  today its timer covers only the whole run and the target's evaluations.
- [ ] **Porting gaps** listed in `pybads/bads/README.md` (periodic
  variables, benchmarking on neurobench). A port of periodic variables also
  assigns `period_check`'s result at every call site, as MATLAB BADS does,
  where the poll discards it, and gives the initial design `optim_state`'s
  boolean mask of the periodic variables, where it passes the option, a
  list of indices (rows W3-35 and W4-11 of the port review).
- [ ] **`gp_cov_prior="ard"`.** MATLAB's per-dimension empirical prior of
  the GP length scales (`gpdef/gpdefBads.m:254-274`) is not ported; by the
  ruling on row W1-28 of the port review
  (`experiments/port_review_20260925/verification/wave1.md`), PyBADS
  refuses the value with a message instead. A port needs its own population
  comparison with the option set.
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
  port would call, keeps checks of its own on the value and its SD, and
  what it records for a repeated point is settled with it (row W4-10).
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
  GP, in PyBADS or in gpyreg, with a test on the thin band.
- [ ] **The example notebooks' saved outputs.** Nothing runs the notebooks
  of `examples/`, and the saved outputs of all five predate the port
  review, whose fix passes change their numbers, and some of their
  messages: `pybads_example_2_nonbox_constraints.ipynb` shows the warning
  `bads:TooCloseBounds`, which W2-4 removed;
  `pybads_example_4_user_provided_noise.ipynb` a termination message on
  `tol_mesh` that speaks of the change in the function value, which wave 0
  corrected; `pybads_example_5_extended_usage.ipynb` a result with
  `'fsd': 0` and without `status` (W2-12, W2-13), from version
  `0.8.3.dev21`. Rerun all five once the review's fix passes have landed,
  with the headless run of the examples before the release, so that the
  outputs are not regenerated at every pass
  (`experiments/port_review_20260925/verification/wave2.md`, "Fix pass").
- [ ] **Minor items of slices B1 and B2 of the port review**, whose wave has
  passed (`experiments/port_review_20260925/verification/wave2.md`, "Found
  while fixing" and "Doublecheck", with the details). Not fixed:
  - the options: `Options.descriptions` has no entry for an advanced
    option that the user set, so `str(options)` prints `(None)` for it;
    the checks `stobads is None` and `specify_target_noise is None` cannot
    fire since W2-19; `test_options.ini` and `test_options2.ini` ship in
    the wheel and nothing reads them; a 0-d array for `max_fun_evals` or a
    boolean option is refused, where 1.1.0 took it;
  - the display: `optim_state["cache_active"]` is always False since W2-7,
    so the cache branches of the display cannot run; the reports of the log
    transform and of periodic variables are logged at INFO, so that
    `"notify"` and `"final"` hide them, and the caution for infinite bounds
    at WARNING, so that `"off"` shows it, where MATLAB BADS prints all three
    from `"notify"` on (`setupvars.m:30`, `119`, `122`);
  - the inputs: a missing `x0` with plausible bounds given as a list or a
    Python scalar raises `AttributeError`; `__init__` fills missing
    plausible bounds without `bads:pbUnspecified`, which MATLAB BADS logs
    whenever it fills them; the redraw of a random start tests it before it
    is put on the mesh (KD-B1-9); the test of fixed variables leaves `x0`
    out (KD-B1-7);
  - the run's control: the test of the reserve for the final samples does
    not reach its floor at 0; the docstring of
    `test_iterations_count_from_one` says "8th" where the run reports 7;
  - `VariableTransformer` used directly: a scalar `apply_log_t` raises
    `AttributeError`, a 1-D bound `IndexError`, and a NumPy scalar hard
    bound with the plausible bounds omitted fails; the `else` branches of
    its four bounds cannot run.
- [ ] **Minor items of slices B7 and O of the port review**, whose wave has
  passed (`experiments/port_review_20260925/verification/wave4.md`, "Found
  while fixing", with the details). Not fixed:
  - the search: `_search_step_` calls `acq_fcn_lcb` on the chosen search
    point without `search_acq_fcn`'s `sqrt_beta`, where `bads.m:578`
    applies `SearchAcqFcn`; only its mean is read, so nothing moves, but a
    callable `sqrt_beta` is not called there; `acq_fcn_lcb`'s comment
    `# Returns z, dz,ymu,ys,fmu,fs,*fpi*` lists MATLAB's outputs; the port
    floors the ES search's `mu = n_search / n_search_iter`, where
    `private/setupvars.m:186` does not; `n_search` and `search_method` are
    not checked (an empty `search_method` fails at the first search), and a
    `search_acq_fcn` that is not a pair fails when `BADS` is created with an
    unrelated `IndexError` or `TypeError`, while a first element other than
    `"acq_LCB"` fails only at the first search; `_search_step_`'s docstring
    gives `search_dist` as an array and has the typo "thecurrent";
  - the checks: a NumPy complex scalar passes the checks of `hedge_gamma`,
    `hedge_beta`, `hedge_decay` and `improvement_quantile`, since NumPy
    orders complex numbers, and a one-element array is accepted and stored
    as an array; `tol_fun` is not checked (0 raises a bare
    `ZeroDivisionError` while the `.ini` default of `hedge_beta` is
    evaluated, and a negative value is refused only through `hedge_beta`);
  - the function logger: `FunctionLogger.add` keeps checks of its own (a
    string value raises `TypeError`, a Python complex of zero imaginary part
    passes `np.isreal` and fails while recorded, a one-element array is
    refused); the final samples still add to the incumbent's `n_evals` and
    average their times into its row, as the noise test did before W4-6, at
    the end of the run; `test_function_logger.py` calls
    `test_add_parameter_transform()` at module level;
  - the rest: the description of `periodic_vars` does not say that the
    option is refused, and `test_options.py` asserts its text;
    `_get_gp_training_options`'s docstring gives the type `dic`, leaves out
    `function_logger` and `second_fit` and lists `hyp_dict`, which it does
    not read; `init_sobol` keeps two commented-out lines; comments,
    strings and docstrings longer than 79 characters in
    `constraints_check.py`, `grid_functions.py`, `es_search.py` and
    `init_sobol.py`; the comment typo "Re-evalate" in `bads.py`.
- [ ] **The resolution of `Timer`.** `pybads/utils/timer/timer.py` measures
  with `time.time()`, whose resolution on Windows before Python 3.13 is
  about 15.6 ms: for a fast target most evaluations time as 0, and the
  returned `overhead` of two near-identical runs can differ by orders of
  magnitude. `time.perf_counter` fixes it, and changes the returned
  `overhead`, which needs a changelog line.
- [ ] **Coding-agent skill**, after PyVBMC's (`skills/pyvbmc/SKILL.md`): a
  `skills/pybads/SKILL.md` that points a coding agent to the parts of the
  documentation relevant to its task, linked from the README.
- [ ] **gpyreg releases after 1.3.3.** PyBADS's minimum gpyreg
  (`pyproject.toml`) and its CI pin (`GPYREG_PIN`) name one release, 1.3.3
  as of 2026-09-25 ([assessment](results/2026-09-25-gpyreg-1.3.3.md)).
  Each new release moves both, after the population comparison
  (`dev/scripts/population.py compare`) against the current reference
  shows that it has no effect on PyBADS, or explains the one it has.
- [ ] **For gpyreg's maintainers.** gpyreg lists pytest and
  pytest-rerunfailures among its runtime dependencies (`pyproject.toml`,
  every release from 1.0.4 to 1.3.3), so installing PyBADS still installs
  them.
