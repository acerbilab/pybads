# PyBADS: open work

Updated 2026-09-30. The next release is 1.5.0 (tag `v1.5.0`), the version
the PI has decided on; "the next release" below means it. The release
waits for the first item of its section, gpyreg 1.4.0, and the last, the
conda-forge recipe, follows its upload to PyPI; otherwise the order of the
items is not a priority. `README.md`, the documentation and the skill on
`dev-next` describe PyBADS 1.5 with gpyreg 1.4.0, and `docs.yml` publishes
the documentation of `main`, so `dev-next` reaches `main` with the release.
Other records name the items by their titles, the
port review's ledger
([results/2026-09-28-port-correctness-review.md](results/2026-09-28-port-correctness-review.md))
among them, so a title stays as it is while its item is open.

## The release of 1.5.0

- [ ] **gpyreg releases after 1.3.3.** PyBADS's minimum gpyreg
  (`pyproject.toml`) is 1.3.3 as of 2026-09-25
  ([assessment](results/2026-09-25-gpyreg-1.3.3.md)), and its CI pin
  (`GPYREG_PIN`) the merge commit of acerbilab/gpyreg#61 on gpyreg's
  `main`, `b44634f`, which carries the kernels' periods (below). Each new
  release moves both, the pin to the release's tag, after the population
  comparison (`dev/scripts/population.py compare`) against the current
  reference shows that it has no effect on PyBADS, or explains the one it
  has. A move is a change for users: an entry in `CHANGELOG.md` and a line
  in its "Upgrading from" list. PyBADS's next release waits for gpyreg's
  next one (PI, 2026-09-28), 1.4.0 (PI, 2026-09-29), which is to hold:
  - `periods` on the ARD kernels (`SquaredExponential`, `Matern`,
    `RationalQuadraticARD`), on gpyreg's `main` since acerbilab/gpyreg#61
    (merge commit `b44634f`), which PyBADS's `periodic_vars` needs:
    `_gp_periods` passes them to the kernel of a run with periodic
    variables, and a run without them builds its kernel as before, so the
    comparison of this release is expected to flag nothing on runs
    without periodic variables. PyBADS 1.5.0 needs the move: under gpyreg
    1.3.3, `BADS` refuses periodic variables with `ImportError`
    (`_gpyreg_takes_periods` in `bads.py`), and the tests of periodic
    variables fail. With the minimum at 1.4.0, the check and its test,
    `test_periodic_vars_need_a_gpyreg_with_periods`, can go. The
    changelog's entry "Requirements" and the line of "Upgrading from
    1.1.0" name the new minimum. The `default` suite holds the six
    configurations of the `periodic` suite (PI, 2026-09-30), which its
    references, the Windows one at 100 seeds and the Linux one at 30,
    lack: `population.py compare` tests only the configurations that both
    populations hold, and lists the others on one line, outside its
    verdict and its exit code. The release's comparison runs the whole
    `default` suite with the release's clone on both platforms, at the
    references' numbers of seeds, and its populations become the new
    references, the periodic configurations included (PI, 2026-09-29),
    and the `default` suite the gate of a change to the handling of
    periodic variables in `AGENTS.md`. On Linux its periodic
    configurations are compared with the "on" arm of
    [experiments/population_periodic_linux_20260928/](experiments/population_periodic_linux_20260928/README.md)
    as well; Windows has no earlier run of them, and runs no "off" arm
    (PI, 2026-09-30);
  - a periodic kernel that costs less (PI, 2026-09-29). At `b44634f`,
    gpyreg's kernel with periods, which computes an `fmod` and a sine of
    every pair of inputs, takes 2.5 to 3.9 times as long as the same calls
    without periods on the runs of the `periodic` suite on Linux, a
    difference of 27 to 38 % of each run (10 to 11 % on `periodic_D2`),
    most of it in the predictions at the ES search's candidates. gpyreg's
    commit `0f27db5`, on `b44634f`, evaluates the trigonometric functions
    once per point instead of once per pair, by mapping each periodic
    coordinate onto a circle, as MATLAB BADS's `covPPERard_fast` does, and
    `91ea28e` after it adds a test. With them the deterministic
    configurations take no longer per evaluation than without
    `periodic_vars`, and the noisy ones 1.2 to 1.5 times as long instead
    of 1.8 to 2.3
    ([results/2026-09-28-periodic-variables.md](results/2026-09-28-periodic-variables.md),
    "Time per evaluation"). `dev/scripts/fingerprint.py` keeps
    `4146a986863602cb` with `0f27db5` (Linux, SciPy 1.17.1, one BLAS
    thread), and the `periodic` suite at 30 seeds flags nothing against
    its Linux reference. The periodic runs take other paths from a
    difference in the kernel's last bits, so the release's comparison of
    those configurations with that reference, above, is a statistical one,
    not an identity, and the reference that the code at 1.4.0 reproduces
    run by run is the release's own population of the `default` suite.
    Both commits are on gpyreg's `main` since acerbilab/gpyreg#62 (merge
    commit `e10120c`), with their evidence in
    [experiments/periodic_kernel_linux_20260929/](experiments/periodic_kernel_linux_20260929/README.md);
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
    `CHANGELOG.md`'s `Unreleased`, which changes with it;
  - the kernels, `predict` and the objective of `fit` computed without
    intermediate arrays or SciPy's layers, every value the same to the
    last bit, on gpyreg's `main` since acerbilab/gpyreg#63 (merge commit
    `4126dbe`) and acerbilab/gpyreg#64 (merge commit `d84bf39`), which
    gives the factorization of the training covariance back to
    `scipy.linalg.cholesky`, whose bits a direct call of LAPACK misses
    under SciPy 1.18. With them and PyBADS's one line that takes the
    priors once per rebuild, PyBADS's own time is 23 to 28 % lower on the
    `profile` suite under SciPy 1.17 and 19 to 27 % lower under SciPy
    1.18, with the fingerprint, the replay and `gpyreg_bitwise.py`
    identical under both, and the oracles' `--against` under SciPy 1.17
    ([results/2026-09-29-bit-identical-speedups.md](results/2026-09-29-bit-identical-speedups.md)).
    These measurements compare with gpyreg `e10120c`, which already holds
    acerbilab/gpyreg#60's gradients, and include that line of PyBADS's.
    The release measures PyBADS's own time with gpyreg 1.4.0 against
    gpyreg 1.3.3, and the move's entry in `CHANGELOG.md` states it; the
    item "Faster Gaussian processes" of "What's new in PyBADS 1.5"
    (`README.md`, `docsrc/source/index.rst`) gives no figure until then;
  - the recommended bounds of the kernels and of `NegativeQuadratic`
    computed without NumPy's `RuntimeWarning`s on inputs without spread,
    every value the same to the last bit, on gpyreg's `main` since
    acerbilab/gpyreg#66 (squash commit `3e56dce`), with the fingerprint, the
    replay and `gpyreg_bitwise.py` identical against its parent, `1260d68`
    (Windows, SciPy 1.18.1, one BLAS thread). `fit` computes those bounds
    even where PyBADS sets every bound, so that under gpyreg up to `1260d68`
    PyBADS's refits on inputs that a poll leaves without spread in a
    coordinate print a log of zero, in every run of `sphere_band_D3`, and a
    refit on one point at D = 1 NumPy's warnings on a sample of one
    ([experiments/one_point_gp_linux_20260928/](experiments/one_point_gp_linux_20260928/README.md)).
    Of 25 runs on Windows (`sphere_band_D3` and `sphere_band_D2_hetero` at
    seeds 0-9, and that experiment's `band1` at seeds 0-4), 15 print them
    under `1260d68` and none under gpyreg#66. The move updates, in its
    commit, what `CHANGELOG.md`'s entry "Targets without spread" says of
    gpyreg's warnings, KD-B6-5's "gpyreg's recommendations warn"
    (`pybads/bads/README.md`), and the filter of `RuntimeWarning` in
    `test_one_point_gp_falls_back_to_fit`, which it no longer needs. The
    helpers still replace a single target by `[0, 1]`, which centres the
    constant mean's recommendation at 0.5 whatever the target, and which
    only a fit on one point reaches (that refit at D = 1, and the fit to
    which `init_and_train_gp` falls back when the posterior with MATLAB
    BADS's definition values fails): the PI left it as it is (2026-09-29).

  At `1893eff`, with the switch off, `main` gives gpyreg 1.3.3's records in
  all 1,080 runs of the `default`, `geometry`, `oned` and `bounds` suites
  on Linux
  ([experiments/gp_switch_linux_20260928/](experiments/gp_switch_linux_20260928/README.md)).
  With gpyreg at `280d8c0` (the tree of the last commit of
  acerbilab/gpyreg#60, `1095742`) and PyBADS at `bec8a57a`,
  `dev/scripts/fingerprint.py` gives 1.3.3's hash, `4146a986863602cb`
  (Linux, NumPy 2.4.6, SciPy 1.17.1, one BLAS thread), as the kernel
  change's bit identity implies. The release's own gate, the comparison
  run with its clone, flags nothing on Linux (below), and takes the
  Windows comparison too. conda-forge's `gpyreg-feedstock` lists
  the three test packages among its run requirements (`recipe/meta.yaml`),
  and its bot merges its version-update PR once a CI that only imports
  gpyreg passes: that PR has them dropped before it merges.

  Since 2026-09-30 gpyreg's `main` holds all of the above, at `3e56dce`
  (the squash commit of #66). The release's comparison runs on that commit
  before the tag: if the tag adds only the date of the release notes, its
  populations are the release's own, and a surprise shows before the
  upload to PyPI. Its clone, listed in `dev/scripts/runs/LOCAL.md` once
  made, is `git clone https://github.com/acerbilab/gpyreg
  dev/scripts/runs/gpyreg/main_3e56dce` followed by `git -C
  dev/scripts/runs/gpyreg/main_3e56dce checkout 3e56dce`. The Linux half
  ran on 2026-09-30, in a cloud container, with PyBADS at `60ad9e0f`
  ([experiments/population_linux_gpyreg140_20260930/](experiments/population_linux_gpyreg140_20260930/README.md)):
  the `default` suite's 24 configurations at 30 seeds flag nothing against
  the Linux reference, whose 540 runs they repeat run by run, and the six
  periodic ones nothing against the periodic reference, their 180 runs
  equal to those of gpyreg `0f27db5`'s gate; no run crashed. The Windows
  half ran on the same day, with PyBADS at `bef26ec2`
  ([experiments/population_gpyreg140_20260930/](experiments/population_gpyreg140_20260930/README.md)):
  the `default` suite's 24 configurations at 100 seeds flag nothing
  against the Windows reference, whose 1800 runs they repeat run by run,
  and no run crashed. gpyreg 1.4.0 was released on 2026-09-30: its tag,
  `v1.4.0`, is `682585f`, which differs from `3e56dce` only in the date of
  the release notes. Still open: the measurement of PyBADS's own time
  against gpyreg 1.3.3 (above), which takes the same clone, and the move
  itself. The commit of the move names
  the release's populations as the references of both platforms in
  `dev/README.md`, and the `default` suite as the gate of periodic
  variables in `AGENTS.md`, in place of the `periodic` suite against
  `population_periodic_linux_20260928`.
- [ ] **The example notebooks' saved outputs.** No CI job runs the
  notebooks of `examples/`, and the saved outputs of the first five predate the port
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
  outputs are regenerated by the run of the notebooks before the release,
  which `AGENTS.md` describes. On 2026-09-29 all six ran to their end on
  PyBADS at `618652d6` with gpyreg at `e10120c`, before and after the
  seeding of Examples 1 to 4, and what they printed agrees with their
  text; the outputs of those runs were not kept. The saved outputs of
  Example 6 (periodic variables) come from gpyreg's development branch
  (`3f1a732`), before any gpyreg release had `periods`; it is rerun with
  the others, with gpyreg 1.4.0.
- [ ] **conda-forge recipe.** The test command of `conda-forge/pybads-feedstock`
  (`recipe/meta.yaml`) passes `--reruns=5` and requires
  pytest-rerunfailures. The tests of 1.1.0, which it runs, are not all
  seeded, so both stay until the first release after 1.1.0, whose tests
  are: drop them in the version-update PR that the feedstock's bot opens
  for that release, before it is merged.

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
