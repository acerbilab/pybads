# Changelog

All notable changes to PyBADS are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.5.1] - 2026-10-06

Changes since PyBADS 1.5.0.

### Upgrading from 1.5.0

- An exception that the objective function raises reaches the caller with
  its `args` as raised, where 1.5.0 appended an element to them; from
  Python 3.11 on, it carries a note with the point at which it was raised.
- Messages no longer begin with identifiers such as `bads:pbUnspecified`
  or `FunctionLogger:InvalidFuncValue`: a script that matched them matches
  the exception's type or the message's text instead ("Messages" under
  Fixed).

### Fixed

- **Messages.** The errors and warnings about a value or a noise SD that
  the objective returns, about the bounds and about the starting point
  say what is wrong, at which point or variables, and what is expected,
  and each error is printed once; failures that a run recovers from, such
  as a failed Cholesky decomposition while fitting the Gaussian process,
  are no longer logged as warnings. Without final samples, a noisy run
  reports the observation at its returned point.
- **Documentation.** Notebook launch and edit links point to their
  repository sources. The FAQ and API reference clarify noisy results
  with few or no final samples, timing and function logger return values.

## [1.5.0] - 2026-10-06

Changes since PyBADS 1.1.0. Most of the fixes make PyBADS follow MATLAB
BADS 1.1.3, the reference implementation, with which it was compared line
by line: the [catalogue of deliberate
differences](https://github.com/acerbilab/pybads/blob/main/pybads/bads/README.md)
says where it still differs, and why.

### Upgrading from 1.1.0

- PyBADS needs gpyreg 1.4.0 or later, NumPy 2.0 or later, SciPy 1.13 or
  later and matplotlib 3.9 or later ("Requirements" under Changed).
- Results differ from 1.1.0, also with a fixed seed.
- `BADS` checks the values of many options when it is created and raises
  `ValueError` for some that 1.1.0 accepted, such as `30.5` for
  `max_fun_evals`, `inf` for `accelerate_mesh_steps`, or `1` or `"off"` for
  a boolean option ("Checks of the options" under Changed).
- Setting one of the 66 options that had no effect in 1.1.0 and that MATLAB
  BADS does not have, such as `min_iter` or `gp_cov_fun`, raises
  `ValueError` ("Options without effect" under Removed).
- A user value of `None` stands for the option's default, as an empty
  value does in MATLAB BADS: `nonlinear_scaling=None`, for instance, keeps
  the log transform on, where 1.1.0 turned it off.
- The eighth positional argument of `BADS` is `options`, as in MATLAB BADS,
  where 1.1.0 took it as `gamma_uncertain_interval` and ignored the
  options; `gamma_uncertain_interval` is keyword-only.
- `output_fcn` is called as `output_fcn(x, optim_state, state)`, at the
  start (`"init"`), after each poll (`"iter"`) and at the end (`"done"`),
  and a true return value stops the run; 1.1.0 called `output_fcn(x,
  "init")` once.
- The result's `success` is False when a run ends on `max_fun_evals` or
  `max_iter`, or is stopped by `output_fcn`, where 1.1.0 always reported
  True, and its `iterations`, like the display's iteration column, counts
  from 1, as in MATLAB BADS, one more than in 1.1.0.
- A second call of `optimize()` on the same `BADS` object raises
  `RuntimeError`, where 1.1.0 ran again from the state that the first call
  had left, past `max_fun_evals`: create a new `BADS` object for each run.
- With `specify_target_noise=True`, the returned `fval` and `fsd` are the
  precision-weighted mean of the final samples and its SD, where 1.1.0
  ignored the SDs that the target returns, and with `noise_final_samples=1`,
  `yval_vec` and `ysd_vec` hold one value.
- Without `specify_target_noise`, the returned `fsd` of a noisy run
  normalizes the SD of the `n` final samples by `n - 1`, not `n`, which
  makes it larger by `sqrt(n/(n - 1))`, 1.05 at the default 10; with
  `noise_final_samples=1`, `yval_vec` has shape (2,), not (2, 1).
- With `uncertainty_handling=False`, a run makes no noise test, and a noisy
  target is optimized as a deterministic one. With it left empty, the
  noise test compares the two values at the start with the option
  `tol_noise`, whose default is `sqrt(eps) * tol_fun`, where it was
  `eps * tol_fun`: a difference at most that large counts as no noise.
- `x0` and the plausible bounds are used as given ("Bounds and the starting
  point" under Fixed).
- A value or a noise SD that the target returns and that is not a finite
  real number raises `ValueError`, a complex one included, even when its
  imaginary part is zero.
- `pybads.init_functions.init_sobol` returns the number of points of its
  design as its second value, not its base-2 logarithm, and requires `lb`
  and `ub`; a run's function logger holds `S` only with
  `specify_target_noise=True`, and `FunctionLogger.add` checks a value and
  an SD as those of the target, the SD required at uncertainty level 2.
  `pybads.search.ESSearchCMA` and `FunctionLogger.y_max` are removed
  ("Removed").

### Added

- **Periodic variables.** `periodic_vars` lists the indices, from 0, of
  the variables that are periodic, such as angles: BADS wraps each of them
  around its hard bounds, which need to be finite, and the Gaussian process
  is periodic along it.
  [Example 6](https://acerbilab.github.io/pybads/_examples/pybads_example_6_periodic_variables.html)
  and the
  [FAQ](https://acerbilab.github.io/pybads/faq.html#faq-does-pybads-support-periodic-variables-such-as-angles)
  show how to set it up.
- **Evaluations made before the run.** `BADS(...,
  precomputed_evaluations=(X, y))`, or `(X, y, y_sd)` with
  `specify_target_noise=True`, gives a run evaluations of the target made
  before it. The run evaluates `x0` and its design all the same, but for
  the points of the design that they hold; those nearest the incumbent
  join its Gaussian process from the first poll on, none counts against
  `max_fun_evals`, and the result counts them in
  `precomputed_observations`, and their distinct points in
  `precomputed_locations`.
- **Fixed variables.** A variable whose four bounds, `lb`, `ub`, `plb` and
  `pub`, are equal is fixed at that value, where 1.1.0 raised `ValueError`:
  BADS optimizes the others, the defaults that depend on D count only
  those, and the target, `non_box_cons`, `output_fcn` and the result see
  points of all the variables. The
  [FAQ](https://acerbilab.github.io/pybads/faq.html#faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value)
  shows an example.
- **FAQ and a coding-agent skill.** The documentation has a [page of
  frequently asked questions](https://acerbilab.github.io/pybads/faq.html),
  and `skills/pybads/SKILL.md` in the repository points a coding agent to
  the documentation relevant to its task.
- **Tips.** A run with the iteration display may print a short tip, with
  a link to the documentation, before its first iteration line: the first
  such run of a Python session, then every third, each tip at most once
  per session. `options={"show_tips": False}` turns them off.
- **Update reminders.** In an interactive session, a run of a release more
  than a year old shows, in the place of a tip, a note that a newer version
  may exist, at most three times for each installed version, without a
  network request (`options={"show_tips": False}` turns it off with the
  tips). `pybads.check_for_updates()` asks PyPI whether a newer version
  exists and gives the command that installs it.
- **Stage times.** A `BADS` object's `iteration_history["timer"]` and
  `optim_state["stage_times"]` hold the seconds that its run has spent in
  each of its stages; their format can change in any release.

### Changed

- **Speed.** PyBADS's own computations, a run's time besides the target's
  evaluations, take 44 % less time than in 1.1.0 over seven benchmark
  problems at 30 seeds each, which is 1.8 times as fast, and 19 to 54 %
  less per problem (Windows, one BLAS thread, against 1.1.0 with gpyreg
  1.3.3), with equal or better results. Each evaluation costs PyBADS less
  time, partly because gpyreg 1.4.0 computes the same Gaussian processes
  as 1.3.3, to the last bit, faster, and runs end after fewer evaluations
  ([the measurement](https://github.com/acerbilab/pybads/blob/main/dev/experiments/own_time_v110_20261001/README.md)).
- **Requirements.** PyBADS needs gpyreg 1.4.0 or later, NumPy 2.0 or later,
  SciPy 1.13 or later and matplotlib 3.9 or later, and installing it no
  longer installs pytest, pytest-rerunfailures and numdifftools.
- **Seeded initial design.** `random_seed` decides the initial design, as
  it decides every other random draw of a run; in 1.1.0 every start inside
  the plausible box gave the same design at a given D, whatever the seed.
- **Checks of the options.** `BADS` checks these options when it is
  created and raises `ValueError`, naming the option, for any other value;
  1.1.0 ran some of these as meant, such as `1` for `True`, and most with
  wrong results or until an unrelated error. An integer option takes a
  float that is a whole number; a boolean, a string or a complex number is
  refused for a number, and so is an array, except in `noise_size` and
  `sqrt_beta`.
  - `max_fun_evals`, `max_iter`, `tol_stall_iters`: a positive integer or
    `inf`; `search_n_try`, `noise_final_samples`: an integer at least 0;
  - `n_search`, `n_search_iter`, `accelerate_mesh_steps`: a positive
    integer, `n_search_iter` at most `n_search`;
  - `tol_mesh`: a positive finite number; `tol_fun`: a positive number at
    most e^6, about 403; `improvement_quantile`: between 0 and 1, both
    excluded;
  - `hedge_gamma`: from 0 to 1/n, n the number of search methods in
    `search_method` (2 by default); `hedge_beta`: a finite number at least 0;
    `hedge_decay`: from 0 to 1;
  - `noise_size`: a number, or MATLAB BADS's pair of the base noise SD and
    the SD of the prior over its logarithm; unless `specify_target_noise`
    is set, which ignores it, a positive finite base and a positive SD;
  - `search_method`: a non-empty list of pairs naming `"ES-wcm"` or
    `"ES-ell"`; `search_acq_fcn`: a pair `("acq_LCB", sqrt_beta)`, with
    `sqrt_beta` `None`, a positive finite number or a callable that returns
    one (checked at each search);
  - `gp_mean_fun`: `"const"` or `"zero"`; `gp_cov_prior`: `"iso"`;
    `acq_hedge`: `False`; `fit_lik`: `True`; `f_vals`: no finite value;
    `fun_values`: empty (`precomputed_evaluations` takes its evaluations);
  - `uncertainty_handling` and the options whose default is `True` or
    `False`, except `plot`: `True` or `False`.
- **Move to an earlier iterate in noisy runs.** When the re-estimation at
  the end of an iteration of a noisy run finds an earlier iterate better
  by more than `tol_fun`, the incumbent moves to that iterate, where 1.1.0,
  like MATLAB BADS, moved only its value.

### Deprecated

- **Sto-BADS.** `stobads`, with `opp_stobads`,
  `stobads_frame_size_scaling_power` and the argument
  `gamma_uncertain_interval`, is deprecated and may be removed in a future
  release: Sto-BADS is experimental and has not been found to improve noisy
  runs; a noisy target takes `uncertainty_handling=True`, without
  Sto-BADS. Setting `stobads=True` logs a warning.

### Removed

- **Options without effect.** The 66 options that had no effect in 1.1.0
  and that MATLAB BADS does not have, most of them leftovers of PyVBMC, are
  removed; the 12 without effect that are named after options of MATLAB
  BADS, such as `poll_method`, stay, described as unused.
- **`ESSearchCMA` and `y_max`.** `pybads.search.ESSearchCMA`, a CMA-ES
  search that no `search_method` selected and that failed when called, is
  removed, and so is `FunctionLogger.y_max`, which 1.1.0 held at -inf.

### Fixed

#### Noisy targets

- With `specify_target_noise=True`, the Gaussian process takes the squares
  of the noise SDs that the target returns as its noise variances, where it
  took the SDs themselves. Results change in both directions: runs on a
  3-D sphere with target noise end closer to the minimum, and runs on an
  ill-conditioned 3-D ellipsoid with target noise farther from it.
- With `specify_target_noise=True`, a point evaluated again is merged with
  its own earlier evaluation in the log, where it was merged with the first
  one that shared any of its coordinates, usually another point's.
- At the end of each iteration of a noisy run, every earlier iterate is
  re-estimated under the hyperparameters of its own iteration, where some
  were re-estimated under a later iteration's: noisy runs take 12 to 18 %
  fewer evaluations on PyBADS's benchmark, with no significant change of
  their errors.
- A noisy run that ends within its first iteration returns its result, and
  `specify_target_noise=True` alone turns uncertainty handling on, where
  each raised an error.
- `noise_size` as a list or a pair no longer makes a noisy run fail, and
  with `specify_target_noise=True` a scalar `noise_size` is ignored, with a
  warning, where it made the creation of `BADS` fail.
- In a noisy run, the search hedge rewards a search with its expected
  improvement, where a misplaced parenthesis over-rewarded a search whose
  point was worse than the incumbent, the more so the worse the point.

#### The Gaussian process

- At each rebuild of the local Gaussian process, the prior over its mean is
  centred at the 90th percentile of the training targets, as in MATLAB
  BADS, a prior that 1.1.0 computed and never applied. Runs on
  ill-conditioned targets end much closer to the minimum: on 3-, 6- and
  10-D ellipsoids with condition number 1e6, the median error falls by a
  factor of 6 to 110.
- The upper bound of each log length scale of the GP is the logarithm of
  min(100, 10 times the variable's range), where it was that number
  itself, which let the GP take a direction along which the target varies
  slowly to be flat.
- The check of the GP's predictions that calls for a refit follows MATLAB
  BADS in its count of the predictions, its chi-square quantiles and its
  scaling of the errors by the SD of an observation; it failed on GPs that
  MATLAB BADS accepts.
- A failed update of the GP no longer stops the run with `LinAlgError`:
  the run carries on and rebuilds the GP at its next step.
- A failed refit of the hyperparameters is retried as in MATLAB BADS and,
  if every retry fails, keeps its best starting hyperparameters, where
  1.1.0 stopped the run at the fifth failure; the initial fit, which 1.1.0
  retried without end, stops the run with a `RuntimeError` after 10
  failures.
- A GP on targets that are all equal, on one point or on two points, as a
  penalty plateau or a thin feasible region can give, no longer stops the
  run with gpyreg's `ValueError`.
- The GP follows MATLAB BADS in smaller details that change results: its
  mean is unbounded, the prior over its output scale takes the targets' SD
  normalized by N - 1, in deterministic runs the prior over its noise
  follows the mesh size, and after a move it is rebuilt as often as in
  MATLAB BADS, not at every later step.
- Smaller changes of PyBADS's own, which MATLAB BADS does not share,
  change results too: the noise test no longer counts in the schedule of
  the GP's fits, at D = 1 the GP chooses its training points with its
  fitted length scale, where it took 1, and with `poll_training=False` a
  poll no longer counts a refit that it skips as made.

#### The search and the poll

- The initial design, the search and the poll no longer evaluate again a
  point already evaluated, and a point halfway between two points of the
  search grid goes to the one farther from zero, as in MATLAB BADS.
- The evolution-strategy search follows MATLAB BADS in the covariance of
  ES-wcm, the offspring of each candidate, the order of ties, the scale of
  its generations (with `n_search_iter` of 3 or more) and the floor
  `search_factor_min` of its scale after failed searches; in a variable
  without bounds, ES-ell's scale, which was constant, follows the GP's
  length scales.
- The search and the poll skip candidates whose acquisition value is NaN,
  where they evaluated the first one, and a search that leaves no
  candidate counts as failed, where it stopped the run.
- The poll's early stop takes the D points of highest probability of
  improvement, not the last D + 1, and the accelerated reduction of the
  mesh starts one iteration earlier, as in MATLAB BADS.
- With `stobads=True`, a poll succeeds when any of its points succeeds,
  where its last point decided, and with `opp_stobads` an uncertain search
  moves the incumbent only to a point estimated better.
- Runs with `hedge_gamma=0`, `uncertain_incumbent=False` or
  `sloppy_improvement=False` no longer stop with an error.

#### Bounds and the starting point

- `x0` and the plausible bounds are used as given, as in MATLAB BADS, where
  1.1.0 moved them 0.1 % of the range away from the hard bounds and widened
  the plausible box to a start near a hard bound. Omitted plausible bounds
  are the hard bounds, with the warning `bads:pbUnspecified`.
- A problem that mixes bounded and unbounded variables, or has a variable
  bounded on one side only, is accepted, where 1.1.0 refused it; the
  plausible bound on an infinite side is required.
- A scalar bound stands for the same bound in every variable, and integer
  bounds are taken as floats, which 1.1.0 mapped to the wrong range on a
  log scale; bounds from about 1e10 in magnitude are no longer refused.
  `VariableTransformer` used directly takes bounds as lists, scalars or
  arrays of shape (D,).
- Without a finite `x0`, the start is drawn uniformly in the transformed
  plausible box, so log-uniformly for a variable on a log scale, and a
  start that violates `non_box_cons` is drawn again, up to 1000 draws in
  all.

#### Termination and the result

- The result's `status` is MATLAB BADS's exit flag, where
  `result["status"]` raised `KeyError`.
- A run whose `max_fun_evals` is 1, or no larger than its initial design,
  returns its result, where it could raise an error, and makes at most
  `max_fun_evals` evaluations, except that a budget of 1 with the noise
  test takes 2.
- The result holds the `fun` and `non_box_cons` passed to `BADS`, not
  copies, which failed for a target that holds a lock or an open file.
- `total_time` and `overhead` are timed with `time.perf_counter`, not
  `time.time`, which timed fast evaluations as 0 on Windows before Python
  3.13, and `overhead` counts the final samples in the target's time.

#### Messages, warnings and errors

- `display` takes MATLAB BADS's levels, `"notify"`, `"final"` and `"iter"`
  among them, where 1.1.0 showed everything for any value but `"off"` and
  `"full"`, and every message of a run goes to the `BADS` logger.
- PyBADS no longer emits `SyntaxWarning` or `DeprecationWarning` on Python
  3.12 and later; NumPy's `RuntimeWarning`s on a GP whose points have no
  spread in a coordinate or that holds a single point; the warning of the
  Shapiro-Wilk test when the GP predicted its last evaluations exactly; or
  an overflow
  `RuntimeWarning` for a variable with a bound above about 700 beside one
  on a log scale. A run leaves NumPy's error handling as it found it.
- `BADS` checks, when it is created, that `non_box_cons`, given N points,
  returns a NumPy array of shape (N,) or (N, 1), and raises `ValueError`
  otherwise; 1.1.0 failed on other outputs with unrelated errors, then or
  during the run, and on (N, 1) too.
- `pybads.stats.kde1d` no longer raises `AttributeError` under NumPy 2.

## [1.1.0] - 2026-09-25

Changes since PyBADS 1.0.6.

### Upgrading from 1.0.6

- PyBADS needs Python 3.10 or later and gpyreg 1.3.3 or later.
- Results differ from 1.0.6, also with a fixed seed.
- `random_seed` no longer seeds NumPy's global random state.
- The seed is read when the `BADS` object is created: a `random_seed` set in
  `options` afterwards, or `np.random.seed` called between creation and
  `optimize()`, does not fix a run.
- `random_seed` refuses a float that is not a whole number, and a string,
  with a `TypeError`.

### Added

- **Seeded runs through a random generator.** Every random draw of a run
  comes from one `numpy.random.Generator`, `bads.rng`, which `BADS` creates
  from the `random_seed` option when the object is created. `random_seed`
  takes what `numpy.random.default_rng` takes, such as an integer or a
  `SeedSequence`, or a `Generator`, which `BADS` uses as given; a float that
  is a whole number is converted to an integer, and another float, or a
  string, raises a `TypeError`. A seeded run neither reads nor writes NumPy's
  global random state, so on one machine two runs with the same seed give
  the same result whatever else draws from that state. With
  `random_seed=None`, the default, the generator is derived from NumPy's
  global random state when the `BADS` object is created, so
  `np.random.seed(...)` before that fixes a run. The draws, and so the
  results, differ from those of 1.0.6, also with the same seed.
  - `random_seed` does not seed the target: a target that draws random
    numbers of its own, for example from `np.random`, has to be seeded
    separately. In 1.0.6 every draw came from NumPy's global random state,
    which `random_seed` seeded, the target's included.
  - In 1.0.6, `optimize()` reseeded NumPy's global random state from the
    option, so a `random_seed` set in `options` after creating the `BADS`
    object, or `np.random.seed(...)` called before `optimize()` with
    `random_seed=None`, fixed a run; neither does.
  - `optimize_result["random_seed"]` holds the seed when it is an integer
    (or a float converted to one), and `None` otherwise.

### Changed

- **Requirements.** PyBADS needs Python 3.10 or later (Python 3.9 has
  reached its end of life) and gpyreg 1.3.3 or later, the gpyreg release its
  tests run against. From gpyreg 1.3.2 on, the predictions of a GP with very
  small noise are more accurate. BADS's GP reaches that regime on many
  deterministic targets; on several benchmark problems runs then end closer
  to the minimum, some by orders of magnitude, and on the others the results
  change little.
- **Test dependencies.** PyBADS no longer lists pytest, pytest-mock and
  pytest-rerunfailures among its dependencies. `pip install "pybads[test]"`
  installs what its test suite needs. (gpyreg 1.3.3 itself still installs
  pytest and pytest-rerunfailures.)

### Fixed

- **Repeated evaluations with user-specified noise.** With
  `specify_target_noise=True`, a run could stop with `ValueError: setting an
  array element with a sequence` after the optimizer evaluated a point a
  second time. The run now continues, with the two observations merged.
