# Changelog

All notable changes to PyBADS are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Upgrading from 1.1.0

- PyBADS needs NumPy 2.0 or later, SciPy 1.13 or later and matplotlib 3.9
  or later.
- Results differ from 1.1.0, also with a fixed seed.
- With `specify_target_noise=True`, the returned `fval` and `fsd` weight
  the final samples by the precisions that the target returns, and with
  `noise_final_samples=1`, `yval_vec` and `ysd_vec` hold one value.
- Without `specify_target_noise`, the returned `fsd` of a noisy run is
  larger by a factor `sqrt(n/(n - 1))`, `n` the number of final samples:
  1.05 at the default 10.
- The returned `iterations` is one more than in 1.1.0.
- `output_fcn` is called as `output_fcn(x, optim_state, state)`, with
  `state` one of `"init"`, `"iter"` and `"done"`, where 1.1.0 called
  `output_fcn(x, "init")` once.
- `gamma_uncertain_interval` is keyword-only and follows `options`: a
  script that passed it as the 8th positional argument of `BADS` passes it
  by name.
- With `uncertainty_handling=False`, a run makes no noise test: it takes one
  evaluation fewer, and a noisy target is optimized as a deterministic one.
- `BADS` raises `ValueError` for a `gp_mean_fun` other than `"const"` or
  `"zero"`, `"negquad"` included.
- `BADS` raises `ValueError` for a `gp_cov_prior` other than `"iso"`,
  `"ard"` included, which 1.1.0 accepted and ignored.

### Changed

- **Requirements.** PyBADS needs NumPy 2.0 or later, SciPy 1.13 or later
  and matplotlib 3.9 or later (1.1.0 accepted NumPy 1.22.1, SciPy 1.7.3 and
  matplotlib 3.5.1). The `test` extra no longer lists pytest-rerunfailures,
  which the tests do not need (gpyreg 1.3.3 still installs it).
- **GP mean function.** `gp_mean_fun` accepts `"const"`, the default, and
  `"zero"`, and `BADS` refuses any other name when it is created. 1.1.0
  accepted ten more: nine stopped the run when the Gaussian process was
  built, and `"negquad"`, a concave mean made for log densities, has the
  wrong shape for a minimizer and could stop the run at a failed fit.
- **Prior over the GP length scales.** `gp_cov_prior` accepts only
  `"iso"`, the default, and `BADS` refuses any other value when it is
  created. 1.1.0 accepted MATLAB BADS's `"ard"`, which PyBADS does not
  implement, and any other value, and then kept the initial prior over the
  length scales for the whole run.
- **Large noise.** `BADS` warns when `noise_size` exceeds e^5, about 148,
  the largest noise standard deviation its Gaussian process can represent
  (the bound of MATLAB BADS): the noise it infers then stays at that bound,
  and the target is better rescaled.

### Fixed

- **User-specified noise.** With `specify_target_noise=True`, the Gaussian
  process treated the noise standard deviations that the target returns as
  variances after its initial fit, so that it underrated the noise wherever
  the standard deviation exceeded 1, and overrated it below 1. It now uses
  their squares, as MATLAB BADS does. Results change in both directions. On
  the 3-D sphere of PyBADS's tests, whose noise standard deviation is
  `2 + sqrt(f)`, the median error of 100 seeded runs of 200 evaluations
  falls from 0.33 to 0.14, and the largest from 2.3 to 0.56. On a 3-D
  ellipsoid with condition number 1e6 and noise standard deviation
  `1 + sqrt(f)`, runs end farther from the minimum: over 90 seeds, the
  median error rises from 0.21 to 0.54.
- **Repeated points with user-specified noise.** With
  `specify_target_noise=True`, a point evaluated again was merged with the
  first earlier evaluation that shared any one of its coordinates, usually
  that of another point, whose value and noise it then changed. It is now
  merged with its own earlier evaluation. On the 3-D ellipsoid above, run
  over 90 seeds on Linux, where 54 runs made such a merge, the median error
  falls from 0.58 to 0.48, and the number of runs with an error of 1 or
  more from 31 to 20.
- **Prior of the GP mean.** At each rebuild of the local Gaussian process,
  the prior over its constant mean is centred at the 90th percentile of the
  training targets, with a width set by their spread, as in MATLAB BADS.
  PyBADS computed that prior and never applied it, so the prior set on the
  initial design held for the whole run. Results change at default
  options, most on ill-conditioned targets: on 3-, 6- and 10-D ellipsoids
  with condition number 1e6, the median error of 30 seeded runs falls by a
  factor of 6 to 110, with as many evaluations or fewer, and on a 6-D
  Rosenbrock function the runs that reach the global minimum end 17 times
  closer to it.
- **Length scales of the GP.** The upper bound of each log length scale of
  the Gaussian process was the largest length scale itself, up to 100,
  instead of its logarithm, as in MATLAB BADS, so that a length scale
  could grow far beyond the size of the search space and the GP could
  treat as flat a direction along which the target varies slowly. Results
  change at default options. On the 3-D ellipsoid with target noise above,
  run over 90 seeds on Linux, the median error falls from 0.46 (with the
  two fixes above) to 0.25, and the number of runs with an error of 1 or
  more from 20 to 8.
- **Final estimate with user-specified noise.** With
  `specify_target_noise=True`, the returned `fval` and `fsd` weight the
  final samples at the returned point (`yval_vec`) by their precisions, the
  inverse squares of the standard deviations that the target returns with
  them (`ysd_vec`), as MATLAB BADS does: `fval` is their precision-weighted
  mean, and `fsd = 1/sqrt(sum(1/ysd_vec**2))`. They were the plain mean of
  the samples and its standard error estimated from their spread, whatever
  standard deviations the target returned. When the target returns the same
  standard deviation for every sample, `fval` is their mean as before, and
  `fsd` is that standard deviation divided by the square root of the number
  of samples. With `noise_final_samples=1`, `fval` and `fsd` are the one
  sample and its standard deviation; before, the incumbent's earlier
  observation was averaged in, and `yval_vec` and `ysd_vec` held two values.
  The returned `x` and the number of evaluations are unchanged.
- **`specify_target_noise` alone.** With `specify_target_noise=True` and
  `uncertainty_handling` left empty, PyBADS turns uncertainty handling on,
  as MATLAB BADS does and as the error message asked; it raised
  `ValueError` unless `uncertainty_handling=True` was set as well.
- **Noisy runs that end within their first iteration.** A run with
  uncertainty handling that ends within its first iteration, for instance
  with `max_iter=1`, returns its result, with the incumbent's observation
  in `yval_vec` and `ysd_vec` set to `None`, instead of raising
  `KeyError: 'yval_vec'`.
- **`noise_size` with user-specified noise.** With
  `specify_target_noise=True`, a scalar `noise_size` made the creation of
  `BADS` fail with `IndexError`. It now gives the warning about
  `noise_size` that an array gave, and `noise_size` is ignored, as the
  warning says. It set the threshold of a check for a Gaussian process that
  explains the data as noise, so that `noise_size=0`, which the warning
  proposed, made every refit of the GP a second fit, and runs ended at
  other points.
- **`noise_size` as a list or a pair.** A list `noise_size`, or MATLAB
  BADS's pair of the base noise standard deviation and the standard
  deviation of the prior over its logarithm, made a noisy run fail with
  `TypeError` or `ValueError`; both are accepted. Without
  `specify_target_noise`, a `noise_size` of 0 or less, which made the run
  fail in the prior of the Gaussian process, raises `ValueError` that names
  `noise_size` when `BADS` is created, as in MATLAB BADS.
- **Final estimate without user-specified noise.** The returned `fsd` of a
  noisy run is the standard error of the final samples computed from their
  standard deviation normalized by `n - 1`, as MATLAB BADS computes it; it
  was normalized by `n`, which made `fsd` smaller by a factor
  `sqrt((n - 1)/n)`, 0.95 at the default 10 samples. The final `fval` and
  `fsd` are recorded in `iteration_history` at the iterate they describe,
  the returned point, instead of the last iterate.
- **Iteration count.** The returned `iterations` and the iteration column
  of the display count from 1, as in MATLAB BADS: a run that ends on
  `max_iter` reports `max_iter` iterations. They were one lower.
- **Output function.** `output_fcn` is called as in MATLAB BADS, as
  `output_fcn(x, optim_state, state)` at the start (`state="init"`), after
  each poll (`"iter"`) and at the end (`"done"`), with the incumbent `x` in
  the original space and a copy of the internal state `optim_state`; a
  true return value stops the run. It was called once, at the start, as
  `output_fcn(x, "init")`, and a true return value raised
  `UnboundLocalError`.
- **Arguments in MATLAB BADS's order.** `BADS(fun, x0, lb, ub, plb, pub,
  non_box_cons, options)` passes `options`. They went to the undocumented
  argument `gamma_uncertain_interval`, which stood before `options`, and
  were ignored, the seed included. `gamma_uncertain_interval`, the
  multiplier of the uncertainty interval of Sto-BADS (`stobads=True`), is
  keyword-only and documented.
- **Sto-BADS poll.** With `stobads=True`, a poll succeeds when one of its
  points succeeds, as in Sto-MADS, and moves the incumbent to that point;
  before, the outcome of the last polled point decided, so that a success
  followed by other points was lost and the mesh contracted. With
  `opp_stobads`, a poll without a success moves to its best point when any
  of its points is uncertain, not only when the last one is.
- **Search without a candidate.** A search that leaves no candidate, for
  instance under a `non_box_cons` that does not always give the same answer
  for a point, counts as a failed search, as in MATLAB BADS; the run
  stopped with `UnboundLocalError` or `IndexError`.
- **`uncertainty_handling=False`.** With `uncertainty_handling=False`, the
  starting point is not evaluated a second time to test for noise, as in
  MATLAB BADS; the test ran unless uncertainty handling was on, and a
  target that it found noisy was optimized as a noisy one although the
  option declared it deterministic. The test runs when
  `uncertainty_handling` is left empty.
- **Random starting point.** Without a finite `x0`, the starting point is
  drawn uniformly in the transformed plausible box, as in MATLAB BADS:
  log-uniformly in the original space for a variable on a log scale (all
  its bounds positive, and `pub/plb >= 10`). It was drawn uniformly in the
  original plausible box, which put most starting points in the upper
  decade of such a variable.
- **One function evaluation.** A run with `max_fun_evals=1` returns the
  starting point, where it raised `KeyError: 'eff_starting_points'`. As in
  MATLAB BADS, the starting point is evaluated a second time when
  `uncertainty_handling` is left empty, to test for noise.
- **`kde1d` with NumPy 2.** `pybads.stats.kde1d` no longer raises
  `AttributeError` under NumPy 2.
- **Termination message.** A run that ends because the mesh size fell below
  `tol_mesh` says so; the message spoke of the change in the function
  value.
- **Failed GP updates.** A run no longer stops with `LinAlgError`
  ("Singular matrix for L Cholesky decomposition") when a Gaussian-process
  update fails while adding a point, predicting the optimization target or
  rebuilding the local GP. The run carries on, as in MATLAB BADS: a point
  the GP could not take is included when the local GP is rebuilt at the
  next step, a target that cannot be predicted under the best
  hyperparameters is predicted from the current GP, and a point the GP
  cannot estimate counts as no improvement, with Sto-BADS
  (`stobads=True`) too. A failed rebuild is retried at the next step with
  refitted hyperparameters (in the poll, with `poll_training` on or in the
  first iteration), where MATLAB BADS refits only when its check of the
  GP's predictions calls for it. Runs without such a failure give the same
  results.
- **Re-evaluation of the iterates in noisy runs.** At the end of each
  iteration of a noisy run, PyBADS re-estimated the target at the iterates
  and then went on with a Gaussian process of the iteration history as its
  working GP; the history kept that same GP, which the next refits
  changed, so that some iterates were re-estimated under another
  iteration's hyperparameters. As in MATLAB BADS, each iterate is now
  re-estimated with the hyperparameters recorded at its iteration, the
  working GP is kept, and `iteration_history` keeps its GPs as they were
  recorded. Noisy runs end earlier, more often on `tol_fun`: on the noisy
  targets of PyBADS's benchmark, 12 to 18% fewer evaluations, with no
  significant change of the error (over 90 seeds on a 3-D ellipsoid with
  inferred noise, a median error of 0.080 against 0.076).
- **Messages on the BADS logger.** PyBADS logs every message of a run to the
  `BADS` logger, whose level `display` sets; `display="full"` shows the
  debug messages. The warnings of the GP fits (a failed initial fit, failed
  hyperparameter optimizations) and the debug message of a stalling run went
  to `asyncio`'s logger, and the debug messages of failed GP updates to the
  root logger. PyBADS's warnings no longer come with the
  `DeprecationWarning` of `Logger.warn`.
- **Warnings on Python 3.12 and later.** Importing PyBADS no longer emits
  `SyntaxWarning: invalid escape sequence` (from a docstring), and a run no
  longer emits a `DeprecationWarning` for `~` applied to a `bool`, an
  operation that later Python versions remove.
- **Shipped tests.** The tests that ship with PyBADS
  (`pytest --pyargs pybads`) give the same result on each run on a given
  machine: every test whose outcome depends on random draws is seeded. Each
  optimization test checks a tolerance of its own, tighter than before for
  most. Two tests of the poll, which pytest did not collect, are renamed so
  that it does, and a third is added. The module `pybads.testing.run_tests`,
  which failed on import, and six data files that no test read are no
  longer installed.
- **Refits without poll training.** With `poll_training=False`, a poll
  after the first iteration skipped a refit of the Gaussian process that was
  due but counted it as made, which delayed the next refit of the search
  (MATLAB BADS does the same). The poll now leaves that refit to the search,
  and its stopping rule reads the calibration of the GP as it is. Results
  change only with `poll_training=False`.
- **Fixed noise.** `fit_lik=False`, a fixed noise level, which MATLAB BADS
  does not support either, is refused when `BADS` is created, with MATLAB
  BADS's message "Fixed noise not supported"; 1.1.0 stopped the run at its
  start with gpyreg's "Unknown hyperprior type delta".
- **Initial fit of the GP.** A run whose initial Gaussian-process fit
  keeps failing stops after 10 tries with a `RuntimeError` that says so;
  1.1.0 retried without end.
- **Targets without spread.** A run no longer stops with `ValueError` from
  gpyreg when the targets of a Gaussian-process fit are all equal: a target
  flat on the initial design (a penalty plateau over the plausible box), or
  a feasible region (`non_box_cons`) so thin that the initial design leaves
  the GP a single point. The prior of the GP mean then takes the width 1,
  and a rebuild keeps the previous centre of the prior of the output scale.

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
