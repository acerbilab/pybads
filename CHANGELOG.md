# Changelog

All notable changes to PyBADS are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Upgrading from 1.1.0

- PyBADS needs NumPy 2.0 or later, SciPy 1.13 or later and matplotlib 3.9
  or later.
- Results of runs with `specify_target_noise=True` differ from 1.1.0, also
  with a fixed seed.
- With `specify_target_noise=True`, the returned `fval` and `fsd` weight
  the final samples by the precisions that the target returns, and with
  `noise_final_samples=1`, `yval_vec` and `ysd_vec` hold one value.

### Changed

- **Requirements.** PyBADS needs NumPy 2.0 or later, SciPy 1.13 or later
  and matplotlib 3.9 or later (1.1.0 accepted NumPy 1.22.1, SciPy 1.7.3 and
  matplotlib 3.5.1). The `test` extra no longer lists pytest-rerunfailures,
  which the tests do not need (gpyreg 1.3.3 still installs it).

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
- **`noise_size` with user-specified noise.** With
  `specify_target_noise=True`, a scalar `noise_size` made the creation of
  `BADS` fail with `IndexError`. It now gives the warning about
  `noise_size` that an array gave.
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
  cannot estimate counts as no improvement. A failed rebuild is retried at
  the next step with refitted hyperparameters (in the poll, only with
  `poll_training` on). Runs without such a failure give the same results.
- **Messages on the BADS logger.** PyBADS logs every message of a run to the
  `BADS` logger, whose level `display` sets; `display="full"` shows the
  debug messages. The warnings of the GP fits (a
  failed initial fit, failed hyperparameter optimizations) and the debug
  message of a stalling run went to `asyncio`'s logger, and the debug
  messages of failed GP updates to the root logger.
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
