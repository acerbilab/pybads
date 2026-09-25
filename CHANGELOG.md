# Changelog

All notable changes to PyBADS are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since PyBADS 1.0.6.

### Upgrading from 1.0.6

What can stop an existing script, or change what it returns. Each point has
its entry below.

- PyBADS needs Python 3.10 or later and gpyreg 1.3.3 or later.
- Results differ from 1.0.6, also with a fixed seed.
- `random_seed` no longer seeds NumPy's global random state.
- `random_seed` refuses a float that is not a whole number, and a string,
  with a `TypeError`.

### Added

- **Seeded runs through a random generator.** Every random draw of a run
  comes from one `numpy.random.Generator`, `bads.rng`, which `BADS` creates
  from the `random_seed` option. `random_seed` takes what
  `numpy.random.default_rng` takes: an integer, a `SeedSequence`, or a
  `Generator`, which is used as given; a float that is a whole number is
  converted to an integer, and another float, or a string, raises a
  `TypeError`. On one machine, two runs with the same seed give the same
  result whatever happens to NumPy's global random state, which a seeded run
  neither reads nor writes. With `random_seed=None`, the default, the
  generator is derived from NumPy's global random state, so
  `np.random.seed(...)` before creating the `BADS` object still fixes a run,
  as in 1.0.6. In 1.0.6 every draw came from NumPy's global random state,
  which `random_seed` seeded, so a target that drew from that state shifted
  the optimizer's draws. The draws differ from those of 1.0.6, and so do the
  results, also with the same seed.
  - `random_seed` does not reach inside the target. A target that draws
    random numbers of its own, for example from `np.random`, has to be seeded
    separately; in 1.0.6, `random_seed` seeded it too, through NumPy's global
    random state.
  - `optimize_result["random_seed"]` holds the seed when it is an integer
    (or a float converted to one), and `None` when it is a `SeedSequence` or
    a `Generator`.

### Changed

- **Requirements.** PyBADS needs Python 3.10 or later (Python 3.9 has
  reached its end of life) and gpyreg 1.3.3 or later, the gpyreg release its
  tests run against. From gpyreg 1.3.2 on, the predictions of a GP with very
  small noise are more accurate; on deterministic targets, where BADS's GP
  reaches that regime near the optimum, runs end closer to the minimum.
- **Test dependencies.** PyBADS no longer lists pytest, pytest-mock and
  pytest-rerunfailures among its dependencies. `pip install "pybads[test]"`
  installs what its test suite needs. (gpyreg 1.3.3 itself still installs
  pytest and pytest-rerunfailures.)

### Fixed

- **Repeated evaluations with user-specified noise.** With
  `specify_target_noise=True`, a run could stop with `ValueError: setting an
  array element with a sequence` after the optimizer evaluated a point a
  second time. The run now continues, with the two observations merged.
