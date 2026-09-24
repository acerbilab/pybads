# Changelog

All notable changes to PyBADS are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since PyBADS 1.0.6.

### Upgrading from 1.0.6

What can stop an existing script, or change what it returns. Each point has
its entry below.

- PyBADS needs Python 3.10 or later and gpyreg 1.3.1 or later.

### Changed

- **Requirements.** PyBADS needs Python 3.10 or later (Python 3.9 has
  reached its end of life) and gpyreg 1.3.1 or later, the gpyreg release its
  tests run against.
- **Test dependencies.** PyBADS no longer lists pytest, pytest-mock and
  pytest-rerunfailures among its dependencies. `pip install "pybads[test]"`
  installs what its test suite needs. (gpyreg 1.3.1 itself still installs
  pytest and pytest-rerunfailures.)

### Fixed

- **Repeated evaluations with user-specified noise.** With
  `specify_target_noise=True`, a run could stop with `ValueError: setting an
  array element with a sequence` after the optimizer evaluated a point a
  second time. The two observations are now merged and the run continues.
