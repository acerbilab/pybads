# Plan: tooling, CI, seeded runs and the benchmark harness

Created: 2026-09-24
Status: DONE (2026-09-25); approved 2026-09-24, open questions settled at their defaults

## Summary

Bring PyBADS's repository tooling to the level of PyVBMC's (the
`dev-port-review` branch of `../pyvbmc` is the reference): line endings and
formatting, packaging and a changelog, a shared CI test workflow with gpyreg
pinned, a release workflow, generated example scripts, seed tests, a
benchmark suite with a population comparison, and random-number generator
objects in place of NumPy's global stream. Then assess gpyreg 1.3.3 for
PyBADS with that comparison and require it. Each
change that can move numerical results is made alone and checked against a
recorded reference.

## Context: gpyreg 1.3.2

gpyreg 1.3.2 is prepared on the gpyreg branch `w6-leftovers` (worktree
`../gpyreg-w6-leftovers`, not pushed when this plan was written); its
changes are listed in that worktree's `docsrc/source/release_notes.rst`,
section "1.3.2 (unreleased)". Two of them reach PyBADS: where the smallest
noise variance of a GP is below `1e-6`, the predictive covariance is formed
from a Cholesky factor, and PyBADS's GPs reach that regime at default
options (the noise lower bound corresponds to a variance of about
`1.35e-7`); and several invalid inputs are now refused with `ValueError`.
PyBADS requires only `gpyreg >= 0.1.0`, so its users get 1.3.2 on release.
The PyVBMC maintainers' release plan (`../pyvbmc/dev/TODO.md`, the item that
releases gpyreg 1.3.2) runs PyBADS against the branch before the tag and
asks PyBADS for: the suite's outcomes on 1.3.1 and on the branch; each
difference with its cause (the prediction change, a refusal, or chance);
whether anything in 1.3.2 should change before the tag; and PyBADS's
proposals, including on two issues found in gpyreg's review, the `"delta"`
noise prior of the known-noise path and the bound inversion of
`_robust_gp_fit_` (Phase 9, step 5). Phase 0 answers the first two
questions early; Phase 9 answers all of them.

Amended on 2026-09-24: gpyreg 1.3.2 was released during Phase 7 and gpyreg
1.3.3 after it (tag `v1.3.3`, `98ab5a4`, acerbilab/gpyreg#54; release notes
in gpyreg's `docsrc/source/release_notes.rst`, section "1.3.3
(2026-09-24)", with the 1.3.2 section corrected in place). 1.3.3 makes a
`fit`, `update` or `set_hyperparameters` that raises restore the GP's data,
bounds, priors and posteriors before re-raising (1.3.2 left the new data
beside posteriors that did not match them); makes a GP whose `s2` holds a
number work again where 1.3.2's count checks raised; and refuses NaN
hyperparameters in `update` before it changes anything. Every `fit` and
`update` that completes gives the same result as in 1.3.2, bit for bit, by
the gpyreg maintainers' account. Phase 9 therefore assesses 1.3.3, the
release PyBADS would require, and moves PyBADS's minimum and CI pin to it
when its comparison shows no flag, or, as it turned out, on the user's
decision when the flags are explained.

## Scope

- **In scope**: the phases below.
- **Out of scope**: the bug hunt and the verification against MATLAB BADS
  (`dev/TODO.md`, deferred); the candidate defects of
  `dev/results/2026-09-23-codebase-survey.md` other than the crash fixed in
  Phase 5 (new candidates found on the way are recorded there, not fixed);
  the user-facing agent skill `skills/pybads/SKILL.md` (after this plan);
  exact step-by-step replay, numerical oracles and the profiler (later, see
  Decisions); the conda-forge recipes (a TODO for the next release,
  Phase 2).

## Conventions for every phase

- Work on the development branch (Open Question 4), from the repository
  root, in Git Bash, with the project venv (`.venv/Scripts/python.exe` on
  Windows, `.venv/bin/python` elsewhere), written below as `python`. The
  agent's shell does not activate the venv: commands name the interpreter
  by its path, and a bare `python` there is the system installation.
- Commits follow conventional commits and end with the `Co-Authored-By:`
  line; never a `Claude-Session:` trailer (`AGENTS.md`). One or more
  commits per phase; no push without the user's go. Code is committed
  before any run whose records name the commit (Phases 7–9).
- One heavy process at a time: the test suite, the population runs and the
  example scripts never run concurrently. Long runs write unbuffered logs to
  the gitignored `dev/scripts/runs/` (`mkdir -p dev/scripts/runs` first;
  `python -u ... > dev/scripts/runs/<name>_$(date +%s).log 2>&1`) and are
  read from the log.
- If a check contradicts an assumption a step rests on, stop and report the
  mismatch to the user rather than improvise.
- gpyreg is selected explicitly for every run that serves as evidence (the
  fingerprint, the populations, the suite comparisons): `PYTHONPATH` names a
  clone checked out at the release tag, `dev/scripts/runs/gpyreg/v1.3.1`,
  `v1.3.2` or `v1.3.3` there (gitignored; listed with the command that
  recreates them in `dev/scripts/runs/LOCAL.md`). The venv's editable
  install follows `../gpyreg`, which other sessions move: a population run
  on 2026-09-24 was split by its move from 1.3.1 to 1.3.2. The fingerprint
  hash was `fcf9451180c5172e` with the v1.3.1 clone before Phase 8,
  `91ca34c6b51e7f20` with it after Phase 8, and is `57241c985a68c78b` with
  the v1.3.3 clone from Phase 9 on.
- From Phase 2 on, each user-visible change gets its `CHANGELOG.md` entry in
  the commit that makes it.
- Until Phase 5, `--reruns=5 -x` fails about 5% of the time on
  `test_he_noisy_sphere_opt` (its crash, below). A suite failure there with
  `ValueError: setting an array element with a sequence` is that crash:
  rerun the suite once before treating it as a finding.
- After each phase, append to the Worklog at the end of this file: date,
  commits, and the results of the phase's checks.
- On approval, set Status to APPROVED and delete the closing review line.
- Fingerprint check: `dev/scripts/fingerprint.py` (created in Phase 1)
  prints one hash of the results of six seeded runs, three deterministic
  and three with inferred noise. A phase that must not move results shows
  the same hash before and after, on one machine (BLAS and platform
  differences can change the value). It exercises neither
  `specify_target_noise`, non-box constraints, infinite bounds nor a random
  `x0`; the phases that touch those have their own checks.

```python
# dev/scripts/fingerprint.py
"""Hash of six seeded PyBADS runs: equal before and after a change that
must not move results. Run from the repository root."""
import hashlib

import numpy as np

import pybads
from pybads import BADS


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


g = np.random.default_rng(0)


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


h = hashlib.sha256()
for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        r = BADS(
            fun, np.ones(3) * 4, -100 * np.ones(3), 100 * np.ones(3),
            -8 * np.ones(3), 12 * np.ones(3), options=o,
        ).optimize()
        h.update(np.asarray(r["x"], dtype=float).tobytes())
        h.update(np.float64(r["fval"]).tobytes())
        h.update(np.int64(r["func_count"]).tobytes())
        if r["yval_vec"] is not None:
            h.update(np.asarray(r["yval_vec"], dtype=float).tobytes())
print(pybads.__file__, h.hexdigest()[:16])
```

At `fd4bda9` with gpyreg 1.3.1, on Windows 11 with Python 3.12.6 and NumPy
2.5.3, it printed `fcf9451180c5172e`, identically in two processes.

## Phases

### Phase 0 (optional, Open Question 1): gpyreg 1.3.2 breakage check

**Executor**: Opus (orchestrator)
**Status**: [x] done (2026-09-24)
**Goal**: tell the PyVBMC maintainers early whether gpyreg 1.3.2 makes
PyBADS raise, before the long phases below (see Context). The effect of
1.3.2 on results is Phase 9.

**Steps**:
1. Record the gpyreg heads: `git -C ../gpyreg log -1 --format=%h` (expect
   `1dbbfc5`, tag `v1.3.1`) and
   `git -C ../gpyreg-w6-leftovers log -1 --format=%h` (the branch head at
   run time). Change nothing in either checkout.
2. Run the suite three times against each, reruns off:
   `python -u -m pytest -p no:rerunfailures -q -rfE` for 1.3.1, and the
   same with `PYTHONPATH=../gpyreg-w6-leftovers` for the branch
   (`PYTHONPATH` takes precedence over the editable install). Before each
   run print `python -c "import gpyreg, pybads; print(gpyreg.__file__, pybads.__file__)"`.
   Identify gpyreg by that path, not by its version string, which reads
   1.3.1 in both cases.
3. Tabulate per test the failures with their exception type and message.
   `test_he_noisy_sphere_opt` fails in about 60% of runs under 1.3.1 (the
   crash of Phase 5): compare its rate and exception between the two.
4. Trace a failure that appears only on the branch to its gpyreg call and
   value (a refusal of 1.3.2 is a `ValueError` that names it).
5. Worklog entry and report.

**Verification**:
- [x] Report to the user, for the PyVBMC maintainers: gpyreg heads,
      per-test outcomes on each, each difference with its cause.

### Phase 1: line endings and formatting

**Executor**: Opus (orchestrator)
**Status**: [x] done (2026-09-24)
**Goal**: consistent line endings and a codebase formatted once, so later
diffs show only their change.

**Steps**:
1. Precondition: `git status --porcelain` is empty (this plan and the
   `dev/README.md` index committed first); `git add --renormalize` stages
   every modified tracked file.
2. Create `dev/scripts/fingerprint.py` from the listing above; run it and
   note the hash (expected `fcf9451180c5172e` in the environment above; a
   different value elsewhere is noted, not a stop: the gate is equality
   before and after). Record the public names:
   `python -c "import pybads, pybads.bads; print(sorted(n for n in dir(pybads) if not n.startswith('_'))); print(sorted(n for n in dir(pybads.bads) if not n.startswith('_')))" > dev/scripts/runs/public_names_before.txt`.
   Commit the script (`chore(dev): fingerprint script`), so that the
   formatting of step 5 covers it.
3. Create `.gitattributes` with PyVBMC's content (`../pyvbmc/.gitattributes`:
   `* text=auto`, LF for `*.sh` and `*.sbatch`, CRLF for `*.bat` and
   `*.cmd`). Then `git add .gitattributes && git add --renormalize .`.
   Expected staged files: `.gitattributes`, the six
   `pybads/testing/bads/*.dat` (read by no test) and `docsrc/make.bat`
   (stored LF in the index, checked out CRLF, as PyVBMC's). Any other file:
   stop and report. Commit (`chore: normalize line endings`).
4. Replace `.pre-commit-config.yaml` with PyVBMC's hook versions
   (pre-commit-hooks v4.4.0, isort 5.12.0, black 23.3.0, pycln v2.6.0) and
   PyVBMC's coverage: the `^docs/` excludes of the two whitespace hooks,
   isort's `docs/tutorials` exclude, and no exclude for black (PyVBMC's
   calibration-script exclude does not apply). The current black exclude,
   `examples/*.py`, is a regular expression that excludes the notebooks and
   `pybads/function_examples.py` rather than the example scripts; with no
   exclude, black formats every Python file and the notebooks' code cells
   (Decisions). Commit (`chore: update pre-commit hooks`).
5. `python -m pip install pre-commit`, `python -m pre_commit install` (the
   git hook, so later commits are checked), then `python -m pre_commit run -a`
   until it passes. Review `git diff --stat`: Python modules and tests
   reformatted, whitespace fixes (the largest in
   `advanced_bads_options.ini`), import order, and in the five notebooks
   changes to code-cell `source` only (outputs untouched: check with
   `git diff examples/*.ipynb | grep '^[-+] *"output'` printing nothing).
   An import removed by pycln is acceptable only if unused; list it in the
   commit message.
6. Checks before committing: `python dev/scripts/fingerprint.py` prints the
   hash of step 2; the public-name command of step 2 prints the content of
   `dev/scripts/runs/public_names_before.txt`;
   `python -m pytest --reruns=5 -x -q` passes (89 tests). Commit
   (`style: format the codebase with the pre-commit hooks`).
7. Create `.git-blame-ignore-revs` with a comment line and the full hash of
   the formatting commit; commit.
8. `AGENTS.md`, "Setup and commands": the sentence that several modules are
   not black-formatted becomes: the whole tree passes the hooks, and
   `git config blame.ignoreRevsFile .git-blame-ignore-revs` hides the
   formatting commit from `git blame`. Commit.

**Verification**:
- [x] `python -m pre_commit run -a` passes with no changes.
- [x] Fingerprint and public names unchanged; suite green.

### Phase 2: packaging, changelog and documentation

**Executor**: Opus (orchestrator)
**Status**: [x] done (2026-09-24)
**Goal**: dependencies that match what CI tests, pytest out of PyBADS's
runtime dependencies, and a changelog. Before Phase 3, whose workflow
installs the `test` extra and whose pin comment names the minimum set here.

**Steps**:
1. `pybads/bads/gaussian_process_train.py`: delete
   `from pytest import Function` (unused). Confirm that no package module
   imports pytest:
   `grep -rnE "^\s*(import pytest|from pytest)" pybads --include=*.py | grep -v pybads/testing`
   prints nothing.
2. `pyproject.toml`: `requires-python = ">=3.10"`; `gpyreg >= 1.3.1`;
   remove `pytest`, `pytest-mock` and `pytest-rerunfailures` from
   `dependencies`; add a `test` extra with those three (PyVBMC's comment
   adapted: what the test suite needs; CI installs it) and add the same
   three to `dev`; keep the NumPy, SciPy and matplotlib floors, which equal
   gpyreg's. Keep the tests in the wheel (Decisions).
3. `.coveragerc`: `[run]` with `omit = pybads/testing/*` (PyVBMC's, without
   its HTML stylesheet).
4. `CHANGELOG.md` (new), Keep a Changelog 1.1.0, laid out as PyVBMC's:
   title and the format line; `## [Unreleased]` with "Changes since PyBADS
   1.0.6."; `### Upgrading from 1.0.6` ("What can stop an existing script,
   or change what it returns. Each point has its entry below."), then
   `### Added`, `### Changed`, `### Fixed`, `### Removed`. First entries:
   under Upgrading, "PyBADS needs Python 3.10 or later and gpyreg 1.3.1 or
   later."; under Changed, the same requirement with its reason (Python 3.9
   is past its end of life; 1.3.1 is the gpyreg release CI tests), and
   "PyBADS no longer lists pytest, pytest-mock and pytest-rerunfailures
   among its dependencies; `pip install "pybads[test]"` installs what the
   test suite needs (gpyreg 1.3.1 itself still installs pytest and
   pytest-rerunfailures)." `MANIFEST.in`: `include CHANGELOG.md`.
5. User and developer documentation: `README.md` (the Python requirement);
   `docsrc/source/installation.rst` (the Python requirement; the test
   instructions gain `pip install "pybads[test]"`; "PyVBMC's internal
   tests" becomes "PyBADS's internal tests");
   `docsrc/source/development.rst` (the 3.9 mentions; the install steps
   name the `.venv`, `pip install -e ".[dev]"` and the test command);
   `environment.yml` (`python>=3.10`).
6. `AGENTS.md`: the packaging paragraph (the `test` extra; pytest no longer
   among PyBADS's dependencies) and a changelog bullet under "Conventions",
   adapted from PyVBMC's `AGENTS.md`: a change a user can notice is listed
   under `Unreleased` in the commit that makes it, written for users and
   relative to the last release, and one that can stop a script written
   for the last release, or change what it returns, also has a line in the
   "Upgrading from" list.
7. `dev/TODO.md`: an item for the conda-forge recipes at the next release.
   `conda-forge/pybads-feedstock` (`recipe/meta.yaml`): run requirements
   `gpyreg >=1.3.1`, without pytest, pytest-mock, pytest-rerunfailures and
   the stale cma, corner, dill, imageio and plotly; `test.requires` gains
   pytest and pytest-rerunfailures, since its test command
   `python -m pytest --pyargs pybads --reruns=5 -x -vv` runs the tests of
   the installed package; `python_min` 3.10. It needs
   `conda-forge/gpyreg-feedstock`, at gpyreg 1.0.2 when this plan was
   written, to reach 1.3.1 first. And a note for the gpyreg maintainers:
   gpyreg lists pytest and pytest-rerunfailures as runtime dependencies.
8. Commit (`build: Python 3.10+, gpyreg 1.3.1+, test extra, changelog`).
   Then check the committed state: clone the branch into the scratchpad,
   `python -m pip wheel --no-deps -w <dir> .` there; create a fresh venv
   (`python -m venv <scratch>/venv-wheel`), `pip install <wheel>` (it pulls
   gpyreg 1.3.1 from PyPI, and pytest with it), then
   `pip uninstall -y pytest pytest-rerunfailures pytest-mock`; from a
   directory outside any source tree, `python -c "import pybads; print(pybads.__file__)"`
   succeeds and prints a `site-packages` path. Then
   `pip install pytest pytest-rerunfailures` and, still outside the source
   tree, `python -m pytest --pyargs pybads --reruns=5 -x -q` collects 89
   tests and passes. In the project venv: `pip install -e ".[dev]"`, the
   suite passes, the fingerprint is unchanged. A failed check is fixed in a
   follow-up commit.

**Verification**:
- [x] The wheel imports without pytest; the `--pyargs` run passes from the
      installed wheel.
- [x] Fingerprint unchanged.

### Phase 3: CI and release workflows

**Executor**: Opus (orchestrator)
**Status**: [x] done (2026-09-24)
**Goal**: PyVBMC's CI structure: one test job defined once, gpyreg pinned,
drift detection on schedule, a smoke run on development branches; action
versions kept current; a release workflow.

**Steps**:
1. `.github/workflows/test-matrix.yml` (new): copy
   `../pyvbmc/.github/workflows/test-matrix.yml` and adapt it: PyBADS names
   and paths; defaults `os` = the three runners and `python-version` =
   `["3.10", "3.11", "3.12"]`;
   `GPYREG_PIN: 1dbbfc5f8785d514a3d5774026810af5ed44a093` (the commit of the
   annotated tag `v1.3.1`) with PyVBMC's comment adapted (the tagged commit,
   so setuptools_scm reads 1.3.1, the minimum in `pyproject.toml`); gpyreg
   checkout with `fetch-depth: 0`; `fail-fast: false`. Drop the optional
   feature steps (torch, ArviZ, PyMC). Install PyBADS with
   `python -m pip install -e ".[test]"`, then gpyreg with `git log --oneline -1`
   and `python -m pip install -e .` (gpyreg has no `test` extra). Run
   `python -m pytest --reruns=5 -x -vv`.
2. `.github/workflows/tests.yml`: PyVBMC's, adapted: dispatch with a
   `gpyreg-ref` input; schedule on the 13th and 28th testing gpyreg `main`;
   push to `dev*` branches touching `pybads/**`, `pyproject.toml`,
   `setup.py` or the two workflow files, Ubuntu with Python 3.12 only; the
   concurrency group.
3. `.github/workflows/merge-tests.yml`: `check_changes` keeps its logic with
   `actions/checkout@v6`; the `tests` job becomes
   `uses: ./.github/workflows/test-matrix.yml`.
4. `build.yml` and `docs.yml`: action versions as PyVBMC's
   (`actions/checkout@v6`, `actions/setup-python@v6`,
   `actions/upload-artifact@v7`); build on Python 3.10.
5. `.github/dependabot.yml`: PyVBMC's (github-actions, monthly).
6. `.github/workflows/release.yml` (new), once Open Question 2 is settled:
   on `release: types: [published]` and `workflow_dispatch`; calls
   `./.github/workflows/build.yml` (the local file, not PyVBMC's); a
   publish job downloads the artifact and runs
   `pypa/gh-action-pypi-publish@v1.14.0` (PyVBMC's pin), either by trusted
   publishing (`permissions: id-token: write`, `environment: pypi`, no
   secret; on PyPI, the project's publisher settings name owner
   `acerbilab`, repository `pybads`, workflow `release.yml`, environment
   `pypi`) or with a `pypi_password` secret as PyVBMC's. A dispatch from a
   commit without a release tag builds a `.dev` version: dispatch only from
   the tag.
7. Check: every workflow parses
   (`python -c "import yaml, glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/**/*.yml', recursive=True)]"`)
   and `diff -r .github ../pyvbmc/.github` shows only the intended
   differences. The workflows run only on GitHub: the push smoke run is
   their first real check (Open Question 3); the full matrix runs on the
   pull request to `main`.
8. Documentation: `AGENTS.md`, "Setup and commands": the CI paragraph (the
   pin and its variable, the drift run, the `dev*` smoke run, what a PR
   runs, how a release is made: tag, GitHub release, the workflow uploads);
   `docsrc/source/development.rst`: how a release is made. Commit
   (`ci: shared test matrix with gpyreg pinned, release workflow`; the
   release workflow may be its own commit).

**Verification**:
- [x] YAML parses; after the user's go to push, the smoke run on the branch
      passes.

### Phase 4: generated example scripts

**Executor**: Opus (orchestrator)
**Status**: [x] done (2026-09-24)
**Goal**: `examples/scripts/*.py` generated from the notebooks, as in PyVBMC.

**Steps**:
1. `examples/scripts/Makefile`: adapt `../pyvbmc/examples/scripts/Makefile`
   to PyBADS's script names (`pybads_example_<n>_<topic>.py`, the stem of
   each notebook), for instance with a pattern rule from `<stem>.ipynb` to
   `<stem>.py`; drop PyVBMC's `.pkl` path rewrite (no PyBADS notebook opens
   a file). The generated scripts lose the hand-written two-line header,
   as PyVBMC's have none.
2. Tools: if `make --version` fails, stop and ask the user to install GNU
   Make (a system tool; PyVBMC's scripts were regenerated with GNU Make
   4.4.1). `python -m pip install nbconvert black==23.3.0 isort==5.12.0`
   (tools of this step, at the hook versions, not declared dependencies).
   Then `make -C examples/scripts` from Git Bash with the venv activated,
   so that `jupyter` and `python` resolve to it. If make fails on Windows
   paths (`VPATH`, `realpath`), stop and report.
3. Review each script's diff: changes other than the header and formatting
   mean the script had drifted from its notebook; list them in the commit
   message. Expected among them: the `options["rng_seed"] = ...` lines of
   the example 2 script, which the notebook does not have; update that
   row of `dev/results/2026-09-23-codebase-survey.md` (the script no longer
   has them; the notebook never had).
4. Run each script headless, one at a time:
   `MPLBACKEND=Agg python examples/scripts/<script>.py`; each exits 0.
5. `AGENTS.md`: the sentence on `examples/scripts/` becomes: generated from
   the notebooks by `examples/scripts/Makefile` (GNU Make, nbconvert, and
   black and isort at the hook versions); regenerate, do not edit. Commit
   (`docs: generate the example scripts from the notebooks`).

**Verification**:
- [x] A second `make` produces no diff; the five scripts exit 0.

### Phase 5: the crash with user-specified noise

**Executor**: Opus (orchestrator)
**Status**: [x] done (2026-09-24)
**Goal**: remove the crash recorded in the survey, so that noisy benchmark
runs can be measured; runs that did not crash stay identical.

**Steps**:
1. Before the change, record a paired reference with a scratch script: the
   problem of `test_he_noisy_sphere_opt` (`he_noisy_sphere` in
   `pybads/testing/bads/test_bads_optimization.py`, with its noise drawn
   from `np.random.default_rng(seed + 1000)` instead of `np.random`; the
   bounds of `get_test_opt_conf`, D=3), options
   `uncertainty_handling=True`, `specify_target_noise=True` (the first is
   required: with `uncertainty_handling=None` construction raises),
   `max_fun_evals=200`, `display="off"`, `random_seed` 0 to 29. Save per
   seed either the exception or `(x, fval, func_count, yval_vec)`.
2. `pybads/function_logger/function_logger.py`, `FunctionLogger._record`,
   the duplicate branch taken when `fsd is not None`: return the merged
   value as a scalar (`self.Y[idx].item()`) instead of the shape-`(1,)`
   row.
3. Unit test in `pybads/testing/function_logger/test_function_logger.py`:
   with `uncertainty_handling_level=2`, logging the same point twice
   returns a scalar the second time, equal to the precision-weighted mean
   of the two observations.
4. Rerun the script of step 1: every seed that did not crash gives
   identical results; every seed that crashed now finishes. A changed
   non-crashing seed, or a seed that still raises: stop and report.
5. `python -m pytest pybads/testing/bads/test_bads_optimization.py::test_he_noisy_sphere_opt -p no:rerunfailures -q`
   ten times: no `ValueError`. Report assertion failures separately (the
   test's tolerance is a different matter).
6. Records: `CHANGELOG.md`, under Fixed: a run with
   `specify_target_noise=True` could stop with `ValueError: setting an
   array element with a sequence` when a point was evaluated again.
   `dev/results/2026-09-23-codebase-survey.md`: the crash section gains the
   fix commit; a new candidate row: with `specify_target_noise=True` and
   `uncertainty_handling=None`, `BADS.__init__` raises although its message
   says to leave `uncertainty_handling` empty. `dev/TODO.md`: the crash item
   removed. `AGENTS.md`: the sentence about the crash of
   `test_he_noisy_sphere_opt` removed. Fingerprint unchanged. Commit
   (`fix: return a scalar for a repeated evaluation with user-specified noise`).

**Verification**:
- [x] Paired check of step 4; ten runs of the test without `ValueError`;
      fingerprint unchanged.

### Phase 6: seed tests

**Executor**: Opus (orchestrator)
**Status**: [x] done (2026-09-24)
**Goal**: tests that pin the seeding contract as it stands, so that the
generator change of Phase 8 is held to it.

**Steps**:
1. New `pybads/testing/bads/test_bads_seed.py`, after
   `../pyvbmc/pyvbmc/testing/vbmc/test_vbmc_seed.py`: the autouse fixture
   that saves and restores `np.random.get_state()`; a module-scoped seeded
   run (D=3, `max_fun_evals` about 60, display off) that saves and
   restores the global state itself; targets whose noise comes from their
   own `np.random.default_rng(<fixed seed>)`, created per run.
2. Tests, each comparing `x`, `fval`, `func_count`, `yval_vec` and the
   logger's `X[X_flag]` and `Y[X_flag]`:
   - `test_seed_fixes_run`: the shared run equals a second run with the same
     `random_seed`; another seed gives a different run.
   - `test_seed_ignores_global_draws`: `np.random.seed(12345)` before
     construction and `np.random.rand(7)` between construction and
     `optimize()` leave a seeded run unchanged.
   - `test_seed_none_follows_global_seed`: with `random_seed=None`,
     `np.random.seed(5)` before construction fixes the run.
   - `test_seed_fixes_noisy_run`: as the first, with
     `uncertainty_handling=True`, and with `uncertainty_handling=True` plus
     `specify_target_noise=True`.
3. Reach check: delete the call to `_init_random_seed_` in
   `_init_optimization_`; `test_seed_ignores_global_draws` must fail;
   restore it. If the test passes, it does not reach the code: fix the
   test.
4. The file runs in a few seconds (`--durations=5`). Commit
   (`test: seed tests`).

**Verification**:
- [x] The new tests pass; the reach check fails as expected.

### Phase 7: benchmark targets and population comparison

**Executor**: Opus sub-agent (implementation, steps 1–5), Opus
(orchestrator) (review and reference, steps 6–9)
**Status**: [x] done (2026-09-24)
**Goal**: the gate for every later change that moves results: a benchmark
suite, populations of seeded runs of it, and a statistical comparison of
two populations.

**Design (settled; the sub-agent implements it)**, amended on 2026-09-24
after the first baseline: every configuration runs at BADS's default budget,
500·D, not at a budget fitted to a 30-minute population. A calibration at
500·D (4 seeds per configuration) found every run ending on BADS's own
termination after 55 to 863 evaluations, whereas at the fitted budgets four
configurations solved none of their runs and never reached the end of the
algorithm (the fine mesh, the stopping rules, the GP near the optimum).
Three real-data configurations join the suite: `timing_D5` and
`multisensory_s1_D6`, the negative log-likelihoods of two of the lab's
models ported from PyVBMC's benchmark with their pins and data
(`dev/scripts/data/`), and `multisensory_s1_D6_homo`, the latter with
additive noise; their reference minima come from
`dev/scripts/make_reference_optima.py`, and a run is solved within 0.5
log-likelihood units. The population runs in about 80 minutes. The positive
control of step 8 becomes a few configurations at `--budget-scale 0.1`
(50·D), where runs stop on the budget: halving 500·D binds no run.
The original design follows.
- `dev/scripts/benchmark_targets.py`, after
  `../pyvbmc/dev/scripts/benchmark_targets.py` (`Problem`, frozen `Config`,
  `SUITES`, `STRUCTURE_SEED`, `--list/--check/--smoke`), for optimization:
  - `Problem`: `name`, `D`, `f_true` (noiseless, on a `(D,)` array),
    `f_min`, `x_min`, `lb`, `ub`, `plb`, `pub`, `x0`, `noise` (`"none"`,
    `"homo"` with a standard deviation, or `"hetero"` with a rule for the
    standard deviation), `non_box_cons`, `options`, `tolerance` (the error
    below which a run counts as solved); `fun(x)` adds noise drawn from the
    problem's own generator and returns `(y, sd)` for `"hetero"`.
  - `Config`: `name`, `D`, `noise`, `tag`, `options` (a tuple of pairs),
    `budget` (`max_fun_evals` as a multiple of `D`); `label` as PyVBMC's;
    `make(seed)`.
  - Streams: `STRUCTURE_SEED` fixes shifts and rotations per `(name, D)`;
    per run, `SeedSequence(seed).spawn(2)` gives the `x0` stream (uniform
    in the plausible box) and the noise stream; BADS gets
    `random_seed=seed`.
  - Targets: sphere, ill-conditioned ellipsoid, rotated Rosenbrock, Ackley
    and Rastrigin (shifted, with analytic minima); noisy sphere and
    ellipsoid with `"homo"` noise (`uncertainty_handling=True`) and with
    `"hetero"` noise (`uncertainty_handling=True`,
    `specify_target_noise=True`); the non-box sphere of the tests; one
    configuration with unbounded variables (infinite `lb`/`ub`, finite
    plausible box). Dimensions 2, 3 and 6, plus 10 for sphere and
    ellipsoid.
  - Budgets chosen from `--smoke` timings (which include the process
    start-up) so that the `default` suite at 30 seeds runs in about 30
    minutes as one process. Suites `smoke` (a few configurations) and
    `default`.
- `dev/scripts/population.py`, after `../pyvbmc/dev/scripts/golden_trace.py`,
  without its traces:
  - `run --suite NAME --seeds 0-29 --out DIR [--only a,b] [--options JSON]
    [--budget-scale S] [--workers 1]`: each run in a fresh spawned process
    (`ProcessPoolExecutor(max_workers=workers,
    mp_context=get_context("spawn"), max_tasks_per_child=1)`), with
    `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to
    `1` and `MPLBACKEND=Agg`; resumable (a run whose record exists is
    skipped); an exception is an outcome, recorded with its type and
    message; one progress line per finished run (`flush=True`).
    `--budget-scale` multiplies every configuration's budget.
  - One JSON record per run: `label`, `seed`, `problem`, `D`, `noise`,
    requested and effective options; `final`: `x`, `fval`, `fsd`,
    `true_error` (`f_true(x) - f_min`), `func_count`, `iterations`,
    `message`, `wall_s`, `crashed`, `exception`, and `min_noise_var`: the
    smallest of the quantity gpyreg compares with `1e-6` to choose its
    low-noise representation (`np.min(sn2)`; the same test in 1.3.1 and on
    the branch, `gpyreg/gaussian_process.py`), over the GPs of
    `bads.iteration_history["gp"]` (one per poll iteration; skip `None`
    slots); `meta`: git SHA and dirty flag, Python, NumPy, SciPy, PyBADS
    and gpyreg versions, gpyreg's source path and git SHA (the identity of
    gpyreg: under `PYTHONPATH` its version string misleads), the thread
    variables, start and end times. Provenance helpers written here after
    `../pyvbmc/dev/scripts/profile_run.py` (`git_info`, `pkg_version`,
    `module_source`, `thread_env`); no import from PyVBMC, no psutil.
  - `summary DIR`: per configuration, median [IQR] of `true_error` and
    `func_count`, the fraction solved, the crash count; writes `summary.md`.
  - `compare REF NEW [--alpha 0.05]`: per configuration and metric
    (`true_error`, `func_count`), a two-sample KS test where both sides
    have at least 3 runs, and a paired Wilcoxon signed-rank test on
    `log10(true_error + 1e-12)` over the seeds present in both (REF and
    NEW share each seed's `x0` and noise streams); Holm correction over the
    whole family; a flag also for any configuration whose crash count rises
    from zero; exit 1 on any flag. It prints effect sizes per
    configuration: the median paired `log10` error ratio with a bootstrap
    95% interval, and the difference in fraction solved.
    `compare REF --split` compares the even and odd seeds of REF (KS only;
    the null check).
- Records: output goes to `dev/scripts/runs/population/<name>/`; a
  population that serves as a reference is copied (JSON records,
  `summary.md`) to `dev/experiments/<name>/` with a `README.md` holding the
  command, the provenance, the null check, the positive control and the
  smallest effect the comparison detects at 30 seeds (the KS statistic and
  the paired shift that reach significance after Holm over the family).
  Names: `population_baseline_<YYYYMMDD>` (Phase 7),
  `population_generator_<YYYYMMDD>` (Phase 8),
  `population_gpyreg132_<YYYYMMDD>` (Phase 9), dated by the day of the run.

**Steps**:
1. (Sub-agent) Implement `benchmark_targets.py`: `--check` verifies
   `f_true(x_min) == f_min` and that `x_min` lies inside the bounds and
   satisfies `non_box_cons`; `--smoke` runs each configuration for one
   seed and prints its wall time.
2. (Sub-agent) Implement `population.py` as designed.
3. (Sub-agent) `dev/scripts/test_population.py` (run by path; default
   discovery covers only `pybads/testing`): the record schema,
   resumability, `compare` on synthetic records (identical populations do
   not flag; shifted ones do, in the KS and in the paired test), Holm.
4. (Sub-agent) `dev/README.md`, section "Scripts": usage of
   `benchmark_targets.py`, `population.py` and `fingerprint.py`, what is
   committed where, one process at a time.
5. (Sub-agent) Report: files, test results, `--smoke` timings, the
   proposed budget per configuration.
6. (Orchestrator) Review; commit the harness
   (`feat(dev): benchmark targets and population comparison`).
7. (Orchestrator) At that commit with gpyreg 1.3.1:
   `python -u dev/scripts/population.py run --suite default --seeds 0-29 --out dev/scripts/runs/population/population_baseline_<YYYYMMDD>`,
   logged. No crashes are expected after Phase 5; a crash is a finding:
   stop and report.
8. (Orchestrator) `compare <baseline> --split` flags nothing; a positive
   control, the same suite at `--budget-scale 0.5` into a scratch
   directory, flags on `true_error` in at least one configuration. If the
   null check flags or the control does not flag on `true_error`, stop and
   report.
9. (Orchestrator) Copy the baseline to
   `dev/experiments/population_baseline_<YYYYMMDD>/` with its `README.md`;
   `AGENTS.md` gains a "Numerical gates" section: the population
   comparison is the gate for a change that moves results, and its "no
   flag" means no change beyond the detectable effect recorded with the
   reference; a gate must reach the changed code; a gpyreg change is gated
   by PyBADS's comparison run against that gpyreg checkout, with
   `gpyreg.__file__` printed. Commit.

**Verification**:
- [x] Harness tests pass; null check clean; positive control flags on
      `true_error`; reference committed with provenance.

### Phase 8: random-number generator objects

**Executor**: Opus sub-agent (implementation, steps 1–6), Opus
(orchestrator) (steps 7–10)
**Status**: [x] done (2026-09-24)
**Goal**: every random draw of a run goes through one
`numpy.random.Generator`, with PyVBMC's contract as it stands: the
randomness bullet of `../pyvbmc/AGENTS.md`, `../pyvbmc/pyvbmc/rng.py` and
`../pyvbmc/pyvbmc/testing/vbmc/test_vbmc_seed.py`.
(`../pyvbmc/dev/plans/stage1-rng-generator.md` §§2–3 describe an earlier
design that reseeded the global stream; its §8 follow-up 1 removed it.)

**Contract**:
- The `random_seed` option takes what `numpy.random.default_rng` takes (an
  integer, a `SeedSequence`, a `Generator`) or `None`; an integral float is
  converted with `int()`, as before (as implemented, another float or a
  string raises `TypeError`). `BADS.__init__` sets
  `self.rng = get_rng(...)` before its first draw (the random `x0`);
  `optimize()` draws from it; nothing reseeds in `_init_optimization_`.
- `get_rng` (new `pybads/rng.py`, after PyVBMC's; internal: not exported
  from `pybads/__init__.py`, no API page): `None` derives the generator
  from four `uint32` draws of the global stream, so `np.random.seed`
  before construction fixes an unseeded run; it never reseeds the global
  stream; a `Generator` is used as given.
- A seeded run leaves NumPy's global stream untouched: `random_seed` no
  longer seeds it, so a target that draws from `np.random` is not made
  reproducible by `random_seed` (Decisions).
- The generator is passed explicitly: an `rng=` keyword on the GP functions
  of `gaussian_process_train.py` and on to each `gp.fit(..., rng=rng)` and
  `SliceSampler(..., rng=rng)`; a constructor argument of `ESSearchHedge`
  and the `ESSearch` classes; an argument of `init_sobol` and of
  `poll_mads_2n`. A keyword that defaults to `None` resolves through
  `get_rng`. The generator is never stored in `optim_state` nor in the
  `OptimizeResult` (which deep-copies its values).
- The Sobol seed keeps its MATLAB derivation from the digits of `u0`; only
  its fallback draw moves to the generator.
- `OptimizeResult["random_seed"]` holds the option's value when it is an
  integer or `None`, and `None` otherwise.

**Steps**:
1. [x] (Sub-agent) Add `pybads/rng.py`.
2. [x] (Sub-agent) Replace every draw: in `bads.py`, the random `x0`
   (`BADS.__init__`), `_init_random_seed_` (becomes the creation of the
   generator) and its call in `_init_optimization_` (removed), the two
   fallback `np.random.randint` of `_search_step_` and `_poll_step_` (same
   range, via `rng.integers`), and the call of `poll_mads_2n`; in
   `pybads/poll/poll_mads_2n.py`, which imports `from numpy import random as rnd`,
   its two `rnd.randint` and one `rnd.permutation`; in
   `gaussian_process_train.py`, the `np.random.choice`, `np.random.randn`
   and `np.random.normal` draws, the four `gp.fit` calls and the
   `SliceSampler`; in `init_sobol.py`, the fallback seed; in
   `es_search.py`, three draws; in `search_hedge.py`, two draws. Leave
   `pybads/function_examples.py` (noisy example targets) and
   `pybads/stats/kde1d.py` (its only mention is a docstring example).
3. [x] (Sub-agent)
   `grep -rnE "np\.random|numpy\.random|from numpy import random|\brnd\." pybads --include=*.py | grep -v pybads/testing`
   lists only `pybads/rng.py`, `pybads/function_examples.py` and the
   docstring of `pybads/stats/kde1d.py`.
4. [x] (Sub-agent) Update the callers in the tests (`test_search.py`,
   `pybads/testing/bads/poll/test_poll_mads.py`, any direct construction of
   the changed classes). Extend `test_bads_seed.py`:
   `test_seeded_run_leaves_global_state_untouched` (the global state equals
   before and after a seeded run, deterministic and noisy: the gate for a
   missed draw site in PyBADS or a gpyreg call without the generator);
   `test_seed_none_does_not_reseed` (construction consumes exactly the four
   draws); `test_seed_accepts_generator` (`bads.rng is rng`);
   `test_seed_none_ignores_draws_after_construction`. Reach check: add a
   temporary `np.random.rand()` inside `_poll_step_`;
   `test_seeded_run_leaves_global_state_untouched` must fail; remove it.
   The tests of Phase 6 stay green.
5. [x] (Sub-agent) Documentation: the `random_seed` description in
   `basic_bads_options.ini`, the `BADS` docstring and the `random_seed`
   entry of the `OptimizeResult` docstring; `CHANGELOG.md` (Added: seeded
   runs through a generator, `bads.rng`; Upgrading: "Results differ from
   1.0.6, also with a fixed seed." and "`random_seed` no longer seeds
   NumPy's global random state."); `AGENTS.md`, the randomness bullet
   rewritten to the contract.
6. [x] (Sub-agent)
   `PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.1 python -m pytest --reruns=5 -x -vv`
   passes (the CI pin; the editable install follows `../gpyreg`, at 1.3.3
   since its release); report.
7. [x] (Orchestrator) Review; commit
   (`feat: random draws through a numpy Generator`).
8. [x] (Orchestrator) At that commit with gpyreg 1.3.1, run the `default`
   population into
   `dev/scripts/runs/population/population_generator_<YYYYMMDD>`;
   `compare dev/experiments/population_baseline_<YYYYMMDD> <new>`. Every
   trajectory changes; the distributions should not. A flag stops the
   phase: report it with the configuration, metric and effect size. Large
   effect sizes without a flag are reported too.
9. [x] (Orchestrator) With no flag, copy the population to
   `dev/experiments/population_generator_<YYYYMMDD>/` (its README cites the
   comparison and its effect sizes); it is the reference from now on. Run
   `python dev/scripts/fingerprint.py` and record the new hash in the
   Worklog: the fingerprint of later phases.
10. [x] (Orchestrator) Commit the records.

**Verification**:
- [x] No global draw in a seeded run (test, with its reach check); seed
      tests green; population comparison without flags; changelog and docs
      updated.

### Phase 9: gpyreg 1.3.3 for PyBADS

**Executor**: Opus (orchestrator)
**Status**: [x] done (2026-09-25)
**Goal**: the effect of gpyreg 1.3.3 on PyBADS, measured with the
population comparison against the generator reference of Phase 8; the
answers the PyVBMC maintainers asked for (Context); and PyBADS's minimum
and CI pin moved to 1.3.3, when the comparison shows no flag or the user
accepts the flags it explains.

Amended on 2026-09-24, twice. gpyreg 1.3.2 was released during Phase 7
(tag `v1.3.2`, `29b868c`, the merge of the branch `w6-leftovers`, whose
worktree `../gpyreg-w6-leftovers` was then removed), and 1.3.3 after it
(Context). The phase as first written assessed the 1.3.2 branch before its
tag; it now assesses the release PyBADS would require, 1.3.3. The questions
of Phase 0 and the two issues of step 5 were answered before the 1.3.2 tag
(Worklog, Phase 7 entries). 1.3.2 gets no population of its own: step 3
checks at PyBADS's default options that 1.3.3 gives the same results as
1.3.2, including in the retries of `_robust_gp_fit_`, where the two
releases leave a failed GP in different states. At default options that
retry reads only bounds that PyBADS sets itself (`_gp_hyp` in
`gaussian_process_train.py`), so no difference is expected; with
`use_slice_sampler=True` or the `negquad` mean it reads more (survey
candidate, `_robust_gp_fit_` row).

**Steps**:
1. [x] Record the heads of the clones: `git -C dev/scripts/runs/gpyreg/<tag>
   log -1 --format=%h` for `v1.3.1`, `v1.3.2` and `v1.3.3` (expect
   `1dbbfc5`, `29b868c`, `98ab5a4`). Change nothing in them or in
   `../gpyreg`.
2. [x] Suite at the head of Phase 8 on 1.3.1 and on 1.3.3, reruns off,
   three times each:
   `PYTHONPATH=dev/scripts/runs/gpyreg/<tag> python -u -m pytest -p no:rerunfailures -q -rfE`,
   with `python -c "import gpyreg, pybads; print(gpyreg.__file__, pybads.__file__)"`
   printed before each run. Tabulate per test the failures with their
   exception type and message; trace a failure that appears only on 1.3.3
   to its gpyreg call (a refusal is a `ValueError` that names it).
3. [x] 1.3.2 against 1.3.3 at default options. `fingerprint.py` under each
   clone prints the same hash. Then six of the seven configurations where
   the 1.3.1 robust-fit counts of Phase 7 found failed fits in every run
   (`sphere_D2`, `ellipsoid_D3`, `rosenbrock_D2`, `ellipsoid_D3_homo`,
   `sphere_nonbox_D3`, `ellipsoid_D3_unbounded`; not `ellipsoid_D6`),
   seeds 0–9:
   `population.py run --suite default --only <those> --seeds 0-9` under
   each clone into scratch directories under
   `dev/scripts/runs/population/`; per record, the `final` fields other
   than `wall_s` are equal. A difference contradicts the gpyreg
   maintainers' account (Context): stop and report it with the
   configuration and seed.
4. [x] Population: the `default` suite, seeds 0–29, at the head of Phase 8
   with `PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3`, into
   `dev/scripts/runs/population/population_gpyreg133_<YYYYMMDD>`, logged;
   `summary` of it; `compare dev/experiments/population_generator_<YYYYMMDD> <new>`.
   From the records: the share of runs with `min_noise_var` below `1e-6`
   (the low-noise representation, whose predictions changed in 1.3.2), and
   whether the flagged configurations are those runs; the crashes of both
   populations, and for each crash of the generator reference whether its
   seed crashes under 1.3.3 (the question of the `dev/TODO.md` item on
   unguarded GP updates).
5. [x] The two PyBADS-side issues, rerun at the released tag:
   `PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 python -u dev/scripts/gpyreg_issue_checks.py dev/scripts/runs/issues_<ts>/gpyreg133.json`,
   logged. Expected, as under 1.3.1 and 1.3.2: `fit_lik=False` raises
   `Unknown hyperprior type delta`; the robust-fit counts reach no bound
   inversion. Note that the counts of Phase 7 were taken with the global
   stream: the rerun under 1.3.3 is at the head of Phase 8, so its
   trajectories differ.
6. [x] With no flag in step 4 and no difference in step 3, require 1.3.3:
   `pyproject.toml` `gpyreg >= 1.3.3`; `.github/workflows/test-matrix.yml`
   `GPYREG_PIN: 98ab5a4adecf37eb188521360bd87835979f47e7` with its comment
   naming v1.3.3; `CHANGELOG.md`, the Upgrading line and the Changed entry
   (gpyreg 1.3.3 or later; the release CI tests); the conda-forge item of
   `dev/TODO.md` (`gpyreg >=1.3.3`); the gpyreg-minimum Decision below;
   any other mention of the minimum (`grep -rn "1\.3\.1"` outside
   `dev/experiments/`, `dev/plans/` and the records that name the gpyreg a
   run used). The suite passed on 1.3.3 in step 2. Commit
   (`build: require gpyreg 1.3.3`). A push, which runs the `dev*` smoke
   test against the new pin, waits for the user's go. With a flag or a
   difference: no move; stop and report. Done on the user's decision
   (2026-09-25) after step 4 flagged five configurations, every one
   improved by 1.3.2's low-noise predictions (Worklog).
7. [x] Records: a result note `dev/results/<YYYY-MM-DD>-gpyreg-1.3.3.md` (the
   heads, the suite tables, the 1.3.2–1.3.3 check, the population
   comparison with effect sizes, the low-noise share and the crashes, the
   two issues with their counts, the verdict), linked from
   `dev/README.md`'s index. When 1.3.3 is required, its population is the
   reference from then on: copy it to
   `dev/experiments/population_gpyreg133_<YYYYMMDD>/` with its `README.md`
   (command, provenance, the comparison it passed, the null check
   `compare <it> --split`, and what the comparison detects at 30 seeds,
   citing the baseline's positive control), and record in the Worklog the
   fingerprint hash under the v1.3.3 clone. `dev/results/2026-09-23-codebase-survey.md`:
   the known-noise and robust-fit sections cite the 1.3.3 results.
   `dev/TODO.md`: the unguarded-updates item answers whether 1.3.3 changes
   the crashing runs; the gpyreg-releases item names 1.3.3 as the required
   release. `dev/README.md`, the reference population's index entry.
   Commit.
8. [x] Report to the user, for the PyVBMC maintainers: suite results on
   1.3.1 and 1.3.3, each difference and its cause, the 1.3.2–1.3.3 check,
   the population comparison, and the PyBADS-side proposals (the two
   issues, and passing the data explicitly in the retry of
   `_robust_gp_fit_`).

**Verification**:
- [x] 1.3.2 and 1.3.3 agree at default options; the population comparison
      against the generator reference is reported with its effect sizes.
- [x] `gpyreg >= 1.3.3` and the pin committed (the user's decision, the
      flags explained), the reference replaced; result note and records
      committed; report delivered.

## Documentation

Updated in the phase that makes the change:
- `AGENTS.md`: Phases 1 (formatting), 2 (packaging, changelog convention),
  3 (CI, releases), 4 (example scripts), 5 (the crash sentence), 7
  ("Numerical gates"), 8 (randomness).
- `CHANGELOG.md`: created in Phase 2; entries in Phases 2, 5 and 8; the
  gpyreg requirement moved in Phase 9.
- `dev/README.md`: the index (this plan; the Phase 9 note); "Scripts"
  (Phase 7).
- `dev/TODO.md`: Phases 2, 5, 9.
- `dev/results/2026-09-23-codebase-survey.md`: Phases 4, 5, 9.
- `README.md`, `docsrc/source/installation.rst`, `environment.yml`:
  Phase 2. `docsrc/source/development.rst`: Phases 2 and 3.
- `basic_bads_options.ini` (`random_seed`), the `BADS` and `OptimizeResult`
  docstrings: Phase 8.
- `pyproject.toml` and `GPYREG_PIN` (the gpyreg requirement): Phase 9.

New records, each holding what nothing else holds: the reference
populations under `dev/experiments/` (Phases 7, 8, 9) and the result note
of Phase 9. This plan holds the execution status in its Worklog.

## Decisions

- **One formatting commit for the whole tree, listed in
  `.git-blame-ignore-revs`** — every later diff shows only its change, and
  a trial on a scratch clone left seeded results bit-identical. Rejected:
  formatting each file when it is next touched (no churn now, but a
  whole-file rewrite inside a later review, the RNG change among them).
- **black covers the notebooks and the example scripts, as in PyVBMC** — one
  rule for all Python code; notebook outputs are untouched. Rejected:
  keeping an examples exclude (the notebooks keep their layout, but the
  current regex excludes the notebooks and `function_examples.py` by
  accident, and a corrected one would still differ from PyVBMC).
- **gpyreg minimum equal to the CI pin (1.3.1, then 1.3.3)** — the minimum names a
  version CI tests; both moved to 1.3.3 in Phase 9 (1.3.2 is skipped:
  1.3.3 restores the GP after a failed call, which 1.3.2 left
  inconsistent), on the user's decision after its comparison flagged five
  configurations, every one improved by 1.3.2's low-noise predictions.
  Rejected: `>= 1.1.0`,
  the oldest with `fit(rng=)` (lower, but untested).
- **NumPy, SciPy and matplotlib floors unchanged** — they equal gpyreg's,
  and PyBADS cannot need less than gpyreg. Rejected: PyVBMC's floors
  (NumPy 2.0, SciPy 1.15; consistent across the lab's packages, but
  nothing in PyBADS needs them).
- **Tests stay in the wheel** — the conda-forge recipe tests the installed
  package with `pytest --pyargs pybads`. Rejected: PyVBMC's exclusion (a
  smaller wheel, but it breaks that recipe).
- **The crash fix precedes the baseline** — noisy benchmark runs would
  otherwise crash in most seeds; the fix leaves non-crashing runs identical
  (checked in Phase 5). Rejected: leaving it to the bug hunt.
- **A population harness without traces or exact replay** — BADS runs are
  cheap enough for distribution comparisons over 30 seeds; exact replay
  needs a trace format and matters once oracles exist. Rejected: PyVBMC's
  full golden-trace machinery now.
- **Benchmark runs at BADS's default budget, ending on its own termination**
  — every run covers the whole algorithm, and a calibration found the
  suite at 30 seeds taking about an hour (80 minutes with the real
  targets). Rejected: budgets fitted to a 30-minute population (half the
  cost, but four configurations then solved no run and never reached the
  fine mesh or the stopping rules).
- **Real targets as maximum-likelihood fits** (negative log-likelihood
  within the paper's bounds) — how BADS is used; the timing data carries
  the paper's maximum-likelihood point to check the reference against.
  Rejected: MAP under PyVBMC's priors (keeps the optimum off the bounds,
  but has no published reference and tests less of BADS's bound handling).
- **KS tests plus a paired signed-rank test, with effect sizes** —
  PyVBMC's KS test detects only gross changes at 30 seeds; the paired test
  uses the shared `x0` and noise streams of each seed, and the effect
  sizes make "no flag" interpretable. Rejected: KS alone, as PyVBMC
  (simpler, weaker).
- **One generator per run, PyVBMC's contract** — one design across the
  lab's packages. Rejected: separate child streams for GP fitting and
  search (a gpyreg change in its number of draws would leave the search
  draws in place, but the design would differ from PyVBMC's).
- **`random_seed` no longer seeds NumPy's global stream** — a seeded run
  touches no global state, as in PyVBMC; no example relies on the side
  effect (examples 3 and 4 draw noise from `np.random` without
  `random_seed`). Rejected: keeping `np.random.seed(random_seed)` for
  compatibility (scripts whose target draws from `np.random` stay
  reproducible, at the cost of a hidden global side effect). Listed under
  "Upgrading" in the changelog.

## Open Questions

1. **Run Phase 0 now?** Default: yes, before Phase 1 (about ten minutes);
   the gpyreg tag waits on PyBADS's answer.
2. **Release publishing.** Default: PyPI trusted publishing (no stored
   secret; the PyPI project's owner adds the publisher, Phase 3 step 6).
   Alternative: PyVBMC's `pypi_password` secret.
3. **Pushing the branch.** Default: push after Phase 3, so that the `dev*`
   smoke run tests the new workflows; each push only with the user's go.
4. **Branch.** Default: rename `dev-agent-docs` to `dev-next` (the
   long-lived development branch, as in PyVBMC); pull requests to `main`
   later.

## Worklog

(Appended after each phase: date, commits, check results.)


### Phase 0 — 2026-09-24

- gpyreg 1.3.1: `../gpyreg` at `1dbbfc5` (tag `v1.3.1`, clean). Branch:
  `../gpyreg-w6-leftovers` at `38e8ada` (clean). PyBADS at `82e3a81`.
- Suite, reruns off, three runs each (`-W ignore::DeprecationWarning`):
  1.3.1: 88 passed, `test_he_noisy_sphere_opt` failed 3 of 3, each time on
  its tolerance (`Error [1.49897281] is not smaller than tolerance`), not
  with the crash. Branch: 89 passed, 3 of 3. No other test differs; no
  `ValueError` from a gpyreg refusal on the branch.
- Cause: in the full suite, `test_small_noisy_func` calls
  `np.random.seed(42343)`, so the tests after it draw from a fixed global
  stream and `test_he_noisy_sphere_opt` has one outcome per gpyreg version.
  `test_small_noisy_func` (noise standard deviation `1e-4`, a variance of
  `1e-8`) is in the low-noise representation, and its run changes under
  1.3.2 (seeded, as in the test: error `1.2e-6` on 1.3.1, `1.6e-8` on the
  branch; 164 and 168 evaluations; both pass) and leaves the global stream
  in another state. `test_he_noisy_sphere_opt` itself does not change:
  with `np.random.seed(s)`, `s` = 0–19, its 20 runs are bit-identical on
  both versions, including the 6 that crash (PyBADS's crash of Phase 5).
- Verdict for the gpyreg maintainers: no refusal fires and no test breaks
  on the branch; the one difference is a prediction change in the
  low-noise representation, passed through the global random stream to a
  later test. Records: `dev/scripts/runs/phase0_1790247784/` (gitignored).

### Phase 1 — 2026-09-24

- Commits: `e82c1f8` fingerprint script, `82594df` line endings
  (staged exactly `.gitattributes`, `docsrc/make.bat` and the six `.dat`
  files), `3e99f1e` hooks, `69be885` formatting (53 files), `d882cf8`
  `.git-blame-ignore-revs`, then `AGENTS.md`.
- Fingerprint `fcf9451180c5172e` before and after; public names unchanged;
  suite 89 passed (5 reruns); every changed `.py` file has the same syntax
  tree up to import order and string whitespace; no notebook output
  changed; pycln removed nothing (it leaves `from pytest import Function`).

### Phase 2 — 2026-09-24

- Commit `8eb0266`. `CHANGELOG.md` starts with the sections that have
  entries (Upgrading, Changed); Keep a Changelog omits empty sections.
- Wheel check on a clone of `8eb0266`: the wheel installs gpyreg 1.3.1 from
  PyPI (with pytest); after `pip uninstall pytest pytest-rerunfailures
  pytest-mock`, `import pybads` works from `site-packages`; with pytest
  back, `pytest --pyargs pybads` from outside the source tree: 89 passed.
- Project venv: `pip install -e ".[dev]"`; suite 89 passed; fingerprint
  `fcf9451180c5172e`.

### Phase 4 — 2026-09-24

- `5b066dd`: black (Phase 1) had split `x0 = ...;  # Starting point` into a
  statement and a comment line in three notebooks and two test scripts;
  the comments are back inline.
- The Makefile names the notebook as an explicit prerequisite
  (`%.py: ../%.ipynb`) instead of PyVBMC's `VPATH`, and calls
  `python -m nbconvert`; GNU Make 4.4.1 (Chocolatey) runs it from Git Bash.
- The regenerated scripts differ from the old copies beyond the header and
  formatting where the copies had drifted from the notebooks: example 1
  imports `from pybads.bads import BADS` (as the notebook), example 2 loses
  the `options["rng_seed"]` lines and writes `> 1.0`, example 4 loses three
  result lines the notebook does not have. A second `make -B` changes
  nothing; the five scripts exit 0 with `MPLBACKEND=Agg`.

### Phase 5 — 2026-09-24

- Fix `4566acb` (with its unit test, which fails on the old code at the
  scalar check, and the changelog entry); records in the next commit.
- Paired check, 30 seeds (`random_seed` = seed, target noise from
  `default_rng(seed + 1000)`): before, 10 crashes; after, 0; the 20
  non-crashing seeds bit-identical. `test_he_noisy_sphere_opt`, ten runs
  with reruns off: no `ValueError`; 1 run failed its tolerance (error
  1.16 > 1), which is the test's statistical check, not the crash.
  Fingerprint `fcf9451180c5172e`.
- New survey candidate: `specify_target_noise=True` with
  `uncertainty_handling=None` raises, against its own message.

### Phase 6 — 2026-09-24

- `pybads/testing/bads/test_bads_seed.py`: 5 tests (the noisy one for
  inferred and for specified noise), 5.5 s. Reach check: with the reseed in
  `_init_optimization_` replaced by the stored seed,
  `test_seed_ignores_global_draws` fails; restored.

### Phase 3 — 2026-09-24

- Commits `588dc17` (test matrix, tests, merge-tests, build, docs,
  dependabot) and `f4e00d6` (release workflow by trusted publishing through
  the `pypi` environment, admitting `v*` tags; AGENTS.md and the developer
  guide). `build.yml` also checks out with `fetch-depth: 0`, so that a
  build reads its version from the tags.
- `dev-next` pushed at `989e360` (user's go): the `tests` smoke run
  (Ubuntu, Python 3.12, gpyreg pinned at `1dbbfc5`) passed, 95 tests.

### Phase 7 (in progress) — 2026-09-24

- Harness `4d9a127` (sub-agent; reviewed). The `default` suite has 15
  configurations, not the full grid, to fit about 30 minutes at 30 seeds;
  the non-box target uses MATLAB's constraint (`runtest.m`). Harness tests
  14 passed; `--check` and `--smoke` pass.
- The two issues of Phase 9 step 4, run early and reported for the gpyreg
  tag: `fit_lik=False` raises `Unknown hyperprior type delta` under both
  versions; the bound inversion of `_robust_gp_fit_` is not reached at
  default options (at most 3 consecutive failures in about 1,500 calls per
  version). Recorded in the survey (`028f000`).
- A first baseline at `028f000` was split by the move of `../gpyreg` to
  v1.3.2 (59 runs on 1.3.1, 391 on 1.3.2) and is discarded; the pinned
  clones replace the editable install for evidence runs (Conventions).

### Phase 7 (continued) — 2026-09-24

- The short-budget baseline (`8baccaa`, 450 runs on the v1.3.1 clone,
  clean) was superseded before its checks, when the budgets moved to 500·D
  (`3f693a4`, after the calibration). Its null check was clean; the
  positive control was stopped.
- Real targets `469a505` (sub-agent; pins reproduced: timing exactly,
  multisensory to 1.8e-10). Reference minima regenerated at the clean
  commit (`5bcc430`) with identical `f_min` and `x_min`: timing
  3839.1732706078164 (17 of 20 restarts within 0.5; a second basin 4.1 to
  4.6 higher; 4.7e-8 below the paper's MLE, the same point),
  multisensory_s1 483.5133436051275 (20 of 20 within 0.5; its minimum is a
  curve in the noise parameters). `--smoke`: 163 s per seed, about 82
  minutes at 30 seeds.

### Phase 7 (done) — 2026-09-24

- Reference `dev/experiments/population_baseline_20260924/`: 540 runs at
  `2226883` with the v1.3.1 clone, clean, 75.7 minutes; 2 crashes (the
  unguarded GP updates, TODO and survey). Null check clean (36 tests);
  positive control (3 configurations at 50·D) flagged all three,
  `ellipsoid_D10` on the error. Detectable at 30 seeds: KS 0.50, a paired
  shift of about 0.87 SD (README). AGENTS.md gains "Numerical gates".

### Resume point — 2026-09-24

Phases 0–7 are done; the next session resumes at Phase 8, its steps 1–6
briefed to an Opus sub-agent with the phase text, the orchestrator keeping
the populations (steps 7–10). State at this entry:

- Branch `dev-next` at the commit of this entry; `origin/dev-next` is at
  `989e360` (the first push); the later commits are local, and a push
  waits for the user's go.
- Environment: the venv `.venv` (Python 3.12.6), with PyBADS editable and
  gpyreg editable from `../gpyreg`, which now holds gpyreg 1.3.2 and moves
  with other work; evidence runs therefore use the pinned clones of
  `dev/scripts/runs/gpyreg/` (Conventions, `dev/scripts/runs/LOCAL.md`).
  The venv also has pre-commit (hook installed), nbconvert, black 23.3.0
  and isort 5.12.0; GNU Make 4.4.1 is on the `PATH`.
- Reference for Phase 8's comparison:
  `dev/experiments/population_baseline_20260924/` (18 configurations × 30
  seeds, about 76 minutes). Phase 8's own population takes as long; its
  `compare` should show no flag, with the effect sizes reported. The two
  crashing seeds of the reference (`ellipsoid_D3` 20, `ellipsoid_D10` 7)
  may crash again or not under the generator: a crash rising from zero in
  another configuration flags.
- The fingerprint changes with the generator (step 9 records the new
  hash); until then `fcf9451180c5172e` with the v1.3.1 clone.
- Phase 9 after the release of gpyreg 1.3.2 (its amendment above): the
  clone `dev/scripts/runs/gpyreg/v1.3.2` stands for the branch; step 4's
  two issues were run and reported before the tag (Phase 7 entries, the
  survey), so step 4 needs only a rerun of the `fit_lik=False` check and
  of the robust-fit counts at the released tag if the result note is to
  cite them (`dev/scripts/gpyreg_issue_checks.py`, with
  `PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.2`); the result note also says whether 1.3.2 changes the two
  crashing runs (`dev/TODO.md`). The PyVBMC maintainers have both earlier
  reports (Phase 0, the two issues); Phase 9's report is the remaining one.

### Plan amended — 2026-09-24: gpyreg 1.3.3

- gpyreg 1.3.3 released (`98ab5a4`, Context). Phase 9 now assesses 1.3.3,
  with a check that 1.3.3 and 1.3.2 agree at default options in place of a
  1.3.2 population, and requires 1.3.3 when its comparison shows no flag
  (the user's decision). The clone `dev/scripts/runs/gpyreg/v1.3.3` is
  made; `../gpyreg` is at `v1.3.3`, so Phase 8 runs the suite under the
  v1.3.1 clone.
- At default options the retry of `_robust_gp_fit_` reads only bounds set
  by `_gp_hyp` (checked in the code), so 1.3.3's restore-on-failure does
  not reach it; with the slice sampler or the `negquad` mean it does (new
  survey candidate). 1.3.1's `fit` stores `X` and fills the recommended
  bounds before the optimization that can raise.

### Phase 8 — 2026-09-24

- Implementation `551aa47` (sub-agent; reviewed), plan `c85cddb`, records
  `be11a52`. Suite on
  the v1.3.1 clone: 109 passed, no reruns. `test_bads_seed.py`: 19 tests,
  8 s; the global-state test covers a random `x0`, inferred noise, prior
  samples and the slice sampler, and all four cases failed with a
  temporary `np.random.rand()` in `_poll_step_`. Beyond the plan: a float
  seed that is not a whole number, or a string, raises `TypeError` (an
  Upgrading line of the changelog); nothing else holds the generator than
  `bads.rng` and the stored hedge, `bads.search_es_hedge.rng`.
- Population `population_generator_20260924` at `c85cddb` with the v1.3.1
  clone: 540 runs, 84.5 minutes, 2 crashes (`ellipsoid_D10` seeds 13 and
  26, a third unguarded GP call, in the recovery of `local_gp_fitting`;
  both reproduce; TODO and survey). A first launch ran the system Python
  (NumPy 2.5.0, SciPy 1.18.0) through a bare `python`; stopped after 12
  runs and discarded (Conventions, `AGENTS.md`).
- `compare` against the baseline: no flag in 54 tests. Effect sizes within
  ±0.37 in log10; two intervals exclude zero, `ellipsoid_D3_homo` +0.26
  [+0.06, +0.41] and `ellipsoid_D3_hetero` +0.21 [+0.005, +0.38], with
  signed-rank p = 0.25 and 0.096 before correction (about one of 18
  expected by chance). Null check of the new population: no flag in 36
  tests. Copied to `dev/experiments/population_generator_20260924/`, the
  reference from now on.
- Fingerprint with the v1.3.1 clone: `91ca34c6b51e7f20` (two processes),
  the fingerprint of later phases in place of `fcf9451180c5172e`.

### Phase 9 (in progress) — 2026-09-24

- Clones: v1.3.1 `1dbbfc5`, v1.3.2 `29b868c`, v1.3.3 `98ab5a4`, all clean.
  PyBADS at `be11a52`.
- Suite, reruns off, three runs each: 109 passed in every run on 1.3.1 and
  on 1.3.3 (logs `dev/scripts/runs/phase9_suite/`).
- 1.3.2 against 1.3.3: fingerprint `57241c985a68c78b` under both (1.3.1:
  `91ca34c6b51e7f20`); six of the seven configurations with failed fits in
  every run of the 1.3.1 counts, seeds 0–9: the 60 records identical in every `final` field but
  `wall_s`.
- Population `population_gpyreg133_20260924` at `2059506` with the v1.3.3
  clone: 540 runs, 72.2 minutes, no crash. `compare` against the generator
  reference flags five configurations, all with smaller errors under
  1.3.3: `sphere_D10` (median paired log10 error ratio -2.43, median error
  4.8e-5 → 8.5e-8, evaluations 538 → 449, flagged also on the
  evaluations), `timing_D5` (-2.63; 6.2e-5 → 2.3e-7), `multisensory_s1_D6`
  (-1.06; 2.4e-6 → 2.1e-7), `rosenbrock_D6` (-0.35; solved 0.63 → 0.77)
  and `rastrigin_D3` (every pair slightly lower within the same local
  minima, median error 3.98 in both). Unflagged: `ackley_D6` +0.10
  [+0.007, +0.30]. Null check of the new population: no flag in 36 tests.
- Every changed run went through the low-noise representation: 279 runs
  differ from the reference, each with `min_noise_var` below `1e-6`, and
  the 261 others are identical in every result field but `wall_s`. Of the
  150 noisy-target runs, 2 changed (`ellipsoid_D3_homo`). `ellipsoid_D10` seeds 13 and
  26, which crashed in the reference, finish under 1.3.3; both are in the
  low-noise regime before the crash, so their trajectories differ and this
  does not show whether 1.3.3 avoids the crash.
- Issue checks under 1.3.3 at `2059506` (`dev/scripts/runs/issues_1790283783/`):
  `fit_lik=False` raises `Unknown hyperprior type delta` (from
  `_write_prior_block`); robust fit over the 18 configurations, seeds
  0–9: 3,385 calls, failures per call {0: 2865, 1: 307, 2: 201, 3: 12}, no
  bound inversion, every run finished.
- Step 6 says no move with a flag: stopped for the user's decision.

### Phase 9 (continued) — 2026-09-25

- The user decided to move to 1.3.3 despite the flags, all improvements
  explained by 1.3.2's low-noise predictions. `c588686`: `gpyreg >= 1.3.3`,
  `GPYREG_PIN` at `98ab5a4`, the changelog (Upgrading line; the Changed
  entry says that runs on deterministic targets end closer to the
  minimum).
- Records: result note `dev/results/2026-09-25-gpyreg-1.3.3.md`; the 1.3.3
  population copied to `dev/experiments/population_gpyreg133_20260924/`,
  the reference from now on; survey, TODO and `dev/README.md` updated; the
  dev scripts' usage lines name the v1.3.3 clone. The issue checks under
  1.3.3 took 20 minutes.
- Fingerprint of later phases, with the v1.3.3 clone: `57241c985a68c78b`.

### Phase 9 (done) — 2026-09-25

- Report to the user for the PyVBMC maintainers: nothing in gpyreg to
  change for PyBADS; the one gpyreg-side item is packaging (pytest among
  its runtime dependencies, `dev/TODO.md`).
- `/doublecheck` by three fresh-context reviewers (code and tests, records
  against the data, docs and CI): no correctness finding in the code or
  the build; the records' numbers all recomputed and matching. Fixed: the
  TODO's claim that gpyreg 1.3.3's restore leaves a consistent GP (false
  where PyBADS assigns `gp.X` before `gp.update`), the description of the
  six configurations of the 1.3.2–1.3.3 check, overclaims in the verdict,
  the changelog and `dev/README.md`, a missing Upgrading line (the seed is
  read when `BADS` is created), stale references to the reference
  population, and the plan text above. Survey: three new candidate rows.
- The venv's gpyreg reinstalled editable from `../gpyreg` at `v1.3.3`
  (metadata 1.3.3); the suite passes against it.
