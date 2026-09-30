# Update reminders: an old-release reminder and `check_for_updates()`

Created 2026-09-30, for the release of 1.5.0. Users who installed PyBADS
once and never updated learn that a newer release may exist, without
PyBADS making a network request they did not ask for:

1. **An old-release reminder, with no network access.** The package ships
   the date of its release. When a run starts in an interactive session
   and the installed release is more than a year old, the run notes that a
   newer version may exist and how to find out: at most once per Python
   session, and three times for each installed version, at least 90 days
   apart.
2. **`pybads.check_for_updates()`, on request.** It asks PyPI for the
   latest release, compares it with the installed version, and says how to
   update. It is the only code in the package that opens a network
   connection, and only when the user calls it.

Neither touches a run's results, its random stream or NumPy's global state.

The design is PyVBMC's (`acerbilab/pyvbmc`, branch `dev-next`, merged at
`01bd00de`: `pyvbmc/_release.py`, `pyvbmc/vbmc/_release_reminder.py`,
`pyvbmc/_update_check.py`), copied, since PyBADS does not import PyVBMC.
Its plan, `dev/plans/version-check.md` there, holds the survey of the
update notices of other tools and the PI's rulings (2026-09-30) that this
copy keeps: the threshold of a year, counted in calendar dates, which
assumes a release of PyBADS at least once a year (D1); `show_tips` as
the switch (D2); the wording (D4); versions compared as `X.Y.Z` integer
tuples, with no new dependency (D5); the release date as a tracked
constant checked against the changelog (D7); and the cap of three
showings per version, 90 days apart (D8).

## Where PyBADS differs from PyVBMC (PI, 2026-09-30)

- **The cache directory.** PyVBMC takes its cache directory from
  `platformdirs`, a dependency of its calibration cache. PyBADS has no
  such dependency, and the reminder computes the same directory itself:
  `%LOCALAPPDATA%\pybads` on Windows, `~/Library/Caches/pybads` on macOS,
  `$XDG_CACHE_HOME/pybads` or `~/.cache/pybads` elsewhere, or the
  directory that `PYBADS_CACHE_DIR` names. Where none can be found, the
  once-per-session flag is the only cap.
- **The output.** Every message of a run goes to the `BADS` logger, the
  reminder too: one INFO record, as a tip is, in the tip's slot of
  `BADS._init_mesh_`, after the opening message and before the column
  headers. It shows under the conditions of a tip (`show_tips` on and the
  logger enabled for INFO, the displays `"iter"`, `"all"` and `"full"`),
  and in an interactive session only.
- **The slot.** PyBADS has no calibration reminder: a start considers the
  old-release reminder, then a tip. When the reminder shows, no tip shows
  and the tips' cadence does not advance.
- **What counts as a showing.** A showing counts when the reminder is
  logged: logging cannot tell whether a handler delivered the record, and
  the check of an interactive session reads standard output, not the
  logger's handlers, so a session that sends the `BADS` logger to a file
  uses up the showings there. PyVBMC counts a showing only when its
  `print` succeeded.
- **The fork hook.** As `_runtime_tips.py` does, a forked child gets a
  lock of its own; it keeps the parent's session flag.

## Design

- `pybads/_release.py` holds `RELEASE_DATE`, the ISO date of the latest
  release in the source tree. The pull request of a release sets it to
  the date of its changelog heading, `## [X.Y.Z] - YYYY-MM-DD`
  (`AGENTS.md`, "Setup and commands"); a test checks that the two agree.
- `pybads/bads/_release_reminder.py`:
  `consider_release_reminder(*, logger, enabled, today=None,
  installed=None, release_date=None, interactive=None, environ=None,
  state_path=None) -> bool`, which returns whether the reminder was logged.
  It logs nothing when `enabled` is false or the logger is above INFO; when
  it has shown in this Python session; when `CI`,
  `PYBADS_NO_UPDATE_REMINDER` or `NO_UPDATE_NOTIFIER` is set to a value
  other than empty, `0` or `false`; when the installed version is not a
  final `X.Y.Z` (a development install); when the release date is
  unreadable; when the date of the run is not past the release's first
  anniversary (a clock behind the release included); when neither
  standard output is a terminal nor the code runs in an IPython kernel;
  and when the state file forbids it.
- The state file, `update_reminder.json` in the cache directory, maps each
  installed version to the dates of its showings, as in
  `{"1.5.0": ["2027-11-02"]}`. A version with three dates, or whose last
  date is less than 90 days before the date of the run, logs nothing.
  After a showing the date is appended, and the file replaced through a
  temporary file and a rename, every error swallowed. A file whose content
  is malformed, or larger than 64 KiB, counts as empty and is replaced at
  the next showing; a file that cannot be read leaves the session flag as
  the only cap.
- The record: `Note: PyBADS {version} was released {age}. Run
  pybads.check_for_updates() to see whether a newer version is
  available.`, with ` This is the last reminder for PyBADS {version}.` on
  the third showing, then `https://pypi.org/project/pybads/` on a line of
  its own and an empty line, as a tip ends. `{age}` is "more than a year
  ago" up to the second anniversary, then "more than N years ago".
- `pybads/_update_check.py`: `check_for_updates(*, timeout=5.0) ->
  UpdateCheck`, exported as `pybads.check_for_updates`, as PyVBMC's: a GET
  of `https://pypi.org/pypi/pybads/json` through `urllib.request`,
  imported inside the function, with the `User-Agent`
  `pybads/<installed version> (check_for_updates)`; the latest release is
  the highest final `X.Y.Z` with a file that is not yanked, or PyPI's
  `info.version` in a reply without `releases`; one printed message; the
  update command from the distribution's `INSTALLER`; `ValueError` only
  for an invalid `timeout`.
- `pybads/testing/conftest.py`: autouse fixtures, one for the session
  (the fixtures of a module or a session are set up before a test's own)
  and one for each test, mark the reminder as shown and point its state
  file into pytest's temporary directory, so that no test depends on the
  calendar or writes the user's cache directory.

## Records

`CHANGELOG.md` (Added, "Update reminders"); the FAQ's question "How do I
know whether a newer version of PyBADS exists?" under "Installing PyBADS"
and its answer on silencing PyBADS; the API page
`docsrc/source/api/functions/check_for_updates.rst`;
`docsrc/source/installation.rst`; the description of `show_tips`;
KD-B2-3 of `pybads/bads/README.md`; `AGENTS.md` (the release date in the
release's steps, the network access, the FAQ's new label); the skill
(`skills/pybads/SKILL.md`); `dev/README.md`.

## Checklist

- [x] Branch `feat-update-reminders` from `dev-next` (`8a9a31ac`).
- [x] Fingerprint of the parent, `8a9a31ac` (Windows, NumPy 2.5.3, SciPy
  1.18.1, one BLAS thread, gpyreg `v1.4.0`): `093cb1d05a16d889`.
- [x] `pybads/_release.py`, `pybads/bads/_release_reminder.py`, the slot
  in `_init_mesh_`, `consider_runtime_tip`'s flag, the option's
  description.
- [x] `pybads/_update_check.py` and the export.
- [x] `pybads/testing/conftest.py`.
- [x] Tests: `pybads/testing/bads/test_release_reminder.py` and
  `pybads/testing/test_update_check.py`, ported from PyVBMC's (PyVBMC's
  cases for the calibration reminder, resumed runs and the log file
  dropped: PyBADS has none of them). The port found that `_cache_root`
  raised when the home directory could not be found: it returns `None`
  there, and ignores a relative `XDG_CACHE_HOME`, as the XDG specification
  asks.
- [x] Records (above).
- [x] Gates:
  - [x] the focused tests (with `test_runtime_tips.py`: 180 passed, 2
    skipped, the forks, on Windows), then the whole suite (1378 passed,
    2 skipped; Windows, Python 3.12, gpyreg `v1.4.0`);
  - [x] the fingerprint, `replay.py check` and the oracles' `--against`
    identical to `8a9a31ac`: at `0e666b2a` (Windows, NumPy 2.5.3, SciPy
    1.18.1, OpenBLAS's Haswell kernels, one BLAS thread, gpyreg `v1.4.0`),
    the fingerprint `093cb1d05a16d889`, as at the parent; `replay.py
    check`, 8 runs of 8 identical; the oracles' `--check --exact
    --against` a `--dump` of `8a9a31ac`, 1056 outputs of 1056 identical;
  - [x] the documentation built, with the example notebooks copied in as
    `make github` does, with no warning at all (`0e666b2a`);
  - [x] one call of `pybads.check_for_updates()` against PyPI, from the
    development install (2026-09-30): "a development version; the latest
    release is 1.1.0", and `import pybads` left `urllib.request`
    unimported.
- [x] Review (`/doublecheck`, 2026-09-30): three reviewers (the reminder,
  `check_for_updates()`, the documentation). Resolved: the tests of
  `check_for_updates()` use pytest's `monkeypatch` rather than
  pytest-mock's `mocker`, which no other shipped test uses and the
  conda-forge recipe need not install; the release's steps in
  `development.rst` set `RELEASE_DATE`; `make -C examples/scripts run`
  sets `PYBADS_NO_UPDATE_REMINDER`, so that the outputs that ship with a
  release show no reminder; a session-scoped fixture keeps the reminder
  inert for the fixtures of a module or a session; the FAQ's
  reproducibility answer, Example 1's text, `dev/TODO.md`'s item on
  PyVBMC's review of the tips and the tips' plan name the reminder; what
  counts as a showing is recorded above; wording. The whole suite after
  the fixes, 1379 passed, 2 skipped; the documentation built with no
  warning.
- [ ] Merge into `dev-next` (PI).
