# Developer notes

`dev/` is for human review: major findings, proposals, discussions and
consolidated decisions. Use dated names (`YYYY-MM-DD-short-slug.md`) for
these notes. Update or consolidate a related narrative instead of creating
a top-level file for each agent, working session or experiment phase.
Separate notes are appropriate for genuinely separate topics.

- `TODO.md` lists the open work.
- `plans/` holds implementation plans, checklists and execution worklogs.
  Keep them current while the work is open, updating the existing file in
  place, and retain them afterwards.
- `results/` holds detailed findings and experiment writeups, with dated
  names and no further date or campaign subdirectory. A top-level note
  summarizes the evidence and decisions and links to them.
- `experiments/` holds the machine-readable evidence that a result cites.
- `scripts/` holds developer tooling that is not part of the package or the
  test suite. Run it from the repository root with the project venv, as
  `python dev/scripts/<name>.py`. Its output goes under `scripts/runs/`,
  which is gitignored: a result that matters is summarized in a plan or a
  result, not committed raw. A machine that keeps raw artifacts there lists
  them in the gitignored `scripts/runs/LOCAL.md`; tracked documents point
  at that file and never say "this machine".

A directory is created with its first file. These are maintainer records,
not user documentation: `docs/` is gitignored Sphinx output published to
`gh-pages`, so it cannot hold source notes.

## Index

- [Codebase survey](results/2026-09-23-codebase-survey.md) — failures
  observed in the test suite at `273a5b7`, the candidate defects found by a
  read of the code (not verified), and the tests that check less than they
  appear to. The starting point of the deferred bug hunt in `TODO.md`.
