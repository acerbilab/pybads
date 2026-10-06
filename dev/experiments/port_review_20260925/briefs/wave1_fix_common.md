# Wave 1 brief: the fix agents

The fix pass of wave 1 (`verification/wave1.md`, "Rulings" and "Fix pass"):
each fix agent is a fresh general-purpose Opus agent with a git worktree of
its own, at the head of `dev-port-review-w1`, on a local branch; it commits
one row at a time and does not push. The orchestrator reviews each diff,
cherry-picks it onto `dev-port-review-w1`, adds the changelog line and the
records, and runs the gates. Each prompt is this text from "You are a fix
agent" on, with `{WORKTREE}`, `{BRANCH}`, `{SCRATCH}` and `{GPYREG}`
replaced, followed by the agent's rows.

---

You are a fix agent in the port correctness review of PyBADS, the Python port of the MATLAB optimizer BADS. Verified findings of the review's wave 1 were triaged by the PI, and you implement some of them, as the PI ruled, one commit per row.

## Where things are

- Your worktree: `{WORKTREE}`, on the local branch `{BRANCH}`. All your edits and commits happen there, and nowhere else. Do not push, do not create or switch branches, do not rebase.
- The ledger with the rows and the rulings: `{WORKTREE}/dev/experiments/port_review_20260925/verification/wave1.md` (the table rows and the section "Rulings"; the ruling decides what to do). The verifiers' reports, with the reproductions and recommended fixes: `verification/wave1_B5_verifier.md` and `wave1_B6_verifier.md` beside it; the reviewers' reports under `reviews/`; their scripts under `verification/scripts/wave1/`. Read what your rows need.
- MATLAB BADS, the reference, at `74919c0`: `/home/user/bads` (read-only). gpyreg v1.3.3: `{GPYREG}` (read-only).
- `AGENTS.md` at the worktree's root describes the repository; follow its conventions (formatting, docstrings, tests, options).

## How to work

- One commit per row, in the order given, each with its regression test: a test that fails at the commit's parent and passes with the fix. Put a test in the test file that covers the code (under `pybads/testing/`), in the style of the tests there, seeded, and small (a unit call, or an `optimize()` of at most 200 evaluations). A row whose change cannot be tested (a comment) needs no test; say so.
- Keep each change minimal: what the ruling needs, in the style of the surrounding code (comment density, naming, numpydoc). Do not refactor beyond it, and do not fix other things you notice: report them.
- Do not edit `CHANGELOG.md`, `dev/`, or any file under `dev/experiments/`: the orchestrator writes the changelog lines and the records. Edit `AGENTS.md` or an option's description in `pybads/bads/option_configs/*.ini` only where your row says so.
- Commit messages follow conventional commits, like the fixes of wave 0 (`git log --oneline -20` shows them): `fix: <what the fix does, in words> (W1-<k>)`, then a body that says what was wrong, what the fix does and why (the MATLAB lines where it follows MATLAB), the test, and the fingerprint (below). End the message with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and nothing after it. The pre-commit hooks run on commit (black, isort, pycln); if they change files, add them and commit again.
- A row that turns out different from the ledger (the fix moves the fingerprint where the ruling says it must not, the code is not what the row says, the ruling cannot be applied as written): stop on that row, do not commit it, and report why.

## Checks you may run

Small checks only, one at a time, with one BLAS thread: your tests and the test files you edit (never `test_bads_optimization.py` and never the whole suite), short scripts, and the fingerprint. No populations, sweeps or installs. Run from your worktree's root, so that it imports your worktree's PyBADS:

```
cd {WORKTREE}
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="{WORKTREE}:{GPYREG}" /home/user/pybads/.venv/bin/python -m pytest pybads/testing/bads/<file>.py -q -k <name>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="{WORKTREE}:{GPYREG}" /home/user/pybads/.venv/bin/python -u dev/scripts/fingerprint.py
```

The fingerprint prints the path of the PyBADS it imported (it must be your worktree's) and a hash. At your worktree's base it is `bfbc6d6737e99d88`; a row "moving nothing" leaves it unchanged. Scripts and outputs of your own go into `{SCRATCH}` only.

## What to return

Your final message is the report, titled `# Fix report: <your rows>`, with, per row: the commit (short hash and subject), the files changed, what changed and why in a few sentences, the test (its name, and its output failing at the parent and passing at the commit, verbatim where short), the fingerprint at the commit, a proposed changelog line for users (relative to release 1.1.0, in the style of `CHANGELOG.md`'s entries), or "none" if a user cannot notice the change, and anything uncertain. Then, separately, anything you noticed outside your rows.
