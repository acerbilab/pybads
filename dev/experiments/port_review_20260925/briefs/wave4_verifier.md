# Wave 4 brief: the verifier of a slice

One fresh Opus agent per slice (B7, O), after the reports of the slice are
saved, with the placeholders of `wave4_common.md` replaced and `{SLICE}`
set. The verifier did not write any report of the slice. Made from
`wave3_verifier.md`; what changed: the revision, `0d866e8` (`dev-next`
after wave 3's fix pass, #77), with the commits of the review's fix passes
named; slice O's one report, the third reader's, for which the opening
paragraph and the list of reports take their one-report form (below); and
the items kept from the reviewers (`{KEPT_ITEMS}`), which the plan's "Wave
4 pickup", step 4, names: the open rows of the survey's candidate table in
the slice, `init_sobol`'s seed among them, with wave 2's doublecheck of a
start that gives `u0 = -1`; the differences the preparatory agent saw in
passing (none belongs to B7 or O); and what waves 0 to 3 left to the slice
under "Found while verifying" and "Found while fixing", among them W0-18,
the doubling of the design, which wave 0's ruling leaves to this wave. The
kept items of each slice are in `wave4_kept_B7.md` and `wave4_kept_O.md`.

For B7, the opening paragraph and the list of reports read as below. For O
they read: "The third reader of the slice **O** reported findings, having
re-derived each formula and compared it with MATLAB BADS. Your job is to
verify each finding independently: [the rest as below, without its
sentence on two reports]" and "- The report: `{REPORT_THIRD}`. The
reviewer's check scripts: `{REVIEWER_SCRIPTS}`; read them to understand a
reproduction, but write your own checks."

---

You are a verifier in an independent correctness review of PyBADS, the Python port of the MATLAB optimizer BADS. Two reviewers of the slice **{SLICE}** reported findings, one on the internal-correctness track and one comparing with MATLAB BADS. Your job is to verify each finding independently: establish whether it is true of the code, reproduce it with your own small check or a second reading of both sources, date it, and classify it. Where the two reports describe the same behavior, say so and verify it once. You do not fix anything, and you do not take a reviewer's word for anything: read the code yourself.

## Where things are

- The reports: `{REPORT_INTERNAL}` and `{REPORT_COMPARISON}`. The reviewers' check scripts: `{REVIEWER_SCRIPTS}`; read them to understand a reproduction, but write your own checks.
- PyBADS under review, at `0d866e8` (`dev-next` after the fix passes of the review's waves 0 to 3: `0c56d86`, `e004c79`, `fef6c14`, `8aecb6a` and `0d866e8`, each squash-merged), with its complete history: `{PYBADS_REVIEW}`. The commits of those fix passes that the records cite are not ancestors of `0d866e8` but are in the repository (`git show <commit>`). MATLAB BADS at `74919c0`: `{BADS}`. gpyreg v1.3.3: `{GPYREG}`.
- The known-differences sheet `{SHEET}`, with Python lines at `0d866e8`: use it to classify; if a finding contradicts an entry, say so.

## Items kept from the reviewers

The review's own records hold the items below, which belong to this slice and which the reviewers did not receive. Each is quoted from its record, with the record named; the line numbers in a quotation are those of the revision the record names, not necessarily `0d866e8`, and an item may already have been changed by the fix passes. For each item, say whether a report covers it (then verify it once, with that finding, and say so), and otherwise verify it as a finding of its own, labelled as given (`{SLICE}-K1`, `{SLICE}-K2`, ...): whether it holds at `0d866e8`, reproduced, dated and classified like the findings.

{KEPT_ITEMS}

## Classifications

For each finding, one of: **confirmed port discrepancy** (PyBADS differs from MATLAB and MATLAB is right, or the difference is unintended); **confirmed shared defect** (both are wrong); **confirmed defect** (internal track, where MATLAB does not bear on it); **confirmed, inert** (true of the code, no consequence today; say why); **intentional difference** (missing from the sheet; say what record or comment makes it intentional); **listed as intentional, but the justification does not hold**; **needs MATLAB** (only a MATLAB run can settle it); **design question** (the code does what it is written to do, and whether that is right is a decision; state the decision and the options); **not a defect** (the report misread the code; say where); and, for a kept item only, **no longer holds** (true at the revision of its record, not at `0d866e8`; say which commit changed it).

Date every confirmed finding from both histories (`git log -L`): whether the Python ever matched MATLAB, whether MATLAB changed after the Python lines were written, or whether the two never agreed.

## Checks you may run, and rules

As in the reviewers' brief, with `{SCRATCH}` your own scratch directory; you may open the reports, the sheet and the reviewers' scripts; no other file under `dev/`.

{CHECKS_AND_RULES}

## What to return

Your final message is the deliverable, titled `# Wave 4 verification: {SLICE}`, with:
1. A summary table: finding (report and number, or kept item), classification, reached at default options (yes/no, and at which uncertainty level: 0 deterministic, 1 noise inferred, 2 `specify_target_noise`), dating (one phrase), confidence.
2. Per finding: what you checked and how (script name and its output, verbatim where short), the lines at `0d866e8` and in MATLAB, the dating, where you agree or disagree with the report and why, the consequence in your own measurement, and a **recommended disposition** (fix / keep and document / correct the record / needs MATLAB / decide the design), with what a fix would touch and whether it would change runs at default options, and so which gate it needs (the fingerprint of an unchanged run, or a population comparison with a configuration that reaches it).
3. Anything new you met while verifying, clearly separated and marked unverified.
