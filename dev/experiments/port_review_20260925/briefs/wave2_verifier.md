# Wave 2 brief: the verifier of a slice

One fresh Opus agent per slice (B1, B2), after both reports of the slice
are saved, with the placeholders of `wave2_common.md` replaced and
`{SLICE}` set. The verifier did not write either report. Made from
`wave1_verifier.md`; what changed: the revision, `fef6c14`; the sections
"Checks you may run" and "Rules" of `wave2_common.md` quoted in the prompt
(`{CHECKS_AND_RULES}`) instead of named, their rule on `dev/` extended to
the two reports and the reviewers' scripts; and the items kept from the
reviewers (`{KEPT_ITEMS}`), which the plan's "Wave 2 pickup", step 4,
names: the open rows of the survey's candidate table in the slice, the
differences the preparatory agent saw in passing, and what waves 0 and 1
left to the slice, each quoted from its record with the record named,
verified as a finding when neither report covers it, and otherwise matched
to the finding that does.

---

You are a verifier in an independent correctness review of PyBADS, the Python port of the MATLAB optimizer BADS. Two reviewers of the slice **{SLICE}** reported findings, one on the internal-correctness track and one comparing with MATLAB BADS. Your job is to verify each finding independently: establish whether it is true of the code, reproduce it with your own small check or a second reading of both sources, date it, and classify it. Where the two reports describe the same behavior, say so and verify it once. You do not fix anything, and you do not take a reviewer's word for anything: read the code yourself.

## Where things are

- The reports: `{REPORT_INTERNAL}` and `{REPORT_COMPARISON}`. The reviewers' check scripts: `{REVIEWER_SCRIPTS}`; read them to understand a reproduction, but write your own checks.
- PyBADS under review, at `fef6c14` (`dev-next` after the fix passes of the review's waves 0 and 1: `0c56d86`, `e004c79` and `fef6c14`): `{PYBADS_REVIEW}`. MATLAB BADS at `74919c0`: `{BADS}`. gpyreg v1.3.3: `{GPYREG}`.
- The known-differences sheet `{SHEET}`, with Python lines at `fef6c14`: use it to classify; if a finding contradicts an entry, say so.

## Items kept from the reviewers

The review's own records hold the items below, which belong to this slice and which the reviewers did not receive. Each is quoted from its record, with the record named; the line numbers in a quotation are those of the revision the record names, not necessarily `fef6c14`, and an item may already have been changed by the fix passes. For each item, say whether a report covers it (then verify it once, with that finding, and say so), and otherwise verify it as a finding of its own, labelled as given (`{SLICE}-K1`, `{SLICE}-K2`, ...): whether it holds at `fef6c14`, reproduced, dated and classified like the findings.

{KEPT_ITEMS}

## Classifications

For each finding, one of: **confirmed port discrepancy** (PyBADS differs from MATLAB and MATLAB is right, or the difference is unintended); **confirmed shared defect** (both are wrong); **confirmed defect** (internal track, where MATLAB does not bear on it); **confirmed, inert** (true of the code, no consequence today; say why); **intentional difference** (missing from the sheet; say what record or comment makes it intentional); **listed as intentional, but the justification does not hold**; **needs MATLAB** (only a MATLAB run can settle it); **design question** (the code does what it is written to do, and whether that is right is a decision; state the decision and the options); **not a defect** (the report misread the code; say where); and, for a kept item only, **no longer holds** (true at the revision of its record, not at `fef6c14`; say which commit changed it).

Date every confirmed finding from both histories (`git log -L`): whether the Python ever matched MATLAB, whether MATLAB changed after the Python lines were written, or whether the two never agreed.

## Checks you may run, and rules

As in the reviewers' brief, with `{SCRATCH}` your own scratch directory; you may open the two reports, the sheet and the reviewers' scripts; no other file under `dev/`.

{CHECKS_AND_RULES}

## What to return

Your final message is the deliverable, titled `# Wave 2 verification: {SLICE}`, with:
1. A summary table: finding (report and number, or kept item), classification, reached at default options (yes/no, and at which uncertainty level: 0 deterministic, 1 noise inferred, 2 `specify_target_noise`), dating (one phrase), confidence.
2. Per finding: what you checked and how (script name and its output, verbatim where short), the lines at `fef6c14` and in MATLAB, the dating, where you agree or disagree with the report and why, the consequence in your own measurement, and a **recommended disposition** (fix / keep and document / correct the record / needs MATLAB / decide the design), with what a fix would touch and whether it would change runs at default options, and so which gate it needs (the fingerprint of an unchanged run, or a population comparison with a configuration that reaches it).
3. Anything new you met while verifying, clearly separated and marked unverified.
