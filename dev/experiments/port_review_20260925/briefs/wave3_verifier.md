# Wave 3 brief: the verifier of a slice

One fresh Opus agent per slice (B3, B4), after both reports of the slice
are saved, with the placeholders of `wave3_common.md` replaced and
`{SLICE}` set. The verifier did not write either report. Made from
`wave2_verifier.md`; what changed: the revision, `8aecb6a` (`dev-next`
after wave 2's fix pass, #76), with the commits of the review's fix passes
named; the items kept from the reviewers (`{KEPT_ITEMS}`), which the plan's
"Wave 3 pickup", step 4, names: the open rows of the survey's candidate
table in the slice (among them rows that wave 0's fixes may have closed,
whose status the survey does not yet say), the differences the preparatory
agent saw in passing, and what waves 0 to 2 left to the slice under "Found
while verifying" and "Found while fixing", among them the two findings of
wave 2 fixed in these slices' code (W2-16, W2-29), which the verifier
checks as fixed code. The kept items of each slice are in
`wave3_kept_B3.md` and `wave3_kept_B4.md`.

---

You are a verifier in an independent correctness review of PyBADS, the Python port of the MATLAB optimizer BADS. Two reviewers of the slice **{SLICE}** reported findings, one on the internal-correctness track and one comparing with MATLAB BADS. Your job is to verify each finding independently: establish whether it is true of the code, reproduce it with your own small check or a second reading of both sources, date it, and classify it. Where the two reports describe the same behavior, say so and verify it once. You do not fix anything, and you do not take a reviewer's word for anything: read the code yourself.

## Where things are

- The reports: `{REPORT_INTERNAL}` and `{REPORT_COMPARISON}`. The reviewers' check scripts: `{REVIEWER_SCRIPTS}`; read them to understand a reproduction, but write your own checks.
- PyBADS under review, at `8aecb6a` (`dev-next` after the fix passes of the review's waves 0 to 2: `0c56d86`, `e004c79`, `fef6c14` and `8aecb6a`, each squash-merged), with its complete history: `{PYBADS_REVIEW}`. MATLAB BADS at `74919c0`: `{BADS}`. gpyreg v1.3.3: `{GPYREG}`.
- The known-differences sheet `{SHEET}`, with Python lines at `8aecb6a`: use it to classify; if a finding contradicts an entry, say so.

## Items kept from the reviewers

The review's own records hold the items below, which belong to this slice and which the reviewers did not receive. Each is quoted from its record, with the record named; the line numbers in a quotation are those of the revision the record names, not necessarily `8aecb6a`, and an item may already have been changed by the fix passes. For each item, say whether a report covers it (then verify it once, with that finding, and say so), and otherwise verify it as a finding of its own, labelled as given (`{SLICE}-K1`, `{SLICE}-K2`, ...): whether it holds at `8aecb6a`, reproduced, dated and classified like the findings.

{KEPT_ITEMS}

## Classifications

For each finding, one of: **confirmed port discrepancy** (PyBADS differs from MATLAB and MATLAB is right, or the difference is unintended); **confirmed shared defect** (both are wrong); **confirmed defect** (internal track, where MATLAB does not bear on it); **confirmed, inert** (true of the code, no consequence today; say why); **intentional difference** (missing from the sheet; say what record or comment makes it intentional); **listed as intentional, but the justification does not hold**; **needs MATLAB** (only a MATLAB run can settle it); **design question** (the code does what it is written to do, and whether that is right is a decision; state the decision and the options); **not a defect** (the report misread the code; say where); and, for a kept item only, **no longer holds** (true at the revision of its record, not at `8aecb6a`; say which commit changed it).

Date every confirmed finding from both histories (`git log -L`): whether the Python ever matched MATLAB, whether MATLAB changed after the Python lines were written, or whether the two never agreed.

## Checks you may run, and rules

As in the reviewers' brief, with `{SCRATCH}` your own scratch directory; you may open the two reports, the sheet and the reviewers' scripts; no other file under `dev/`.

{CHECKS_AND_RULES}

## What to return

Your final message is the deliverable, titled `# Wave 3 verification: {SLICE}`, with:
1. A summary table: finding (report and number, or kept item), classification, reached at default options (yes/no, and at which uncertainty level: 0 deterministic, 1 noise inferred, 2 `specify_target_noise`), dating (one phrase), confidence.
2. Per finding: what you checked and how (script name and its output, verbatim where short), the lines at `8aecb6a` and in MATLAB, the dating, where you agree or disagree with the report and why, the consequence in your own measurement, and a **recommended disposition** (fix / keep and document / correct the record / needs MATLAB / decide the design), with what a fix would touch and whether it would change runs at default options, and so which gate it needs (the fingerprint of an unchanged run, or a population comparison with a configuration that reaches it).
3. Anything new you met while verifying, clearly separated and marked unverified.
