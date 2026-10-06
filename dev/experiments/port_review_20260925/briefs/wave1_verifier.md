# Wave 1 brief: the verifier of a slice

One fresh Opus agent per slice (B5, B6), after both reports of the slice
are saved, with the placeholders of `wave1_common.md` replaced and
`{SLICE}` set. The verifier did not write either report.

---

You are a verifier in an independent correctness review of PyBADS, the Python port of the MATLAB optimizer BADS. Two reviewers of the slice **{SLICE}** reported findings, one on the internal-correctness track and one comparing with MATLAB BADS. Your job is to verify each finding independently: establish whether it is true of the code, reproduce it with your own small check or a second reading of both sources, date it, and classify it. Where the two reports describe the same behavior, say so and verify it once. You do not fix anything, and you do not take a reviewer's word for anything: read the code yourself.

## Where things are

- The reports: `{REPORT_INTERNAL}` and `{REPORT_COMPARISON}`. The reviewers' check scripts: `{REVIEWER_SCRIPTS}`; read them to understand a reproduction, but write your own checks.
- PyBADS at the review's freeze `95da7f1`: `{PYBADS_REVIEW}`. MATLAB BADS at `74919c0`: `{BADS}`. gpyreg v1.3.3: `{GPYREG}`.
- The known-differences sheet `{SHEET}`: use it to classify; if a finding contradicts an entry, say so.

## Classifications

For each finding, one of: **confirmed port discrepancy** (PyBADS differs from MATLAB and MATLAB is right, or the difference is unintended); **confirmed shared defect** (both are wrong); **confirmed defect** (internal track, where MATLAB does not bear on it); **confirmed, inert** (true of the code, no consequence today; say why); **intentional difference** (missing from the sheet; say what record or comment makes it intentional); **listed as intentional, but the justification does not hold**; **needs MATLAB** (only a MATLAB run can settle it); **design question** (the code does what it is written to do, and whether that is right is a decision; state the decision and the options); **not a defect** (the report misread the code; say where).

Date every confirmed finding from both histories (`git log -L`): whether the Python ever matched MATLAB, whether MATLAB changed after the Python lines were written, or whether the two never agreed.

## Checks you may run, and rules

As in the reviewers' brief (`wave1_common.md`, "Checks you may run" and "Rules"), with `{SCRATCH}` your own scratch directory; you may open the two reports, the sheet and the reviewers' scripts; no other file under `dev/`.

## What to return

Your final message is the deliverable, titled `# Wave 1 verification: {SLICE}`, with:
1. A summary table: finding (report and number), classification, reached at default options (yes/no, and at which uncertainty level), dating (one phrase), confidence.
2. Per finding: what you checked and how (script name and its output, verbatim where short), the lines at `95da7f1` and in MATLAB, the dating, where you agree or disagree with the report and why, the consequence in your own measurement, and a **recommended disposition** (fix / keep and document / correct the record / needs MATLAB / decide the design), with what a fix would touch and whether it would change runs at default options, and so which gate it needs (the fingerprint of an unchanged run, or a population comparison with a configuration that reaches it).
3. Anything new you met while verifying, clearly separated and marked unverified.
