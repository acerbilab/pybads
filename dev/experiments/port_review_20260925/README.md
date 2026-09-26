# Records of the port correctness review

The records of the review planned and logged in
[`dev/plans/port-correctness-review.md`](../../plans/port-correctness-review.md):
an independent reading of PyBADS against MATLAB BADS v1.1.3 (`74919c0`) and
against its own specification, in waves of fresh reviewers.

- `known_differences.md`: the known-differences sheet, the settled and
  deliberate differences between PyBADS and MATLAB BADS that a reviewer
  does not report as new, with the claims of the repository's records that
  did not check out. Its Python line citations are at `ab4dded`.
- `counterpart_map.md`: every MATLAB file outside the bundled GPML library,
  with its Python counterpart, or "unported", and the slice that owns it.
- `prep_report.md`: the preparatory agent's report: the corrections of the
  plan's slice table, what it left off the sheet and why, and its coverage.
- `reviews/<slice>_<track>.md`: each reviewer's report, saved verbatim from
  its final message under a header that says what it read. Nothing in a
  report is verified; the verification of a wave goes to
  `verification/wave<N>.md`.
- `extract_report.py`: saves a reviewer's final message verbatim from its
  transcript (copied from PyVBMC's review).

The reviewers' check scripts and outputs are kept on the machine that ran
them (`dev/scripts/runs/LOCAL.md`).
