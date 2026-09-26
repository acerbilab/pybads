# Records of the port correctness review

The records of the review planned and logged in
[`dev/plans/port-correctness-review.md`](../../plans/port-correctness-review.md):
an independent reading of PyBADS against MATLAB BADS v1.1.3 (`74919c0`) and
against its own specification, in waves of fresh reviewers.

- `known_differences.md`: the known-differences sheet, the settled and
  deliberate differences between PyBADS and MATLAB BADS that a reviewer
  does not report as new, with the claims of the repository's records that
  did not check out. Its Python line citations are at `8aecb6a`, the
  revision of wave 3, carried from `ab4dded` to `95da7f1`, the freeze of
  wave 1, by `refresh_citations.py --base ab4dded`, from there to
  `fef6c14`, the revision of wave 2, by `--base 95da7f1`, and on by
  `--base fef6c14`; the claims keep the lines of `95da7f1`.
- `counterpart_map.md`: every MATLAB file outside the bundled GPML library,
  with its Python counterpart, or "unported", and the slice that owns it.
- `prep_report.md`: the preparatory agent's report: the corrections of the
  plan's slice table, what it left off the sheet and why, and its coverage.
- `reviews/<slice>_<track>.md`: each reviewer's report, saved verbatim from
  its final message under a header that says what it read. Nothing in a
  report is verified; the verification of a wave goes to
  `verification/wave<N>.md`.
- `briefs/`: the prompts of waves 1 to 3 (reviewers and verifiers), with
  placeholders for the paths, for a session away from the orchestrator's
  machine (the plan's "Wave 1 pickup" to "Wave 3 pickup").
- `verification/wave<N>.md`: the ledger of a wave, and
  `verification/wave<N>_<slice>_verifier.md` the reports of its verifiers,
  saved verbatim.
- `matlab_side_defects.md`: what the review finds wrong or questionable in
  MATLAB BADS itself, with PyBADS's disposition.
- `briefs/wave1_fix_common.md`: the brief of the fix agents of wave 1's fix
  pass; `fixes/`: their reports, saved verbatim.
- `extract_report.py`: saves a reviewer's final message verbatim from its
  transcript (copied from PyVBMC's review).
- `refresh_citations.py`: carries the `pybads/...:<line>` citations of a
  document from the commit at which they were checked to the working tree,
  and reports the cited lines that changed (adapted from PyVBMC's review).

The check scripts and outputs of wave 0 are kept on the machine that ran
them (`dev/scripts/runs/LOCAL.md`). Those of wave 1, which ran in a cloud
session, are under `verification/scripts/wave1/<slice>_<track>/` and
`verification/scripts/wave1/<slice>_verifier/`, and those of the fix agents
of its fix pass under `verification/scripts/wave1/fix_<agent>/` (A to J, the
letters of `fixes/`), without the copies of package and test files that
they made for their checks at a parent commit, which git holds; all were
formatted by the pre-commit hooks after they ran, and the reports cite them
at the sandbox's scratch paths. The raw outputs of the fix pass's inject
gate are in `verification/wave1_fixpass/inject/`; its populations are
summarized by the comparisons beside it, and only the final one is kept
whole, as the Linux reference `population_linux_wave1_20260926`, which
wave 2's `population_linux_wave2_20260926` replaces.

Those of wave 2, also in a cloud session, are under
`verification/scripts/wave2/<slice>_<track>/` and
`verification/scripts/wave2/<slice>_verifier/`, with the orchestrator's
check under `verification/scripts/wave2/orchestrator/`, formatted by the
pre-commit hooks in the same way. `reviews/B1_comparison_history.md` is the
B1 comparison reviewer's re-dating of its report on the complete history,
saved verbatim beside the report. The fix agents of wave 2's fix pass (A to
F, the letters of `fixes/`) have theirs under
`verification/scripts/wave2/fix_<agent>/`, without their copies of a
parent's tree, and the orchestrator's scripts of the pass (the
cherry-picks, the changelog lines, the fingerprint at every commit, the
comparisons of populations run by run) are with its check. The pass's
comparisons are in `verification/wave2_fixpass/`, and its last population
is kept whole as the Linux reference `population_linux_wave2_20260926`.

Those of wave 3, in a cloud session too, are under
`verification/scripts/wave3/<slice>_<track>/` and
`verification/scripts/wave3/<slice>_verifier/`, formatted by the
pre-commit hooks in the same way. The items of the records kept from wave
3's reviewers and given to its verifiers are quoted in
`briefs/wave3_kept_B3.md` and `briefs/wave3_kept_B4.md`.
