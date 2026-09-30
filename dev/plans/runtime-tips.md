# Runtime tips

Created 2026-09-30, for the release of 1.5.0 (`dev/TODO.md`, "Runtime tips,
as in PyVBMC."). A run with the iteration display may print a short tip,
with a link to the documentation, before its first iteration line. The
design is PyVBMC's (`acerbilab/pyvbmc`, branch `dev-next`, from
`1c3d4f25`: `pyvbmc/vbmc/_runtime_tips.py`, `_tip_catalog.py` and its plan
`dev/plans/runtime-tips.md`), copied, since PyBADS does not import PyVBMC,
and simplified: PyBADS has no calibration reminder that would share the
slot, and a `BADS` object runs once, with no save or resume.

## Decisions (PI, 2026-09-30)

- The tips come with 1.5.0; PyBADS does not wait for the PI's review of
  PyVBMC's tips, whose outcome a later edit of PyBADS's copy can follow.
- They show only with the iteration lines, `display` `"iter"`, `"all"` or
  `"full"`, and `show_tips=False` turns them off.
- Every message of a run goes to the `BADS` logger, the tips too, so a
  handler that a user attaches, such as a file, receives them.
- A run in a process of its own, as in a script, a cluster job or a pool of
  workers, is the first of its session and shows a tip; a forked worker
  reshuffles, so that workers forked together do not show the same one.
- The example notebooks, rerun before the release, show the tips that
  their runs draw.
- "What's new" in `README.md` and `index.rst` mentions them.
- No filtering by what the run uses (a tip on periodic variables shows in
  a run that has them).
- The wording of the tips below.

## Behaviour

- Basic option `show_tips = True`; an option whose default is a boolean
  takes only a boolean (`Options.validate_boolean_options`).
- `BADS._init_mesh_` considers a tip once per `BADS` object, after the
  opening message ("Beginning optimization of a ... objective function")
  and before the column headers of the iteration display.
- A tip is eligible when `show_tips` is on, the `BADS` logger is enabled
  for INFO when the tip would be logged (the logger is shared by every
  `BADS` object, and the last one created sets its level), and a tip
  remains. The first eligible run of a Python session shows one, then
  every third (the eligible runs 1, 4, 7, ...); a run that is not eligible
  neither advances that count nor uses a tip.
- The catalogue is shuffled once per session, by a private
  `random.Random()` seeded from the operating system: no draw of a run, no
  NumPy stream and no state of the `random` module is touched, and
  `random_seed` does not fix which tip shows. Each tip shows at most once
  per session, then no more tips. After a low-frequency tip, the next is an
  ordinary one when one remains.
- The tip is one INFO record of the `BADS` logger: `Tip: <text>`, each URL
  on a line of its own, and an empty line.
- `os.register_at_fork` (POSIX) gives a forked child a new generator, a new
  lock and a new shuffle of the tips its parent has not shown.
- The catalogue, `pybads/bads/_tip_catalog.py`, is data: a tip's `id`,
  `text` (plain ASCII, no Markdown), `frequency` (`"normal"` or
  `"low_frequency"`) and `urls`. The scheduler,
  `pybads/bads/_runtime_tips.py`, reads none of its sizes or ids.

## Catalogue

Six ordinary tips and two low-frequency ones. The URLs are those of the
published documentation, built from `main`, so the FAQ's labels that they
link are coupled with it (`AGENTS.md`).

1. `multiple_starts`: "Run PyBADS from at least 10 different starting
   points, ideally dozens, depending on your problem; with x0=None, each
   run draws its start at random in the plausible box. Keep the run with
   the lowest fval: a single run can stop in a local minimum, and if
   several runs reach nearly the same fval, you can be more confident in
   the solution." FAQ, "How do I run PyBADS from several starting points?".
2. `plausible_bounds`: "Set plb and pub around where you expect the
   solution, a box you would bet contains it with over 90% probability.
   PyBADS draws its first points inside that box and scales its steps by
   it; lb and ub only bound the search." FAQ, "How do I choose plb and
   pub?".
3. `noisy_target`: "If your objective is noisy, for instance a negative
   log-likelihood estimated by simulation, set
   options['uncertainty_handling'] = True rather than relying on PyBADS's
   noise test at x0. PyBADS works best when the noise SD near the solution
   is about 1 or less; if you can estimate the SD of each evaluation, set
   options['specify_target_noise'] = True and return (f, sd)." FAQ, "Noisy
   objective function".
4. `fixed_variables`: "To hold a parameter at a known value, set its four
   bounds (lb, ub, plb and pub) to that value, and its entry of x0 to the
   same value or NaN. PyBADS optimizes only the other parameters and still
   passes all of them to your objective, so your code needs no change."
   FAQ, "Can I set lb = ub for some variable to fix it to a given value?".
5. `periodic_vars`: "For angles and other periodic parameters, list their
   indices (from 0) in options['periodic_vars'], with the period as the
   hard bounds (e.g. lb = -np.pi, ub = np.pi), usually as the plausible
   bounds too. PyBADS then treats the two bounds as the same point and can
   move across them." FAQ, "Does PyBADS support periodic variables, such
   as angles?".
6. `stop_reason`: "The message at the end of the run, also in
   optimize_result['message'], says why it ended. If it used up its budget
   (max_fun_evals, 500 * D by default, or max_iter) before settling, raise
   it; if it converged but the solution seems imprecise, the FAQ lists
   options that make it search longer." FAQ, "On some problems, PyBADS
   seems to get stuck and stop too early...".
7. `pyvbmc` (low frequency): "If your objective is a negative
   log-likelihood, you can also estimate the uncertainty over the
   parameters and the model evidence: run PyVBMC on the same model and
   data, with PyBADS's solution as its starting point x0." FAQ, "I have
   run PyBADS on my problem. How do I run PyVBMC?".
8. `agent_skill` (low frequency): "If you work with a coding agent, give it
   the PyBADS skill from GitHub, or copy its skills/pybads folder into the
   agent's skill directory: it points the agent to the parts of PyBADS's
   documentation relevant to your task." The skill on GitHub.

## Records

`CHANGELOG.md` (Added, "Tips"); "What's new" in `README.md` and
`docsrc/source/index.rst`; the FAQ's "How do I silence PyBADS, or send its
output elsewhere?" (`show_tips`) and "How do I make a run reproducible?"
(the tips' order is not the seed's); `AGENTS.md` (the randomness paragraph,
the FAQ's labels that the catalogue links); the catalogue of deliberate
differences (KD-B1-1, KD-B1-4, KD-B2-3); `dev/TODO.md`.

## Verification

- `pybads/testing/bads/test_runtime_tips.py`: the cadence (runs 1, 4, 7),
  each tip once and then none, the spacing of low-frequency tips, runs that
  are not eligible (tips off, a logger above INFO, a display other than the
  iteration display) neither counting nor using a tip, the record's format,
  the shipped catalogue (unique ids, known frequencies, ASCII text, URLs of
  the documentation), the fork hook, and whole runs: the tip between the
  opening message and the column headers, a seeded run with a tip equal to
  the same run without, and NumPy's and the `random` module's states left
  as they were.
- The whole suite; `dev/scripts/fingerprint.py`, `replay.py check` and the
  oracles' `--against`, identical to the parent commit, since the runs of
  those tools have no iteration display.
- The example notebooks rerun (`AGENTS.md`, "Setup and commands").

## Checklist

- [x] Catalogue, scheduler, option, call site.
- [x] Tests: `test_runtime_tips.py`, 17 passed; the whole suite, 1215
  passed (Windows, gpyreg `v1.4.0`).
- [ ] The gates against the parent commit.
- [x] Records.
- [ ] Notebooks rerun.
