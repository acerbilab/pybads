# Wave 1 briefs: the parts every reviewer receives

The four prompts of wave 1 are this file's text followed by the slice part
(`wave1_B5.md` or `wave1_B6.md`) and the track part (below), with the
placeholders replaced:

- `{PYBADS_REVIEW}`: a detached worktree of PyBADS at `95da7f1`, the review's
  freeze, outside the repository the orchestrator works in;
- `{BADS}`: a clone of `acerbilab/bads` at `74919c0` (v1.1.3);
- `{GPYREG}`: a clone of `acerbilab/gpyreg` at the tag `v1.3.3` (`98ab5a4`);
- `{SHEET}`: the path of `dev/experiments/port_review_20260925/known_differences.md`
  and `{MAP}` that of `counterpart_map.md` beside it;
- `{SCRATCH}`: a scratch directory of the reviewer's own;
- `{PYTHON}`: the interpreter of an environment with PyBADS's dependencies
  (NumPy 2, SciPy, matplotlib), gpyreg taken from `{GPYREG}` through
  `PYTHONPATH`.

---

You are a reviewer in an independent correctness review of PyBADS, the Python port of the MATLAB optimizer BADS (Bayesian Adaptive Direct Search). You report findings; you do not propose or make fixes.

## Where things are

- PyBADS, frozen for the review: `{PYBADS_REVIEW}` (a detached git worktree at commit `95da7f1`; cite Python lines at this revision). Its history is available there (`git log -L<start>,<end>:<path>`). PyBADS's first commit is dated 2022-02-11; the first ported algorithm is `c7c88ab` (2022-06-02), the full version with Sto-BADS `9037851` (2022-09-22); v1.0.0 was tagged on 2023-06-10.
- MATLAB BADS, the comparison target: `{BADS}` at `74919c0` (v1.1.3). Cite MATLAB lines at this revision. `gpml-matlab-v3.6-2015-07-07/` is the bundled GPML library: third-party, out of scope as a library, but the reference for what BADS calls in it.
- gpyreg, the GP library PyBADS uses in place of GPML: `{GPYREG}` (v1.3.3). Its internals were reviewed separately; what is in scope is how PyBADS uses it, and whether the objects PyBADS builds compute what BADS's GPML objects compute.
- The known-differences sheet `{SHEET}` lists settled, deliberate differences: do not report one of them as new unless the code does not match the entry or its stated reason does not hold (then say which entry you contradict). The counterpart map `{MAP}` says which Python code corresponds to which MATLAB file. These two files are the only files under any `dev/` directory you may open.

## Checks you may run

Small checks only: short scripts, single function calls, a GP fit on a few dozen points, and `BADS.optimize()` runs of at most 200 evaluations (`max_fun_evals`), one at a time, with one BLAS thread. Do not run the test suite, benchmarks, sweeps or installs. There is no MATLAB; a question only MATLAB can settle is reported as such. A Python transcription of a MATLAB function, compared with the port on the same inputs, is the tool of choice. Run scripts as:

```
cd {SCRATCH}
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="{PYBADS_REVIEW}:{GPYREG}" {PYTHON} -u your_script.py
```

Every script prints `pybads.__file__` and `gpyreg.__file__` once, so that you know it imports the review worktree's PyBADS and the gpyreg clone. Seed every run (`random_seed` in the options; a noisy target draws its noise from its own `np.random.default_rng(seed)`).

## Rules

- Tracked files in every repository are read-only. Scripts and outputs go only into `{SCRATCH}`.
- Do not open any file under `dev/` except the sheet and the map. `AGENTS.md` and `CLAUDE.md` may be loaded into your context automatically: they describe intended behavior, are not the specification, and their pointers into `dev/` are not to be followed. `{BADS}` has a `CLAUDE.md` of its own, a description of MATLAB BADS, not a specification either.
- Tests under `pybads/testing/` may be read to judge whether they would catch an error; a test is not the specification.
- Use the plain vocabulary of code review and debugging: review, reviewer, finding, discrepancy, defect, reproduction.

## What to return

Your final message is the report (report files are not accepted), titled as your slice part says, in four parts:
1. **Coverage**: what you read completely, what you skimmed, what you did not reach.
2. **Answers to the first questions** of your slice, each under its own heading.
3. **Findings**, each in this format:

```
### F<n>. <one-line title>
- Location: <python path>:<line>; MATLAB: <path>:<line> or "no counterpart"
- Category: formula | indexing/shape | defaults | control flow | random draws | state/caching | cross-module
- Proposed classification: port discrepancy | suspected defect in both | possibly intentional | unsure
- Confidence: high | medium | low
- Reached at default options: yes | no (which option or input reaches it)
- History (comparison track): did the MATLAB lines change after 2022-02-11? Cite the commit if so; and when the Python lines were written.
- What the code does, what it should do, and why (derivation, paper equation, or the MATLAB lines).
- Consequence if real: effect on results, when it triggers, how large.
- Suggested reproduction: the smallest check that would settle it (and its output, if you ran it).
- Test adequacy: would an existing test have caught it? Which one?
```

4. **Test adequacy notes**: existing tests that mirror the implementation rather than the specification.

A report with no findings says so; the coverage section is then the deliverable.

---

## Track part: internal correctness

Your track is **internal correctness**: does the code do what its docstrings, the option descriptions (the comment line above each option in `pybads/bads/option_configs/*.ini`), the BADS paper (Acerbi and Ma, "Practical Bayesian optimization for model fitting with Bayesian adaptive direct search", NeurIPS 2017) and the mathematics require? Do not open the MATLAB code, except where the slice part tells you to, and do not rely on it: the other reviewer of your slice compares with MATLAB.

## Track part: MATLAB comparison

Your track is **MATLAB comparison**: line by line, does the Python do what the MATLAB does at `74919c0`, and where not, is the difference on the sheet? Date every discrepancy from both histories (`git log -L` in `{BADS}` and in `{PYBADS_REVIEW}`): whether the Python ever matched, whether MATLAB changed after the Python lines were written, or whether the two never agreed.
