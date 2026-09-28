# Seeded runs on macOS arm64: not repeatable bit for bit

`test_run_control.py::test_output_fcn_that_changes_nothing_leaves_the_run_unchanged`
ran one seeded optimization twice in one process, the second with an output
function that alters only its copy of `optim_state`, and required the same
result. In the CI of #90 it failed on `macos-latest` with Python 3.10
(NumPy 2.2.6, SciPy 1.15.3, gpyreg 1.3.3): the two runs returned different
points. It passed on the job's re-run, and on Linux and Windows.

## Method

`dev/scripts/divergence_trace.py` runs that test's optimization (a 3-D
sphere, `random_seed=3`, 80 evaluations) repeatedly in one process,
alternating runs without and with the output function:

- `loop` compares the sequence of evaluated points of each run with the
  first run's;
- `trace` records a digest of the arguments and of the return value of
  every Python call in PyBADS, gpyreg, `numpy.linalg`, `scipy.linalg` and
  `scipy.optimize`, and reports the first event at which a run departs from
  a reference run of its configuration;
- `align` calls the linear algebra of a GP fit on fixed inputs placed at
  each 8-byte offset from a 128-byte boundary, and repeated with the
  allocator's state shifted, and counts the distinct results bit for bit.

It ran in CI on `macos-latest` (Apple Silicon, where NumPy and SciPy use
Accelerate for BLAS and LAPACK) with Python 3.10, 3.11 and 3.12, and on
`ubuntu-latest` with Python 3.10 (OpenBLAS), each with the default number
of BLAS threads and with one (`VECLIB_MAXIMUM_THREADS=1` and its kin), in
two rounds on 2026-09-28: runs
[36470887355](https://github.com/acerbilab/pybads/actions/runs/36470887355)
(`loop` and `trace`) and
[36471817628](https://github.com/acerbilab/pybads/actions/runs/36471817628)
(`align` and `loop`), with gpyreg at the v1.3.3 pin of `test-matrix.yml`.
The workflow installed PyBADS and gpyreg as `test-matrix.yml` does and ran
the script's modes as separate steps; the numbers below are from its logs,
which GitHub keeps for its retention period only.

Controls, on Linux (Python 3.11, NumPy 2.4.6, SciPy 1.17.1, OpenBLAS): 20
alternating runs in one process evaluate the same points; 6 traced runs
repeat their reference over all of their 193,000 events; and a one-ulp
change injected in one run, into the matrix that gpyreg factorizes at its
300th factorization, is reported at the call of `scipy.linalg.cholesky`
that receives it, with that matrix as the one variable of the calling frame
that differs. That run's 80 evaluations are unchanged: the mesh absorbs a
difference in the last bits, unless it changes a decision.

## Results

| Runner | Python | NumPy | SciPy | `loop`: runs whose evaluations differ | `trace` |
|---|---|---|---|---|---|
| macOS arm64 | 3.10 | 2.2.6 | 1.15.3 | round 1: 12 of 60 with the default threads, 12 of 60 with one; round 2: 0 of 60 | default threads: 9 of 18 runs depart, at one event; one thread: none |
| macOS arm64 | 3.11 | 2.4.6 | 1.17.1 | 0 of 60 in each step | none |
| macOS arm64 | 3.12 | 2.5.3 | 1.18.1 | 0 of 60 in each step | none |
| Ubuntu | 3.10 | 2.2.6 | 1.15.3 | 0 of 60 in each step | none |

- In round 1 on macOS with Python 3.10, every fifth run departed from the
  first (runs 3, 8, ..., 58 with the default threads, 2, 7, ..., 57 with
  one), with or without the output function; all 60 still returned the
  same point. In round 2, in another process on another runner, no run
  departed.
- Where the traced runs departed, the first difference was a return of
  gpyreg's `_solve_triangular` (LAPACK's `dtrtrs`), in a hyperparameter
  fit, whose arguments had the same values as in the reference.
- `align`: on each of the three macOS stacks, with either number of
  threads, the product of a 1 x 11 and an 11 x 1 array (`@`, a dot product
  in BLAS) gives two different results depending on the alignment of its
  second operand. The other operations measured (`dtrtrs`, `dpotrf`,
  N x N products, N x N by N x 1 products, `np.sum`, at N from 6 to 81)
  gave one result at every offset and every repeat. On Ubuntu every
  operation gave one result.

## Conclusion

On macOS arm64, Accelerate's results depend on where the arrays lie in
memory, which the allocations of the process decide, and not on the number
of threads. Two runs of one seed, in one process or in two, need not match
bit for bit there: a difference in the last bits usually leaves the
evaluations unchanged, sometimes changes them (one run in five, in one
process of round 1), and rarely the returned point (the CI of #90). The
same versions of NumPy and SciPy repeat every run on Linux. Later stacks
showed no departure in these runs, but their BLAS has the same dependence,
so no stack of macOS arm64 is known to repeat. Nothing in PyBADS or gpyreg
reads the clock, an object's identity or uninitialized memory in a way that
changes a run: the traced runs repeat on Linux over every event.

The test is now `test_output_fcn_alters_only_its_copy_of_optim_state`,
which checks what the comparison of two runs stood for: the output
function's changes, in place included, reach only its copy of
`optim_state`, and the run ends on its budget.
