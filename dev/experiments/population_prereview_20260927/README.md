# Pre-review baseline on Windows: the default suite, 100 seeds, gpyreg 1.3.3

PyBADS as it stood before the fixes of the port correctness review
(`dev/plans/port-correctness-review.md`): `ab4dded`, the revision that the
review's wave 0 read, with every random draw through the run's
`numpy.random.Generator`, the squared noise standard deviations of
`020d6a8`, the three GP fixes of #66 and the noise options of #67. 18
configurations of the `default` suite of `dev/scripts/benchmark_targets.py`
× seeds 0-99, each run at BADS's default budget (500 D) and ending on
BADS's own termination criteria, with gpyreg 1.3.3.

This population is a fixed baseline, not a reference that later gates
replace: it is the "before" of the whole review's comparison (in
[`population_wave4_20260928`](../population_wave4_20260928/README.md)), and
any later version of PyBADS can be compared with it on this platform. It
extends [`population_gpfixes_20260925`](../population_gpfixes_20260925/README.md),
the Windows reference at the same commit, from 30 seeds to 100.

Pairing by seed holds only on the platform and versions below. After a
change of any of them (Windows, Python, NumPy, SciPy, the gpyreg clone, the
number of BLAS threads), regenerate the population from a clean worktree at
`ab4dded` with the command below before comparing with it.

## Command and provenance

From the main checkout's root, with a clean detached worktree at `ab4dded`
under `dev/scripts/runs/worktrees/`, and the 540 records of
`population_gpfixes_20260925` placed in the output directory first
(`population.py` runs only the seeds whose record is missing):

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 .venv/Scripts/python.exe -u dev/scripts/runs/worktrees/winref_ab4dded/dev/scripts/population.py run --suite default --seeds 0-99 --workers 8 --out dev/scripts/runs/population/population_reviewbefore_20260927
```

- PyBADS at `ab4dded`, run by the worktree's own `population.py`, which
  puts that checkout first on `sys.path`: every record's `meta.git` and
  `meta.pybads_source` name the worktree, clean, as `ab4dded` (the 510
  records of `population_gpfixes_20260925`) or `ab4ddedc` (the 1290 run for
  this population), the same commit. gpyreg 1.3.3 from a clone checked out
  at the tag `v1.3.3` (`98ab5a4`), selected with `PYTHONPATH`.
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1; one BLAS thread per
  run (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1), a
  fresh process per run.
- Seeds 0-29 of 17 configurations are the records of
  `population_gpfixes_20260925` (one run at a time, 2026-09-25 and 26).
  The other 1290 runs (seeds 30-99 of every configuration, and seeds 0-29
  of `sphere_D2`) ran with 8 at a time on 2026-09-27, from 22:52 to 23:53.
  Seeds 0-29 of `sphere_D2`, run in both, are equal in every `final` field
  but `wall_s` (30 of 30): the number of runs at a time changes the wall
  time alone, which is about twice as long at 8 as at one.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md` and `null_check.md`.

## Outcome

All 1800 runs finished, none crashed. The fraction solved ranges from 0.01
(`rastrigin_D3`, whose runs end in local minima) to 1.00.

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 36 tests.
- **Positive control**: the one in the first reference
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)).
  This population has the same configurations and tests.

## What the comparison detects at 100 seeds

A full comparison holds 54 tests; after Holm's correction, the first step
needs p ≤ 9.3e-4. For 100 against 100 runs, that is a KS statistic of at
least 0.28. The paired signed-rank test detects, with 80% power, a shift of
the paired log10 error ratios of about 0.44 of their standard deviation
(simulated; the same simulation gives the 0.87 of 30 seeds that the first
reference states).

## The records and later code

The records share the notes of `population_gpfixes_20260925`, "The records
and later code": code from `7b50a3a` (#71) on counts `iterations` from 1
and reports a larger final `fsd` for the configurations with inferred
noise, two fields that `compare` does not read.
