# Reference population on Linux after the noise-variance fix: the default suite, 30 seeds, gpyreg 1.3.3

Replaced as the reference on Linux by
[`population_linux_meanprior_20260925`](../population_linux_meanprior_20260925/README.md),
at `8afbe16`, whose GP mean prior changes the runs of every configuration.

The reference on Linux for `dev/scripts/population.py compare` at
`1c8c71d` (the one on Windows is
[`population_targetnoise_20260925`](../population_targetnoise_20260925/README.md)):
18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3. It
replaces [`population_linux_20260925`](../population_linux_20260925/README.md):
`020d6a8` (in `1c8c71d`, #65) gives the GP the squares of the noise
standard deviations that a target returns under `specify_target_noise`,
which changes the runs of `sphere_D3_hetero` and `ellipsoid_D3_hetero` and
no other. Pairing by seed is only meaningful against a population from the
same platform and versions, so a change made on Linux compares with this
reference, and one made on Windows with the Windows reference.

## Command and provenance

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 .venv/bin/python -u dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out dev/scripts/runs/population/population_linux_targetnoise_20260925
```

- PyBADS at `1c8c71d` (`dev-next`), clean tree (the records' `meta.git`;
  their `pybads` version string, `1.1.1.dev23+g1c8c71d2b`, is the metadata
  of the venv's editable install of the same checkout). gpyreg 1.3.3 from
  a clone checked out at the tag `v1.3.3` (`98ab5a4`), selected with
  `PYTHONPATH`, and installed editable from a checkout at the same tag, so
  the version string also reads 1.3.3.
- Linux (a cloud container), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, the
  versions of the previous Linux reference. One BLAS thread per run
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1), four
  runs at a time, a fresh process per run; 27.2 minutes, from 14:53 to
  15:20 UTC on 2026-09-25.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`.

## Outcome

All 540 runs finished. The fraction solved ranges from 0.03 (Rastrigin,
whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_linux_20260925 <this population>`
(`comparison.md`) flags no configuration in 54 tests. The records of 16
configurations equal the previous reference's in every `final` field
except `wall_s` and, in the runs that end on `tol_mesh`, the wording of
`message`, which #65 also changed ("mesh size less than
options['tol_mesh']."). The two configurations with target noise differ in
every run:

| Configuration | Median error, previous → this | Largest error | Solved | Median evaluations | Signed-rank p (Holm) |
|---|---|---|---|---|---|
| `sphere_D3_hetero` | 0.21 → 0.10 | 2.26 → 0.53 | 0.23 → 0.50 | 376 → 394 | 0.021 (1) |
| `ellipsoid_D3_hetero` | 0.24 → 0.58 | 1.88 → 7.37 | 0.20 → 0.10 | 342 → 371 | 0.0050 (0.27) |

The median paired log10 error ratio is -0.29 [-0.68, -0.09] for
`sphere_D3_hetero` and +0.39 [+0.23, +0.65] for `ellipsoid_D3_hetero`
(bootstrap intervals), the same directions as on Windows
([`population_targetnoise_20260925`](../population_targetnoise_20260925/README.md)).

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 36 tests.
- **Positive control**: the one in the first reference
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)).
  This population has the same configurations, seeds and tests.

## What the comparison detects at 30 seeds

As for the first reference: 54 tests, the first Holm step at p ≤ 9.3e-4, a
KS statistic of at least 0.50, and, for the paired signed-rank test, a
shift of about 0.87 of the standard deviation of the paired log10 error
ratios at 80% power. Between two versions on this platform, a run that a
change does not reach is identical in both populations, as the 16
unchanged configurations show.
