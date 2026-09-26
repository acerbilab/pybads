# Reference population on Linux after wave 0 of the port review: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Linux for `dev/scripts/population.py compare` until a
later reference replaces it, and the baseline of the fix pass of the port
review's wave 1 (step 0 of the rulings in
[`port_review_20260925/verification/wave1.md`](../port_review_20260925/verification/wave1.md)).
The one on Windows is
[`population_gpfixes_20260925`](../population_gpfixes_20260925/README.md),
at `ab4dded`. 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3. Its
package code is that of `dev-next` at `e004c79`: wave 0 of the port review
and its fix pass (#72) and W0-1 (#73). It replaces
[`population_linux_gpfixes_20260925`](../population_linux_gpfixes_20260925/README.md)
(`97b2c66`), from which it differs, in the fields that `compare` reads, by
W0-1 alone (the step below).

## Command and provenance

```console
git worktree add --detach dev/scripts/runs/worktrees/baseline_ac3dfed ac3dfed
PYTHONPATH=<gpyreg clone at v1.3.3> <repository>/.venv/bin/python -u dev/scripts/runs/worktrees/baseline_ac3dfed/dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out <repository>/dev/scripts/runs/population/baseline_ac3dfed
```

- PyBADS at `ac3dfed`, the head of `dev-port-review-w1` (the records of
  wave 1 on top of `e004c79`, with no change under `pybads/`), run from a
  clean worktree at that commit, whose own `population.py` puts that
  checkout first on `sys.path` (the records' `meta.git` and
  `meta.pybads_source`; their `pybads` version string,
  `1.1.1.dev45+g3e9646c05`, is the metadata of the venv's editable
  install). gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3`
  (`98ab5a4`), selected with `PYTHONPATH`.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, as for the previous Linux reference. One BLAS
  thread per run, four runs at a time, a fresh process per run; 18.3
  minutes, from 09:23 to 09:42 UTC on 2026-09-26.
- The fingerprint of `dev/scripts/fingerprint.py` at `ac3dfed` on this
  platform, with the same gpyreg clone: `bfbc6d6737e99d88`.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference), `null_check.md` and `steps/`. The runs went to
  `dev/scripts/runs/population/baseline_ac3dfed` (and `baseline_138a141`
  for the step); the headers of the `.md` files name those directories.

## Outcome

All 540 runs finished; none crashed. The fraction solved ranges from 0.00
(Rastrigin, whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_linux_gpfixes_20260925 <this population>`
(`comparison.md`) flags three configurations, on the number of evaluations
alone: fewer evaluations, the same errors.

| Configuration | Median evaluations, previous → this | Median error | KS on evaluations (p Holm) | Median paired log10 error ratio [95% CI] |
|---|---|---|---|---|
| `ellipsoid_D3_homo` | 374 → 326 | 0.071 → 0.082 | 0.77 (3.5e-7) | +0.129 [-0.112, +0.368] |
| `multisensory_s1_D6_homo` | 639 → 488 | 0.180 → 0.178 | 0.63 (3.1e-4) | +0.019 [-0.163, +0.127] |
| `ellipsoid_D3_hetero` | 363 → 310 | 0.298 → 0.259 | 0.53 (0.015) | +0.000 [-0.107, +0.225] |

The other two configurations with noise change too, unflagged:
`sphere_D3_homo` 366 → 205 evaluations (KS 0.43, p Holm 0.33) and
`sphere_D3_hetero` 384 → 330. No test of the error is flagged, and every
interval of the median paired log10 error ratio contains zero. The 13
configurations without noise are identical run by run: W0-1 changes only
the re-evaluation of the iterates of a noisy run. More noisy runs end on
the stall criterion (the change of the function value over
`tol_stall_iters` iterations): 82 of the 150, against 42. These are the
flags of W0-1's gate on Windows
([`port_review_20260925/w01_investigation/`](../port_review_20260925/w01_investigation/README.md)),
whose investigation found no significant change of the error over seeds
0-89 of `ellipsoid_D3_homo`; W0-1 stays (PI, 2026-09-26).

## Steps

- **`138a141`** (`steps/138a141_comparison.md`, against the previous
  reference): the fix pass of wave 0 without W0-1, whose package code
  `dev-next` carried at #72. No flag in 54 tests, and every median paired
  log10 error ratio is exactly zero: `x`, `true_error`, `func_count`,
  `crashed` and `message` are identical in all 540 runs. Other fields change
  as the changelog says: `iterations` is one more in every run (#71), `fsd`
  is larger by `sqrt(10/9)` in the 90 runs with inferred noise (#71), and
  `fval` in 43 and `fsd` in 60 of the runs with target noise take the
  weighted final estimate (#67). So the platform and the versions reproduce
  the previous reference, and the comparison above measures W0-1: this
  population compared with the step gives the same table as with the
  previous reference.

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
change does not reach is identical in both populations.
