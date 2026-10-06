# Reference population on Linux after wave 1 of the port review: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Linux for `dev/scripts/population.py compare` until a
later reference replaces it. The one on Windows is
[`population_gpfixes_20260925`](../population_gpfixes_20260925/README.md),
at `ab4dded`. 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3. Its
package code is that of `dev-port-review-w1` at `6e22d32`: the previous
reference's (`e004c79`) with wave 1's fix pass up to W1-1
([`port_review_20260925/verification/wave1.md`](../port_review_20260925/verification/wave1.md),
"Fix pass"). The pass's last fix, W1-17 (`cfacb98`), changes only runs at
D = 1, of which the default suite has none, so this population also stands
for `cfacb98`: the fingerprint of `dev/scripts/fingerprint.py` is
`91f947f78e1087c2` at both. It replaces
[`population_linux_wave0_20260926`](../population_linux_wave0_20260926/README.md)
(`ac3dfed`).

## Command and provenance

```console
git worktree add --detach dev/scripts/runs/worktrees/h_6e22d32 6e22d32
PYTHONPATH=<gpyreg clone at v1.3.3> <repository>/.venv/bin/python -u dev/scripts/runs/worktrees/h_6e22d32/dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out <repository>/dev/scripts/runs/population/w1-1_6e22d32
```

- PyBADS at `6e22d32`, run from a worktree at that commit, whose own
  `population.py` puts that checkout first on `sys.path` (the records'
  `meta.git` and `meta.pybads_source`). The records say `dirty`: the
  worktree's `dev/scripts/benchmark_targets.py` carried the `oned` suite of
  `db09fb6`, for W1-17's gate, which leaves the `default` suite as it is;
  nothing under `pybads/` differed from the commit. Their `pybads` version
  string, `1.1.1.dev45+g3e9646c05`, is the metadata of the venv's editable
  install. gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3`
  (`98ab5a4`), selected with `PYTHONPATH`.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, as for the previous Linux reference. One BLAS
  thread per run, four runs at a time, a fresh process per run; 18.4
  minutes, from 13:28 to 13:46 UTC on 2026-09-26.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`. The comparisons of the pass's batches,
  one against the other, are in
  [`port_review_20260925/verification/wave1_fixpass/`](../port_review_20260925/verification/wave1_fixpass/).

## Outcome

All 540 runs finished; none crashed. The fraction solved ranges from 0.00
(Rastrigin, whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_linux_wave0_20260926 <this population>`
(`comparison.md`), the net change of wave 1's fix pass, flags one
configuration, a lower error:

| Configuration | Median error, previous → this | Median evaluations | KS on the error (p Holm) | Median paired log10 error ratio [95% CI] |
|---|---|---|---|---|
| `ackley_D6` | 4.3e-4 → 1.9e-4 | 394 → 388 | 0.77 (3.5e-7) | -0.299 [-0.401, -0.210] |

Unflagged, two intervals of the median paired log10 error ratio exclude
zero, both lower errors: `rosenbrock_D2`, 1.9e-5 → 2.7e-6 (-1.03
[-1.60, -0.42], solved 0.97 → 1.00), and `sphere_nonbox_D3`, 1.1e-5 → 5.8e-6
(-0.36 [-0.64, -0.09]). The fraction solved falls on `ellipsoid_D3_hetero`,
0.33 → 0.17 (median error 0.26 → 0.36, ratio +0.14 [-0.37, +0.46]), and
on `multisensory_s1_D6_homo`, 1.00 → 0.90, neither flagged. On the noisy
configurations the fraction solved moved by as much between the batches of
the pass (`ellipsoid_D3_homo`: 0.47 to 0.70), and over seeds 0-29 it is a
coarse measure (the W0-1 investigation,
[`port_review_20260925/w01_investigation/`](../port_review_20260925/w01_investigation/README.md)).

## Checks

- **Null check** (`compare <this population> --split`, even against odd
  seeds, KS tests alone; `null_check.md`): no flag in 36 tests.
- **Positive control**: the one in the first reference
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)).
  This population has the same configurations, seeds and tests.

## What the comparison detects at 30 seeds

As for the first reference: 54 tests, the first Holm step at p ≤ 9.3e-4, a
KS statistic of at least 0.50, and, for the paired signed-rank test, a
shift of about 0.87 of the standard deviation of the paired log10 error
ratios at 80% power. Between two versions on this platform, a run that a
change does not reach is identical in both populations.
