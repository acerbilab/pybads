# Reference population on Linux after wave 3 of the port review: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Linux for `dev/scripts/population.py compare` until a
later reference replaces it. The one on Windows is
[`population_gpfixes_20260925`](../population_gpfixes_20260925/README.md),
at `ab4dded`. 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3. Its
package code is that of `dev-port-review-w3` at `a14524d`: the previous
reference's (`8510ca8`, wave 2's), with wave 2's W2-45 to W2-47, which
reach no run of this suite (`8aecb6a`), and wave 3's fix pass
([`port_review_20260925/verification/wave3.md`](../port_review_20260925/verification/wave3.md),
"Fix pass"), the doublecheck of wave 2 merged from `dev-next` (`68d4516`),
and without W3-24, which the PI reverted after its gate (`b03a320`). Every
run is identical, in every field but the wall time, to the runs of the
pass's step W3-29 (`0b7add3`): the later commits, the merge (an `f_vals`
without a finite value, and docstrings), the revert, W3-39 (a check of
`accelerate_mesh_steps` when `BADS` is created) and W3-40 (the prior of the
length scales on two points), reach no run of this suite, and the
fingerprint of `dev/scripts/fingerprint.py` is `360971bf1f0ba6cb` (Linux,
gpyreg 1.3.3 from the clone, one BLAS thread) at all of them. It replaces
[`population_linux_wave2_20260926`](../population_linux_wave2_20260926/README.md)
(`8510ca8`'s runs).

## Command and provenance

```console
git worktree add --detach dev/scripts/runs/worktrees/h_a14524d a14524d
PYTHONPATH=<gpyreg clone at v1.3.3> <repository>/.venv/bin/python -u dev/scripts/runs/worktrees/h_a14524d/dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out <repository>/dev/scripts/runs/population/head_default_a14524d
```

- PyBADS at `a14524d`, run from a worktree at that commit, whose own
  `population.py` puts that checkout first on `sys.path` (the records'
  `meta.git` and `meta.pybads_source`, clean). Their `pybads` version
  string, `1.1.1.dev35+g8aecb6a80`, is the metadata of the venv's editable
  install. gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3`
  (`98ab5a4`), selected with `PYTHONPATH`; the records' gpyreg version
  string, `1.3.4.dev10+gd96d0d9f7`, is the metadata of the venv's editable
  install of `../gpyreg`, and `meta.gpyreg_source` names the clone.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, as for the previous Linux reference. One BLAS
  thread per run, four runs at a time, a fresh process per run; 19
  minutes, from 09:29 to 09:48 UTC on 2026-09-27.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`. The comparisons of the pass's steps, one
  against the other, are in
  [`port_review_20260925/verification/wave3_fixpass/`](../port_review_20260925/verification/wave3_fixpass/).

## Outcome

All 540 runs finished; none crashed. The fraction solved ranges from 0.03
(Rastrigin, whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_linux_wave2_20260926 <this population>`
(`comparison.md`), the net change of wave 3's fix pass without W3-24,
flags no configuration in 54 tests. Every step of the pass changed runs:
W3-14 moved 31 (at a fine search mesh), the ES batch (W3-4, W3-5, W3-15)
all 540, W3-1 325, W3-6 the 128 of the noisy configurations, W3-19 5 and
W3-29 277, none of them flagged. Unflagged, the largest shifts are fewer
evaluations on `sphere_D3_homo` (270 → 220) and `sphere_D3_hetero` (372 →
332), a higher error on `ellipsoid_D3_hetero` (median 0.32 → 0.43; paired
log10 ratio +0.13 [-0.04, +0.28]; solved 0.13 → 0.10) and a lower one on
`timing_D5` (ratio -0.29 [-0.48, +0.07]). The fraction solved over seeds
0-29 is a coarse measure on the noisy configurations (the W0-1
investigation,
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
