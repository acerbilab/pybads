# Reference population on Linux after wave 4 of the port review: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Linux for `dev/scripts/population.py compare` until a
later reference replaces it. The one on Windows is
[`population_gpfixes_20260925`](../population_gpfixes_20260925/README.md),
at `ab4dded`. 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3. Its
package code is that of `dev-port-review-w4` at `46af65a`, the last step
of wave 4's fix pass
([`port_review_20260925/verification/wave4.md`](../port_review_20260925/verification/wave4.md),
"Fix pass"): the previous reference's (`a14524d`, wave 3's), with the
doublecheck of wave 3 merged from `dev-next` (#79, `6ceed6f`) and wave 4's
fix pass. The later commits of the branch change no file of the package,
and the fingerprint of `dev/scripts/fingerprint.py` is `4146a986863602cb` (Linux,
gpyreg 1.3.3 from the clone, one BLAS thread) at `46af65a` and at the
branch's head. It replaces
[`population_linux_wave3_20260927`](../population_linux_wave3_20260927/README.md)
(`a14524d`'s runs).

## Command and provenance

```console
git worktree add --detach dev/scripts/runs/worktrees/h_46af65a 46af65a
PYTHONPATH=<gpyreg clone at v1.3.3> <repository>/.venv/bin/python -u dev/scripts/runs/worktrees/h_46af65a/dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out <repository>/dev/scripts/runs/population/w46_46af65a
```

- PyBADS at `46af65a`, run from a worktree at that commit, whose own
  `population.py` puts that checkout first on `sys.path` (the records'
  `meta.git` and `meta.pybads_source`, clean). gpyreg 1.3.3 from a clone
  checked out at the tag `v1.3.3` (`98ab5a4`), selected with `PYTHONPATH`;
  the records' gpyreg version string, `1.3.4.dev10+gd96d0d9f7`, is the
  metadata of the venv's editable install of `../gpyreg`, and
  `meta.gpyreg_source` names the clone. Their `pybads` version string,
  `1.1.1.dev38+ged82ec01b`, is the metadata of the venv's editable
  install.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, as for the previous Linux reference. One BLAS
  thread per run, four runs at a time, a fresh process per run; 24 minutes, from 17:11 to 17:35 UTC on 2026-09-27.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`. The comparisons of the pass's steps, one
  against the other, are in
  [`port_review_20260925/verification/wave4_fixpass/`](../port_review_20260925/verification/wave4_fixpass/).

## Outcome

All 540 runs finished; none crashed. The fraction solved ranges from 0.07
(Rastrigin, whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_linux_wave3_20260927 <this population>`
(`comparison.md`), the net change of wave 4's fix pass, flags no
configuration in 54 tests. Three steps of the pass changed runs of this
suite: W4-21 (the rounding of `contraints_check`'s bins) 34, W4-1 (the
initial design seeded from the run's generator) all 540, and W4-6 with
its completion (the noise test out of the GP's fit schedule) the 390 runs
of the 13 configurations without noise, which take the noise test; the
other steps leave the fingerprint unchanged, and those between W4-21 and
W4-1 have no population of their own (they lie within W4-1's comparison,
where every run changed). Unflagged, the fraction solved moves most on the
noisy 3-D configurations: `ellipsoid_D3_homo` ends farther from its minimum
(paired log10 error ratio +0.39 [-0.10, +0.86]; solved 0.73 → 0.43, of
which W4-21 moved 0.10 and W4-1 0.20), and `sphere_D3_hetero` (-0.21
[-0.49, +0.00]; solved 0.40 → 0.60) and `ellipsoid_D3_hetero` (-0.23
[-0.54, -0.07], the one interval that excludes 0; solved 0.10 → 0.23)
closer. The fraction solved over seeds
0-29 is a coarse measure on the noisy configurations (the W0-1
investigation,
[`port_review_20260925/w01_investigation/`](../port_review_20260925/w01_investigation/README.md)).
Before W4-1 every run of a given D started from the same initial design,
whatever its seed; in this population each seed has its own, so that the
30 seeds of a configuration also sample the design.

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
