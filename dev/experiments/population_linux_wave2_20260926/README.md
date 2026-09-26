# Reference population on Linux after wave 2 of the port review: the default suite, 30 seeds, gpyreg 1.3.3

The reference on Linux for `dev/scripts/population.py compare` until a
later reference replaces it. The one on Windows is
[`population_gpfixes_20260925`](../population_gpfixes_20260925/README.md),
at `ab4dded`. 18 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py` × seeds 0-29, each run at BADS's default
budget (500 D) and ending on BADS's own termination criteria, every random
draw through the run's `numpy.random.Generator`, with gpyreg 1.3.3. Its
package code is that of `dev-port-review-w2` at `8510ca8`: the previous
reference's (`fef6c14`) with wave 2's fix pass up to W2-2
([`port_review_20260925/verification/wave2.md`](../port_review_20260925/verification/wave2.md),
"Fix pass"). The pass's later commits reach no run of the default suite:
W2-45 (`dc7383a`) casts integer-typed bounds to float, and every
configuration passes float arrays
([`reach_int.out`](../port_review_20260925/verification/scripts/wave2/orchestrator/reach_int.out)),
W2-46 (`885fc33`) silences a warning without changing a value, W2-47
(`a07be4e`) makes `VariableTransformer`'s copies of the bounds float, which
`BADS` already passes since W2-45, and the rest change docstrings and
option descriptions; the fingerprint of `dev/scripts/fingerprint.py` is
`dc11118754b18b47` at all of them. It
replaces
[`population_linux_wave1_20260926`](../population_linux_wave1_20260926/README.md)
(`fef6c14`'s runs).

## Command and provenance

```console
git worktree add --detach dev/scripts/runs/worktrees/h_8510ca8 8510ca8
PYTHONPATH=<gpyreg clone at v1.3.3> <repository>/.venv/bin/python -u dev/scripts/runs/worktrees/h_8510ca8/dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out <repository>/dev/scripts/runs/population/w2-head_8510ca8
```

- PyBADS at `8510ca8`, run from a worktree at that commit, whose own
  `population.py` puts that checkout first on `sys.path` (the records'
  `meta.git` and `meta.pybads_source`, clean). Their `pybads` version
  string, `0.1.dev60+g95a87bc9c`, is the metadata of the venv's editable
  install. gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3`
  (`98ab5a4`), selected with `PYTHONPATH`.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, as for the previous Linux reference. One BLAS
  thread per run, four runs at a time, a fresh process per run; 20.5
  minutes, from 19:22 to 19:42 UTC on 2026-09-26.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`. The comparisons of the pass's steps, one
  against the other, are in
  [`port_review_20260925/verification/wave2_fixpass/`](../port_review_20260925/verification/wave2_fixpass/).

## Outcome

All 540 runs finished; none crashed. The fraction solved ranges from 0.03
(Rastrigin, whose runs end in local minima) to 1.00.

## Comparison with the previous reference

`compare dev/experiments/population_linux_wave1_20260926 <this population>`
(`comparison.md`), the net change of wave 2's fix pass, flags one
configuration, on its number of evaluations:

| Configuration | Median error, previous → this | Median evaluations | KS on the evaluations (p Holm) | Median paired log10 error ratio [95% CI] |
|---|---|---|---|---|
| `ellipsoid_D10` | 4.8e-7 → 7.3e-7 | 643 → 676 | 0.57 (0.0047) | +0.089 [-0.083, +0.218] |

It is W2-16's (the floor `search_factor_min` on the search factor after a
failed search, as in MATLAB BADS), the pass's only step that moves the
deterministic configurations. Of the other steps, W2-25 (the move after
the re-estimate takes the incumbent's location with its value) changes the
five noisy configurations and flags nothing, and W2-29, W2-4 and W2-2
change no run of this suite (`wave2_fixpass/`; W2-4's gate is the `bounds`
suite). Unflagged, `ellipsoid_D3_homo` has the largest shift, a lower
error (0.10 → 0.057, ratio -0.33 [-0.67, +0.19], solved 0.47 → 0.67), and
the noisy configurations take more evaluations (`sphere_D3_homo` 203 →
270, `sphere_D3_hetero` 332 → 372, `ellipsoid_D3_hetero` 284 → 322,
`multisensory_s1_D6_homo` 526 → 564; KS p Holm ≥ 0.83). The fraction
solved over seeds 0-29 is a coarse measure on the noisy configurations (the
W0-1 investigation,
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
