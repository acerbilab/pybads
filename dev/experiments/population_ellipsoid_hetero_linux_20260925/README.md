# `ellipsoid_D3_hetero` on Linux: the regression after `020d6a8` and three candidate causes

Seeds 0-89 of the configuration `ellipsoid_D3_hetero` of the `default`
suite of `dev/scripts/benchmark_targets.py`, at the commit before `020d6a8`
and after it, and under the candidates of `dev/TODO.md` (item
"`ellipsoid_D3_hetero` after `020d6a8`"). The Windows counterpart is
[`population_ellipsoid_hetero_20260925`](../population_ellipsoid_hetero_20260925/README.md).
The target is a 3-D ellipsoid `sum(a_i (x_i - c_i)**2)` with `a` = (1, 1e3,
1e6), with noise of standard deviation `1 + sqrt(f)` that the target
returns (`specify_target_noise=True`).

## The runs

| Directory | Seeds | PyBADS |
|---|---|---|
| `before/` | 30-89 | `685da15`, before `020d6a8`; seeds 0-29 are those of [`population_linux_20260925`](../population_linux_20260925/README.md), equal run by run to the same seeds at `685da15` |
| `after/` | 30-89 | `1c8c71d`, with `020d6a8`; seeds 0-29 are those of [`population_linux_targetnoise_20260925`](../population_linux_targetnoise_20260925/README.md), equal run by run to the same seeds at `1c8c71d` |
| `mean_prior/` | 0-89 | `1c8c71d` with `mean_prior.patch`: the GP mean prior re-centred at each rebuild, as in MATLAB's `gpdefBads.m` (the package code of `8afbe16` without `032dfcb`) |
| `row_fix/` | 0-89 | the package code of `032dfcb`: a repeated point merged into the row of the function log that matches it in every coordinate |
| `row_fix_observation/` | 0-89 | `032dfcb` with `row_fix_observation.patch`: in addition, the log returns the observation of a repeated point, as MATLAB's `funlogger` does, instead of the merged value |
| `row_fix_mean_prior/` | 0-89 | the package code of `8afbe16`: the row fix and the mean prior |

The records of the four candidate directories name local commits of the
same package code (`meta.git`); the patches give the two that are not
commits of the repository. Each command was

```console
cd <clean worktree at the commit>
PYTHONPATH=<repository>/dev/scripts/runs/gpyreg/v1.3.3 <repository>/.venv/bin/python -u dev/scripts/population.py run --suite default --only ellipsoid_D3_hetero --seeds 0-89 --workers 4 --out <repository>/dev/scripts/runs/population/<name>
```

- gpyreg 1.3.3 from a clone checked out at the tag `v1.3.3` (`98ab5a4`).
- Linux (a cloud container), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1;
  one BLAS thread per run, four runs at a time, a fresh process per run;
  about 6 minutes a directory, on 2026-09-25.
- The seeds 0-29 of `row_fix/` and `row_fix_mean_prior/` equal, run by
  run, those of the gate populations of `032dfcb` and `8afbe16`.

## Outcome

All runs finished. Over the 90 seeds, with the p value of the signed-rank
test on the paired log10 errors:

| Variant | Median error | IQR | 90th pct | Largest | Error ≥ 1 | Solved | Median evals | Smaller / equal vs after | p vs after | p vs before |
|---|---|---|---|---|---|---|---|---|---|---|
| before (`685da15`) | 0.18 | 0.08–0.35 | 0.53 | 6.2 | 2 | 0.31 | 327 | 70 / 0 | 3.4e-07 | — |
| after (`1c8c71d`) | 0.58 | 0.17–1.39 | 2.91 | 11.0 | 31 | 0.17 | 356 | — | — | 3.4e-07 |
| mean prior | 0.45 | 0.20–1.00 | 2.61 | 8.4 | 23 | 0.13 | 356 | 52 / 0 | 0.13 | 7e-06 |
| row fix (`032dfcb`) | 0.48 | 0.10–0.95 | 2.32 | 11.0 | 20 | 0.26 | 376 | 33 / 39 | 0.012 | 0.00032 |
| row fix + observation | 0.52 | 0.20–0.99 | 2.32 | 11.0 | 22 | 0.18 | 363 | 30 / 38 | 0.18 | 2.1e-06 |
| row fix + mean prior (`8afbe16`) | 0.46 | 0.16–0.85 | 2.62 | 8.1 | 20 | 0.19 | 352 | 52 / 0 | 0.17 | 7.9e-06 |

"Solved" is an error below 0.1, the configuration's tolerance.

- **The regression reproduces on Linux**: the median error rises from 0.18
  to 0.58, and the error is smaller after the fix in 20 of the 90 pairs
  (p = 3.4e-7; seeds 30-89 alone, 15 of 60, p = 1.5e-5), as on Windows
  (0.21 to 0.54). As there, it comes from the flat coordinate (`a_1 = 1`):
  its median contribution to the error rises from 0.065 to 0.24, and it
  carries more than half of the error in 23 of the 31 runs at or above 1.
- **The mean prior** lowers the median error to 0.45, not significantly
  (p = 0.13).
- **The row fix**: in 58 of the 90 runs at `1c8c71d` a point is evaluated
  again, 227 times in all, and 188 of those merges, in 54 runs, go into
  another point's row (counted with a wrapper of `FunctionLogger._record`).
  The fix changes 51 runs and improves 33 of them (p = 0.012).
- **Returning the observation** of a repeated point, on top of the row fix,
  changes 20 runs and worsens 16 of them (p = 0.0019 against the row fix
  alone).
- **Both the mean prior and the row fix** do no better than either alone
  (against the row fix, p = 0.85; against the mean prior, p = 0.74).
- **The lower bound of the noise hyperparameter**, the third candidate,
  moves only after a failed fit, and none of the 2,530 fits of the 90 runs
  at `1c8c71d` fails (the same wrapper, with one of `gpyreg.GP.fit`).

No candidate, alone or with another, brings the runs back to those before
`020d6a8` (every p against `before` at or below 3.2e-4).

## Beyond the candidates: no repeated points

The repeats that the row fix concerns exist because `contraints_check`
keeps a candidate that repeats an evaluated point, where MATLAB's
`uCheck.m` drops it (`dev/TODO.md`, "Previously evaluated points
evaluated again"). With the correct noise they are more frequent: seeds
0-19 make 17 repeats at `685da15` and 53 at `1c8c71d`.
`row_fix_mean_prior_no_repeats/` holds seeds 0-89 at `8afbe16` with
`no_repeats.patch`: `contraints_check` drops such candidates, as MATLAB's
`setdiff` does, and, since a search can then be left without a
candidate, the ES search returns an empty set, as MATLAB's `searchES.m`
does, and the search step sets no search point for an empty set. Without
the last two changes, 10 of the 90 runs stopped with `IndexError` in
`es_search.py` (`return us[0], z[0]` on an empty set), and with the second
alone, 8 with `UnboundLocalError` for `u_search` in `_search_step_`.

| Variant | Median error | IQR | 90th pct | Largest | Error ≥ 1 | Solved | Median evals |
|---|---|---|---|---|---|---|---|
| `8afbe16` | 0.46 | 0.16–0.85 | 2.62 | 8.1 | 20 | 0.19 | 352 |
| `8afbe16` without repeats | 0.33 | 0.11–0.76 | 2.37 | 8.1 | 19 | 0.23 | 367 |

Against `8afbe16`, 78 runs change, and 45 of them improve (p = 0.10);
against `1c8c71d`, the error is smaller in 57 of the 90 pairs
(p = 0.02); against `685da15`, in 31 (p = 0.0011), with a median paired
log10 error ratio of +0.14, where `1c8c71d` has +0.54. With the row fix
and the mean prior, dropping the repeats removes about three quarters of
the regression on this measure, and what remains is still significant.
The change is not a commit of the repository: it reaches every run that
repeats a point, and its gate is that of the TODO item.

## Gates of the two commits

- **`032dfcb`** (the row fix): the default suite × seeds 0-29 at that
  commit, compared with
  [`population_linux_targetnoise_20260925`](../population_linux_targetnoise_20260925/README.md)
  (`gate_032dfcb_comparison.md`; its records of `sphere_D3_hetero` are in
  `gate_032dfcb_sphere_D3_hetero/`, and those of `ellipsoid_D3_hetero`
  equal seeds 0-29 of `row_fix/`), flags nothing in 54 tests. The
  16 configurations without target noise are identical run by run, since
  the changed branch runs only when the target returns a noise standard
  deviation. `ellipsoid_D3_hetero` changes in 18 runs, and
  `sphere_D3_hetero` in 25, with no change of its median error (0.10) and
  a smaller largest error (0.52 to 0.34).
- **`8afbe16`** (with the mean prior): its population is the Linux
  reference
  [`population_linux_meanprior_20260925`](../population_linux_meanprior_20260925/README.md),
  whose comparison with the previous one flags five configurations, all
  better.
