# Periodic variables on Linux: the `periodic` suite with `periodic_vars` on and off, 30 seeds

The evidence of the port of periodic variables
([`results/2026-09-28-periodic-variables.md`](../../results/2026-09-28-periodic-variables.md)),
and the reference on Linux of the `periodic` suite of
`dev/scripts/benchmark_targets.py` for a later change to the handling of
periodic variables. The suite's six configurations set `periodic_vars`
("on"); the same problems run as bounded ones with `--options
'{"periodic_vars": null}'` ("off"), paired by seed (each seed's start point
and noise stream are the same in both).

## Command and provenance

```console
R=dev/scripts/runs/periodic_20260928
PYTHONPATH=<gpyreg at 3f1a732> .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --out $R/on
PYTHONPATH=<gpyreg at 3f1a732> .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --options '{"periodic_vars": null}' --out $R/off
python dev/scripts/population.py compare $R/off $R/on
PYTHONPATH=<gpyreg at 3f1a732> OMP_NUM_THREADS=1 .venv/bin/python -u dev/experiments/periodic_linux_20260928/control_centre.py $R/control_centre.json
python dev/experiments/periodic_linux_20260928/decompose.py
```

- PyBADS at `66ef459`, from the main checkout. The records say "dirty":
  the working tree held the documentation of the feature, committed next
  in `54e4080`, whose only change to the package is the description of
  `periodic_vars` in `advanced_bads_options.ini`, a comment.
- gpyreg at `3f1a732`, the head of its branch
  `claude/todo-discussion-q8c9im` (`periods` on the ARD kernels), on
  gpyreg's `main` at `280d8c0`, selected with `PYTHONPATH`; the records'
  version string, 1.3.3, is the metadata of the venv's editable install,
  and `meta.gpyreg_source` names the clone. With it,
  `dev/scripts/fingerprint.py` prints `4146a986863602cb`, the hash of
  gpyreg 1.3.3 and of the Linux reference.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, one BLAS thread per run, three runs at a
  time; "on" from 21:26 to 21:36 UTC, "off" from 21:36 to 21:41, on
  2026-09-28. Other work ran on the machine during "on", so the wall
  times of the two arms do not compare.
- The files: `on/` and `off/`, the records of each run; `summary_on.md`
  and `summary_off.md`; `compare_periodic.md`, "off" as REF and "on" as
  NEW; `decompose.py` and `decompose.txt`, the error of the noisy
  configurations split between the periodic variables and the other one;
  `control_centre.py` and `control_centre.txt`, the control below.

## Outcome

No run crashed. Every configuration is flagged, all but one in favour of
"on":

| Configuration | Median error, off → on | Solved, off → on | Median evaluations, off → on | Median paired log10 error ratio [95% CI] |
|---|---|---|---|---|
| `periodic_D2` | 0.0093 → 3.1e-8 | 0.47 → 1.00 | 52.5 → 54 | -5.01 [-5.44, -0.75] |
| `periodic_D4` | 0.021 → 2.4e-8 | 0.40 → 1.00 | 121 → 124 | -5.87 [-6.10, -0.78] |
| `periodic_D6` | 0.054 → 3.2e-8 | 0.13 → 1.00 | 199 → 203 | -6.20 [-6.41, -5.85] |
| `periodic_D3_homo` | 0.091 → 0.024 | 0.70 → 0.97 | 207 → 359 | -0.56 [-0.69, -0.42] |
| `periodic_D3_hetero` | 0.104 → 0.136 | 0.50 → 0.27 | 310 → 374 | +0.08 [-0.16, +0.30] |
| `periodic_rosenbrock_D4` | 1.6e-6 → 1.5e-7 | 1.00 → 1.00 | 131 → 168 | -0.99 [-1.12, -0.60] |

Without `periodic_vars`, a run whose start lies across the bounds from a
minimum of a periodic variable stops on the bound: every run of the two
noisy configurations ends there, 0.082 from the minimum in the target's
periodic terms (`decompose.txt`), which the three deterministic ones solve
in 13 to 47% of their runs. With `periodic_vars`, all of them are solved.
MATLAB BADS's Example 5, whose minima lie on the bounds, is solved either
way, with a tenth of the error and a quarter more evaluations with
`periodic_vars`.

`periodic_D3_hetero` is flagged for its evaluations only: its errors are
not distinguishable (signed-rank p = 0.78), and its fraction solved falls
because both medians lie near the tolerance of 0.1. Its noise has a
standard deviation of at least 1, and "off" gains from the bound, which
holds each run 0.082 from the minimum. The control shows that the handling
of periodic variables does not worsen noisy runs: with the minima of the
periodic variables moved to the middle of the period, where no run meets
the bounds, "on" is as good as "off" or better (`control_centre.txt`,
seeds 0-29, a budget of 1500):

| Noise | Median error, off → on | Solved (error < 0.1), off → on | Median evaluations, off → on |
|---|---|---|---|
| hetero | 0.137 → 0.114 | 0.40 → 0.47 | 377 → 384 |
| homo | 0.054 → 0.020 | 0.80 → 0.93 | 346 → 263 |
