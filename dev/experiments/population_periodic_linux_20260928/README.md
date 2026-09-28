# Periodic variables on Linux: the `periodic` suite with `periodic_vars` on and off, 30 seeds

The evidence of the port of periodic variables
([`results/2026-09-28-periodic-variables.md`](../../results/2026-09-28-periodic-variables.md)),
and, in its "on" arm, the reference on Linux of the `periodic` suite of
`dev/scripts/benchmark_targets.py`, the gate of a change to the handling of
periodic variables, which the `default` suite does not reach. The suite's
six configurations set `periodic_vars` ("on"); the same problems run as
bounded ones with `--options '{"periodic_vars": null}'` ("off"), paired by
seed (each seed's start point and noise stream are the same in both).

## Command and provenance

```console
R=dev/scripts/runs/periodic_20260928
E=dev/experiments/population_periodic_linux_20260928
G=<gpyreg clone at 3f1a732>
PYTHONPATH=$G .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --out $R/on
PYTHONPATH=$G .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --options '{"periodic_vars": null}' --out $R/off
for arm in on off; do
  python dev/scripts/population.py summary $R/$arm
  mkdir -p $E/$arm && cp $R/$arm/*.json $E/$arm/ && cp $R/$arm/summary.md $E/summary_$arm.md
done
python dev/scripts/population.py compare $E/off $E/on > $E/compare_periodic.md
python dev/scripts/population.py compare $E/on --split > $E/null_check.md
python dev/experiments/population_periodic_linux_20260928/decompose.py > $E/decompose.txt
PYTHONPATH=$G OMP_NUM_THREADS=1 .venv/bin/python -u $E/control_centre.py $E/control_centre.json > $E/control_centre.txt
```

- PyBADS at `66ef459`, from the main checkout. The records say "dirty":
  the working tree held the documentation of the feature, committed next
  in `54e4080`, whose only change to code that a run executes is the
  description of `periodic_vars` in `advanced_bads_options.ini`, a
  comment.
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
  NEW; `null_check.md`; `decompose.py` and `decompose.txt`, the error of
  the noisy configurations split between the periodic variables and the
  other one, with the runs that end with a periodic coordinate on a bound;
  `control_centre.py`, `control_centre.json` (one record per run, without
  its returned point) and `control_centre.txt`, the control below.

## Checks

- **Null check** (`compare on --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 12 tests.
- **Positive control**: the comparison of "off" with "on" below, which
  flags the deterministic configurations, whose runs without
  `periodic_vars` stop on the bounds, with paired log10 error ratios of -5
  to -6.

## What the comparison detects at 30 seeds

The suite alone: 18 tests, the first Holm step at p ≤ 0.0028, a KS
statistic of at least 0.467, and, for the paired signed-rank test, a shift
of about 0.72 of the standard deviation of the paired log10 error ratios
at 80% power. Between two versions on this platform, a run that a change
does not reach is identical in both populations.

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
minimum of a periodic variable stops on the bound: of the noisy
configurations' runs, 29 of 30 (homo) and 25 of 30 (hetero) end with a
periodic coordinate on a bound, most of them 0.082 from the minimum in the
target's periodic terms, the median in both (`decompose.txt`); the three
deterministic configurations are solved in 13 to 47% of their runs. With
`periodic_vars`, no run of the noisy configurations ends on a bound, and
every deterministic run is solved. MATLAB BADS's Example 5, whose minima lie on the bounds, is solved
either way, with a tenth of the error and a quarter more evaluations with
`periodic_vars`.

`periodic_D3_hetero` is flagged for its evaluations only: its errors are
not distinguishable (signed-rank p = 0.78), and its fraction solved falls
because both medians lie near the tolerance of 0.1. Its noise has a
standard deviation of at least 1, and in "off" the bound holds most runs
0.082 from the minimum in the periodic terms.

The control moves the minima of the periodic variables to the middle of
the period, half a period from the bounds, so that reaching them does not
cross a bound, and runs both kinds of noise with and without
`periodic_vars` (`control_centre.txt`; seeds 0-29, a budget of 1500).
Paired by seed, the errors do not differ detectably: the median paired
log10 error ratio (on/off) is +0.03 with noise "hetero" (signed-rank
p = 0.97) and -0.07 with noise "homo" (p = 0.34).

| Noise | Median error, off → on | Solved (error < 0.1), off → on | Median evaluations, off → on |
|---|---|---|---|
| hetero | 0.137 → 0.114 | 0.40 → 0.47 | 377 → 384 |
| homo | 0.054 → 0.020 | 0.80 → 0.93 | 346 → 263 |
