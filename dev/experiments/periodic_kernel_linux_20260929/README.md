# gpyreg's periodic kernel on the circle: where periodic runs spend their time, and the `periodic` suite, on Linux

The evidence of
[`results/2026-09-28-periodic-variables.md`](../../results/2026-09-28-periodic-variables.md),
"Time per evaluation":
stage profiles of the `periodic` suite of `dev/scripts/benchmark_targets.py`
with and without `periodic_vars`, gpyreg's periodic kernel against the
same calls without periods, and the change to gpyreg that maps each
periodic coordinate onto a circle once per point (gpyreg `0f27db5`, below)
with its gate, the whole `periodic` suite against the reference of the
suite on Linux,
[`population_periodic_linux_20260928`](../population_periodic_linux_20260928/README.md)
("on" arm).

## The gpyreg change

`gpyreg-perf-periodic-circle.patch` holds two commits of the branch
`perf-periodic-circle` of a clone of gpyreg whose `main` was at `b44634f`
(the merge of acerbilab/gpyreg#61, the CI pin `GPYREG_PIN`): `0f27db5`,
the change, whose tree is `8638145` and which every measurement here ran,
and `91ea28e`, whose tree is `935929b`, which adds a test (coordinates at
either end of the wrap interval are one point) and changes docstrings, a
comment and `AGENTS.md`, not the kernels' code. `git am` of the patch on
`b44634f` rebuilds both trees. The records name the code by its commit,
`0f27db5`.

## Command and provenance

```console
E=dev/experiments/periodic_kernel_linux_20260929
P=dev/scripts/runs/perf_periodic
R=dev/scripts/runs/population/periodic_circle_20260929
BASE=<gpyreg clone at b44634f>
NEW=<gpyreg clone at 0f27db5>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Stage profiles: seeds 0-2 of each configuration, one run at a time;
# "on" with the counterfactual at both gpyreg commits, "off" at b44634f
for lab in periodic_D3_homo periodic_D3_hetero periodic_D4 periodic_D6 periodic_D2 periodic_rosenbrock_D4; do
  for s in 0 1 2; do
    PYTHONPATH=$BASE .venv/bin/python -u $E/profile_periodic.py $lab $s on --counterfactual --out $P/base_${lab}_s${s}_on.json
    PYTHONPATH=$BASE .venv/bin/python -u $E/profile_periodic.py $lab $s off --out $P/base_${lab}_s${s}_off.json
    PYTHONPATH=$NEW .venv/bin/python -u $E/profile_periodic.py $lab $s on --counterfactual --out $P/new_${lab}_s${s}_on.json
  done
done
python $E/tabulate.py $P base new > $E/profiles.md

# cProfile of seed 0 of periodic_D3_homo, both arms, at b44634f (PyBADS's udist)
PYTHONPATH=$BASE .venv/bin/python -u $E/profile_periodic.py periodic_D3_homo 0 on --cprofile $P/base_D3homo_s0_on.prof
PYTHONPATH=$BASE .venv/bin/python -u $E/profile_periodic.py periodic_D3_homo 0 off --cprofile $P/base_D3homo_s0_off.prof
# (cprofile_udist.txt: udist, period_check and local_gp_fitting of the two dumps)

# The kernel alone, both commits side by side
.venv/bin/python $E/kernel_compare.py $BASE $NEW > $E/kernel_compare.txt

# The periodic suite, 30 seeds, three workers, one population at a time
PYTHONPATH=$NEW .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --out $R/new_on
PYTHONPATH=$BASE .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --out $R/base_on
PYTHONPATH=$BASE .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --options '{"periodic_vars": null}' --out $R/off
python dev/scripts/population.py compare dev/experiments/population_periodic_linux_20260928/on $R/new_on > $E/compare_new_vs_reference.md
python dev/scripts/population.py summary $R/new_on && cp $R/new_on/*.json $E/new_on/ && cp $R/new_on/summary.md $E/summary_new_on.md
python $E/timing.py collect $R/new_on $R/base_on $R/off $E/timing.json
python $E/timing.py table $E/timing.json > $E/timing.md
```

- PyBADS at `9a705918` (`dev-next`), from the main checkout (clean).
- gpyreg from clones on `PYTHONPATH`: `b44634f` ("base") and `0f27db5`
  ("new"); the records' version string, `1.3.4.dev21+gb44634f2a`, is the
  metadata of the venv's editable install, and `meta.gpyreg_source` names
  the clone and its commit.
- Linux (a cloud container, kernel `6.18.44-fc-v37`, 4 cores, Intel Xeon
  at 2.8 GHz), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, one BLAS thread
  per run, nothing else running during the timed runs; 2026-09-29.
- With gpyreg at `0f27db5`, `dev/scripts/fingerprint.py` prints
  `4146a986863602cb`, the hash at `b44634f` and of gpyreg 1.3.3, on this
  machine with one BLAS thread.
- The files: `profile_periodic.py` (the stage timers and the
  counterfactual), `tabulate.py`, `profiles/` (one JSON per profiled run)
  and `profiles.md`; `cprofile_udist.txt`; `kernel_compare.py` and
  `kernel_compare.txt`;
  `new_on/`, the records of the gate's population, `summary_new_on.md` and
  `compare_new_vs_reference.md`; `timing.py`, `timing.json` (evaluations
  and wall time of every run of the three populations) and `timing.md`.

## Checks

- **This machine reproduces the reference.** Every run of `base_on` and
  of `off` has the record of the same run in the reference's "on" and
  "off" arms, every field of its result but the wall time (180 of 180 in
  each arm).
- **Null check**: the reference's (`null_check.md` of
  `population_periodic_linux_20260928`), no flag in 12 tests.
- **Positive control**: the reference's comparison of its "off" arm with
  its "on" arm, which flags every configuration.

## What the comparison detects at 30 seeds

As the reference: 18 tests, the first Holm step at p ≤ 0.0028, a KS
statistic of at least 0.467, and a paired shift of about 0.72 of the
standard deviation of the paired log10 error ratios at 80 % power.

## Outcome

**The gate.** `compare_new_vs_reference.md` flags no configuration: no KS
test and no paired test comes near the first Holm step (smallest
unadjusted p = 0.071, `periodic_D6`'s errors), no run crashed, and every
median paired log10 error ratio lies between -0.15 and 0. 34 of the 180
runs return the reference's result to the last bit (17 of 30 of
`periodic_D3_hetero`, 12 of `periodic_D3_homo`, 5 of `periodic_D2`); the
others differ from the rounding of the kernel on, as a change of
arithmetic moves a trajectory.

**Where the time goes** (`profiles.md` and `profiles/`, seeds 0-2): at
gpyreg `b44634f`, the kernel with periods takes 2.5 to 3.9 times as long
as the same calls without periods (1.6 on `periodic_D2`), and the
difference is 27 to 38 % of each run (10 to 11 % on `periodic_D2`); most
of it in the predictions at the ES search's candidates, the
cross-covariance of the training set with 2048 candidates, where each
periodic dimension took an `fmod` and a sine of every pair. At `0f27db5`
the kernel with periods takes 1.4 to 1.6 times as long as the same calls
without them (1.3 on `periodic_D2`), a difference of 8 to 13 % of each run
(5 to 6 %), and the runs that end after as many evaluations as at
`b44634f` take 13 to 31 % less time (`periodic_D2`: -3 to 20 %, runs of
half a second).

**The kernel alone** (`kernel_compare.txt`, one BLAS thread, nothing else
running): the cross-covariance of 100 training points with 2048
candidates, RQ-ARD at D = 3 with two periodic dimensions, as in
`periodic_D3_*`, takes 5.4 ms instead of 16.0 (4.2 without periods), and
the kernel with its gradient at 200 points 1.6 ms instead of 4.2 (1.7
without periods). Repeats of the script vary by some 10 %; the timings in
the commit message of `0f27db5` come from an earlier run of it. The
kernels and gradients of the five ARD kernels agree with `b44634f`'s to
2.1e-15 of the output variance over 200 random draws each (dimensions 1
to 6, periodic and not, with close pairs and duplicates).

**The suite's time per evaluation** (`timing.md`; medians over 30 seeds,
three runs at a time):

| Configuration | ms/eval, off | on, `b44634f` | on, `0f27db5` | on / off, `b44634f` → `0f27db5` | `0f27db5` / `b44634f`, paired |
|---|---|---|---|---|---|
| `periodic_D2` | 9.3 | 10.1 | 9.2 | 1.09 → 0.99 | 0.93 |
| `periodic_D4` | 11.6 | 14.2 | 11.5 | 1.22 → 0.99 | 0.81 |
| `periodic_D6` | 14.3 | 19.3 | 13.3 | 1.35 → 0.93 | 0.69 |
| `periodic_rosenbrock_D4` | 19.5 | 19.0 | 14.9 | 0.98 → 0.77 | 0.79 |
| `periodic_D3_homo` | 22.6 | 51.0 | 33.5 | 2.26 → 1.48 | 0.66 |
| `periodic_D3_hetero` | 29.4 | 54.0 | 35.9 | 1.84 → 1.22 | 0.67 |

The paired ratio is that of each seed's time per evaluation, median over
the seeds. The time per evaluation of the noisy configurations grows with
the length of the run, and their runs with `periodic_vars` take more
evaluations (medians 348 and 364 against 207 and 310 without it), which
accounts for part of what remains of their ratio: a least-squares line of
the time per evaluation on the number of evaluations, per arm, gives at
the median evaluations of "off" 25.5 against 21.5 ms (`homo`, 1.19) and
32.0 against 29.3 ms (`hetero`, 1.09) with `0f27db5`, and 37.6 and 48.3 ms
with `b44634f`.
