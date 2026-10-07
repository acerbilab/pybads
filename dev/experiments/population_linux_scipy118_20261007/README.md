# The Linux reference under Python 3.12, NumPy 2.5.3 and SciPy 1.18.1: the `default` suite, 30 seeds

The reference on Linux for `dev/scripts/population.py compare`, under the
versions of the Windows reference,
[`population_gpyreg140_20260930`](../population_gpyreg140_20260930/README.md):
the 24 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py`, the six with periodic variables
included, × seeds 0-29, each run at BADS's default budget (500 D) and
ending on BADS's own termination criteria, with PyBADS at `0dc5932f`,
whose package code is that of the release 1.5.1, and gpyreg 1.4.0. A gate
compared with it runs under the same versions: Python 3.12, NumPy 2.5.3
and SciPy 1.18.1. The previous Linux reference,
[`population_linux_gpyreg140_20260930`](../population_linux_gpyreg140_20260930/README.md),
ran under Python 3.11, NumPy 2.4.6 and SciPy 1.17.1, and runs under the
newer versions do not pair with it seed by seed. A gate run under the
older versions still compares with it: under them, the same code repeats
each of its 720 runs (below).

## Command and provenance

```console
E=dev/experiments/population_linux_scipy118_20261007
R=dev/scripts/runs/population
S=dev/experiments/population_linux_gpyreg140_20260930/same_runs.py
G=dev/scripts/runs/gpyreg/v1.4.0   # git clone https://github.com/acerbilab/gpyreg $G && git -C $G checkout v1.4.0
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.3 scipy==1.18.1
.venv/bin/python -m pip install -e "../gpyreg[dev]" -e ".[dev]"   # ../gpyreg at v1.4.0
PYTHONPATH=$G .venv/bin/python -u dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out $R/linuxref_0dc5932f_20261007
cp $R/linuxref_0dc5932f_20261007/*_seed*.json $E/
.venv/bin/python dev/scripts/population.py summary $E
.venv/bin/python dev/scripts/population.py compare $E --split > $E/null_check.md
.venv/bin/python dev/scripts/population.py compare dev/experiments/population_linux_gpyreg140_20260930 $E > $E/comparison.md
.venv/bin/python $S dev/experiments/population_linux_gpyreg140_20260930 $E > $E/same_runs.txt
```

The same code under the previous reference's versions, from a second venv
(`$V`: Python 3.11, `numpy==2.4.6`, `scipy==1.17.1` and `gpyreg==1.4.0`,
then `pip install -e . --no-deps`), whose records are not kept:

```console
PYTHONPATH=$G $V/bin/python -u dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out $R/oldstack_0dc5932f_20261007
.venv/bin/python $S dev/experiments/population_linux_gpyreg140_20260930 $R/oldstack_0dc5932f_20261007 > $E/same_runs_oldstack.txt
```

- PyBADS at `0dc5932f` (`main`, #118), from the main checkout (clean). Its
  package code is that of `6e5133a3`, the tag `v1.5.1`: `git diff
  6e5133a3 0dc5932f -- pybads` is empty. The records' version string,
  `0.1.dev50+g0dc5932f0`, is that of an editable install in a shallow
  clone without tags; `meta.git` names the commit.
- gpyreg 1.4.0, the tag `v1.4.0` (`682585f`, the CI pin), from a clone
  selected with `PYTHONPATH` (`meta.gpyreg_source`).
- Linux (a cloud container, kernel `6.18.44-fc-v77`, 4 cores, Intel Xeon
  at 2.80 GHz with AVX-512), Python 3.12.3, NumPy 2.5.3 (OpenBLAS
  0.3.34), SciPy 1.18.1 (OpenBLAS 0.3.31), each OpenBLAS on its SkylakeX
  kernel, one BLAS thread per run, a fresh process per run, four runs at
  a time, nothing else running; from 16:23 to 16:46 UTC on 2026-10-07.
  The run under the older versions followed, from 17:55 to 18:16 UTC.
- `dev/scripts/fingerprint.py`, with one BLAS thread: `917c279f2c777dbb`
  under these versions; `4146a986863602cb`, the hash of the previous
  reference's setting, for the same code on the same machine under Python
  3.11.17, NumPy 2.4.6 and SciPy 1.17.1 (OpenBLAS 0.3.31, SkylakeX).
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `null_check.md`, `comparison.md` (with the previous Linux
  reference), and `same_runs.txt` and `same_runs_oldstack.txt`, the output
  of the previous reference's `same_runs.py`, which counts per
  configuration the runs whose start and result (`x`, `fval`, `fsd`,
  `func_count`, `iterations` and `message`) equal the previous
  reference's.

## Outcome

All 720 runs finished; none crashed.

**The same code under the previous reference's versions**
(`same_runs_oldstack.txt`): all 720 runs equal the previous reference's,
in every field of the records but the provenance, the wall time and the
stage times. PyBADS's changes from `60ad9e0f` to 1.5.1 leave the suite's
runs as they were; this machine, whose processor is not the previous
reference's (an Intel Xeon at 2.10 GHz), repeats them; and every
difference between the two references comes from the versions of Python,
NumPy and SciPy, with their OpenBLAS.

**Against the previous reference** (`comparison.md`): no configuration
flagged in 72 tests. 56 of the 720 runs equal the previous reference's
(`same_runs.txt`), all of them noisy or periodic but one (16 of
`sphere_D3_homo`, 15 of `periodic_D3_hetero`, 11 of `sphere_D3_hetero`,
9 of `periodic_D3_homo`, 4 of `periodic_D2` and 1 of `ackley_D6`); the
others take other paths from the last bits that the versions change. The
median paired log10 error ratios lie between -0.19 and +0.53, and two of
the 24 95% intervals exclude 0: `ellipsoid_D3_unbounded`, +0.53 (median
error 8.6e-7 against 2.5e-6, both far below its tolerance of 1e-3), and
`ellipsoid_D3_homo`, -0.19 (median error 0.119 against 0.067, the fraction
solved 0.43 against 0.60). The fraction solved changes most on
`ellipsoid_D3_hetero`, 0.23 to 0.00, whose median error, 0.261 in the
previous reference and 0.344 here, lies above its tolerance of 0.1 in
both.

Four at a time on this machine, the runs under the newer versions took
8 % longer than the same code's under the older ones, summed over the
suite, and 3 to 29 % longer per evaluation (the median of each
configuration): a wall time compared between the two references includes
this, beside any difference between their machines.

## Checks

- **Null check** (`compare <population> --split`, even against odd seeds,
  KS tests alone; `null_check.md`): no flag in 48 tests.
- **Positive controls**: the one in the first reference
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)),
  whose tests this comparison shares; for the periodic configurations, the
  comparison of the arms of
  [`population_periodic_linux_20260928`](../population_periodic_linux_20260928/README.md),
  which flags every one of them.

## What the comparison detects at 30 seeds

Against a population that holds the same 24 configurations: 72 tests, the
first Holm step at p ≤ 6.9e-4, a KS statistic of at least 0.533, and, for
the paired signed-rank test, a shift of about 0.87 of the standard
deviation of the paired log10 error ratios at 80% power (the simulation of
the previous reference, whose design this one shares). Between two
versions of PyBADS run under the same versions of Python, NumPy and SciPy
on this platform, a run that a change does not reach is identical in both
populations.
