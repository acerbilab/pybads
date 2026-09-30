# gpyreg 1.4.0's comparison on Linux: the `default` suite, 30 seeds, gpyreg at `3e56dce`

The Linux half of the gate of PyBADS's move to gpyreg 1.4.0
([`TODO.md`](../../TODO.md), "gpyreg releases after 1.3.3"): the 24
configurations of the `default` suite of `dev/scripts/benchmark_targets.py`,
the six with periodic variables (the `periodic` suite) included, seeds
0-29, with gpyreg at `3e56dce`, the head of gpyreg's `main` that holds
everything planned for 1.4.0 (the squash commit of acerbilab/gpyreg#66).
It is compared with the Linux reference of the suite,
[`population_linux_wave4_20260927`](../population_linux_wave4_20260927/README.md)
(gpyreg 1.3.3, which cannot run the periodic configurations), and its
periodic configurations with the "on" arm of
[`population_periodic_linux_20260928`](../population_periodic_linux_20260928/README.md)
(gpyreg at `2c9cdfb`). If 1.4.0's tag carries `3e56dce`'s code, it is the
Linux reference of the `default` suite from the move to 1.4.0 on.

## Command and provenance

The runs were made as two populations, when the `default` suite held two
of the periodic configurations and the `periodic` suite all six, and
merged into one of the 24 configurations of the suite as it stands
(`bef26ec2`):

```console
E=dev/experiments/population_linux_gpyreg140_20260930
R=dev/scripts/runs/population
G=dev/scripts/runs/gpyreg/main_3e56dce   # git clone https://github.com/acerbilab/gpyreg $G && git -C $G checkout 3e56dce
PYTHONPATH=$G .venv/bin/python -u dev/scripts/population.py run --suite default --seeds 0-29 --workers 4 --out $R/release140_default_20260930
PYTHONPATH=$G .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --out $R/release140_periodic_20260930
python $E/same_runs.py $R/release140_default_20260930 $R/release140_periodic_20260930   # periodic_D4, periodic_D3_homo: 60/60 identical
cp $R/release140_default_20260930/*_seed*.json $E/
cp $R/release140_periodic_20260930/periodic_{D2,D6,D3_hetero,rosenbrock_D4}_seed*.json $E/
python dev/scripts/population.py summary $E
python dev/scripts/population.py compare $E --split > $E/null_check.md
python dev/scripts/population.py compare dev/experiments/population_linux_wave4_20260927 $E > $E/comparison.md
python dev/scripts/population.py compare dev/experiments/population_periodic_linux_20260928/on $E > $E/compare_periodic.md
```

`same_runs.txt` holds the output of `same_runs.py REF NEW`, which counts
per configuration the runs whose start and result (`x`, `fval`, `fsd`,
`func_count`, `iterations`, `message`) are equal in both populations, and
of the port review's
`verification/scripts/wave3/orchestrator/same_fields.py REF NEW`, which
names every field of the records that differs, the wall time and the
provenance left out, for the pairs of populations below; `default` and
`periodic` there are the two populations as run.

- PyBADS at `60ad9e0f` (`dev-next`), from the main checkout (clean), whose
  package code is that of `e6733592` (#103) and of `bef26ec2`, which moves
  the periodic configurations into the `default` suite. The records'
  version string, `0.1.dev50+g60ad9e0f5`, is that of an editable install
  in a shallow clone without tags; `meta.git` names the commit.
- gpyreg at `3e56dce`, from a clone selected with `PYTHONPATH`
  (`meta.gpyreg_source`); the records' version string,
  `1.3.4.dev35+g3e56dce0f`, is that of the venv's editable install of
  `../gpyreg` at the same commit. `dev/scripts/fingerprint.py` prints
  `4146a986863602cb`, the hash of gpyreg 1.3.3 and of the Linux
  references, with gpyreg at `3e56dce` and at `v1.3.3`, on this machine
  and one BLAS thread.
- Linux (a cloud container, kernel `6.18.44-fc-v50`, 4 cores, Intel Xeon
  at 2.10 GHz), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, OpenBLAS 0.3.31,
  one BLAS thread per run, a fresh process per run, nothing else running;
  the `default` run from 04:50 to 05:06 UTC, four runs at a time, and the
  `periodic` run from 05:06 to 05:11 UTC, three at a time, on 2026-09-30.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (with the reference of the suite),
  `compare_periodic.md` (with the periodic reference), `null_check.md`,
  `same_runs.py` and `same_runs.txt`.

## Outcome

All 780 runs of the two populations finished; none crashed. The 60 runs
of `periodic_D4` and `periodic_D3_homo` that both made are equal, run by
run, in every field but the wall time and the stage times.

**Against the reference of the suite** (`comparison.md`): no configuration
flagged in 54 tests. The 540 runs of the 18 configurations that the
reference holds are equal to the reference's, run by run, in every field
but the wall time and `precomputed`, a field that the reference's records
predate (null here): at default options, gpyreg's changes after 1.3.3
leave PyBADS's runs as they were, as the fingerprint's identity implies.
The six periodic configurations, which the reference lacks, are listed
outside the verdict.

**The periodic configurations against the periodic reference**
(`compare_periodic.md`): no configuration flagged in 18 tests. 146 of the
180 runs differ from the reference's (all 30 of `periodic_D4`,
`periodic_D6` and `periodic_rosenbrock_D4`, 25 of `periodic_D2`, 18 of
`periodic_D3_homo`, 13 of `periodic_D3_hetero`): the periodic kernel of
gpyreg's `0f27db5` (acerbilab/gpyreg#62), which maps each periodic
coordinate onto a circle, gives other last bits than `2c9cdfb`'s, and the
runs take other paths from them. All 180 runs are equal, run by run, to
those of the gate of that change,
[`periodic_kernel_linux_20260929/new_on`](../periodic_kernel_linux_20260929/README.md)
(gpyreg `0f27db5`, PyBADS `9a705918`): gpyreg's changes after `0f27db5`
leave the periodic runs as they were too. The median paired log10 error
ratios against the reference lie between -0.15 and 0.00, every 95%
interval containing 0; the fraction solved is unchanged but for
`periodic_D3_hetero`, 0.27 → 0.20, whose median error, 0.147 in the
reference and 0.136 here, lies near its tolerance of 0.1.

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
deviation of the paired log10 error ratios at 80% power (simulated; the
same simulation gives 0.86 for the 54 tests of 18 configurations, which
the earlier references state as 0.87). Between two versions on this
platform, a run that a change does not reach is identical in both
populations.
