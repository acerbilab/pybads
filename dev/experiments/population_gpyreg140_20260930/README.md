# gpyreg 1.4.0's comparison on Windows: the `default` suite, 100 seeds, gpyreg at `3e56dce`

The Windows half of the gate of PyBADS's move to gpyreg 1.4.0
([`TODO.md`](../../TODO.md), "gpyreg releases after 1.3.3"), and the
reference on Windows for `dev/scripts/population.py compare` from that move
on. The one on Linux, under the same versions of Python, NumPy and SciPy,
is
[`population_linux_scipy118_20261007`](../population_linux_scipy118_20261007/README.md).
The 24 configurations of the `default` suite of
`dev/scripts/benchmark_targets.py`, the six with periodic variables
included, × seeds 0-99, each run at BADS's default budget (500 D) and
ending on BADS's own termination criteria, with gpyreg at `3e56dce`, the
head of gpyreg's `main` that holds everything of 1.4.0 (the squash commit
of acerbilab/gpyreg#66). The tag `v1.4.0` (`682585f`) differs from
`3e56dce` only in the date of the release notes.

It is compared with the previous reference on Windows,
[`population_wave4_20260928`](../population_wave4_20260928/README.md)
(PyBADS `a4dcd65`, gpyreg 1.3.3), which lacks the periodic configurations:
gpyreg 1.3.3 cannot run them, and Windows has no earlier run of them.

## Command and provenance

From the main checkout's root, with a clean detached worktree at
`bef26ec2` under `dev/scripts/runs/worktrees/`:

```console
R=dev/scripts/runs/population/population_release150_20260930
E=dev/experiments/population_gpyreg140_20260930
PYTHONPATH=dev/scripts/runs/gpyreg/main_3e56dce .venv/Scripts/python.exe -u dev/scripts/runs/worktrees/winref_bef26ec2/dev/scripts/population.py run --suite default --seeds 0-99 --workers 4 --out $R
mkdir -p $E && cp $R/*_seed*.json $E/
python dev/scripts/population.py summary $E
python dev/scripts/population.py compare $E --split > $E/null_check.md
python dev/scripts/population.py compare dev/experiments/population_wave4_20260928 $E > $E/comparison.md
```

- PyBADS at `bef26ec2`, run by the worktree's own `population.py`, which
  puts that checkout first on `sys.path`: every record's
  `meta.pybads_source` names the worktree at `bef26ec2`, clean. Its
  package code is that of `e6733592` (#103); `bef26ec2` moves the
  periodic configurations into the `default` suite. The records' `pybads`
  version string is the metadata of the venv's editable install of the
  main checkout.
- gpyreg at `3e56dce`, from a clone selected with `PYTHONPATH`
  (`meta.gpyreg_source`); the records' version string, 1.3.3, is the
  metadata of the venv's editable install of `../gpyreg`. With one BLAS
  thread, `dev/scripts/fingerprint.py` prints `093cb1d05a16d889` at
  `bef26ec2` with gpyreg at `3e56dce` and at `v1.3.3`, the hash of the
  previous reference's code with gpyreg 1.3.3.
- Windows 11 (Intel Core Ultra 7 155H), Python 3.12.6, NumPy 2.5.3, SciPy
  1.18.1, as for the previous reference; one BLAS thread per run
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS` = 1), a fresh process per run, four runs at a
  time, the laptop's fan profile at "Standard" (under "Performance", four
  or more busy cores cut the machine's power; see the previous
  reference). 86 minutes, from 08:16 to 09:43 (UTC+3) on 2026-09-30.
- The files: one JSON record per run (`<label>_seed<seed>.json`),
  `summary.md`, `comparison.md` (the comparison with the previous
  reference) and `null_check.md`.

## Outcome

All 2400 runs finished; none crashed. The fraction solved ranges from 0.04
(`rastrigin_D3`, whose runs end in local minima) to 1.00.

`compare dev/experiments/population_wave4_20260928 <this population>`
(`comparison.md`) flags no configuration in 54 tests. The 1800 runs of
the 18 configurations that the previous reference holds are equal to its
runs, run by run, in every field of their results but the wall time,
and `precomputed` and the stage times, which the previous reference's
records predate: at default options, gpyreg's changes after 1.3.3
leave PyBADS's runs as they were, as the fingerprint's identity
implies, and as on Linux. The six periodic configurations are listed
outside the verdict.

## Checks

- **Null check** (`compare REF --split`, even against odd seeds, KS tests
  alone; `null_check.md`): no flag in 48 tests.
- **Positive control**: the one in the first reference
  ([`population_baseline_20260924`](../population_baseline_20260924/README.md)),
  whose tests this comparison shares; for the periodic configurations, the
  comparison of the arms of
  [`population_periodic_linux_20260928`](../population_periodic_linux_20260928/README.md)
  (Linux), which flags every one of them.

## What the comparison detects at 100 seeds

Against a population that holds the same 24 configurations: 72 tests, the
first Holm step at p ≤ 6.9e-4, a KS statistic of at least 0.290 for 100
against 100 runs, and, for the paired signed-rank test, a shift of about
0.45 of the standard deviation of the paired log10 error ratios at 80%
power (simulated; the same simulation gives the 0.44 of the 54 tests of 18
configurations that the previous reference states). Between two versions
on this platform, a run that a change does not reach is identical in both
populations.
