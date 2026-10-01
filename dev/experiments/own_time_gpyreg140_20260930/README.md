# PyBADS's own time with gpyreg 1.4.0 against gpyreg 1.3.3 (Windows)

The time that gpyreg 1.4.0 saves PyBADS, part of the speed-up that the
entry "Speed" of `CHANGELOG.md` states: the same PyBADS, run under gpyreg 1.3.3 (arm A) and under gpyreg at
`3e56dce` (arm B), whose code the tag `v1.4.0` carries (the tag adds only
the date of the release notes). With gpyreg 1.4.0, PyBADS's own time, a run's time less
its target's evaluations, is 27 to 39 % lower on the seven configurations
of the `profile` suite (configuration medians of 0.614 to 0.731 of the
time under 1.3.3), 30 % at the median of the 21 pairs of runs, every run
giving the same result under both.

## Command and provenance

From the main checkout's root, with a clean detached worktree at
`bef26ec2` under `dev/scripts/runs/worktrees/winref_bef26ec2` and the two
gpyreg clones of `dev/scripts/runs/LOCAL.md` (`gpyreg/v1.3.3`, `98ab5a4`;
`gpyreg/main_3e56dce`):

```console
E=dev/experiments/own_time_gpyreg140_20260930
R=dev/scripts/runs/release150/timing_abba
.venv/Scripts/python.exe -u $E/timing_abba.py $R --cpu 12 --seeds 0-2 --reps 2
python $E/tabulate.py collect $R > $E/timing.json
python $E/tabulate.py table $E/timing.json > $E/timing.md
```

- `timing_abba.py` runs `dev/scripts/profile_run.py` of the worktree, plain
  (no cProfile), for each configuration of the `profile` suite and seeds
  0-2, twice in each arm, one run at a time; the arms alternate ABBA over
  the two repetitions of each configuration and seed, so that a slow drift
  of the machine reaches both. Each run is its own process with one BLAS
  thread, gpyreg selected by an absolute `PYTHONPATH`; every record names
  the gpyreg commit of its arm and the worktree at `bef26ec2`, clean.
- Each run is pinned to one performance core (logical CPU 12) with
  Windows's power throttling off: the machine's Intel Core Ultra 7 155H
  has performance, efficiency and low-power efficiency cores, and, under
  the "Balanced" power plan, the slowest of four unpinned runs of the
  same code, configuration and seed took up to 2.0 times as long as the
  fastest (1.5 times at the median). Pinned, the slower of the two runs
  of one arm takes 2.8 % longer than the faster at the median, and 42 %
  longer in one pair (`ellipsoid_D3_homo` seed 2, arm B); the table takes
  the faster run of each arm.
- Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1, the laptop's fan
  profile at "Standard", nothing else running; from 09:54 to 10:06
  (UTC+3) on 2026-09-30.
- The files: `timing_abba.py` and `tabulate.py`; `timing.json`, one row per
  run; `timing.md`, the table.

## Outcome

| configuration | own time, 1.4.0 / 1.3.3 (median of 3 seeds) |
|---|---|
| ellipsoid_D10 | 0.614 |
| multisensory_s1_D6_homo | 0.676 |
| ellipsoid_D3 | 0.701 |
| rosenbrock_D6 | 0.704 |
| ellipsoid_D3_homo | 0.718 |
| ackley_D6 | 0.719 |
| sphere_D3_hetero | 0.731 |

The four runs of each configuration and seed give the same number of
evaluations and the same returned value: gpyreg 1.4.0 computes what 1.3.3
computes, to the last bit, on these runs as on the runs of the `default`
suite without periodic variables
([`population_gpyreg140_20260930`](../population_gpyreg140_20260930/README.md)).
The saving comes from gpyreg's kernels, their gradients, `predict` and the
objective of `fit` computed with fewer intermediate arrays and without
SciPy's layers (acerbilab/gpyreg#60, #63 and #64). The earlier
measurement of #63 and #64 alone, against gpyreg `e10120c`, which already
held #60, gave 19 to 27 % under SciPy 1.18 on the same configurations, in
a Linux container
([`results/2026-09-29-bit-identical-speedups.md`](../../results/2026-09-29-bit-identical-speedups.md));
that measurement counted PyBADS's one line that takes the priors once per
rebuild, which both arms here hold.
