# PyBADS's own time against PyBADS 1.1.0 (Windows)

What a user of PyBADS 1.1.0 gains in time by moving to the code of
`dev-next` at `1ecfeb61` (2026-09-30): PyBADS 1.1.0 with gpyreg 1.3.3, the
minimum it names (arm A), against PyBADS at `1ecfeb61` with gpyreg 1.4.0,
the minimum it names (arm B). On the seven configurations of the `profile`
suite at 30 seeds, PyBADS's own time, a run's time less its target's
evaluations, summed over the 210 runs of each arm, is 0.562 of 1.1.0's
(1120 s against 1992 s). Per configuration, the median ratio of the two
runs of one seed is 0.46 to 0.81 and B is faster in 22 to 30 of the 30
seeds. The runs of B also end after fewer evaluations, at errors that are
equal or smaller, and each evaluation costs less of PyBADS's time. The
entry "Speed" of `CHANGELOG.md` states these figures, and "What's new", in
`README.md` and in `docsrc/source/index.rst`, states the speed-up, 1.78
times over all the runs (1.72 as the geometric mean of the
configurations' medians, 1.23 to 2.17 per configuration), as "almost
twice as fast".

The two arms run different code, so the runs of one seed share their
problem, start point and noise, but not their course: PyBADS 1.1.0 already
draws through a seeded `numpy.random.Generator`, but the port review
changed the algorithm, and the seed decides the initial design, which in
1.1.0 it did not. The comparison is therefore between distributions over
seeds,
unlike [`own_time_gpyreg140_20260930`](../own_time_gpyreg140_20260930/README.md),
whose arms ran the same PyBADS and so the same runs.

## Command and provenance

From the main checkout's root, with two detached worktrees under
`dev/scripts/runs/worktrees/` and the gpyreg clones of
`dev/scripts/runs/LOCAL.md` (`gpyreg/v1.3.3`, `98ab5a4`; `gpyreg/v1.4.0`,
`682585f`):

- `timing_v110`, at the tag `v1.1.0` (`1d075ab0`), with
  `dev/scripts/profile_run.py`, `population.py`, `benchmark_targets.py`
  and `harness.py` of `1ecfeb61` copied into it, so that both arms build
  the same problems, start points and noise (`dev/README.md` describes the
  measurement of a commit from before the stage timers this way). Its
  package is as tagged, and the records name it clean.
- `timing_1ecfeb61`, at `1ecfeb61`, clean.

```console
E=dev/experiments/own_time_v110_20261001
R=dev/scripts/runs/release150/timing_v110
.venv/Scripts/python.exe -u $E/timing_ab.py $R --cpu 12 --seeds 0-29
python $E/tabulate.py collect $R > $E/timing.json
python $E/tabulate.py table $E/timing.json > $E/timing.md
```

- `timing_ab.py` runs `dev/scripts/profile_run.py` of each arm's worktree,
  plain (no cProfile), for each configuration of the `profile` suite and
  seeds 0-29, one run at a time, seed by seed; the order of the arms
  alternates AB and BA from one configuration and seed to the next. Each
  run is its own process with one BLAS thread, gpyreg selected by an
  absolute `PYTHONPATH`, pinned to logical CPU 12, a performance core
  (efficiency class 1, as `GetSystemCpuSetInformation` reports), with
  Windows's power throttling off. Beside each run's `summary.json`,
  `machine.json` holds the load of the whole machine while it ran.
- PyBADS 1.1.0 has no stage timers, so the own time of both arms is
  computed one way: the wall time of `optimize()` less the function
  logger's `total_fun_eval_time` (`tabulate.py`). The targets' evaluations
  took 0.6 % of the wall time in arm A and 0.9 % in arm B.
- Windows 11, Intel Core Ultra 7 155H, Python 3.12.6, NumPy 2.5.3, SciPy
  1.18.1, the "Balanced" power plan on mains power; from 03:48 to 04:52
  (UTC+3) on 2026-10-01, with no other job running. The load of all 22
  logical CPUs during a run was 6.6 to 20.3 %, 11.2 % at the median, of
  which one pinned run is 4.5 %; the 21 runs above 15 % are 11 of arm A
  and 10 of arm B. All 420 runs exited with 0.
- The files: `timing_ab.py` and `tabulate.py`; `timing.json`, one row per
  run; `timing.md`, the table; `compare_populations.md`, the comparison of
  the results below.
- The raw runs, each run's `summary.json`, `machine.json` and console log
  and the log of the whole measurement, are attached to the release
  v1.5.0 as
  [`own_time_v110_20261001_raw.zip`](https://github.com/acerbilab/pybads/releases/download/v1.5.0/own_time_v110_20261001_raw.zip)
  (1.4 MB). `tabulate.py collect` of its `timing_v110/` rebuilds
  `timing.json` byte for byte.

## Outcome

| configuration | own time A (s) | own time B (s) | B / A per seed, median [95 % CI] | B faster | evaluations A | evaluations B | median error A | median error B |
|---|---|---|---|---|---|---|---|---|
| rosenbrock_D6 | 7.82 | 3.72 | 0.460 [0.421, 0.532] | 30/30 | 468 | 424 | 4.6e-05 | 2.9e-06 |
| multisensory_s1_D6_homo | 14.12 | 7.59 | 0.530 [0.472, 0.570] | 28/30 | 626 | 554 | 0.18 | 0.19 |
| ackley_D6 | 4.85 | 2.62 | 0.546 [0.527, 0.564] | 30/30 | 394 | 382 | 3.9e-04 | 1.7e-04 |
| ellipsoid_D10 | 18.64 | 9.77 | 0.553 [0.458, 0.643] | 30/30 | 705 | 672 | 6.8e-05 | 4.1e-07 |
| sphere_D3_hetero | 7.96 | 4.10 | 0.564 [0.453, 0.678] | 26/30 | 401 | 338 | 0.20 | 0.098 |
| ellipsoid_D3_homo | 9.84 | 6.56 | 0.667 [0.610, 0.736] | 29/30 | 370 | 310 | 0.11 | 0.087 |
| ellipsoid_D3 | 1.91 | 1.64 | 0.810 [0.714, 0.949] | 22/30 | 147 | 138 | 2.4e-05 | 1.8e-06 |

Medians over the 30 seeds; the interval is a bootstrap of the median of
the 30 ratios. The own time per evaluation is 0.51 to 0.91 of 1.1.0's
(`timing.md`): the saving per run comes both from cheaper evaluations and
from fewer of them. gpyreg 1.4.0 alone, under the same PyBADS, accounts for
0.61 to 0.73 on these configurations
([`own_time_gpyreg140_20260930`](../own_time_gpyreg140_20260930/README.md)).

## The results of the two versions

The time is compared on runs that end at least as well. The populations of
the `default` suite on Windows hold both versions on the same problems:
[`population_gpyreg133_20260924`](../population_gpyreg133_20260924/README.md)
(30 seeds, at `2059506`, whose package differs from `v1.1.0` in docstrings
alone, with gpyreg 1.3.3) and
[`population_gpyreg140_20260930`](../population_gpyreg140_20260930/README.md)
(100 seeds, at `bef26ec2` with gpyreg 1.4.0's code; the package commits
from there to `1ecfeb61` change only messages, a warning and the minimum
gpyreg). `population.py compare` of the two (`compare_populations.md`)
flags 8 of the 18 configurations that both hold, each in favour of the
later code: smaller errors on `ackley_D6`, `ellipsoid_D3`,
`ellipsoid_D3_unbounded`, `ellipsoid_D6`, `ellipsoid_D10`, `rosenbrock_D2`
and `rosenbrock_D6` (median error ratios of 10^-0.4 to 10^-2.6, and the
fraction solved of `ellipsoid_D3` and `ellipsoid_D3_unbounded` from 0.93
and 0.83 to 1.00), and fewer evaluations at an unchanged error on
`ellipsoid_D3_homo` (median 370 to 310). None is flagged against it; the
largest unflagged move against it is `multisensory_s1_D6_homo`, +0.12 in
the log10 error ratio (95 % CI -0.02 to +0.22), with a fraction solved of
1.00 of 30 runs against 0.97 of 100.
The runs timed here agree: B's median error is equal or smaller on every
configuration but `multisensory_s1_D6_homo` (0.19 against 0.18), at fewer
evaluations on all seven.
