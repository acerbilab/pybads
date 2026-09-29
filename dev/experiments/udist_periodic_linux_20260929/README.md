# `udist` on periodic variables one variable at a time, on Linux

`udist` (`pybads/search/grid_functions.py`) with periodic variables at
`0a7f7af4`: one `(N, M)` array of squared differences per variable,
`np.mod` taken only for the differences of a period or more, which it
leaves as they are below, and the sum over the variables in the order in
which `np.sum` adds along an axis (`_pairwise_sum`), against its base
`20fc40f6`, which built the `(N, M, D)` array of every difference and summed
it along the variables. The base is the head of `dev-next`; the two commits
between them, `d757ef28` (the checks of option values when `BADS` is
created) and `31890cb0` (documentation), are covered by the same
comparisons.

## Command and provenance

Run from the main checkout's root at `0a7f7af4`; `E` is this directory,
`P` a worktree at `20fc40f6`, `G` a clone of gpyreg at `b44634f`, `R` a
scratch directory.

```console
PER=periodic_D2,periodic_D4,periodic_D6,periodic_D3_homo,periodic_D3_hetero,periodic_rosenbrock_D4
PYTHONPATH=$G .venv/bin/python -u $P/dev/scripts/replay.py record --out $R/replay_parent
PYTHONPATH=$G .venv/bin/python -u dev/scripts/replay.py record --out $R/replay_head
PYTHONPATH=$G .venv/bin/python -u $P/dev/scripts/replay.py record --configs $PER --out $R/replay_periodic_parent
PYTHONPATH=$G .venv/bin/python -u dev/scripts/replay.py record --configs $PER --out $R/replay_periodic_head
.venv/bin/python dev/scripts/replay.py check $R/replay_parent $R/replay_head > $E/replay_default.txt
.venv/bin/python dev/scripts/replay.py check $R/replay_periodic_parent $R/replay_periodic_head > $E/replay_periodic.txt
PYTHONPATH=$G .venv/bin/python -u $P/dev/scripts/make_oracle_fixtures.py --dump $R/oracles_parent
PYTHONPATH=$G .venv/bin/python -u dev/scripts/make_oracle_fixtures.py --check --exact --against $R/oracles_parent   # oracles.txt
PYTHONPATH=$G .venv/bin/python -u $P/dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --out $R/periodic/parent
PYTHONPATH=$G .venv/bin/python -u dev/scripts/population.py run --suite periodic --seeds 0-29 --workers 3 --out $R/periodic/head
.venv/bin/python $E/identity.py $R/periodic/parent $R/periodic/head > $E/identity.md
.venv/bin/python $E/speed.py $R/periodic/parent $R/periodic/head > $E/speed.md
.venv/bin/python $E/random_cases.py $P/pybads/search/grid_functions.py
(cd $P && OMP_NUM_THREADS=1 PYTHONPATH=$G <repository>/.venv/bin/python -u <repository>/$E/udist_calls.py periodic_D3_homo 0)
OMP_NUM_THREADS=1 PYTHONPATH=$G .venv/bin/python -u $E/udist_calls.py periodic_D3_homo 0
```

- PyBADS: the base `20fc40f6` from the worktree, the change `0a7f7af4`
  from the main checkout, both clean; each side ran its own scripts, which
  are the same at both commits.
- gpyreg at `b44634f`, CI's `GPYREG_PIN` (the kernels' periods), from the
  clone, selected with `PYTHONPATH`. `dev/scripts/fingerprint.py` prints
  `917c279f2c777dbb` at both commits.
- Linux (a cloud container, kernel `6.18.44-fc-v37`, Intel Xeon at
  2.10 GHz, four virtual CPUs), Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1,
  one BLAS thread per run (OpenBLAS's Haswell kernels for the replays);
  the populations three runs at a time, the base from 10:52 to 11:00 UTC
  and the change from 11:00 to 11:08, on 2026-09-29, nothing else running.
- The files: `identity.py` and `identity.md`, the populations compared run
  by run; `speed.py` and `speed.md`, the time of the GP's rebuilds, paired
  by run; `replay_default.txt` and `replay_periodic.txt`; `oracles.txt`;
  `random_cases.py`, `udist` against the base's on random cases;
  `udist_calls.py`, the number, time and shapes of the calls of `udist` in
  one run, which runs the `pybads` of the checkout it is started from.
- The per-run records, traces and dump were not kept: they were written in
  a cloud container that is gone. The commands above regenerate them,
  seeded as they were, on a machine whose fingerprint is the one above.

## Outcome

- **The `periodic` suite at 30 seeds**: all 180 runs identical, timings
  aside (`identity.md`).
- **Replays**: the default set (eight configurations without periodic
  variables, seed 0) identical, and the six configurations of the
  `periodic` suite, seed 0, identical at every evaluation, step and GP
  computation (`replay_default.txt`, `replay_periodic.txt`).
- **Oracles**: all 1056 outputs identical to the base's dump
  (`oracles.txt`); no oracle state has periodic variables.
- **Random cases**: 3000 of 3000 identical to the base's `udist`, D from 1
  to 20. The positive control, the same script against a copy of the
  change that adds the variables' terms in turn instead of in `np.sum`'s
  order, finds 1088 of 3000 identical, every case at D of 8 or more
  differing, by up to 9e-16 relative; below 8 terms the two orders are the
  same.
- **Time** (`speed.md`): on the two noisy configurations, whose local
  training sets reach 200 points, the GP's rebuilds take 0.68 of the base's
  time, a saving of 7.7 and 8.5 % of the run's own time; on the four
  deterministic ones, 0.6 to 1.2 %. `udist` itself (`udist_calls.py`,
  seed 0 of `periodic_D3_homo`, three runs of each commit, one at a time):
  697 calls, 300 of them between all the training inputs, up to 200 × 200
  × 3; 0.52 to 0.57 s in the base (0.75 to 0.82 ms per call, about 2.9 ms
  at 200 × 200) and 0.13 to 0.14 s in the change (0.19 to 0.20 ms per
  call). The wall times of populations run three at a time are noisier:
  0.90 to 1.02 of the base's.
