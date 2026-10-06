<!-- The steps of the two batches of wave 1's fix pass whose comparison flags a configuration, on that configuration alone: 30 seeds at each commit of the batch, run from a clean worktree at the commit with population.py run --only (the runs, gitignored, in dev/scripts/runs/population/steps/), gpyreg 1.3.3, Linux, 2026-09-26. The first and last rows are the records of the batch populations. -->

# Steps of the flagged batches

The median of the true error over seeds 0-29 at each commit, and against
the commit before it the median paired log10 ratio of the errors and the
number of runs that differ.

## Batch 2, `ackley_D6` (flagged: a lower error)

| Commit | Row | Median error | Median log10 ratio vs previous | Runs that differ |
|---|---|---|---|---|
| `1c7200b` | batch 1's end | 4.01e-4 | | |
| `d1caf6b` | W0-8 | 3.42e-4 | +0.000 | 29 |
| `ee9d5d6` | W1-22 | 3.35e-4 | +0.038 | 30 |
| `1d03801` | W1-29 | 3.48e-4 | +0.021 | 30 |
| `172df00` | W1-23 | 1.87e-4 | -0.313 | 30 |
| `3236c2f` | W0-7 | 1.87e-4 | +0.000 | 0 |

W1-23, the constant mean unbounded, lowers the error; the other rows move
it at the level of rounding (W0-8 changes the order of the training rows
alone), and W0-7 does not reach this configuration.

## Batch 3, `sphere_D2` (flagged: a lower error)

| Commit | Row | Median error | Median log10 ratio vs previous | Runs that differ |
|---|---|---|---|---|
| `3236c2f` | batch 2's end | 2.65e-6 | | |
| `e420486` | W1-4 | 7.09e-7 | -0.457 | 30 |
| `bd49445` | W1-5 | 7.09e-7 | +0.000 | 0 |
| `d883cf9` | W1-3 | 7.22e-7 | +0.000 | 5 |

W1-4, the calibration test that counts its statistics and refits at the
refit period, lowers the error; W1-5 changes no run of this configuration,
and W1-3 five, by little.
