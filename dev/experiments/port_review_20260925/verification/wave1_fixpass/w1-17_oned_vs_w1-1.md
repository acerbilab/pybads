<!-- W1-17 at cfacb98 against W1-1 (6e22d32) on the oned suite of db09fb6 (the 1-D configurations); seeds 0-29, gpyreg 1.3.3, Linux, 2026-09-26. 43 of the 180 runs differ between the two. Saved verbatim from the output of `population.py compare` (the runs, gitignored, in `dev/scripts/runs/population/`). -->

# Compare oned_6e22d32 (REF) and oned_cfacb98 (NEW)

- REF, 180 runs: pybads 6e22d32 (dirty), gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 180 runs: pybads cfacb98 (dirty), gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D1 | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ackley_D1 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ackley_D1 | signed-rank | log10 true_error | 30 | 30 | 0 | 0.0431 | 0.776 | ok |
| ellipsoid_D1_unbounded | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ellipsoid_D1_unbounded | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D1_unbounded | signed-rank | log10 true_error | 30 | 30 | 20 | 0.767 | 1 | ok |
| rastrigin_D1 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D1 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D1 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D1 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1 | signed-rank | log10 true_error | 30 | 30 | 31 | 0.311 | 1 | ok |
| sphere_D1_hetero | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D1_hetero | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D1_hetero | signed-rank | log10 true_error | 30 | 30 | 6 | 0.686 | 1 | ok |
| sphere_D1_homo | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D1_homo | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D1_homo | signed-rank | log10 true_error | 30 | 30 | 13 | 0.26 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D1 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D1_unbounded | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rastrigin_D1 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D1 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D1_hetero | 30 | +0.000 [+0.000, +0.000] | 0.67 | 0.67 | +0.00 | 0 | 0 |
| sphere_D1_homo | 30 | +0.000 [+0.000, +0.000] | 0.93 | 0.97 | +0.03 | 0 | 0 |

Holm family: 18 tests at alpha 0.05. A flag needs p <= 0.0028 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**no configuration flagged (18 tests)**
