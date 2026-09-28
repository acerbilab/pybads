# Compare pop (REF) and pop (NEW)

- REF, 450 runs: pybads 0cc795f, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 450 runs: pybads 58e7dd5, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 90 | 90 | 0.0778 | 0.95 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 90 | 90 | 0.111 | 0.638 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 90 | 90 | 1.59e+03 | 0.897 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 90 | 90 | 0.0556 | 0.999 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 90 | 90 | 0.0778 | 0.95 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 90 | 90 | 1.63e+03 | 0.893 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 90 | 90 | 0.1 | 0.762 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 90 | 90 | 0.211 | 0.036 | 0.539 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 90 | 90 | 1.58e+03 | 0.28 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 90 | 90 | 0.1 | 0.762 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 90 | 90 | 0.0667 | 0.989 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 90 | 90 | 1.82e+03 | 0.991 | 1 | ok |
| sphere_D3_homo | KS | true_error | 90 | 90 | 0.0667 | 0.989 | 1 | ok |
| sphere_D3_homo | KS | func_count | 90 | 90 | 0.122 | 0.515 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 90 | 90 | 766 | 0.197 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 90 | +0.000 [-0.023, +0.031] | 0.16 | 0.16 | +0.00 | 0 | 0 |
| ellipsoid_D3_homo | 90 | +0.000 [-0.137, +0.014] | 0.62 | 0.61 | -0.01 | 0 | 0 |
| multisensory_s1_D6_homo | 90 | -0.002 [-0.029, +0.001] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 90 | -0.007 [-0.083, +0.024] | 0.54 | 0.50 | -0.04 | 0 | 0 |
| sphere_D3_homo | 90 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 90 vs 90 runs that is a KS statistic of at least 0.267.

**no configuration flagged (15 tests)**
