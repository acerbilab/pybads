# Compare pop (REF) and pop (NEW)

- REF, 150 runs: pybads 46af65a, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 150 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 300 runs: pybads 73d5c28, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 60 | 60 | 0.183 | 0.267 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 690 | 0.821 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 60 | 60 | 0.0667 | 0.999 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 60 | 60 | 737 | 0.477 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 60 | 60 | 0.133 | 0.665 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 60 | 60 | 652 | 0.166 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 60 | 60 | 0.15 | 0.513 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 588 | 0.127 | 1 | ok |
| sphere_D3_homo | KS | true_error | 60 | 60 | 0.15 | 0.513 | 1 | ok |
| sphere_D3_homo | KS | func_count | 60 | 60 | 0.25 | 0.0467 | 0.7 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 60 | 60 | 454 | 0.115 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 60 | +0.000 [-0.091, +0.150] | 0.17 | 0.17 | +0.00 | 0 | 0 |
| ellipsoid_D3_homo | 60 | +0.000 [-0.164, +0.284] | 0.57 | 0.52 | -0.05 | 0 | 0 |
| multisensory_s1_D6_homo | 60 | -0.026 [-0.152, +0.088] | 0.98 | 0.95 | -0.03 | 0 | 0 |
| sphere_D3_hetero | 60 | +0.070 [-0.001, +0.146] | 0.52 | 0.50 | -0.02 | 0 | 0 |
| sphere_D3_homo | 60 | +0.005 [+0.000, +0.064] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 60 vs 60 runs that is a KS statistic of at least 0.333.

**no configuration flagged (15 tests)**
