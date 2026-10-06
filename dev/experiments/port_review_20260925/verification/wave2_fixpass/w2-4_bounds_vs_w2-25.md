# Compare bounds_a9fbb97 (REF) and bounds_a236eb7 (NEW)

- REF, 150 runs: pybads a9fbb97, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 150 runs: pybads a236eb7, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| logsphere_D3 | KS | true_error | 30 | 30 | 0.533 | 0.000293 | 0.00352 | FLAG |
| logsphere_D3 | KS | func_count | 30 | 30 | 0.733 | 4.33e-08 | 6.06e-07 | FLAG |
| logsphere_D3 | signed-rank | log10 true_error | 30 | 30 | 49 | 4.97e-05 | 0.000646 | FLAG |
| logsphere_D3_homo | KS | true_error | 30 | 30 | 0.433 | 0.00655 | 0.0589 | ok |
| logsphere_D3_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| logsphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 80 | 0.00113 | 0.0124 | FLAG |
| logsphere_D3_nopb | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| logsphere_D3_nopb | KS | func_count | 30 | 30 | 0.467 | 0.00253 | 0.0253 | FLAG |
| logsphere_D3_nopb | signed-rank | log10 true_error | 30 | 30 | 188 | 0.371 | 1 | ok |
| sphere_D3_nopb | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D3_nopb | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_nopb | signed-rank | log10 true_error | 30 | 30 | 179 | 0.28 | 1 | ok |
| sphere_D3_x0lb | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_x0lb | KS | func_count | 30 | 30 | 0.867 | 8.25e-12 | 1.24e-10 | FLAG |
| sphere_D3_x0lb | signed-rank | log10 true_error | 30 | 30 | 181 | 0.299 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| logsphere_D3 | 30 | +0.632 [+0.304, +0.924] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| logsphere_D3_homo | 30 | -0.405 [-0.570, -0.134] | 0.33 | 0.73 | +0.40 | 0 | 0 |
| logsphere_D3_nopb | 30 | +0.162 [-0.145, +0.350] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_nopb | 30 | +0.210 [-0.117, +0.429] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_x0lb | 30 | +0.045 [-0.274, +0.616] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**4 configuration(s) flagged: ['logsphere_D3', 'logsphere_D3_homo', 'logsphere_D3_nopb', 'sphere_D3_x0lb']**
