# Compare base (REF) and change (NEW)

- REF, 330 runs: pybads 07280f79, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4
- NEW, 330 runs: pybads d9772a04, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| edgesphere_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D3_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D3_homo | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D4 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D4 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D4 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D4 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D4 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D4 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2_hetero | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_band_D2_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_band_D2_hetero | signed-rank | log10 true_error | 30 | 30 | 97 | 0.765 | 1 | ok |
| sphere_band_D2_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_band_D2_homo | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_band_D2_homo | signed-rank | log10 true_error | 30 | 30 | 120 | 0.0974 | 1 | ok |
| sphere_band_D3 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_band_D3 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_band_D3 | signed-rank | log10 true_error | 30 | 30 | 192 | 0.416 | 1 | ok |
| sphere_band_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_band_D3_hetero | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_band_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 176 | 0.37 | 1 | ok |
| sphere_band_D3_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_band_D3_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_band_D3_homo | signed-rank | log10 true_error | 30 | 30 | 157 | 0.191 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D4 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ridge_D2 | 30 | +0.000 [+0.000, +0.000] | 0.90 | 0.90 | +0.00 | 0 | 0 |
| ridge_D4 | 30 | +0.000 [+0.000, +0.000] | 0.87 | 0.87 | +0.00 | 0 | 0 |
| sphere_band_D2 | 30 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| sphere_band_D2_hetero | 30 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| sphere_band_D2_homo | 30 | -0.000 [-0.005, +0.000] | 0.03 | 0.03 | +0.00 | 0 | 0 |
| sphere_band_D3 | 30 | -0.035 [-0.249, +0.128] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_band_D3_hetero | 30 | +0.137 [-0.253, +0.601] | 0.33 | 0.43 | +0.10 | 0 | 0 |
| sphere_band_D3_homo | 30 | -0.091 [-0.476, +0.157] | 0.87 | 0.90 | +0.03 | 0 | 0 |

Holm family: 33 tests at alpha 0.05. A flag needs p <= 0.0015 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (33 tests)**
