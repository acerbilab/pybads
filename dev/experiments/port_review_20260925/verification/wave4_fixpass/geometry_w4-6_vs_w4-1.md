# Compare geo_w41_efe5e95 (REF) and geo_w46_46af65a (NEW)

- REF, 210 runs: pybads efe5e95, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 210 runs: pybads 46af65a, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| edgesphere_D2 | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 0.692 | ok |
| edgesphere_D2 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| edgesphere_D2 | signed-rank | log10 true_error | 30 | 30 | 89 | 0.028 | 0.588 | ok |
| edgesphere_D3_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D3_homo | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D4 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| edgesphere_D4 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| edgesphere_D4 | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |
| ridge_D2 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ridge_D2 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ridge_D2 | signed-rank | log10 true_error | 30 | 30 | 195 | 0.452 | 1 | ok |
| ridge_D4 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ridge_D4 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ridge_D4 | signed-rank | log10 true_error | 30 | 30 | 217 | 0.761 | 1 | ok |
| sphere_band_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_band_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_band_D3 | signed-rank | log10 true_error | 30 | 30 | 221 | 0.824 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | 30 | -0.400 [-1.432, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D4 | 30 | -0.033 [-0.303, +0.321] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ridge_D2 | 30 | -0.045 [-0.434, +0.265] | 0.83 | 0.90 | +0.07 | 0 | 0 |
| ridge_D4 | 30 | +0.084 [-0.262, +0.422] | 0.87 | 0.87 | +0.00 | 0 | 0 |
| sphere_band_D2 | 30 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| sphere_band_D3 | 30 | -0.050 [-0.144, +0.170] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 21 tests at alpha 0.05. A flag needs p <= 0.0024 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (21 tests)**
