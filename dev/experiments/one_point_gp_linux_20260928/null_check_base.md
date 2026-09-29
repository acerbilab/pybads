# Null check: even vs odd seeds of base

- REF, 330 runs: pybads 07280f79, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| edgesphere_D2 | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| edgesphere_D2 | KS | func_count | 15 | 15 | 0.0667 | 1 | 1 | ok |
| edgesphere_D3_homo | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| edgesphere_D3_homo | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| edgesphere_D4 | KS | true_error | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| edgesphere_D4 | KS | func_count | 15 | 15 | 0.0667 | 1 | 1 | ok |
| ridge_D2 | KS | true_error | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| ridge_D2 | KS | func_count | 15 | 15 | 0.533 | 0.0262 | 0.577 | ok |
| ridge_D4 | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| ridge_D4 | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_band_D2 | KS | true_error | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| sphere_band_D2 | KS | func_count | 15 | 15 | 0 | 1 | 1 | ok |
| sphere_band_D2_hetero | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| sphere_band_D2_hetero | KS | func_count | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| sphere_band_D2_homo | KS | true_error | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| sphere_band_D2_homo | KS | func_count | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| sphere_band_D3 | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_band_D3 | KS | func_count | 15 | 15 | 0.133 | 1 | 1 | ok |
| sphere_band_D3_hetero | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| sphere_band_D3_hetero | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_band_D3_homo | KS | true_error | 15 | 15 | 0.533 | 0.0262 | 0.577 | ok |
| sphere_band_D3_homo | KS | func_count | 15 | 15 | 0.333 | 0.386 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D3_homo | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D4 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ridge_D2 | - | - | 0.93 | 0.87 | -0.07 | 0 | 0 |
| ridge_D4 | - | - | 0.87 | 0.87 | +0.00 | 0 | 0 |
| sphere_band_D2 | - | - | 0.00 | 0.00 | +0.00 | 0 | 0 |
| sphere_band_D2_hetero | - | - | 0.00 | 0.00 | +0.00 | 0 | 0 |
| sphere_band_D2_homo | - | - | 0.00 | 0.07 | +0.07 | 0 | 0 |
| sphere_band_D3 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_band_D3_hetero | - | - | 0.47 | 0.20 | -0.27 | 0 | 0 |
| sphere_band_D3_homo | - | - | 0.93 | 0.80 | -0.13 | 0 | 0 |

Holm family: 22 tests at alpha 0.05. A flag needs p <= 0.0023 for the first step; for 15 vs 15 runs that is a KS statistic of at least 0.667.

**no configuration flagged (22 tests)**
