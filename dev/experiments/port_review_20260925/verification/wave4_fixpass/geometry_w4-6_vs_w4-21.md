# Compare geo_w421_86512c9 (REF) and geo_w46_46af65a (NEW)

- REF, 210 runs: pybads 86512c9, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 210 runs: pybads 46af65a, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| edgesphere_D2 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| edgesphere_D2 | KS | func_count | 30 | 30 | 0.6 | 2.37e-05 | 0.000497 | FLAG |
| edgesphere_D2 | signed-rank | log10 true_error | 30 | 30 | 118 | 0.147 | 1 | ok |
| edgesphere_D3_homo | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| edgesphere_D3_homo | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| edgesphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 100 | 0.00538 | 0.108 | ok |
| edgesphere_D4 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| edgesphere_D4 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| edgesphere_D4 | signed-rank | log10 true_error | 30 | 30 | 220 | 0.808 | 1 | ok |
| ridge_D2 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ridge_D2 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ridge_D2 | signed-rank | log10 true_error | 30 | 30 | 184 | 0.328 | 1 | ok |
| ridge_D4 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ridge_D4 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ridge_D4 | signed-rank | log10 true_error | 30 | 30 | 162 | 0.152 | 1 | ok |
| sphere_band_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_band_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_band_D3 | signed-rank | log10 true_error | 30 | 30 | 130 | 0.0345 | 0.656 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | 30 | -0.130 [-0.762, +0.120] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D3_homo | 30 | -0.488 [-1.071, -0.019] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| edgesphere_D4 | 30 | -0.135 [-0.329, +0.273] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ridge_D2 | 30 | -0.110 [-0.766, +0.290] | 0.77 | 0.90 | +0.13 | 0 | 0 |
| ridge_D4 | 30 | -0.206 [-0.282, -0.064] | 0.87 | 0.87 | +0.00 | 0 | 0 |
| sphere_band_D2 | 30 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| sphere_band_D3 | 30 | +0.194 [+0.082, +0.274] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 21 tests at alpha 0.05. A flag needs p <= 0.0024 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**1 configuration(s) flagged: ['edgesphere_D2']**
