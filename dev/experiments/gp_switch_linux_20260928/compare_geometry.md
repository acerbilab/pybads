# Compare pop (REF) and pop (NEW)

- REF, 210 runs: pybads f8a1cad, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-1893eff/gpyreg at 1893eff
- NEW, 210 runs: pybads f8a1cad, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-1893eff/gpyreg at 1893eff

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| edgesphere_D2 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| edgesphere_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D2 | signed-rank | log10 true_error | 30 | 30 | 14 | 0.00185 | 0.0388 | FLAG |
| edgesphere_D3_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D3_homo | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D4 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| edgesphere_D4 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| edgesphere_D4 | signed-rank | log10 true_error | 30 | 30 | 37 | 0.0196 | 0.392 | ok |
| ridge_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ridge_D4 | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| ridge_D4 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| ridge_D4 | signed-rank | log10 true_error | 30 | 30 | 0 | 0.317 | 1 | ok |
| sphere_band_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_band_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_band_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_band_D3 | signed-rank | log10 true_error | 30 | 30 | 200 | 0.946 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | 30 | +0.000 [-0.924, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D4 | 30 | -0.019 [-0.292, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ridge_D2 | 30 | +0.000 [+0.000, +0.000] | 0.90 | 0.90 | +0.00 | 0 | 0 |
| ridge_D4 | 30 | +0.000 [+0.000, +0.000] | 0.87 | 0.87 | +0.00 | 0 | 0 |
| sphere_band_D2 | 30 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| sphere_band_D3 | 30 | +0.009 [-0.131, +0.178] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 21 tests at alpha 0.05. A flag needs p <= 0.0024 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**1 configuration(s) flagged: ['edgesphere_D2']**
