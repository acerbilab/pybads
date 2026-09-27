# Compare geo_w329_0b7add3 (REF) and geo_w324_869a033 (NEW)

- REF, 210 runs: pybads 0b7add3, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 210 runs: pybads 869a033, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| edgesphere_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| edgesphere_D2 | KS | func_count | 30 | 30 | 0.767 | 6.53e-09 | 1.31e-07 | FLAG |
| edgesphere_D2 | signed-rank | log10 true_error | 30 | 30 | 199 | 0.927 | 1 | ok |
| edgesphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| edgesphere_D3_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| edgesphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 226 | 0.903 | 1 | ok |
| edgesphere_D4 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| edgesphere_D4 | KS | func_count | 30 | 30 | 0.933 | 2.99e-14 | 6.29e-13 | FLAG |
| edgesphere_D4 | signed-rank | log10 true_error | 30 | 30 | 207 | 0.612 | 1 | ok |
| ridge_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ridge_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ridge_D2 | signed-rank | log10 true_error | 30 | 30 | 184 | 0.328 | 1 | ok |
| ridge_D4 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ridge_D4 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ridge_D4 | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |
| sphere_band_D2 | KS | true_error | 30 | 28 | 0.0976 | 0.994 | 1 | ok |
| sphere_band_D2 | KS | func_count | 30 | 28 | 0.143 | 0.88 | 1 | ok |
| sphere_band_D2 | signed-rank | log10 true_error | 28 | 28 | 0 | 0.18 | 1 | ok |
| sphere_band_D3 | KS | true_error | 30 | 29 | 0.285 | 0.15 | 1 | ok |
| sphere_band_D3 | KS | func_count | 30 | 29 | 0.489 | 0.00119 | 0.0225 | FLAG |
| sphere_band_D3 | signed-rank | log10 true_error | 29 | 29 | 118 | 0.0308 | 0.554 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | 30 | +0.120 [-0.794, +0.856] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| edgesphere_D3_homo | 30 | +0.011 [-0.651, +0.522] | 0.97 | 0.93 | -0.03 | 0 | 0 |
| edgesphere_D4 | 30 | -0.011 [-0.308, +0.142] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ridge_D2 | 30 | +0.134 [-0.509, +0.830] | 0.83 | 0.73 | -0.10 | 0 | 0 |
| ridge_D4 | 30 | +0.007 [-0.267, +0.178] | 0.87 | 0.83 | -0.03 | 0 | 0 |
| sphere_band_D2 | 28 | +0.000 [+0.000, +0.000] | 0.00 | 0.07 | +0.07 | 0 | 2 |
| sphere_band_D3 | 29 | +0.176 [-0.085, +0.469] | 1.00 | 0.77 | -0.23 | 0 | 1 |

Holm family: 21 tests at alpha 0.05. A flag needs p <= 0.0024 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

Crash count rising from zero: ['sphere_band_D2', 'sphere_band_D3'].

**4 configuration(s) flagged: ['edgesphere_D2', 'edgesphere_D4', 'sphere_band_D2', 'sphere_band_D3']**
