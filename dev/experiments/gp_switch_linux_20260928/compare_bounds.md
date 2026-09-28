# Compare pop (REF) and pop (NEW)

- REF, 150 runs: pybads f8a1cad, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-1893eff/gpyreg at 1893eff
- NEW, 150 runs: pybads f8a1cad, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-1893eff/gpyreg at 1893eff

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| logsphere_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| logsphere_D3 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| logsphere_D3 | signed-rank | log10 true_error | 30 | 30 | 25 | 0.477 | 1 | ok |
| logsphere_D3_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| logsphere_D3_homo | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| logsphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| logsphere_D3_nopb | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| logsphere_D3_nopb | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| logsphere_D3_nopb | signed-rank | log10 true_error | 30 | 30 | 67 | 0.42 | 1 | ok |
| sphere_D3_nopb | KS | true_error | 30 | 30 | 0.7 | 2.5e-07 | 3.75e-06 | FLAG |
| sphere_D3_nopb | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_nopb | signed-rank | log10 true_error | 30 | 30 | 41 | 1.82e-05 | 0.000237 | FLAG |
| sphere_D3_x0lb | KS | true_error | 30 | 30 | 0.567 | 8.74e-05 | 0.00105 | FLAG |
| sphere_D3_x0lb | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_x0lb | signed-rank | log10 true_error | 30 | 30 | 28 | 2.76e-06 | 3.87e-05 | FLAG |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| logsphere_D3 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| logsphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 0.87 | 0.87 | +0.00 | 0 | 0 |
| logsphere_D3_nopb | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_nopb | 30 | -1.107 [-1.385, -0.561] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_x0lb | 30 | -0.827 [-1.084, -0.447] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**2 configuration(s) flagged: ['sphere_D3_nopb', 'sphere_D3_x0lb']**
