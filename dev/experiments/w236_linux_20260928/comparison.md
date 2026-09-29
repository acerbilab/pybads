# Compare head (REF) and variant (NEW)

- REF, 450 runs: pybads 58e7dd5, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 450 runs: pybads 9a7c361e, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 90 | 90 | 0.0111 | 1 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 90 | 90 | 0.0111 | 1 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 90 | 90 | 0 | 0.317 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 90 | 90 | 0.0111 | 1 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 90 | 90 | 0.0111 | 1 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 90 | 90 | 0 | 0.317 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 90 | 90 | 0 | 1 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 90 | 90 | 0 | 1 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 90 | 90 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 90 | 90 | 0.144 | 0.306 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 90 | 90 | 0.0889 | 0.872 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 90 | 90 | 85 | 0.0125 | 0.187 | ok |
| sphere_D3_homo | KS | true_error | 90 | 90 | 0.0111 | 1 | 1 | ok |
| sphere_D3_homo | KS | func_count | 90 | 90 | 0.0444 | 1 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 90 | 90 | 23 | 0.646 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 90 | +0.000 [+0.000, +0.000] | 0.16 | 0.14 | -0.01 | 0 | 0 |
| ellipsoid_D3_homo | 90 | +0.000 [+0.000, +0.000] | 0.61 | 0.60 | -0.01 | 0 | 0 |
| multisensory_s1_D6_homo | 90 | +0.000 [+0.000, +0.000] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 90 | +0.000 [+0.000, +0.000] | 0.50 | 0.61 | +0.11 | 0 | 0 |
| sphere_D3_homo | 90 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 90 vs 90 runs that is a KS statistic of at least 0.267.

**no configuration flagged (15 tests)**
