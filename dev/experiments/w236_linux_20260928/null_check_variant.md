# Null check: even vs odd seeds of variant

- REF, 450 runs: pybads 9a7c361e, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 45 | 45 | 0.356 | 0.00638 | 0.0638 | ok |
| ellipsoid_D3_hetero | KS | func_count | 45 | 45 | 0.222 | 0.218 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 45 | 45 | 0.111 | 0.948 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 45 | 45 | 0.133 | 0.825 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 45 | 45 | 0.133 | 0.825 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 45 | 45 | 0.2 | 0.332 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 45 | 45 | 0.156 | 0.653 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 45 | 45 | 0.178 | 0.48 | 1 | ok |
| sphere_D3_homo | KS | true_error | 45 | 45 | 0.0889 | 0.995 | 1 | ok |
| sphere_D3_homo | KS | func_count | 45 | 45 | 0.2 | 0.332 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | - | - | 0.18 | 0.11 | -0.07 | 0 | 0 |
| ellipsoid_D3_homo | - | - | 0.62 | 0.58 | -0.04 | 0 | 0 |
| multisensory_s1_D6_homo | - | - | 0.96 | 0.98 | +0.02 | 0 | 0 |
| sphere_D3_hetero | - | - | 0.60 | 0.62 | +0.02 | 0 | 0 |
| sphere_D3_homo | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 10 tests at alpha 0.05. A flag needs p <= 0.005 for the first step; for 45 vs 45 runs that is a KS statistic of at least 0.378.

**no configuration flagged (10 tests)**
