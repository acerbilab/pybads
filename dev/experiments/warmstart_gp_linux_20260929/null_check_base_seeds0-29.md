# Null check: even vs odd seeds of base

- REF, 300 runs: pybads ee0d9c29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | KS | true_error | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| ellipsoid_D3_other | KS | func_count | 15 | 15 | 0.133 | 1 | 1 | ok |
| ellipsoid_D3_rerun | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| ellipsoid_D3_rerun | KS | func_count | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| rosenbrock_D6_other | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| rosenbrock_D6_other | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| rosenbrock_D6_rerun | KS | true_error | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| rosenbrock_D6_rerun | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D3_hetero_other | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D3_hetero_other | KS | func_count | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| sphere_D3_hetero_rerun | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D3_hetero_rerun | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D3_homo_other | KS | true_error | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| sphere_D3_homo_other | KS | func_count | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| sphere_D3_homo_rerun | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D3_homo_rerun | KS | func_count | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| sphere_D3_other | KS | true_error | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| sphere_D3_other | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D3_rerun | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D3_rerun | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_rerun | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6_other | - | - | 1.00 | 0.87 | -0.13 | 0 | 0 |
| rosenbrock_D6_rerun | - | - | 0.87 | 0.73 | -0.13 | 0 | 0 |
| sphere_D3_hetero_other | - | - | 0.73 | 0.73 | +0.00 | 0 | 0 |
| sphere_D3_hetero_rerun | - | - | 0.60 | 0.60 | +0.00 | 0 | 0 |
| sphere_D3_homo_other | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_homo_rerun | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_other | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_rerun | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 20 tests at alpha 0.05. A flag needs p <= 0.0025 for the first step; for 15 vs 15 runs that is a KS statistic of at least 0.667.

**no configuration flagged (20 tests)**
