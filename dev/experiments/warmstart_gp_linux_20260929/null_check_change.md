# Null check: even vs odd seeds of change

- REF, 900 runs: pybads 58d922a1, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | KS | true_error | 45 | 45 | 0.111 | 0.948 | 1 | ok |
| ellipsoid_D3_other | KS | func_count | 45 | 45 | 0.222 | 0.218 | 1 | ok |
| ellipsoid_D3_rerun | KS | true_error | 45 | 45 | 0.111 | 0.948 | 1 | ok |
| ellipsoid_D3_rerun | KS | func_count | 45 | 45 | 0.156 | 0.653 | 1 | ok |
| rosenbrock_D6_other | KS | true_error | 45 | 45 | 0.133 | 0.825 | 1 | ok |
| rosenbrock_D6_other | KS | func_count | 45 | 45 | 0.156 | 0.653 | 1 | ok |
| rosenbrock_D6_rerun | KS | true_error | 45 | 45 | 0.178 | 0.48 | 1 | ok |
| rosenbrock_D6_rerun | KS | func_count | 45 | 45 | 0.111 | 0.948 | 1 | ok |
| sphere_D3_hetero_other | KS | true_error | 45 | 45 | 0.0889 | 0.995 | 1 | ok |
| sphere_D3_hetero_other | KS | func_count | 45 | 45 | 0.111 | 0.948 | 1 | ok |
| sphere_D3_hetero_rerun | KS | true_error | 45 | 45 | 0.111 | 0.948 | 1 | ok |
| sphere_D3_hetero_rerun | KS | func_count | 45 | 45 | 0.0889 | 0.995 | 1 | ok |
| sphere_D3_homo_other | KS | true_error | 45 | 45 | 0.178 | 0.48 | 1 | ok |
| sphere_D3_homo_other | KS | func_count | 45 | 45 | 0.222 | 0.218 | 1 | ok |
| sphere_D3_homo_rerun | KS | true_error | 45 | 45 | 0.178 | 0.48 | 1 | ok |
| sphere_D3_homo_rerun | KS | func_count | 45 | 45 | 0.111 | 0.948 | 1 | ok |
| sphere_D3_other | KS | true_error | 45 | 45 | 0.178 | 0.48 | 1 | ok |
| sphere_D3_other | KS | func_count | 45 | 45 | 0.0667 | 1 | 1 | ok |
| sphere_D3_rerun | KS | true_error | 45 | 45 | 0.178 | 0.48 | 1 | ok |
| sphere_D3_rerun | KS | func_count | 45 | 45 | 0.156 | 0.653 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_rerun | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6_other | - | - | 0.89 | 0.80 | -0.09 | 0 | 0 |
| rosenbrock_D6_rerun | - | - | 0.76 | 0.82 | +0.07 | 0 | 0 |
| sphere_D3_hetero_other | - | - | 0.58 | 0.60 | +0.02 | 0 | 0 |
| sphere_D3_hetero_rerun | - | - | 0.64 | 0.56 | -0.09 | 0 | 0 |
| sphere_D3_homo_other | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_homo_rerun | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_other | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_rerun | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 20 tests at alpha 0.05. A flag needs p <= 0.0025 for the first step; for 45 vs 45 runs that is a KS statistic of at least 0.400.

**no configuration flagged (20 tests)**
