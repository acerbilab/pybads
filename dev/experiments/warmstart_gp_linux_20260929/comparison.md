# Compare base (REF) and change (NEW)

- REF, 900 runs: pybads ee0d9c29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4
- NEW, 900 runs: pybads 58d922a1, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | KS | true_error | 90 | 90 | 0.0889 | 0.872 | 1 | ok |
| ellipsoid_D3_other | KS | func_count | 90 | 90 | 0.0778 | 0.95 | 1 | ok |
| ellipsoid_D3_other | signed-rank | log10 true_error | 90 | 90 | 1.97e+03 | 0.761 | 1 | ok |
| ellipsoid_D3_rerun | KS | true_error | 90 | 90 | 0.0889 | 0.872 | 1 | ok |
| ellipsoid_D3_rerun | KS | func_count | 90 | 90 | 0.122 | 0.515 | 1 | ok |
| ellipsoid_D3_rerun | signed-rank | log10 true_error | 90 | 90 | 2.04e+03 | 0.97 | 1 | ok |
| rosenbrock_D6_other | KS | true_error | 90 | 90 | 0.0667 | 0.989 | 1 | ok |
| rosenbrock_D6_other | KS | func_count | 90 | 90 | 0.133 | 0.402 | 1 | ok |
| rosenbrock_D6_other | signed-rank | log10 true_error | 90 | 90 | 1.94e+03 | 0.68 | 1 | ok |
| rosenbrock_D6_rerun | KS | true_error | 90 | 90 | 0.0778 | 0.95 | 1 | ok |
| rosenbrock_D6_rerun | KS | func_count | 90 | 90 | 0.3 | 0.000561 | 0.0168 | FLAG |
| rosenbrock_D6_rerun | signed-rank | log10 true_error | 90 | 90 | 2.04e+03 | 0.973 | 1 | ok |
| sphere_D3_hetero_other | KS | true_error | 90 | 90 | 0.111 | 0.638 | 1 | ok |
| sphere_D3_hetero_other | KS | func_count | 90 | 90 | 0.111 | 0.638 | 1 | ok |
| sphere_D3_hetero_other | signed-rank | log10 true_error | 90 | 90 | 2.02e+03 | 0.899 | 1 | ok |
| sphere_D3_hetero_rerun | KS | true_error | 90 | 90 | 0.133 | 0.402 | 1 | ok |
| sphere_D3_hetero_rerun | KS | func_count | 90 | 90 | 0.133 | 0.402 | 1 | ok |
| sphere_D3_hetero_rerun | signed-rank | log10 true_error | 90 | 90 | 1.83e+03 | 0.381 | 1 | ok |
| sphere_D3_homo_other | KS | true_error | 90 | 90 | 0.111 | 0.638 | 1 | ok |
| sphere_D3_homo_other | KS | func_count | 90 | 90 | 0.0556 | 0.999 | 1 | ok |
| sphere_D3_homo_other | signed-rank | log10 true_error | 90 | 90 | 1.85e+03 | 0.42 | 1 | ok |
| sphere_D3_homo_rerun | KS | true_error | 90 | 90 | 0.1 | 0.762 | 1 | ok |
| sphere_D3_homo_rerun | KS | func_count | 90 | 90 | 0.133 | 0.402 | 1 | ok |
| sphere_D3_homo_rerun | signed-rank | log10 true_error | 90 | 90 | 1.93e+03 | 0.625 | 1 | ok |
| sphere_D3_other | KS | true_error | 90 | 90 | 0.1 | 0.762 | 1 | ok |
| sphere_D3_other | KS | func_count | 90 | 90 | 0.0889 | 0.872 | 1 | ok |
| sphere_D3_other | signed-rank | log10 true_error | 90 | 90 | 2.02e+03 | 0.918 | 1 | ok |
| sphere_D3_rerun | KS | true_error | 90 | 90 | 0.0889 | 0.872 | 1 | ok |
| sphere_D3_rerun | KS | func_count | 90 | 90 | 0.0889 | 0.872 | 1 | ok |
| sphere_D3_rerun | signed-rank | log10 true_error | 90 | 90 | 1.99e+03 | 0.817 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | 90 | +0.080 [-0.313, +0.303] | 0.99 | 1.00 | +0.01 | 0 | 0 |
| ellipsoid_D3_rerun | 90 | -0.036 [-0.125, +0.237] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6_other | 90 | +0.018 [-0.140, +0.212] | 0.87 | 0.84 | -0.02 | 0 | 0 |
| rosenbrock_D6_rerun | 90 | -0.050 [-0.219, +0.111] | 0.83 | 0.79 | -0.04 | 0 | 0 |
| sphere_D3_hetero_other | 90 | +0.021 [-0.117, +0.146] | 0.64 | 0.59 | -0.06 | 0 | 0 |
| sphere_D3_hetero_rerun | 90 | +0.014 [-0.050, +0.087] | 0.67 | 0.60 | -0.07 | 0 | 0 |
| sphere_D3_homo_other | 90 | -0.057 [-0.181, +0.049] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_homo_rerun | 90 | -0.087 [-0.225, +0.053] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_other | 90 | +0.009 [-0.299, +0.289] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_rerun | 90 | -0.141 [-0.313, +0.209] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 30 tests at alpha 0.05. A flag needs p <= 0.0017 for the first step; for 90 vs 90 runs that is a KS statistic of at least 0.289.

**1 configuration(s) flagged: ['rosenbrock_D6_rerun']**
