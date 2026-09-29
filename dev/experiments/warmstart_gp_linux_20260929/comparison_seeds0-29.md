# Compare base (REF) and change (NEW)

- REF, 300 runs: pybads ee0d9c29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4
- NEW, 300 runs: pybads 58d922a1, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_other | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_other | signed-rank | log10 true_error | 30 | 30 | 199 | 0.503 | 1 | ok |
| ellipsoid_D3_rerun | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_rerun | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_rerun | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 1 | ok |
| rosenbrock_D6_other | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6_other | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6_other | signed-rank | log10 true_error | 30 | 30 | 147 | 0.0803 | 1 | ok |
| rosenbrock_D6_rerun | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D6_rerun | KS | func_count | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| rosenbrock_D6_rerun | signed-rank | log10 true_error | 30 | 30 | 230 | 0.968 | 1 | ok |
| sphere_D3_hetero_other | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero_other | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D3_hetero_other | signed-rank | log10 true_error | 30 | 30 | 203 | 0.556 | 1 | ok |
| sphere_D3_hetero_rerun | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_hetero_rerun | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_hetero_rerun | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |
| sphere_D3_homo_other | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo_other | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo_other | signed-rank | log10 true_error | 30 | 30 | 221 | 0.824 | 1 | ok |
| sphere_D3_homo_rerun | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_homo_rerun | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D3_homo_rerun | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |
| sphere_D3_other | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_other | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D3_other | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 1 | ok |
| sphere_D3_rerun | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_rerun | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D3_rerun | signed-rank | log10 true_error | 30 | 30 | 187 | 0.36 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | 30 | -0.331 [-0.771, +0.205] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_rerun | 30 | -0.069 [-0.376, +0.402] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6_other | 30 | +0.122 [-0.024, +0.569] | 0.93 | 0.77 | -0.17 | 0 | 0 |
| rosenbrock_D6_rerun | 30 | -0.029 [-0.421, +0.231] | 0.80 | 0.80 | +0.00 | 0 | 0 |
| sphere_D3_hetero_other | 30 | -0.018 [-0.094, +0.275] | 0.73 | 0.57 | -0.17 | 0 | 0 |
| sphere_D3_hetero_rerun | 30 | +0.056 [-0.165, +0.276] | 0.60 | 0.50 | -0.10 | 0 | 0 |
| sphere_D3_homo_other | 30 | -0.023 [-0.215, +0.190] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_homo_rerun | 30 | -0.012 [-0.264, +0.419] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_other | 30 | +0.134 [-0.313, +0.580] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_rerun | 30 | -0.258 [-0.496, +0.156] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 30 tests at alpha 0.05. A flag needs p <= 0.0017 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (30 tests)**
