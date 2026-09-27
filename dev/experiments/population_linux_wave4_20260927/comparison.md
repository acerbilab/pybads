# Compare population_linux_wave3_20260927 (REF) and w46_46af65a (NEW)

- REF, 540 runs: pybads a14524d, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 46af65a, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.813 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 202 | 0.543 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 168 | 0.191 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 153 | 0.105 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.4 | 0.0156 | 0.813 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 114 | 0.0137 | 0.724 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 149 | 0.0879 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 164 | 0.164 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 206 | 0.598 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 174 | 0.237 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 211 | 0.67 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 205 | 0.584 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.433 | 0.00655 | 0.354 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 208 | 0.626 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 194 | 0.44 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 191 | 0.404 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 123 | 0.0234 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 190 | 0.393 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 177 | 0.262 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 203 | 0.556 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.012 [-0.154, +0.104] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | +0.114 [-0.192, +0.364] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | -0.369 [-0.749, +0.020] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.231 [-0.541, -0.067] | 0.10 | 0.23 | +0.13 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.387 [-0.097, +0.860] | 0.73 | 0.43 | -0.30 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.337 [-0.922, +0.205] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | -0.015 [-0.218, +0.199] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | -0.155 [-0.365, +0.256] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.074 [-0.039, +0.172] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rastrigin_D3 | 30 | -0.000 [-0.218, +0.111] | 0.03 | 0.07 | +0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.001 [-0.915, +0.619] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | +0.283 [-0.321, +0.716] | 0.77 | 0.83 | +0.07 | 0 | 0 |
| sphere_D10 | 30 | -0.019 [-0.172, +0.101] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.176 [-0.127, +0.407] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.208 [-0.490, +0.002] | 0.40 | 0.60 | +0.20 | 0 | 0 |
| sphere_D3_homo | 30 | -0.105 [-0.213, +0.144] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.299 [-0.547, +0.206] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.083 [-0.236, +0.384] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
