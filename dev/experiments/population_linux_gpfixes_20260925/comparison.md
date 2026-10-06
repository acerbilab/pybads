# Compare population_linux_targetnoise_20260925 (REF) and population_linux_gpfixes_20260925 (NEW)

- REF, 540 runs: pybads 1c8c71d, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 97b2c66, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 174 | 0.237 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.9 | 5.79e-13 | 3.12e-11 | FLAG |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.733 | 4.33e-08 | 2.21e-06 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 3 | 9.31e-09 | 4.84e-07 | FLAG |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.633 | 5.8e-06 | 0.000278 | FLAG |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 8 | 4.66e-08 | 2.33e-06 | FLAG |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 123 | 0.0234 | 0.983 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 209 | 0.641 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.6 | 2.37e-05 | 0.00109 | FLAG |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.5 | 0.0009 | 0.0387 | FLAG |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 48 | 4.41e-05 | 0.00198 | FLAG |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.8 | 8.47e-10 | 4.49e-08 | FLAG |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.567 | 8.74e-05 | 0.00384 | FLAG |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 11 | 1.02e-07 | 5.02e-06 | FLAG |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 138 | 0.0523 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 219 | 0.792 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 201 | 0.529 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.633 | 5.8e-06 | 0.000278 | FLAG |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 141 | 0.0606 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 227 | 0.919 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 189 | 0.382 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 193 | 0.428 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 226 | 0.903 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 211 | 0.67 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.070 [-0.033, +0.207] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -2.043 [-2.300, -1.742] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| ellipsoid_D3 | 30 | -1.291 [-1.866, -0.505] | 0.90 | 1.00 | +0.10 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.273 [-0.461, +0.013] | 0.10 | 0.20 | +0.10 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.105 [-0.302, +0.100] | 0.60 | 0.67 | +0.07 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -1.121 [-1.815, -0.439] | 0.90 | 1.00 | +0.10 | 0 | 0 |
| ellipsoid_D6 | 30 | -2.124 [-2.426, -1.797] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.235 [-0.059, +0.514] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.101 [-0.187, +0.148] | 0.90 | 0.97 | +0.07 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [-0.062, +0.301] | 0.03 | 0.00 | -0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.033 [-0.966, +0.494] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.655 [-1.107, -0.136] | 0.83 | 0.77 | -0.07 | 0 | 0 |
| sphere_D10 | 30 | -0.021 [-0.098, +0.131] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.234 [-0.923, +0.461] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.065 [-0.190, +0.427] | 0.50 | 0.50 | +0.00 | 0 | 0 |
| sphere_D3_homo | 30 | +0.083 [-0.240, +0.279] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.036 [-0.534, +0.301] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.000 [-0.204, +0.363] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**5 configuration(s) flagged: ['ellipsoid_D10', 'ellipsoid_D3', 'ellipsoid_D3_unbounded', 'ellipsoid_D6', 'rosenbrock_D6']**
