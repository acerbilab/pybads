# Compare population_linux_wave2_20260926 (REF) and g6_w324_869a033 (NEW)

- REF, 540 runs: pybads 8510ca8, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 869a033, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.4 | 0.0156 | 0.735 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 92 | 0.00299 | 0.143 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 217 | 0.761 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 196 | 0.465 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.735 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 157 | 0.124 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 179 | 0.28 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.5 | 0.0009 | 0.045 | FLAG |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 208 | 0.626 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.567 | 8.74e-05 | 0.00472 | FLAG |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 77 | 0.000872 | 0.0445 | FLAG |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.4 | 0.0156 | 0.735 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.735 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 61 | 0.000189 | 0.01 | FLAG |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 194 | 0.44 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 210 | 0.655 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 200 | 0.516 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 220 | 0.808 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 87 | 0.00202 | 0.099 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 146 | 0.0767 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 147 | 0.0803 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 223 | 0.855 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.533 | 0.000293 | 0.0153 | FLAG |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 143 | 0.0667 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 135 | 0.0449 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.112 [+0.055, +0.234] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | +0.102 [-0.117, +0.341] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.140 [-0.304, +0.738] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.161 [-0.068, +0.460] | 0.13 | 0.10 | -0.03 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.025 [-0.132, +0.398] | 0.67 | 0.57 | -0.10 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | +0.028 [-0.828, +0.362] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.788 [+0.309, +1.114] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.572 [+0.230, +0.791] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.150 [-0.177, +0.291] | 0.90 | 0.93 | +0.03 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [-0.040, +0.301] | 0.03 | 0.03 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.216 [-0.578, +0.256] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.092 [-0.537, +0.547] | 0.73 | 0.80 | +0.07 | 0 | 0 |
| sphere_D10 | 30 | +0.208 [+0.132, +0.304] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.152 [-0.425, +0.110] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.109 [-0.335, +0.063] | 0.37 | 0.60 | +0.23 | 0 | 0 |
| sphere_D3_homo | 30 | -0.015 [-0.202, +0.220] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.198 [-0.063, +0.755] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.248 [+0.098, +0.432] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**4 configuration(s) flagged: ['ellipsoid_D3_unbounded', 'ellipsoid_D6', 'multisensory_s1_D6', 'sphere_nonbox_D3']**
