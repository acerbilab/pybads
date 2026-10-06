# Compare the default suite at 8afbe16 (REF) and at 97b2c66 (NEW)

- REF, 540 runs: pybads 8afbe16, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 97b2c66, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 148 | 0.0841 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 220 | 0.808 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 193 | 0.428 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 221 | 0.824 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 177 | 0.262 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 164 | 0.164 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 192 | 0.416 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 200 | 0.516 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.433 | 0.00655 | 0.354 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 160 | 0.14 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 221 | 0.824 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 210 | 0.655 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 116 | 0.0155 | 0.819 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 155 | 0.274 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 186 | 0.349 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 198 | 0.49 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 195 | 0.452 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.115 [-0.018, +0.240] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.144 [-0.454, +0.450] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | -0.271 [-0.722, +0.371] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.055 [-0.471, +0.386] | 0.27 | 0.20 | -0.07 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.011 [-0.276, +0.156] | 0.63 | 0.67 | +0.03 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.342 [-1.013, +0.492] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | -0.324 [-0.536, +0.118] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.034 [-0.307, +0.384] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.078 [-0.115, +0.347] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| rastrigin_D3 | 30 | -0.000 [-0.128, +0.000] | 0.07 | 0.00 | -0.07 | 0 | 0 |
| rosenbrock_D2 | 30 | +0.989 [-0.191, +1.355] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.000 [-0.251, +0.709] | 0.73 | 0.77 | +0.03 | 0 | 0 |
| sphere_D10 | 30 | -0.079 [-0.187, +0.080] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.457 [-0.927, -0.032] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.001 [-0.077, +0.211] | 0.50 | 0.50 | +0.00 | 0 | 0 |
| sphere_D3_homo | 30 | +0.065 [-0.036, +0.192] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.046 [-0.417, +0.545] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.090 [-0.106, +0.286] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
