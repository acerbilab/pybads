# Compare g0_batch1_a1bf658 (REF) and g1_es_c276d79 (NEW)

- REF, 540 runs: pybads a1bf658, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads c276d79, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 206 | 0.598 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 198 | 0.49 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 225 | 0.887 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.845 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 200 | 0.516 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 191 | 0.404 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 193 | 0.428 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 191 | 0.404 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 179 | 0.28 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 173 | 0.229 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 224 | 0.871 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 221 | 0.824 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 212 | 0.685 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 213 | 0.7 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 191 | 0.404 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 215 | 0.73 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 228 | 0.935 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 160 | 0.14 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 182 | 0.309 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.001 [-0.133, +0.165] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.112 [-0.294, +0.220] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.029 [-0.384, +0.366] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.089 [-0.164, +0.366] | 0.13 | 0.20 | +0.07 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.121 [-0.185, +0.433] | 0.67 | 0.60 | -0.07 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | +0.012 [-0.459, +0.149] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.119 [-0.195, +0.399] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.254 [-0.297, +0.500] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | -0.110 [-0.319, +0.133] | 0.90 | 1.00 | +0.10 | 0 | 0 |
| rastrigin_D3 | 30 | -0.000 [-0.239, +0.239] | 0.03 | 0.03 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.022 [-0.557, +0.334] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.033 [-0.666, +0.448] | 0.73 | 0.80 | +0.07 | 0 | 0 |
| sphere_D10 | 30 | -0.164 [-0.301, +0.118] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.173 [-0.381, +0.223] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.026 [-0.217, +0.146] | 0.37 | 0.47 | +0.10 | 0 | 0 |
| sphere_D3_homo | 30 | -0.039 [-0.169, +0.133] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.200 [-0.038, +0.441] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.287 [-0.474, +0.132] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
