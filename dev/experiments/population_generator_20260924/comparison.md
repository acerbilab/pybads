# Compare population_baseline_20260924 (REF) and population_generator_20260924 (NEW)

- REF, 540 runs: pybads 2226883, gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.1\gpyreg at 1dbbfc5
- NEW, 540 runs: pybads c85cddb, gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.1\gpyreg at 1dbbfc5

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 194 | 0.44 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 29 | 28 | 0.225 | 0.395 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 29 | 28 | 0.151 | 0.838 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 27 | 27 | 163 | 0.546 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 29 | 30 | 0.218 | 0.411 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 29 | 30 | 0.254 | 0.243 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 29 | 29 | 217 | 1 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 151 | 0.0961 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 176 | 0.253 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 217 | 0.761 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 174 | 0.237 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 187 | 0.36 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 190 | 0.393 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 166 | 0.177 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 207 | 0.612 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 165 | 0.171 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 211 | 0.67 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 188 | 0.371 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 191 | 0.404 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | -0.055 [-0.325, +0.128] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 27 | -0.138 [-0.398, +0.303] | 0.97 | 0.93 | -0.03 | 1 | 2 |
| ellipsoid_D3 | 29 | +0.095 [-0.763, +0.540] | 0.87 | 0.90 | +0.03 | 1 | 0 |
| ellipsoid_D3_hetero | 30 | +0.207 [+0.005, +0.383] | 0.23 | 0.20 | -0.03 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.262 [+0.061, +0.407] | 0.63 | 0.50 | -0.13 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.348 [-0.950, +0.319] | 0.93 | 0.83 | -0.10 | 0 | 0 |
| ellipsoid_D6 | 30 | -0.034 [-0.422, +0.306] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.323 [-0.101, +0.895] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.003 [-0.270, +0.079] | 0.93 | 1.00 | +0.07 | 0 | 0 |
| rastrigin_D3 | 30 | -0.010 [-0.301, +0.079] | 0.03 | 0.07 | +0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | +0.351 [-0.205, +1.131] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | +0.256 [-0.927, +1.036] | 0.73 | 0.63 | -0.10 | 0 | 0 |
| sphere_D10 | 30 | -0.004 [-0.134, +0.238] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.057 [-1.013, +0.280] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.164 [-0.283, +0.216] | 0.23 | 0.40 | +0.17 | 0 | 0 |
| sphere_D3_homo | 30 | -0.045 [-0.276, +0.186] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.168 [-0.656, +0.149] | 0.90 | 0.90 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.366 [-1.038, +0.683] | 0.80 | 0.97 | +0.17 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
