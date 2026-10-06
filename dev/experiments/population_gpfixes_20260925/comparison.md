# Compare population_targetnoise_20260925 (REF) and population_gpfixes_20260925 (NEW)

- REF, 540 runs: pybads c044fea, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4
- NEW, 540 runs: pybads ab4dded, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 214 | 0.715 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.9 | 5.79e-13 | 3.12e-11 | FLAG |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.6 | 2.37e-05 | 0.00116 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 1 | 3.73e-09 | 1.94e-07 | FLAG |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.467 | 0.00253 | 0.111 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 60 | 0.00017 | 0.00782 | FLAG |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 176 | 0.253 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 150 | 0.0919 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.567 | 8.74e-05 | 0.00419 | FLAG |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.673 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 66 | 0.000313 | 0.0141 | FLAG |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.8 | 8.47e-10 | 4.49e-08 | FLAG |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.567 | 8.74e-05 | 0.00419 | FLAG |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 13 | 1.64e-07 | 8.36e-06 | FLAG |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 191 | 0.404 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 184 | 0.328 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 196 | 0.465 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 208 | 0.626 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.667 | 1.28e-06 | 6.38e-05 | FLAG |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 164 | 0.164 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 167 | 0.184 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 155 | 0.114 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 230 | 0.968 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 200 | 0.516 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.070 [-0.150, +0.233] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -1.975 [-2.296, -1.656] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | -1.088 [-1.707, -0.313] | 0.93 | 1.00 | +0.07 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.403 [-0.534, +0.223] | 0.17 | 0.23 | +0.07 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.338 [-0.633, +0.021] | 0.47 | 0.73 | +0.27 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.960 [-1.749, -0.538] | 0.83 | 1.00 | +0.17 | 0 | 0 |
| ellipsoid_D6 | 30 | -2.323 [-2.856, -1.383] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.186 [-0.253, +0.530] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | -0.000 [-0.120, +0.242] | 1.00 | 0.90 | -0.10 | 0 | 0 |
| rastrigin_D3 | 30 | -0.062 [-0.301, +0.066] | 0.07 | 0.00 | -0.07 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.219 [-0.777, +0.427] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -1.132 [-2.103, -0.523] | 0.77 | 0.70 | -0.07 | 0 | 0 |
| sphere_D10 | 30 | -0.167 [-0.616, +0.103] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.199 [-0.639, -0.048] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.002 [-0.206, +0.214] | 0.50 | 0.53 | +0.03 | 0 | 0 |
| sphere_D3_homo | 30 | -0.044 [-0.190, +0.167] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.101 [-0.211, +0.524] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.024 [-0.294, +0.504] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**5 configuration(s) flagged: ['ellipsoid_D10', 'ellipsoid_D3', 'ellipsoid_D3_unbounded', 'ellipsoid_D6', 'rosenbrock_D6']**
