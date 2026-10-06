# Compare population_gpyreg133_20260924 (REF) and population_linux_20260925 (NEW)

- REF, 540 runs: pybads 2059506, gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4
- NEW, 540 runs: pybads 676083d, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 219 | 0.792 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 181 | 0.299 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 219 | 0.792 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 232 | 1 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 159 | 0.135 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 216 | 0.746 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 206 | 0.598 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 224 | 0.871 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 223 | 0.855 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 213 | 0.7 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 228 | 0.935 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 201 | 0.529 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 169 | 0.198 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 210 | 0.655 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 231 | 0.984 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 197 | 0.658 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 177 | 0.262 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 188 | 0.371 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.006 [-0.150, +0.188] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | +0.066 [-0.276, +0.660] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| ellipsoid_D3 | 30 | -0.057 [-0.810, +0.674] | 0.93 | 0.90 | -0.03 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.004 [-0.215, +0.186] | 0.20 | 0.20 | +0.00 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.280 [-0.611, +0.155] | 0.47 | 0.60 | +0.13 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.087 [-0.773, +0.260] | 0.83 | 0.90 | +0.07 | 0 | 0 |
| ellipsoid_D6 | 30 | -0.179 [-0.653, +0.315] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.077 [-0.223, +0.383] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | -0.037 [-0.140, +0.260] | 1.00 | 0.90 | -0.10 | 0 | 0 |
| rastrigin_D3 | 30 | -0.000 [-0.101, +0.000] | 0.07 | 0.03 | -0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.114 [-0.410, +0.600] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.126 [-0.691, +0.443] | 0.77 | 0.83 | +0.07 | 0 | 0 |
| sphere_D10 | 30 | -0.069 [-0.643, +0.292] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.138 [-0.487, +0.314] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.052 [-0.290, +0.161] | 0.40 | 0.23 | -0.17 | 0 | 0 |
| sphere_D3_homo | 30 | -0.030 [-0.135, +0.087] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.025 [-0.337, +0.698] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.110 [-0.356, +0.019] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
