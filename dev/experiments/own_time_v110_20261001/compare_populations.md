# Compare population_gpyreg133_20260924 (REF) and population_gpyreg140_20260930 (NEW)

- REF, 540 runs: pybads 2059506, gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4
- NEW, 2400 runs: pybads bef26ec2, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\main_3e56dce\gpyreg at 3e56dce

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 100 | 0.557 | 3.86e-07 | 1.74e-05 | FLAG |
| ackley_D6 | KS | func_count | 30 | 100 | 0.3 | 0.0251 | 0.703 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 58 | 0.000137 | 0.00536 | FLAG |
| ellipsoid_D10 | KS | true_error | 30 | 100 | 0.99 | 2.37e-28 | 1.28e-26 | FLAG |
| ellipsoid_D10 | KS | func_count | 30 | 100 | 0.417 | 0.000416 | 0.0158 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 1 | 3.73e-09 | 1.75e-07 | FLAG |
| ellipsoid_D3 | KS | true_error | 30 | 100 | 0.683 | 6.7e-11 | 3.48e-09 | FLAG |
| ellipsoid_D3 | KS | func_count | 30 | 100 | 0.247 | 0.102 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 46 | 3.45e-05 | 0.00145 | FLAG |
| ellipsoid_D3_hetero | KS | true_error | 30 | 100 | 0.173 | 0.445 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 100 | 0.337 | 0.00808 | 0.259 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 100 | 0.197 | 0.296 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 100 | 0.68 | 8.69e-11 | 4.43e-09 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 221 | 0.824 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 100 | 0.617 | 8.75e-09 | 4.02e-07 | FLAG |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 100 | 0.557 | 3.86e-07 | 1.74e-05 | FLAG |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 32 | 5.14e-06 | 0.000221 | FLAG |
| ellipsoid_D6 | KS | true_error | 30 | 100 | 0.92 | 4.04e-22 | 2.14e-20 | FLAG |
| ellipsoid_D6 | KS | func_count | 30 | 100 | 0.667 | 2.43e-10 | 1.22e-08 | FLAG |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 0 | 1.86e-09 | 8.94e-08 | FLAG |
| multisensory_s1_D6 | KS | true_error | 30 | 100 | 0.13 | 0.785 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 100 | 0.383 | 0.00156 | 0.0576 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 212 | 0.685 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 100 | 0.163 | 0.519 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 100 | 0.33 | 0.01 | 0.308 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 185 | 0.339 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 100 | 0.133 | 0.759 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 100 | 0.123 | 0.833 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 180 | 0.289 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 100 | 0.46 | 6.19e-05 | 0.00247 | FLAG |
| rosenbrock_D2 | KS | func_count | 30 | 100 | 0.12 | 0.856 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 90 | 0.00256 | 0.0922 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 100 | 0.653 | 6.58e-10 | 3.23e-08 | FLAG |
| rosenbrock_D6 | KS | func_count | 30 | 100 | 0.283 | 0.0401 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 50 | 5.59e-05 | 0.00229 | FLAG |
| sphere_D10 | KS | true_error | 30 | 100 | 0.157 | 0.572 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 100 | 0.0633 | 1 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 215 | 0.73 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 100 | 0.177 | 0.421 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 100 | 0.173 | 0.445 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 167 | 0.184 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 100 | 0.347 | 0.00579 | 0.197 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 100 | 0.343 | 0.00648 | 0.214 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 109 | 0.00993 | 0.308 | ok |
| sphere_D3_homo | KS | true_error | 30 | 100 | 0.303 | 0.0228 | 0.66 | ok |
| sphere_D3_homo | KS | func_count | 30 | 100 | 0.353 | 0.00461 | 0.161 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 182 | 0.309 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 100 | 0.227 | 0.161 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 100 | 0.12 | 0.856 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 213 | 0.7 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 100 | 0.193 | 0.315 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 100 | 0.16 | 0.545 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 151 | 0.0961 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | -0.382 [-0.458, -0.145] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -2.274 [-2.578, -1.962] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | -0.981 [-1.696, -0.552] | 0.93 | 1.00 | +0.07 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.033 [-0.226, +0.307] | 0.20 | 0.16 | -0.04 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.014 [-0.448, +0.113] | 0.47 | 0.55 | +0.08 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -1.046 [-2.293, -0.558] | 0.83 | 1.00 | +0.17 | 0 | 0 |
| ellipsoid_D6 | 30 | -2.586 [-3.024, -1.749] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | -0.008 [-0.381, +0.265] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.124 [-0.023, +0.222] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| rastrigin_D3 | 30 | -0.040 [-0.317, +0.048] | 0.07 | 0.04 | -0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.830 [-1.309, -0.459] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.911 [-1.387, -0.684] | 0.77 | 0.81 | +0.04 | 0 | 0 |
| sphere_D10 | 30 | -0.069 [-0.219, +0.166] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.043 [-0.288, +0.593] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.206 [-0.593, -0.130] | 0.40 | 0.49 | +0.09 | 0 | 0 |
| sphere_D3_homo | 30 | +0.057 [-0.199, +0.536] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.076 [-0.479, +0.310] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.133 [-0.522, +0.006] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Only in REF: []; only in NEW: ['periodic_D2', 'periodic_D3_hetero', 'periodic_D3_homo', 'periodic_D4', 'periodic_D6', 'periodic_rosenbrock_D4'].

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 100 runs that is a KS statistic of at least 0.400.

**8 configuration(s) flagged: ['ackley_D6', 'ellipsoid_D10', 'ellipsoid_D3', 'ellipsoid_D3_homo', 'ellipsoid_D3_unbounded', 'ellipsoid_D6', 'rosenbrock_D2', 'rosenbrock_D6']**
