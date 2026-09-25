# Compare population_generator_20260924 (REF) and population_gpyreg133_20260924 (NEW)

- REF, 540 runs: pybads c85cddb, gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.1\gpyreg at 1dbbfc5
- NEW, 540 runs: pybads 2059506, gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 118 | 0.0175 | 0.768 | ok |
| ellipsoid_D10 | KS | true_error | 28 | 30 | 0.193 | 0.569 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 28 | 30 | 0.195 | 0.549 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 28 | 28 | 79 | 0.332 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 10 | 0.917 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 1 | 0.655 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 7 | 0.893 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 38 | 0.937 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.7 | 2.5e-07 | 1.2e-05 | FLAG |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 25 | 1.68e-06 | 7.91e-05 | FLAG |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 0 | 1.86e-09 | 9.5e-08 | FLAG |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 0.00335 | 0.151 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 37 | 0.000157 | 0.00721 | FLAG |
| sphere_D10 | KS | true_error | 30 | 30 | 0.9 | 5.79e-13 | 3.07e-11 | FLAG |
| sphere_D10 | KS | func_count | 30 | 30 | 0.833 | 9.24e-11 | 4.8e-09 | FLAG |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 0 | 1.86e-09 | 9.5e-08 | FLAG |
| sphere_D2 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 62 | 0.492 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 91 | 0.0319 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.933 | 2.99e-14 | 1.62e-12 | FLAG |
| timing_D5 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 0 | 1.86e-09 | 9.5e-08 | FLAG |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.097 [+0.007, +0.302] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 28 | +0.000 [-0.166, +0.020] | 0.93 | 1.00 | +0.07 | 2 | 0 |
| ellipsoid_D3 | 30 | +0.000 [+0.000, +0.000] | 0.90 | 0.93 | +0.03 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.000 [+0.000, +0.000] | 0.20 | 0.20 | +0.00 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.000 [+0.000, +0.000] | 0.50 | 0.47 | -0.03 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | +0.000 [+0.000, +0.000] | 0.83 | 0.83 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | -1.063 [-1.383, -0.776] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rastrigin_D3 | 30 | -0.000 [-0.000, -0.000] | 0.07 | 0.07 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | +0.000 [-0.056, +0.000] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.352 [-0.521, -0.000] | 0.63 | 0.77 | +0.13 | 0 | 0 |
| sphere_D10 | 30 | -2.427 [-2.812, -1.986] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.000 [+0.000, +0.000] | 0.40 | 0.40 | +0.00 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.159 [-0.902, +0.000] | 0.90 | 1.00 | +0.10 | 0 | 0 |
| timing_D5 | 30 | -2.631 [-3.003, -1.881] | 0.97 | 1.00 | +0.03 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**5 configuration(s) flagged: ['multisensory_s1_D6', 'rastrigin_D3', 'rosenbrock_D6', 'sphere_D10', 'timing_D5']**
