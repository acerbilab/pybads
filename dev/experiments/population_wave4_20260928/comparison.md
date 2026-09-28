# Compare population_prereview_20260927 (REF) and population_wave4_20260928 (NEW)

- REF, 510 runs: pybads ab4dded, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4
- REF, 1290 runs: pybads ab4ddedc, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4
- NEW, 1800 runs: pybads a4dcd651, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 100 | 100 | 0.5 | 1e-11 | 5.11e-10 | FLAG |
| ackley_D6 | KS | func_count | 100 | 100 | 0.43 | 1.12e-08 | 5.46e-07 | FLAG |
| ackley_D6 | signed-rank | log10 true_error | 100 | 100 | 600 | 3.62e-11 | 1.81e-09 | FLAG |
| ellipsoid_D10 | KS | true_error | 100 | 100 | 0.24 | 0.00613 | 0.215 | ok |
| ellipsoid_D10 | KS | func_count | 100 | 100 | 0.59 | 1.77e-16 | 9.37e-15 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 100 | 100 | 1.7e+03 | 0.00441 | 0.159 | ok |
| ellipsoid_D3 | KS | true_error | 100 | 100 | 0.21 | 0.0241 | 0.722 | ok |
| ellipsoid_D3 | KS | func_count | 100 | 100 | 0.14 | 0.282 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 100 | 100 | 2.05e+03 | 0.101 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 100 | 100 | 0.12 | 0.47 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 100 | 100 | 0.56 | 8.77e-15 | 4.56e-13 | FLAG |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 100 | 100 | 2.46e+03 | 0.81 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 100 | 100 | 0.12 | 0.47 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 100 | 100 | 0.72 | 2.68e-25 | 1.45e-23 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 100 | 100 | 2.14e+03 | 0.187 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 100 | 100 | 0.17 | 0.111 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 100 | 100 | 0.09 | 0.815 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 100 | 100 | 1.74e+03 | 0.00667 | 0.22 | ok |
| ellipsoid_D6 | KS | true_error | 100 | 100 | 0.19 | 0.0539 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 100 | 100 | 0.22 | 0.0156 | 0.483 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 100 | 100 | 1.91e+03 | 0.0336 | 0.974 | ok |
| multisensory_s1_D6 | KS | true_error | 100 | 100 | 0.07 | 0.968 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 100 | 100 | 0.25 | 0.00373 | 0.149 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 100 | 100 | 2.47e+03 | 0.85 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 100 | 100 | 0.15 | 0.211 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 100 | 100 | 0.33 | 3.21e-05 | 0.00148 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 100 | 100 | 2.15e+03 | 0.2 | 1 | ok |
| rastrigin_D3 | KS | true_error | 100 | 100 | 0.09 | 0.815 | 1 | ok |
| rastrigin_D3 | KS | func_count | 100 | 100 | 0.09 | 0.815 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 100 | 100 | 2.17e+03 | 0.226 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 100 | 100 | 0.34 | 1.61e-05 | 0.000755 | FLAG |
| rosenbrock_D2 | KS | func_count | 100 | 100 | 0.11 | 0.583 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 100 | 100 | 1.36e+03 | 6.65e-05 | 0.00293 | FLAG |
| rosenbrock_D6 | KS | true_error | 100 | 100 | 0.1 | 0.702 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 100 | 100 | 0.19 | 0.0539 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 100 | 100 | 2.39e+03 | 0.64 | 1 | ok |
| sphere_D10 | KS | true_error | 100 | 100 | 0.28 | 0.000738 | 0.0302 | FLAG |
| sphere_D10 | KS | func_count | 100 | 100 | 0.08 | 0.908 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 100 | 100 | 1.47e+03 | 0.000286 | 0.0123 | FLAG |
| sphere_D2 | KS | true_error | 100 | 100 | 0.33 | 3.21e-05 | 0.00148 | FLAG |
| sphere_D2 | KS | func_count | 100 | 100 | 0.06 | 0.994 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 100 | 100 | 1.21e+03 | 6.35e-06 | 0.000305 | FLAG |
| sphere_D3_hetero | KS | true_error | 100 | 100 | 0.08 | 0.908 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 100 | 100 | 0.25 | 0.00373 | 0.149 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 100 | 100 | 2.44e+03 | 0.762 | 1 | ok |
| sphere_D3_homo | KS | true_error | 100 | 100 | 0.25 | 0.00373 | 0.149 | ok |
| sphere_D3_homo | KS | func_count | 100 | 100 | 0.29 | 0.000412 | 0.0173 | FLAG |
| sphere_D3_homo | signed-rank | log10 true_error | 100 | 100 | 1.79e+03 | 0.0116 | 0.372 | ok |
| sphere_nonbox_D3 | KS | true_error | 100 | 100 | 0.25 | 0.00373 | 0.149 | ok |
| sphere_nonbox_D3 | KS | func_count | 100 | 100 | 0.15 | 0.211 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 100 | 100 | 1.73e+03 | 0.0062 | 0.215 | ok |
| timing_D5 | KS | true_error | 100 | 100 | 0.19 | 0.0539 | 1 | ok |
| timing_D5 | KS | func_count | 100 | 100 | 0.17 | 0.111 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 100 | 100 | 2.08e+03 | 0.124 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 100 | -0.273 [-0.343, -0.240] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 100 | -0.162 [-0.294, -0.074] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 100 | -0.230 [-0.565, +0.049] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 100 | -0.003 [-0.244, +0.141] | 0.19 | 0.16 | -0.03 | 0 | 0 |
| ellipsoid_D3_homo | 100 | +0.112 [-0.104, +0.234] | 0.64 | 0.55 | -0.09 | 0 | 0 |
| ellipsoid_D3_unbounded | 100 | -0.434 [-0.534, -0.128] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 100 | -0.189 [-0.352, +0.038] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 100 | -0.010 [-0.136, +0.178] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 100 | -0.019 [-0.145, +0.071] | 0.96 | 0.97 | +0.01 | 0 | 0 |
| rastrigin_D3 | 100 | -0.000 [-0.109, +0.000] | 0.01 | 0.04 | +0.03 | 0 | 0 |
| rosenbrock_D2 | 100 | -0.513 [-0.861, -0.245] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 100 | -0.000 [-0.213, +0.176] | 0.72 | 0.81 | +0.09 | 0 | 0 |
| sphere_D10 | 100 | +0.196 [+0.044, +0.321] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 100 | +0.327 [+0.172, +0.581] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 100 | +0.019 [-0.103, +0.137] | 0.46 | 0.49 | +0.03 | 0 | 0 |
| sphere_D3_homo | 100 | +0.168 [-0.015, +0.335] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 100 | -0.174 [-0.347, -0.037] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 100 | -0.133 [-0.419, +0.061] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 100 vs 100 runs that is a KS statistic of at least 0.280.

**9 configuration(s) flagged: ['ackley_D6', 'ellipsoid_D10', 'ellipsoid_D3_hetero', 'ellipsoid_D3_homo', 'multisensory_s1_D6_homo', 'rosenbrock_D2', 'sphere_D10', 'sphere_D2', 'sphere_D3_homo']**
