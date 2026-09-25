# Compare population_linux_targetnoise_20260925 (REF) and population_linux_meanprior_20260925 (NEW)

- REF, 540 runs: pybads 1c8c71d, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 8afbe16, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 207 | 0.612 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.933 | 2.99e-14 | 1.62e-12 | FLAG |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.733 | 4.33e-08 | 2.21e-06 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 0 | 1.86e-09 | 9.87e-08 | FLAG |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.5 | 0.0009 | 0.0405 | FLAG |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 46 | 3.45e-05 | 0.00169 | FLAG |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 175 | 0.245 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 219 | 0.792 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.567 | 8.74e-05 | 0.00419 | FLAG |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 75 | 0.00073 | 0.0336 | FLAG |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.7 | 2.5e-07 | 1.25e-05 | FLAG |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.433 | 0.00655 | 0.288 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 4 | 1.3e-08 | 6.78e-07 | FLAG |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 154 | 0.109 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 184 | 0.328 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 213 | 0.7 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 148 | 0.0841 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.567 | 8.74e-05 | 0.00419 | FLAG |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 165 | 0.171 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 201 | 0.529 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 189 | 0.382 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 175 | 0.245 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 169 | 0.198 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 223 | 0.855 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | -0.049 [-0.147, +0.068] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -1.991 [-2.309, -1.613] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| ellipsoid_D3 | 30 | -1.284 [-1.940, -0.519] | 0.90 | 1.00 | +0.10 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.121 [-0.495, +0.149] | 0.10 | 0.27 | +0.17 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.029 [-0.240, +0.225] | 0.60 | 0.63 | +0.03 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.861 [-1.720, -0.226] | 0.90 | 1.00 | +0.10 | 0 | 0 |
| ellipsoid_D6 | 30 | -1.760 [-2.170, -1.398] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.216 [-0.153, +0.535] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | -0.088 [-0.177, +0.055] | 0.90 | 0.97 | +0.07 | 0 | 0 |
| rastrigin_D3 | 30 | +0.040 [-0.111, +0.269] | 0.03 | 0.07 | +0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.337 [-1.018, +0.416] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.870 [-1.434, -0.137] | 0.83 | 0.73 | -0.10 | 0 | 0 |
| sphere_D10 | 30 | -0.074 [-0.242, +0.390] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.260 [-0.422, +0.616] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.031 [-0.322, +0.227] | 0.50 | 0.50 | +0.00 | 0 | 0 |
| sphere_D3_homo | 30 | -0.220 [-0.345, +0.096] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.390 [-0.825, +0.401] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.010 [-0.488, +0.390] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**5 configuration(s) flagged: ['ellipsoid_D10', 'ellipsoid_D3', 'ellipsoid_D3_unbounded', 'ellipsoid_D6', 'rosenbrock_D6']**
