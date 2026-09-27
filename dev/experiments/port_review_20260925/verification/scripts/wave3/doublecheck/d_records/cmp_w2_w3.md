# Compare population_linux_wave2_20260926 (REF) and population_linux_wave3_20260927 (NEW)

- REF, 540 runs: pybads 8510ca8, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads a14524d, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 195 | 0.452 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 157 | 0.124 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 230 | 0.968 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 179 | 0.28 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 226 | 0.903 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 183 | 0.318 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.845 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 214 | 0.715 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 215 | 0.73 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 230 | 0.968 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 219 | 0.792 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 227 | 0.919 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 191 | 0.404 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 228 | 0.935 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 160 | 0.14 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 169 | 0.198 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.048 [-0.108, +0.161] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.189 [-0.417, +0.051] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.057 [-0.641, +0.614] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.131 [-0.039, +0.284] | 0.13 | 0.10 | -0.03 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.089 [-0.412, +0.420] | 0.67 | 0.73 | +0.07 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.065 [-0.573, +0.126] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.091 [-0.243, +0.269] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.199 [-0.294, +0.326] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.019 [-0.207, +0.146] | 0.90 | 1.00 | +0.10 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [-0.199, +0.176] | 0.03 | 0.03 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | +0.026 [-0.571, +0.334] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.000 [-0.398, +0.522] | 0.73 | 0.77 | +0.03 | 0 | 0 |
| sphere_D10 | 30 | -0.170 [-0.345, +0.216] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.173 [-0.381, +0.223] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.081 [-0.145, +0.183] | 0.37 | 0.40 | +0.03 | 0 | 0 |
| sphere_D3_homo | 30 | -0.036 [-0.223, +0.108] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.200 [-0.038, +0.441] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.287 [-0.476, +0.067] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
