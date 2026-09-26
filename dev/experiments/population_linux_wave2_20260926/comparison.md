<!-- `population.py compare dev/experiments/population_linux_wave1_20260926 dev/scripts/runs/population/w2-head_8510ca8`: the net change of wave 2's fix pass on the default suite -->

# Compare population_linux_wave1_20260926 (REF) and w2-head_8510ca8 (NEW)

- REF, 540 runs: pybads 6e22d32 (dirty), gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 8510ca8, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 157 | 0.124 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.567 | 8.74e-05 | 0.00472 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 160 | 0.14 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 229 | 0.952 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.829 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 223 | 0.855 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 178 | 0.271 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 158 | 0.129 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 164 | 0.164 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 214 | 0.715 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 216 | 0.746 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 224 | 0.871 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 199 | 0.503 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 145 | 0.0732 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 75 | 0.0321 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 210 | 0.871 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 162 | 0.517 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 124 | 0.67 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 210 | 0.655 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | -0.048 [-0.193, +0.079] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | +0.089 [-0.083, +0.218] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.037 [-0.567, +0.246] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.020 [-0.196, +0.275] | 0.17 | 0.13 | -0.03 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.332 [-0.666, +0.191] | 0.47 | 0.67 | +0.20 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | +0.278 [-0.042, +1.131] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | -0.289 [-0.469, +0.138] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.047 [-0.193, +0.274] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | -0.065 [-0.259, +0.196] | 0.90 | 0.90 | +0.00 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [-0.000, +0.000] | 0.00 | 0.03 | +0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.124 [-0.559, +0.385] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.133 [-0.431, +0.092] | 0.77 | 0.73 | -0.03 | 0 | 0 |
| sphere_D10 | 30 | +0.171 [-0.074, +0.360] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.155 [+0.000, +0.558] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.002 [-0.107, +0.108] | 0.43 | 0.37 | -0.07 | 0 | 0 |
| sphere_D3_homo | 30 | -0.030 [-0.176, +0.139] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.000 [-0.086, +0.029] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.148 [-0.324, +0.040] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**1 configuration(s) flagged: ['ellipsoid_D10']**
