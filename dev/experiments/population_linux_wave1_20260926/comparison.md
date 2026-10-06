<!-- `population.py compare dev/experiments/population_linux_wave0_20260926 dev/scripts/runs/population/w1-1_6e22d32`: the net change of wave 1's fix pass on the default suite -->

# Compare population_linux_wave0_20260926 (REF) and w1-1_6e22d32 (NEW)

- REF, 540 runs: pybads ac3dfed, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 6e22d32 (dirty), gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.767 | 6.53e-09 | 3.53e-07 | FLAG |
| ackley_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 15 | 2.55e-07 | 1.35e-05 | FLAG |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 163 | 0.158 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 219 | 0.792 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 210 | 0.655 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 214 | 0.715 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.433 | 0.00655 | 0.334 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 192 | 0.416 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 217 | 0.761 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.773 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 166 | 0.177 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 225 | 0.887 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 193 | 0.428 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.467 | 0.00253 | 0.132 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 116 | 0.0155 | 0.773 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 165 | 0.171 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 228 | 0.935 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 148 | 0.0841 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 210 | 0.655 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 175 | 0.358 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 138 | 0.0523 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 220 | 0.808 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | -0.299 [-0.401, -0.210] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.327 [-0.493, +0.191] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.333 [-0.469, +0.432] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.136 [-0.371, +0.456] | 0.33 | 0.17 | -0.17 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.077 [-0.455, +0.263] | 0.53 | 0.47 | -0.07 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.307 [-0.886, +0.448] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.013 [-0.698, +0.517] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | -0.217 [-0.402, +0.145] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | -0.008 [-0.148, +0.172] | 1.00 | 0.90 | -0.10 | 0 | 0 |
| rastrigin_D3 | 30 | -0.079 [-0.284, +0.040] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | -1.034 [-1.595, -0.424] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.141 [-0.479, +0.008] | 0.77 | 0.77 | +0.00 | 0 | 0 |
| sphere_D10 | 30 | +0.016 [-0.241, +0.220] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.338 [-0.288, +0.879] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.065 [-0.089, +0.187] | 0.47 | 0.43 | -0.03 | 0 | 0 |
| sphere_D3_homo | 30 | +0.099 [-0.042, +0.210] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.355 [-0.644, -0.094] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.127 [-0.211, +0.345] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**1 configuration(s) flagged: ['ackley_D6']**
