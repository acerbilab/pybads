# Compare g1_es_c276d79 (REF) and g2_w31_149d528 (NEW)

- REF, 540 runs: pybads c276d79, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 149d528, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 113 | 0.931 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 201 | 0.721 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 12 | 0.214 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 89 | 0.55 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 42 | 0.51 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 20 | 0.445 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 113 | 0.931 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 157 | 0.638 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 120 | 0.159 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 109 | 0.57 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 3 | 0.465 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 135 | 0.195 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 131 | 0.397 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 135 | 0.459 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 31 | 0.859 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 11 | 0.173 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.000 [-0.004, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.005 [-0.039, +0.018] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.000 [+0.000, +0.000] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.000 [-0.001, +0.029] | 0.20 | 0.17 | -0.03 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.000 [+0.000, +0.000] | 0.60 | 0.63 | +0.03 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.000 [-0.055, +0.090] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | -0.006 [-0.224, +0.046] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.004 [+0.000, +0.037] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [+0.000, +0.000] | 0.03 | 0.07 | +0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | +0.000 [+0.000, +0.000] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | +0.000 [-0.000, +0.285] | 0.80 | 0.73 | -0.07 | 0 | 0 |
| sphere_D10 | 30 | -0.001 [-0.016, +0.001] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.000 [-0.010, +0.091] | 0.47 | 0.43 | -0.03 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
