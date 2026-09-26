# Compare population_linux_gpfixes_20260925 (REF) and baseline_ac3dfed (NEW)

- REF, 540 runs: pybads 97b2c66, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads ac3dfed, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.533 | 0.000293 | 0.0153 | FLAG |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 199 | 0.927 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.767 | 6.53e-09 | 3.53e-07 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 169 | 0.294 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.633 | 5.8e-06 | 0.000307 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 213 | 0.7 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 189 | 0.538 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.433 | 0.00655 | 0.334 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 161 | 0.222 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.000 [-0.107, +0.225] | 0.20 | 0.33 | +0.13 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.129 [-0.112, +0.368] | 0.67 | 0.53 | -0.13 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.019 [-0.163, +0.127] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | +0.000 [+0.000, +0.000] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | +0.000 [+0.000, +0.000] | 0.77 | 0.77 | +0.00 | 0 | 0 |
| sphere_D10 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.016 [-0.180, +0.231] | 0.50 | 0.47 | -0.03 | 0 | 0 |
| sphere_D3_homo | 30 | +0.007 [-0.062, +0.227] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**3 configuration(s) flagged: ['ellipsoid_D3_hetero', 'ellipsoid_D3_homo', 'multisensory_s1_D6_homo']**
