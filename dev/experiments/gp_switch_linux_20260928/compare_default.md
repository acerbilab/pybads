# Compare pop (REF) and pop (NEW)

- REF, 540 runs: pybads f8a1cad, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-1893eff/gpyreg at 1893eff
- NEW, 540 runs: pybads f8a1cad, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-1893eff/gpyreg at 1893eff

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.533 | 0.000293 | 0.0144 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 222 | 0.839 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.4 | 0.0156 | 0.704 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.6 | 2.37e-05 | 0.00121 | FLAG |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 173 | 0.229 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.4 | 0.0156 | 0.704 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 215 | 0.73 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 27 | 0.444 | 0.00459 | 0.211 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 27 | 0.556 | 0.000128 | 0.00642 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 27 | 27 | 65 | 0.00205 | 0.0961 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.733 | 4.33e-08 | 2.34e-06 | FLAG |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 194 | 0.44 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.667 | 1.28e-06 | 6.76e-05 | FLAG |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 144 | 0.0699 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 0 | 0.109 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 230 | 0.968 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 44 | 0.0707 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 162 | 0.152 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.533 | 0.000293 | 0.0144 | FLAG |
| sphere_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 37 | 1.06e-05 | 0.000552 | FLAG |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 120 | 0.0197 | 0.845 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 26 | 0.878 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | +0.051 [-0.362, +0.404] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.297 [-0.802, +1.456] | 1.00 | 0.80 | -0.20 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.083 [-0.259, +0.149] | 0.23 | 0.17 | -0.07 | 0 | 0 |
| ellipsoid_D3_homo | 27 | -0.376 [-0.712, -0.145] | 0.43 | 0.70 | +0.27 | 0 | 3 |
| ellipsoid_D3_unbounded | 30 | +0.126 [-0.888, +1.107] | 1.00 | 0.87 | -0.13 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.257 [-0.055, +0.468] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [+0.000, +0.000] | 0.07 | 0.07 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.060 [-0.613, +0.520] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | +0.000 [-0.215, +0.000] | 0.83 | 0.87 | +0.03 | 0 | 0 |
| sphere_D10 | 30 | -0.199 [-0.485, +0.017] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.964 [-1.221, -0.633] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.000 [+0.000, +0.000] | 0.60 | 0.60 | +0.00 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.502 [-0.819, -0.044] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

Crash count rising from zero: ['ellipsoid_D3_homo'].

**6 configuration(s) flagged: ['ellipsoid_D10', 'ellipsoid_D3', 'ellipsoid_D3_homo', 'ellipsoid_D3_unbounded', 'ellipsoid_D6', 'sphere_D2']**
