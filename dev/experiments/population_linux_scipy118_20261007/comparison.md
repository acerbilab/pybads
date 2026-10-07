# Compare population_linux_gpyreg140_20260930 (REF) and population_linux_scipy118_20261007 (NEW)

- REF, 720 runs: pybads 60ad9e0f, gpyreg 1.3.4.dev35+g3e56dce0f from /home/user/pybads/dev/scripts/runs/gpyreg/main_3e56dce/gpyreg at 3e56dce
- NEW, 720 runs: pybads 0dc5932f, gpyreg 1.4.0 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.4.0/gpyreg at 682585f

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 206 | 0.804 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 211 | 0.67 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 166 | 0.177 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 120 | 0.0197 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 150 | 0.0919 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 105 | 0.00761 | 0.548 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 134 | 0.0427 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 168 | 0.191 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 195 | 0.452 | 1 | ok |
| periodic_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| periodic_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| periodic_D2 | signed-rank | log10 true_error | 30 | 30 | 153 | 0.568 | 1 | ok |
| periodic_D3_hetero | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| periodic_D3_hetero | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| periodic_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 28 | 0.221 | 1 | ok |
| periodic_D3_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| periodic_D3_homo | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| periodic_D3_homo | signed-rank | log10 true_error | 30 | 30 | 64 | 0.349 | 1 | ok |
| periodic_D4 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| periodic_D4 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| periodic_D4 | signed-rank | log10 true_error | 30 | 30 | 219 | 0.792 | 1 | ok |
| periodic_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| periodic_D6 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| periodic_D6 | signed-rank | log10 true_error | 30 | 30 | 228 | 0.935 | 1 | ok |
| periodic_rosenbrock_D4 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| periodic_rosenbrock_D4 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| periodic_rosenbrock_D4 | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 210 | 0.655 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 223 | 0.855 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 229 | 0.952 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 215 | 0.73 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 172 | 0.221 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 55 | 0.184 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 33 | 0.638 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 217 | 0.761 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 209 | 0.641 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.003 [-0.106, +0.069] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.098 [-0.322, +0.284] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.292 [-0.224, +0.926] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.108 [-0.055, +0.515] | 0.23 | 0.00 | -0.23 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.185 [-0.304, -0.004] | 0.43 | 0.60 | +0.17 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | +0.530 [+0.153, +0.791] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.173 [-0.042, +0.565] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.103 [-0.201, +0.524] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.006 [-0.008, +0.116] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_D2 | 30 | +0.004 [-0.167, +0.297] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_D3_hetero | 30 | +0.000 [+0.000, +0.000] | 0.20 | 0.23 | +0.03 | 0 | 0 |
| periodic_D3_homo | 30 | +0.000 [+0.000, +0.000] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| periodic_D4 | 30 | +0.017 [-0.145, +0.151] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_D6 | 30 | +0.020 [-0.121, +0.102] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_rosenbrock_D4 | 30 | -0.123 [-0.671, +0.247] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rastrigin_D3 | 30 | -0.000 [-0.000, +0.000] | 0.07 | 0.07 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.061 [-0.486, +0.532] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.004 [-0.406, +0.353] | 0.83 | 0.83 | +0.00 | 0 | 0 |
| sphere_D10 | 30 | +0.121 [-0.214, +0.261] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.114 [-0.563, +0.032] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.000 [+0.000, +0.116] | 0.60 | 0.60 | +0.00 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.039 [-0.174, +0.185] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.130 [-0.307, +0.339] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 72 tests at alpha 0.05. A flag needs p <= 0.00069 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.533.

**no configuration flagged (72 tests)**
