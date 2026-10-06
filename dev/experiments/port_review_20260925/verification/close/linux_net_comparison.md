<!-- The net change of the port review on Linux at 30 seeds, from the records of the two references alone (no run): `python -u dev/scripts/population.py compare dev/experiments/population_linux_gpfixes_20260925 dev/experiments/population_linux_wave4_20260927`, at `3d31f3d`, with Python 3.11.15, NumPy 2.4.6 and SciPy 1.17.1, on 2026-09-28, at the close of the review; its output, verbatim. It exits 1, on the flags. -->

# Compare population_linux_gpfixes_20260925 (REF) and population_linux_wave4_20260927 (NEW)

- REF, 540 runs: pybads 97b2c66, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 46af65a, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.8 | 8.47e-10 | 4.57e-08 | FLAG |
| ackley_D6 | KS | func_count | 30 | 30 | 0.433 | 0.00655 | 0.301 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 17 | 3.86e-07 | 2e-05 | FLAG |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.567 | 8.74e-05 | 0.00446 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 198 | 0.49 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 152 | 0.1 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.467 | 0.00253 | 0.124 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 172 | 0.221 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.8 | 8.47e-10 | 4.57e-08 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 100 | 0.00538 | 0.253 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 149 | 0.0879 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 113 | 0.0128 | 0.539 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.467 | 0.00253 | 0.124 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 183 | 0.318 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 212 | 0.685 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 151 | 0.0961 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.5 | 0.0009 | 0.045 | FLAG |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 107 | 0.00871 | 0.383 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 179 | 0.28 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 184 | 0.328 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 109 | 0.00993 | 0.427 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 148 | 0.0841 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.433 | 0.00655 | 0.301 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 192 | 0.416 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 128 | 0.031 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 199 | 0.503 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | -0.410 [-0.492, -0.267] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.195 [-0.512, +0.270] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | -0.275 [-0.690, +0.018] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.043 [-0.437, +0.138] | 0.20 | 0.23 | +0.03 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.196 [+0.065, +0.500] | 0.67 | 0.43 | -0.23 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.385 [-0.975, +0.101] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | -0.224 [-0.732, -0.092] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | -0.068 [-0.586, +0.351] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.006 [-0.164, +0.220] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| rastrigin_D3 | 30 | -0.128 [-0.261, -0.000] | 0.00 | 0.07 | +0.07 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.698 [-1.887, +0.023] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.187 [-0.805, +0.346] | 0.77 | 0.83 | +0.07 | 0 | 0 |
| sphere_D10 | 30 | +0.113 [-0.162, +0.248] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.737 [+0.135, +0.977] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.083 [-0.413, +0.119] | 0.50 | 0.60 | +0.10 | 0 | 0 |
| sphere_D3_homo | 30 | +0.121 [-0.143, +0.252] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.520 [-0.719, +0.060] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | -0.189 [-0.424, +0.232] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**4 configuration(s) flagged: ['ackley_D6', 'ellipsoid_D10', 'ellipsoid_D3_homo', 'rosenbrock_D2']**
