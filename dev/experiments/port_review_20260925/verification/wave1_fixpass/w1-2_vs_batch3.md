<!-- W1-2 at d3640ab (with W1-35, 463312f, which moves no default run) against batch 3's end (d883cf9); default suite, seeds 0-29, gpyreg 1.3.3, Linux, 2026-09-26. Saved verbatim from the output of `population.py compare` (the runs, gitignored, in `dev/scripts/runs/population/`). -->

# Compare batch3_d883cf9 (REF) and w1-2_d3640ab (NEW)

- REF, 540 runs: pybads d883cf9, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads d3640ab, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 202 | 0.543 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 201 | 0.529 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 208 | 0.626 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 182 | 0.309 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 174 | 0.237 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 210 | 0.655 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 193 | 0.428 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 224 | 0.871 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 223 | 0.855 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 223 | 0.855 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 189 | 0.382 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 180 | 0.289 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 64 | 0.212 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 53 | 0.438 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 20 | 0.248 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 195 | 0.452 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 211 | 0.67 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.012 [-0.078, +0.167] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.155 [-0.586, +0.204] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.181 [-0.541, +0.558] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.143 [-0.372, +0.123] | 0.07 | 0.17 | +0.10 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.211 [-0.052, +0.371] | 0.70 | 0.47 | -0.23 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | +0.159 [-0.410, +0.470] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.193 [-0.240, +0.394] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.007 [-0.218, +0.257] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.000 [-0.021, +0.043] | 1.00 | 0.90 | -0.10 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [-0.000, +0.000] | 0.03 | 0.00 | -0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.216 [-0.477, +0.729] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.209 [-0.577, +0.256] | 0.70 | 0.77 | +0.07 | 0 | 0 |
| sphere_D10 | 30 | -0.016 [-0.299, +0.077] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.000 [+0.000, +0.067] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.000 [-0.000, +0.000] | 0.40 | 0.43 | +0.03 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.083 [-0.316, +0.106] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.109 [-0.212, +0.411] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
