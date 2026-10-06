<!-- Batch 3 of wave 1's fix pass (the calibration test: W1-4, W1-5, W1-3) at d883cf9 against batch 2's end (3236c2f); default suite, seeds 0-29, gpyreg 1.3.3, Linux, 2026-09-26. Saved verbatim from the output of `population.py compare` (the runs, gitignored, in `dev/scripts/runs/population/`). -->

# Compare batch2_3236c2f (REF) and batch3_d883cf9 (NEW)

- REF, 540 runs: pybads 3236c2f, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads d883cf9, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 231 | 0.984 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.467 | 0.00253 | 0.132 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 152 | 0.1 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 215 | 0.73 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 135 | 0.0449 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 195 | 0.452 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 162 | 0.152 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 219 | 0.792 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 142 | 0.0636 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 216 | 0.974 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 168 | 0.191 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 191 | 0.404 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 185 | 0.339 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 231 | 0.984 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.5 | 0.0009 | 0.0477 | FLAG |
| sphere_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 67 | 0.000345 | 0.0186 | FLAG |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 206 | 0.598 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 199 | 0.927 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 216 | 0.746 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 120 | 0.833 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | -0.009 [-0.107, +0.132] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.152 [-0.687, +0.029] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.051 [-0.739, +0.605] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.199 [-0.067, +0.423] | 0.13 | 0.07 | -0.07 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.139 [-0.418, +0.256] | 0.70 | 0.70 | +0.00 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.312 [-0.926, +0.313] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | -0.135 [-0.396, +0.229] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | -0.058 [-0.277, -0.009] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | -0.000 [-0.154, +0.078] | 0.93 | 1.00 | +0.07 | 0 | 0 |
| rastrigin_D3 | 30 | -0.000 [-0.243, -0.000] | 0.00 | 0.03 | +0.03 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.090 [-0.389, +0.087] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.000 [-0.015, +0.381] | 0.77 | 0.70 | -0.07 | 0 | 0 |
| sphere_D10 | 30 | -0.037 [-0.222, +0.416] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.427 [-0.617, -0.137] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.113 [-0.352, +0.303] | 0.57 | 0.40 | -0.17 | 0 | 0 |
| sphere_D3_homo | 30 | +0.005 [-0.107, +0.111] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| sphere_nonbox_D3 | 30 | +0.187 [-0.453, +0.262] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.000 [+0.000, +0.118] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**1 configuration(s) flagged: ['sphere_D2']**
