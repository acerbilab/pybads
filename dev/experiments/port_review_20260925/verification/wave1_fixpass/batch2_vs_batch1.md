<!-- Batch 2 of wave 1's fix pass (the priors and bounds: W0-8, W1-22, W1-29, W1-23, W0-7) at 3236c2f against batch 1's end (1c7200b); default suite, seeds 0-29, gpyreg 1.3.3, Linux, 2026-09-26. Saved verbatim from the output of `population.py compare` (the runs, gitignored, in `dev/scripts/runs/population/`). -->

# Compare batch1_1c7200b (REF) and batch2_3236c2f (NEW)

- REF, 540 runs: pybads 1c7200b, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 3236c2f, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.733 | 4.33e-08 | 2.34e-06 | FLAG |
| ackley_D6 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 31 | 4.42e-06 | 0.000234 | FLAG |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.433 | 0.00655 | 0.327 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 139 | 0.0549 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 173 | 0.229 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 212 | 0.685 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 178 | 0.271 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 221 | 0.824 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 228 | 0.935 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 223 | 0.855 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 209 | 0.641 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 211 | 0.67 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 214 | 0.715 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 201 | 0.529 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 230 | 0.968 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.467 | 0.00253 | 0.129 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 82 | 0.00134 | 0.0697 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 215 | 0.73 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 152 | 0.778 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 232 | 1 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 207 | 0.612 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | -0.340 [-0.429, -0.236] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | +0.245 [-0.065, +0.675] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.356 [-0.306, +0.559] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.010 [-0.300, +0.258] | 0.33 | 0.13 | -0.20 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.162 [-0.333, +0.099] | 0.47 | 0.70 | +0.23 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.174 [-0.346, +0.702] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.032 [-0.284, +0.250] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.075 [-0.229, +0.298] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.016 [-0.151, +0.176] | 1.00 | 0.93 | -0.07 | 0 | 0 |
| rastrigin_D3 | 30 | -0.151 [-0.301, +0.048] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | +0.096 [-0.541, +0.750] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | 30 | -0.075 [-0.551, +0.287] | 0.77 | 0.77 | +0.00 | 0 | 0 |
| sphere_D10 | 30 | +0.102 [-0.290, +0.246] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.433 [+0.284, +0.786] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.130 [-0.249, +0.253] | 0.47 | 0.57 | +0.10 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [-0.088, +0.095] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.001 [-0.351, +0.353] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.024 [-0.138, +0.169] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**1 configuration(s) flagged: ['ackley_D6']**
