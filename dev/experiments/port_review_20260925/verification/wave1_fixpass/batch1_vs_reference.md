<!-- Batch 1 of wave 1's fix pass (the fit retries: W1-21, W1-13, W1-12, W1-14, W1-16) at 1c7200b against the Linux reference population_linux_wave0_20260926 (ac3dfed); default suite, seeds 0-29, gpyreg 1.3.3, Linux, 2026-09-26. Saved verbatim from the output of `population.py compare` (the runs, gitignored, in `dev/scripts/runs/population/`). -->

# Compare population_linux_wave0_20260926 (REF) and batch1_1c7200b (NEW)

- REF, 540 runs: pybads ac3dfed, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 1c7200b, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 14 | 0.169 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 190 | 0.393 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 194 | 0.44 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 206 | 0.598 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 207 | 0.612 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 145 | 0.0732 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 1 | 0.655 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 0.317 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.433 | 0.00655 | 0.354 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 96 | 0.00861 | 0.456 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 6 | 0.0929 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 31 | 0.53 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D2 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 161 | 0.339 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 0.317 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 162 | 0.152 | 1 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 1 | 0.285 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | -0.136 [-0.238, +0.273] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | -0.190 [-0.845, +0.315] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | +0.000 [+0.000, +0.000] | 0.33 | 0.33 | +0.00 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.013 [-0.215, +0.051] | 0.53 | 0.47 | -0.07 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.242 [-0.736, +0.407] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | -0.245 [-0.617, +0.133] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rastrigin_D3 | 30 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.966 [-1.472, -0.228] | 0.97 | 1.00 | +0.03 | 0 | 0 |
| rosenbrock_D6 | 30 | +0.000 [+0.000, +0.000] | 0.77 | 0.77 | +0.00 | 0 | 0 |
| sphere_D10 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | +0.079 [-0.213, +0.704] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.000 [+0.000, +0.000] | 0.47 | 0.47 | +0.00 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.196 [-0.344, +0.106] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**no configuration flagged (54 tests)**
