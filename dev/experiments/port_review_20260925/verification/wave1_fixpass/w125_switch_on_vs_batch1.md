<!-- Exploratory measurement of W1-25: batch 1's end with W1-35 and gpyreg's raise_on_cholesky_failure turned on (local commit 4515621 on w125-exploratory, never pushed; gpyreg from the branch of acerbilab/gpyreg#56, whose default is bit-identical to 1.3.3 and gives batch 1's fingerprint ed8f953edbd6f141 with the switch off), against batch 1's end (1c7200b); default suite, seeds 0-29, Linux, 2026-09-26. Saved verbatim from the output of `population.py compare` (the runs, gitignored, in `dev/scripts/runs/population/`). -->

# Compare batch1_1c7200b (REF) and w125_on_4515621 (NEW)

- REF, 540 runs: pybads 1c7200b, gpyreg 1.3.3 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 540 runs: pybads 4515621, gpyreg 1.3.3 from /home/user/gpyreg-wt/switch/gpyreg at 5ec308f

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ackley_D6 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| ackley_D6 | signed-rank | log10 true_error | 30 | 30 | 22 | 0.101 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| ellipsoid_D10 | signed-rank | log10 true_error | 30 | 30 | 107 | 0.00871 | 0.4 | ok |
| ellipsoid_D3 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 30 | 30 | 0.7 | 2.5e-07 | 1.33e-05 | FLAG |
| ellipsoid_D3 | signed-rank | log10 true_error | 30 | 30 | 190 | 0.393 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 132 | 0.0384 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 160 | 0.14 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 30 | 30 | 0.767 | 6.53e-09 | 3.53e-07 | FLAG |
| ellipsoid_D3_unbounded | signed-rank | log10 true_error | 30 | 30 | 168 | 0.191 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 30 | 30 | 0.5 | 0.0009 | 0.0459 | FLAG |
| ellipsoid_D6 | KS | func_count | 30 | 30 | 0.467 | 0.00253 | 0.121 | ok |
| ellipsoid_D6 | signed-rank | log10 true_error | 30 | 30 | 72 | 0.000555 | 0.0289 | FLAG |
| multisensory_s1_D6 | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6 | signed-rank | log10 true_error | 30 | 30 | 93 | 0.936 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 10 | 0.499 | 1 | ok |
| rastrigin_D3 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rastrigin_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| rastrigin_D3 | signed-rank | log10 true_error | 30 | 30 | 164 | 0.77 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| rosenbrock_D2 | signed-rank | log10 true_error | 30 | 30 | 194 | 0.44 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| rosenbrock_D6 | signed-rank | log10 true_error | 30 | 30 | 146 | 0.657 | 1 | ok |
| sphere_D10 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_D10 | signed-rank | log10 true_error | 30 | 30 | 229 | 0.952 | 1 | ok |
| sphere_D2 | KS | true_error | 30 | 30 | 0.5 | 0.0009 | 0.0459 | FLAG |
| sphere_D2 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D2 | signed-rank | log10 true_error | 30 | 30 | 92 | 0.00299 | 0.14 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 4 | 0.715 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 2 | 0.593 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| sphere_nonbox_D3 | signed-rank | log10 true_error | 30 | 30 | 88 | 0.00219 | 0.107 | ok |
| timing_D5 | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| timing_D5 | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| timing_D5 | signed-rank | log10 true_error | 30 | 30 | 48 | 0.301 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | 30 | +0.271 [+0.069, +0.472] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | 30 | +0.537 [-0.691, +1.286] | 1.00 | 0.90 | -0.10 | 0 | 0 |
| ellipsoid_D3_hetero | 30 | -0.327 [-0.654, +0.011] | 0.33 | 0.40 | +0.07 | 0 | 0 |
| ellipsoid_D3_homo | 30 | -0.305 [-0.536, -0.021] | 0.47 | 0.67 | +0.20 | 0 | 0 |
| ellipsoid_D3_unbounded | 30 | -0.592 [-1.221, +0.239] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | 30 | +0.493 [+0.214, +0.930] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | 30 | +0.000 [+0.000, +0.013] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 0.97 | -0.03 | 0 | 0 |
| rastrigin_D3 | 30 | -0.000 [-0.000, +0.000] | 0.00 | 0.00 | +0.00 | 0 | 0 |
| rosenbrock_D2 | 30 | -0.295 [-0.973, +0.198] | 1.00 | 0.93 | -0.07 | 0 | 0 |
| rosenbrock_D6 | 30 | +0.000 [-0.066, +0.017] | 0.77 | 0.77 | +0.00 | 0 | 0 |
| sphere_D10 | 30 | -0.017 [-0.199, +0.220] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | 30 | -0.474 [-1.315, -0.106] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 30 | +0.000 [+0.000, +0.000] | 0.47 | 0.50 | +0.03 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | 30 | -0.410 [-0.612, -0.115] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 54 tests at alpha 0.05. A flag needs p <= 0.00093 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.500.

**4 configuration(s) flagged: ['ellipsoid_D3', 'ellipsoid_D3_unbounded', 'ellipsoid_D6', 'sphere_D2']**
