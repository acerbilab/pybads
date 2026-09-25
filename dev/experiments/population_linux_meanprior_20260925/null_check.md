# Null check: even vs odd seeds of population_linux_meanprior_20260925

- REF, 540 runs: pybads 8afbe16, gpyreg 1.3.3 from /home/user/pybads/dev/scripts/runs/gpyreg/v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| ackley_D6 | KS | func_count | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 15 | 15 | 0.467 | 0.0755 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 15 | 15 | 0.133 | 1 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| rastrigin_D3 | KS | true_error | 15 | 15 | 0.533 | 0.0262 | 0.945 | ok |
| rastrigin_D3 | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 15 | 15 | 0.133 | 1 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 15 | 15 | 0.467 | 0.0755 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| sphere_D10 | KS | true_error | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| sphere_D10 | KS | func_count | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| sphere_D2 | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D2 | KS | func_count | 15 | 15 | 0.0667 | 1 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| sphere_D3_homo | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| sphere_D3_homo | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| timing_D5 | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| timing_D5 | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | - | - | 0.20 | 0.33 | +0.13 | 0 | 0 |
| ellipsoid_D3_homo | - | - | 0.60 | 0.67 | +0.07 | 0 | 0 |
| ellipsoid_D3_unbounded | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | - | - | 0.93 | 1.00 | +0.07 | 0 | 0 |
| rastrigin_D3 | - | - | 0.07 | 0.07 | +0.00 | 0 | 0 |
| rosenbrock_D2 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | - | - | 0.67 | 0.80 | +0.13 | 0 | 0 |
| sphere_D10 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | - | - | 0.53 | 0.47 | -0.07 | 0 | 0 |
| sphere_D3_homo | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 36 tests at alpha 0.05. A flag needs p <= 0.0014 for the first step; for 15 vs 15 runs that is a KS statistic of at least 0.733.

**no configuration flagged (36 tests)**
