# Null check: even vs odd seeds of population_prereview_20260927

- REF, 510 runs: pybads ab4dded, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4
- REF, 1290 runs: pybads ab4ddedc, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D6 | KS | true_error | 50 | 50 | 0.16 | 0.549 | 1 | ok |
| ackley_D6 | KS | func_count | 50 | 50 | 0.08 | 0.998 | 1 | ok |
| ellipsoid_D10 | KS | true_error | 50 | 50 | 0.1 | 0.967 | 1 | ok |
| ellipsoid_D10 | KS | func_count | 50 | 50 | 0.08 | 0.998 | 1 | ok |
| ellipsoid_D3 | KS | true_error | 50 | 50 | 0.14 | 0.717 | 1 | ok |
| ellipsoid_D3 | KS | func_count | 50 | 50 | 0.22 | 0.179 | 1 | ok |
| ellipsoid_D3_hetero | KS | true_error | 50 | 50 | 0.34 | 0.00584 | 0.21 | ok |
| ellipsoid_D3_hetero | KS | func_count | 50 | 50 | 0.12 | 0.869 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 50 | 50 | 0.14 | 0.717 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 50 | 50 | 0.12 | 0.869 | 1 | ok |
| ellipsoid_D3_unbounded | KS | true_error | 50 | 50 | 0.22 | 0.179 | 1 | ok |
| ellipsoid_D3_unbounded | KS | func_count | 50 | 50 | 0.28 | 0.0392 | 1 | ok |
| ellipsoid_D6 | KS | true_error | 50 | 50 | 0.14 | 0.717 | 1 | ok |
| ellipsoid_D6 | KS | func_count | 50 | 50 | 0.12 | 0.869 | 1 | ok |
| multisensory_s1_D6 | KS | true_error | 50 | 50 | 0.12 | 0.869 | 1 | ok |
| multisensory_s1_D6 | KS | func_count | 50 | 50 | 0.1 | 0.967 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 50 | 50 | 0.2 | 0.272 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 50 | 50 | 0.12 | 0.869 | 1 | ok |
| rastrigin_D3 | KS | true_error | 50 | 50 | 0.18 | 0.396 | 1 | ok |
| rastrigin_D3 | KS | func_count | 50 | 50 | 0.14 | 0.717 | 1 | ok |
| rosenbrock_D2 | KS | true_error | 50 | 50 | 0.22 | 0.179 | 1 | ok |
| rosenbrock_D2 | KS | func_count | 50 | 50 | 0.16 | 0.549 | 1 | ok |
| rosenbrock_D6 | KS | true_error | 50 | 50 | 0.1 | 0.967 | 1 | ok |
| rosenbrock_D6 | KS | func_count | 50 | 50 | 0.16 | 0.549 | 1 | ok |
| sphere_D10 | KS | true_error | 50 | 50 | 0.14 | 0.717 | 1 | ok |
| sphere_D10 | KS | func_count | 50 | 50 | 0.1 | 0.967 | 1 | ok |
| sphere_D2 | KS | true_error | 50 | 50 | 0.12 | 0.869 | 1 | ok |
| sphere_D2 | KS | func_count | 50 | 50 | 0.08 | 0.998 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 50 | 50 | 0.2 | 0.272 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 50 | 50 | 0.16 | 0.549 | 1 | ok |
| sphere_D3_homo | KS | true_error | 50 | 50 | 0.22 | 0.179 | 1 | ok |
| sphere_D3_homo | KS | func_count | 50 | 50 | 0.16 | 0.549 | 1 | ok |
| sphere_nonbox_D3 | KS | true_error | 50 | 50 | 0.14 | 0.717 | 1 | ok |
| sphere_nonbox_D3 | KS | func_count | 50 | 50 | 0.12 | 0.869 | 1 | ok |
| timing_D5 | KS | true_error | 50 | 50 | 0.26 | 0.0678 | 1 | ok |
| timing_D5 | KS | func_count | 50 | 50 | 0.16 | 0.549 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D6 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D10 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D3_hetero | - | - | 0.24 | 0.14 | -0.10 | 0 | 0 |
| ellipsoid_D3_homo | - | - | 0.62 | 0.66 | +0.04 | 0 | 0 |
| ellipsoid_D3_unbounded | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D6 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| multisensory_s1_D6_homo | - | - | 0.96 | 0.96 | +0.00 | 0 | 0 |
| rastrigin_D3 | - | - | 0.00 | 0.02 | +0.02 | 0 | 0 |
| rosenbrock_D2 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rosenbrock_D6 | - | - | 0.72 | 0.72 | +0.00 | 0 | 0 |
| sphere_D10 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D2 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D3_hetero | - | - | 0.42 | 0.50 | +0.08 | 0 | 0 |
| sphere_D3_homo | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_nonbox_D3 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| timing_D5 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 36 tests at alpha 0.05. A flag needs p <= 0.0014 for the first step; for 50 vs 50 runs that is a KS statistic of at least 0.380.

**no configuration flagged (36 tests)**
