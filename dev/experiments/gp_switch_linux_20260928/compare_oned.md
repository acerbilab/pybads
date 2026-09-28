# Compare pop (REF) and pop (NEW)

- REF, 180 runs: pybads f8a1cad, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-1893eff/gpyreg at 1893eff
- NEW, 180 runs: pybads f8a1cad, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-1893eff/gpyreg at 1893eff

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ackley_D1 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ackley_D1 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ackley_D1 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D1_unbounded | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| ellipsoid_D1_unbounded | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| ellipsoid_D1_unbounded | signed-rank | log10 true_error | 30 | 30 | 55 | 0.184 | 1 | ok |
| rastrigin_D1 | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D1 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| rastrigin_D1 | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1 | KS | true_error | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D1 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1 | signed-rank | log10 true_error | 30 | 30 | 26.5 | 0.0319 | 0.573 | ok |
| sphere_D1_hetero | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1_hetero | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1_hetero | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1_homo | KS | true_error | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1_homo | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| sphere_D1_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ackley_D1 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| ellipsoid_D1_unbounded | 30 | +0.000 [-0.576, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| rastrigin_D1 | 30 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D1 | 30 | +0.000 [-0.514, +0.000] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| sphere_D1_hetero | 30 | +0.000 [+0.000, +0.000] | 0.83 | 0.83 | +0.00 | 0 | 0 |
| sphere_D1_homo | 30 | +0.000 [+0.000, +0.000] | 0.97 | 0.97 | +0.00 | 0 | 0 |

Holm family: 18 tests at alpha 0.05. A flag needs p <= 0.0028 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**no configuration flagged (18 tests)**
