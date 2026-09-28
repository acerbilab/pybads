# Compare pop (REF) and pop (NEW)

- REF, 286 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 12 runs: pybads b276da0, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 2 runs: pybads b276da0 (dirty), gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 300 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 60 | 60 | 0.283 | 0.0158 | 0.142 | ok |
| ellipsoid_D3_hetero | KS | func_count | 60 | 60 | 0.567 | 3.16e-09 | 3.79e-08 | FLAG |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 524 | 0.004 | 0.04 | FLAG |
| ellipsoid_D3_homo | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 60 | 60 | 0.433 | 1.94e-05 | 0.000213 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 60 | 60 | 781 | 0.324 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 60 | 60 | 0.267 | 0.0276 | 0.193 | ok |
| multisensory_s1_D6_homo | KS | func_count | 60 | 60 | 0.967 | 1.48e-31 | 2.22e-30 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 60 | 60 | 596 | 0.0189 | 0.151 | ok |
| sphere_D3_hetero | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 60 | 60 | 0.867 | 1.74e-23 | 2.44e-22 | FLAG |
| sphere_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 777 | 0.31 | 1 | ok |
| sphere_D3_homo | KS | true_error | 60 | 60 | 0.133 | 0.665 | 1 | ok |
| sphere_D3_homo | KS | func_count | 60 | 60 | 0.767 | 1.39e-17 | 1.8e-16 | FLAG |
| sphere_D3_homo | signed-rank | log10 true_error | 60 | 60 | 797 | 0.385 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 60 | +0.287 [+0.075, +0.397] | 0.10 | 0.07 | -0.03 | 0 | 0 |
| ellipsoid_D3_homo | 60 | +0.115 [-0.067, +0.267] | 0.63 | 0.55 | -0.08 | 0 | 0 |
| multisensory_s1_D6_homo | 60 | +0.137 [-0.013, +0.291] | 0.95 | 0.97 | +0.02 | 0 | 0 |
| sphere_D3_hetero | 60 | -0.196 [-0.299, +0.156] | 0.48 | 0.52 | +0.03 | 0 | 0 |
| sphere_D3_homo | 60 | -0.059 [-0.229, +0.112] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 60 vs 60 runs that is a KS statistic of at least 0.333.

**5 configuration(s) flagged: ['ellipsoid_D3_hetero', 'ellipsoid_D3_homo', 'multisensory_s1_D6_homo', 'sphere_D3_hetero', 'sphere_D3_homo']**
