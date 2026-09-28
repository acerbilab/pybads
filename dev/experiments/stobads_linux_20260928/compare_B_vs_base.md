# Compare pop (REF) and pop (NEW)

- REF, 150 runs: pybads 46af65a, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 150 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 300 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 60 | 60 | 0.433 | 1.94e-05 | 0.000213 | FLAG |
| ellipsoid_D3_hetero | KS | func_count | 60 | 60 | 0.617 | 5.57e-11 | 7.24e-10 | FLAG |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 369 | 5.83e-05 | 0.000583 | FLAG |
| ellipsoid_D3_homo | KS | true_error | 60 | 60 | 0.05 | 1 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 60 | 60 | 0.35 | 0.00117 | 0.0105 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 60 | 60 | 879 | 0.791 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 60 | 60 | 0.133 | 0.665 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 60 | 60 | 0.917 | 3.95e-27 | 5.92e-26 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 60 | 60 | 752 | 0.23 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 60 | 60 | 0.1 | 0.928 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 60 | 60 | 0.783 | 1.81e-18 | 2.54e-17 | FLAG |
| sphere_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 884 | 0.819 | 1 | ok |
| sphere_D3_homo | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| sphere_D3_homo | KS | func_count | 60 | 60 | 0.483 | 1.02e-06 | 1.22e-05 | FLAG |
| sphere_D3_homo | signed-rank | log10 true_error | 60 | 60 | 672 | 0.0736 | 0.589 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 60 | +0.353 [+0.145, +0.498] | 0.17 | 0.07 | -0.10 | 0 | 0 |
| ellipsoid_D3_homo | 60 | +0.000 [-0.344, +0.157] | 0.57 | 0.55 | -0.02 | 0 | 0 |
| multisensory_s1_D6_homo | 60 | +0.101 [-0.069, +0.196] | 0.98 | 0.97 | -0.02 | 0 | 0 |
| sphere_D3_hetero | 60 | +0.018 [-0.090, +0.211] | 0.52 | 0.52 | +0.00 | 0 | 0 |
| sphere_D3_homo | 60 | -0.125 [-0.346, +0.042] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 60 vs 60 runs that is a KS statistic of at least 0.333.

**5 configuration(s) flagged: ['ellipsoid_D3_hetero', 'ellipsoid_D3_homo', 'multisensory_s1_D6_homo', 'sphere_D3_hetero', 'sphere_D3_homo']**
