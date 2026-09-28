# Compare pop (REF) and pop (NEW)

- REF, 150 runs: pybads 46af65a, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 150 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 300 runs: pybads 73d5c28, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 60 | 60 | 0.283 | 0.0158 | 0.126 | ok |
| ellipsoid_D3_hetero | KS | func_count | 60 | 60 | 0.45 | 7.57e-06 | 9.84e-05 | FLAG |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 546 | 0.0105 | 0.105 | ok |
| ellipsoid_D3_homo | KS | true_error | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 60 | 60 | 0.35 | 0.00117 | 0.0129 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 60 | 60 | 835 | 0.556 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 60 | 60 | 0.0833 | 0.987 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 60 | 60 | 0.867 | 1.74e-23 | 2.61e-22 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 60 | 60 | 896 | 0.889 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 60 | 60 | 0.783 | 1.81e-18 | 2.54e-17 | FLAG |
| sphere_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 725 | 0.162 | 1 | ok |
| sphere_D3_homo | KS | true_error | 60 | 60 | 0.133 | 0.665 | 1 | ok |
| sphere_D3_homo | KS | func_count | 60 | 60 | 0.45 | 7.57e-06 | 9.84e-05 | FLAG |
| sphere_D3_homo | signed-rank | log10 true_error | 60 | 60 | 415 | 0.0126 | 0.113 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 60 | +0.254 [-0.045, +0.466] | 0.17 | 0.10 | -0.07 | 0 | 0 |
| ellipsoid_D3_homo | 60 | -0.038 [-0.271, +0.200] | 0.57 | 0.50 | -0.07 | 0 | 0 |
| multisensory_s1_D6_homo | 60 | +0.041 [-0.088, +0.105] | 0.98 | 0.93 | -0.05 | 0 | 0 |
| sphere_D3_hetero | 60 | +0.061 [-0.044, +0.228] | 0.52 | 0.50 | -0.02 | 0 | 0 |
| sphere_D3_homo | 60 | +0.033 [+0.000, +0.145] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 60 vs 60 runs that is a KS statistic of at least 0.333.

**5 configuration(s) flagged: ['ellipsoid_D3_hetero', 'ellipsoid_D3_homo', 'multisensory_s1_D6_homo', 'sphere_D3_hetero', 'sphere_D3_homo']**
