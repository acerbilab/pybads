# Compare pop (REF) and pop (NEW)

- REF, 300 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 300 runs: pybads 73d5c28, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 60 | 60 | 0.233 | 0.0761 | 0.609 | ok |
| ellipsoid_D3_hetero | KS | func_count | 60 | 60 | 0.283 | 0.0158 | 0.189 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 723 | 0.158 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 60 | 60 | 781 | 0.324 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 60 | 60 | 0.4 | 0.000112 | 0.00157 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 60 | 60 | 789 | 0.354 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 60 | 60 | 0.2 | 0.182 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 60 | 60 | 0.5 | 3.51e-07 | 5.27e-06 | FLAG |
| sphere_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 616 | 0.0277 | 0.276 | ok |
| sphere_D3_homo | KS | true_error | 60 | 60 | 0.267 | 0.0276 | 0.276 | ok |
| sphere_D3_homo | KS | func_count | 60 | 60 | 0.35 | 0.00117 | 0.0152 | FLAG |
| sphere_D3_homo | signed-rank | log10 true_error | 60 | 60 | 594 | 0.0181 | 0.199 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 60 | -0.093 [-0.286, +0.131] | 0.07 | 0.10 | +0.03 | 0 | 0 |
| ellipsoid_D3_homo | 60 | +0.053 [-0.077, +0.245] | 0.55 | 0.50 | -0.05 | 0 | 0 |
| multisensory_s1_D6_homo | 60 | -0.043 [-0.182, +0.044] | 0.97 | 0.93 | -0.03 | 0 | 0 |
| sphere_D3_hetero | 60 | +0.148 [+0.018, +0.252] | 0.52 | 0.50 | -0.02 | 0 | 0 |
| sphere_D3_homo | 60 | +0.116 [-0.014, +0.369] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 60 vs 60 runs that is a KS statistic of at least 0.333.

**3 configuration(s) flagged: ['multisensory_s1_D6_homo', 'sphere_D3_hetero', 'sphere_D3_homo']**
