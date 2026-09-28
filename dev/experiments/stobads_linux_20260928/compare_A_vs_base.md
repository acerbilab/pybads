# Compare pop (REF) and pop (NEW)

- REF, 150 runs: pybads 46af65a, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 150 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 286 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 12 runs: pybads b276da0, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 2 runs: pybads b276da0 (dirty), gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 60 | 60 | 0.217 | 0.12 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 765 | 0.365 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 60 | 60 | 0.133 | 0.665 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 60 | 60 | 0.183 | 0.267 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 60 | 60 | 768 | 0.279 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 60 | 60 | 0.183 | 0.267 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 60 | 60 | 715 | 0.199 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 770 | 0.286 | 1 | ok |
| sphere_D3_homo | KS | true_error | 60 | 60 | 0.133 | 0.665 | 1 | ok |
| sphere_D3_homo | KS | func_count | 60 | 60 | 0.367 | 0.000557 | 0.00835 | FLAG |
| sphere_D3_homo | signed-rank | log10 true_error | 60 | 60 | 700 | 0.229 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 60 | -0.002 [-0.099, +0.177] | 0.17 | 0.10 | -0.07 | 0 | 0 |
| ellipsoid_D3_homo | 60 | -0.173 [-0.281, +0.022] | 0.57 | 0.63 | +0.07 | 0 | 0 |
| multisensory_s1_D6_homo | 60 | -0.034 [-0.193, +0.100] | 0.98 | 0.95 | -0.03 | 0 | 0 |
| sphere_D3_hetero | 60 | +0.088 [-0.089, +0.227] | 0.52 | 0.48 | -0.03 | 0 | 0 |
| sphere_D3_homo | 60 | -0.063 [-0.206, +0.088] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 60 vs 60 runs that is a KS statistic of at least 0.333.

**1 configuration(s) flagged: ['sphere_D3_homo']**
