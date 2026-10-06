# Compare pop (REF) and pop (NEW)

- REF, 286 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 12 runs: pybads b276da0, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 2 runs: pybads b276da0 (dirty), gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 300 runs: pybads 73d5c28, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 60 | 60 | 0.133 | 0.665 | 1 | ok |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 753 | 0.714 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 60 | 60 | 0.233 | 0.0761 | 1 | ok |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 60 | 60 | 720 | 0.151 | 1 | ok |
| multisensory_s1_D6_homo | KS | true_error | 60 | 60 | 0.1 | 0.928 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 60 | 60 | 0.2 | 0.182 | 1 | ok |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 60 | 60 | 735 | 0.949 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 60 | 60 | 0.0833 | 0.987 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 688 | 0.993 | 1 | ok |
| sphere_D3_homo | KS | true_error | 60 | 60 | 0.2 | 0.182 | 1 | ok |
| sphere_D3_homo | KS | func_count | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 60 | 60 | 601 | 0.0488 | 0.732 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 60 | +0.000 [-0.080, +0.118] | 0.10 | 0.17 | +0.07 | 0 | 0 |
| ellipsoid_D3_homo | 60 | +0.135 [-0.058, +0.400] | 0.63 | 0.52 | -0.12 | 0 | 0 |
| multisensory_s1_D6_homo | 60 | +0.000 [-0.068, +0.124] | 0.95 | 0.95 | +0.00 | 0 | 0 |
| sphere_D3_hetero | 60 | +0.000 [-0.082, +0.137] | 0.48 | 0.50 | +0.02 | 0 | 0 |
| sphere_D3_homo | 60 | +0.116 [-0.005, +0.289] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 60 vs 60 runs that is a KS statistic of at least 0.333.

**no configuration flagged (15 tests)**
