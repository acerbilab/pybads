# Compare pop (REF) and pop (NEW)

- REF, 286 runs: pybads 886bbff, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 12 runs: pybads b276da0, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- REF, 2 runs: pybads b276da0 (dirty), gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4
- NEW, 300 runs: pybads 73d5c28, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 60 | 60 | 0.183 | 0.267 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 60 | 60 | 0.367 | 0.000557 | 0.00613 | FLAG |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 764 | 0.266 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 60 | 60 | 0.2 | 0.182 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 60 | 60 | 0.45 | 7.57e-06 | 9.08e-05 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 60 | 60 | 618 | 0.0288 | 0.264 | ok |
| multisensory_s1_D6_homo | KS | true_error | 60 | 60 | 0.167 | 0.378 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 60 | 60 | 0.883 | 1.23e-24 | 1.85e-23 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 60 | 60 | 750 | 0.224 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 60 | 60 | 0.117 | 0.813 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 60 | 60 | 0.85 | 2.16e-22 | 3.03e-21 | FLAG |
| sphere_D3_hetero | signed-rank | log10 true_error | 60 | 60 | 856 | 0.664 | 1 | ok |
| sphere_D3_homo | KS | true_error | 60 | 60 | 0.2 | 0.182 | 1 | ok |
| sphere_D3_homo | KS | func_count | 60 | 60 | 0.767 | 1.39e-17 | 1.8e-16 | FLAG |
| sphere_D3_homo | signed-rank | log10 true_error | 60 | 60 | 547 | 0.0264 | 0.264 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 60 | +0.029 [-0.147, +0.271] | 0.10 | 0.10 | +0.00 | 0 | 0 |
| ellipsoid_D3_homo | 60 | +0.176 [-0.060, +0.312] | 0.63 | 0.50 | -0.13 | 0 | 0 |
| multisensory_s1_D6_homo | 60 | +0.079 [-0.065, +0.153] | 0.95 | 0.93 | -0.02 | 0 | 0 |
| sphere_D3_hetero | 60 | +0.036 [-0.113, +0.123] | 0.48 | 0.50 | +0.02 | 0 | 0 |
| sphere_D3_homo | 60 | +0.123 [+0.000, +0.298] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 60 vs 60 runs that is a KS statistic of at least 0.333.

**5 configuration(s) flagged: ['ellipsoid_D3_hetero', 'ellipsoid_D3_homo', 'multisensory_s1_D6_homo', 'sphere_D3_hetero', 'sphere_D3_homo']**
