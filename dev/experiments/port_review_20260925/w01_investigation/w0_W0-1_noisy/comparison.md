# Compare population_gpfixes_20260925 (REF) and w0_W0-1_noisy (NEW)

- REF, 540 runs: pybads ab4dded, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4
- NEW, 150 runs: pybads d6e3f61, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | KS | true_error | 30 | 30 | 0.133 | 0.958 | 1 | ok |
| ellipsoid_D3_hetero | KS | func_count | 30 | 30 | 0.467 | 0.00253 | 0.0329 | FLAG |
| ellipsoid_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 212 | 0.685 | 1 | ok |
| ellipsoid_D3_homo | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| ellipsoid_D3_homo | KS | func_count | 30 | 30 | 0.667 | 1.28e-06 | 1.91e-05 | FLAG |
| ellipsoid_D3_homo | signed-rank | log10 true_error | 30 | 30 | 107 | 0.0169 | 0.203 | ok |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 0.633 | 5.8e-06 | 8.11e-05 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 173 | 0.229 | 1 | ok |
| sphere_D3_hetero | KS | true_error | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| sphere_D3_hetero | KS | func_count | 30 | 30 | 0.233 | 0.393 | 1 | ok |
| sphere_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 196 | 0.642 | 1 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 0.3 | 0.135 | 1 | ok |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 186 | 0.943 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 30 | +0.016 [-0.209, +0.309] | 0.23 | 0.17 | -0.07 | 0 | 0 |
| ellipsoid_D3_homo | 30 | +0.156 [-0.000, +0.494] | 0.73 | 0.50 | -0.23 | 0 | 0 |
| multisensory_s1_D6_homo | 30 | +0.102 [-0.032, +0.219] | 0.90 | 0.80 | -0.10 | 0 | 0 |
| sphere_D3_hetero | 30 | -0.017 [-0.170, +0.066] | 0.53 | 0.57 | +0.03 | 0 | 0 |
| sphere_D3_homo | 30 | +0.000 [-0.094, +0.045] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Only in REF: ['ackley_D6', 'ellipsoid_D10', 'ellipsoid_D3', 'ellipsoid_D3_unbounded', 'ellipsoid_D6', 'multisensory_s1_D6', 'rastrigin_D3', 'rosenbrock_D2', 'rosenbrock_D6', 'sphere_D10', 'sphere_D2', 'sphere_nonbox_D3', 'timing_D5']; only in NEW: [].

Holm family: 15 tests at alpha 0.05. A flag needs p <= 0.0033 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**3 configuration(s) flagged: ['ellipsoid_D3_hetero', 'ellipsoid_D3_homo', 'multisensory_s1_D6_homo']**
