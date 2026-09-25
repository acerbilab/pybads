# Compare population_baseline_20260924 (REF) and control_50D_20260924 (NEW)

- REF, 540 runs: pybads 2226883, gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.1\gpyreg at 1dbbfc5
- NEW, 27 runs: pybads 2226883, gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.1\gpyreg at 1dbbfc5
- NEW, 63 runs: pybads 2226883 (dirty), gpyreg 1.3.1 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.1\gpyreg at 1dbbfc5

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| ellipsoid_D10 | KS | true_error | 29 | 30 | 1 | 3.38e-17 | 2.37e-16 | FLAG |
| ellipsoid_D10 | KS | func_count | 29 | 30 | 1 | 3.38e-17 | 2.37e-16 | FLAG |
| ellipsoid_D10 | signed-rank | log10 true_error | 29 | 29 | 0 | 3.73e-09 | 1.86e-08 | FLAG |
| multisensory_s1_D6_homo | KS | true_error | 30 | 30 | 0.2 | 0.594 | 0.926 | ok |
| multisensory_s1_D6_homo | KS | func_count | 30 | 30 | 1 | 1.69e-17 | 1.52e-16 | FLAG |
| multisensory_s1_D6_homo | signed-rank | log10 true_error | 30 | 30 | 143 | 0.172 | 0.687 | ok |
| sphere_D3_homo | KS | true_error | 30 | 30 | 0.233 | 0.393 | 0.926 | ok |
| sphere_D3_homo | KS | func_count | 30 | 30 | 1 | 1.69e-17 | 1.52e-16 | FLAG |
| sphere_D3_homo | signed-rank | log10 true_error | 30 | 30 | 182 | 0.309 | 0.926 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| ellipsoid_D10 | 29 | +1.592 [+1.278, +1.728] | 0.97 | 0.30 | -0.67 | 1 | 0 |
| multisensory_s1_D6_homo | 30 | +0.002 [-0.014, +0.104] | 0.93 | 0.87 | -0.07 | 0 | 0 |
| sphere_D3_homo | 30 | -0.057 [-0.287, +0.094] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Only in REF: ['ackley_D6', 'ellipsoid_D3', 'ellipsoid_D3_hetero', 'ellipsoid_D3_homo', 'ellipsoid_D3_unbounded', 'ellipsoid_D6', 'multisensory_s1_D6', 'rastrigin_D3', 'rosenbrock_D2', 'rosenbrock_D6', 'sphere_D10', 'sphere_D2', 'sphere_D3_hetero', 'sphere_nonbox_D3', 'timing_D5']; only in NEW: [].

Holm family: 9 tests at alpha 0.05. A flag needs p <= 0.0056 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**3 configuration(s) flagged: ['ellipsoid_D10', 'multisensory_s1_D6_homo', 'sphere_D3_homo']**
