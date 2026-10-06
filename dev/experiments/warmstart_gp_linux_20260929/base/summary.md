# Population base

- 900 runs: pybads ee0d9c29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | 90 | 0 | 3.5e-06 [1.36e-06, 8.69e-06] | 120 [114, 125] | 0.99 | 0.001 | 4.26 [3.63, 6.66] |
| ellipsoid_D3_rerun | 90 | 0 | 3.79e-06 [1.26e-06, 1.12e-05] | 117 [110, 124] | 1.00 | 0.001 | 4.06 [3.35, 6.75] |
| rosenbrock_D6_other | 90 | 0 | 1.47e-06 [7.27e-07, 3.54e-06] | 406 [377, 440] | 0.87 | 0.001 | 8.64 [7.48, 14.9] |
| rosenbrock_D6_rerun | 90 | 0 | 2.03e-06 [9.73e-07, 5.35e-06] | 394 [371, 420] | 0.83 | 0.001 | 8.44 [7.63, 14.1] |
| sphere_D3_hetero_other | 90 | 0 | 0.0682 [0.0348, 0.147] | 359 [302, 409] | 0.64 | 0.1 | 11.9 [9, 17.3] |
| sphere_D3_hetero_rerun | 90 | 0 | 0.0704 [0.036, 0.125] | 336 [274, 396] | 0.67 | 0.1 | 11.4 [7.68, 15.9] |
| sphere_D3_homo_other | 90 | 0 | 0.00663 [0.00381, 0.011] | 332 [268, 385] | 1.00 | 0.1 | 10.5 [7.75, 14.9] |
| sphere_D3_homo_rerun | 90 | 0 | 0.0106 [0.005, 0.0215] | 262 [163, 314] | 1.00 | 0.1 | 8.01 [4.21, 11.7] |
| sphere_D3_other | 90 | 0 | 1.79e-07 [7.1e-08, 1.06e-06] | 72 [67, 82] | 1.00 | 0.001 | 2.21 [1.61, 3.3] |
| sphere_D3_rerun | 90 | 0 | 1.58e-06 [5.92e-07, 5.28e-06] | 75 [68, 78] | 1.00 | 0.001 | 1.83 [1.44, 2.28] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
