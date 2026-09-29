# Population change

- 900 runs: pybads 58d922a1, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | 90 | 0 | 3.09e-06 [1.34e-06, 7.91e-06] | 118 [112, 124] | 1.00 | 0.001 | 4.04 [3.63, 4.45] |
| ellipsoid_D3_rerun | 90 | 0 | 3.57e-06 [1.5e-06, 8.38e-06] | 120 [114, 125] | 1.00 | 0.001 | 3.59 [3.1, 3.99] |
| rosenbrock_D6_other | 90 | 0 | 1.58e-06 [7.51e-07, 3.13e-06] | 416 [387, 444] | 0.84 | 0.001 | 8.27 [7.38, 8.79] |
| rosenbrock_D6_rerun | 90 | 0 | 1.88e-06 [7.95e-07, 5.7e-06] | 426 [397, 455] | 0.79 | 0.001 | 8.65 [7.77, 9.25] |
| sphere_D3_hetero_other | 90 | 0 | 0.0809 [0.0368, 0.152] | 357 [294, 389] | 0.59 | 0.1 | 11.4 [8.7, 12.9] |
| sphere_D3_hetero_rerun | 90 | 0 | 0.087 [0.0523, 0.149] | 326 [258, 402] | 0.60 | 0.1 | 9.58 [6.84, 12.5] |
| sphere_D3_homo_other | 90 | 0 | 0.00667 [0.00313, 0.00989] | 329 [269, 378] | 1.00 | 0.1 | 9.65 [7.8, 11.5] |
| sphere_D3_homo_rerun | 90 | 0 | 0.00863 [0.0042, 0.0215] | 290 [172, 338] | 1.00 | 0.1 | 7.86 [3.82, 10] |
| sphere_D3_other | 90 | 0 | 1.89e-07 [5.16e-08, 1.18e-06] | 77 [67, 82] | 1.00 | 0.001 | 2.05 [1.7, 2.33] |
| sphere_D3_rerun | 90 | 0 | 1.8e-06 [7.46e-07, 5.06e-06] | 68 [67, 78] | 1.00 | 0.001 | 1.73 [1.49, 2] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
