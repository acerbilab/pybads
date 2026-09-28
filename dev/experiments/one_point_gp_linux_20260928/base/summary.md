# Population base

- 330 runs: pybads 07280f79, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | 30 | 0 | 1.65e-09 [1.13e-09, 4.43e-08] | 48 [46.2, 49] | 1.00 | 0.001 | 0.511 [0.434, 0.573] |
| edgesphere_D3_homo | 30 | 0 | 0.00558 [0.000914, 0.0121] | 218 [207, 290] | 1.00 | 0.1 | 4.24 [3.64, 6.31] |
| edgesphere_D4 | 30 | 0 | 6.67e-07 [2.56e-07, 1.16e-06] | 92 [91, 97] | 1.00 | 0.001 | 0.931 [0.885, 0.98] |
| ridge_D2 | 30 | 0 | 0.000161 [9.31e-05, 0.000353] | 186 [176, 196] | 0.90 | 0.001 | 1.71 [1.61, 1.85] |
| ridge_D4 | 30 | 0 | 0.000273 [0.000135, 0.000511] | 370 [350, 385] | 0.87 | 0.001 | 4.22 [3.76, 4.58] |
| sphere_band_D2 | 30 | 0 | 15.1 [6.09, 28] | 2 [2, 2] | 0.00 | 0.001 | 0.0533 [0.0508, 0.0567] |
| sphere_band_D2_hetero | 30 | 0 | 13.4 [6.09, 26.6] | 94.5 [51, 120] | 0.00 | 0.1 | 0.99 [0.506, 1.39] |
| sphere_band_D2_homo | 30 | 0 | 7.99 [4.27, 24.2] | 116 [89.5, 212] | 0.03 | 0.1 | 1.24 [0.898, 3.19] |
| sphere_band_D3 | 30 | 0 | 9.7e-06 [6.4e-06, 1.42e-05] | 57 [53, 60] | 1.00 | 0.001 | 0.582 [0.507, 0.654] |
| sphere_band_D3_hetero | 30 | 0 | 0.143 [0.0376, 0.293] | 240 [179, 292] | 0.33 | 0.1 | 4.23 [2.74, 5.71] |
| sphere_band_D3_homo | 30 | 0 | 0.0331 [0.0144, 0.0661] | 217 [174, 284] | 0.87 | 0.1 | 3.29 [2.22, 5.05] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
