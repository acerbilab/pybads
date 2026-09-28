# Population change

- 330 runs: pybads d9772a04, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | 30 | 0 | 1.65e-09 [1.13e-09, 4.43e-08] | 48 [46.2, 49] | 1.00 | 0.001 | 0.46 [0.41, 0.543] |
| edgesphere_D3_homo | 30 | 0 | 0.00558 [0.000914, 0.0121] | 218 [207, 290] | 1.00 | 0.1 | 4.16 [3.61, 6.05] |
| edgesphere_D4 | 30 | 0 | 6.67e-07 [2.56e-07, 1.16e-06] | 92 [91, 97] | 1.00 | 0.001 | 0.922 [0.875, 0.963] |
| ridge_D2 | 30 | 0 | 0.000161 [9.31e-05, 0.000353] | 186 [176, 196] | 0.90 | 0.001 | 1.66 [1.56, 1.74] |
| ridge_D4 | 30 | 0 | 0.000273 [0.000135, 0.000511] | 370 [350, 385] | 0.87 | 0.001 | 4.2 [3.86, 4.46] |
| sphere_band_D2 | 30 | 0 | 15.1 [6.09, 28] | 2 [2, 2] | 0.00 | 0.001 | 0.0282 [0.0274, 0.0296] |
| sphere_band_D2_hetero | 30 | 0 | 13.9 [6.1, 26.8] | 72 [32, 121] | 0.00 | 0.1 | 0.722 [0.3, 1.39] |
| sphere_band_D2_homo | 30 | 0 | 6.45 [2.82, 22.3] | 124 [102, 182] | 0.03 | 0.1 | 1.48 [1.1, 2.42] |
| sphere_band_D3 | 30 | 0 | 9.94e-06 [5.78e-06, 1.24e-05] | 56.5 [52, 61.5] | 1.00 | 0.001 | 0.535 [0.482, 0.645] |
| sphere_band_D3_hetero | 30 | 0 | 0.179 [0.0482, 0.399] | 235 [179, 306] | 0.43 | 0.1 | 4.59 [2.85, 6.31] |
| sphere_band_D3_homo | 30 | 0 | 0.0188 [0.0113, 0.0585] | 244 [206, 293] | 0.90 | 0.1 | 4.05 [3.08, 5.07] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
