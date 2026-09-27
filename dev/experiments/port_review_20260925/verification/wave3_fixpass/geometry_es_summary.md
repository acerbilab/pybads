# Population geo_es_c276d79

- 210 runs: pybads c276d79, gpyreg 1.3.4.dev10+gd96d0d9f7 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| edgesphere_D2 | 30 | 0 | 3.34e-08 [1.13e-09, 4.43e-08] | 47 [47, 47] | 1.00 | 0.001 | 0.474 [0.431, 0.554] |
| edgesphere_D3_homo | 30 | 0 | 0.0112 [0.00274, 0.0273] | 212 [189, 250] | 0.97 | 0.1 | 4.1 [3.46, 5.59] |
| edgesphere_D4 | 30 | 0 | 7.96e-07 [3.54e-07, 1.22e-06] | 93 [90, 100] | 1.00 | 0.001 | 1.05 [0.891, 1.13] |
| ridge_D2 | 30 | 0 | 0.000115 [5.55e-05, 0.000536] | 180 [171, 195] | 0.83 | 0.001 | 1.85 [1.65, 1.99] |
| ridge_D4 | 30 | 0 | 0.000265 [0.000188, 0.000342] | 360 [340, 372] | 0.90 | 0.001 | 4.67 [4.12, 4.85] |
| sphere_band_D2 | 30 | 0 | 15.1 [6.09, 28] | 2 [2, 2] | 0.00 | 0.001 | 0.0486 [0.0464, 0.0511] |
| sphere_band_D3 | 30 | 0 | 8.55e-06 [4.78e-06, 1.12e-05] | 60 [55, 63] | 1.00 | 0.001 | 0.656 [0.479, 0.767] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
