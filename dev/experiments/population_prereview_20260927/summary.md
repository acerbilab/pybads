# Population population_prereview_20260927

- 510 runs: pybads ab4dded, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4
- 1290 runs: pybads ab4ddedc, gpyreg 1.3.3 from C:\Users\luigi\Documents\GitHub\pybads\dev\scripts\runs\gpyreg\v1.3.3\gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 100 | 0 | 0.000344 [0.000227, 0.000492] | 400 [388, 412] | 1.00 | 0.5 | 12.3 [5.97, 14.3] |
| ellipsoid_D10 | 100 | 0 | 7.33e-07 [3.05e-07, 1.84e-06] | 629 [609, 649] | 1.00 | 0.001 | 35.1 [18.6, 41.8] |
| ellipsoid_D3 | 100 | 0 | 2.43e-06 [5.37e-07, 1.16e-05] | 142 [131, 150] | 1.00 | 0.001 | 3.35 [2.2, 3.93] |
| ellipsoid_D3_hetero | 100 | 0 | 0.289 [0.143, 0.587] | 366 [331, 397] | 0.19 | 0.1 | 20.2 [12.2, 25.3] |
| ellipsoid_D3_homo | 100 | 0 | 0.0758 [0.0399, 0.142] | 380 [353, 405] | 0.64 | 0.1 | 32 [19.5, 36.7] |
| ellipsoid_D3_unbounded | 100 | 0 | 3.15e-06 [9.61e-07, 1.01e-05] | 147 [137, 155] | 1.00 | 0.001 | 5.18 [3.37, 6.66] |
| ellipsoid_D6 | 100 | 0 | 1.32e-07 [5.9e-08, 5.44e-07] | 348 [335, 366] | 1.00 | 0.001 | 11.8 [7.24, 14] |
| multisensory_s1_D6 | 100 | 0 | 2.47e-07 [1.24e-07, 5.85e-07] | 276 [262, 288] | 1.00 | 0.5 | 9.91 [3.99, 12.3] |
| multisensory_s1_D6_homo | 100 | 0 | 0.197 [0.105, 0.277] | 668 [598, 752] | 0.96 | 0.5 | 37.8 [21.5, 48.7] |
| rastrigin_D3 | 100 | 0 | 4.48 [1.99, 5.97] | 148 [135, 157] | 0.01 | 0.5 | 2.22 [1.54, 2.68] |
| rosenbrock_D2 | 100 | 0 | 7.39e-06 [1.66e-06, 3.54e-05] | 95 [83, 103] | 1.00 | 0.001 | 1.81 [1.22, 2.78] |
| rosenbrock_D6 | 100 | 0 | 2.23e-06 [9.62e-07, 3.97] | 443 [390, 491] | 0.72 | 0.001 | 16.3 [9.65, 19.7] |
| sphere_D10 | 100 | 0 | 7.03e-08 [4.06e-08, 1.01e-07] | 454 [449, 478] | 1.00 | 0.001 | 14.4 [8.12, 16.3] |
| sphere_D2 | 100 | 0 | 5.49e-07 [2.12e-07, 1.49e-06] | 55 [55, 59] | 1.00 | 0.001 | 0.756 [0.668, 0.816] |
| sphere_D3_hetero | 100 | 0 | 0.108 [0.0672, 0.185] | 376 [302, 450] | 0.46 | 0.1 | 18.5 [11.3, 25.3] |
| sphere_D3_homo | 100 | 0 | 0.0078 [0.00356, 0.0169] | 313 [218, 390] | 1.00 | 0.1 | 11.5 [8.71, 24.6] |
| sphere_nonbox_D3 | 100 | 0 | 9.14e-06 [3.7e-06, 2.86e-05] | 97 [93, 102] | 1.00 | 0.001 | 2.63 [1.46, 3.98] |
| timing_D5 | 100 | 0 | 2.32e-07 [9.52e-08, 4.73e-07] | 215 [194, 270] | 1.00 | 0.5 | 34.8 [17.7, 47.6] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
