# Population pop

- 450 runs: pybads 58e7dd5, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 90 | 0 | 0.311 [0.15, 0.511] | 302 [270, 322] | 0.16 | 0.1 | 7.72 [6.79, 8.61] |
| ellipsoid_D3_homo | 90 | 0 | 0.0781 [0.0335, 0.168] | 308 [289, 328] | 0.61 | 0.1 | 12.7 [11.5, 13.7] |
| multisensory_s1_D6_homo | 90 | 0 | 0.188 [0.112, 0.27] | 538 [476, 699] | 0.97 | 0.5 | 14.6 [12.3, 20.5] |
| sphere_D3_hetero | 90 | 0 | 0.103 [0.0556, 0.152] | 322 [252, 380] | 0.50 | 0.1 | 7.33 [5.6, 9.43] |
| sphere_D3_homo | 90 | 0 | 0.0111 [0.00546, 0.0206] | 206 [180, 305] | 1.00 | 0.1 | 3.64 [2.74, 7.02] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
