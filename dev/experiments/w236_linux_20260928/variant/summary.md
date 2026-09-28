# Population variant

- 450 runs: pybads 9a7c361e, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 90 | 0 | 0.316 [0.154, 0.523] | 302 [269, 322] | 0.14 | 0.1 | 17.2 [14.6, 19.4] |
| ellipsoid_D3_homo | 90 | 0 | 0.0807 [0.0356, 0.174] | 309 [290, 328] | 0.60 | 0.1 | 43.7 [37.6, 48.6] |
| multisensory_s1_D6_homo | 90 | 0 | 0.188 [0.112, 0.27] | 538 [476, 699] | 0.97 | 0.5 | 30.3 [25.2, 40.1] |
| sphere_D3_hetero | 90 | 0 | 0.0732 [0.0282, 0.144] | 324 [239, 396] | 0.61 | 0.1 | 24.2 [15.5, 31.9] |
| sphere_D3_homo | 90 | 0 | 0.0114 [0.00524, 0.0206] | 206 [182, 289] | 1.00 | 0.1 | 9.67 [6.52, 17.1] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
