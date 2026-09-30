# Population release140_periodic_20260930

- 180 runs: pybads 60ad9e0f, gpyreg 1.3.4.dev35+g3e56dce0f from /home/user/pybads/dev/scripts/runs/gpyreg/main_3e56dce/gpyreg at 3e56dce

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 30 | 0 | 3.64e-08 [7.27e-09, 5.17e-08] | 54 [54, 56] | 1.00 | 0.001 | 0.375 [0.36, 0.398] |
| periodic_D3_hetero | 30 | 0 | 0.136 [0.107, 0.243] | 364 [302, 433] | 0.20 | 0.1 | 7.23 [5.95, 9.36] |
| periodic_D3_homo | 30 | 0 | 0.0264 [0.0205, 0.0461] | 348 [232, 404] | 0.97 | 0.1 | 6.57 [3.72, 8.89] |
| periodic_D4 | 30 | 0 | 1.92e-08 [1.19e-08, 4.49e-08] | 124 [113, 124] | 1.00 | 0.001 | 0.915 [0.865, 0.976] |
| periodic_D6 | 30 | 0 | 2.23e-08 [1.73e-08, 4.1e-08] | 203 [203, 203] | 1.00 | 0.001 | 1.76 [1.72, 1.84] |
| periodic_rosenbrock_D4 | 30 | 0 | 1.41e-07 [7.08e-08, 7.32e-07] | 168 [159, 176] | 1.00 | 0.001 | 1.62 [1.56, 1.72] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
