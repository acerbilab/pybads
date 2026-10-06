# Population new_on

- 180 runs: pybads 9a705918, gpyreg 1.3.4.dev21+gb44634f2a from /home/user/gpyreg/gpyreg at 0f27db5

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 30 | 0 | 3.64e-08 [7.27e-09, 5.17e-08] | 54 [54, 56] | 1.00 | 0.001 | 0.507 [0.487, 0.547] |
| periodic_D3_hetero | 30 | 0 | 0.136 [0.107, 0.243] | 364 [302, 433] | 0.20 | 0.1 | 12.5 [9.77, 16.6] |
| periodic_D3_homo | 30 | 0 | 0.0264 [0.0205, 0.0461] | 348 [232, 404] | 0.97 | 0.1 | 11.4 [6.71, 14.9] |
| periodic_D4 | 30 | 0 | 1.92e-08 [1.19e-08, 4.49e-08] | 124 [113, 124] | 1.00 | 0.001 | 1.41 [1.28, 1.5] |
| periodic_D6 | 30 | 0 | 2.23e-08 [1.73e-08, 4.1e-08] | 203 [203, 203] | 1.00 | 0.001 | 2.74 [2.61, 2.79] |
| periodic_rosenbrock_D4 | 30 | 0 | 1.41e-07 [7.08e-08, 7.32e-07] | 168 [159, 176] | 1.00 | 0.001 | 2.49 [2.32, 2.63] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
