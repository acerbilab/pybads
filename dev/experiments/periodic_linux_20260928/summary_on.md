# Population on

- 180 runs: pybads 66ef459c (dirty), gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 3f1a732

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 30 | 0 | 3.08e-08 [8.61e-09, 4.52e-08] | 54 [54, 56] | 1.00 | 0.001 | 0.555 [0.526, 0.61] |
| periodic_D3_hetero | 30 | 0 | 0.136 [0.0824, 0.237] | 374 [345, 496] | 0.27 | 0.1 | 20.2 [16.1, 29.3] |
| periodic_D3_homo | 30 | 0 | 0.0243 [0.0175, 0.0452] | 359 [232, 414] | 0.97 | 0.1 | 18.8 [9.58, 22.6] |
| periodic_D4 | 30 | 0 | 2.43e-08 [1.06e-08, 4.23e-08] | 124 [113, 124] | 1.00 | 0.001 | 1.9 [1.72, 2.34] |
| periodic_D6 | 30 | 0 | 3.24e-08 [1.46e-08, 5.63e-08] | 203 [203, 203] | 1.00 | 0.001 | 3.85 [3.76, 3.99] |
| periodic_rosenbrock_D4 | 30 | 0 | 1.46e-07 [8.36e-08, 4.71e-07] | 168 [160, 175] | 1.00 | 0.001 | 3.2 [3.02, 3.38] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
