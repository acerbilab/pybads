# Population on

- 180 runs: pybads 12cf2f29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 2c9cdfb

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 30 | 0 | 3.61e-08 [1.37e-08, 7.23e-08] | 54 [54, 56] | 1.00 | 0.001 | 0.535 [0.512, 0.592] |
| periodic_D3_hetero | 30 | 0 | 0.147 [0.085, 0.25] | 368 [321, 437] | 0.27 | 0.1 | 19 [14.7, 25.3] |
| periodic_D3_homo | 30 | 0 | 0.0284 [0.0175, 0.0467] | 382 [263, 455] | 0.97 | 0.1 | 19.7 [12.1, 24.8] |
| periodic_D4 | 30 | 0 | 2.8e-08 [1.53e-08, 4.5e-08] | 124 [113, 124] | 1.00 | 0.001 | 1.73 [1.51, 1.82] |
| periodic_D6 | 30 | 0 | 3.97e-08 [2.24e-08, 5.09e-08] | 203 [203, 205] | 1.00 | 0.001 | 3.9 [3.82, 4.06] |
| periodic_rosenbrock_D4 | 30 | 0 | 3.35e-07 [9.51e-08, 8.73e-07] | 168 [161, 175] | 1.00 | 0.001 | 3.18 [3.08, 3.43] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
