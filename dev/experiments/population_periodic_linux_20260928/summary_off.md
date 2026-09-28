# Population off

- 180 runs: pybads 12cf2f29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 2c9cdfb

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 30 | 0 | 0.0093 [3.31e-08, 0.0093] | 52.5 [49, 57] | 0.47 | 0.001 | 0.5 [0.465, 0.574] |
| periodic_D3_hetero | 30 | 0 | 0.104 [0.0882, 0.139] | 310 [252, 338] | 0.50 | 0.1 | 8.46 [6.55, 9.88] |
| periodic_D3_homo | 30 | 0 | 0.0906 [0.0828, 0.101] | 207 [164, 266] | 0.70 | 0.1 | 4.56 [2.84, 6.61] |
| periodic_D4 | 30 | 0 | 0.0212 [1.07e-07, 0.0293] | 121 [114, 127] | 0.40 | 0.001 | 1.41 [1.28, 1.51] |
| periodic_D6 | 30 | 0 | 0.0541 [0.0153, 0.06] | 199 [181, 208] | 0.13 | 0.001 | 2.88 [2.71, 2.95] |
| periodic_rosenbrock_D4 | 30 | 0 | 1.63e-06 [6.33e-07, 5.28e-06] | 131 [125, 136] | 1.00 | 0.001 | 2.45 [2.11, 2.66] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
