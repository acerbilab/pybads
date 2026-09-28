# Population pop

- 450 runs: pybads 0cc795f, gpyreg 1.3.4.dev11+g1893effc9 from /home/user/gpyreg-v1.3.3/gpyreg at 98ab5a4

| config | runs | crashed | true_error | func_count | solved | tolerance | wall s |
|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 90 | 0 | 0.336 [0.128, 0.549] | 298 [274, 317] | 0.16 | 0.1 | 8.09 [6.83, 8.49] |
| ellipsoid_D3_homo | 90 | 0 | 0.0757 [0.035, 0.173] | 313 [288, 329] | 0.62 | 0.1 | 12.6 [11.5, 13.6] |
| multisensory_s1_D6_homo | 90 | 0 | 0.178 [0.125, 0.266] | 502 [462, 596] | 0.97 | 0.5 | 13 [11.9, 16.5] |
| sphere_D3_hetero | 90 | 0 | 0.0862 [0.0341, 0.162] | 322 [217, 372] | 0.54 | 0.1 | 6.96 [4.16, 9.13] |
| sphere_D3_homo | 90 | 0 | 0.0131 [0.00603, 0.0213] | 212 [190, 327] | 1.00 | 0.1 | 4.03 [2.84, 7.8] |

true_error, func_count and wall time: median [interquartile range] over the runs that did not crash; solved: fraction of all runs with true_error below the tolerance.
