| configuration | seeds | own A (s) | own B (s) | B / A of medians | median of B / A per seed [95 % CI] | B faster | evals A | evals B | own per eval A (ms) | own per eval B (ms) | error A | error B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ackley_D6 | 30 | 4.85 | 2.62 | 0.540 | 0.546 [0.527, 0.564] | 30/30 | 394 | 382 | 12.2 | 6.8 | 0.000386 | 0.00017 |
| ellipsoid_D10 | 30 | 18.64 | 9.77 | 0.524 | 0.553 [0.458, 0.643] | 30/30 | 705 | 672 | 24.1 | 14.6 | 6.77e-05 | 4.14e-07 |
| ellipsoid_D3 | 30 | 1.91 | 1.64 | 0.860 | 0.810 [0.714, 0.949] | 22/30 | 147 | 138 | 12.8 | 11.6 | 2.39e-05 | 1.76e-06 |
| ellipsoid_D3_homo | 30 | 9.84 | 6.56 | 0.666 | 0.667 [0.610, 0.736] | 29/30 | 370 | 310 | 26.8 | 21.7 | 0.107 | 0.0865 |
| multisensory_s1_D6_homo | 30 | 14.12 | 7.59 | 0.538 | 0.530 [0.472, 0.570] | 28/30 | 626 | 554 | 22.8 | 14.0 | 0.18 | 0.194 |
| rosenbrock_D6 | 30 | 7.82 | 3.72 | 0.476 | 0.460 [0.421, 0.532] | 30/30 | 468 | 424 | 17.0 | 8.6 | 4.61e-05 | 2.85e-06 |
| sphere_D3_hetero | 30 | 7.96 | 4.10 | 0.515 | 0.564 [0.453, 0.678] | 26/30 | 401 | 338 | 19.4 | 12.5 | 0.2 | 0.0985 |

Own time: the wall time of `optimize()` less the target's evaluations. Summed over every run: A 1992 s, B 1120 s, B / A 0.562.
Load of the whole machine during a run (all logical CPUs): median 11.2 %, largest 20.3 %.
