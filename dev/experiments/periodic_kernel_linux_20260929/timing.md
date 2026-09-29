| configuration | ms/eval base | ms/eval new | ms/eval off | new/base, paired (same evals: n, median) | new/base, paired (all) | base/off | new/off |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 10.1 | 9.2 | 9.3 | 30, 0.93 | 0.93 | 1.09 | 0.99 |
| periodic_D3_hetero | 54.0 | 35.9 | 29.4 | 19, 0.69 | 0.67 | 1.84 | 1.22 |
| periodic_D3_homo | 51.0 | 33.5 | 22.6 | 13, 0.67 | 0.66 | 2.26 | 1.48 |
| periodic_D4 | 14.2 | 11.5 | 11.6 | 30, 0.81 | 0.81 | 1.22 | 0.99 |
| periodic_D6 | 19.3 | 13.3 | 14.3 | 27, 0.69 | 0.69 | 1.35 | 0.93 |
| periodic_rosenbrock_D4 | 19.0 | 14.9 | 19.5 | 25, 0.78 | 0.79 | 0.98 | 0.77 |

Time per evaluation against evaluations (least squares, ms per 100 evaluations):
| configuration | arm | slope | intercept at the median evals of off |
|---|---|---|---|
| periodic_D2 | base | 0.64 | 10.3 |
| periodic_D2 | new | -3.24 | 9.4 |
| periodic_D2 | off | -1.52 | 9.5 |
| periodic_D3_hetero | base | 5.86 | 48.3 |
| periodic_D3_hetero | new | 4.05 | 32.0 |
| periodic_D3_hetero | off | 6.59 | 29.3 |
| periodic_D3_homo | base | 7.61 | 37.6 |
| periodic_D3_homo | new | 5.18 | 25.5 |
| periodic_D3_homo | off | 6.12 | 21.5 |
| periodic_D4 | base | 4.01 | 14.3 |
| periodic_D4 | new | 4.37 | 11.7 |
| periodic_D4 | off | 1.77 | 11.6 |
| periodic_D6 | base | 2.86 | 19.2 |
| periodic_D6 | new | 0.62 | 13.3 |
| periodic_D6 | off | -0.60 | 14.2 |
| periodic_rosenbrock_D4 | base | 0.09 | 19.1 |
| periodic_rosenbrock_D4 | new | 1.88 | 14.2 |
| periodic_rosenbrock_D4 | off | 3.19 | 19.2 |
