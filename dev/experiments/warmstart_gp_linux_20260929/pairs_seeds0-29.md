| config | pairs | changed | median error ref → new | median evaluations ref → new | solved ref → new | changed runs: median log10 error ratio | signed-rank p | median change in evaluations | crashed ref, new |
|---|---|---|---|---|---|---|---|---|---|
| `ellipsoid_D3_other` | 30 | 30 | 2.89e-06 → 1.97e-06 | 120 → 118 | 1.00 → 1.00 | -0.331 | 0.503 | +3.0 | 0, 0 |
| `ellipsoid_D3_rerun` | 30 | 30 | 3.95e-06 → 3.91e-06 | 117.5 → 120 | 1.00 → 1.00 | -0.069 | 0.777 | +1.0 | 0, 0 |
| `rosenbrock_D6_other` | 30 | 30 | 1.21e-06 → 1.71e-06 | 417 → 417 | 0.93 → 0.77 | +0.122 | 0.080 | +12.0 | 0, 0 |
| `rosenbrock_D6_rerun` | 30 | 30 | 2.27e-06 → 2.18e-06 | 398.5 → 437 | 0.80 → 0.80 | -0.029 | 0.968 | +35.0 | 0, 0 |
| `sphere_D3_hetero_other` | 30 | 30 | 0.0582 → 0.0853 | 384 → 358.5 | 0.73 → 0.57 | -0.018 | 0.556 | -10.0 | 0, 0 |
| `sphere_D3_hetero_rerun` | 30 | 30 | 0.0744 → 0.0963 | 362 → 353 | 0.60 → 0.50 | +0.056 | 0.570 | -13.5 | 0, 0 |
| `sphere_D3_homo_other` | 30 | 30 | 0.00777 → 0.00624 | 307 → 339.5 | 1.00 → 1.00 | -0.023 | 0.824 | -2.5 | 0, 0 |
| `sphere_D3_homo_rerun` | 30 | 30 | 0.00823 → 0.00904 | 240 → 296.5 | 1.00 → 1.00 | -0.012 | 0.570 | +7.5 | 0, 0 |
| `sphere_D3_other` | 30 | 30 | 1.74e-07 → 3.02e-07 | 74.5 → 72 | 1.00 → 1.00 | +0.134 | 0.777 | +0.0 | 0, 0 |
| `sphere_D3_rerun` | 30 | 30 | 1.49e-06 → 1.29e-06 | 75 → 68 | 1.00 → 1.00 | -0.258 | 0.360 | +0.0 | 0, 0 |

| config | pairs | changed | solved ref | solved new | new only | ref only | McNemar p | paired difference [95% CI] | median log10 error ratio, changed runs | signed-rank p, changed runs |
|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | 30 | 30 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.331 | 0.503 |
| ellipsoid_D3_rerun | 30 | 30 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.069 | 0.777 |
| rosenbrock_D6_other | 30 | 30 | 0.93 | 0.77 | 1 | 6 | 0.125 | -0.167 [-0.333, +0.000] | +0.122 | 0.080 |
| rosenbrock_D6_rerun | 30 | 30 | 0.80 | 0.80 | 5 | 5 | 1.000 | +0.000 [-0.200, +0.200] | -0.029 | 0.968 |
| sphere_D3_hetero_other | 30 | 30 | 0.73 | 0.57 | 3 | 8 | 0.227 | -0.167 [-0.367, +0.033] | -0.018 | 0.556 |
| sphere_D3_hetero_rerun | 30 | 30 | 0.60 | 0.50 | 5 | 8 | 0.581 | -0.100 [-0.333, +0.133] | +0.056 | 0.570 |
| sphere_D3_homo_other | 30 | 30 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.023 | 0.824 |
| sphere_D3_homo_rerun | 30 | 30 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.012 | 0.570 |
| sphere_D3_other | 30 | 30 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | +0.134 | 0.777 |
| sphere_D3_rerun | 30 | 30 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.258 | 0.360 |
| all | 300 | 300 | 0.91 | 0.86 | 14 | 27 | 0.060 | -0.043 [-0.083, -0.003] | -0.007 | 0.705 |

The paired difference is NEW minus REF; its interval is the percentile interval of 10000 bootstrap resamples of the pairs, within each configuration.
