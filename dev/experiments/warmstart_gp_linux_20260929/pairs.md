| config | pairs | changed | median error ref → new | median evaluations ref → new | solved ref → new | changed runs: median log10 error ratio | signed-rank p | median change in evaluations | crashed ref, new |
|---|---|---|---|---|---|---|---|---|---|
| `ellipsoid_D3_other` | 90 | 90 | 3.5e-06 → 3.09e-06 | 119.5 → 118 | 0.99 → 1.00 | +0.080 | 0.761 | +0.0 | 0, 0 |
| `ellipsoid_D3_rerun` | 90 | 90 | 3.79e-06 → 3.57e-06 | 117 → 120 | 1.00 → 1.00 | -0.036 | 0.970 | +2.0 | 0, 0 |
| `rosenbrock_D6_other` | 90 | 90 | 1.47e-06 → 1.58e-06 | 406.5 → 416.5 | 0.87 → 0.84 | +0.018 | 0.680 | +12.0 | 0, 0 |
| `rosenbrock_D6_rerun` | 90 | 90 | 2.03e-06 → 1.88e-06 | 394 → 426 | 0.83 → 0.79 | -0.050 | 0.973 | +29.0 | 0, 0 |
| `sphere_D3_hetero_other` | 90 | 90 | 0.0682 → 0.0809 | 359 → 357 | 0.64 → 0.59 | +0.021 | 0.899 | -8.5 | 0, 0 |
| `sphere_D3_hetero_rerun` | 90 | 90 | 0.0704 → 0.087 | 335.5 → 326 | 0.67 → 0.60 | +0.014 | 0.381 | -16.5 | 0, 0 |
| `sphere_D3_homo_other` | 90 | 90 | 0.00663 → 0.00667 | 332 → 329 | 1.00 → 1.00 | -0.057 | 0.420 | -2.0 | 0, 0 |
| `sphere_D3_homo_rerun` | 90 | 90 | 0.0106 → 0.00863 | 262 → 290 | 1.00 → 1.00 | -0.087 | 0.625 | +3.5 | 0, 0 |
| `sphere_D3_other` | 90 | 90 | 1.79e-07 → 1.89e-07 | 72 → 77 | 1.00 → 1.00 | +0.009 | 0.918 | +0.0 | 0, 0 |
| `sphere_D3_rerun` | 90 | 90 | 1.58e-06 → 1.8e-06 | 75 → 68 | 1.00 → 1.00 | -0.141 | 0.817 | +0.0 | 0, 0 |

| config | pairs | changed | solved ref | solved new | new only | ref only | McNemar p | paired difference [95% CI] | median log10 error ratio, changed runs | signed-rank p, changed runs |
|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_other | 90 | 90 | 0.99 | 1.00 | 1 | 0 | 1.000 | +0.011 [+0.000, +0.033] | +0.080 | 0.761 |
| ellipsoid_D3_rerun | 90 | 90 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.036 | 0.970 |
| rosenbrock_D6_other | 90 | 90 | 0.87 | 0.84 | 10 | 12 | 0.832 | -0.022 [-0.122, +0.078] | +0.018 | 0.680 |
| rosenbrock_D6_rerun | 90 | 90 | 0.83 | 0.79 | 12 | 16 | 0.572 | -0.044 [-0.156, +0.067] | -0.050 | 0.973 |
| sphere_D3_hetero_other | 90 | 90 | 0.64 | 0.59 | 14 | 19 | 0.487 | -0.056 [-0.178, +0.067] | +0.021 | 0.899 |
| sphere_D3_hetero_rerun | 90 | 90 | 0.67 | 0.60 | 16 | 22 | 0.418 | -0.067 [-0.200, +0.067] | +0.014 | 0.381 |
| sphere_D3_homo_other | 90 | 90 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.057 | 0.420 |
| sphere_D3_homo_rerun | 90 | 90 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.087 | 0.625 |
| sphere_D3_other | 90 | 90 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | +0.009 | 0.918 |
| sphere_D3_rerun | 90 | 90 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | -0.141 | 0.817 |
| all | 900 | 900 | 0.90 | 0.88 | 53 | 69 | 0.174 | -0.018 [-0.041, +0.007] | -0.020 | 0.983 |

The paired difference is NEW minus REF; its interval is the percentile interval of 10000 bootstrap resamples of the pairs, within each configuration.
