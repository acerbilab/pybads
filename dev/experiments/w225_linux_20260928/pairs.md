| config | pairs | changed | solved ref | solved new | new only | ref only | McNemar p | paired difference [95% CI] | median log10 error ratio, changed runs | signed-rank p, changed runs |
|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid_D3_hetero | 90 | 89 | 0.16 | 0.16 | 8 | 8 | 1.000 | +0.000 [-0.089, +0.089] | +0.000 | 0.897 |
| ellipsoid_D3_homo | 90 | 88 | 0.62 | 0.61 | 18 | 19 | 1.000 | -0.011 [-0.144, +0.122] | +0.000 | 0.893 |
| multisensory_s1_D6_homo | 90 | 88 | 0.97 | 0.97 | 1 | 1 | 1.000 | +0.000 [-0.033, +0.033] | -0.003 | 0.280 |
| sphere_D3_hetero | 90 | 90 | 0.54 | 0.50 | 14 | 18 | 0.597 | -0.044 [-0.167, +0.078] | -0.007 | 0.991 |
| sphere_D3_homo | 90 | 84 | 1.00 | 1.00 | 0 | 0 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 | 0.197 |
| all | 450 | 439 | 0.66 | 0.65 | 41 | 46 | 0.668 | -0.011 [-0.051, +0.029] | +0.000 | 0.315 |

The paired difference is NEW minus REF; its interval is the percentile interval of 10000 bootstrap resamples of the pairs, within each configuration.

| config | seeds | solved ref | solved new |
|---|---|---|---|
| ellipsoid_D3_hetero | 0-29 | 0.13 | 0.23 |
| ellipsoid_D3_hetero | 30-89 | 0.17 | 0.12 |
| ellipsoid_D3_homo | 0-29 | 0.60 | 0.43 |
| ellipsoid_D3_homo | 30-89 | 0.63 | 0.70 |
| multisensory_s1_D6_homo | 0-29 | 1.00 | 1.00 |
| multisensory_s1_D6_homo | 30-89 | 0.95 | 0.95 |
| sphere_D3_hetero | 0-29 | 0.60 | 0.60 |
| sphere_D3_hetero | 30-89 | 0.52 | 0.45 |
| sphere_D3_homo | 0-29 | 1.00 | 1.00 |
| sphere_D3_homo | 30-89 | 1.00 | 1.00 |

Reproduced: 150 of 150 runs of NEW equal the records of REPRO_DIR (every field of final but wall_s).
