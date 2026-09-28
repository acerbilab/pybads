| config | runs | initial evaluations | median f - f_min | median yval - f | median abs(yval - f) | median GP mean - f | median abs(GP mean - f) | GP closer | median SD of yval | median GP SD | yval within 2 SD | GP within 2 SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sphere_D3_homo | 90 | 33 | 2.88 | -0.34 | 0.78 | -0.39 | 0.45 | 0.73 | 1 | 0.477 | 0.94 | 0.81 |
| ellipsoid_D3_homo | 90 | 33 | 1.69e+04 | +0.09 | 0.63 | +0.06 | 0.76 | 0.36 | 1 | 0 | 0.94 | 0.44 |
| sphere_D3_hetero | 90 | 33 | 3.89 | -2.88 | 2.88 | -1.30 | 1.55 | 0.84 | 2.97 | 2.22 | 0.90 | 0.91 |
| ellipsoid_D3_hetero | 90 | 33 | 1.69e+04 | +11.23 | 62.92 | +8.64 | 71.30 | 0.43 | 131 | 130 | 0.94 | 0.94 |
| multisensory_s1_D6_homo | 90 | 33 | 48.7 | -0.16 | 0.75 | -0.15 | 0.75 | 0.52 | 1 | 0.999 | 0.91 | 0.91 |
| all | 450 | 33 | 48.7 | -0.38 | 1.24 | -0.34 | 1.08 | 0.58 | 1 | 1.03 | 0.93 | 0.80 |

f is the target's noiseless value at the first incumbent, yval its observation there, the raw minimum of the initial design, and the GP mean and SD the initial GP's prediction there. The SD of yval is the one the run gives it: `noise_size`, 1, or with `specify_target_noise` the SD that the target returned there, its true noise SD. "GP closer": the fraction of runs whose GP mean is closer to f than yval is.
