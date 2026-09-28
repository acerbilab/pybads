# Compare off (REF) and on (NEW)

- REF, 180 runs: pybads 66ef459c (dirty), gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 3f1a732
- NEW, 180 runs: pybads 66ef459c (dirty), gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 3f1a732

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| periodic_D2 | KS | true_error | 30 | 30 | 0.633 | 5.8e-06 | 5.8e-05 | FLAG |
| periodic_D2 | KS | func_count | 30 | 30 | 0.5 | 0.0009 | 0.0045 | FLAG |
| periodic_D2 | signed-rank | log10 true_error | 30 | 30 | 46 | 3.45e-05 | 0.000276 | FLAG |
| periodic_D3_hetero | KS | true_error | 30 | 30 | 0.367 | 0.0346 | 0.138 | ok |
| periodic_D3_hetero | KS | func_count | 30 | 30 | 0.567 | 8.74e-05 | 0.000612 | FLAG |
| periodic_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 218 | 0.777 | 0.786 | ok |
| periodic_D3_homo | KS | true_error | 30 | 30 | 0.967 | 1.01e-15 | 1.83e-14 | FLAG |
| periodic_D3_homo | KS | func_count | 30 | 30 | 0.533 | 0.000293 | 0.00176 | FLAG |
| periodic_D3_homo | signed-rank | log10 true_error | 30 | 30 | 0 | 1.86e-09 | 2.61e-08 | FLAG |
| periodic_D4 | KS | true_error | 30 | 30 | 0.833 | 9.24e-11 | 1.39e-09 | FLAG |
| periodic_D4 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 0.786 | ok |
| periodic_D4 | signed-rank | log10 true_error | 30 | 30 | 2 | 5.59e-09 | 7.26e-08 | FLAG |
| periodic_D6 | KS | true_error | 30 | 30 | 0.9 | 5.79e-13 | 9.84e-12 | FLAG |
| periodic_D6 | KS | func_count | 30 | 30 | 0.367 | 0.0346 | 0.138 | ok |
| periodic_D6 | signed-rank | log10 true_error | 30 | 30 | 3 | 9.31e-09 | 1.12e-07 | FLAG |
| periodic_rosenbrock_D4 | KS | true_error | 30 | 30 | 0.6 | 2.37e-05 | 0.000213 | FLAG |
| periodic_rosenbrock_D4 | KS | func_count | 30 | 30 | 0.9 | 5.79e-13 | 9.84e-12 | FLAG |
| periodic_rosenbrock_D4 | signed-rank | log10 true_error | 30 | 30 | 21 | 8.33e-07 | 9.16e-06 | FLAG |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 30 | -5.010 [-5.438, -0.747] | 0.47 | 1.00 | +0.53 | 0 | 0 |
| periodic_D3_hetero | 30 | +0.079 [-0.156, +0.296] | 0.50 | 0.27 | -0.23 | 0 | 0 |
| periodic_D3_homo | 30 | -0.561 [-0.692, -0.422] | 0.70 | 0.97 | +0.27 | 0 | 0 |
| periodic_D4 | 30 | -5.874 [-6.095, -0.784] | 0.40 | 1.00 | +0.60 | 0 | 0 |
| periodic_D6 | 30 | -6.196 [-6.414, -5.853] | 0.13 | 1.00 | +0.87 | 0 | 0 |
| periodic_rosenbrock_D4 | 30 | -0.991 [-1.124, -0.604] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 18 tests at alpha 0.05. A flag needs p <= 0.0028 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**6 configuration(s) flagged: ['periodic_D2', 'periodic_D3_hetero', 'periodic_D3_homo', 'periodic_D4', 'periodic_D6', 'periodic_rosenbrock_D4']**
