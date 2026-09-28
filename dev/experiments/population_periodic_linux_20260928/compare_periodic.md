# Compare off (REF) and on (NEW)

- REF, 180 runs: pybads 12cf2f29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 2c9cdfb
- NEW, 180 runs: pybads 12cf2f29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 2c9cdfb

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| periodic_D2 | KS | true_error | 30 | 30 | 0.567 | 8.74e-05 | 0.000699 | FLAG |
| periodic_D2 | KS | func_count | 30 | 30 | 0.5 | 0.0009 | 0.0063 | FLAG |
| periodic_D2 | signed-rank | log10 true_error | 30 | 30 | 47 | 3.9e-05 | 0.000351 | FLAG |
| periodic_D3_hetero | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 0.213 | ok |
| periodic_D3_hetero | KS | func_count | 30 | 30 | 0.5 | 0.0009 | 0.0063 | FLAG |
| periodic_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 216 | 0.746 | 0.786 | ok |
| periodic_D3_homo | KS | true_error | 30 | 30 | 0.967 | 1.01e-15 | 1.83e-14 | FLAG |
| periodic_D3_homo | KS | func_count | 30 | 30 | 0.6 | 2.37e-05 | 0.000237 | FLAG |
| periodic_D3_homo | signed-rank | log10 true_error | 30 | 30 | 1 | 3.73e-09 | 4.84e-08 | FLAG |
| periodic_D4 | KS | true_error | 30 | 30 | 0.833 | 9.24e-11 | 1.39e-09 | FLAG |
| periodic_D4 | KS | func_count | 30 | 30 | 0.233 | 0.393 | 0.786 | ok |
| periodic_D4 | signed-rank | log10 true_error | 30 | 30 | 0 | 1.86e-09 | 2.61e-08 | FLAG |
| periodic_D6 | KS | true_error | 30 | 30 | 0.9 | 5.79e-13 | 9.84e-12 | FLAG |
| periodic_D6 | KS | func_count | 30 | 30 | 0.4 | 0.0156 | 0.0626 | ok |
| periodic_D6 | signed-rank | log10 true_error | 30 | 30 | 3 | 9.31e-09 | 1.12e-07 | FLAG |
| periodic_rosenbrock_D4 | KS | true_error | 30 | 30 | 0.5 | 0.0009 | 0.0063 | FLAG |
| periodic_rosenbrock_D4 | KS | func_count | 30 | 30 | 0.9 | 5.79e-13 | 9.84e-12 | FLAG |
| periodic_rosenbrock_D4 | signed-rank | log10 true_error | 30 | 30 | 42 | 2.08e-05 | 0.000229 | FLAG |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 30 | -4.935 [-5.249, -0.696] | 0.47 | 1.00 | +0.53 | 0 | 0 |
| periodic_D3_hetero | 30 | +0.049 [-0.152, +0.297] | 0.50 | 0.27 | -0.23 | 0 | 0 |
| periodic_D3_homo | 30 | -0.541 [-0.689, -0.445] | 0.70 | 0.97 | +0.27 | 0 | 0 |
| periodic_D4 | 30 | -5.825 [-6.090, -1.058] | 0.40 | 1.00 | +0.60 | 0 | 0 |
| periodic_D6 | 30 | -6.118 [-6.269, -5.903] | 0.13 | 1.00 | +0.87 | 0 | 0 |
| periodic_rosenbrock_D4 | 30 | -0.828 [-1.230, -0.583] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 18 tests at alpha 0.05. A flag needs p <= 0.0028 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**6 configuration(s) flagged: ['periodic_D2', 'periodic_D3_hetero', 'periodic_D3_homo', 'periodic_D4', 'periodic_D6', 'periodic_rosenbrock_D4']**
