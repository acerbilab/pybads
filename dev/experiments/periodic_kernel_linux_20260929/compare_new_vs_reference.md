# Compare on (REF) and new_on (NEW)

- REF, 180 runs: pybads 12cf2f29, gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 2c9cdfb
- NEW, 180 runs: pybads 9a705918, gpyreg 1.3.4.dev21+gb44634f2a from /home/user/gpyreg/gpyreg at 0f27db5

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| periodic_D2 | KS | true_error | 30 | 30 | 0.167 | 0.808 | 1 | ok |
| periodic_D2 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| periodic_D2 | signed-rank | log10 true_error | 30 | 30 | 139 | 0.527 | 1 | ok |
| periodic_D3_hetero | KS | true_error | 30 | 30 | 0.0667 | 1 | 1 | ok |
| periodic_D3_hetero | KS | func_count | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| periodic_D3_hetero | signed-rank | log10 true_error | 30 | 30 | 26 | 0.534 | 1 | ok |
| periodic_D3_homo | KS | true_error | 30 | 30 | 0.1 | 0.999 | 1 | ok |
| periodic_D3_homo | KS | func_count | 30 | 30 | 0.2 | 0.594 | 1 | ok |
| periodic_D3_homo | signed-rank | log10 true_error | 30 | 30 | 72 | 0.557 | 1 | ok |
| periodic_D4 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| periodic_D4 | KS | func_count | 30 | 30 | 0 | 1 | 1 | ok |
| periodic_D4 | signed-rank | log10 true_error | 30 | 30 | 193 | 0.428 | 1 | ok |
| periodic_D6 | KS | true_error | 30 | 30 | 0.333 | 0.0709 | 1 | ok |
| periodic_D6 | KS | func_count | 30 | 30 | 0.0333 | 1 | 1 | ok |
| periodic_D6 | signed-rank | log10 true_error | 30 | 30 | 173 | 0.229 | 1 | ok |
| periodic_rosenbrock_D4 | KS | true_error | 30 | 30 | 0.267 | 0.239 | 1 | ok |
| periodic_rosenbrock_D4 | KS | func_count | 30 | 30 | 0.0667 | 1 | 1 | ok |
| periodic_rosenbrock_D4 | signed-rank | log10 true_error | 30 | 30 | 204 | 0.57 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| periodic_D2 | 30 | +0.000 [-0.336, +0.214] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_D3_hetero | 30 | +0.000 [+0.000, +0.000] | 0.27 | 0.20 | -0.07 | 0 | 0 |
| periodic_D3_homo | 30 | +0.000 [-0.001, +0.000] | 0.97 | 0.97 | +0.00 | 0 | 0 |
| periodic_D4 | 30 | -0.003 [-0.218, +0.113] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_D6 | 30 | -0.145 [-0.310, +0.168] | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_rosenbrock_D4 | 30 | -0.061 [-0.342, +0.233] | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 18 tests at alpha 0.05. A flag needs p <= 0.0028 for the first step; for 30 vs 30 runs that is a KS statistic of at least 0.467.

**no configuration flagged (18 tests)**
