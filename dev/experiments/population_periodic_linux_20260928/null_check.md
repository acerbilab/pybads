# Null check: even vs odd seeds of on

- REF, 180 runs: pybads 66ef459c (dirty), gpyreg 1.3.3 from /home/user/gpyreg/gpyreg at 3f1a732

| config | test | metric | n ref | n new | statistic | p | p Holm | |
|---|---|---|---|---|---|---|---|---|
| periodic_D2 | KS | true_error | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| periodic_D2 | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| periodic_D3_hetero | KS | true_error | 15 | 15 | 0.4 | 0.184 | 1 | ok |
| periodic_D3_hetero | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| periodic_D3_homo | KS | true_error | 15 | 15 | 0.467 | 0.0755 | 0.906 | ok |
| periodic_D3_homo | KS | func_count | 15 | 15 | 0.267 | 0.678 | 1 | ok |
| periodic_D4 | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| periodic_D4 | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| periodic_D6 | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| periodic_D6 | KS | func_count | 15 | 15 | 0.2 | 0.938 | 1 | ok |
| periodic_rosenbrock_D4 | KS | true_error | 15 | 15 | 0.333 | 0.386 | 1 | ok |
| periodic_rosenbrock_D4 | KS | func_count | 15 | 15 | 0.333 | 0.386 | 1 | ok |

| config | pairs | median log10 error ratio (new/ref) [95% CI] | solved ref | solved new | change | crashed ref | crashed new |
|---|---|---|---|---|---|---|---|
| periodic_D2 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_D3_hetero | - | - | 0.27 | 0.27 | +0.00 | 0 | 0 |
| periodic_D3_homo | - | - | 0.93 | 1.00 | +0.07 | 0 | 0 |
| periodic_D4 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_D6 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |
| periodic_rosenbrock_D4 | - | - | 1.00 | 1.00 | +0.00 | 0 | 0 |

Holm family: 12 tests at alpha 0.05. A flag needs p <= 0.0042 for the first step; for 15 vs 15 runs that is a KS statistic of at least 0.667.

**no configuration flagged (12 tests)**
