### Refits of the local GP (`_robust_gp_fit_`)

A try is one `gp.fit`; it fails when a factorization of the objective raises (`LinAlgError`: after ten failures, or at the first under gpyreg's switch). A refit whose every try fails keeps the best of its starts (exit flag -1). The fits that raised include the initial fits of `init_and_train_gp`; the run time is the population record's `wall_s` (`--pop`).

| configuration | level | runs | refits | ok at the first try | ok after retries | every try failed | failed tries | fit time in fits that raised | fits that raised / run time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 60 | 1397 | 42.8% | 57.1% | 0.1% | 2681 | 56.0% | 23.5% |
| multisensory_s1_D6_homo | 1 | 60 | 1492 | 99.9% | 0.1% | 0.0% | 1 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 60 | 1345 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 60 | 1327 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 60 | 1555 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |

### Factorizations of the training covariance

Inflated: the factorization failed at least once and succeeded with the noise multiplied by ten per failure; raised: it failed ten times, or once under gpyreg's switch (`raise_on_cholesky_failure`). Low-noise repr.: the share of factorizations with the noise variance below 1e-6 (gpyreg's `L_chol = False`).

| configuration | level | objective evaluations | inflated | raised | posteriors | inflated | raised | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 330233 | 32.0% | 0.8% | 59371 | 54.7% | 0.0% | 1.2% |
| multisensory_s1_D6_homo | 1 | 76014 | 0.0% | 0.0% | 107740 | 0.0% | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 69672 | 0.0% | 0.0% | 57867 | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 101185 | 33.4% | 0.0% | 55282 | 69.4% | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 69694 | 0.0% | 0.0% | 73314 | 0.0% | 0.0% | 0.0% |

### Posteriors that keep an inflated noise

The share of returns of each method whose posterior keeps a noise multiplier above one, and of the acquisition's calls on such a GP. A dash for set_hyperparameters: counters that did not count its returns without a posterior apart.

| configuration | level | after fit | after update | after set_hyperparameters | largest multiplier (median over runs) | search predictions on one | poll predictions on one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 40.2% | 56.5% | — | 1e+04 | 51.1% | 41.6% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 60.9% | 71.0% | — | 1e+03 | 64.5% | 60.8% |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |

### Zero predictive SDs (latent variance returned as exactly 0)

A poll acquisition with a zero SD makes the poll's `gamma_z` infinite and marks the GP unreliable (W3-28).

| configuration | level | poll acquisitions with one | poll points | search points | target predictions | negative before the clamp | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 33.0% | 29.3% | 53.8% | 40.0% | 97.8% | 0.4% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ellipsoid_D3_hetero | 2 | 18.9% | 12.1% | 32.8% | 25.8% | 93.1% | 0.0% |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

### Log priors, priors outside their bounds, small training sets

| configuration | level | NaN log priors | -inf log priors | NaN objectives | fits with the mean's prior outside its bounds | ... the covariance's | ... the noise's | runs with 2 or fewer distinct training points | hook errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| multisensory_s1_D6_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D3_hetero | 2 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_hetero | 2 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |

### Sto-BADS decisions

Each call of `_sto_success_improvement_` at the search or the poll: success (1), uncertain (0), failure (-1, certain or no estimate); mu is the estimated improvement and SD its standard deviation. The shares of certain outcomes are over the certain outcomes with an estimate; one with an SD of 0 is decided by the sign of mu alone. A certain outcome whose abs(mu) is under 0.5 SD is right with a probability of at most about 69%. With `opp_stobads`, an uncertain outcome moves the incumbent where the estimated improvement is positive (the search did on every uncertain outcome before W0-13's fix).

| configuration | level | search decisions | search success | search uncertain | search uncertain with mu < 0 | search certain with SD 0 | search certain with abs(mu) < 0.5 SD | search certain with abs(mu) < 1.96 SD | poll decisions | poll success | poll uncertain | poll uncertain with mu < 0 | poll certain with SD 0 | poll certain with abs(mu) < 0.5 SD | poll certain with abs(mu) < 1.96 SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 8965 | 21.6% | 6.6% | 58.4% | 43.9% | 16.4% | 25.4% | 6721 | 1.6% | 1.6% | 82.7% | 26.5% | 6.1% | 14.1% |
| multisensory_s1_D6_homo | 1 | 15616 | 17.4% | 2.1% | 56.7% | 0.0% | 64.9% | 87.9% | 16042 | 5.4% | 1.2% | 60.3% | 0.0% | 39.0% | 76.7% |
| sphere_D3_homo | 1 | 8445 | 21.8% | 4.7% | 65.0% | 0.0% | 90.6% | 98.9% | 6689 | 7.2% | 0.0% | 33.3% | 0.0% | 61.0% | 80.6% |
| ellipsoid_D3_hetero | 2 | 7835 | 17.1% | 4.0% | 59.7% | 21.5% | 30.0% | 50.3% | 6321 | 2.7% | 2.4% | 65.3% | 10.6% | 14.2% | 27.0% |
| sphere_D3_hetero | 2 | 10070 | 18.3% | 3.1% | 58.1% | 0.0% | 89.5% | 99.9% | 8000 | 8.5% | 0.3% | 91.7% | 0.0% | 64.7% | 88.5% |

### Zero SDs: the value before the clamp and where they occur

Level 1, 17387290 points.

- floor(log10(|raw| / kss)): exact0: 383966, -16: 2803693, -15: 13124617, -14: 989586, -13: 62255, -12: 16308, -11: 5427, -10: 1178, -9: 207, -8: 46, -7: 5, -6: 2
- distance to the nearest training input (ell): 0: 17494, <1e-6: 846726, <1e-3: 14754908, <1e-1: 1767901, >=1e-1: 261
- floor(log10(kss / effective noise)): 12: 256996, 13: 10613102, 14: 6308511, 15: 70992, 16: 137689

Level 2, 9495560 points.

- floor(log10(|raw| / kss)): exact0: 659582, -16: 3417858, -15: 5000222, -14: 343665, -13: 61680, -12: 10672, -11: 1663, -10: 188, -9: 28, -8: 1, -7: 1
- distance to the nearest training input (ell): 0: 11542, <1e-6: 806080, <1e-3: 8527382, <1e-1: 150555, >=1e-1: 1
- floor(log10(kss / effective noise)): 12: 3, 13: 3403997, 14: 5883802, 15: 207758
