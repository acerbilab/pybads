### Refits of the local GP (`_robust_gp_fit_`)

A try is one `gp.fit`; it fails when a factorization of the objective raises (`LinAlgError`: after ten failures, or at the first under gpyreg's switch). A refit whose every try fails keeps the best of its starts (exit flag -1). The fits that raised include the initial fits of `init_and_train_gp`; the run time is the population record's `wall_s` (`--pop`).

| configuration | level | runs | refits | ok at the first try | ok after retries | every try failed | failed tries | fit time in fits that raised | fits that raised / run time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 60 | 1319 | 40.0% | 59.3% | 0.8% | 2701 | 57.8% | 25.8% |
| multisensory_s1_D6_homo | 1 | 60 | 1153 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 60 | 861 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 60 | 1242 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 60 | 1013 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |

### Factorizations of the training covariance

Inflated: the factorization failed at least once and succeeded with the noise multiplied by ten per failure; raised: it failed ten times, or once under gpyreg's switch (`raise_on_cholesky_failure`). Low-noise repr.: the share of factorizations with the noise variance below 1e-6 (gpyreg's `L_chol = False`).

| configuration | level | objective evaluations | inflated | raised | posteriors | inflated | raised | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 323676 | 32.6% | 0.8% | 55609 | 59.9% | 0.0% | 1.1% |
| multisensory_s1_D6_homo | 1 | 67289 | 0.0% | 0.0% | 70259 | 0.0% | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 58775 | 0.0% | 0.0% | 30099 | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 97392 | 34.2% | 0.0% | 49911 | 76.7% | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 59338 | 0.0% | 0.0% | 38295 | 0.0% | 0.0% | 0.0% |

### Posteriors that keep an inflated noise

The share of returns of each method whose posterior keeps a noise multiplier above one, and of the acquisition's calls on such a GP. A dash for set_hyperparameters: counters that did not count its returns without a posterior apart.

| configuration | level | after fit | after update | after set_hyperparameters | largest multiplier (median over runs) | search predictions on one | poll predictions on one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 43.0% | 61.9% | — | 1e+05 | 53.0% | 50.3% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 67.1% | 78.2% | — | 1e+03 | 69.2% | 72.0% |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |

### Zero predictive SDs (latent variance returned as exactly 0)

A poll acquisition with a zero SD makes the poll's `gamma_z` infinite and marks the GP unreliable (W3-28).

| configuration | level | poll acquisitions with one | poll points | search points | target predictions | negative before the clamp | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 39.7% | 35.3% | 57.1% | 41.9% | 98.2% | 0.3% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ellipsoid_D3_hetero | 2 | 30.1% | 23.5% | 39.0% | 32.4% | 93.5% | 0.0% |
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
| ellipsoid_D3_homo | 1 | 7749 | 17.2% | 25.2% | 69.9% | 59.4% | 0.0% | 0.0% | 6813 | 1.2% | 15.3% | 91.5% | 32.4% | 0.0% | 0.0% |
| multisensory_s1_D6_homo | 1 | 7676 | 2.2% | 79.4% | 74.9% | 0.0% | 0.0% | 0.0% | 13779 | 0.8% | 78.3% | 86.1% | 0.0% | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 3320 | 1.2% | 97.8% | 72.4% | 0.0% | 0.0% | 0.0% | 5015 | 0.0% | 76.2% | 79.2% | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 6369 | 12.3% | 46.3% | 80.4% | 51.9% | 0.0% | 0.0% | 6599 | 1.1% | 32.5% | 90.4% | 23.2% | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 3957 | 0.1% | 99.8% | 71.1% | 0.0% | 0.0% | 0.0% | 6234 | 0.0% | 86.2% | 76.9% | 0.0% | 0.0% | 0.0% |

### Zero SDs: the value before the clamp and where they occur

Level 1, 15408623 points.

- floor(log10(|raw| / kss)): exact0: 282905, -16: 2218860, -15: 11874651, -14: 938892, -13: 65850, -12: 17143, -11: 7143, -10: 2889, -9: 237, -8: 46, -7: 5, -6: 2
- distance to the nearest training input (ell): 0: 18297, <1e-6: 1673843, <1e-3: 12633380, <1e-1: 1082842, >=1e-1: 261
- floor(log10(kss / effective noise)): 12: 208455, 13: 9211355, 14: 5770744, 15: 80380, 16: 137689

Level 2, 8367110 points.

- floor(log10(|raw| / kss)): exact0: 541474, -16: 3056199, -15: 4460626, -14: 261063, -13: 35781, -12: 7990, -11: 3532, -10: 413, -9: 30, -8: 1, -7: 1
- distance to the nearest training input (ell): 0: 13159, <1e-6: 2892060, <1e-3: 5372351, <1e-1: 89539, >=1e-1: 1
- floor(log10(kss / effective noise)): 12: 24176, 13: 4025736, 14: 4157461, 15: 159665, 16: 72
