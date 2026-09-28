### Refits of the local GP (`_robust_gp_fit_`)

A try is one `gp.fit`; it fails when a factorization of the objective fails ten times (`LinAlgError`). A refit whose every try fails keeps the best of its starts (exit flag -1). The run time is the population record's `wall_s` (`--pop`).

| configuration | level | runs | refits | ok at the first try | ok after retries | every try failed | failed tries | fit time in failed tries | failed tries / run time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 60 | 1315 | 40.5% | 59.2% | 0.3% | 2538 | 56.7% | 24.1% |
| multisensory_s1_D6_homo | 1 | 60 | 1116 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 60 | 888 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 60 | 1162 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 60 | 914 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |

### Factorizations of the training covariance

Inflated: the factorization failed at least once and succeeded with the noise multiplied by ten per failure; raised: it failed ten times. Low-noise repr.: the share of factorizations with the noise variance below 1e-6 (gpyreg's `L_chol = False`).

| configuration | level | objective evaluations | inflated | raised | posteriors | inflated | raised | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 310566 | 31.7% | 0.8% | 55985 | 55.4% | 0.0% | 1.2% |
| multisensory_s1_D6_homo | 1 | 66432 | 0.0% | 0.0% | 68344 | 0.0% | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 58630 | 0.0% | 0.0% | 32412 | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 94815 | 33.3% | 0.0% | 46308 | 78.7% | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 57401 | 0.0% | 0.0% | 33860 | 0.0% | 0.0% | 0.0% |

### Posteriors that keep an inflated noise

The share of returns of each method whose posterior keeps a noise multiplier above one, and of the acquisition's calls on such a GP.

| configuration | level | after fit | after update | after set_hyperparameters | largest multiplier (median over runs) | search predictions on one | poll predictions on one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 38.1% | 57.5% | 23.3% | 1e+05 | 47.7% | 46.1% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 67.1% | 80.2% | 36.8% | 1e+03 | 71.2% | 75.7% |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |

### Zero predictive SDs (latent variance returned as exactly 0)

A poll acquisition with a zero SD makes the poll's `gamma_z` infinite and marks the GP unreliable (W3-28).

| configuration | level | poll acquisitions with one | poll points | search points | target predictions | negative before the clamp | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 39.4% | 35.1% | 57.6% | 42.7% | 97.8% | 0.5% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ellipsoid_D3_hetero | 2 | 28.1% | 21.3% | 29.8% | 26.5% | 93.1% | 0.0% |
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

Each call of `_sto_success_improvement_` at the search or the poll: success (1), uncertain (0), failure (-1, certain or no estimate); mu is the estimated improvement and SD its standard deviation. With `opp_stobads` every uncertain outcome of the search moves its incumbent; a certain outcome whose abs(mu) is under 0.5 SD is right with a probability of at most about 69%.

| configuration | level | search decisions | search success | search uncertain | search uncertain with mu < 0 | search certain with abs(mu) < 0.5 SD | search certain with abs(mu) < 1.96 SD | poll decisions | poll success | poll uncertain | poll uncertain with mu < 0 | poll certain with abs(mu) < 0.5 SD | poll certain with abs(mu) < 1.96 SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 8006 | 19.5% | 22.7% | 50.8% | 0.0% | 0.0% | 6491 | 1.4% | 14.2% | 68.8% | 0.0% | 0.0% |
| multisensory_s1_D6_homo | 1 | 7440 | 2.6% | 71.8% | 47.5% | 0.0% | 0.0% | 12925 | 0.9% | 79.6% | 62.0% | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 3450 | 1.2% | 97.0% | 52.3% | 0.0% | 0.0% | 5142 | 0.0% | 77.7% | 60.0% | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 5914 | 12.9% | 49.7% | 58.5% | 0.0% | 0.0% | 5951 | 1.1% | 42.6% | 66.6% | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 3471 | 0.1% | 99.3% | 55.4% | 0.0% | 0.0% | 5496 | 0.0% | 85.8% | 60.5% | 0.0% | 0.0% |

### Zero SDs: the value before the clamp and where they occur

Level 1, 16710802 points.

- floor(log10(|raw| / kss)): -16: 2576641, -15: 12662001, -14: 1007857, -13: 65353, -12: 19635, -11: 7711, -10: 1878, -9: 355, -8: 66, -7: 6, -6: 2, 0: 369297
- distance to the nearest training input (ell): 0: 17744, <1e-6: 1558942, <1e-3: 13755098, <1e-1: 1378685, >=1e-1: 333
- floor(log10(kss / effective noise)): 11: 2, 12: 242069, 13: 10318192, 14: 5942169, 15: 70681, 16: 137689

Level 2, 6065903 points.

- floor(log10(|raw| / kss)): -16: 2199574, -15: 3147955, -14: 264090, -13: 29171, -12: 7501, -11: 788, -10: 133, -9: 10, 0: 416681
- distance to the nearest training input (ell): 0: 8693, <1e-6: 1744895, <1e-3: 4243453, <1e-1: 68862
- floor(log10(kss / effective noise)): 12: 7998, 13: 1657360, 14: 4187779, 15: 212602, 16: 164
