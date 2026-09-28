### Refits of the local GP (`_robust_gp_fit_`)

A try is one `gp.fit`; it fails when a factorization of the objective fails ten times (`LinAlgError`). A refit whose every try fails keeps the best of its starts (exit flag -1). The run time is the population record's `wall_s` (`--pop`).

| configuration | level | runs | refits | ok at the first try | ok after retries | every try failed | failed tries | fit time in failed tries | failed tries / run time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 60 | 1466 | 45.7% | 54.0% | 0.3% | 2661 | 55.6% | 22.7% |
| multisensory_s1_D6_homo | 1 | 60 | 1523 | 99.9% | 0.1% | 0.0% | 1 | 0.1% | 0.0% |
| sphere_D3_homo | 1 | 60 | 1501 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 60 | 1373 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 60 | 1576 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |

### Factorizations of the training covariance

Inflated: the factorization failed at least once and succeeded with the noise multiplied by ten per failure; raised: it failed ten times. Low-noise repr.: the share of factorizations with the noise variance below 1e-6 (gpyreg's `L_chol = False`).

| configuration | level | objective evaluations | inflated | raised | posteriors | inflated | raised | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 328674 | 31.0% | 0.8% | 64079 | 50.2% | 0.0% | 1.1% |
| multisensory_s1_D6_homo | 1 | 77217 | 0.0% | 0.0% | 110812 | 0.0% | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 73192 | 0.0% | 0.0% | 68165 | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 101501 | 33.3% | 0.0% | 57468 | 69.9% | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 70043 | 0.0% | 0.0% | 75014 | 0.0% | 0.0% | 0.0% |

### Posteriors that keep an inflated noise

The share of returns of each method whose posterior keeps a noise multiplier above one, and of the acquisition's calls on such a GP.

| configuration | level | after fit | after update | after set_hyperparameters | largest multiplier (median over runs) | search predictions on one | poll predictions on one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 35.0% | 52.1% | 21.2% | 6e+04 | 45.8% | 37.0% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 62.0% | 71.5% | 31.2% | 1e+03 | 65.9% | 60.5% |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |

### Zero predictive SDs (latent variance returned as exactly 0)

A poll acquisition with a zero SD makes the poll's `gamma_z` infinite and marks the GP unreliable (W3-28).

| configuration | level | poll acquisitions with one | poll points | search points | target predictions | negative before the clamp | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3_homo | 1 | 33.2% | 29.1% | 52.9% | 38.0% | 97.6% | 0.1% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ellipsoid_D3_hetero | 2 | 20.0% | 13.5% | 33.9% | 27.1% | 92.5% | 0.0% |
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
| ellipsoid_D3_homo | 1 | 9711 | 22.1% | 5.5% | 56.1% | 30.9% | 48.1% | 6998 | 2.4% | 1.6% | 69.1% | 12.3% | 24.8% |
| multisensory_s1_D6_homo | 1 | 16213 | 17.9% | 1.9% | 53.9% | 64.9% | 87.8% | 16360 | 5.2% | 1.5% | 61.3% | 36.8% | 76.8% |
| sphere_D3_homo | 1 | 9799 | 22.9% | 4.5% | 60.7% | 89.1% | 98.7% | 7568 | 7.3% | 0.1% | 25.0% | 61.1% | 82.8% |
| ellipsoid_D3_hetero | 2 | 8161 | 16.6% | 4.1% | 60.5% | 41.5% | 64.9% | 6595 | 2.5% | 2.5% | 62.3% | 17.9% | 30.6% |
| sphere_D3_hetero | 2 | 10430 | 19.1% | 3.5% | 59.2% | 89.6% | 99.7% | 8039 | 8.1% | 0.3% | 82.1% | 63.9% | 88.0% |

### Zero SDs: the value before the clamp and where they occur

Level 1, 18326577 points.

- floor(log10(|raw| / kss)): -16: 2682027, -15: 14027185, -14: 1060056, -13: 77211, -12: 24019, -11: 7768, -10: 2031, -9: 385, -8: 66, -7: 6, -6: 2, 0: 445821
- distance to the nearest training input (ell): 0: 17827, <1e-6: 728609, <1e-3: 15441281, <1e-1: 2138518, >=1e-1: 342
- floor(log10(kss / effective noise)): 12: 371949, 13: 11556723, 14: 6191030, 15: 69186, 16: 137689

Level 2, 10032283 points.

- floor(log10(|raw| / kss)): -16: 3873778, -15: 4942657, -14: 409357, -13: 46828, -12: 8247, -11: 604, -10: 129, -9: 9, 0: 750674
- distance to the nearest training input (ell): 0: 12509, <1e-6: 1206670, <1e-3: 8705919, <1e-1: 107185
- floor(log10(kss / effective noise)): 12: 24029, 13: 4155140, 14: 5720223, 15: 132727, 16: 164
