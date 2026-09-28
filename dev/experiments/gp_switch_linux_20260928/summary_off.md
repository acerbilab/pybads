### Refits of the local GP (`_robust_gp_fit_`)

A try is one `gp.fit`; it fails when a factorization of the objective fails ten times (`LinAlgError`). A refit whose every try fails keeps the best of its starts (exit flag -1). The run time is the population record's `wall_s` (`--pop`).

| configuration | level | runs | refits | ok at the first try | ok after retries | every try failed | failed tries | fit time in failed tries | failed tries / run time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 30 | 159 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ackley_D6 | 0 | 30 | 622 | 99.8% | 0.2% | 0.0% | 2 | 0.5% | 0.1% |
| edgesphere_D2 | 0 | 30 | 142 | 74.6% | 25.4% | 0.0% | 67 | 28.9% | 10.3% |
| edgesphere_D4 | 0 | 30 | 270 | 97.8% | 2.2% | 0.0% | 8 | 4.4% | 1.3% |
| ellipsoid_D10 | 0 | 30 | 815 | 86.1% | 13.9% | 0.0% | 175 | 24.0% | 8.2% |
| ellipsoid_D1_unbounded | 0 | 30 | 94 | 70.2% | 29.8% | 0.0% | 46 | 25.4% | 10.3% |
| ellipsoid_D3 | 0 | 30 | 434 | 35.5% | 58.1% | 6.5% | 1302 | 71.6% | 42.6% |
| ellipsoid_D3_unbounded | 0 | 30 | 441 | 36.1% | 59.4% | 4.5% | 1236 | 70.5% | 40.8% |
| ellipsoid_D6 | 0 | 30 | 627 | 65.2% | 34.4% | 0.3% | 557 | 47.7% | 21.9% |
| logsphere_D3 | 0 | 30 | 228 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| logsphere_D3_nopb | 0 | 30 | 227 | 99.6% | 0.4% | 0.0% | 1 | 0.3% | 0.1% |
| multisensory_s1_D6 | 0 | 30 | 531 | 99.4% | 0.6% | 0.0% | 3 | 1.1% | 0.2% |
| rastrigin_D1 | 0 | 30 | 139 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| rastrigin_D3 | 0 | 30 | 459 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ridge_D2 | 0 | 30 | 544 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ridge_D4 | 0 | 30 | 829 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| rosenbrock_D2 | 0 | 30 | 286 | 63.3% | 36.7% | 0.0% | 281 | 47.4% | 20.7% |
| rosenbrock_D6 | 0 | 30 | 665 | 99.4% | 0.6% | 0.0% | 4 | 1.1% | 0.2% |
| sphere_D1 | 0 | 30 | 96 | 75.0% | 25.0% | 0.0% | 45 | 25.4% | 10.5% |
| sphere_D10 | 0 | 30 | 473 | 97.5% | 2.5% | 0.0% | 17 | 5.5% | 1.3% |
| sphere_D2 | 0 | 30 | 161 | 68.9% | 31.1% | 0.0% | 147 | 38.4% | 15.6% |
| sphere_D3_nopb | 0 | 30 | 291 | 58.8% | 40.9% | 0.3% | 415 | 53.4% | 27.1% |
| sphere_D3_x0lb | 0 | 30 | 255 | 68.2% | 31.8% | 0.0% | 227 | 41.5% | 18.0% |
| sphere_band_D2 | None | 30 | 0 | — | — | — | 0 | 0.0% | 0.0% |
| sphere_band_D3 | 0 | 30 | 154 | 70.1% | 29.9% | 0.0% | 112 | 36.1% | 16.5% |
| sphere_nonbox_D3 | 0 | 30 | 290 | 62.4% | 37.6% | 0.0% | 339 | 47.9% | 24.7% |
| timing_D5 | 0 | 30 | 558 | 99.3% | 0.7% | 0.0% | 5 | 1.1% | 0.1% |
| edgesphere_D3_homo | 1 | 30 | 565 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ellipsoid_D3_homo | 1 | 30 | 695 | 44.5% | 55.1% | 0.4% | 1316 | 56.6% | 23.0% |
| logsphere_D3_homo | 1 | 30 | 726 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| multisensory_s1_D6_homo | 1 | 30 | 811 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D1_homo | 1 | 30 | 282 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 30 | 576 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 30 | 714 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D1_hetero | 2 | 30 | 347 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 30 | 706 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |

### Factorizations of the training covariance

Inflated: the factorization failed at least once and succeeded with the noise multiplied by ten per failure; raised: it failed ten times. Low-noise repr.: the share of factorizations with the noise variance below 1e-6 (gpyreg's `L_chol = False`).

| configuration | level | objective evaluations | inflated | raised | posteriors | inflated | raised | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 15371 | 0.0% | 0.0% | 3107 | 0.0% | 0.0% | 6.2% |
| ackley_D6 | 0 | 31580 | 0.0% | 0.0% | 18819 | 0.0% | 0.0% | 20.6% |
| edgesphere_D2 | 0 | 24548 | 3.2% | 0.3% | 2687 | 8.2% | 0.0% | 36.8% |
| edgesphere_D4 | 0 | 26080 | 0.7% | 0.0% | 4866 | 0.7% | 0.0% | 38.2% |
| ellipsoid_D10 | 0 | 68542 | 25.5% | 0.3% | 35672 | 35.5% | 0.0% | 36.8% |
| ellipsoid_D1_unbounded | 0 | 19030 | 3.1% | 0.2% | 1710 | 7.0% | 0.0% | 25.9% |
| ellipsoid_D3 | 0 | 102905 | 49.9% | 1.3% | 8003 | 64.9% | 0.0% | 34.1% |
| ellipsoid_D3_unbounded | 0 | 102000 | 47.5% | 1.2% | 8160 | 64.3% | 0.0% | 33.8% |
| ellipsoid_D6 | 0 | 89667 | 38.4% | 0.6% | 18275 | 48.6% | 0.0% | 32.3% |
| logsphere_D3 | 0 | 22222 | 0.1% | 0.0% | 3833 | 0.0% | 0.0% | 35.7% |
| logsphere_D3_nopb | 0 | 22668 | 0.2% | 0.0% | 4098 | 0.0% | 0.0% | 35.4% |
| multisensory_s1_D6 | 0 | 31769 | 0.0% | 0.0% | 15044 | 0.0% | 0.0% | 27.4% |
| rastrigin_D1 | 0 | 15369 | 0.0% | 0.0% | 2795 | 0.0% | 0.0% | 13.4% |
| rastrigin_D3 | 0 | 27516 | 0.0% | 0.0% | 8409 | 0.0% | 0.0% | 17.1% |
| ridge_D2 | 0 | 29865 | 0.0% | 0.0% | 10562 | 0.0% | 0.0% | 17.4% |
| ridge_D4 | 0 | 40497 | 0.0% | 0.0% | 20209 | 0.0% | 0.0% | 14.3% |
| rosenbrock_D2 | 0 | 44867 | 33.7% | 0.6% | 5462 | 53.7% | 0.0% | 22.8% |
| rosenbrock_D6 | 0 | 35654 | 1.5% | 0.0% | 22341 | 1.7% | 0.0% | 22.4% |
| sphere_D1 | 0 | 19383 | 3.1% | 0.2% | 1759 | 7.5% | 0.0% | 27.6% |
| sphere_D10 | 0 | 34536 | 12.8% | 0.0% | 21415 | 20.8% | 0.0% | 53.0% |
| sphere_D2 | 0 | 31745 | 13.7% | 0.5% | 3039 | 36.6% | 0.0% | 37.5% |
| sphere_D3_nopb | 0 | 53909 | 24.1% | 0.8% | 5191 | 55.8% | 0.0% | 50.5% |
| sphere_D3_x0lb | 0 | 40432 | 20.1% | 0.6% | 4086 | 46.5% | 0.0% | 45.7% |
| sphere_band_D2 | None | 4111 | 0.0% | 0.0% | 30 | 0.0% | 0.0% | 0.0% |
| sphere_band_D3 | 0 | 26974 | 12.4% | 0.4% | 3701 | 30.5% | 0.0% | 43.1% |
| sphere_nonbox_D3 | 0 | 49192 | 20.4% | 0.7% | 5479 | 49.5% | 0.0% | 53.1% |
| timing_D5 | 0 | 35098 | 0.3% | 0.0% | 13167 | 0.1% | 0.0% | 33.0% |
| edgesphere_D3_homo | 1 | 31299 | 0.0% | 0.0% | 23733 | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_homo | 1 | 160957 | 30.9% | 0.8% | 29226 | 51.2% | 0.0% | 1.3% |
| logsphere_D3_homo | 1 | 34715 | 0.0% | 0.0% | 33924 | 0.0% | 0.0% | 0.0% |
| multisensory_s1_D6_homo | 1 | 40033 | 0.0% | 0.0% | 62470 | 0.0% | 0.0% | 0.0% |
| sphere_D1_homo | 1 | 25391 | 0.0% | 0.0% | 11709 | 0.0% | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 32562 | 0.0% | 0.0% | 23640 | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 50952 | 31.0% | 0.0% | 31012 | 65.8% | 0.0% | 0.0% |
| sphere_D1_hetero | 2 | 25716 | 0.0% | 0.0% | 14388 | 0.0% | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 33377 | 0.0% | 0.0% | 33054 | 0.0% | 0.0% | 0.0% |

### Posteriors that keep an inflated noise

The share of returns of each method whose posterior keeps a noise multiplier above one, and of the acquisition's calls on such a GP.

| configuration | level | after fit | after update | after set_hyperparameters | largest multiplier (median over runs) | search predictions on one | poll predictions on one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| ackley_D6 | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| edgesphere_D2 | 0 | 0.0% | 7.6% | 7.8% | 1e+01 | 8.1% | 7.6% |
| edgesphere_D4 | 0 | 0.3% | 0.6% | 0.7% | 1e+00 | 0.7% | 0.6% |
| ellipsoid_D10 | 0 | 20.0% | 34.0% | 34.2% | 1e+04 | 38.8% | 29.1% |
| ellipsoid_D1_unbounded | 0 | 0.0% | 6.9% | 6.3% | 1e+00 | 6.9% | 7.0% |
| ellipsoid_D3 | 0 | 45.9% | 62.6% | 45.2% | 1e+06 | 64.1% | 68.0% |
| ellipsoid_D3_unbounded | 0 | 45.7% | 62.2% | 45.3% | 1e+06 | 64.1% | 66.2% |
| ellipsoid_D6 | 0 | 36.2% | 46.6% | 43.1% | 1e+06 | 49.6% | 43.5% |
| logsphere_D3 | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| logsphere_D3_nopb | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| multisensory_s1_D6 | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| rastrigin_D1 | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| rastrigin_D3 | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| ridge_D2 | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| ridge_D4 | 0 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| rosenbrock_D2 | 0 | 33.5% | 52.0% | 43.8% | 1e+04 | 52.0% | 60.2% |
| rosenbrock_D6 | 0 | 0.6% | 1.3% | 1.9% | 1e+00 | 1.2% | 2.5% |
| sphere_D1 | 0 | 0.0% | 7.6% | 6.6% | 1e+00 | 7.2% | 8.4% |
| sphere_D10 | 0 | 16.9% | 18.3% | 20.9% | 1e+02 | 19.9% | 21.4% |
| sphere_D2 | 0 | 16.2% | 35.4% | 30.8% | 1e+01 | 36.3% | 40.0% |
| sphere_D3_nopb | 0 | 26.9% | 53.0% | 45.0% | 1e+02 | 52.7% | 59.0% |
| sphere_D3_x0lb | 0 | 28.4% | 42.7% | 38.7% | 1e+02 | 43.7% | 51.4% |
| sphere_band_D2 | None | 0.0% | — | — | 1e+00 | — | — |
| sphere_band_D3 | 0 | 15.2% | 29.9% | 26.3% | 1e+01 | 30.3% | 34.0% |
| sphere_nonbox_D3 | 0 | 27.2% | 47.9% | 40.0% | 1e+02 | 49.2% | 49.4% |
| timing_D5 | 0 | 0.0% | 0.0% | 0.1% | 1e+00 | 0.0% | 0.1% |
| edgesphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_homo | 1 | 36.8% | 52.9% | 22.7% | 1e+04 | 48.0% | 38.1% |
| logsphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| sphere_D1_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 57.9% | 67.2% | 28.7% | 1e+03 | 60.6% | 57.3% |
| sphere_D1_hetero | 2 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | 0.0% | 1e+00 | 0.0% | 0.0% |

### Zero predictive SDs (latent variance returned as exactly 0)

A poll acquisition with a zero SD makes the poll's `gamma_z` infinite and marks the GP unreliable (W3-28).

| configuration | level | poll acquisitions with one | poll points | search points | target predictions | negative before the clamp | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ackley_D6 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| edgesphere_D2 | 0 | 24.4% | 22.0% | 27.9% | 26.8% | 81.7% | 100.0% |
| edgesphere_D4 | 0 | 19.2% | 15.3% | 13.3% | 9.8% | 60.8% | 100.0% |
| ellipsoid_D10 | 0 | 7.6% | 3.3% | 7.6% | 25.7% | 94.8% | 77.6% |
| ellipsoid_D1_unbounded | 0 | 17.8% | 16.3% | 16.5% | 9.9% | 81.7% | 100.0% |
| ellipsoid_D3 | 0 | 14.3% | 9.3% | 15.1% | 23.7% | 96.7% | 66.6% |
| ellipsoid_D3_unbounded | 0 | 20.4% | 13.9% | 16.2% | 29.1% | 96.2% | 68.2% |
| ellipsoid_D6 | 0 | 19.0% | 11.2% | 16.0% | 30.9% | 96.9% | 76.9% |
| logsphere_D3 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 9.7% | 100.0% |
| logsphere_D3_nopb | 0 | 2.3% | 1.3% | 0.8% | 0.2% | 60.0% | 100.0% |
| multisensory_s1_D6 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| rastrigin_D1 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| rastrigin_D3 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ridge_D2 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ridge_D4 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| rosenbrock_D2 | 0 | 21.9% | 18.2% | 15.9% | 10.5% | 94.2% | 33.8% |
| rosenbrock_D6 | 0 | 4.6% | 3.5% | 2.1% | 3.1% | 76.5% | 100.0% |
| sphere_D1 | 0 | 12.5% | 11.7% | 15.2% | 8.6% | 83.5% | 100.0% |
| sphere_D10 | 0 | 15.6% | 12.3% | 11.4% | 17.1% | 90.6% | 100.0% |
| sphere_D2 | 0 | 38.0% | 33.5% | 29.7% | 25.5% | 88.5% | 95.7% |
| sphere_D3_nopb | 0 | 40.5% | 35.0% | 21.1% | 24.5% | 91.0% | 84.2% |
| sphere_D3_x0lb | 0 | 39.0% | 35.6% | 42.9% | 37.3% | 92.2% | 95.4% |
| sphere_band_D2 | None | — | — | — | — | — | — |
| sphere_band_D3 | 0 | 22.0% | 21.0% | 36.4% | 16.8% | 82.5% | 99.9% |
| sphere_nonbox_D3 | 0 | 36.9% | 33.3% | 27.9% | 29.0% | 90.9% | 97.5% |
| timing_D5 | 0 | 0.0% | 0.0% | 0.2% | 0.2% | 83.6% | 100.0% |
| edgesphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ellipsoid_D3_homo | 1 | 33.9% | 29.6% | 53.8% | 42.3% | 97.8% | 0.9% |
| logsphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D1_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ellipsoid_D3_hetero | 2 | 17.5% | 11.6% | 32.9% | 29.6% | 93.6% | 0.0% |
| sphere_D1_hetero | 2 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

### Log priors, priors outside their bounds, small training sets

| configuration | level | NaN log priors | -inf log priors | NaN objectives | fits with the mean's prior outside its bounds | ... the covariance's | ... the noise's | runs with 2 or fewer distinct training points | hook errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 1.6% | 0 | 0 |
| ackley_D6 | 0 | 0 | 0 | 0 | 0.0% | 0.3% | 10.7% | 0 | 0 |
| edgesphere_D2 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| edgesphere_D4 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D10 | 0 | 0 | 0 | 0 | 0.0% | 3.3% | 14.2% | 0 | 0 |
| ellipsoid_D1_unbounded | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 1.4% | 0 | 0 |
| ellipsoid_D3_unbounded | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 1.1% | 0 | 0 |
| ellipsoid_D6 | 0 | 0 | 0 | 0 | 0.0% | 0.2% | 7.8% | 0 | 0 |
| logsphere_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| logsphere_D3_nopb | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| multisensory_s1_D6 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 3.9% | 0 | 0 |
| rastrigin_D1 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| rastrigin_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.6% | 0 | 0 |
| ridge_D2 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 18.5% | 0 | 0 |
| ridge_D4 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 16.6% | 0 | 0 |
| rosenbrock_D2 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| rosenbrock_D6 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 3.9% | 0 | 0 |
| sphere_D1 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D10 | 0 | 0 | 0 | 0 | 0.0% | 12.3% | 15.5% | 0 | 0 |
| sphere_D2 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_nopb | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_x0lb | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_band_D2 | None | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 30 | 0 |
| sphere_band_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 30 | 0 |
| sphere_nonbox_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.9% | 2 | 0 |
| timing_D5 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.3% | 0 | 0 |
| edgesphere_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| logsphere_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| multisensory_s1_D6_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D1_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D3_hetero | 2 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D1_hetero | 2 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_hetero | 2 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |

### Zero SDs: the value before the clamp and where they occur

Level 0, 19130122 points.

- floor(log10(|raw| / kss)): -16: 8417623, -15: 7400228, -14: 868906, -13: 331529, -12: 197465, -11: 110061, -10: 54634, -9: 28481, -8: 12923, -7: 5230, -6: 1828, -5: 554, -4: 184, -3: 47, -2: 37, 0: 1700392
- distance to the nearest training input (ell): 0: 16797, <1e-6: 90335, <1e-3: 14124807, <1e-1: 4893470, >=1e-1: 4713
- floor(log10(kss / effective noise)): 12: 54885, 13: 2215840, 14: 15653198, 15: 785052, 16: 160832, 17: 177392, 18: 70865, 19: 12058

Level 1, 9161531 points.

- floor(log10(|raw| / kss)): -16: 1537813, -15: 6958684, -14: 423425, -13: 31889, -12: 7670, -11: 2624, -10: 840, -9: 147, -8: 33, -7: 3, 0: 198403
- distance to the nearest training input (ell): 0: 8775, <1e-6: 292217, <1e-3: 7166633, <1e-1: 1693760, >=1e-1: 146
- floor(log10(kss / effective noise)): 12: 213619, 13: 5041459, 14: 3807409, 15: 40959, 16: 58085

Level 2, 5452745 points.

- floor(log10(|raw| / kss)): -16: 2241695, -15: 2690980, -14: 132464, -13: 32026, -12: 7773, -11: 1329, -10: 108, 0: 346370
- distance to the nearest training input (ell): 0: 6927, <1e-6: 458149, <1e-3: 4899195, <1e-1: 88474
- floor(log10(kss / effective noise)): 12: 7, 13: 2292083, 14: 3087545, 15: 73110
