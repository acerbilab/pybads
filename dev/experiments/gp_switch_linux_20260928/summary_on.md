### Refits of the local GP (`_robust_gp_fit_`)

A try is one `gp.fit`; it fails when a factorization of the objective raises (`LinAlgError`: after ten failures, or at the first under gpyreg's switch). A refit whose every try fails keeps the best of its starts (exit flag -1). The fits that raised include the initial fits of `init_and_train_gp`; the run time is the population record's `wall_s` (`--pop`).

| configuration | level | runs | refits | ok at the first try | ok after retries | every try failed | failed tries | fit time in fits that raised | fits that raised / run time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 30 | 159 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ackley_D6 | 0 | 30 | 622 | 99.8% | 0.2% | 0.0% | 2 | 0.5% | 0.1% |
| edgesphere_D2 | 0 | 30 | 143 | 46.9% | 53.1% | 0.0% | 235 | 43.5% | 18.9% |
| edgesphere_D4 | 0 | 30 | 270 | 82.6% | 17.0% | 0.4% | 164 | 31.8% | 12.3% |
| ellipsoid_D10 | 0 | 30 | 979 | 43.8% | 45.5% | 10.7% | 2867 | 67.7% | 36.9% |
| ellipsoid_D1_unbounded | 0 | 30 | 94 | 57.4% | 42.6% | 0.0% | 100 | 30.4% | 13.1% |
| ellipsoid_D3 | 0 | 30 | 1579 | 6.9% | 21.5% | 71.6% | 13366 | 90.6% | 59.5% |
| ellipsoid_D3_unbounded | 0 | 30 | 1482 | 6.7% | 23.7% | 69.6% | 12487 | 90.5% | 58.9% |
| ellipsoid_D6 | 0 | 30 | 1023 | 25.1% | 37.0% | 37.8% | 5837 | 81.4% | 46.4% |
| logsphere_D3 | 0 | 30 | 228 | 94.7% | 5.3% | 0.0% | 17 | 3.7% | 1.3% |
| logsphere_D3_nopb | 0 | 30 | 226 | 88.1% | 11.9% | 0.0% | 41 | 8.9% | 3.2% |
| multisensory_s1_D6 | 0 | 30 | 531 | 98.9% | 1.1% | 0.0% | 6 | 2.2% | 0.5% |
| rastrigin_D1 | 0 | 30 | 139 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| rastrigin_D3 | 0 | 30 | 459 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ridge_D2 | 0 | 30 | 544 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ridge_D4 | 0 | 30 | 829 | 99.9% | 0.1% | 0.0% | 1 | 0.3% | 0.0% |
| rosenbrock_D2 | 0 | 30 | 261 | 34.5% | 53.3% | 12.3% | 1084 | 71.9% | 33.8% |
| rosenbrock_D6 | 0 | 30 | 664 | 96.2% | 3.8% | 0.0% | 56 | 8.3% | 1.4% |
| sphere_D1 | 0 | 30 | 96 | 62.5% | 37.5% | 0.0% | 96 | 29.9% | 13.3% |
| sphere_D10 | 0 | 30 | 483 | 69.4% | 30.0% | 0.6% | 855 | 57.5% | 25.4% |
| sphere_D2 | 0 | 30 | 162 | 30.9% | 53.1% | 16.0% | 813 | 70.1% | 38.5% |
| sphere_D3_nopb | 0 | 30 | 314 | 21.0% | 34.7% | 44.3% | 2035 | 85.5% | 49.8% |
| sphere_D3_x0lb | 0 | 30 | 270 | 22.6% | 56.7% | 20.7% | 1559 | 76.2% | 48.7% |
| sphere_band_D2 | None | 30 | 0 | — | — | — | 0 | 0.0% | 0.0% |
| sphere_band_D3 | 0 | 30 | 150 | 34.7% | 64.7% | 0.7% | 443 | 55.6% | 29.8% |
| sphere_nonbox_D3 | 0 | 30 | 300 | 21.7% | 64.3% | 14.0% | 1637 | 76.9% | 47.9% |
| timing_D5 | 0 | 30 | 559 | 97.0% | 3.0% | 0.0% | 25 | 4.3% | 0.3% |
| edgesphere_D3_homo | 1 | 30 | 565 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ellipsoid_D3_homo | 1,None | 30 | 1163 | 6.9% | 38.0% | 55.1% | 9266 | 85.5% | 50.0% |
| logsphere_D3_homo | 1 | 30 | 726 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| multisensory_s1_D6_homo | 1 | 30 | 812 | 99.9% | 0.1% | 0.0% | 1 | 0.1% | 0.0% |
| sphere_D1_homo | 1 | 30 | 282 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 30 | 576 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 30 | 668 | 25.9% | 74.0% | 0.1% | 2186 | 59.5% | 25.5% |
| sphere_D1_hetero | 2 | 30 | 347 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 30 | 706 | 100.0% | 0.0% | 0.0% | 0 | 0.0% | 0.0% |

### Factorizations of the training covariance

Inflated: the factorization failed at least once and succeeded with the noise multiplied by ten per failure; raised: it failed ten times, or once under gpyreg's switch (`raise_on_cholesky_failure`). Low-noise repr.: the share of factorizations with the noise variance below 1e-6 (gpyreg's `L_chol = False`).

| configuration | level | objective evaluations | inflated | raised | posteriors | inflated | raised | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 15371 | 0.0% | 0.0% | 3107 | 0.0% | 0.0% | 6.2% |
| ackley_D6 | 0 | 31580 | 0.0% | 0.0% | 18819 | 0.0% | 0.0% | 20.6% |
| edgesphere_D2 | 0 | 31693 | 0.0% | 0.7% | 2701 | 0.0% | 0.1% | 31.2% |
| edgesphere_D4 | 0 | 32386 | 0.0% | 0.5% | 4862 | 0.0% | 0.0% | 42.0% |
| ellipsoid_D10 | 0 | 169300 | 0.0% | 3.2% | 38090 | 0.0% | 2.1% | 22.6% |
| ellipsoid_D1_unbounded | 0 | 21315 | 0.0% | 0.5% | 1718 | 0.0% | 0.0% | 18.9% |
| ellipsoid_D3 | 0 | 347788 | 0.0% | 8.8% | 11554 | 0.0% | 25.3% | 4.9% |
| ellipsoid_D3_unbounded | 0 | 337908 | 0.0% | 8.1% | 12027 | 0.0% | 21.7% | 5.9% |
| ellipsoid_D6 | 0 | 204141 | 0.0% | 4.4% | 22407 | 0.0% | 5.2% | 14.4% |
| logsphere_D3 | 0 | 23102 | 0.0% | 0.1% | 3827 | 0.0% | 0.0% | 34.5% |
| logsphere_D3_nopb | 0 | 24513 | 0.0% | 0.2% | 4086 | 0.0% | 0.0% | 34.2% |
| multisensory_s1_D6 | 0 | 31899 | 0.0% | 0.0% | 15044 | 0.0% | 0.0% | 27.6% |
| rastrigin_D1 | 0 | 15369 | 0.0% | 0.0% | 2795 | 0.0% | 0.0% | 13.4% |
| rastrigin_D3 | 0 | 27516 | 0.0% | 0.0% | 8409 | 0.0% | 0.0% | 17.1% |
| ridge_D2 | 0 | 29865 | 0.0% | 0.0% | 10562 | 0.0% | 0.0% | 17.4% |
| ridge_D4 | 0 | 40465 | 0.0% | 0.0% | 20200 | 0.0% | 0.0% | 14.2% |
| rosenbrock_D2 | 0 | 49326 | 0.0% | 2.7% | 4947 | 0.0% | 0.6% | 8.5% |
| rosenbrock_D6 | 0 | 37477 | 0.0% | 0.2% | 22364 | 0.0% | 0.0% | 21.6% |
| sphere_D1 | 0 | 21206 | 0.0% | 0.5% | 1754 | 0.0% | 0.0% | 20.3% |
| sphere_D10 | 0 | 70108 | 0.0% | 1.3% | 21471 | 0.0% | 1.1% | 38.6% |
| sphere_D2 | 0 | 54002 | 0.0% | 1.5% | 3036 | 0.0% | 0.2% | 19.2% |
| sphere_D3_nopb | 0 | 87250 | 0.0% | 2.4% | 5276 | 0.0% | 0.6% | 17.9% |
| sphere_D3_x0lb | 0 | 81185 | 0.0% | 2.0% | 4130 | 0.0% | 1.0% | 22.4% |
| sphere_band_D2 | None | 4111 | 0.0% | 0.0% | 30 | 0.0% | 0.0% | 0.0% |
| sphere_band_D3 | 0 | 39047 | 0.0% | 1.1% | 3603 | 0.0% | 0.0% | 24.4% |
| sphere_nonbox_D3 | 0 | 82075 | 0.0% | 2.0% | 5463 | 0.0% | 0.8% | 28.1% |
| timing_D5 | 0 | 35671 | 0.0% | 0.1% | 13200 | 0.0% | 0.0% | 32.7% |
| edgesphere_D3_homo | 1 | 31299 | 0.0% | 0.0% | 23733 | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_homo | 1,None | 699301 | 0.0% | 2.8% | 24441 | 0.0% | 20.1% | 0.1% |
| logsphere_D3_homo | 1 | 34715 | 0.0% | 0.0% | 33924 | 0.0% | 0.0% | 0.0% |
| multisensory_s1_D6_homo | 1 | 40130 | 0.0% | 0.0% | 62518 | 0.0% | 0.0% | 0.0% |
| sphere_D1_homo | 1 | 25391 | 0.0% | 0.0% | 11709 | 0.0% | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 32562 | 0.0% | 0.0% | 23640 | 0.0% | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 163484 | 0.0% | 2.8% | 25887 | 0.0% | 7.5% | 0.0% |
| sphere_D1_hetero | 2 | 25716 | 0.0% | 0.0% | 14388 | 0.0% | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 33377 | 0.0% | 0.0% | 33054 | 0.0% | 0.0% | 0.0% |

### Posteriors that keep an inflated noise

The share of returns of each method whose posterior keeps a noise multiplier above one, and of the acquisition's calls on such a GP. A dash for set_hyperparameters: counters that did not count its returns without a posterior apart.

| configuration | level | after fit | after update | after set_hyperparameters | largest multiplier (median over runs) | search predictions on one | poll predictions on one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ackley_D6 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| edgesphere_D2 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| edgesphere_D4 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D10 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D1_unbounded | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_unbounded | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D6 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| logsphere_D3 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| logsphere_D3_nopb | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| multisensory_s1_D6 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| rastrigin_D1 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| rastrigin_D3 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ridge_D2 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ridge_D4 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| rosenbrock_D2 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| rosenbrock_D6 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D1 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D10 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D2 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D3_nopb | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D3_x0lb | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_band_D2 | None | 0.0% | — | — | 1e+00 | — | — |
| sphere_band_D3 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_nonbox_D3 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| timing_D5 | 0 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| edgesphere_D3_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_homo | 1,None | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| logsphere_D3_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D1_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D3_homo | 1 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| ellipsoid_D3_hetero | 2 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D1_hetero | 2 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | — | 1e+00 | 0.0% | 0.0% |

### Zero predictive SDs (latent variance returned as exactly 0)

A poll acquisition with a zero SD makes the poll's `gamma_z` infinite and marks the GP unreliable (W3-28).

| configuration | level | poll acquisitions with one | poll points | search points | target predictions | negative before the clamp | low-noise repr. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ackley_D6 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| edgesphere_D2 | 0 | 6.0% | 5.0% | 11.7% | 14.6% | 62.4% | 100.0% |
| edgesphere_D4 | 0 | 13.9% | 10.2% | 8.7% | 6.3% | 49.3% | 100.0% |
| ellipsoid_D10 | 0 | 1.1% | 0.4% | 1.6% | 12.2% | 95.8% | 26.0% |
| ellipsoid_D1_unbounded | 0 | 1.3% | 0.8% | 2.0% | 0.3% | 64.3% | 100.0% |
| ellipsoid_D3 | 0 | 19.5% | 13.0% | 21.4% | 44.0% | 95.8% | 7.0% |
| ellipsoid_D3_unbounded | 0 | 13.3% | 8.0% | 15.2% | 40.3% | 96.8% | 1.4% |
| ellipsoid_D6 | 0 | 2.2% | 1.0% | 3.6% | 15.5% | 93.0% | 15.4% |
| logsphere_D3 | 0 | 0.2% | 0.1% | 0.0% | 0.0% | 13.3% | 100.0% |
| logsphere_D3_nopb | 0 | 2.3% | 1.8% | 1.4% | 1.0% | 84.4% | 100.0% |
| multisensory_s1_D6 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| rastrigin_D1 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| rastrigin_D3 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ridge_D2 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ridge_D4 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| rosenbrock_D2 | 0 | 2.5% | 1.8% | 3.3% | 3.2% | 75.0% | 44.8% |
| rosenbrock_D6 | 0 | 3.6% | 2.9% | 1.9% | 2.9% | 78.0% | 100.0% |
| sphere_D1 | 0 | 0.0% | 0.0% | 0.6% | 0.0% | 52.4% | 100.0% |
| sphere_D10 | 0 | 7.6% | 5.1% | 5.5% | 11.8% | 86.9% | 100.0% |
| sphere_D2 | 0 | 2.9% | 2.3% | 3.2% | 3.2% | 72.5% | 96.8% |
| sphere_D3_nopb | 0 | 12.8% | 11.8% | 9.4% | 12.4% | 77.0% | 8.7% |
| sphere_D3_x0lb | 0 | 7.7% | 5.9% | 5.1% | 7.1% | 65.9% | 51.3% |
| sphere_band_D2 | None | — | — | — | — | — | — |
| sphere_band_D3 | 0 | 1.2% | 0.9% | 1.6% | 1.4% | 48.5% | 100.0% |
| sphere_nonbox_D3 | 0 | 5.0% | 3.6% | 4.2% | 6.0% | 70.8% | 80.4% |
| timing_D5 | 0 | 0.0% | 0.0% | 0.0% | 0.1% | 40.5% | 100.0% |
| edgesphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ellipsoid_D3_homo | 1,None | 34.7% | 28.4% | 52.6% | 48.4% | 93.8% | 0.0% |
| logsphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| multisensory_s1_D6_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D1_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D3_homo | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| ellipsoid_D3_hetero | 2 | 7.7% | 4.4% | 17.3% | 23.6% | 88.8% | 0.0% |
| sphere_D1_hetero | 2 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| sphere_D3_hetero | 2 | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

### Log priors, priors outside their bounds, small training sets

| configuration | level | NaN log priors | -inf log priors | NaN objectives | fits with the mean's prior outside its bounds | ... the covariance's | ... the noise's | runs with 2 or fewer distinct training points | hook errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_D1 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 1.6% | 0 | 0 |
| ackley_D6 | 0 | 0 | 0 | 0 | 0.0% | 0.3% | 10.7% | 0 | 0 |
| edgesphere_D2 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| edgesphere_D4 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D10 | 0 | 0 | 0 | 0 | 0.0% | 2.2% | 13.7% | 0 | 0 |
| ellipsoid_D1_unbounded | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 4.2% | 0 | 0 |
| ellipsoid_D3_unbounded | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 4.8% | 0 | 0 |
| ellipsoid_D6 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 11.4% | 0 | 0 |
| logsphere_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| logsphere_D3_nopb | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| multisensory_s1_D6 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 3.9% | 0 | 0 |
| rastrigin_D1 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| rastrigin_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.6% | 0 | 0 |
| ridge_D2 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 18.5% | 0 | 0 |
| ridge_D4 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 16.5% | 0 | 0 |
| rosenbrock_D2 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.4% | 0 | 0 |
| rosenbrock_D6 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 4.0% | 0 | 0 |
| sphere_D1 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D10 | 0 | 0 | 0 | 0 | 0.0% | 12.2% | 16.7% | 0 | 0 |
| sphere_D2 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_nopb | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.5% | 0 | 0 |
| sphere_D3_x0lb | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_band_D2 | None | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 30 | 0 |
| sphere_band_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 30 | 0 |
| sphere_nonbox_D3 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 1.0% | 2 | 0 |
| timing_D5 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.3% | 0 | 0 |
| edgesphere_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D3_homo | 1,None | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| logsphere_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| multisensory_s1_D6_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D1_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_homo | 1 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| ellipsoid_D3_hetero | 2 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D1_hetero | 2 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| sphere_D3_hetero | 2 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |

### Zero SDs: the value before the clamp and where they occur

Level 0, 8912400 points.

- floor(log10(|raw| / kss)): exact0: 1156557, -16: 2851360, -15: 1689570, -14: 1178033, -13: 865816, -12: 587088, -11: 321689, -10: 145190, -9: 70769, -8: 32132, -7: 10883, -6: 2759, -5: 515, -4: 34, -3: 5
- distance to the nearest training input (ell): 0: 10304, <1e-6: 3075, <1e-3: 4451697, <1e-1: 4428507, >=1e-1: 18817
- floor(log10(kss / effective noise)): 12: 36822, 13: 982652, 14: 3425201, 15: 829591, 16: 822807, 17: 1639374, 18: 998959, 19: 176994

Level 1, 6348598 points.

- floor(log10(|raw| / kss)): exact0: 393382, -16: 2344527, -15: 3032389, -14: 409736, -13: 110694, -12: 38483, -11: 15563, -10: 3612, -9: 194, -8: 11, -7: 7
- distance to the nearest training input (ell): 0: 6137, <1e-6: 75668, <1e-3: 4435733, <1e-1: 1830974, >=1e-1: 86
- floor(log10(kss / effective noise)): 12: 49028, 13: 2659056, 14: 2903911, 15: 112596, 16: 624007

Level 2, 2693047 points.

- floor(log10(|raw| / kss)): exact0: 302192, -16: 1219774, -15: 1102912, -14: 31703, -13: 15234, -12: 10282, -11: 6117, -10: 3926, -9: 806, -8: 89, -7: 10, -6: 2
- distance to the nearest training input (ell): 0: 3772, <1e-6: 181165, <1e-3: 2441155, <1e-1: 66955
- floor(log10(kss / effective noise)): 13: 1709271, 14: 939370, 15: 44406
