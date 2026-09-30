| configuration | seed | A rep 1 | A rep 2 | B rep 1 | B rep 2 | B / A | same result |
|---|---|---|---|---|---|---|---|
| ackley_D6 | 0 | 3.92 | 3.74 | 2.57 | 2.54 | 0.679 | yes |
| ackley_D6 | 1 | 3.65 | 4.25 | 2.94 | 2.81 | 0.770 | yes |
| ackley_D6 | 2 | 3.79 | 3.73 | 2.68 | 2.87 | 0.719 | yes |
| ellipsoid_D10 | 0 | 12.95 | 13.25 | 8.46 | 8.94 | 0.653 | yes |
| ellipsoid_D10 | 1 | 15.90 | 15.80 | 9.79 | 9.70 | 0.614 | yes |
| ellipsoid_D10 | 2 | 18.89 | 18.88 | 11.25 | 11.36 | 0.596 | yes |
| ellipsoid_D3 | 0 | 2.98 | 3.05 | 2.05 | 2.19 | 0.688 | yes |
| ellipsoid_D3 | 1 | 2.40 | 2.39 | 1.70 | 1.68 | 0.701 | yes |
| ellipsoid_D3 | 2 | 2.00 | 1.94 | 1.38 | 1.44 | 0.711 | yes |
| ellipsoid_D3_homo | 0 | 10.37 | 9.72 | 6.97 | 7.04 | 0.717 | yes |
| ellipsoid_D3_homo | 1 | 10.76 | 11.27 | 7.72 | 7.84 | 0.718 | yes |
| ellipsoid_D3_homo | 2 | 9.09 | 8.91 | 6.91 | 9.81 | 0.775 | yes |
| multisensory_s1_D6_homo | 0 | 7.09 | 7.12 | 4.75 | 5.11 | 0.670 | yes |
| multisensory_s1_D6_homo | 1 | 13.26 | 13.06 | 9.26 | 9.02 | 0.691 | yes |
| multisensory_s1_D6_homo | 2 | 13.03 | 12.69 | 8.58 | 10.20 | 0.676 | yes |
| rosenbrock_D6 | 0 | 5.11 | 4.81 | 3.42 | 3.39 | 0.704 | yes |
| rosenbrock_D6 | 1 | 6.11 | 5.73 | 3.96 | 3.92 | 0.684 | yes |
| rosenbrock_D6 | 2 | 4.56 | 4.28 | 3.61 | 3.48 | 0.814 | yes |
| sphere_D3_hetero | 0 | 3.65 | 3.93 | 2.68 | 3.28 | 0.733 | yes |
| sphere_D3_hetero | 1 | 4.71 | 4.71 | 4.13 | 3.29 | 0.699 | yes |
| sphere_D3_hetero | 2 | 4.62 | 4.30 | 3.14 | 3.17 | 0.731 | yes |

| configuration | median B / A |
|---|---|
| ellipsoid_D10 | 0.614 |
| multisensory_s1_D6_homo | 0.676 |
| ellipsoid_D3 | 0.701 |
| rosenbrock_D6 | 0.704 |
| ellipsoid_D3_homo | 0.718 |
| ackley_D6 | 0.719 |
| sphere_D3_hetero | 0.731 |

Own time in seconds; B / A is the faster run of B over the faster of A. Median over all 21 pairs: 0.701; configuration medians 0.614 to 0.731.
