## dev/experiments/one_point_gp_linux_20260928/warnings_base.jsonl

pybads /home/user/pybads/dev/scripts/runs/worktrees/p3_base/pybads, gpyreg /home/user/gpyreg/gpyreg/__init__.py

| configuration | runs | runs warned at init | runs warned in a refit | runs warned elsewhere | runs with a one-point refit | init on one point | init mean = its target |
|---|---|---|---|---|---|---|---|
| `sphere_band_D2` | 30 | 30 | 0 | 0 | 0 | 30 | 0 |
| `sphere_band_D3` | 30 | 30 | 5 | 0 | 0 | 30 | 0 |
| `sphere_band_D2_homo` | 30 | 28 | 0 | 0 | 0 | 28 | 0 |
| `sphere_band_D3_homo` | 30 | 29 | 0 | 0 | 0 | 29 | 0 |
| `sphere_band_D2_hetero` | 30 | 28 | 0 | 0 | 0 | 28 | 0 |
| `sphere_band_D3_hetero` | 30 | 29 | 0 | 0 | 0 | 29 | 0 |

| where | warning | runs |
|---|---|---|
| init | RuntimeWarning: Degrees of freedom <= 0 for slice | 174 |
| init | RuntimeWarning: divide by zero encountered in log | 174 |
| init | RuntimeWarning: invalid value encountered in divide | 174 |
| refit | RuntimeWarning: divide by zero encountered in log | 5 |

## dev/experiments/one_point_gp_linux_20260928/warnings_change.jsonl

pybads /home/user/pybads/dev/scripts/runs/worktrees/p3_change/pybads, gpyreg /home/user/gpyreg/gpyreg/__init__.py

| configuration | runs | runs warned at init | runs warned in a refit | runs warned elsewhere | runs with a one-point refit | init on one point | init mean = its target |
|---|---|---|---|---|---|---|---|
| `sphere_band_D2` | 30 | 0 | 0 | 0 | 0 | 30 | 30 |
| `sphere_band_D3` | 30 | 0 | 30 | 0 | 0 | 30 | 30 |
| `sphere_band_D2_homo` | 30 | 0 | 0 | 0 | 0 | 28 | 28 |
| `sphere_band_D3_homo` | 30 | 0 | 0 | 0 | 0 | 29 | 29 |
| `sphere_band_D2_hetero` | 30 | 0 | 3 | 0 | 0 | 28 | 28 |
| `sphere_band_D3_hetero` | 30 | 0 | 0 | 0 | 0 | 29 | 29 |

| where | warning | runs |
|---|---|---|
| refit | RuntimeWarning: divide by zero encountered in log | 33 |

## dev/experiments/one_point_gp_linux_20260928/warnings_band1_base.jsonl

pybads /home/user/pybads/dev/scripts/runs/worktrees/p3_base/pybads, gpyreg /home/user/gpyreg/gpyreg/__init__.py

| configuration | runs | runs warned at init | runs warned in a refit | runs warned elsewhere | runs with a one-point refit | init on one point | init mean = its target |
|---|---|---|---|---|---|---|---|
| `band1` | 30 | 27 | 27 | 0 | 27 | 27 | 0 |

| where | warning | runs |
|---|---|---|
| init | RuntimeWarning: Degrees of freedom <= 0 for slice | 27 |
| init | RuntimeWarning: divide by zero encountered in log | 27 |
| init | RuntimeWarning: invalid value encountered in divide | 27 |
| refit | RuntimeWarning: Degrees of freedom <= 0 for slice | 27 |
| refit | RuntimeWarning: divide by zero encountered in log | 27 |
| refit | RuntimeWarning: invalid value encountered in divide | 27 |

## dev/experiments/one_point_gp_linux_20260928/warnings_band1_change.jsonl

pybads /home/user/pybads/dev/scripts/runs/worktrees/p3_change/pybads, gpyreg /home/user/gpyreg/gpyreg/__init__.py

| configuration | runs | runs warned at init | runs warned in a refit | runs warned elsewhere | runs with a one-point refit | init on one point | init mean = its target |
|---|---|---|---|---|---|---|---|
| `band1` | 30 | 0 | 27 | 0 | 27 | 27 | 27 |

| where | warning | runs |
|---|---|---|
| refit | RuntimeWarning: Degrees of freedom <= 0 for slice | 27 |
| refit | RuntimeWarning: divide by zero encountered in log | 27 |
| refit | RuntimeWarning: invalid value encountered in divide | 27 |
