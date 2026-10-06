# W0-1: the gate, and the investigation of its flags

W0-1 of the port review's wave 0 ([ledger](../verification/wave0.md)): at
the end of each iteration of a noisy run, PyBADS continued with a GP of the
iteration history as its working GP, and that GP was the history's slot
itself, so that later rebuilds changed it in place and later re-evaluations
of the iterates used drifted hyperparameters. The fix follows MATLAB BADS
(`bads.m:1091-1120`, `reevaluateIterList`): the working GP stays, and each
iterate is re-evaluated, from a copy of the working GP, with the
hyperparameters recorded at its iteration. Its commit was `d6e3f61`,
rebased onto `dev-next` as `7c73704` with the same package code.

## Provenance

Every population here was run on Windows 11 (Python 3.12.6, NumPy 2.5.3,
SciPy 1.18.1, one BLAS thread, one run at a time), gpyreg 1.3.3 from a
clone at the tag (`98ab5a4`), by the `population.py` of a clean detached
worktree at the commit named, from the main checkout's root:

```console
PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.3 .venv/Scripts/python.exe -u <worktree>/dev/scripts/population.py run --suite default --only <labels> --seeds <seeds> [--options '{"tol_stall_iters": 100000}'] --out <dir>
```

The commits: `010eeb4`, the fix pass of wave 0 without W0-1, whose package
code `dev-next` carries since #72 (the base); `d6e3f61`, W0-1 on top of it;
`76abaf1` and `6b5b2e6`, two variants of the base for the investigation,
local commits whose changes are `variant_a.patch` and `variant_b.patch`.

| Directory | Commit | Configurations × seeds |
|---|---|---|
| `w0_groupA_noisy`, `w0_groupA_det` | `010eeb4` | the five with noise × 0-3; `sphere_D2`, `ellipsoid_D3`, `rosenbrock_D2` × 0-1 |
| `w0_W0-1_noisy` (with `comparison.md`), `w0_W0-1_det` | `d6e3f61` | the five with noise × 0-29; the same three × 0-1 |
| `w01_base_ellhomo` | `010eeb4` | `ellipsoid_D3_homo` × 0-89 |
| `w01_a_ellhomo` | `76abaf1`, variant a: the old code with the swap on a deep copy of the stored GP, so that no slot is shared | the same |
| `w01_b_ellhomo` | `6b5b2e6`, variant b: the old code without the swap (the old re-evaluation, each stored GP rebuilt in place with its own geometry) | the same |
| `w01_c_ellhomo` | `d6e3f61` | the same |
| `w01_base_nostall`, `w01_c_nostall` | `010eeb4`, `d6e3f61` | the same, with `tol_stall_iters` 100000 |

`w01_analysis.py` prints the paired comparisons below from these records;
`compare_records.py` compares two directories run by run.

## The gate

At `010eeb4` the records equal those of the Windows reference
`population_gpfixes_20260925` in `x`, `fval`, `func_count` and `true_error`
(the fix pass of wave 0 moves no run of the benchmark; `iterations` and
the `fsd` with inferred noise differ, as #71 made them). At `d6e3f61`
(`comparison.md`, against the reference) the deterministic runs are
unchanged, and the five configurations with noise flag their number of
evaluations in three of them (`ellipsoid_D3_hetero`, `ellipsoid_D3_homo`,
`multisensory_s1_D6_homo`, 12 to 18% fewer), no error test: more runs end
on the stall criterion. The median error of `ellipsoid_D3_homo` rose from
0.053 to 0.099 over seeds 0-29 (signed-rank p = 0.017 before the Holm
correction), which the investigation took up.

## The investigation, `ellipsoid_D3_homo`, seeds 0-89

| Code | Median error [quartiles] | Solved | Median evaluations | Runs ended by `tol_fun` | Median paired log10 error ratio to the base (signed-rank p) |
|---|---|---|---|---|---|
| base | 0.076 [0.039, 0.139] | 0.63 | 380 | 17 | — |
| a, swap on a copy | 0.082 [0.052, 0.134] | 0.56 | 346 | 36 | +0.001 (0.24) |
| b, no swap | 0.087 [0.044, 0.187] | 0.54 | 318 | 40 | +0.101 (0.16) |
| c, W0-1 | 0.080 [0.039, 0.154] | 0.59 | 313 | 43 | +0.044 (0.22) |

The evaluations fall in all three (signed-rank p between 1.6e-11 and
6.2e-6). No variant changes the error significantly; the rise that seeds
0-29 showed for W0-1 is not in seeds 30-89 (median error 0.082 before,
0.078 after). Variant a, which only stops the history from sharing the
working GP, already doubles the runs that the stall criterion ends (17 to
36 of 90): the drifting estimates of the old code inflated the historic
improvement that the criterion tests.

With the stall criterion off (`*_nostall`), every run ends on `tol_mesh`:
W0-1 takes 330 evaluations to the base's 385 (median, p = 5.6e-14), at a
median error of 0.075 to 0.074 (median paired log10 ratio +0.06, p =
0.21). Over seeds 0-29 alone the error rises (p = 0.007), over seeds 30-89
it does not (+0.002, p = 0.79).

## Ruling

W0-1 stays as committed (PI, 2026-09-26): MATLAB's semantics, the drift
removed, noisy runs 12 to 18% shorter, and no change of the error that 90
seeds detect on the configuration whose error rose at 30. Not checked at
90 seeds: `multisensory_s1_D6_homo`, whose median error rose from 0.20 to
0.22 over seeds 0-29 (not significant, solved 0.90 to 0.80).
