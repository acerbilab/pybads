# gpyreg's `raise_on_cholesky_failure` in PyBADS at `a4dcd65`, on Linux

W1-25's switch measured at the head of the port review: every GP of a run
constructed with `raise_on_cholesky_failure=True`, MATLAB BADS's rule of an
error at the first failed factorization, against the same gpyreg with the
switch off, paired by seed. The findings are in
[`results/2026-09-28-gp-health.md`](../../results/2026-09-28-gp-health.md),
"gpyreg's switch at the head".

## Command and provenance

```console
R=dev/scripts/runs/gp_switch_20260928
for arm in on off; do   # GP_FORCE_RAISE_ON_CHOLESKY_FAILURE=1 for "on" only
  for suite in default geometry oned bounds; do
    GP_HEALTH_OUT=$R/$arm/$suite/health \
    PYTHONPATH=<repository>/dev/scripts/gp_health_hooks:<gpyreg worktree at 1893eff> \
      .venv/bin/python -u dev/scripts/population.py run --suite $suite \
        --seeds 0-29 --workers 4 --out $R/$arm/$suite/pop
  done
done
python dev/scripts/population.py compare $R/off/<suite>/pop $R/on/<suite>/pop
```

- PyBADS at `f8a1cad`, whose package code is `a4dcd65`'s, from the main
  checkout. The switch is set by the knob
  `GP_FORCE_RAISE_ON_CHOLESKY_FAILURE=1` of
  `dev/scripts/gp_health_hooks/sitecustomize.py` (`f8a1cad`), which wraps
  `GP.__init__`; PyBADS is unchanged. The GP-health counters run in both
  arms.
- gpyreg `main` at `1893eff` (after 1.3.3: acerbilab/gpyreg#56, the switch;
  #57, the log prior mass; #58, the fit that starts no optimization from a
  design point whose factorization failed; #59, documentation), from a
  worktree selected with `PYTHONPATH`. With the switch off,
  `dev/scripts/fingerprint.py` prints `4146a986863602cb`, the hash of gpyreg
  1.3.3 and of the Linux reference, with and without the counters; with it
  on, `5991322b841c122c`.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1, one BLAS thread per run, four runs at a time.
- The files: `compare_<suite>.md`, the comparison of each suite, "off" as
  REF and "on" as NEW; `summary_on.md` and `summary_off.md`, the counters'
  tables of each arm over the four suites.
- The per-run records and counter files stay on the machine that ran
  them, under the gitignored `dev/scripts/runs/gp_switch_20260928/`.

## "off" is gpyreg 1.3.3

With the switch off, gpyreg `main` gives the records of gpyreg 1.3.3 in all
1,080 runs, the wall time aside: the `default` suite against the Linux
reference `population_linux_wave4_20260927`, the other three against the
runs of [`gp_health_linux_20260928`](../gp_health_linux_20260928/README.md)
(`verification/scripts/wave3/orchestrator/same_fields.py` of the port
review). The comparisons with "off" are therefore also comparisons with
1.3.3 and with the Linux reference.

## Outcome

All runs of "off" finished; with the switch on, 3 runs of
`ellipsoid_D3_homo` (seeds 23, 26 and 29) stop at their start with
`RuntimeError: bads:gp: The initial fit of the GP failed 10 times`. The
comparisons flag 9 of the 36 configurations (`default` 6 of 18, `geometry`
1 of 7, `oned` 0 of 6, `bounds` 2 of 5); every configuration with a
changed run is below, "off" → "on", medians over the 30 seeds, runs above
the configuration's tolerance, the median paired ratio of wall times (both
arms under the counters).

| configuration | changed runs | evaluations | above tolerance | crashed | wall time | flag |
| --- | --- | --- | --- | --- | --- | --- |
| ellipsoid_D3 | 30 | 148 → 178 | 0 → 6 | 0 | ×1.76 | evaluations |
| ellipsoid_D3_unbounded | 30 | 150 → 188 | 0 → 4 | 0 | ×1.85 | evaluations |
| ellipsoid_D6 | 30 | 347 → 404 | 0 → 0 | 0 | ×1.24 | evaluations |
| ellipsoid_D10 | 30 | 668 → 714 | 0 → 0 | 0 | ×1.31 | evaluations |
| ellipsoid_D3_homo | 30 | 312 → 266 | 17 → 6 of 27 | 3 | ×1.11 | evaluations; crashes |
| ellipsoid_D3_hetero | 30 | 302 → 281 | 23 → 25 | 0 | ×1.06 | — |
| sphere_D2 | 30 | 55 → 55 | 0 → 0 | 0 | ×1.07 | error, lower |
| sphere_nonbox_D3 | 30 | 98 → 98 | 0 → 0 | 0 | ×1.17 | — (error lower) |
| sphere_D10 | 30 | 454 → 468 | 0 → 0 | 0 | ×1.17 | — |
| sphere_D3_nopb | 30 | 98 → 98 | 0 → 0 | 0 | ×1.21 | error, lower |
| sphere_D3_x0lb | 30 | 77 → 87 | 0 → 0 | 0 | ×1.16 | error, lower |
| edgesphere_D2 | 18 | 48 → 48 | 0 → 0 | 0 | ×1.03 | error, lower |
| rosenbrock_D2 | 30 | 94 → 83 | 0 → 0 | 0 | ×0.81 | — |
| rosenbrock_D6 | 18 | 415 → 415 | 5 → 4 | 0 | ×0.96 | — |
| sphere_band_D3 | 28 | 57 → 56 | 0 → 0 | 0 | ×1.07 | — |
| edgesphere_D4, ellipsoid_D1_unbounded, sphere_D1, logsphere_D3, logsphere_D3_nopb, timing_D5, multisensory_s1_D6, multisensory_s1_D6_homo, ridge_D4 | 1 to 19 | unchanged medians | unchanged | 0 | ×0.94 to ×1.05 | — |

The median paired log10 error ratios of the flagged errors ("on" over
"off"): `sphere_D2` −0.96, `sphere_D3_nopb` −1.11, `sphere_D3_x0lb`
−0.83; unflagged, `sphere_nonbox_D3` −0.50 and `ellipsoid_D3_homo` −0.38
[−0.71, −0.15] on its 27 pairs. Every run of the spheres, in both arms,
ends within its tolerance of 1e-3. The counters of "on"
(`summary_on.md`): on `ellipsoid_D3`, 72% of the refits fail all ten tries
(6.5% with the switch off) and the failed tries take 60% of the run time;
`ellipsoid_D6` 38%, `sphere_D2` 16%, `ellipsoid_D3_homo` 55%.
