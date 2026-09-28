# The GP layer's numerical health at `a4dcd65`, on Linux, gpyreg 1.3.3

Counts of what the GP layer does in PyBADS runs at default options: the
factorizations of the training covariance that fail and the noise
multiplier with which gpyreg rescues them, the posteriors that keep that
multiplier, the refits of `_robust_gp_fit_` and their failed tries, the
predictive SDs returned as exactly 0 and their value before gpyreg's clamp,
the NaN log priors and the priors outside their bounds, and the smallest
training sets. The findings and what they mean for the open items of
`dev/TODO.md` are in
[`results/2026-09-28-gp-health.md`](../../results/2026-09-28-gp-health.md).

## Command and provenance

```console
R=dev/scripts/runs/gp_health_20260928
for suite in default geometry oned bounds; do
  GP_HEALTH_OUT=$R/$suite/health \
  PYTHONPATH=<repository>/dev/scripts/gp_health_hooks:<gpyreg clone at v1.3.3> \
    .venv/bin/python -u dev/scripts/population.py run --suite $suite \
      --seeds 0-29 --workers 4 --out $R/$suite/pop
done
.venv/bin/python dev/scripts/gp_health.py summary $R/*/health \
  --pop $R/*/pop --md summary.md --csv runs.csv
```

- PyBADS at `b150b8a`, whose package code is `a4dcd65`'s (`dev-next`
  after #81), run from the main checkout. The counters are
  `dev/scripts/gp_health_hooks/sitecustomize.py` at `b150b8a`, loaded in
  every spawned run through `PYTHONPATH`; the tables are
  `dev/scripts/gp_health.py`.
- gpyreg 1.3.3 from a clone at the tag `v1.3.3` (`98ab5a4`), selected with
  `PYTHONPATH`.
- Linux (a cloud container, kernel `6.18.44-fc-v37`), Python 3.11.15,
  NumPy 2.4.6, SciPy 1.17.1: the environment of the Linux reference
  [`population_linux_wave4_20260927`](../population_linux_wave4_20260927/README.md),
  whose fingerprint (`4146a986863602cb`, one BLAS thread) the checkout
  printed before the runs. One BLAS thread per run, four runs at a time, a
  fresh process per run.
- The files: `summary.md` (the tables, all four suites) and `runs.csv`
  (one row per run with the counts the tables sum). The per-run counter
  files and the population records stay on the machine that ran them, under
  the gitignored `dev/scripts/runs/gp_health_20260928/`.

## The counters change nothing

The wrappers draw no random number and set nothing on the GP. The default
suite's 540 records equal the reference's field by field, the wall time
aside (`verification/scripts/wave3/orchestrator/same_fields.py` of the port
review against `population_linux_wave4_20260927`: only `final.wall_s`
differs, in every run). The counters add a median 11% to a run's wall time
(interquartile range 4% to 20%), which the shares of run time in
`summary.md` include. The other three suites have no reference on this
platform; their runs are the same code under the same counters.

## Outcome

The four suites at 30 seeds each (seeds 0-29): `default` (18
configurations, 540 runs, 26 minutes), `geometry` (7, 210), `oned` (6,
180) and `bounds` (5, 150). All 1,080 runs finished; none crashed.
`summary.md` has one table per topic, one row per configuration; its last
section sums the histograms of the zero SDs by level of uncertainty
handling. `sphere_band_D2`'s level is `None` there: its runs stop after 2
evaluations, with no search or poll prediction, so the counters never
read the level.

## Known gaps of the counters that wrote these files

The doublecheck of these records (a read-only review on 2026-09-28) found
four defects of the counters at `b150b8a`, fixed in
`dev/scripts/gp_health_hooks/sitecustomize.py` afterwards. None affects
the runs, which equal the reference's.

- `_mults` tested `gp.posteriors` for truth, which fails when it holds
  more than one hyperparameter sample: 81 returns of
  `set_hyperparameters`, in 60 runs (one to three each) of 11
  configurations (ackley_D6, ellipsoid_D10, ellipsoid_D3,
  ellipsoid_D3_unbounded, ellipsoid_D6, multisensory_s1_D6, rosenbrock_D6,
  ellipsoid_D3_homo, ridge_D2, ridge_D4, sphere_band_D3), are missing from
  the table of posteriors, each a hook error counted in the last table;
  tens of thousands of returns are counted. Rerun with the fix, seeds 0,
  10 and 11 of `ridge_D2` count their missing return, with no hook error
  and the same records.
- A `set_hyperparameters` that computes no posterior
  (`compute_posterior=False`, as `local_gp_fitting`, `_robust_gp_fit_` and
  the re-estimate call it) counted as a return with no inflated noise, so
  the column "after set_hyperparameters" understated the share; the tables
  now show a dash there for these counters, and no conclusion used it.
- The zero-SD recomputation solved over the zero points only, whose
  rounding differs from `predict`'s in a few points: on `ellipsoid_D3`,
  seed 0, all 6 values counted as positive before the clamp are that
  artefact, and the 1,528 of 33.7 million over all the runs are likely
  the same.
- The histogram of `|raw| / kss` wrote an exact 0 under the key `"0"`,
  which a value between 1 and 10 times `kss` would share; none occurred,
  and the tables read the key as an exact 0.
