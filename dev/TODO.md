# PyBADS: open work

Updated 2026-09-30. The next release is 1.5.0 (tag `v1.5.0`), the version
the PI has decided on; "the next release" below means it. `AGENTS.md`
("Setup and commands") gives the release's steps; its section below holds
the decision on Sto-BADS and the conda-forge recipe, which follows the
release's upload to PyPI.
`README.md`, the documentation and the skill on `dev-next` describe
PyBADS 1.5 with gpyreg 1.4.0, and `docs.yml` publishes the documentation
of `main`, so `dev-next` reaches `main` with the release. Other records
name the items by their titles, the port review's ledger
([results/2026-09-28-port-correctness-review.md](results/2026-09-28-port-correctness-review.md))
among them, so a title stays as it is while its item is open.

## The release of 1.5.0

- [ ] **Sto-BADS: experimental, not for users.** Sto-BADS (`stobads`,
  with `opp_stobads`, `stobads_frame_size_scaling_power` and the
  keyword-only argument `gamma_uncertain_interval` of `BADS`) is PyBADS's
  own, with no counterpart in MATLAB BADS (KD-S-1 of
  `pybads/bads/README.md`). The PI (2026-09-30) holds it experimental and
  wants it kept from users, who could take `stobads=True` for the right
  setting for a noisy target; what to do with it is to be decided. At
  default options it brings none of the benchmark's five noisy
  configurations closer to its minimum than BADS without it, and spends
  more evaluations on `sphere_D3_homo`
  ([results/2026-09-28-stobads-rule.md](results/2026-09-28-stobads-rule.md)).
  Users meet it on the options page, which includes
  `advanced_bads_options.ini` verbatim and whose description of `stobads`,
  there since 1.1.0, reads "if True switch to stochastic optimization and
  uncertain incumbent"; in `BADS`'s docstring, which documents
  `gamma_uncertain_interval`, undocumented in 1.1.0; and in the
  `Unreleased` section of `CHANGELOG.md`, the notes of the GitHub release,
  which names `gamma_uncertain_interval` under "Upgrading from 1.1.0" and
  the fixes to Sto-BADS under Fixed.
- [ ] **conda-forge recipe.** The test command of `conda-forge/pybads-feedstock`
  (`recipe/meta.yaml`) passes `--reruns=5` and requires
  pytest-rerunfailures. The tests of 1.1.0, which it runs, are not all
  seeded, so both stay until the first release after 1.1.0, whose tests
  are: drop them in the version-update PR that the feedstock's bot opens
  for that release, before it is merged. That PR needs gpyreg 1.4.0 on
  conda-forge, whose update PR on `conda-forge/gpyreg-feedstock` drops
  pytest, pytest-rerunfailures and numdifftools from the run
  requirements of its `recipe/meta.yaml` before it merges: gpyreg 1.4.0
  no longer needs them, and the feedstock's bot merges its update PR
  once a CI that only imports gpyreg passes.

## Waiting on MATLAB BADS

Each needs MATLAB and the BADS toolbox, and one session with them can serve
all three.

- [ ] **`ellipsoid_D3_hetero` after `020d6a8`.** Squaring the target's noise
  standard deviations, as MATLAB does, made the runs of this benchmark
  configuration worse: over 90 seeds the median error rose from 0.21 to
  0.54 on Windows and from 0.18 to 0.58 on Linux, mostly along the flat
  axis of the ellipsoid
  ([Windows](experiments/population_ellipsoid_hetero_20260925/README.md),
  [Linux](experiments/population_ellipsoid_hetero_linux_20260925/README.md)),
  while the spheres with target noise improved. The fix stays. Three
  differences from MATLAB, fixed on Linux, bring the median to 0.25 and
  the flat axis back to its error before `020d6a8`: the bound of the GP log
  length scales (`97b2c66`, the largest effect), a repeated point merged
  into another point's row of the function log (`032dfcb`), and the GP
  mean prior, re-centred at each rebuild (`8afbe16`). Returning the
  observation of a repeated point, as MATLAB's `funlogger` does, makes the
  runs worse, and the lower bound of the noise hyperparameter never moves,
  since no fit fails. The runs remain worse than before `020d6a8` (p =
  0.0008), now along the two steep axes. The port review fixed two more
  differences:
  - the evaluated points that `contraints_check` kept: W3-1 (`149d528`,
    wave 3) removes them, as MATLAB does; over seeds 0-29 of this
    configuration its 100 repeats (of 8800 evaluations, in 17 runs) are
    gone, and the median error moved from 0.43 to 0.36, unflagged
    (`experiments/port_review_20260925/verification/wave3_fixpass/`);
  - the bounds of the GP mean, which the port set from the initial design
    and MATLAB leaves infinite; once `8afbe16` re-centred the prior of the
    mean, it could fall outside them, which made the log prior NaN in the
    fits of 67 runs of the Windows population at `ab4dded`
    ([experiments/population_gpfixes_20260925/](experiments/population_gpfixes_20260925/README.md)).
    W1-23 (`172df00`, wave 1) leaves them infinite, as MATLAB does.

  Still open: a run of MATLAB BADS on this problem, which would show
  whether correct noise handling alone gives such runs. Measured on
  2026-09-28: its GP runs at an output variance near 1e15 times its noise
  variance, and most of its fits keep a noise that gpyreg multiplied
  (KD-B6-6); with gpyreg's switch to MATLAB's rule its errors are not
  measurably smaller, nor with any of the four Sto-BADS arms measured
  ([results/2026-09-28-gp-health.md](results/2026-09-28-gp-health.md),
  [results/2026-09-28-stobads-rule.md](results/2026-09-28-stobads-rule.md)).
- [ ] **Zero predictive SDs: how often MATLAB gives them.** The predictive
  SD of the GP is exactly 0 at about a tenth of the poll's acquisitions
  over the four suites, up to 40% on some configurations, noisy ones
  included, and each makes the poll's GP unreliable (W3-28). Counted and
  traced on 2026-09-28
  ([results/2026-09-28-gp-health.md](results/2026-09-28-gp-health.md)):
  rounding, the latent variance `kss - v'v` cancelled below the rounding
  of `kss` near the training inputs and clamped at 0, where the output
  variance exceeds the noise by 1e12 or more, the same cause as KD-B6-6;
  MATLAB's `mygp.m:187` clamps the same way. Open only: how often MATLAB's
  own fits reach them, which needs MATLAB.
- [ ] **Numerical oracles computed by MATLAB BADS.** The oracles of
  `pybads/testing/oracles/` are PyBADS's own numbers on stored states: they
  pin the numerics against change, not the port against MATLAB BADS.
  Oracles computed by MATLAB BADS on the same states would check the pure
  pieces that are not deliberate differences (`pybads/bads/README.md`):
  `transvars.m`, `udist.m`, `force2grid.m`, `ucov.m`, the priors of
  `gpdefBads.m` but the cases of KD-B6-2 and the centre of the mean's
  prior on one point (KD-B6-5), `acqLCB.m` with `gppred.m` at
  fixed hyperparameters, the ES search's weights, `searchHedge.m`'s update
  and `pollMADS2N.m` with injected draws. The fixtures are the inputs such
  a harness would take: plain arrays and JSON, with prescribed draws
  (`ScriptedGenerator` in `_oracles.py`), which can be handed to MATLAB as
  arrays. Generating the references needs MATLAB and the BADS toolbox.

## Not adopted

- [ ] **Rank-1 GP update when adding a point: not adopted, to revisit if
  its terms change.** MATLAB BADS adds a point to the GP by a rank-1
  update of the posterior (`private/gpupdate.m`, `utils/update_posterior.m`);
  PyBADS's `add_and_update_gp` recomputes every posterior in full, and says
  why at its call of `gp.update`. The measurement
  ([results/2026-09-28-where-pybads-spends-its-time.md](results/2026-09-28-where-pybads-spends-its-time.md))
  settled the questions: gpyreg's rank-1 path agrees with the full
  recomputation to rounding, target noise included (so there is no reason
  to skip it under `specify_target_noise`, as MATLAB does); it would save
  at most 2.5 % of PyBADS's own time, and is slower below about 60
  training points; but on the ellipsoids 40 to 44 % of the additions meet a
  GP whose factorization needed gpyreg's noise multiplier, where it
  differs from the recomputation by up to 1e-2 of the targets' spread, and
  by 0.21 when the multiplier it carries over differs from the one the
  recomputation picks. The PI did not adopt it (2026-09-28). Revisit if the
  training sets grow well beyond 200 points, if gpyreg's handling of the
  multiplier changes (KD-B6-6 of `pybads/bads/README.md`, kept as it is
  by the PI on 2026-09-28), or if a profile shows the update after
  a new point taking a larger share. Adopting it moves the ellipsoids'
  runs, so it needs the population comparison, and the target's reuse of
  the GP's own posterior (`_get_target_from_gp_`), which relies on
  posteriors computed in full, needs revisiting with it.

## Porting work

- [ ] **Benchmarking on neurobench**, open porting work listed in
  `pybads/bads/README.md`: PyBADS on cognitive and neural science models
  ([neurobench](https://github.com/lacerbi/neurobench)).
