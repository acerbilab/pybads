# PyBADS: open work

Updated 2026-10-01. The next release is 1.5.0 (tag `v1.5.0`), the version
the PI has decided on; "the next release" below means it. `AGENTS.md`
("Setup and commands") gives the release's steps. The PI (2026-09-30)
holds the release's pull request for further work before it. A change
before it that alters what a run prints or returns reruns the example
notebooks again, which also refreshes their timings: those of the rerun
at `110d8dc6` are longer than the previous rerun's from the state of the
machine, not the code. The release's section below holds a check of the
links to the lab before it, and the conda-forge recipe, which follows its
upload to PyPI.
`README.md`, the documentation and the skill on `dev-next` describe
PyBADS 1.5 with gpyreg 1.4.0, and `docs.yml` publishes the documentation
of `main`, so `dev-next` reaches `main` with the release. Other records
name the items by their titles, the port review's ledger
([results/2026-09-28-port-correctness-review.md](results/2026-09-28-port-correctness-review.md))
among them, so a title stays as it is while its item is open.

## The release of 1.5.0

- [ ] **Where the documentation points users.** The lab's website is
  https://acerbilab.org, and https://acerbilab.org/model-fitting is a
  landing page for the lab's model-fitting methods (a work in progress,
  functional on 2026-10-01). Before the release, a pass over what ships or
  is published decides which links go there instead (PI, 2026-10-01): the
  University of Helsinki group pages that `README.md`,
  `docsrc/source/index.rst` and `docsrc/source/about_us.rst` link for the
  lab and its people, `lacerbi.github.io` at the head of the FAQ, and the
  places that send a user to the lab's other methods: the FAQ's "What do I
  do if PyBADS is not suited for my problem?" and "I have run PyBADS on my
  problem. How do I run PyVBMC?", the runtime tips
  (`pybads/bads/_tip_catalog.py`) and the skill. The couplings of
  `AGENTS.md` apply: a tip restates the advice of the answer that it links,
  `README.md` and `index.rst` carry the same blocks, and a change to a tip
  changes what a run prints, which reruns the example notebooks.
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
  once a CI that only imports gpyreg passes. The same PR on
  `conda-forge/pybads-feedstock` raises the recipe's host requirement
  `setuptools >=45` to `>=77`, which the license
  field of `pyproject.toml` needs; conda-forge resolves the newest
  setuptools, so builds work meanwhile.

## Later releases

- [ ] **Sto-BADS: whether to remove it.** Sto-BADS (`stobads`, with
  `opp_stobads`, `stobads_frame_size_scaling_power` and the keyword-only
  argument `gamma_uncertain_interval` of `BADS`) is PyBADS's own,
  experimental, and brings no measured gain
  ([results/2026-09-28-stobads-rule.md](results/2026-09-28-stobads-rule.md)).
  The PI (2026-09-30) keeps it where users find it, on the options page
  since 1.1.0, but deprecated from 1.5.0, so that no user takes
  `stobads=True` for the setting of a noisy target (KD-S-1 of
  `pybads/bads/README.md`). Open: whether a later release removes it. Its
  options would then raise `ValueError` as unknown ones, as the 66 removed
  in 1.5.0 do, and `gamma_uncertain_interval` a `TypeError`, a break that
  needs a line under "Upgrading from"; only `stobads=True` warns now, so a
  script that sets another of these names meets the removal unwarned. The
  `improvement` oracle computes its `sto_flags` through
  `BADS._sto_success_improvement_` (`pybads/testing/oracles/_oracles.py`),
  and `dev/scripts/make_oracle_fixtures.py` reads
  `stobads_frame_size_scaling_power`: the removal changes that oracle's
  recipe, which takes `--write --reason` (`AGENTS.md`, "Numerical
  gates").
- [ ] **Tips: PyVBMC's review.** PyBADS's runtime tips copy PyVBMC's,
  whose tips the PI is to review before PyVBMC's release (PyVBMC's
  `dev/TODO.md`, "Review of the tips."). When that review concludes, its
  outcome for the policy or the wording is applied to PyBADS's copy
  (`pybads/bads/_runtime_tips.py`, `_tip_catalog.py`;
  `dev/plans/runtime-tips.md`). The review covers the old-release reminder
  as well, which shares the tips' slot and switch: its outcome reaches
  `pybads/bads/_release_reminder.py`, `pybads/_update_check.py`,
  `dev/plans/version-check.md`, the FAQ's "How do I know whether a newer
  version of PyBADS exists?", the API page of `check_for_updates` and the
  changelog's "Update reminders".

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

## The film

The narrated film about PyBADS 1.5, in `dev/film/` (its `NOTES.md` holds
the decisions and the open points).

- [ ] **The film's delivery**: its masters at 1920 x 1080, with captions
  and without, and its subtitles (`.srt` and `.vtt`) are made; a poster
  frame for the model-fitting page remains, and a web encode of the clean
  master if the page needs one smaller than the master.
- [ ] **The film's open picture points** (`dev/film/NOTES.md`, "Decisions",
  items 5 and 6): the labels that describe events, and a footnote crediting
  Bayesian optimization.
