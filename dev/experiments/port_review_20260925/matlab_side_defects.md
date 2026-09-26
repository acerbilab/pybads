# Defects and design observations on the MATLAB side

The port review ([plan](../../plans/port-correctness-review.md)) reads PyBADS
against MATLAB BADS at `74919c0` (v1.1.3). What it finds wrong or
questionable in MATLAB BADS itself, whether or not PyBADS shares it, is
collected here, for a developer of MATLAB BADS and for the close of the
review. Each item names the MATLAB lines at `74919c0`, what is wrong, the
evidence (the ledger row that verified it) and what PyBADS does about it.
Nothing here was run in MATLAB.

## Shared defects that PyBADS fixes

- **With `PollTraining` off, the poll records a refit that it then
  cancels** (W1-8, `verification/wave1.md`). `bads.m:822-823`: `IsRefitTime`
  resets `lastfitgp` and the GP statistics, and the poll then drops the
  refit when `PollTraining` is off, so the next refit of the search waits
  for the minimum refit time counted from one that did not happen. Off by
  default. PyBADS: with `poll_training` off, the poll neither performs nor
  records the refit (`c9ebdc7`).
- **At D = 1 the GP length scale used for the training set is 1**
  (W1-17). `private/gpupdate.m:285-292` takes the ARD length scales only
  when `gpstruct.ncovlen > 1`, the test meant to tell per-dimension length
  scales from an isotropic one; at D = 1 the ARD kernel has a single length
  scale, so the distances that choose the training set are not in its
  units. The effect is bounded (the training set has between `NtrainMin`
  and `NtrainMax` points). PyBADS: the fitted length scale at every D (`cfacb98`; on
  a 1-D suite of 30 seeds it changes 43 of 180 runs, with no flag).

- **The transform's self-test refuses valid bounds of large magnitude**
  (W2-5, `verification/wave2.md`). `utils/transvars.m:30`, `169-178` check
  that the inverse of the transform returns each bound within an absolute
  1e-6, which rounding alone exceeds from bounds of about 1e10 (an upper
  bound of about 1e9 for a log-scaled variable), where the transform is
  exact to machine precision: BADS refuses such a problem as if its bounds
  could not be transformed. PyBADS: a tolerance of `1e-6 · max(1, |b|)`,
  which accepts only bounds refused before (`a2b8d38`; KD-B1-10).
- **A random start that violates the non-box constraints stops the run**
  (W2-11). `private/setupvars.m:83-85` draws a start that is not given in
  the plausible box, without regard to `NONBCON`, and
  `private/evalinitmesh.m:22-26` then stops with an error when it violates
  the constraint: 4 of 20 seeds on a disc in a square. PyBADS:
  draws again, up to 1000 draws, before the same error; a run whose first
  draw is feasible is unchanged (`c3d7815`; KD-B1-11).
- **The noise test is left out of the budget** (W2-27).
  `private/evalinitmesh.m:37-42` evaluates `x0` a second time when
  `UncertaintyHandling` is empty, and `98-104` caps the initial design at
  `MaxFunEvals - 1` without counting it, so a budget below the design is
  exceeded by one evaluation (D = 2 with `MaxFunEvals` 3: 4 evaluations).
  PyBADS: counts it, and keeps the design within the evaluations left
  (`381bf32`; KD-B2-6).
- **After the re-estimate, the move to an earlier iterate takes its value
  and not its location** (W2-25). `bads.m:1111-1118` set `u`, `yval`,
  `fval`, `fsd` and the target's hyperparameters from the chosen iterate
  and leave `ubest` at the old incumbent, and `769` sets `u = ubest` at the
  next iteration: the next search's target is predicted at the old point,
  and a poll that no successful search precedes runs around the old point
  while it is judged by the chosen iterate's value (9 of 22 polls of 6
  noisy runs in the verifier's check). PyBADS: the incumbent moves with its
  value (`a9fbb97`; KD-B2-7).

## Shared design observations (PyBADS keeps MATLAB's behavior)

- **The calibration test for three or more points tests normality only**
  (W1-7). `utils/gppredcheck.m:30` applies `swtest`, a test of normality
  with unspecified mean and variance, to the z-scores of the GP's
  predictions, so a GP whose predictions are biased, or whose standard
  deviations are off by a common factor, passes; the chi-square test for one
  or two points is scale-sensitive. A test of the scale (chi-square on the
  sum of the squared z-scores, at every n) would check what the comments
  call calibration.
- **The noise of the GP is bounded above at a log SD of 5** (W1-30).
  `gpdef/gpdefBads.m:161` bounds the log noise SD by the constant 5 (SD
  about 148), whatever the scale of the target, so a target with noise
  larger than that cannot be represented, and a `NoiseSize` above it is a
  prior outside its bounds; the fitted noise then sits at the bound. PyBADS
  keeps the bound and warns when `BADS` is created with such a
  `noise_size`.
- **A zero spread of the training targets gives a degenerate prior**
  (W1-26, the rebuild case; needs MATLAB). `gpdef/gpdefBads.m:219-222` sets
  the variance of the mean's prior to `yrange.^2/4` and `293-295` centres
  the output scale's prior at `log(std(y))`, which are zero and `-Inf` when
  the local targets are equal (a plateau); what the fit then does inside
  `gpHyperOptimize`'s `try` is not known without MATLAB. PyBADS keeps the
  previous prior in that case (KD-B6-2).

- **A noisy run's first incumbent is the raw minimum of its initial
  design** (W2-36). The re-estimate starts at `iter > 1` (`bads.m:1097`),
  so for two iterations the incumbent of a noisy run is the lowest noisy
  observation of the design, a biased order statistic, with `fsd` set to
  `NoiseSize`, and the searches and polls of those iterations are judged
  against it. PyBADS keeps MATLAB's behaviour.
- **A feasible region thinner than the mesh can resolve ends the run on
  its stall criterion** (W2-37; needs MATLAB for the GP on one point). With
  `|x1 - x2| <= 0.005` as `NONBCON` at D = 2, no point of the initial design
  is feasible, the GP is trained on `x0` alone (`log(std(y))` is `-Inf`),
  no search runs, the axis-aligned poll points are all infeasible, and the
  stall criterion, which counts iterations without an evaluation, ends the
  run at `x0`. PyBADS keeps the criterion, and the documentation of
  `non_box_cons` says that such a region can end the run early and
  suggests a reparametrization (`bdaef58`).
