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
  records the refit.
- **At D = 1 the GP length scale used for the training set is 1**
  (W1-17). `private/gpupdate.m:285-292` takes the ARD length scales only
  when `gpstruct.ncovlen > 1`, the test meant to tell per-dimension length
  scales from an isotropic one; at D = 1 the ARD kernel has a single length
  scale, so the distances that choose the training set are not in its
  units. The effect is bounded (the training set has between `NtrainMin`
  and `NtrainMax` points). PyBADS: the fitted length scale at every D.

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
