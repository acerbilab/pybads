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
- **A zero spread of the training targets gives a degenerate prior**
  (W1-26, the rebuild case; needs MATLAB for what its fit then does).
  `gpdef/gpdefBads.m:219-222` sets the variance of the mean's prior to
  `yrange.^2/4` and `293-295` centres the output scale's prior at
  `log(std(y))`, which are zero and `-Inf` when the local targets are equal
  (a plateau); what the fit then does inside `gpHyperOptimize`'s `try` is
  not known without MATLAB. PyBADS computed the same at a rebuild, which
  gpyreg refuses; it now keeps the previous prior there (`cd1831f`;
  KD-B6-2).
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

- **With `HedgeGamma` 0, the search hedge fails at its first update**
  (W3-7, `verification/wave3.md`). `acq/acqPortfolio.m:40` scores every
  search at the same point, the search point taken as a row, its evident
  intent, and `:47` then predicts with `gpstructnew`, which is undefined
  there (its assignment was commented out before `d4fead5`), so a run with
  `HedgeGamma = 0` stops at its first search. Off by default (0.125).
  PyBADS failed too, earlier, by slicing the point's coordinates; it now
  scores each search at the search point (`4d357e4`).
- **A non-finite target prediction gives a NaN target** (W3-23).
  `bads.m:1310-1311` replace a non-finite prediction of the target by the
  incumbent's `fval` and `fsd`, but `1321` computes the target from the
  non-finite variance, so that the target is NaN (or `-Inf` for an infinite
  variance), and the poll then treats the GP as unreliable. In MATLAB this
  follows a failed rebuild, whose `post` is empty. PyBADS did the same, where
  gpyreg's predictions are non-finite only on overflow; it now computes the
  target from the incumbent's SD (`dac062e`).
- **`AccelerateMeshSteps` below 1 stops the run** (W3-39, from the
  doublecheck of wave 2). `bads.m:976-979` compare the incumbent with
  `iterList.fval(iter - AccelerateMeshSteps)`, and `iterList` starts empty
  (`private/setupvars.m:179-182`), so a value of 0 or less reads an
  iteration not recorded yet at the first failed poll. Off by default (3).
  PyBADS failed too, with `TypeError`; it now refuses a value that is not a
  positive integer when `BADS` is created (`5d711bf`), `inf` included,
  with which MATLAB runs without the accelerated reduction (`iter > Inf`
  never holds; KD-B4-6).
- **The prior of the length scales on two points has a zero width**
  (W3-40, found by W3-24's gate). `gpdef/gpdefBads.m:240-251` centre the
  empirical prior of the log length scales between the logs of the
  largest and the smallest pairwise distance of the training set, with
  half their difference as its SD, which is 0 when the local GP holds two
  distinct points; what the fit then does with a zero-variance prior is
  not known without MATLAB. PyBADS computed the same, and gpyreg 1.3.3
  refuses it with `ValueError`, which stopped the run; it now keeps the
  previous prior (`a14524d`; KD-B6-2).
- **A noisy run that ends in its first iteration takes none of the final
  samples it reserves** (W4-14, `verification/wave4.md`). `bads.m:1138`
  makes the final estimate only at `iter > 1`, so a noisy run that ends
  within its first iteration, on `MaxIter = 1` or on a `MaxFunEvals` that
  the initial design nearly uses up, leaves unused the evaluations it
  reserved for the final samples and reports the incumbent's single
  observation, with an `fsd` that at uncertainty level 1 is the default
  `NoiseSize` (`bads.m:448-452`, as the verifier read them). PyBADS had the
  same rule, over more budgets, since its design is larger (at D = 2, up to
  44 evaluations with the design alone, against MATLAB's 32, and up to 48
  with the first poll); it now takes the reserved samples at
  the incumbent, the run's only iterate, and reports their estimate
  (`b61a880`; KD-B2-8).
- **The search hedge's parameters are not checked** (W4-18, W4-29).
  `search/searchHedge.m:45-46` chooses a search with the probabilities
  `(1 - n γ) softmax(β g) + γ`, n the number of searches, and
  `acq/acqPortfolio.m:69` decays the gains by `HedgeDecay` at each update;
  none of `HedgeGamma`, `HedgeBeta` and `HedgeDecay` is checked, and every
  bad value runs without a warning. A `HedgeGamma` above `1/n` favors the
  search of lower gain, and above `1/(n - 1)` gives some searches a negative
  probability, so that they are never chosen; a negative `HedgeBeta`
  inverts the hedge, and an infinite or NaN one makes the probabilities NaN
  (PyBADS then chose at random every time); a `HedgeDecay` above 1 makes
  the gains grow until they overflow, and a negative one makes them
  alternate in sign. Off by default (0.125, `1e-3/TolFun` and
  `0.1^(1/(2*nvars))`). PyBADS ran with such values too; it now refuses, when
  `BADS` is created, a `hedge_gamma` outside `[0, 1/n]` (`6e24519`), a
  `hedge_beta` that is not a finite number at least 0 and a `hedge_decay`
  outside `[0, 1]` (`bd793f2`; KD-B3-8).
- **`Nsearchiter` is not checked** (W4-25). `private/setupoptions.m:26`
  evaluates it and `search/searchES.m:125` loops over `1:Nsearchiter`; what
  MATLAB does with 0, a negative value or a non-integer was not run. PyBADS
  stopped its run at the first search, with `ZeroDivisionError` for 0,
  `ValueError` for a negative value and `TypeError` for a float; it now
  refuses a value that is not a positive integer when `BADS` is created
  (`36e8b70`; KD-B4-6).

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
- **The covariance of ES-wcm is the unweighted scatter of the best points**
  (W3-3). `utils/ucov.m:19` sums the weighted copies of the scatter matrix
  of the best points about the incumbent, and the weights sum to one, so
  the result is the unweighted scatter, where the comments ("weighted
  covariance matrix"), the log weights and the name point to the weighted
  sum of the outer products. Weighting would change the normalized search
  covariance by 13-46% (median 33%) over 20 GP states; the final values of
  10 seeds on two problems did not move beyond their spread. PyBADS keeps
  the unweighted scatter, and its comments say so (`7e09887`).
- **At uncertainty level 0 the poll's GP does not take the poll's
  evaluations** (W3-26). Only a noisy run adds the poll's points to the GP
  (`bads.m:908-916`), so in a deterministic run, after an improving poll
  point, the target is predicted at `upollbest` (`841`), where the GP has
  no data (observed 4.155, predicted 86.41; observed 26.35, predicted
  11.44), and the LCB and the probabilities of improvement of the remaining
  poll points ignore the poll's observations. A GP that holds the points changed 1 of
  9 runs. PyBADS keeps MATLAB's behaviour.
- **The poll's basis is bounded by the inverse of LTMADS's ratio, so the
  poll steps along the coordinates** (W3-24). `poll/pollMADS2N.m:7` sets
  `Nmax = max(1, round(SearchMeshSize/MeshSize))`, the ratio of the search
  mesh to the poll mesh, which is below 1 at every default state (the
  locked search mesh is `2^(2k-10)` at the poll mesh `2^k`): `Nmax` is 1,
  the entries below the diagonal are 0, and the basis is a signed
  permutation of the identity, so that the poll is a coordinate search, as
  the user documents of both describe it. LTMADS (Audet and Dennis, 2006),
  whose basis the code draws, bounds it by the ratio of the poll size to
  the mesh size, and its tilted directions are dense in the limit. PyBADS
  tried LTMADS's bound, with the directions in units of the search mesh
  size (`869a033`), and reverted it (`b03a320`) after its gate: on
  PyBADS's benchmark the deterministic problems ended with higher errors,
  far below their tolerances (`ellipsoid_D6` flagged), a thin feasible band
  at D = 3 was solved in 23 of 30 runs instead of 30, and nonsmooth ridges
  along the diagonal, the case that the tilted directions address, did
  not improve (`verification/wave3.md`, "Fix pass"). PyBADS keeps MATLAB's
  poll.
- **The noise test's time counts as the optimizer's time** (W4-7).
  `private/evalinitmesh.m:41` calls the target directly for the noise test,
  and `private/funlogger.m:130` times only the logged calls, so the
  overhead that `bads.m:1186` reports counts the test's evaluation as the
  optimizer's time. PyBADS keeps this accounting (W2-20), and the
  description of `overhead` says so (`e744ed9`).

## Defects that PyBADS does not share

- **An empty search set moves to a stale point, or stops the run** (W3-11,
  `verification/wave3.md`). `bads.m:667-725`, `1257-1279`: a search set is
  empty when `uCheck` removes every candidate, as violating the constraint
  or already evaluated. At `ImprovementQuantile` > 0.5 in a noisy run, the
  empty set counts as an incremental search and moves the incumbent to the
  previous search's point `usearch`, with its `fval` and an SD of 0; when
  the run's first search set is empty, `usearch` is undefined and the run
  stops with an error.
  PyBADS: an empty set is a failed search on every path (W0-15, `0c56d86`),
  and the hedge's gains decay as MATLAB's (`4388e6d`; KD-B3-5).
- **The scale of the ES search becomes NaN after a generation without
  candidates** (found while verifying wave 3, B3 verifier, by reading).
  `search/searchES.m:170-193` updates the scale by the fraction of new
  candidates among the best, `nnew/ntest`, which is 0/0 when `uCheck`
  removed every candidate of the generation; from `Nsearchiter` 3 (the
  default is 2) the scale is then NaN, and `uCheck`'s projection, whose
  `min` and `max` ignore NaN, sends every later candidate of the search to
  the corner `UBsearch`. PyBADS keeps the scale after such a generation
  (`a77d95d`; KD-B3-6).
- **The ring of evaluations never writes its last row, and reads it**
  (W4-12, found while verifying). `private/funlogger.m:120-121` advances
  the row as `Xn = max(1, mod(Xn + 1, CacheSize))`, rows 1, 2, 3, 4, 1, …
  for a `CacheSize` of 5, so that its last row is never written, while
  `Xmax = min(Xmax + 1, CacheSize)` reaches it: once the ring has wrapped,
  `U(1:Xmax)` and `Y(1:Xmax)` (`utils/uCheck.m:23`,
  `private/gpupdate.m:30`) take in the row that `funlogger`'s `'init'` left
  unwritten (the verifier, by reading). Reached past 9999 logged
  evaluations at the default `CacheSize` of 1e4. PyBADS's log grows
  instead (KD-B7-4).
- **The rank-1 update takes a non-finite value before its penalty** (found
  while measuring the rank-1 update, by reading;
  [`dev/results/2026-09-28-where-pybads-spends-its-time.md`](../../results/2026-09-28-where-pybads-spends-its-time.md)).
  `private/gpupdate.m:55` passes `ystar` to `update_posterior`, and only
  at lines 69-78 replaces a non-finite `ystar` by the penalty, the largest
  target of the training set, in `gpstruct.y`; `update_posterior` raises
  nothing on a NaN, so the posterior would take the NaN while the training
  set records the penalty. Unreachable at the defaults:
  `private/funlogger.m:103-104` stops the run on a non-finite value, and
  `FitnessShaping`, which could make one, is off. PyBADS replaces the value
  before the update (`add_and_update_gp`), whose posterior it recomputes in
  full, and its function logger refuses non-finite values too.

## Questions that need MATLAB

- **The seed of the initial design** (W4-1). `init/initSobol.m:9-15` sets
  the skip index into the Sobol sequence to
  `mod(prod(uint64(num2str(u0(1:min(10,end))))), MaxSeed) + 1`, with
  `MaxSeed` 997. If `prod` of a `uint64` array returns a double, as the
  verifier read MATLAB's documentation, the product of a start that is not
  an integer exceeds `flintmax` from D = 2, and the seed then depends on how
  `mod` treats such a double: with an exact remainder, 378 to 395 seeds
  over 500 random starts; with the documented formula `x - floor(x./y).*y`
  or its round-off compensation, 1 for all of them, one design per D, as
  PyBADS had. Under the literal formula one transcribed seed came out
  negative, which `i4_sobol.m:249-250` clamps to 0, the start of the
  sequence (the verifier, unverified). The call that settles it:
  `mod(prod(uint64(num2str([0.25 -0.5]))), 997) + 1`, 966 for an exact
  remainder and 1 otherwise. PyBADS seeds its scrambled design from the
  run's generator, whatever MATLAB computes (`efe5e95`; KD-B7-1).
