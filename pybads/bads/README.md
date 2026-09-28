# PyBADS and MATLAB BADS: deliberate differences and open porting work

PyBADS is the Python port of MATLAB BADS (`acerbilab/bads`). This file
catalogues where PyBADS differs from MATLAB BADS v1.1.3 (`74919c0`) on
purpose, and lists the porting work that is still open. A difference that
is not listed here is not known to be deliberate: it is a defect until
shown otherwise.

The catalogue comes from the port correctness review of 2026-09-25 to
09-28, which compared the port with MATLAB BADS line by line, slice by
slice (`dev/results/2026-09-28-port-correctness-review.md`). Each entry
keeps the identifier it had there (`KD-<slice>-<n>`; the slices are those of
the review, B1 to B7, S and T), so that the review's ledgers, the survey and
`dev/TODO.md` can cite it; the rulings it names are rows of those ledgers
(`W<wave>-<n>`, in
`dev/experiments/port_review_20260925/verification/wave<wave>.md`).
PyBADS is cited by module and function, MATLAB BADS by file and lines at
`74919c0`, relative to the root of its repository. A change that adds,
removes or alters a deliberate difference updates its entry here.

Kinds: *deliberate change* (PyBADS does it otherwise), *unported feature*
(MATLAB BADS has it, PyBADS does not), *removed feature* (neither side
supports it, or PyBADS dropped it), *substituted library*, *Python-only
feature*.

## Open porting work

* Support for periodic variables (KD-B1-6).
* Benchmark PyBADS on cognitive and neural science models
  ([neurobench](https://github.com/lacerbi/neurobench)).

`dev/TODO.md` holds the other open work, among it the unported features
below that have an item of their own: `gp_cov_prior="ard"` (KD-B6-7) and
the prior evaluations of `fun_values` (KD-B1-4).

## Deliberate differences

### Setup, options, bounds, transform and result (B1)

**KD-B1-1. Every random draw comes from one generator, created from `random_seed`.**
MATLAB BADS has no seed option: it draws with `rand`, `randn`, `randi` and
`randperm` from MATLAB's global stream and reports the stream's state.
PyBADS creates one `numpy.random.Generator`, `bads.rng`, when `BADS` is
created, from `random_seed`, passes it to every draw, and never draws from
NumPy's global stream, except that `random_seed=None` seeds the generator
from four draws of it. The scrambling of the initial design draws from a
generator that SciPy seeds with one integer drawn from `bads.rng`
(KD-B7-1). The result reports `random_seed`, the option when it is an
integer and `None` otherwise, not a state. No draw is meant to reproduce
MATLAB's numbers; what is drawn, and from which distribution, follows
MATLAB BADS.
- PyBADS: `pybads/rng.py` (`get_rng`); `BADS.__init__` and `_init_rng_`
  (`pybads/bads/bads.py`); the draws of `poll_mads_2n`, `ESSearchHedge`,
  `ESSearch`, `init_sobol`, and the GP's fits and slice sampler in
  `pybads/bads/gaussian_process_train.py`.
- MATLAB: `private/setupvars.m:83`; `bads.m:589`, `857`;
  `search/searchES.m:117`, `168`, `201`; `search/searchHedge.m:48`, `50`;
  `poll/pollMADS2N.m:10`, `14`, `17`; `private/gpupdate.m:374`, `392`;
  `utils/gppriorrnd.m:75`; `private/bads_output.m:25`.
- Settled by: `dev/plans/tooling-and-rng.md`; the initial design's seed,
  W4-1 (KD-B7-1). Kind: deliberate change, with a Python-only option.

**KD-B1-2. Options live in two `.ini` files, with snake_case names.**
The options are split between `basic_bads_options.ini` and
`advanced_bads_options.ini`; MATLAB's CamelCase names become snake_case,
and some are renamed: `Ninit` → `fun_eval_start`, `Ndata` → `n_train_max`,
`MinNdata` → `n_train_min`, `BufferNdata` → `buffer_ntrain`,
`MeshOverflowsWarning` → `mesh_overflow_warning`, `Nsearch` → `n_search`,
`Nsearchiter` → `n_search_iter`, `Nbasis` → `n_basis`, `TolPoI` →
`tol_poi`, `ESbeta` and `ESstart` → `es_beta` and `es_start`,
`gpSVGDiters` → `gp_svd_iters`, `NormAlphaLevel` → `normalpha_level`,
`InitFcn` → `init_fun`, `gpdefFcn` → `gp_def_fcn`, and `gp*` → `gp_*`.
`PeriodicVars` and `OutputFcn` are basic in MATLAB BADS and advanced in
PyBADS. Function handles become strings (`init_fun = "init_sobol"`),
`'on'` and `'off'` become booleans, and `nvars` becomes `D`.
`search_method` lists the members of the search hedge, which is always
used, where MATLAB BADS names `@searchHedge` (`bads.m:239`).
- PyBADS: `pybads/bads/option_configs/`; `BADS.__init__`.
- MATLAB: `bads.m:149-161` (basic), `187-290` (advanced).
- Kind: deliberate change (interface).

**KD-B1-3. A user's option value is used as given; `None` stands for the default; an unknown name raises.**
The `.ini` values are expressions evaluated with `D` bound, and a user's
value is taken verbatim: a string such as `"200*D"` stays a string, where
MATLAB BADS evaluates `'200*nvars'` (for `max_fun_evals`, which must be a
positive integer of any size, it is refused; one beyond NumPy's 64-bit
integers stands for `inf`). A user's `None` leaves the option at its
default, as MATLAB's empty field does (`private/setupoptions.m:5-9`). The
options whose default is `True` or `False`, and `uncertainty_handling`,
take only booleans: MATLAB's `'on'`, `'off'`, `'yes'` and `'no'` are
refused with `ValueError`. A misspelt option name raises, where MATLAB
BADS ignores it.
- PyBADS: `pybads/bads/options.py` (`Options`,
  `Options.validate_boolean_options`); `BADS.__init__`,
  `BADS._init_optim_state_`.
- MATLAB: `private/setupoptions.m:5-9`, `21-50`.
- Settled by: W2-18, W2-19. Kind: deliberate change.

**KD-B1-4. Options that exist on one side only.**
- *MATLAB BADS only:* `OptimToolbox`, which chooses `fmincon` or
  `fminunc` against `minimizebnd` for the GP's hyperparameters (PyBADS's
  optimizer is gpyreg's, KD-B6-1); `Debug` and
  `TrueMinX`, which only print or plot (`bads.m:161`, `188`, `189`).
- *PyBADS only, and read:* `random_seed` (KD-B1-1); `stobads`,
  `opp_stobads` and `stobads_frame_size_scaling_power` (KD-S-1);
  `gp_mean_fun`, `"const"` (MATLAB's fixed `@meanConst`) or `"zero"`, any
  other name refused when `BADS` is created (W1-34); the options of the
  gpyreg-based GP layer, `gp_train_n_init`, `gp_train_n_init_final`,
  `gp_train_init_method`, `gp_tol_opt` (KD-B5-6), `hpd_frac`,
  `use_slice_sampler`, `gp_hyp_sampler` and `noise_shaping`;
  `init_mesh_size_integer` (default 0, MATLAB's fixed `MeshSizeInteger`).
  `hyp_run_weight` and `fun_evals_per_iter` (a running covariance of the
  hyperparameters that no run fills), `gp_quadratic_mean_bound` and
  `tol_sd` (the `"negquad"` mean, which is refused), and `hessian_update`
  and `hessian_method` (a branch that does nothing) are read only by code
  that no run reaches.
- *PyBADS only, and refused:* `f_vals`, read only by its check: one that
  holds a finite value is refused, and one without, such as an empty list,
  stands for `None` (W2-7).
- *On both sides, unported in PyBADS:* `fun_values` (MATLAB's `FunValues`,
  which imports earlier evaluations into the log and the GP,
  `private/setupvars.m:126-167`): a non-empty value is refused (W2-6); the
  port is an item of `dev/TODO.md`.
- Kind: removed feature (MATLAB's options); Python-only feature (the
  others).

**KD-B1-5. Options that are parsed and have no effect.**
- *Read by neither side:* `skip_poll`, `search_improve_frac`, `gp_cluster`.
- *Hard-coded to MATLAB's default choice:* `poll_method` (KD-B4-1),
  `poll_acq_fcn` (KD-B3-2), `gp_def_fcn` (KD-B6-1), `gp_method` (always
  the nearest neighbours), `chol_attempts` (KD-B6-6).
- *Read by MATLAB BADS only away from its defaults:* `n_basis`,
  `gp_samples` and `gp_svd_iters` (KD-B5-4), `rotate_gp`.
- *Without a MATLAB counterpart (leftovers of PyVBMC or of the port):*
  `gp_cov_fun`, which `_init_optim_state_` overrides
  (`optim_state["gp_cov_fun"] = 1`), `upper_gp_length_factor` (W1-33),
  `min_iter` and `min_fun_evals` (W2-35), and the options of PyVBMC's
  variational machinery and warping, such as `variational_sampler`,
  `warp_*`, `nsgp_*` and `acq_hedge_iter_window`.
- *Read only by a branch that does nothing or refuses:* `plot` (KD-B2-2),
  `restarts` (KD-B2-1), `search_optimize` (KD-B3-4), `acq_hedge`
  (KD-B3-3), `fitness_shaping` (KD-B5-5), `hessian_update` and
  `hessian_method` (KD-B1-4), a nonzero `warp_func` (KD-B6-4),
  `periodic_vars` (KD-B1-6), an `init_fun` other than `"init_sobol"`
  (KD-B7-2).
- This is not a list of every option that no code reads; `dev/TODO.md`
  counts them.
- Kind: removed feature; Python-only feature.

**KD-B1-6. Periodic variables are not supported.**
MATLAB BADS wraps a periodic variable into its range and uses a periodic
kernel. PyBADS refuses any `periodic_vars` that names a variable, before
its first transform of the variables; an empty one names none, as in
MATLAB BADS. `period_check` is a stub, and the periodic branches of
`udist`, `ucov` and the transform cannot be reached.
- PyBADS: `BADS.__init__`; `pybads/utils/period_check.py`.
- MATLAB: `bads.m:152`; `private/setupvars.m:49-57`, `107-116`;
  `utils/periodCheck.m`; `gpdef/gpdefBads.m:58-81`, `277-284`;
  `utils/udist.m`; `utils/ucov.m`; `gpml_fast/covPPERard_fast.m`.
- Settled by: the ruling on `periodic_vars` in wave 4. Kind: unported
  feature (open porting work, above).

**KD-B1-7. Fixed variables are refused.**
A variable whose bounds are all equal makes PyBADS raise `ValueError`;
MATLAB BADS fixes it and optimizes the others.
- PyBADS: `BADS._bounds_check_`.
- MATLAB: `private/boundscheck.m:39-40`; `bads.m:351-382`, `1480-1488`
  (`expandvars`); `private/fixedbads.m`.
- Kind: unported feature.

**KD-B1-8. The result is an `OptimizeResult` dict, not MATLAB's six outputs.**
PyBADS returns a SciPy-style dict (`x`, `x0`, `fval`, `fsd`, `yval_vec`,
`ysd_vec`, `func_count`, `iterations`, `mesh_size`, `message`,
`target_type`, `problem_type`, `total_time`, `overhead`, `random_seed`,
`algorithm`, `version`, `fun`, `non_box_cons`, `success`, `status`), and
the `BADS` object keeps the run's state. `status` is MATLAB's `exitflag`
(0 at `max_fun_evals` or `max_iter`, or when `output_fcn` stops the run; 1
on `tol_mesh`; 2 on the stall criterion), and `success` is `status > 0`.
There is no `rngstate` (KD-B1-1) and no `maxconstraint`. `iterations`
counts from 1 as MATLAB's does, but a run that ends in its initialization
reports 0, where MATLAB BADS reports 1. `fun` and `non_box_cons` are the
objects passed, where MATLAB stores `func2str(fun)`. `yval_vec` is `None`
for a deterministic run, with `noise_final_samples = 0`, and when the
budget leaves no evaluation for the final samples, where MATLAB BADS
returns the incumbent's observation (`bads.m:1136`).
- PyBADS: `pybads/bads/optimize_result.py` (`OptimizeResult`);
  `BADS.optimize`.
- MATLAB: `bads.m:1`, `423`, `1062-1083`, `1185-1194`;
  `private/bads_output.m`.
- Settled by: W0-4, W2-12, W2-13, W2-14, W2-32. Kind: deliberate change
  (interface).

**KD-B1-9. A start that `non_box_cons` rejects once put on the mesh is refused.**
PyBADS tests `non_box_cons` at the start a second time, after
`force_to_grid`, and raises `ValueError` if the point on the mesh violates
it; MATLAB BADS tests only the start as given and evaluates the point on
the mesh. A random start is put on the mesh before MATLAB's test, and
after PyBADS's redraws (KD-B1-11), so that PyBADS refuses a random start
that the mesh makes infeasible, without drawing again.
- PyBADS: `BADS._init_optim_state_`; `BADS.__init__`.
- MATLAB: `private/evalinitmesh.m:22-26`; `private/setupvars.m:84-87`,
  `101`.
- Kind: deliberate change.

**KD-B1-10. The transform's self-test tolerates an error relative to the bounds' magnitude.**
Both sides check that the inverse of the transform returns each finite
bound. MATLAB's absolute tolerance, 1e-6, refuses valid bounds from about
1e10 (an upper bound from about 1e9 on a log scale) through rounding alone;
PyBADS's tolerance is `1e-6 · max(1, |b|)`. A defect that PyBADS shared
and fixes.
- PyBADS: `VariableTransformer.__create_hypercube_trans__`, which
  `VariableTransformer.__init__` calls
  (`pybads/variable_transformer/variables_transformer.py`).
- MATLAB: `utils/transvars.m:30`, `169-178`.
- Settled by: W2-5. Kind: deliberate change.

**KD-B1-11. A random start that violates `non_box_cons` is drawn again.**
When `x0` is missing or not finite, MATLAB BADS draws one start in the
plausible box and stops with an error when it violates `non_box_cons`.
PyBADS draws again, up to 1000 draws in all, and then raises the same
error; a run whose first draw is feasible starts where MATLAB's would from
the same numbers. A defect that PyBADS shared and fixes.
- PyBADS: `BADS.__init__`.
- MATLAB: `private/setupvars.m:83-85`; `private/evalinitmesh.m:22-26`.
- Settled by: W2-11. Kind: deliberate change.

**KD-B1-12. A missing `x0` with only the hard bounds is accepted.**
With `x0` missing and the plausible bounds omitted, MATLAB BADS refuses
the problem (`bads.m:331-342`). PyBADS takes the hard bounds for the
plausible bounds and draws the start in the plausible box, as it draws any
missing start.
- PyBADS: `BADS.__init__`, `BADS._bounds_check_`.
- MATLAB: `bads.m:331-342`.
- Settled by: W2-9. Kind: deliberate change.

**KD-B1-13. `tol_fun` is checked when `BADS` is created.**
A `tol_fun` that is a real number must be positive and at most e^6, and a
boolean is refused; MATLAB BADS does not check it. Above e^6 the bounds of
the GP's log noise SD, `log(tol_fun) - 1` and 5, cross
(`gpdef/gpdefBads.m:161`), which stopped a run at its first fit, and 0
stopped it while the default of `hedge_beta`, `1e-3 / tol_fun`, was
evaluated.
- PyBADS: `BADS._check_tol_fun_`.
- MATLAB: `bads.m:193`; `private/setupoptions.m:23`.
- Settled by: #84. Kind: deliberate change.

### The main loop, termination and the final estimate (B2)

**KD-B2-1. Restarts are not implemented.**
With `restarts > 0`, MATLAB BADS resets the mesh and continues after
termination ("Multiple starts (deprecated)"); PyBADS stops. Both default
to 0.
- PyBADS: `BADS.optimize` (the branch on `restarts` does nothing).
- MATLAB: `bads.m:201`, `479`, `1121-1127`.
- Kind: unported feature.

**KD-B2-2. Plotting is not implemented.**
`plot` has no effect; MATLAB BADS draws a profile (`utils/landscapeplot.m`)
or a scatter plot (`private/scatterplot.m`).
- PyBADS: `BADS.optimize`; `advanced_bads_options.ini`.
- MATLAB: `bads.m:187`, `988-1015`, `1054-1057`.
- Kind: unported feature.

**KD-B2-3. Messages go through Python logging, to the `BADS` logger.**
`display` sets the level of a logger instead of choosing which `fprintf`
calls run. The level follows MATLAB's reading of the first three letters,
lower case: `"off"` and `"none"` show the warnings only, `"notify"` and
any other value also the opening message, `"final"` also the final
message, `"iter"` and `"all"` also the iteration lines, and `"full"`,
which only PyBADS has, the debug messages too. This settles the mechanism;
which message goes at which level is open (`dev/TODO.md`, the minor items
of B1 and B2).
- PyBADS: `BADS.__init__` and the display methods of `BADS`;
  `pybads/bads/gaussian_process_train.py`.
- MATLAB: `bads.m:311-328` and `fprintf` throughout.
- Settled by: W2-15. Kind: deliberate change.

**KD-B2-4. When the re-estimate of the current iterate fails, it keeps its estimate.**
In a noisy run, each iterate is re-estimated from a copy of the working GP
rebuilt around it. When that rebuild fails, a past iterate gets NaN, as in
MATLAB BADS, and the move and the final choice skip NaN; the current
iterate keeps its recorded estimate, so that the incumbent is never NaN,
where MATLAB's becomes NaN. The NaN of past iterates stays in
`iteration_history`.
- PyBADS: `BADS._re_evaluate_history_`.
- MATLAB: `bads.m:1097-1104`; `utils/gppred.m:22-56`.
- Settled by: W1-35, W2-30, W2-39. Kind: deliberate change.

**KD-B2-5. The output function: a stop has a message of its own and is final, and the `"init"` call comes after a noisy run's setup.**
A stop by `output_fcn` ends the run with the message "terminated by
options['output_fcn']", unless a termination criterion fires in the same
iteration, whose message then stands, as on both sides; MATLAB BADS sets a
message only when a criterion fires, so that a stop by its output function
keeps the message of the initialization. A false return at `"init"`
cannot reopen a run that ended there. The `"init"` call comes after the
options of a noisy run are changed and the first GP is fitted, and
MATLAB's before.
- PyBADS: `BADS.optimize`, `BADS._init_optimization_`.
- MATLAB: `bads.m:424` (the initialization's message), `426-428` (the
  `'init'` call), `431-445`, `447-457`, `465-469` (a noisy run's setup, the
  GP defined), `1037-1039` (the `'iter'` call), `1062-1085`.
- Settled by: W2-33. Kind: deliberate change.

**KD-B2-6. The budget counts the noise test, and the initial design keeps within it.**
MATLAB BADS caps the design at `MaxFunEvals - 1` without counting the
noise test at `x0`, so that a budget below the design takes one evaluation
more than `MaxFunEvals`. PyBADS rounds the design up to a power of two
(KD-B7-1) and then keeps its first points within the evaluations left, the
noise test counted, so that only `max_fun_evals=1` with the noise test
exceeds the budget, by the noise test, as in MATLAB BADS; a noisy run's
reserve for its final samples is never negative. At the default budgets
the cap does not bind. The counting of the noise test is a defect that
PyBADS shared and fixes.
- PyBADS: `BADS._init_mesh_`, `BADS._init_optimization_`.
- MATLAB: `private/evalinitmesh.m:37-42`, `98-104`.
- Settled by: W2-27; W4-3 (the design's size). Kind: deliberate change.

**KD-B2-7. A move to an earlier iterate after the re-estimate moves the incumbent's location with its value.**
When the re-estimate of a noisy run finds an earlier iterate better by more
than `tol_fun`, MATLAB BADS gives the incumbent that iterate's value and
leaves `ubest` at the old point, so that the next search's target is
predicted there and a poll that no successful search precedes runs around
the old point while it is judged by the other's value. PyBADS moves the
incumbent, its location with its value. As in MATLAB BADS, the target's
hyperparameters move with it and the working GP stays. A defect that
PyBADS shared and fixes; it moves the noisy runs.
- PyBADS: `BADS.optimize` (the move after `_re_evaluate_history_`,
  through `_update_incumbent_`).
- MATLAB: `bads.m:1111-1118`, `769`.
- Settled by: W2-25, option (b). Kind: deliberate change.

**KD-B2-8. A noisy run that ends within its first iteration takes the final samples it reserved, at the incumbent.**
MATLAB BADS takes the final samples only after its first iteration
(`bads.m:1138`), so that a noisy run that ends within it, on `MaxIter` 1
or on a budget that the initial design nearly uses up, leaves the reserved
evaluations unused and reports the incumbent's single observation. PyBADS
takes them at the incumbent, the run's only iterate, and reports their
estimate. A run that `output_fcn` stops at `"init"` takes none, on both
sides, and its `fsd` is not an estimate (its description says what it
is). A defect that PyBADS shared and fixes.
- PyBADS: `BADS.optimize` (the final estimate).
- MATLAB: `bads.m:448-452`, `1138`.
- Settled by: W4-14, option (a); W4-30. Kind: deliberate change.

**KD-B2-9. `optim_state` keeps the incumbent's values after the re-estimate and the final estimate.**
The re-estimate at the end of an iteration of a noisy run, and the final
estimate before the `"done"` call of `output_fcn`, set `optim_state`'s `u`,
`yval`, `fval` and `fsd` with the incumbent's, so that the copy that
`output_fcn` receives holds them; MATLAB's `optimState` keeps the values of
an earlier iteration there. No result reads these entries, and
`iteration_history` holds the final estimate at the chosen iterate, as
MATLAB's `iterList` does.
- PyBADS: `BADS.optimize`.
- MATLAB: `bads.m:1111-1118`, `1150-1165`.
- Settled by: W3-33, W4-26. Kind: deliberate change.

### The search (B3)

**KD-B3-1. The search hedge chooses between ES-wcm and ES-ell; the other search methods are not ported.**
Only MATLAB's default set of searches exists: `ESSearchWM`, `searchES`'s
method 1 (`'ES-wcm'`), and `ESSearchELL`, its method 2 (`'ES-ell'`). The
other methods of `searchES` (`ES-eye`, `ES-cov`, `ES-cma+`) and the other
search functions are absent. A `search_method` that is not a non-empty
list of pairs named `"ES-wcm"` or `"ES-ell"` is refused when `BADS` is
created; MATLAB BADS checks nothing.
- PyBADS: `pybads/search/search_hedge.py` (`ESSearchHedge`);
  `pybads/search/es_search.py`; `BADS._init_optim_state_`.
- MATLAB: `bads.m:239`; `search/searchES.m:3-12`, `39-101`;
  `search/searchCMA.m`, `searchCombine.m`, `searchCrossover.m`,
  `searchGauss.m`, `searchGrid.m`, `searchMax.m`, `searchMaxAcq.m`,
  `searchNewton.m`, `searchOptim.m`, `searchWCM.m`, `search/private/`.
- Settled by: W3-13. Kind: unported feature.

**KD-B3-2. Only the lower confidence bound (LCB) exists as acquisition function.**
`PollAcqFcn` and `SearchAcqFcn` can name other acquisition functions in
MATLAB BADS; in PyBADS the poll always uses the LCB, with its default
schedule, and a `search_acq_fcn` that is not a pair `("acq_LCB",
sqrt_beta)` is refused when `BADS` is created. Both default to the LCB.
- PyBADS: `pybads/acquisition_functions/acq_fcn_lcb.py`; `ESSearch`;
  `BADS._init_optim_state_`, `BADS._search_step_`, `BADS._poll_step_`.
- MATLAB: `bads.m:269-270`, `577-578`, `852`; `search/searchES.m:147`,
  `156-165`; `acq/acqNegEI.m`, `acqNegEQI.m`, `acqNegPI.m`,
  `acqNegSqEI.m`, `acqRnd.m`, `acq/private/`.
- Kind: unported feature.

**KD-B3-3. The acquisition hedge (`AcqHedge`) is not implemented, and `acq_hedge=True` is refused.**
MATLAB BADS labels it unsupported and its search falls back from it;
PyBADS refuses the option when `BADS` is created. Both default to off. The
search hedge's reward and update are ported (`ESSearchHedge.update_hedge`,
`acqPortfolio.m`'s `'upd'` branch), with the standard normal density of
`acqPortfolio.m:64`.
- PyBADS: `BADS._init_optim_state_`; `ESSearchHedge.update_hedge`.
- MATLAB: `bads.m:271`, `569-573`, `684-686`, `716-719`, `845-848`;
  `acq/acqPortfolio.m`; `acq/acqHedge.m`; `search/searchES.m:139-141`.
- Settled by: the ruling after wave 3's doublecheck; W3-6. Kind:
  unported feature.

**KD-B3-4. The local optimization of the acquisition function (`SearchOptimize`) is not implemented.**
`search_optimize=True` does nothing; MATLAB BADS runs `fmincon` on the
acquisition function. Both default to off, and MATLAB's comment says it
generally does not improve results.
- PyBADS: `BADS._search_step_`.
- MATLAB: `bads.m:248`, `594-616`.
- Kind: unported feature.

**KD-B3-5. An empty search set is a failed search on every path.**
A search set is empty when every candidate violates `non_box_cons` or was
already evaluated. PyBADS counts a failed search and decays the hedge's
gains, as MATLAB BADS does after the run's first search at the default
`ImprovementQuantile` (≤ 0.5) or without noise; there MATLAB's hedge
update scores the previous search's point, with a reward of 0, and
PyBADS's scores none, with the same gains. At `ImprovementQuantile` > 0.5
in a noisy run, MATLAB BADS counts an incremental search and moves the
incumbent to the previous search's point with an SD of 0. When the run's
first search set is empty, that point is undefined and MATLAB BADS stops
with an error, at every quantile (by reading: `bads.m:693`, `704`,
`722`).
- PyBADS: `BADS._search_step_`; `ESSearchHedge.update_hedge`.
- MATLAB: `bads.m:667-725`, `1257-1282`; `acq/acqPortfolio.m:56-69`.
- Settled by: W0-15, W3-11. Kind: deliberate change.

**KD-B3-6. A generation of the ES search that adds no candidate leaves its scale unchanged.**
From `n_search_iter` = 3 (the default is 2), a generation that
`non_box_cons` or the removal of evaluated points empties makes MATLAB's
scale 0/0, NaN, and `uCheck`'s projection, whose `min` and `max` ignore
NaN, then sends every later candidate of the search to the corner
`UBsearch` (by reading). PyBADS updates the scale only when the generation
added a candidate.
- PyBADS: `ESSearch.__call__`.
- MATLAB: `search/searchES.m:170-193`.
- Settled by: W3-8, W3-9. Kind: deliberate change.

**KD-B3-7. The search's `sqrt_beta` is `None`, a callable or a positive finite number.**
MATLAB's `acqLCB` takes an empty value (the schedule of Srinivas et al.), a
function handle or a function's name, or any numeric scalar, zero, negative
and non-finite values included, and uses a function's value unchecked.
PyBADS takes `None`, a callable or a positive finite real number (a
one-element array included), and refuses anything else, a name included,
when `BADS` is created; the search raises `ValueError` when a callable
returns a value that is not a positive finite real number.
- PyBADS: `pybads/acquisition_functions/acq_fcn_lcb.py`
  (`check_sqrt_beta`); `BADS._init_optim_state_`.
- MATLAB: `acq/acqLCB.m:10-21`.
- Settled by: W3-10, W4-19. Kind: deliberate change.

**KD-B3-8. The search hedge's parameters are checked, and `hedge_gamma = 0` works.**
At `hedge_gamma = 0` the searches that the hedge did not choose are scored
by the GP at the search point, taken as a row, MATLAB's evident intent;
MATLAB BADS stops with an error at its first search there, on an undefined
variable. MATLAB BADS checks none of the hedge's parameters: a
`HedgeGamma` above 1/n (n the number of searches) favors the search of
lower gain, and above 1/(n - 1) gives some searches a negative
probability; a negative `HedgeBeta` inverts the hedge, and a non-finite
one makes its probabilities NaN; a `HedgeDecay` above 1 makes the gains
grow until they overflow, and a negative one makes them alternate in sign.
PyBADS refuses, when `BADS` is created, a `hedge_gamma` outside
`[0, 1/n]`, a `hedge_beta` that is not a finite number at least 0, and a
`hedge_decay` outside `[0, 1]`, and any of the three that is not a real
number (a Python or NumPy integer or float, not a boolean). Defects that
PyBADS shared and fixes.
- PyBADS: `ESSearchHedge.update_hedge`; `BADS._init_optim_state_`.
- MATLAB: `acq/acqPortfolio.m:40`, `47`, `69`; `search/searchHedge.m:45-46`.
- Settled by: W3-7, W4-18, W4-29. Kind: deliberate change.

**KD-B3-9. The ES search's number of parents is floored.**
PyBADS takes `mu = n_search / n_search_iter` rounded down, as the
description of `n_search_iter` says, so that an `n_search_iter` that does
not divide `n_search` keeps its rounded-down generations; MATLAB BADS
does not round it (`private/setupvars.m:186`), and its `randn` would not
take the fraction (by reading).
- PyBADS: `ESSearchHedge.__init__` (`pybads/search/search_hedge.py`);
  `advanced_bads_options.ini` (`n_search_iter`).
- MATLAB: `private/setupvars.m:186`; `search/searchES.m:117`.
- Settled by: the rulings of wave 4's doublecheck; `dev/TODO.md`, the
  minor items of B7 and O, asks whether to close it as ruled. Kind:
  deliberate change.

### The poll, the mesh, the incumbent and the target (B4)

**KD-B4-1. The poll is MADS 2N (`poll_mads_2n`, MATLAB's `pollMADS2N`); the other poll methods are not ported.**
`poll_method` is ignored (KD-B1-5). On both sides MADS 2N's basis is the
signed coordinate directions at every default state, since
`pollMADS2N.m:7` bounds its lower-triangular entries by the ratio of the
search mesh to the poll mesh, which is below 1; LTMADS's tilted
directions, tried in the review, did worse on PyBADS's benchmark and were
reverted. PyBADS does not permute the basis's columns as
`pollMADS2N.m:17` does, which only reorders the directions.
- PyBADS: `pybads/poll/poll_mads_2n.py`; `BADS._poll_step_`.
- MATLAB: `bads.m:206`, `791-798`; `poll/pollMADS2N.m`; `poll/pollGPS2N.m`;
  `poll/private/pollBADS2N.m`, `pollBMADS2N.m`.
- Settled by: W3-24. Kind: unported feature.

**KD-B4-2. The target is predicted under the best iteration's hyperparameters, from a posterior computed under them.**
PyBADS predicts the target at the incumbent from the GP itself when the
best iteration's hyperparameters, `hyp_best`, are the GP's own, and
otherwise from a copy of the GP whose posterior is recomputed under
`hyp_best`; when that posterior cannot be computed (`LinAlgError`), it
predicts from the GP as it stands. MATLAB's `UpdateTarget` keeps the
current posterior and evaluates the kernel and the mean under `hyp`
(`bads.m:1301`, `utils/gppred.m:39-47`, `utils/mygp.m:122-123`,
`146-187`), a hybrid that is no GP prediction under one set of
hyperparameters, and cannot fail there. The two agree when `hyp_best` is
the current set. The search computes the target only for an acquisition
function that reads it, of which none is ported
(`_SEARCH_ACQ_FCNS_READING_TARGET`, empty), where MATLAB BADS computes it
at every search (`bads.m:539`).
- PyBADS: `BADS._get_target_from_gp_`, `BADS._update_target_`,
  `_SEARCH_ACQ_FCNS_READING_TARGET` (`pybads/bads/bads.py`).
- MATLAB: `bads.m:539`, `841`, `1296-1312`.
- Settled by: W3-21 (a); #84 (the reuse of the GP's posterior, the
  search's target). Kind: deliberate change.

KD-B4-3 was LTMADS's directions in the poll, which were reverted (KD-B4-1).

**KD-B4-4. A target whose prediction is not finite is computed from the incumbent's SD.**
Both sides replace a prediction that is not finite by the incumbent's
`fval` and `fsd`, but MATLAB's target keeps the prediction's variance, so
that a NaN variance gives a NaN target and an infinite one `-Inf`; PyBADS
computes the target from the incumbent's SD. A defect that PyBADS shared
and fixes; no run has shown a non-finite prediction.
- PyBADS: `BADS._get_target_from_gp_`.
- MATLAB: `bads.m:1309-1311`, `1321`.
- Settled by: W3-23. Kind: deliberate change.

**KD-B4-5. When every acquisition value is NaN, the search and the poll choose a candidate at random.**
With some values NaN, both sides take the smallest of the others. With
every value NaN, MATLAB's `min` returns the first candidate, so that its
fallback "randomly choose index" fires only when the search's acquisition
raises; PyBADS takes a random candidate, drawn from `bads.rng`, with a
warning. No such case has been observed.
- PyBADS: `BADS._search_step_`, `BADS._poll_step_`.
- MATLAB: `bads.m:581-589`, `853-857`.
- Settled by: W3-27. Kind: deliberate change.

**KD-B4-6. `improvement_quantile`, `accelerate_mesh_steps`, `n_search_iter` and `n_search` are checked when `BADS` is created.**
MATLAB BADS refuses an `ImprovementQuantile` outside (0, 1) when it first
evaluates an improvement, and lets NaN through, to NaN improvements; PyBADS
refuses both when `BADS` is created, and any value that is not a real
number. MATLAB BADS does not check `AccelerateMeshSteps`: 0, a negative
value or a non-integer stops its run at the first accelerated mesh
reduction, and `Inf` runs without the reduction. PyBADS refuses every value
that is not a positive integer, `inf` included, converts a whole-number
float, and names `accelerate_mesh=False`, the switch that turns the
reduction off, in its message. `n_search_iter` and `n_search`, unchecked in
MATLAB BADS, must be positive integers, with `n_search_iter` at most
`n_search`. The integers may be of any size. The stop on an
`AccelerateMeshSteps` below 1 is a defect that PyBADS shared and fixes.
- PyBADS: `BADS._init_optim_state_`.
- MATLAB: `bads.m:976-979`, `1269-1271`; `private/setupvars.m:179-182`,
  `185-188`; `private/setupoptions.m:26`; `search/searchES.m:125`.
- Settled by: W3-31, W3-39, W4-25, and the rulings after the doublechecks
  of waves 3 and 4. Kind: deliberate change.

### The GP's training set and refits (B5)

**KD-B5-1. Adding a point recomputes every posterior in full, and a failed addition leaves the point out of the GP until the next rebuild.**
MATLAB BADS adds a point by a rank-1 update of the posterior
(`utils/update_posterior.m`), except under `SpecifyTargetNoise`, and keeps
the point beside an empty posterior when the update fails. PyBADS passes
the point through gpyreg's `update` with the hyperparameters given, which
recomputes the posterior in full; on a failure gpyreg restores the GP, the
point stays out of it, in the function log, until the next rebuild, and
the GP is marked for that rebuild. A noisy poll's estimate at such a point
is NaN, as MATLAB's prediction is. The rank-1 update was measured and not
adopted (`dev/results/2026-09-28-where-pybads-spends-its-time.md`;
`dev/TODO.md`), and `_get_target_from_gp_` relies on posteriors computed
in full (KD-B4-2).
- PyBADS: `add_and_update_gp` (`pybads/bads/gaussian_process_train.py`);
  `BADS._search_step_`, `BADS._poll_step_`.
- MATLAB: `private/gpupdate.m:39-83`, `340-354`; `bads.m:633-641`,
  `908-924`.
- Settled by: `dev/plans/gp-update-guards.md`. Kind: deliberate change.

**KD-B5-2. A failed rebuild restores the GP as it was, marks it, and forces a refit at the next rebuild.**
When the rebuild of the local GP fails, MATLAB BADS keeps the new data and
the failed rebuild's hyperparameters and `pollscale` beside an empty
posterior, rebuilds while the posterior is empty, and refits only when
`gppredcheck` finds its NaN predictions unreliable and `MinRefitTime` has
passed. PyBADS puts back the GP it was given, marks it
(`temporary_data["needs_rebuild"]` and `["needs_refit"]`, standing in for
the empty posterior), and refits at the next rebuild, whatever
`min_refit_time` says. The retry with the previous hyperparameters on the
new training set, which MATLAB BADS lacks, is made only after a refit, and
the GP's geometry then comes from the hyperparameters it keeps. After a
failed rebuild, the search ranks its candidates by the restored GP, a
consistent GP with finite predictions, where MATLAB's empty posterior
scores every candidate 0 and keeps `uCheck`'s first, an arbitrary far
point; the poll treats such a GP as unreliable on both sides.
- PyBADS: `local_gp_fitting` (`pybads/bads/gaussian_process_train.py`);
  `BADS._search_step_`, `BADS._poll_step_`, `BADS._record_gp_refit_`.
- MATLAB: `private/gpupdate.m:340-354`; `bads.m:523-536`, `826-839`,
  `1223-1254`.
- Settled by: `dev/plans/gp-update-guards.md`; W1-10, W1-11, W3-12. Kind:
  deliberate change.

**KD-B5-3. Only `LinAlgError` is caught at the guarded GP calls.**
MATLAB's `try` around the GP's update catches any error; PyBADS catches
`LinAlgError`, so that another error from gpyreg stops the run. MATLAB's
other `try` blocks (around the acquisition, `gppredcheck`, `acqLCB`,
`searchES` and `gppred`) have no counterpart decided.
- PyBADS: `local_gp_fitting`, `add_and_update_gp`, `_robust_gp_fit_`,
  `init_and_train_gp`; `BADS._get_target_from_gp_`.
- MATLAB: `private/gpupdate.m:52-64`, `340-354`.
- Settled by: `dev/plans/gp-update-guards.md`. Kind: deliberate change.

**KD-B5-4. The GP's hyperparameters are optimized, never sampled.**
With `gpSamples` above 1, MATLAB BADS fits several samples of the
hyperparameters by SVGD; PyBADS ignores `gp_samples` and `gp_svd_iters`
and keeps one set. At the default, 0, both optimize one set.
- PyBADS: `pybads/bads/gaussian_process_train.py`.
- MATLAB: `bads.m:254`; `private/gpupdate.m:281`, `411-414`;
  `utils/gpHyperSVGD.m`.
- Kind: unported feature.

**KD-B5-5. Fitness shaping is not implemented.**
`fitness_shaping=True` does nothing; MATLAB BADS lists it under "GP warping
parameters (unsupported)". Both default to off.
- PyBADS: `local_gp_fitting`.
- MATLAB: `bads.m:279-280`; `private/gpupdate.m:43-47`, `252-256`;
  `utils/fitnessTransform.m`.
- Kind: unported feature.

**KD-B5-6. A refit starts from gpyreg's design of prior draws.**
PyBADS evaluates `init_N` draws from the priors beside the given starting
points and optimizes the best one with gpyreg's L-BFGS-B (the best two on
a second fit, the second replaced by gpyreg's low-noise design point).
MATLAB BADS runs one local optimization from the previous hyperparameters
and one from the second fit's point, with `TolFun` 0.1, `TolX` 1e-4 and
150 evaluations. On the same data (14 refits), the design reached the same
optimum in 8, a better one in 3 and a worse one in 3 (by at most 0.56 in
the negative log posterior), and it avoided fit failures that MATLAB's
starts met.
- PyBADS: `_get_gp_training_options`, `local_gp_fitting`,
  `_robust_gp_fit_`; gpyreg's `GP.fit`.
- MATLAB: `private/gpupdate.m:371-408`; `utils/gpHyperOptimize.m:47-75`.
- Settled by: W1-15. Kind: deliberate change.

**KD-B5-7. The normality test of the GP's calibration is SciPy's Shapiro-Wilk.**
MATLAB's `swtest` switches to Shapiro-Francia when the kurtosis of the
z-scores exceeds 3; PyBADS always uses `scipy.stats.shapiro`. At
`normalpha_level = 1e-6` the two decide differently on heavy-tailed
z-scores, which default runs rarely give.
- PyBADS: `BADS._is_gp_refit_time_`.
- MATLAB: `utils/swtest.m:130-160`, `272`; `utils/gppredcheck.m:30`.
- Settled by: W1-6. Kind: substituted library.

**KD-B5-8. Under `specify_target_noise`, the high-noise check of a refit takes the base noise 1.**
MATLAB's check reads a user's `NoiseSize`, although its own warning says
that the option is ignored with `SpecifyTargetNoise`; PyBADS follows the
warning, so that `noise_size=0`, which the warning proposes, does not make
every refit a second fit.
- PyBADS: `BADS._init_optim_state_`; `local_gp_fitting`.
- MATLAB: `private/setupoptions.m:100-101`; `private/gpupdate.m:379-381`.
- Settled by: the PI's ruling in #71. Kind: deliberate change.

**KD-B5-9. With `poll_training` off, the poll neither records a refit that it does not make nor clears the flag of an unreliable GP.**
MATLAB's `IsRefitTime` records the refit and clears the flag before the
poll drops the refit when `PollTraining` is off, so that the next refit of
the search waits for `MinRefitTime` counted from a refit that did not
happen. PyBADS does neither. Off by default. A defect that PyBADS shared
and fixes.
- PyBADS: `BADS._poll_step_`, `BADS._is_gp_refit_time_`.
- MATLAB: `bads.m:822-823`, `1242-1252`.
- Settled by: W1-8, W3-30. Kind: deliberate change.

**KD-B5-10. At D = 1 the training set's distances are in units of the fitted length scale.**
MATLAB BADS takes the ARD length scales for the distances that choose the
training set only when `ncovlen > 1`, its test for a kernel with one
length scale per dimension, which fails at D = 1, where the distances are
then in units of 1. PyBADS takes the fitted length scales at every D. A
defect that PyBADS shared and fixes.
- PyBADS: `local_gp_fitting`.
- MATLAB: `private/gpupdate.m:285-292`; `gpdef/gpdefBads.m:51`.
- Settled by: W1-17. Kind: deliberate change.

### The GP model (B6)

**KD-B6-1. The GP is gpyreg's, with a fixed rational-quadratic ARD kernel.**
Every object of the GP layer (the hyperparameter vector, the priors, the
bounds, the likelihood, the inference, the optimizer, the prediction) is
gpyreg's where MATLAB BADS uses GPML 3.6 with its own fast replacements.
The kernel is `RationalQuadraticARD`, MATLAB's default (`'rq'`, ARD), and
cannot be changed (`gp_cov_fun` and `gp_def_fcn` have no effect). gpyreg's
Gaussian priors take a mean and an SD, where GPML's `priorGauss` takes a
variance. The starting points (KD-B5-6), the fit at initialization
(KD-B6-5) and the handling of a failed factorization (KD-B6-6) have
entries of their own.
- PyBADS: `pybads/bads/gaussian_process_train.py` (`init_and_train_gp`,
  `_gp_hyp`, `_cov_identifier_to_covariance_function`); gpyreg's `GP`,
  `RationalQuadraticARD`, `ConstantMean`, `GaussianNoise`.
- MATLAB: `bads.m:260`; `gpdef/gpdefBads.m`; `gpml_fast/covRQard_fast.m`;
  `private/gpupdate.m:359-419`; `utils/gpHyperOptimize.m`;
  `utils/gppred.m`; `utils/mygp.m`.
- Kind: substituted library.

**KD-B6-2. A GP whose training data have no spread keeps its previous priors.**
When the range of the training targets is 0, MATLAB BADS gives the mean's
prior a zero variance (`yrange.^2/4`) and centres the output scale's prior
at `log(std(y)) = -Inf`; PyBADS keeps the previous width of the mean's
prior and the previous centre of the output scale's, and at the initial
fit gives the mean's prior the SD 1. When the pairwise distances of the
training set have no spread (two distinct points), MATLAB's empirical
prior of the length scales has a zero width, which gpyreg refuses; PyBADS
keeps the previous prior. Otherwise the re-centred priors follow MATLAB
BADS. What MATLAB's fit does with these zero-width priors is not known
without MATLAB.
- PyBADS: `local_gp_fitting`, `_gp_hyp`.
- MATLAB: `gpdef/gpdefBads.m:219-222`, `240-251`, `293-295`.
- Settled by: W1-26, W3-40. Kind: deliberate change.

**KD-B6-3. With `gp_fixed_mean`, the GP's mean is not fixed.**
MATLAB BADS fixes the mean at `ymean` under a delta prior; PyBADS
re-centres a Gaussian prior and keeps its width. Both default to off.
- PyBADS: `local_gp_fitting`.
- MATLAB: `gpdef/gpdefBads.m:168-172`, `220-231`.
- Kind: unported feature.

**KD-B6-4. Warped likelihoods and output warping are unsupported on both sides.**
MATLAB BADS refuses them with a message when it defines the GP; in PyBADS
a nonzero `warp_func` fails at the first rebuild, without one.
- PyBADS: `local_gp_fitting`, `_gp_hyp`; `advanced_bads_options.ini`
  (`warp_*`).
- MATLAB: `bads.m:279-281`; `gpdef/gpdefBads.m:116-118`, `210-215`,
  `287-291`; `warp/`.
- Kind: removed feature.

**KD-B6-5. PyBADS fits a GP on the initial design; MATLAB BADS only defines it.**
PyBADS fits the hyperparameters on the initial design under the priors of
the definition; MATLAB BADS keeps the definition's values until its first
rebuild. Both refit at the first rebuild, so the initial fit reaches a run
as one start of that refit and as the hyperparameters of the first
target's prediction.
- PyBADS: `init_and_train_gp`; `BADS._init_optimization_`.
- MATLAB: `bads.m:465-469`; `gpdef/gpdefBads.m:164-165`.
- Settled by: W1-27. Kind: deliberate change.

**KD-B6-6. A failed Cholesky factorization multiplies the GP's noise.**
gpyreg multiplies the noise by ten per failed attempt, up to ten attempts,
and keeps the multiplier in the posterior (`sn2_mult`), which
`get_hyperparameters` does not show; MATLAB BADS (`CholAttempts = 0`)
treats the failure as an error, restarts the fit with the noise's start
nudged, and empties the posterior. It is reached at default options on
some targets. gpyreg's switch `raise_on_cholesky_failure`, off by default,
gives MATLAB's behavior; measured, it stays off in PyBADS, and
`chol_attempts` is unread.
- PyBADS: gpyreg's `GP` (its training Cholesky factorization, and
  `predict`).
- MATLAB: `bads.m:272`; `gpml_fast/infExact_fastrobust.m:36`, `77-80`;
  `utils/gpHyperOptimize.m:73-176`; `private/gpupdate.m:340-354`.
- Settled by: W1-25; `dev/TODO.md` holds the revisit. Kind: substituted
  library.

**KD-B6-7. `gp_cov_prior="ard"` is not ported, and is refused.**
MATLAB's `'ard'` sets an empirical prior of the length scales per
dimension; PyBADS refuses any value but `"iso"` when `BADS` is created.
- PyBADS: `BADS._init_optim_state_`; `local_gp_fitting`.
- MATLAB: `gpdef/gpdefBads.m:254-274`.
- Settled by: W1-28; the port is an item of `dev/TODO.md`. Kind:
  unported feature.

**KD-B6-8. A fixed noise (`fit_lik=False`) is refused on both sides.**
PyBADS refuses it when `BADS` is created, MATLAB BADS when it defines the
GP, with the same message.
- PyBADS: `BADS._init_optim_state_`.
- MATLAB: `bads.m:466`; `gpdef/gpdefBads.m:139-140`.
- Settled by: W1-32. Kind: removed feature.

### The function logger, the initial design and the utilities (B7)

**KD-B7-1. The initial design is a scrambled Sobol set with a power-of-two number of points, seeded from the run's generator.**
MATLAB BADS draws `Ninit` points of the unscrambled Sobol sequence, from a
skip index that it derives from the start alone, with no random draw
(`init/initSobol.m:9-15`). PyBADS draws `2**ceil(log2(fun_eval_start))`
points, twice as many when that number equals D, from
`scipy.stats.qmc.Sobol`, whose scrambling is seeded by one draw of
`bads.rng`, so that `random_seed` decides the design whatever `x0` is
given. What MATLAB's seed is for a start inside the plausible box needs
MATLAB
(`dev/experiments/port_review_20260925/matlab_side_defects.md`).
- PyBADS: `pybads/init_functions/init_sobol.py` (`init_sobol`);
  `BADS._init_mesh_`.
- MATLAB: `private/evalinitmesh.m:98-104`; `init/initSobol.m:9-16`;
  `init/private/i4_sobol*.m`, `i4_bit_*.m`.
- Settled by: W4-1, option (a); W4-3 (the doubling kept). Kind:
  substituted library, with deliberate changes of the seed and the size.

**KD-B7-2. Only the Sobol initial design exists.**
Latin hypercube and uniform designs, and `initSobol`'s fallback to a Latin
hypercube, are not ported; any other `init_fun` is refused.
- PyBADS: `BADS._init_mesh_`.
- MATLAB: `bads.m:199`; `init/initLHS.m`, `init/initRand.m`,
  `init/private/lhs.m`; `init/initSobol.m:18-21`.
- Kind: unported feature.

**KD-B7-3. With target noise, a repeated point is merged into its own row, and the merged value is returned.**
At uncertainty level 2, PyBADS's function logger keeps one row per point
and merges a repeated evaluation into it by precision weighting; MATLAB's
`funlogger` adds a row per evaluation and returns the observation. No run
of `BADS` reaches the merge, since `contraints_check` removes a point
already evaluated before it is evaluated and the noise test and the final
samples record nothing; only a direct use of `FunctionLogger` merges.
Returning the observation, as MATLAB does, was tested and not adopted. At
levels 0 and 1 a repeat is a new row on both sides.
- PyBADS: `FunctionLogger` (`pybads/function_logger/function_logger.py`).
- MATLAB: `private/funlogger.m:117-129`.
- Settled by: W4-5;
  `dev/experiments/population_ellipsoid_hetero_linux_20260925/`. Kind:
  deliberate change.

**KD-B7-4. The log of evaluations grows when it is full.**
MATLAB's log is a ring of `CacheSize` rows (1e4 by default), which
overwrites its oldest rows past `CacheSize` evaluations and never writes
its last row, which it then reads; PyBADS's arrays grow by half when full,
and `cache_size` is their initial size. The two agree up to 9999 logged
evaluations.
- PyBADS: `FunctionLogger`; `advanced_bads_options.ini` (`cache_size`).
- MATLAB: `private/funlogger.m:120-121`.
- Settled by: W4-12. Kind: deliberate change.

**KD-B7-5. The noise test's second value goes through the logger's checks.**
The noise test's evaluation goes through the function logger, recording no
row, so that a NaN or infinite value raises `ValueError`, as at every other
evaluation; MATLAB BADS calls the target directly and reads NaN as
deterministic and infinity as noisy. The test counts in PyBADS's
`max_fun_evals` (KD-B2-6), and the schedule of the GP's fits leaves it out
of its budget.
- PyBADS: `BADS._init_mesh_`; `FunctionLogger`.
- MATLAB: `private/evalinitmesh.m:41-47`.
- Settled by: W4-6, W4-13. Kind: deliberate change.

### Sto-BADS (S)

**KD-S-1. Sto-BADS is PyBADS's own.**
With `stobads` on, a success rule based on uncertainty intervals, after
Sto-MADS (Audet, Dzahini, Kokkolaras and Le Digabel, 2021), replaces the
improvement tests of the search and the poll; `opp_stobads` and
`stobads_frame_size_scaling_power` tune it, and the keyword-only argument
`gamma_uncertain_interval` of `BADS` sets its interval. It is off by
default and switched off for a deterministic target. The design of its
rule is open (`dev/TODO.md`, "The uncertainty interval of Sto-BADS").
- PyBADS: `BADS._sto_success_improvement_`, and the `stobads` branches of
  `BADS._search_step_` and `BADS._poll_step_`.
- MATLAB: no counterpart.
- Kind: Python-only feature.

### Tests (T)

**KD-T-1. The optimization tests use the problems of MATLAB's `runtest.m`, with tolerances from seed sweeps.**
Each test's tolerance is ten times the largest error over seeds 0-99,
rounded up, or `runtest.m`'s tolerance when that is lower;
`test_sphere_opt` takes `runtest.m`'s constraint and start point.
- PyBADS: `pybads/testing/bads/test_bads_optimization.py`.
- MATLAB: `private/runtest.m:11`.
- Settled by: `dev/results/2026-09-23-codebase-survey.md`, "The seed sweep
  behind the tolerances". Kind: deliberate change.
