# PyBADS and MATLAB BADS: deliberate differences and open porting work

PyBADS is the Python port of MATLAB BADS (`acerbilab/bads`). This file
catalogues where PyBADS differs from MATLAB BADS v1.1.3 (`74919c0`) on
purpose, and lists the porting work that is still open. A difference that
is not listed here is not known to be deliberate: it is a defect until
shown otherwise.

The catalogue comes from the port correctness review of 2026-09-25 to
09-28, which compared the port with MATLAB BADS line by line, slice by
slice (`dev/results/2026-09-28-port-correctness-review.md`). Each entry
keeps the identifier it had there (`KD-<slice>-<n>`, with the slices of
the review, B1 to B7 and S, and T for the tests), so that the review's ledgers, the survey and
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

* Benchmark PyBADS on cognitive and neural science models
  ([neurobench](https://github.com/lacerbi/neurobench)).

`dev/TODO.md` holds the other open work.

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
options whose default is `True` or `False`, but `plot`, and
`uncertainty_handling`, take only booleans: MATLAB's `'on'`, `'off'`,
`'yes'` and `'no'` are refused with `ValueError`. The options checked when
`BADS` is created (the boolean options, `max_fun_evals`, `tol_fun`,
`improvement_quantile`, `hedge_gamma`, `hedge_beta`, `hedge_decay`,
`n_search`, `n_search_iter` and `accelerate_mesh_steps`) refuse a NumPy
array, a 0-d one included, and take a NumPy scalar; the other numeric
options are not checked. A misspelt option name raises, where MATLAB BADS
ignores it.
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
  hyperparameters that no run fills), and `gp_quadratic_mean_bound` and
  `tol_sd` (the `"negquad"` mean, which is refused), are read only by code
  that no run reaches; `hessian_update` and `hessian_method`, only by a
  branch of the search that does nothing.
- *PyBADS only, and refused:* `f_vals`, read only by its check: one that
  holds a finite value is refused, and one without, such as an empty list,
  stands for `None` (W2-7).
- *On both sides, an option in MATLAB BADS and an argument in PyBADS:*
  `fun_values` (MATLAB's `FunValues`, which imports earlier evaluations
  into the log and the GP, `private/setupvars.m:126-167`): a non-empty
  value is refused (W2-6), and `BADS` takes the evaluations as its
  argument `precomputed_evaluations` (KD-B1-15).
- Kind: removed feature (MATLAB's options); deliberate change (interface,
  `fun_values`); Python-only feature (the others).

**KD-B1-5. Options that are parsed and have no effect.**
The options that no code of PyBADS reads are twelve, all named after
MATLAB BADS's options and kept so that a user's setting is not an error;
their descriptions say that they are unused:
- *Read by neither side:* `skip_poll`, `search_improve_frac`, `gp_cluster`,
  `n_basis` (MATLAB reads `Nbasis` only in `poll/private/pollBMADS2N.m`,
  which nothing calls).
- *Hard-coded to MATLAB's default choice:* `poll_method` (KD-B4-1),
  `poll_acq_fcn` (KD-B3-2), `gp_def_fcn` (KD-B6-1), `gp_method` (always
  the nearest neighbours), `chol_attempts` (KD-B6-6).
- *Read by MATLAB BADS only away from its defaults:* `gp_samples` and
  `gp_svd_iters` (KD-B5-4), `rotate_gp`.

The options that no code read and that MATLAB BADS does not have,
leftovers of PyVBMC and of the port (`variational_sampler`, `min_iter`,
`gp_cov_fun`, the `warp_*` options other than `warp_func`, and others, 66
of them in 1.1.0), are not options of PyBADS: setting one raises
`ValueError`, as for any unknown name. Other options are read, but only by
a branch that does nothing or refuses: `plot` (KD-B2-2), `restarts`
(KD-B2-1), `search_optimize` (KD-B3-4), `acq_hedge` (KD-B3-3),
`gp_cov_prior` (KD-B6-7), `fitness_shaping` (KD-B5-5), `hessian_update`
and `hessian_method` (KD-B1-4), a nonzero `warp_func` (KD-B6-4), an
`init_fun` other than `"init_sobol"` (KD-B7-2).
- Settled by: W1-33, W2-35; the PI's ruling at the close of the review
  (the removal of the leftovers). Kind: removed feature.

**KD-B1-6. Periodic variables: indices from 0, a kernel in the units of the others, and a second wrap after the grid.**
As in MATLAB BADS, the hard bounds of a periodic variable, which must be
finite, are its period, and the variable is never taken to log
coordinates. Every point that the initial design, the search (each
generation of the ES search) and the poll propose is wrapped into
`[lb, ub)`; the distances of `udist` and the ES-wcm covariance of `ucov`
take a periodic difference the shorter way round; and the GP's kernel is
periodic along the variable, with its period fixed. The removal of the
closest pair of training points after a failed fit (`_robust_gp_fit_`)
measures Euclidean distances, as `gpHyperOptimize.m` does, so that it
misses a pair close across the bounds. PyBADS differs in three ways.

`periodic_vars` takes indices from 0, as any Python index, where
`PeriodicVars` takes MATLAB's from 1; a boolean mask, a repeated index or
one out of range is refused.

gpyreg's ARD kernels take the periods (`periods`) and replace a periodic
squared difference `d**2` by the squared chord `(p/pi)**2 *
sin(pi*d/p)**2`, which matches `d**2` at short range: a periodic length
scale is in the units of the others, takes their prior and bounds, and
enters `len_scale` and `poll_scale` as they do. MATLAB's `covPPERard_fast`
maps the variable onto the unit circle, `(sin(2*pi*x/p), cos(2*pi*x/p))`,
so that its length scale is in radians. `gpdefBads.m` shifts the centre of
its prior by `-log(p)`, with the `+ log(2*pi)` that would complete the
change of units commented out, which centres the length scale in the
variable's units `2*pi` times shorter than an ordinary one's. `gpupdate.m`
then uses the length scale in radians unconverted in `udist` and
`pollscale`, where it is `pi` times the length in the variable's units
when the period is 2, as it is for a variable whose plausible bounds are
its hard bounds.

A periodic coordinate that the grid takes to its upper bound or past a
bound is wrapped and put on the grid again (`force_to_grid_periodic`): it
stays on the grid, and a point on the upper bound becomes the same point
on the lower one where the grid holds it, so that the removal of the
points already evaluated finds it there; a start on the upper bound is
taken on the lower one too. MATLAB BADS wraps its design, search and poll
candidates only before the grid (its `SearchOptimize`, not ported, wraps
after it), projects a design or search candidate that the grid puts past a
bound onto the bound, drops such a poll candidate under `ForcePollMesh`,
and starts where `x0` lies. Evaluations made before the run are logged
where they are given, as MATLAB BADS logs the points of `FunValues`
(`private/setupvars.m:126-167`, `private/funlogger.m:82`): points on the
two bounds of a periodic variable are two rows of the log, each with its
value, which rounding makes differ for a periodic target.
- PyBADS: `BADS._check_periodic_vars_`, `_variable_transformer_`,
  `_init_optim_state_`, `_init_mesh_`, `_search_step_` and `_poll_step_`;
  `pybads/utils/period_check.py`; `force_to_grid_periodic` and `udist`
  (`pybads/search/grid_functions.py`); `ucov` and `ESSearch.__call__`
  (`pybads/search/es_search.py`); `_gp_periods`
  (`pybads/bads/gaussian_process_train.py`); gpyreg's
  `SquaredExponential`, `Matern` and `RationalQuadraticARD` (`periods`).
- MATLAB: `bads.m:152`, `552`, `807`; `private/setupvars.m:49-57`,
  `107-116`; `private/evalinitmesh.m:106-107`; `search/searchES.m:127-128`;
  `utils/periodCheck.m`; `utils/uCheck.m`; `utils/udist.m`; `utils/ucov.m`;
  `gpdef/gpdefBads.m:58-81`, `277-284`; `private/gpupdate.m:284-308`;
  `gpml_fast/covPPERard_fast.m`.
- Settled by: the PI's rulings of 2026-09-28 (the port, for 1.5; the
  length scale in the units of the other variables; `periods` on gpyreg's
  ARD kernels); W3-35 and W4-11, the call sites of `period_check`. Kind:
  deliberate change.

**KD-B1-7. Fixed variables: a non-finite `x0` stands for the value, and the log and the history hold all the variables.**
A variable whose four bounds are equal is fixed, and the run optimizes the
others, as in MATLAB BADS: the options are evaluated with `D` the number of
free variables, a fixed periodic variable is not periodic in the run, and
the target, `non_box_cons` and the output function receive points of all
the variables.

MATLAB also asks that `x0` equal the bounds, so that at a variable whose
four bounds are equal an `x0` that differs, NaN included, leaves the
variable free, and `setupvars.m` then refuses the order of its bounds.
PyBADS takes a non-finite `x0` there as the value, and refuses a finite
`x0` that differs with a message that names the variable.

MATLAB runs `bads` again on the free variables, with the target,
`non_box_cons` and the output function wrapped by `expandvars`, and lifts
`x` and, with five outputs or more, `optimState.X` back to all the
variables. PyBADS's variable transform, which every point that leaves the
run goes through, drops the fixed coordinates of the points it takes and
puts them back into those it returns, so that the result's `x` and `x0`,
the function log's `X_orig` and the points `"x"` of `iteration_history`
hold all the variables too, while `optim_state`, the transformed points and
the GP cover the free ones, as MATLAB's `optimState` does but for its `X`.

MATLAB rewrites `PeriodicVars` over the free variables through `eval`,
which fails on the numeric value that `bads_examples.m` passes, and drops
the fixed columns of `FunValues.X` unchecked. PyBADS keeps `periodic_vars`
as given, over all the variables, and refuses a point of
`precomputed_evaluations` whose coordinate at a fixed variable differs from
its value, as outside the hard bounds.

With every variable fixed, MATLAB fails in its call on no variables (`If no
starting point is provided, PLB and PUB need to be specified`), and PyBADS
raises a `ValueError` saying that there is nothing to optimize. PyBADS
names the fixed variables from `display="notify"` on, as it names those on
a log scale and the periodic ones; MATLAB does not.
- PyBADS: `_bounds_as_rows`, `_find_fixed_values`, `_run_indices` and
  `_user_indices`, `BADS.__init__`, `_check_periodic_vars_` and
  `_import_precomputed_evaluations_` (`pybads/bads/bads.py`);
  `VariableTransformer`'s `fixed_values`; `FunctionLogger`'s `D_orig`.
- MATLAB: `private/boundscheck.m:39-40`; `bads.m:351-382`, `1480-1488`
  (`expandvars`); `private/fixedbads.m`; `private/setupvars.m:7-9`.
- Kind: deliberate change.

**KD-B1-8. The result is an `OptimizeResult` dict, not MATLAB's six outputs.**
PyBADS returns a SciPy-style dict (`x`, `x0`, `fval`, `fsd`, `yval_vec`,
`ysd_vec`, `func_count`, `iterations`, `mesh_size`, `message`,
`target_type`, `problem_type`, `total_time`, `overhead`, `random_seed`,
`algorithm`, `version`, `fun`, `non_box_cons`, `success`, `status`, and
`precomputed_observations` and `precomputed_locations` for a run given
evaluations made before it, KD-B1-15), and the `BADS` object keeps the
run's state. `status` is MATLAB's `exitflag`
(0 at `max_fun_evals` or `max_iter`, or when `output_fcn` stops the run; 1
on `tol_mesh`; 2 on the stall criterion), and `success` is `status > 0`.
There is no `rngstate` (KD-B1-1) and no `maxconstraint`. `total_time`
times `optimize()` and leaves out the creation of `BADS`, where MATLAB
BADS times the whole call, its setup included (`bads.m:144`, `1186`), and
`overhead` follows it on both sides. `iterations`
counts from 1 as MATLAB's does, but a run that ends in its initialization
reports 0, where MATLAB BADS reports 1. `fun` and `non_box_cons` are the
objects passed, where MATLAB stores `func2str(fun)`. `yval_vec` is `None`
for a deterministic run, with `noise_final_samples = 0`, and when the
budget leaves no evaluation for the final samples, where MATLAB BADS
returns the incumbent's observation (`bads.m:1136`).
- PyBADS: `pybads/bads/optimize_result.py` (`OptimizeResult`);
  `BADS._optimize_`.
- MATLAB: `bads.m:1`, `144`, `423`, `1062-1083`, `1185-1194`;
  `private/bads_output.m`.
- Settled by: W0-4, W2-12, W2-13, W2-14, W2-32; the PI's ruling of
  2026-09-28 on the loose ends of the review (`total_time`), in the
  review's ledger (`dev/results/2026-09-28-port-correctness-review.md`,
  "Open ends"). Kind: deliberate change (interface).

**KD-B1-9. A given start that `non_box_cons` rejects once put on the mesh is refused.**
PyBADS tests `non_box_cons` at a given start, and a second time after
`force_to_grid`, and raises `ValueError` if the point on the mesh violates
it; MATLAB BADS tests only the start as given and evaluates the point on
the mesh. PyBADS tests a random start as drawn and on the mesh, MATLAB
BADS on the mesh (KD-B1-11).
- PyBADS: `BADS.__init__`; `BADS._init_optim_state_`.
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
plausible box, puts it on the mesh, and stops with an error when that point
violates `non_box_cons`. PyBADS draws again while the point as drawn or
the point on the mesh violates it, up to 1000 draws in all, and then raises
the same error; a run whose first draw is feasible, as drawn and on the
mesh, starts where MATLAB's would from the same numbers. A defect that
PyBADS shared and fixes.
- PyBADS: `BADS._init_optim_state_`.
- MATLAB: `private/setupvars.m:83-85`, `101`;
  `private/evalinitmesh.m:22-26`.
- Settled by: W2-11; the PI's ruling at the close of the review (the test
  as drawn and on the mesh). Kind: deliberate change.

**KD-B1-12. A missing `x0` with only the hard bounds is accepted.**
With `x0` missing and the plausible bounds omitted, MATLAB BADS refuses
the problem (`bads.m:332-343`). PyBADS takes the hard bounds for the
plausible bounds and draws the start in the plausible box, as it draws any
missing start.
- PyBADS: `BADS.__init__`.
- MATLAB: `bads.m:332-343`.
- Settled by: W2-9. Kind: deliberate change.

**KD-B1-13. `tol_fun` is checked when `BADS` is created.**
`tol_fun` must be a positive real number (a Python or NumPy integer or
float) at most e^6: a boolean, a string, an array or a complex number is
refused. MATLAB BADS does not check it. Above e^6 the bounds of
the GP's log noise SD, `log(tol_fun) - 1` and 5, cross
(`gpdef/gpdefBads.m:161`); in PyBADS 1.1.0 such a value stopped a run at
its first fit, and 0 stopped it with a `ZeroDivisionError` while the
default of `hedge_beta`, `1e-3 / tol_fun`, was evaluated.
- PyBADS: `BADS._check_tol_fun_`.
- MATLAB: `bads.m:193`; `private/setupoptions.m:23`.
- Settled by: #84; the PI's ruling at the close of the review (real
  numbers only). Kind: deliberate change.

**KD-B1-14. MATLAB's extra arguments to the target and its other calling forms are not ported.**
MATLAB BADS passes the arguments that follow `options` to the target,
`fun(x, varargin{:})`, and `bads('defaults')`, `bads('all')`,
`bads('test')` and `bads('version')` return its basic options, all its
options, the results of its test problems and its version. PyBADS calls
the target with `x` alone, so a target that needs more arguments is bound
to them by the user (with `functools.partial`, for instance); its options
are the two `.ini` files (KD-B1-2), its tests are pytest's (KD-T-1), and
its version is the installed package's
(`importlib.metadata.version("pybads")`).
- PyBADS: `BADS.__init__`; `FunctionLogger`.
- MATLAB: `bads.m:1`, `163-182`, `293-296`, `402-406`.
- Settled by: the PI's ruling at the close of the review. Kind: unported
  feature (interface).

**KD-B1-15. Evaluations made before the run are an argument, `precomputed_evaluations`, with checks of their own.**
MATLAB BADS imports the evaluations of its option `FunValues`, a struct of
points `X`, values `Y` and optionally SDs `S`, into its log when it sets
up, checking their shapes and that they are finite and real; its count of
evaluations starts at 0 after them, and its first incumbent is the best of
the start and the points of the initial design that it evaluates. PyBADS
refuses `fun_values` (KD-B1-4) and takes the evaluations as the keyword
argument `precomputed_evaluations=(X, y)`, or `(X, y, y_sd)`, PyVBMC's
interface, into its log with the same count and the same rule for the
first incumbent. It also refuses:
SDs without `specify_target_noise` and their absence with it, which MATLAB
does not check (an `S` without `SpecifyTargetNoise` makes its logger ask
the target for two outputs, and the converse fails when it pads `S`);
points outside the hard bounds or that violate `non_box_cons`, which
MATLAB takes; and, unless `uncertainty_handling` is `True`, a point given
twice with two different values. The check comes before the noise test, so
that an `uncertainty_handling` left empty counts as none, and a point
given twice with one value is kept once, where MATLAB adds a row per
repeat. With uncertainty handling each repeat is an observation, a row at
level 1 and merged into its point's row at level 2 (KD-B7-3). So at level
2 the run's evaluation of its start merges with an evaluation given at the
start's point, where MATLAB adds a row: PyBADS's first incumbent takes the
merged value and its first GP the merged row, MATLAB's first incumbent the
new observation and its first training set both rows. Neither side
evaluates a point of the initial design that the log holds, so that a run
given the log of an earlier run with the same seed and start evaluates its
start alone, which is its first incumbent. The GP takes the evaluations
given at its first rebuild, at the first poll, among the neighbours of the
incumbent, as MATLAB's does: PyBADS's initial fit leaves them out
(KD-B6-5), as does the schedule of the GP's fits, PyBADS's own (KD-B5-6),
which spans the run's own evaluations. The result counts them in
`precomputed_observations` and `precomputed_locations` (KD-B1-8).
- PyBADS: `BADS.__init__`, `_import_precomputed_evaluations_` and
  `_init_mesh_` (`pybads/bads/bads.py`); `FunctionLogger.add`;
  `init_and_train_gp` and `_get_gp_training_options`
  (`pybads/bads/gaussian_process_train.py`); `OptimizeResult`.
- MATLAB: `private/setupvars.m:126-167`; `private/funlogger.m:30-85`;
  `private/evalinitmesh.m:111-123`.
- Settled by: W2-6, W4-10; the PI's rulings on the port (2026-09-28), in
  the review's ledger (`dev/results/2026-09-28-port-correctness-review.md`,
  "Open ends"). Kind: deliberate change (interface).

**KD-B1-16. `max_iter`, `tol_stall_iters`, `search_n_try`, `noise_final_samples`, `tol_mesh` and `noise_size` are checked when `BADS` is created.**
MATLAB BADS checks none of the first five and only that the base of
`NoiseSize` is positive. PyBADS refuses, with a `ValueError` that names
the option: a `max_iter` or `tol_stall_iters` that is not a positive
integer or `inf`, which turns the limit off, as for `max_fun_evals`; a
`search_n_try` or `noise_final_samples` that is not an integer at least 0
(0 searches is a run whose every iteration is a poll, on both sides); a
`tol_mesh` that is not a positive finite real number; and a `noise_size`
that is not one or two real numbers, or, without `specify_target_noise`,
whose base is not positive and finite or whose SD of the prior over the
log noise SD is finite and not positive. Whole-number floats are
converted to integers. On MATLAB's side, a string such as `'200*nvars'`
is evaluated; a `SearchNtry` that is not whole ends no round of searches,
after which its loop turns without evaluating; a `TolStallIters` or a
`NoiseFinalSamples` that is not whole stops the run at an index or at the
array of the final samples; a `TolMesh` of 0 never ends a run on the
mesh; a NaN or infinite base of `NoiseSize` passes its check; and a
negative SD of the prior runs as its absolute value, since the prior takes
its square. An SD of the prior that is not finite stands for 1 on both
sides.
- PyBADS: `BADS._init_optim_state_`, `_as_limit` (`pybads/bads/bads.py`).
- MATLAB: `bads.m:441-442`, `516`, `744`, `1067`, `1077-1079`,
  `1449-1457`; `private/setupvars.m:105`, `173`;
  `private/setupoptions.m:72-82`; `gpdef/gpdefBads.m:147-151`;
  `private/gpupdate.m:379-381`.
- Settled by: the `dev/TODO.md` item on the checks of option values
  (2026-09-29). Kind: deliberate change.

### The main loop, termination and the final estimate (B2)

**KD-B2-1. Restarts are not implemented.**
With `restarts > 0`, MATLAB BADS resets the mesh and continues after
termination ("Multiple starts (deprecated)"); PyBADS stops. Both default
to 0.
- PyBADS: `BADS._optimize_` (the branch on `restarts` does nothing).
- MATLAB: `bads.m:201`, `479`, `1121-1127`.
- Kind: unported feature.

**KD-B2-2. Plotting is not implemented.**
`plot` has no effect; MATLAB BADS draws a profile (`utils/landscapeplot.m`)
or a scatter plot (`private/scatterplot.m`).
- PyBADS: `BADS._optimize_`; `advanced_bads_options.ini`.
- MATLAB: `bads.m:187`, `988-1015`, `1054-1057`.
- Kind: unported feature.

**KD-B2-3. Messages go through Python logging, to the `BADS` logger.**
`display` sets the level of a logger instead of choosing which `fprintf`
calls run. The level follows MATLAB's reading of the first three letters,
lower case: `"off"` and `"none"` show the warnings only, `"notify"` and
any other value also the opening message, `"final"` also the final
message, `"iter"` and `"all"` also the iteration lines, and `"full"`,
which only PyBADS has, the debug messages too. The reports of the setup
(the caution for infinite bounds, the variables on a log scale) are shown
from `"notify"` on, as MATLAB BADS prints them, and its warnings, such as
`bads:pbUnspecified`, at every level, as MATLAB's `warning` shows whatever
`Display` says. The content and format of the other lines may differ.
- PyBADS: `BADS.__init__`, `BADS._bounds_check_` and
  `BADS._init_optim_state_` (the reports and warnings of the setup), and
  the display methods of `BADS`; `pybads/bads/gaussian_process_train.py`.
- MATLAB: `bads.m:311-328`; `private/setupvars.m:28-39`, `118-123`;
  `private/boundscheck.m:12-16`; `fprintf` throughout.
- Settled by: W2-15; the PI's rulings at the close of the review (the
  setup's reports, `bads:pbUnspecified`). Kind: deliberate change.

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
- PyBADS: `BADS._optimize_`, `BADS._init_optimization_`.
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
PyBADS shared and fixes. It changes nearly every noisy run, but not their
errors or fraction solved measurably (90 seeds against the move of the
value alone, `dev/experiments/w225_linux_20260928/`).
- PyBADS: `BADS._optimize_` (the move after `_re_evaluate_history_`,
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
- PyBADS: `BADS._optimize_` (the final estimate).
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
- PyBADS: `BADS._optimize_`.
- MATLAB: `bads.m:1111-1118`, `1150-1165`.
- Settled by: W3-33, W4-26. Kind: deliberate change.

**KD-B2-10. A search runs once the log holds more than D points.**
Each pass of the loop runs a search while the round has searches left and
more than D points are counted: PyBADS counts the points of the function
log, MATLAB BADS those of the GP's training set (`size(gpstruct.y,1) >
nvars`). The round's first search rebuilds the GP from the log, on both
sides, so that PyBADS counts the points that the search trains on, while
MATLAB's test reads the GP before that rebuild. The two differ when a poll
follows an initial design that leaves D points or fewer, as `non_box_cons`
can: the poll's evaluations join the log and, at uncertainty level 0, not
the GP (W3-26), so that at the next pass PyBADS runs a search that MATLAB
BADS skips. With default options this happens at one pass of each run of
`sphere_band_D3` over seeds 0-6, and of three of the seven runs of
`sphere_nonbox_D3` (`dev/scripts/benchmark_targets.py`).
- PyBADS: `BADS._optimize_` (`do_search_step_flag`); `BADS._search_step_`.
- MATLAB: `bads.m:516-517`, `522-536`.
- Settled by: the PI's ruling of 2026-09-28 on the loose ends of the
  review, in the review's ledger
  (`dev/results/2026-09-28-port-correctness-review.md`, "Open ends").
  Kind: deliberate change.

### The search (B3)

**KD-B3-1. The search hedge chooses between ES-wcm and ES-ell; the other search methods are not ported.**
Only MATLAB's default set of searches exists: `ESSearchWM`, `searchES`'s
method 1 (`'ES-wcm'`), and `ESSearchELL`, its method 2 (`'ES-ell'`). The
other methods of `searchES` (`ES-eye`, `ES-cov`, `ES-cma+`) and the other
search functions are absent. A `search_method` that is not a non-empty
list of pairs (name, sum-rule flag) named `"ES-wcm"` or `"ES-ell"` is
refused when `BADS` is created, an entry with more elements included, such
as MATLAB's triple `{@searchES, 1, 1}`; MATLAB BADS checks nothing.
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
sqrt_beta)`, with no further element, is refused when `BADS` is created.
Both default to the LCB.
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
- Settled by: the rulings of wave 4's doublecheck, confirmed by the PI at
  the close of the review. Kind: deliberate change.

**KD-B3-10. Of the candidates that share a bin, the first is kept.**
Before the initial design, a search or a poll evaluates its candidates,
PyBADS's `contraints_check` and MATLAB BADS's `uCheck` bin them on a grid
of `tol_mesh / 2` and keep one candidate of each bin that holds no
evaluated point, the bins sorted. PyBADS keeps the first of the bin in the
candidates' order; MATLAB BADS, whose `unique(U,'rows')` sorts the
candidates before they are binned, keeps the smallest. The two differ by
less than a bin, 2^-20 in `u` at the default `tol_mesh`, and return their
bins in the same order.
- PyBADS: `contraints_check` (`pybads/function_logger/constraints_check.py`).
- MATLAB: `utils/uCheck.m:14-27`.
- Settled by: W3-2. Kind: deliberate change.

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
- PyBADS: `BADS._init_optimization_` (the setting), `BADS._init_optim_state_`
  (the warning); `local_gp_fitting`.
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
cannot be changed (`gp_def_fcn` has no effect); it is periodic along the
periodic variables (KD-B1-6). gpyreg's
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
prior and the previous centre of the output scale's, and at the
initialization gives the mean's prior the SD 1, the width of MATLAB's
definition, centred on one distinct point at its target (KD-B6-5). When
the pairwise distances of the training set have no spread (two distinct
points), MATLAB's empirical prior of the length scales has a zero width,
which gpyreg refuses; PyBADS keeps the previous prior. Otherwise the
re-centred priors follow MATLAB BADS. What MATLAB's fit does with these
zero-width priors is not known without MATLAB.
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
  (`warp_func`).
- MATLAB: `bads.m:279-281`; `gpdef/gpdefBads.m:116-118`, `210-215`,
  `287-291`; `warp/`.
- Kind: removed feature.

**KD-B6-5. PyBADS fits a GP on the initial design; MATLAB BADS only defines it.**
PyBADS fits the hyperparameters on the start and the initial design,
without the evaluations made before the run (KD-B1-15), under the priors
of the definition; MATLAB BADS keeps the definition's values until its
first refit. Both rebuild the GP first at the first poll, whose
neighbours of the incumbent include the evaluations made before the run.
The first refit needs the run's count of evaluations past D and either
max(10, 2D) of the GP's predictions at points that the run then evaluated
or a failed check of their calibration, which fails when there are none
(`IsRefitTime`). After an initial design the count is past D and there
are none, so that both refit at the first poll's rebuild, and the initial
fit reaches a run as one start of that refit and as the hyperparameters
of the first target's prediction. The mean's start and prior come from
the start and the design, where MATLAB's definition takes the median of
the lowest 80 % of the whole log's values, the evaluations made before
the run included. On one distinct point (a feasible region too thin for
the initial design, `max_fun_evals=2` with the noise test,
`fun_eval_start=0` at level 0, since uncertainty handling raises it to 20,
or a log that holds the whole design, KD-B1-15), where the priors alone
would decide the fit, PyBADS does not fit either: the GP holds the
definition's values, as MATLAB BADS's does, the log length scales, the
log output scale and the log shape at 0, the log noise SD at the log of
the noise size and the mean at the point's
target. The mean's prior is centred at that target with the SD 1, where
MATLAB's definition centres it at 0; both sides re-centre it at each
rebuild, before any fit. A run given the log of an earlier run with the
same seed and start thus works, on both sides, on the whole log's
neighbours with the definition's values until its first refit, which
comes once its own evaluations bring the count past D and then either the
predictions or a failed check: in PyBADS's reruns of the `warmstart`
suite at seeds 0-9, at a `func_count` of 4 or 12 at D = 3 without noise,
11 with it, and 14 (12 in one seed) for Rosenbrock's function at D = 6. A
first GP fitted on the whole log's neighbours in the initialization was
measured on such runs and on runs given the log of a run from another
seed, and not adopted: it cost `rosenbrock_D6_rerun` more evaluations and
improved no configuration significantly
(`dev/experiments/warmstart_gp_linux_20260929/`).
- PyBADS: `init_and_train_gp`, `_gp_hyp`; `BADS._init_optimization_`,
  `BADS._is_gp_refit_time_`.
- MATLAB: `bads.m:465-469`, `821-839`, `1223-1244` (`IsRefitTime`);
  `gpdef/gpdefBads.m:48`, `147-154`, `164-173`, `219-220`;
  `private/setupvars.m:173`.
- Settled by: W1-27; the PI's rulings of 2026-09-28 (one point), in the
  review's ledger (`dev/results/2026-09-28-port-correctness-review.md`,
  "Open ends"), and of 2026-09-29 (the first GP of a run given evaluations
  made before it, `dev/experiments/warmstart_gp_linux_20260929/`). Kind:
  deliberate change.

**KD-B6-6. A failed Cholesky factorization multiplies the GP's noise.**
gpyreg multiplies the noise by ten per failed attempt, up to ten attempts,
and keeps the multiplier in the posterior (`sn2_mult`), which
`get_hyperparameters` does not show; MATLAB BADS (`CholAttempts = 0`)
treats the failure as an error, restarts the fit with the noise's start
nudged, and empties the posterior. It is reached at default options on
some targets: at the end of the review, a fifth of the fits' factorizations
in deterministic runs, half on the 3-D ellipsoids, where the output
variance exceeds the noise floor by far more than double precision holds;
the bounds of both are MATLAB's. gpyreg's switch
`raise_on_cholesky_failure`, off by default, gives MATLAB's behavior;
measured at the head of the review, it made most refits on the 3-D
ellipsoids fail, 10 runs of the deterministic ones end above the tolerance
(none before) and 3 runs of `ellipsoid_D3_homo` stop at their start, for
gains mostly below the tolerance, and it stays off in PyBADS.
`chol_attempts` is unread.
- PyBADS: gpyreg's `GP` (its training Cholesky factorization, and
  `predict`).
- MATLAB: `bads.m:272`; `gpml_fast/infExact_fastrobust.m:36`, `77-80`;
  `utils/gpHyperOptimize.m:73-176`; `private/gpupdate.m:340-354`.
- Settled by: W1-25; the PI's ruling of 2026-09-28 after the measurement
  (`dev/results/2026-09-28-gp-health.md`). Kind: substituted library.

**KD-B6-7. `gp_cov_prior="ard"` is not supported, and is refused.**
MATLAB's `'ard'` sets an empirical prior of the length scales per
dimension. PyBADS has only MATLAB's default, `'iso'`, one empirical prior
shared by all the length scales, and refuses any other value when `BADS`
is created.
- PyBADS: `BADS._init_optim_state_`; `local_gp_fitting`.
- MATLAB: `gpdef/gpdefBads.m:254-274`.
- Settled by: W1-28; the PI's ruling of 2026-09-28 not to port `'ard'`,
  which is off by default in MATLAB BADS. Kind: removed feature.

**KD-B6-8. A fixed noise (`fit_lik=False`) is refused on both sides.**
PyBADS refuses it when `BADS` is created, MATLAB BADS when it defines the
GP, with the same message.
- PyBADS: `BADS._init_optim_state_`.
- MATLAB: `bads.m:466`; `gpdef/gpdefBads.m:139-140`.
- Settled by: W1-32. Kind: removed feature.

**KD-B6-9. A `noise_size` above e^5 is warned about.**
Both sides bound the GP's log noise SD above at 5, whatever the target's
scale, so that a larger noise sits at the bound (a shared design
observation of `dev/experiments/port_review_20260925/matlab_side_defects.md`).
Without `specify_target_noise`, PyBADS warns when `BADS` is created with a
`noise_size` above e^5, about 148, and proposes to rescale the target;
MATLAB BADS does not warn.
- PyBADS: `BADS._init_optim_state_`; `_gp_hyp`.
- MATLAB: `gpdef/gpdefBads.m:161`.
- Settled by: W1-30; the PI's ruling at the close of the review (the
  entry). Kind: Python-only feature.

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
`funlogger` adds a row per evaluation and returns the observation. A run
of `BADS` reaches the merge only through the evaluations made before it
(`precomputed_evaluations`, KD-B1-15), at a point given twice or at the
start given among them, since `contraints_check` removes a point already
evaluated before it is evaluated and the noise test and the final samples
record nothing.
Returning the observation, as MATLAB does, was tested and not adopted. At
levels 0 and 1 a repeat is a new row on both sides.
- PyBADS: `FunctionLogger` (`pybads/function_logger/function_logger.py`);
  `BADS._import_precomputed_evaluations_`.
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

**KD-B7-6. A target that returns `(f, sd)` without `specify_target_noise` is refused.**
Without `SpecifyTargetNoise`, MATLAB's `funlogger` asks the target for one
output, so that a target that also returns an SD runs with the SD dropped.
PyBADS's function logger takes the target's return as its value, and a
tuple is not a scalar: it raises `ValueError`, at the run's first
evaluation, with a message that names `options["specify_target_noise"] =
True` when the tuple has two elements.
- PyBADS: `FunctionLogger.__call__`
  (`pybads/function_logger/function_logger.py`).
- MATLAB: `private/funlogger.m:91`, `95-99`.
- Settled by: the PI's ruling of 2026-09-28 on the loose ends of the
  review, in the review's ledger
  (`dev/results/2026-09-28-port-correctness-review.md`, "Open ends").
  Kind: deliberate change.

### Sto-BADS (S)

**KD-S-1. Sto-BADS is PyBADS's own.**
With `stobads` on, a success rule based on uncertainty intervals, after
Sto-MADS (Audet, Dzahini, Kokkolaras and Le Digabel, 2021), replaces the
improvement tests of the search and the poll; `opp_stobads` and
`stobads_frame_size_scaling_power` tune it, and the keyword-only argument
`gamma_uncertain_interval` of `BADS` sets its interval. It is off by
default and switched off for a deterministic target. It is deprecated
from 1.5.0 and may be removed in a future release (PI, 2026-09-30), so
that no user takes it for the setting of a noisy target: the descriptions
of its options and of `gamma_uncertain_interval` say so, and `stobads=True`
warns. Its rule was measured
at the close of the review (`dev/results/2026-09-28-stobads-rule.md`):
the mesh factor of its interval stays, and the description of
`stobads_frame_size_scaling_power` says what it does (W0-12); an uncertain
search, as an uncertain poll, moves the incumbent only to a point that
improves on it (W0-13).
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
