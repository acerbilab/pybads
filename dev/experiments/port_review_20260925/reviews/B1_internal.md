<!-- Report of the B1 internal reviewer (setup, options, defaults, bounds, transform and result, internal-correctness track), wave 2 of the port review, reading PyBADS at fef6c14 in ../pybads-review and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to verification/scripts/wave2/B1_internal/. Nothing in it is verified. -->

# B1 internal review: setup, options, defaults, bounds, transform and result

## 1. Coverage

**Read completely:**
- `pybads/bads/bads.py` 1-1320 (`__init__`, `_bounds_check_`, `_init_optim_state_`, `_variable_transformer_`, `_init_rng_`, `_init_mesh_`, `_init_optimization_`, the start of `optimize`), 1425-1676 (termination and result), 2721-2780 and 2836-2894.
- `pybads/bads/options.py`, both option files, `options_confs.py`, the two test `.ini` files.
- `pybads/variable_transformer/variables_transformer.py`, `pybads/search/grid_functions.py`, `pybads/rng.py`, `pybads/bads/optimize_result.py`, the package `__init__.py` files.
- Docs: `docsrc/source/api/classes/{bads,optimize_result,options,parameter_transformer,classes}.rst`, `api/options/bads_options.rst`, `api/advanced_docs.rst`, `quickstart.rst`.

**Skimmed:**
- `init_functions/init_sobol.py:40-90` (the design size).
- The reads of `noise_size` in `gaussian_process_train.py` (318-336, 975-1000).
- Tests: `test_variable_transformer.py`, `test_bads_seed.py`, `test_run_control.py`, `test_init_conf.py`, the header of `test_bads_optimization.py`.

**Not reached:**
- The main loop in detail (B2).
- The internals of `FunctionLogger`.
- The BADS paper itself (not consulted; no finding depends on it).
- MATLAB, not opened (internal track).

All checks ran from the scratchpad directory, with `pybads.__file__` = `/home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__` = `/home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. They were construction-only checks or seeded `optimize()` runs of at most 100 evaluations.

## 2. Answers to the first questions

### Q1. The options and their defaults

**Layering does what the documentation says.**
- The basic file is loaded, then the user's dict (`options.py:49-51`), then the advanced file, which skips the user's keys (`options.py:116`). `D` is bound by `exec` into the module globals.
- An advanced expression that reads another option through `self.get` sees the user's value. With `tol_fun=1e-2`, `tol_noise` becomes 2.2e-18 and `hedge_beta` becomes 0.1.
- A misspelt name raises `ValueError` (checked with `maxfunevals`).
- A user expression such as `"20*D"` stays a string and fails with `TypeError` in `optimize` (settled, KD-B1-3).
- **A user value of `None`** is replaced by nothing. Options whose default is `None`, and `specify_target_noise`/`stobads` (None becomes False), treat it as intended. Numeric options fail with `TypeError`: `max_fun_evals`, `max_iter`, `tol_fun`, `fun_eval_start`, `tol_stall_iters`, `search_n_try` and `cache_size` in `optimize`; `tol_mesh`, `poll_mesh_multiplier` and `init_mesh_size_integer` at construction. Boolean options read `None` as False, so `nonlinear_scaling=None` turns off the log transform, whose default is True.
- **Wrong types are not checked:** `uncertainty_handling='off'` gives a noisy run (level 1), and `nonlinear_scaling='off'` keeps the log transform on (F13).

**Descriptions against the code:**
- Correct:
  - `tol_stall_iters` is doubled under uncertainty (`bads.py:1156`).
  - `n_train_max` becomes at least 200 and `n_train_min` is doubled (`1159-1160`).
  - `fun_eval_start` becomes at least 20 for a noisy target (`1067-1071`); rounded up to a power of two, doubled when that equals D (`init_sobol.py:73-75`).
  - `noise_final_samples` is reserved from `max_fun_evals` (`1176-1184`).
  - `uncertainty_handling`, `specify_target_noise` and `noise_size` (scalar or pair) behave as described.
  - `periodic_vars`, `gp_mean_fun`, `gp_cov_prior` and `fit_lik` refuse unsupported values.
  - The `noise_size` default is 1.0 at levels 1 and 2 (as the `BADS` docstring says) and `sqrt(tol_fun)` at level 0 (`1199`). The level-0 default is not in any description.
- Wrong or dead:
  - `display` (F11).
  - `fun_values` and `f_vals` do not work (F6).
  - `min_fun_evals`, `min_iter` and `search_factor_min` are read by nothing, and KD-B1-5 does not list them (F12).
- `tol_noise = eps*tol_fun`, about 2.2e-19 at default, is an absolute threshold below the float resolution of any |f| > 1e-3, so the noise test flags any non-identical repeat. The description gives no magnitude; the comparison track should check it against `defopts`.
- **Cosmetic:**
  - `configparser` splits comment lines at `=` or `:`, so `Options.descriptions` truncates `noise_size`, `periodic_vars` and `stobads_frame_size_scaling_power`. This only affects `str(options)`; the docs page includes the files verbatim.
  - `n_basis`, `gp_cov_fun`, `search_factor_min`, `use_effective_radius` and `gp_fixed_mean` have no description line.
  - Many descriptions end with a stray `'`.

**D-dependent defaults at D = 1 / 2 / 6 / 20:**

| Option | D = 1 | D = 2 | D = 6 | D = 20 |
|---|---|---|---|---|
| `max_iter` | 200 | 400 | 1200 | 4000 |
| `max_fun_evals` | 500 | 1000 | 3000 | 10000 |
| `tol_stall_iters` | 4 | 5 | 7 | 14 |
| `fun_eval_start` | 1 | 2 | 6 | 20 |
| `tol_poi` | 1e-6 | 5e-7 | 1.67e-7 | 5e-8 |
| `mesh_overflow_warning` | 2.5 | 3 | 5 | 12 |
| `search_n_try` | 3 | 4 | 6 | 20 (np.float64) |
| `n_train_max` | 60 | 70 | 110 | 250 |
| `min_refit_time` | 2 | 4 | 12 | 40 |
| `hedge_decay` | 0.316 | 0.562 | 0.825 | 0.944 |
| `min_fun_evals` (unread) | 5 | 10 | 30 | 100 |
| `min_iter` (unread) | 1 | 2 | 6 | 20 |
| `n_basis` (unread) | 200 | 400 | 1200 | 4000 |

### Q2. The bounds and the starting point

**Order of the setup:**
1. `plb`/`pub` default to copies of `lb`/`ub` (`bads.py:181-184`).
2. `x0=None` needs `plb`/`pub` after step 1, or raises; `x0` then becomes a NaN row. `D` is the number of columns of `x0`. `lb`/`ub` of `None` become ∓inf.
3. In `_bounds_check_`, the plausible bounds are estimated from a starting set when still missing and `N0>1`, as `min - width/N0` and `max + width/N0` clipped to the hard bounds. This is reachable only without `lb`/`ub`. Otherwise the plausible bounds become the hard ones.
4. All four bounds must have shape (1, D) (F2).
5. `plb` and `pub` must be finite.
6. All inputs must be real.
7. Fixed variables are refused (KD-B1-7).
8. `plb == pub` is refused.
9. `x0` outside the hard bounds is refused.
10. Effective bounds: 0.1% of the hard range inside each bound; `LB_eff >= UB_eff` is refused. The special case `|lb| <= realmin` gives the same value as the general formula, so it is a no-op.
11. `x0` is clipped into [LB_eff, UB_eff].
12. A permissive order check. Its message says `<` where the check is `<=`.
13. `plb`/`pub` are moved into [LB_eff, UB_eff] (F4).
14. The "x0 inside the plausible bounds" check (F3).
15. The order check again.
16. The half-bounds check (F1).
17. `non_box_cons` is called on `[plb; pub]` as a shape check (F10).
18. A warning for infinite bounds.

**Then in `__init__`:**
- If any coordinate of `x0` is not finite, the whole `x0` is redrawn, the finite coordinates included. It is drawn uniformly in the transformed plausible box, which is log-uniform for log-transformed variables. The docstring says "uniformly ... inside the plausible box", and the quickstart says "from the problem's bounds".
- An infinite `x0` never reaches the redraw. With a finite bound it is refused as outside the bounds. With infinite bounds it expands `pub` to inf and is refused by the transform as a non-finite plausible range.
- `non_box_cons(x0) > 0` raises, also for the random `x0` (F10).

**Then in `_init_optim_state_`:**
- `u0` is put on the search grid and shifted one step back inside the hard bounds.
- `non_box_cons` is checked again and raises if the point on the grid is infeasible. An `x0` within half a grid step of the constraint boundary is therefore refused rather than moved.

**Accepted and refused against the documentation:**
- The documented order `lb <= plb < pub <= ub` and finite plausible bounds hold.
- Refusals that contradict the docstring: mixed bounded and unbounded variables (F1) and scalar bounds (F2).
- A multi-row `x0` is accepted, then crashes (F7).
- With `x0=None`, list-valued plausible bounds raise `AttributeError` (`bads.py:196`). The docstring types are `np.ndarray`, so this is outside the contract.
- The warning at 345-347 has a typo, "Estimatingplausible".

### Q3. The transform and the grid

**The transform is correct on random inputs.** Five cases, 1000 points each, compared with an independent transcription (log where flagged, then an affine map of `plb` to -1 and `pub` to 1): linear and finite; all log; mixed log with an unbounded variable; all unbounded; a ratio of exactly 10.
- The forward and inverse maps agree exactly.
- The round trip errs by at most 2e-15 relative.
- `plb` maps to -1 and `pub` to 1. Infinite bounds stay ∓inf in u.

**Log criterion.** All of `lb`, `ub`, `plb`, `pub` must be > 0, and `pub/plb >= 10`. The docstring says "more than one order of magnitude"; the code comment says "at least".

**`maskindex`.** Correct; it requires 2-D input.

**Shapes.** The forward map returns (1, D) for a 1-D input, while the inverse keeps the input's shape. This is harmless.

**Self-test.** The self-test of the transform uses an absolute tolerance (F5).

**Standalone `VariableTransformer`:**
- A scalar `apply_log_t` raises `AttributeError` (line 104 reads `self.apply_log_t` before it is set).
- Python-float bounds raise at `.copy()`, so the `np.isscalar` branches are dead for Python floats.
- `BADS` always passes arrays, so neither is reached from `BADS`.

**`grid_units`.** Its only caller (`bads.py:672`) passes `scale` positionally into the `x0` parameter. This is harmless, since `var_trans` is given. A 1-D `x` with D>1 would raise `IndexError`; no caller does this.

**Grid, search bounds and `tol_mesh`:**
- The search mesh is `2**min(0, 0*2-10)` = 2^-10.
- `lb_search`/`ub_search` are the first grid points inside the hard bounds.
- `tol_mesh` is placed at `mult**ceil(log(tol)/log(mult))`. Termination at "mesh < placed tol" is equivalent to "mesh < tol" for power-of-mult meshes; checked for 2^-k, k = 1..40.
- The evaluated start point is the point on the grid, not `result["x0"]`: 0.3 became 0.30018 in one check.

### Q4. The result and the seed

**Result fields:**
- `x` is `inverse_transf(u)`, 1-D.
- `x0` is the `x0` after the bounds check and the random draw, 2-D (1, D), and not the evaluated point on the grid.
- `fval` (deterministic) is the observed value at `x` (checked).
- `fsd` is 0 at level 0, as documented. It came back as `int` 0 in the check, while the docstring says float.
- `iterations` is `iter + 1`, and 0 for a run that ends at initialization.
- `func_count` includes the noise test and the final samples.
- `mesh_size` is the poll mesh in u space.
- `message` is that of the last criterion checked.
- `yval_vec`/`ysd_vec` follow their docstring.
- `problem_type` and `target_type` are correct.
- `status` is never set, and `success` is always True (F8).
- `fun` and `non_box_cons` are deep copies (F9).

**`random_seed`:**
- An int, a whole float (Python, `np.float32`, `np.float64`), a `SeedSequence`, a `BitGenerator` and a `Generator` (used as given, `bads.rng is rng`) all behave as documented.
- `None` takes 4 draws from the global state.
- Changing the option after construction has no effect.
- The result reports an int or `None`.
- Undocumented behaviour:
  - `True`/`False` are accepted and reported as 1/0.
  - Lists and arrays are accepted and reported as `None`.
  - A negative int raises `ValueError`, where the `BADS` docstring promises `TypeError` for values that `default_rng` does not take.
  - Non-whole, NaN and inf floats and strings raise `TypeError`, as documented.

## 3. Findings

### F1. The half-bounds check refuses any mix of bounded and unbounded variables
- Location: `pybads/bads/bads.py:533-544`; MATLAB: `private/boundscheck.m` (not opened)
- Category: control flow
- Proposed classification: port discrepancy (probably MATLAB's elementwise `&` rendered as `and` between two `any`; the comparison track should confirm)
- Confidence: high
- Reached at default options: yes, for any problem with at least one finite-bounded variable and one variable in (-inf, inf); all uncertainty levels.
- History: the Python lines have been unchanged since `c7c88ab` (2022-06-02). The docstring promise was added in `ce3a0b3` (2022-11-22).
- What happens: the condition is `any(isfinite(lb)) and any(~isfinite(ub)) or any(~isfinite(lb)) and any(isfinite(ub))`. It is taken over all variables, not per variable. The error message ("Each variable needs to be unbounded or bounded") and the docstring (`bads.py:64-66`: "Set lb[i] = -inf and ub[i] = inf if the i-th coordinate is unbounded (while other coordinates may be bounded)") both mean a per-variable test, `any(isfinite(lb) != isfinite(ub))`. The rest of the setup handles mixed variables: the transform gives ∓inf for the unbounded one, as checked in Q3.
- Consequence: a documented use is impossible; `BADS` raises at construction.
- Reproduction (run): `BADS(f, [0.5,0.5], [0,-inf], [1,inf], [0.1,-1], [0.9,1])` raises `ValueError: bads:HalfBounds`.
- Test adequacy: no test constructs mixed bounds.

### F2. Scalar bounds are refused for D > 1, although the docstring says they are replicated
- Location: `pybads/bads/bads.py:63-64` (docstring), `181-184`, `380-400`; MATLAB: `private/boundscheck.m` (not opened)
- Category: indexing/shape
- Proposed classification: unsure (the docstring against the code)
- Confidence: high
- Reached at default options: no; it needs a scalar `lb`, `ub`, `plb` or `pub` with D>1.
- History: the Python code dates from the original port. The docstring sentence dates from `ce3a0b3`.
- What happens: `np.atleast_2d(scalar)` has shape (1, 1), and the shape test demands (1, D). The docstring says "If scalars, the bound is replicated in each dimension."
- Consequence: `ValueError` at construction for a documented input.
- Reproduction (run): `BADS(f, [0.5]*3, -1.0, 1.0)` raises "need to be of the same dimension D=3".
- Test adequacy: none.

### F3. The "x0 inside the plausible bounds" check compares x0 with the effective hard bounds
- Location: `pybads/bads/bads.py:506-519`; MATLAB: `private/boundscheck.m` (not opened)
- Category: control flow
- Proposed classification: port discrepancy (probable; the code contradicts its own comment and warning)
- Confidence: high
- Reached at default options: yes, whenever `x0` lies outside `[plb, pub]`, or on an effective bound; all levels.
- History: the Python lines have been unchanged since `c7c88ab`.
- What happens: the comment says "Check that all X0 are inside the plausible bounds, move bounds otherwise", and the warning says "not inside the provided plausible bounds ... Expanding". The test, however, is `x0 <= LB_eff or x0 >= UB_eff`, and `x0` was just clipped to [LB_eff, UB_eff]. So:
  - A start outside the plausible box is never detected, and the box is not expanded.
  - The expansion happens only when `x0` was clipped onto an effective bound.
- Consequence: the u-space scale depends discontinuously on `x0`. With lb=0, ub=10, plb=2, pub=8 (run):
  - `x0` = 0.005 gives plb 0.01 and u0 = -1.
  - `x0` = 0.02 gives plb 2 and u0 = -1.66.
  - `x0` = 1.0 gives plb 2 and u0 = -1.33, with no warning.

  The warning is also spurious when `x0` was clipped but `plb` was already at LB_eff (seen in every clipped case). It moves the transform and hence the whole run for starts outside the plausible box.
- Reproduction: the loop above, run.
- Test adequacy: none.

### F4. The effective bounds sit 0.1% of the whole hard range inside each bound, so x0 and the plausible bounds are moved by orders of magnitude or valid inputs are refused
- Location: `pybads/bads/bads.py:450-480`, `494-504`, `521-531`; MATLAB: `private/boundscheck.m` (not opened)
- Category: formula
- Proposed classification: unsure (a suspected defect in both if MATLAB's margin is the same)
- Confidence: medium
- Reached at default options: yes, for a variable whose plausible bound or `x0` lies within `1e-3*(ub-lb)` of a hard bound. This is always the case with the docstring's advice "where in doubt, just set plb = lb and pub = ub". It matters most for log-transformed variables and asymmetric ranges; all levels.
- History: the Python lines have been unchanged since `c7c88ab`.
- What happens: `LB_eff = lb + 1e-3*(ub-lb)` is described as "slightly inside", and the warning speaks of "numerically too close". Checked cases (run):
  - lb=1e-3, ub=1e3, plb=1e-2, pub=1e2, x0=0.05 (log-transformed): `x0` becomes 1.000999 and `plb` 1.000999. Two decades of the plausible box are removed; a user who started at the optimum is moved 20 times away. The run still found it, in 58 evaluations.
  - lb=0, ub=1000, plb=0.1, pub=10, x0=0.5: `x0` and `plb` become 1.0.
  - lb=-1000, ub=1, plb=0, pub=0.99: `UB_eff` = -0.001 < `plb`, so `ValueError: bads:StrictBounds` for valid bounds.
  - In the log case the margin in transformed space would be 1.014e-3, not 1.001.
- Consequence: the starting point, the plausible box (hence the u scale, the Sobol design and the GP priors) and sometimes the acceptance of the problem change for realistic parameter ranges.
- Reproduction: `effbounds.py` and `effrun.py`, both run, outputs as above.
- Test adequacy: none. The optimization tests use symmetric ranges ±100 with plausible box [-8, 12].

### F5. The transform's self-test uses an absolute tolerance and refuses valid bounds of large magnitude
- Location: `pybads/variable_transformer/variables_transformer.py:207-231` (`numeps = 1e-6`, line 218); MATLAB: `utils/transvars.m` (not opened)
- Category: formula
- Proposed classification: unsure
- Confidence: high on the behaviour, medium on its relevance
- Reached at default options: only with large bounds.
- History: last touched in `1d075ab` (2026-09-25); the logic dates from the original port.
- What happens: `|ginv(g(b)) - b| < 1e-6` in absolute terms fails through ordinary rounding once |b| is large. Out of 200 random draws each (run):
  - a log variable with ub ≈ 1e8 was refused 5 times, ≈ 3e8 49 times, ≈ 1e9 143 times;
  - a linear variable with |lb| ≈ 1e10 to 1e12 was refused about 10% of the time.

  `BADS` with lb = -9.53e10, ub = 9.53e10, plb = -2.06, pub = -0.74 raises "Cannot invert the transform". A relative tolerance would express the intent.
- Consequence: `ValueError` at construction for valid, if unusual, bounds.
- Reproduction: `transf2.py` and `transf3.py`, run.
- Test adequacy: `test_variable_transformer.py` uses only ±10 bounds.

### F6. The options for prior evaluations, `fun_values` and `f_vals`, cannot work
- Location: `pybads/bads/bads.py:740-791` (`fun_values`: `not np.isreal(X)` on an array at 758, `range(len())` at 787, and `self.function_logger`, which is created only at 290); `580-604` with `2840-2894` (`f_vals`); `advanced_bads_options.ini:24-25`, `275-276`; MATLAB: no counterpart opened
- Category: control flow
- Proposed classification: unsure (broken features)
- Confidence: high
- Reached at default options: no; it needs a non-empty `fun_values` or a non-`None` `f_vals`.
- History: the Python lines date from the original port (`ce3a0b3` or earlier).
- What happens:
  - A non-empty `fun_values` raises "truth value of an array ... is ambiguous" at construction.
  - `f_vals` is stored in `optim_state["cache"]`, which nothing reads. It sets `cache_active`, which switches the display to a format with 7 fields; `_display_function_log_` passes 6 at level 0.
- Consequence: `fun_values` raises `ValueError` at construction. `f_vals=[f(x0)]` raises `ValueError: Unknown format code 'f' for object of type 'str'` at the first display line in `optimize`.
- Reproduction: `bounds1.py` and `opts1.py`, run.
- Test adequacy: none.

### F7. A starting set (multi-row x0) is accepted at construction and crashes in `optimize`
- Location: `pybads/bads/bads.py:198-199`, `342-368`, `671-696` (`self.u = u0.flatten()`); MATLAB: `bads.m` setup (not opened)
- Category: indexing/shape
- Proposed classification: unsure
- Confidence: high
- Reached at default options: no; it needs an `x0` with ≥ 2 rows.
- History: from the original port.
- What happens: `_bounds_check_` supports a starting set: it estimates the plausible bounds from it and names it in its messages. `f_vals` is described as values "at X0". But `u0` of shape (N0, D) is flattened into a vector of length N0·D.
- Consequence: `optimize()` raises "operands could not be broadcast together with shapes (1,2) (1,4)". Without bounds, the setup silently estimates the plausible box from the set, then fails.
- Reproduction (run): `BADS(f, [[0.1,0.2],[0.3,-0.4]], [-1,-1], [1,1]).optimize()`.
- Test adequacy: none.

### F8. `OptimizeResult` never sets `status`, and `success` is always True
- Location: `pybads/bads/optimize_result.py:62-84`, `159-162`; MATLAB: `bads.m` exitflag, `private/bads_output.m` (not opened)
- Category: control flow
- Proposed classification: possibly intentional (`success`); a discrepancy with the sheet (`status`)
- Confidence: high
- Reached at default options: yes, every run.
- History: `status` has been commented out since `157bd09`/`ce3a0b3` (2022-11-22).
- What happens: `status` is in `_keys`, and KD-B1-8 lists it among the result's keys. `r["status"]` raises `KeyError` and `r.status` raises `AttributeError`. This contradicts KD-B1-8's list of keys.
  - `success` is True even when the run stops on `max_fun_evals` or `max_iter`, with `exit_flag` assigned but unused (`bads.py:1436-1464`).
  - The class claims to be "based on scipy.optimize.OptimizeResult", where `success` is False at an iteration limit.
  - The docstring documents neither field.
- Consequence: a user cannot tell convergence from budget exhaustion except by parsing `message`. Code written for scipy's interface breaks on `status`.
- Reproduction (run): `seed1.py`, which gives `KeyError: 'status'` and `success: True` at "reached maximum number of function evaluations".
- Test adequacy: no test reads either field.

### F9. The result deep-copies the target and the constraint function; a target holding an unpicklable resource makes `optimize()` fail after all its evaluations
- Location: `pybads/bads/optimize_result.py:98-99`, `182-186`; MATLAB: no counterpart
- Category: state/caching
- Proposed classification: suspected defect (Python-only)
- Confidence: high
- Reached at default options: yes. Every run deep-copies `fun`; the crash needs a callable that holds, for example, a lock, a file or a connection.
- History: `__setitem__`'s `deepcopy` has been there since `ce3a0b3`/`157bd09`.
- What happens: `__setitem__` applies `copy.deepcopy` to every value. A bound method or callable object is copied together with its instance. A plain function is kept by reference.
- Consequence: with `fun = model.nll`, where `model` holds a `threading.Lock`, all 20 evaluations are made and then `optimize()` raises `TypeError: cannot pickle '_thread.lock' object`, and the result is lost. Otherwise `result["fun"]` is a snapshot, not the user's object (`is` gives False). A large model is also copied in memory.
- Reproduction: `result1.py` and `result2.py`, run (traceback ends at `optimize_result.py:186`).
- Test adequacy: none.

### F10. A random x0 that violates `non_box_cons` is refused rather than redrawn, and the constraint contract is undocumented
- Location: `pybads/bads/bads.py:259-284`, `546-556`, `684-693`, docstring `79-81`; MATLAB: setup of `bads.m` (not opened)
- Category: control flow
- Proposed classification: unsure
- Confidence: medium
- Reached at default options: when `x0` is missing and `non_box_cons` is given.
- History: from the original port.
- What happens:
  - The docstring promises a random `x0` in the plausible box. With the constraint `sum(x^2) > 1` on the box [-1, 1]^2, construction raises "does not satisfy non-bound constraints" for 4 of 20 seeds (run).
  - The docstring example `lambda x: np.sum(x.^2,1)>1` is MATLAB syntax.
  - The required contract (an N×D input and an (N,) output) is not stated. A function returning a scalar fails with `IndexError`, or `AttributeError` for a Python bool, instead of the intended `ValueError`.
- Consequence: construction fails at random depending on the seed; errors are unclear for non-vectorized constraints.
- Reproduction: `nbc1.py`, run.
- Test adequacy: none.

### F11. `display` lists "notify" and "final", which act as "iter"; "full" is not documented
- Location: `basic_bads_options.ini:2-3`; `pybads/bads/bads.py:225-232`; MATLAB: `bads.m` (not opened)
- Category: defaults
- Proposed classification: port discrepancy (KD-B2-3 settles the mechanism, not the levels)
- Confidence: high
- Reached at default options: no; only with `display` set to "notify", "final" or "full".
- History: from the original port.
- What happens: "notify" and "final" fall through to INFO. With either, 15 INFO records were logged, the same as "iter" (run).
- Consequence: `display="final"` prints every iteration.
- Reproduction: `disp1.py`, run.
- Test adequacy: `test_bads_logger.py` checks only "full".

### F12. `min_fun_evals`, `min_iter` and `search_factor_min` are read by no code and are not in KD-B1-5
- Location: `advanced_bads_options.ini:101`, `281-284`; MATLAB: `bads.m` `defopts` (not opened)
- Category: defaults
- Proposed classification: unsure (they look like PyVBMC leftovers; the comparison track should say whether MATLAB BADS reads them)
- Confidence: high on "unread"
- Reached at default options: no effect in any run.
- History: from the original port.
- What happens: the descriptions promise a minimum number of evaluations or iterations ("Min number of fcn evals", "Min number of iterations"), but the run can stop earlier. `search_factor_min` suggests a floor on `optim_state["search_factor"]`, which `_update_search_stats_` (`bads.py:2758-2764`) never applies. A grep of `pybads/` outside `testing/` finds no reads.
- Consequence: user settings are silently ignored.
- Reproduction: the grep over the `.ini` names.
- Test adequacy: none.

### F13. User option values of `None` or of the wrong type are not checked, and some are silently misread
- Location: `pybads/bads/options.py:49-51`; `pybads/bads/bads.py:900-908`, `975`; MATLAB: `private/setupoptions.m:5-9` (per KD-B1-3; not opened)
- Category: defaults
- Proposed classification: unsure (KD-B1-3 leaves `None` open; strings are not covered)
- Confidence: high on the behaviour
- Reached at default options: no; it needs such a user value.
- History: from the original port.
- What happens:
  - `None` fails with `TypeError` for numeric options (10 checked, some at construction, some in `optimize`), and is read as False for boolean ones: `nonlinear_scaling=None` turns off the log transform, whose default is True.
  - `uncertainty_handling='off'` gives level 1 (`target_type` "stochastic").
  - `nonlinear_scaling='off'` keeps the log transform on.
  - `noise_final_samples=2.0` fails in the noisy run.
- Consequence: MATLAB-style values silently switch the regime of the run.
- Reproduction: `opts1.py`, run.
- Test adequacy: none.

## 4. Test adequacy notes
- `test_variable_transformer.py`:
  - Every case uses finite, symmetric, linear bounds with the plausible box equal to the hard box. No log transform, infinite bound, random round trip or large magnitude is tested.
  - `test_transform_inverse_largeN` builds `np.ones((10 ^ 6, D))`, which is 12 rows (XOR), not 10^6.
  - The `test_init_*` cases check only refusals with missing bounds.
- No test exercises `_bounds_check_`: half bounds, scalars, the effective bounds, the moves of the plausible bounds, starting sets, `non_box_cons` at the start.
- `test_random_x0_is_uniform_in_the_transformed_box` replays the implementation's exact draw (`default_rng(5).uniform(-1, 1, (1, 2))`). It pins the call sequence rather than the distribution, and its bounds stay clear of the effective-bound moves.
- `test_seed_types` and `test_seed_rejects_other_values` mirror `_init_rng_`'s conversions. Negative ints, bools and arrays are not covered.
- `test_init_conf.py::test_version` asserts nothing.
- No test reads `result["status"]` or checks what `success` means, and none uses a callable object as the target.

All scripts and their outputs are in `/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/review/B1_internal/`: `bounds1.py`, `effbounds.py`, `effrun.py`, `pbexp.py`, `transf1.py`, `transf2.py`, `transf3.py`, `vt2.py`, `opts1.py`, `dtable.py`, `disp1.py`, `seed1.py`, `result1.py`, `result2.py`, `result3.py`, `nbc1.py`.
