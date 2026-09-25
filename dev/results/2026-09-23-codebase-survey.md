# Codebase survey: failures and candidate defects

Findings from a read-through of the code at `273a5b7` and runs of the test
suite. The systematic bug hunt and the verification against MATLAB BADS are
deferred (`dev/TODO.md`); this record is their starting point.

Environment of the runs: Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy
1.18.1, gpyreg 1.3.1 installed editable from its checkout at `1dbbfc5`.

## Observed failures

### Crash with user-specified noise

`test_he_noisy_sphere_opt` (`specify_target_noise=True`, uncertainty level
2) failed in 6 of 10 standalone runs with `--reruns` disabled:

```console
python -m pytest pybads/testing/bads/test_bads_optimization.py::test_he_noisy_sphere_opt -p no:rerunfailures
```

In the full suite (`--reruns=5 -x -vv`: 89 passed in 42 s) it failed five
times and passed on the sixth attempt. The reruns hide the failure in CI,
where each of the nine matrix cells runs the test up to six times.

Every failure is the same crash, not an assertion:

```text
bads.py:1288  self._poll_step_(gp)
bads.py:1985  self._is_gp_refit_time_(...)
bads.py:2311  self.gp_stats.get("fval")[...].flatten().astype("float")
ValueError: setting an array element with a sequence.
```

Mechanism: at level 2, when a point already in the log is evaluated again,
`FunctionLogger._record` (`function_logger/function_logger.py`, the
duplicate branch) merges the observations by precision weighting and returns
`self.Y[idx]`, a shape-`(1,)` array, where every other path returns a
scalar. `_save_gp_stats_` stores that value in the `gp_stats["fval"]` object
array, and `_is_gp_refit_time_` then fails to convert the array to float:
NumPy 2.5.3 raises on an object array holding a size-1 array. Whether an
older NumPy accepted the conversion was not checked. The run crashes only
when a search or poll point lands on an evaluated point, which is why the
failure is intermittent.

Fixed in `4566acb`: the duplicate branch returns the merged value as a
scalar. On the problem of the test with its noise from a per-seed generator,
10 of 30 seeded runs crashed before the fix; after it all 30 finish, and
the 20 that did not crash give bit-identical results.

### Bitwise `~` on a bool

`gaussian_process_train.py:275`,
`if prior_mean is not None and ~options["gp_fixed_mean"]:`, emitted all
2130 warnings of the suite run: Python 3.12 deprecates `~` on a `bool` and
names 3.16 for its removal. `~False` is `-1` and `~True` is `-2`, both
truthy, so the condition does not depend on `gp_fixed_mean`.

Fixed in `06badd3` (`not`). The condition decides the width of the
empirical prior of the GP mean, which `local_gp_fitting` never applies
(candidate table), so neither the slip nor the fix changes a run.

### Known-noise path (`fit_lik=False`)

With `fit_lik=False`, `gaussian_process_train.py` (`_gp_hyp`) sets the noise
prior `("delta", noise_mu)`, a prior type gpyreg does not implement: every
such run stops at its first GP setup with `ValueError: Unknown hyperprior
type delta` (checked on 2026-09-24 with gpyreg 1.3.1, from `set_priors`, and
with the 1.3.2 branch `w6-leftovers` at `755a4b3`, from
`_write_prior_block`; on 2026-09-25 with the release 1.3.3, from
`_write_prior_block`). gpyreg's `set_priors` refuses an unknown prior type
in every release from 1.0.2 on. `fit_lik` is an advanced option, `True` by
default.

### Failed fits in `_robust_gp_fit_`

`_robust_gp_fit_` retries a fit that raises `LinAlgError` up to ten times,
raising the lower bound of the noise hyperparameter by 1, 2, 3, ... from
`log(tol_fun) - 1`, about -7.91 at the default `tol_fun`. After the fifth
consecutive failure the lower bound passes the upper bound, 5, and the
inverted pair raises `ValueError` (read from the code, not run: with gpyreg
1.3.2 in `set_bounds` at the fifth failure, with 1.3.1 in the next `fit`);
only `LinAlgError` is caught, so the run stops, and the unbound result of
ten failures cannot be reached. At default options this was not reached: over the 15
configurations of the benchmark suite (`dev/scripts/benchmark_targets.py`,
suite `default`), seeds 0-9, about 1,500 calls under each gpyreg version,
79% had no failure, and the most consecutive failures in one call was 3;
every call recovered and every run finished. With gpyreg 1.3.3 and the
draws through `bads.rng` (`2059506`), over all 18 configurations, seeds
0-9: 3,385 calls, 85% without failure, at most 3 consecutive failures,
every run finished (`dev/results/2026-09-25-gpyreg-1.3.3.md`).

### Crashes on unguarded GP updates

In the first benchmark reference,
`dev/experiments/population_baseline_20260924/` (18 configurations × 30
seeds at 500 D, gpyreg 1.3.1, draws through NumPy's global stream), 2 of
540 runs
stopped with `LinAlgError: Singular matrix for L Cholesky decomposition`
from gpyreg's training Cholesky factorization: `ellipsoid_D3` seed 20 at
150 evaluations (`_poll_step_` → `_get_target_from_gp_` →
`gp.set_hyperparameters`) and `ellipsoid_D10` seed 7 at 951 evaluations
(`_search_step_` → `add_and_update_gp` → `gp.update`). MATLAB BADS guards
both calls, `gppred` and `gpupdate` catching the failure; the port does
not (`dev/TODO.md`, where the MATLAB counterparts are named).

In `dev/experiments/population_generator_20260924/` (the same suite with
the draws through a generator, gpyreg 1.3.1), 2 of 540 runs stopped with the same error at a
third call: `ellipsoid_D10` seeds 13 (751 evaluations, from
`_search_step_`) and 26 (341 evaluations, from `_poll_step_`), in
`local_gp_fitting`, where the `except` that catches a failed
`gp.update(hyp=hyp_gp)` calls `gp.set_hyperparameters(old_hyp_gp)`, which
fails in turn. With gpyreg 1.3.3 no run of the suite crashes (540 runs,
`dev/experiments/population_gpyreg133_20260924/`, the current reference);
those two runs follow other trajectories there, since they pass through
the low-noise regime whose predictions gpyreg 1.3.2 changed, and the three
calls remain unguarded.

## Candidate defects (not verified)

Found by reading the code, without a check of reach or effect and without
the MATLAB comparison. "Seen" means the code at the location reads as
described; the rest are reports of the read not yet looked at.

| Location | Candidate | Status |
|---|---|---|
| `bads.py:626-627` | `optim_state["pub"]` receives the transformed `plb` and `optim_state["plb"]` the transformed `pub`; `gaussian_process_train.py:432-438` reads them for unbounded variables | seen |
| `search/search_hedge.py:141` | Hedge reward written `exp(-0.5*g**2/sqrt(2*pi))`; the MATLAB `acqPortfolio` form is reported as `exp(-0.5*g**2)/sqrt(2*pi)` | seen (MATLAB side not checked) |
| `gaussian_process_train.py:413` | `len_scale += len_scale + ...` doubles the accumulator at each sample | seen |
| `bads.py:1657-1661` (at `06badd3`, lines 1693-1697) | the condition for adding the search point to the GP, `u_search.size > 0 & self.search_es_hedge.count < self.options["search_n_try"]`: `&` binds before the comparisons, so Python reads `u_search.size > (0 & count) < search_n_try`, that is `u_search.size > 0 and 0 < search_n_try`, and the count limit is dropped: every non-empty search point is added. MATLAB (`bads.m:633`) adds it only when `~isempty(usearch) && optimState.searchcount < options.SearchNtry`, counting with `optimState.searchcount`, where the port names the hedge's own `count` | seen (MATLAB side read) |
| `bads.py:1405` | assigns `self.best_u`, a name used nowhere else (the incumbent is `u_best`) | seen |
| `bads.py:500-507` | the half-bounds check tests all variables at once, rejecting any mix of bounded and unbounded variables, while the docstring allows per-variable infinite bounds | seen |
| `bads.py:2183` | appends the bound method `self.u_best.copy` instead of a copy | seen |
| `gaussian_process_train.py:1164` | the posterior update appends `sd_new` to `gp.s2`, where the initial fit stores `S**2` (`:1086`) | not looked at |
| `bads.py:722-770` | a non-empty `fun_values` option is reported to crash | not looked at |
| `optimize_result.py` | `success` is reported always `True`; `exit_flag`, `min_iter` and `min_fun_evals` are reported unread | not looked at |
| `examples/scripts/pybads_example_2_nonbox_constraints.py` | set `options["rng_seed"]`, not a valid option name, in options never passed to `BADS`; the notebook never had these lines, and the script generated from it (tooling plan, Phase 4) no longer has them | resolved |
| `search/es_search.py:239-253` | `ESSearchCMA` calls `ucov` with a signature that does not match its definition (`:294`); no option reaches the class | not looked at |
| `bads.py`, `_init_optim_state_` (at `4566acb`, lines 809-826) | with `specify_target_noise=True`, an `uncertainty_handling` of `None` is set to `False` and then raises `ValueError`, whose message says to leave `uncertainty_handling` empty | seen |
| `gaussian_process_train.py:979-983` (at `1cfe371`) | the fraction of the budget used divides by `min(max_fun_evals, n_train_max) - eff_starting_points`, zero when `max_fun_evals` equals the initial design, and the cubic next to it mixes `x_` and `x` | seen |
| `bads.py:880-893` | the `gp_mean_fun` check accepts twelve names, of which only `zero`, `const` and `negquad` are reported to work | not looked at |
| `gaussian_process_train.py`, `local_gp_fitting` (at `06badd3`, lines 296-346) | computes an empirical prior for the constant GP mean (`prior_mean`: centre at a percentile of `gp.y`, width `y_range ** (1/4)`) and never writes it into the priors it sets; MATLAB (`gpdef/gpdefBads.m`, "Update empirical prior for GP mean") updates the mean prior at each training, with variance `yrange.^2/4`, and with `gpFixedMean` also sets the mean hyperparameter to `ymean` under a delta prior, which the port leaves as a `TODO`. Applying the prior would change runs at default options | seen (MATLAB side read) |
| `gaussian_process_train.py`, `_robust_gp_fit_` (at `2f3d949`, lines 581-617) | after `tmp_gp.fit(X, Y, s2, ...)` raises, the retry reads `tmp_gp`: the bounds it nudges come from `tmp_gp.get_bounds()`, and with `use_slice_sampler=True` it samples hyperparameters on the data `tmp_gp` holds. What a failed fit leaves there depends on gpyreg: up to 1.3.2 the data of the failed fit and the bounds it filled in, from 1.3.3 the GP as it was before the call. At default options every bound the retry reads is set by `_gp_hyp`, so the retry does not depend on it; with the slice sampler, or the `negquad` mean (whose unset bounds are NaN until a fit fills them), it does. Passing the retry's `X`, `Y` and `s2` explicitly would remove the dependence | seen |
| `bads.py`, `_search_step_` and `_poll_step_` (at `894d205`, lines 1662-1669 and 2062-2069) | the fallback when the acquisition fails draws `index_acq` from `rng.integers(0, len(...) + 1)`, which can return one past the last index (the range of the `np.random.randint` it replaced); the branch is not reached, since `np.argmin(z)` is never `None`, empty or non-finite | fixed in `06badd3` |
| `search/search_hedge.py:72-74` (at `894d205`) | `np.argwhere(rand_uni < np.cumsum(self.prob))[0]` raises `IndexError` when nothing matches, before the `len(self.chosen_hedge) == 0` fallback that it guards can run | fixed in `06badd3` |
| `bads.py:39` (at `894d205`), the `BADS` class docstring | a `:math:` role holding `\mathtt` in a docstring that is not a raw string: Python 3.12 emits `SyntaxWarning: invalid escape sequence` for it when it compiles the module | fixed in `06badd3` |

## Tests that checked less than they appeared to

Each defect below is fixed in the commit its entry names. A fix to a test
moves no result, so the test suite was the check: with gpyreg 1.3.3, 109
tests passed at `870fa35` and 112 after the fixes (the three poll tests
added to the collection), none of them needing a rerun; on the full CI
matrix (Ubuntu, Windows, macOS × Python 3.10–3.12), 112 passed in every
job, without reruns.

- `pybads/testing/bads/poll/test_poll_mads.py` named its functions
  `*_test`, so pytest collected none of them. Fixed in `125efac`: renamed
  `test_*`, they check the poll set beyond its shape (the second half
  negates the first, and undoing the division by `poll_scale` gives an
  integer basis of determinant `+-n_max**D`), and a third test reaches
  `n_max > 1`, which neither of the two original cases did. A run at
  default options never does: with `search_size_locked`, the default,
  `search_size_integer` is at most `2 * mesh_size_integer - 10` and at
  most 0, so the search mesh is at least 32 times finer than the poll mesh
  and `n_max` is 1, as in MATLAB BADS.
- `test_high_dim_opt` ran with `assert_flag=False` and asserted nothing,
  and neither did `test_univariate_input_and_opt`. Fixed in `9a99bae`: both
  assert an error bound (table below).
- The optimization tests built on `get_test_opt_conf` passed when the
  error was below 1: `np.any(err < [0.1, 0.1, 1, 1])` compared the error
  with the four tolerances of `runtest.m` at once. Fixed in `9a99bae`:
  each test has one tolerance (table below).
- `test_sphere_opt` marked as infeasible the points with
  `x1 + x2 >= sqrt(2)`, the reverse of the MATLAB test (`runtest.m`,
  infeasible where `x1 + x2 < sqrt(2)`), and started from the origin, so
  the unconstrained minimum 0 was feasible; it expected 1 and passed
  because an error just below 1 met its tolerance. Fixed in `9a99bae`: the
  test takes the constraint and the start point `(4, 4, 4)` of
  `runtest.m`, and every run of the sweep below ends within 2e-4 of the
  constrained minimum.
- Most optimization tests were unseeded, and the noisy targets drew their
  noise from NumPy's global stream (`test_small_noisy_func` after reseeding
  it). Fixed in `9a99bae`: each run sets `random_seed`, and a noisy target
  draws its noise from a generator of its own. The three tests of
  `search/test_search.py` that trained the GP of an unseeded `BADS` are
  seeded in `4fab44d`.
- `pybads/testing/run_tests.py` imported `testing.*` paths that no longer
  exist, and no test read `pybads/testing/bads/*.dat`, six files present
  since the first port (`c7c88ab`) and read in no commit. Both removed in
  `5d5c111`.

### The seed sweep behind the tolerances

Each test of `test_bads_optimization.py` ran with `random_seed = s` and its
noise seed `s + 1000` for `s` from 0 to 99; the tests run at `s = 0`.
Environment: Windows 11, Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1, gpyreg
1.3.3 (a clone at the tag, `98ab5a4`), package code of `870fa35`. No run
crashed. The retry loop of the initial GP fit (`init_and_train_gp`)
logged 49 failed fits, each followed by a successful one: 47 in 32 runs of
`test_small_noisy_func` and 2 in 2 runs of
`test_univariate_input_and_opt`. Called at `s` from 0 to 4 by setting the
module's `SEED` and `NOISE_SEED`, the test functions reproduce the runs of
the sweep bit for bit.

On another platform a seeded run can follow another trajectory, as another
seed would, so a tolerance has to hold beyond the seed the test runs at.
Each tolerance is about ten times the largest error of the sweep, rounded
up to 1, 2 or 5 times a power of ten, and no looser than that of
`runtest.m`, with one exception: the errors of `test_he_noisy_sphere_opt`
exceed the tolerance of `runtest.m`, 1, in 7 of the 100 seeds, the largest
being 2.3, and its tolerance is 5. Whether MATLAB BADS exceeds 1 as often
on this problem, at the same 200 evaluations, is not checked.

| Test | Evaluations | Median error | Largest error | Tolerance | Before | `runtest.m` |
|---|---|---|---|---|---|---|
| `test_ellipsoid_opt` | 67–93 | 8.7e-6 | 9.8e-5 | 1e-3 | 1 | 0.1 |
| `test_sphere_opt` | 65–100 | 2.0e-5 | 1.9e-4 | 2e-3 | 1 | 0.1 |
| `test_noisy_sphere_opt` | 100 | 5.7e-3 | 0.10 | 1 | 1 | 1 |
| `test_he_noisy_sphere_opt` | 192–200 | 0.33 | 2.3 | 5 | 1 | 1 |
| `test_small_noisy_func` | 149–223 | 5.9e-6 | 2.5e-4 | 5e-3 | 0.1 | |
| `test_1D_opt_*` (three tests) | 30 | 5.8e-9 | 2.1e-7 | 5e-6 | 0.1 | |
| `test_univariate_input_and_opt` | 87–141 | 3.3e-5 | 1.4e-3 | 2e-2 | none | |
| `test_high_dim_opt` | 200 | 2.9e-2 | 5.5e-2 | 1 | none | |

The error is `|fval - f_min|` on a target without noise, and the noiseless
value at the returned point minus the minimum on a noisy one. The three 1D
tests pass the same problem in different input shapes; the sweep ran
`test_1D_opt_scalar`, and at `s` from 0 to 4 the other two give the same
runs. The 60-D ellipsoid of `test_high_dim_opt` starts at 17.3, and the
run ends on its budget of 200 evaluations.

### Found while fixing the tests

Two candidate defects, seen in the code (reach and effect not measured):

- `test_incumbent_constraint_check` (`search/test_search.py`) evaluates
  every row of `U` and then asserts that `contraints_check`
  (`function_logger/constraints_check.py`) drops only the duplicate row it
  appends. That holds because `contraints_check` removes no previously
  evaluated point: its step "Remove previously evaluated vectors" keeps the
  first occurrences of `np.unique` over the candidates stacked above the
  evaluated points, and a first occurrence always falls among the
  candidates. MATLAB's `utils/uCheck.m` removes them
  (`setdiff(u1, u2, 'rows')`), which would leave `U_new` empty in the test.
  `contraints_check` filters the candidates of the initial design, the
  search and the poll, so a run can evaluate a point again; without a
  noise estimate from the target, `FunctionLogger` records the repeat as a
  new row, a duplicate training input of the GP. Fixing it moves results.
- `init_sobol` (`init_functions/init_sobol.py`) derives the seed of the
  initial Sobol design from `u0[:11].astype(np.uint64)`, which truncates
  each coordinate and is undefined for one of -1 or below: on macOS,
  `test_small_noisy_func` (whose `x0 = -3` maps to `u0 = -1.5`) emits
  `RuntimeWarning: invalid value encountered in cast` in CI, so its seed
  there can differ from that on Windows and Linux. MATLAB's
  `init/initSobol.m` takes the seed from the characters of
  `num2str(u0(1:min(10,end)))`, the printed values themselves. By
  truncation, every `u0` inside `(-1, 1)^D`, that is, every start point
  inside the plausible box, gives the port the same seed, and so the same
  initial design for a given `D`, whatever the `random_seed` (checked for
  `D = 3`).
