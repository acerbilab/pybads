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
(`_search_step_` → `add_and_update_gp` → `gp.update`). MATLAB BADS cannot
fail at the first call (`UpdateTarget` reuses the current posterior) and
guards the second (`gpupdate`); the port guarded neither until `676083d`
(`dev/plans/gp-update-guards.md`, "What MATLAB BADS does").

In `dev/experiments/population_generator_20260924/` (the same suite with
the draws through a generator, gpyreg 1.3.1), 2 of 540 runs stopped with the same error at a
third call: `ellipsoid_D10` seeds 13 (751 evaluations, from
`_search_step_`) and 26 (341 evaluations, from `_poll_step_`), in
`local_gp_fitting`, where the `except` that catches a failed
`gp.update(hyp=hyp_gp)` calls `gp.set_hyperparameters(old_hyp_gp)`, which
fails in turn. With gpyreg 1.3.3 no run of the suite crashes (540 runs,
`dev/experiments/population_gpyreg133_20260924/`, the current reference on
Windows);
those two runs follow other trajectories there, since they pass through
the low-noise regime whose predictions gpyreg 1.3.2 changed.

**Fixed at `676083d`**
([`dev/plans/gp-update-guards.md`](../plans/gp-update-guards.md)). The
three calls are guarded after MATLAB BADS, and the GP handed on always has
posteriors that match its data:
- `add_and_update_gp` passes the point through `gp.update`, and a failure
  leaves the GP without it, marked for a rebuild at the next step;
- when the recovery in `local_gp_fitting` also fails, the GP of the entry
  is restored and marked for a rebuild with a refit;
- `_get_target_from_gp_` predicts from the current GP when the posterior
  under the best iteration's hyperparameters cannot be computed. Its
  fallback to the incumbent had never run to completion: it raised
  `AttributeError` at the callers' `.item()`.

Runs without a failure are unchanged. On Linux (Python 3.11.15, NumPy
2.4.6, SciPy 1.17.1, gpyreg 1.3.3), the fingerprint is the same before and
after. The default suite × seeds 0-29 gives records identical in every
`final` field except `wall_s`, before the guards (package code of
`09996b5`) and at `676083d`. Its 484,773 guarded calls include no failure
(`dev/scripts/gp_update_failures.py`), so the benchmark does not reach the
new failure paths under gpyreg 1.3.3. Evidence for those paths:
- `test_gp_update_failures.py`, including a real Cholesky failure;
- a stress run in which 2% of the distinct guarded computations fail
  (seeds 0-9; 4,503 failed calls, 2.8% of the calls): every run finished,
  with median errors near those of the reference. The target's fallback to
  the incumbent is reached by the tests alone.

The final review of the change added two fixes at `a83bd51`:
- a NaN estimate in a noisy search whose rebuild around the search point
  fails; before, the search could move the incumbent on an estimate that
  ignored the observation;
- a poll that treats a GP whose rebuild failed as unreliable.

At `a83bd51` the failure count reproduces every run of the Linux
reference (`dev/experiments/population_linux_20260925/`) with no failed
call. The stress run finishes all 180 runs (4,577 failed calls of
166,657).

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
| `gaussian_process_train.py:1164` (at `676083d`, `add_and_update_gp`, line 1246) | the posterior update appends `sd_new` to `gp.s2`, where the initial fit stores `S**2` (`:1086`, at `676083d` line 1163) | fixed in `020d6a8` |
| `gaussian_process_train.py`, `get_grid_search_neighbors` and `local_gp_fitting` (at `676083d`, lines 1134 and 272) | the rebuild of the local GP stores `function_logger.S`, standard deviations, in `gp.s2`, a variance: the slip of the row above, at every rebuild. Only `specify_target_noise` reads `s2`; with an explicit `uncertainty_handling=True` at level 1, `S` is never filled and `gp.s2` holds NaN, which the noise function ignores. MATLAB squares the standard deviations in `likGaussHe`; the effect is in the section "The seed sweep behind the tolerances" | fixed in `020d6a8` |
| `bads.py`, `_get_target_from_gp_` (at `676083d`, line 2479) | recomputes the posterior of a copy of the GP under the best iteration's hyperparameters (`set_hyperparameters(hyp_best)`) to predict the target; MATLAB's `UpdateTarget` (`bads.m:1296-1302`) sets `gptemp.hyp = hyp` but keeps `gptemp.post`, which `gppred` passes on and `mygp` reuses (`mygp.m:123`), so MATLAB predicts from the current posterior with `hyp` in the mean and covariance functions, without refactorizing. The port's recomputation is why the call can fail, and it gives other targets at default options (`uncertain_incumbent`) | seen (MATLAB side read) |
| `bads.py`, `_get_target_from_gp_`, and MATLAB `UpdateTarget` | when the target prediction is not finite, both replace it by the incumbent's `fval` and `fsd` but compute the target from the failed variance, so the target is NaN, and the poll then treats the GP as unreliable. In MATLAB this follows a failed rebuild; in the port, a non-finite prediction from a consistent GP | seen (MATLAB side read) |
| `gaussian_process_train.py`, `local_gp_fitting` (at `676083d`, lines 520-529) | after a failed posterior update, retries with the previous hyperparameters on the new training set, which MATLAB's `gpupdate` does not do (it clears the posterior); without a refit the retry repeats the failed computation. Kept so that runs where it succeeds do not move. With a refit, `temporary_data["poll_scale"]`, `["len_scale"]` and `["effective_radius"]` come from the refit's hyperparameters while the retry puts back the previous ones, so the poll basis and the ES-ell search use a geometry the GP does not hold (before the guards too) | seen (MATLAB side read) |
| `gaussian_process_train.py`, `add_and_update_gp` (at `676083d`) | on a failed update the port leaves the point out of the GP until the next rebuild; MATLAB's `'add'` keeps it beside an empty posterior. It also recomputes every posterior in full, where MATLAB first tries a rank-1 update (skipped under `SpecifyTargetNoise`); `dev/TODO.md` | by design (`dev/plans/gp-update-guards.md`) |
| `bads.py`, `_re_evaluate_history_` (at `676083d`, line 2618) | rebuilds the GPs stored in `IterationHistory` in place, so the recorded GPs change after the fact; after a failed rebuild it records the restored GP's `fval` and `fsd`, where MATLAB would record NaN | seen |
| `bads.py`, `_poll_step_` with `stobads` (at `a83bd51`) | after a failed add in a noisy poll, `f_poll` is NaN and `_sto_success_improvement_` returns 0 (both of its comparisons are false); under `opp_stobads` (on by default) `sto_success > -1` then moves the poll to `u_poll_best`, never to the NaN point, and sets `reset_gp`: the NaN counts as uncertain, not as a failure. `stobads` is off by default | seen |
| `gaussian_process_train.py`, `local_gp_fitting`, and `bads.py`, the forced refit (at `a83bd51`) | after a failed rebuild the port restores the GP of the entry (old data, priors, hyperparameters and geometry) and refits at the next rebuild whatever `min_refit_time`; MATLAB keeps the new data and the failed rebuild's hyperparameters and `pollscale` beside `post = []` (`gpupdate.m`), and refits only when `gppredcheck` finds its NaN predictions unreliable, after `MinRefitTime` (`bads.m:1242-1244`) | by design (`dev/plans/gp-update-guards.md`, Open Question 7) |
| `bads.py`, `_search_step_` (at `a83bd51`) | after a failed rebuild, the search ranks its candidates by the LCB of the previous GP; MATLAB's `acqLCB` sums over the finite prediction samples, zero when none is, so MATLAB evaluates the first candidate. The poll treats such a GP as unreliable, as MATLAB does | seen (MATLAB side read) |
| `gaussian_process_train.py`, `init_and_train_gp` (at `a83bd51`, lines 161-209) | retries a failing initial fit without bound (from the fifth attempt, from random samples of the priors), so a fit that keeps failing loops forever; initialization only | seen |
| `gaussian_process_train.py`, `_robust_gp_fit_`, and the forced refit (at `a83bd51`) | the refit forced after a failed rebuild sends a run that has met a failure into `_robust_gp_fit_`, whose fifth consecutive failed fit raises `ValueError` (section "Failed fits"); the stress run injects no failure into fits, so this is untested | not looked at |
| `function_logger/constraints_check.py`, `contraints_check` | removes no previously evaluated point (its `np.unique` keeps the first occurrences, which fall among the candidates), where MATLAB's `uCheck.m` removes them with `setdiff`; a repeat becomes a duplicate GP training input, a plausible cause of the failed Cholesky factorizations. One exact repeat was evaluated in `ellipsoid_D10` seed 7 on Linux at `8fc1dff` (section "Found while fixing the tests"; `dev/TODO.md`, "Previously evaluated points evaluated again") | seen (MATLAB side read) |
| `init_functions/init_sobol.py`, `init_sobol` | the seed of the initial Sobol design comes from the integer parts of `u0`, so every start point inside the plausible box gives the same design for a given `D`, and the cast is undefined at -1 and below (section "Found while fixing the tests") | seen (MATLAB side read; the saturation of MATLAB's `uint64` product not checked) |
| `gaussian_process_train.py:4` (at `a83bd51`) | imported its `logger` from `asyncio.log`, so the GP fit warnings went to the `asyncio` logger (section "Found while fixing the tests") | fixed at `8fc1dff` (the `BADS` logger) |
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

Each defect below is fixed in pull request #63, in the commit its entry
names. A fix to a test moves no result, so the test suite was the check:
with gpyreg 1.3.3, 109 tests passed at `870fa35` and 112 after the fixes
(the three poll tests added to the collection), none of them needing a
rerun; on the full CI matrix (Ubuntu, Windows, macOS × Python 3.10–3.12),
112 passed in every job, without reruns.

- `pybads/testing/bads/poll/test_poll_mads.py` named its functions
  `*_test`, so pytest collected none of them. Fixed in `125efac`: renamed
  `test_*`, they check the poll set beyond its shape (the second half
  negates the first, and undoing the division by `poll_scale` gives an
  integer basis of determinant `+-n_max**D`), and a third test reaches
  `n_max > 1`, which neither of the two original cases did. A run at
  default options never does: `search_size_integer` stays at most
  `2 * mesh_size_integer - 10` and at most 0 (`search_grid_multiplier` 2,
  `search_grid_number` 10), so with `poll_mesh_multiplier` 2 the search
  mesh is at least 32 times finer than the poll mesh and `n_max` is 1, as
  in MATLAB BADS.
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
noise seed `s + 1000` for `s` from 0 to 99, from a copy of its
configuration; the tests run at `s = 0`. Environment: Windows 11, Python
3.12.6, NumPy 2.5.3, SciPy 1.18.1, gpyreg 1.3.3 (a clone at the tag,
`98ab5a4`), package code of `870fa35`. No run crashed. The retry loop of
the initial GP fit (`init_and_train_gp`) logged 49 failed fits, and every
run's initial fit then succeeded: 47 in 32 runs of `test_small_noisy_func`
(up to four in one run) and 2 in 2 runs of
`test_univariate_input_and_opt`. `dev/scripts/tolerance_sweep.py`, which
runs the test functions themselves with their tolerances disabled,
reproduces the error and the evaluations of every run of the sweep at `s`
from 0 to 4.

On another platform a seeded run can follow another trajectory, as another
seed would, so a tolerance has to hold beyond the seed the test runs at.
Each tolerance is ten times the largest error of the sweep, rounded up to
1, 2 or 5 times a power of ten, or the tolerance of `runtest.m` if that is
lower (`test_noisy_sphere_opt` and `test_he_noisy_sphere_opt`). The row of
`test_he_noisy_sphere_opt` comes from a second sweep of that test, after
the fix of its noise variances (below).

| Test | Evaluations | Median error | Largest error | Tolerance | Before the fix | `runtest.m` |
|---|---|---|---|---|---|---|
| `test_ellipsoid_opt` | 67–93 | 8.7e-6 | 9.8e-5 | 1e-3 | 1 | 0.1 |
| `test_sphere_opt` | 65–100 | 2.0e-5 | 1.9e-4 | 2e-3 | 1 | 0.1 |
| `test_noisy_sphere_opt` | 100 | 5.7e-3 | 0.10 | 1 | 1 | 1 |
| `test_he_noisy_sphere_opt` | 185–200 | 0.14 | 0.56 | 1 | 1 | 1 |
| `test_small_noisy_func` | 149–223 | 5.9e-6 | 2.5e-4 | 1e-2 | 0.1 | |
| `test_1D_opt_*` (three tests) | 30 | 5.8e-9 | 2.1e-7 | 5e-6 | 0.1 | |
| `test_univariate_input_and_opt` | 87–141 | 3.3e-5 | 1.4e-3 | 2e-2 | none | |
| `test_high_dim_opt` | 200 | 2.9e-2 | 5.5e-2 | 1 | none | |

The error is `|fval - f_min|` on a target without noise, and the noiseless
value at the returned point minus the minimum on a noisy one. The three 1D
tests pass the same problem in different input shapes; the sweep ran
`test_1D_opt_scalar`, and at `s` from 0 to 4 the other two give the same
runs. The 60-D ellipsoid of `test_high_dim_opt` starts at 17.3, and the
run ends on its budget of 200 evaluations.

On the arm64 runners of the macOS CI jobs, `test_small_noisy_func` starts
from another initial design than on x86, the same for every seed (the cast
of `init_sobol`, below). Emulated on x86 by clipping `u0` at 0 before
`init_sobol`, on the assumption that the arm64 conversion turns a negative
value into 0, its errors over the same seeds reach 6.5e-4 (median 1.1e-5,
145–207 evaluations), and its tolerance, 1e-2, follows the rule over the
largest error of both designs.

The target of `test_he_noisy_sphere_opt` returns the standard deviation of
its noise (`specify_target_noise`), 2 at the minimum. Before `020d6a8`, the
rebuild of the local GP and each added point stored these standard
deviations in `gp.s2`, which gpyreg reads as variances (the initial fit
squared them). MATLAB BADS keeps them in `gpstruct.s` and squares them in
`likGaussHe` (`sn2 = exp(2*hyp) + s.^2`). Before the fix, the errors over
seeds 0–99 exceeded the tolerance of `runtest.m`, 1, in 7 seeds, with a
median of 0.33 and a largest error of 2.3, and the test had a tolerance of
5; run again at `10d74a7`, the sweep reproduces the first one run by run.
With the squares (`020d6a8`, the environment above), no error reaches 1:
the median is 0.14, the largest 0.56, and the error is smaller in 70 of
the 100 paired seeds (Wilcoxon signed-rank test, p = 4e-8). MATLAB's own
errors on this problem were not measured.

In the benchmark, `020d6a8` changes only the two configurations with
target noise, whose standard deviation is 1 at the minimum, and the
comparison with the previous Windows reference flags neither
([`population_targetnoise_20260925`](../experiments/population_targetnoise_20260925/README.md)).
Over 30 seeds, the median error of `sphere_D3_hetero` falls from 0.20 to
0.10 (signed-rank p = 0.008 before the Holm correction), and that of
`ellipsoid_D3_hetero` rises from 0.26 to 0.37 (p = 0.10), with a largest
error of 4.1 where it was 1.3.

### Found while fixing the tests

Three candidate defects, seen in the code, which are also rows of the
candidate table:

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
  new row, a duplicate training input of the GP. At low noise a duplicate
  input leaves the training covariance of the GP nearly singular. On Linux
  at `8fc1dff`, one exact repeat was evaluated in `ellipsoid_D10` seed 7,
  one of the crashing seeds recorded above; `dev/TODO.md`, "Previously
  evaluated points evaluated again", holds the rest of the check. Fixing
  it moves results.
- `init_sobol` (`init_functions/init_sobol.py`) derives the seed of the
  initial Sobol design from `u0[:11].astype(np.uint64)`, the integer parts
  of the first 11 coordinates, and not from `random_seed`: the seed is the
  product of the character codes of those integers, printed. Every `u0`
  inside `(-1, 1)^D`, that is, every start point inside the plausible box,
  gives the same seed, and so the same initial design for a given `D`
  (checked for `D = 3`). The cast is undefined for a coordinate of -1 or
  below: on x86, `-1.5` gives `2**64 - 1`, while on the arm64 runners of
  the macOS CI jobs `test_small_noisy_func` (whose `x0 = -3` maps to
  `u0 = -1.5`) emits `RuntimeWarning: invalid value encountered in cast`,
  and its design differs (sweep above). The product is taken in NumPy's
  default integer, 64-bit from NumPy 2 but 32-bit with NumPy 1.x on
  Windows (checked with NumPy 1.23.5). For a start point inside the
  plausible box, the 32-bit product overflows from `D = 4` and is 0 from
  `D = 5`, which gives the seed 1; the 64-bit product is 0 from `D = 8`.
  The two give different designs for `D` from 4 to 7, and the same at the
  dimensions of the tests (1, 3 and 60). PyBADS requires NumPy 2
  from `c044fea`. MATLAB's `init/initSobol.m` takes the seed from
  `prod(uint64(num2str(u0(1:min(10,end)))))`, the character codes of the
  printed values of the first 10 coordinates. If MATLAB's `prod` keeps the
  `uint64` class of its argument and saturates, as its integer arithmetic
  does by default, that seed is also the same for most start points, and
  the port differs in mechanism more than in effect; this is not checked in
  MATLAB.
- `gaussian_process_train.py` imported its `logger` from `asyncio.log`
  (line 4), the logger named `asyncio`, not the `BADS` logger whose level
  the `display` option sets. Its warnings (a failed initial fit in
  `init_and_train_gp`, a failed hyperparameter optimization, a failed slice
  sampler) were therefore printed whatever `display` said: the sweep above,
  run with `display="off"`, printed the 49 failed initial fits. Fixed in
  `8fc1dff` (#64): the module logs to the `BADS` logger.

### Reruns and the minimum versions

CI runs each test once from `c044fea`. Under `--reruns=5`, the six
full-matrix runs before it (54 jobs, on 2026-09-25) had rerun no test.

From `c044fea`, `pyproject.toml` requires NumPy 2.0.0, SciPy 1.13.0 and
matplotlib 3.9.0 or later; no CI job installs these minimums. The suite at
`10d74a7` (157 tests, installed from the package, gpyreg 1.3.3) passed on
Windows at these minimums with Python 3.12, and with Python 3.11, NumPy
1.23.5, SciPy 1.9.3 and matplotlib 3.6.3. The previous minimums (NumPy
1.22.1, SciPy 1.7.3 and matplotlib 3.5.1) were not run.
