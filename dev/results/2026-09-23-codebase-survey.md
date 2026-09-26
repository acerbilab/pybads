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
`dev/experiments/population_gpyreg133_20260924/`, then the reference on
Windows);
those two runs follow other trajectories there, since they pass through
the low-noise regime whose predictions gpyreg 1.3.2 changed.

**Training inputs at the failing calls.** `contraints_check` lets a run
evaluate a point again (candidate table), and without a target noise SD
the repeat becomes a second, identical training input of the GP, which can
make the training covariance singular. Each of the four crashing runs was
rerun on Windows at its population's commit (`2226883` or `c85cddb`, both
reachable from `refs/pull/59/head` on GitHub), with a clone of gpyreg at
v1.3.1, through `population.py run` from a worktree at that commit, with
`gpyreg.GP.update`, which `set_hyperparameters` calls, wrapped to observe
each call from PyBADS. Each run crashes again, at the same evaluation and
through the same lines of PyBADS and gpyreg. At the failing call, the
wrapper counts the exact duplicate rows of the training inputs and repeats
the call's computation on copies of the GP as it was before the call:
unchanged, which fails again in all four, and with rows removed. Distances
between inputs are scaled by the length scales of the failing call; below
1e-8, the rational-quadratic kernel of two inputs equals the output
variance to double precision. The wrapper and the records are kept on the
machine that ran them (`dev/scripts/runs/LOCAL.md`).

| Run | Failing call | Inputs | Exact duplicate pairs | Distinct pairs closer than 1e-8 | Updates with duplicates before |
|---|---|---|---|---|---|
| `ellipsoid_D3` seed 20 (baseline) | `_get_target_from_gp_`, `set_hyperparameters(hyp_best)` | 80 | 3 | 78 of 3,157 | 122 of 387, from the 266th |
| `ellipsoid_D10` seed 7 (baseline) | `add_and_update_gp`, `gp.update` | 151 | 0 | 11,325 of 11,325 | 593 of 2,263, from the 1,411th |
| `ellipsoid_D10` seed 13 (generator) | `local_gp_fitting`, `gp.update(hyp=hyp_gp)`, then `set_hyperparameters(old_hyp_gp)` with the same hyperparameters | 150 | 0 | 5,561 of 11,175 | 460 of 1,756, from the 1,132nd |
| `ellipsoid_D10` seed 26 (generator) | the same, from the poll | 50 | 0 | 79 of 1,225 | none of 794 |

Duplicate inputs explain none of the four crashes. In `ellipsoid_D3` seed
20, the computation succeeds without the three duplicate pairs, but it
also succeeds when three other rows are removed instead: one from the
cluster of inputs closer than 1e-8 around each duplicate, the duplicates
kept (20 of 20 draws), or three at random among the rows that are not
duplicates (16 of 20 draws). The factorization fails by a margin that
most sets of three rows decide, and duplicates were in the training sets of
the 122 updates before it, which succeeded. The other three calls hold no
duplicate. All four fail under the same kind of hyperparameters:
- the log output scale within 1.3e-5 of its upper bound,
  `log(1e6 * tol_fun / tol_mesh)` = 20.08 (`tol_mesh` is 2^-19 in
  `optim_state`), for an output variance 2e22 to 2e24 times the noise
  variance;
- long length scales, whose logarithms reach 19.3 in 3-D and 27.5, 59.7
  and 34.7 in 10-D, so that many distinct inputs coincide numerically, all
  11,325 pairs of the 151 inputs of `ellipsoid_D10` seed 7;
- in 3-D, the log shape of the rational-quadratic kernel at its lower
  bound, -5.

Most log length scales lie above 4.38 (2 of 3 in 3-D, 8 to 10 of 10 in
10-D), the upper bound that MATLAB BADS would give them, `log(covrange)`
with its `covrange` 80 on these problems. The port's upper bound is its
`cov_range`, the same 80 without the logarithm (candidate table), which
none of the four reaches.

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
| `bads.py:626-627` | `optim_state["pub"]` receives the transformed `plb` and `optim_state["plb"]` the transformed `pub`; `gaussian_process_train.py:432-438` reads them for unbounded variables | fixed in `6e22d32` (the port review's wave 1, W1-1) |
| `search/search_hedge.py:141` | Hedge reward written `exp(-0.5*g**2/sqrt(2*pi))`; the MATLAB `acqPortfolio` form is reported as `exp(-0.5*g**2)/sqrt(2*pi)` | seen (MATLAB side not checked) |
| `gaussian_process_train.py:413` | `len_scale += len_scale + ...` doubles the accumulator at each sample | fixed in `3cae0e1` (the port review's wave 1, W1-18) |
| `bads.py:1657-1661` (at `06badd3`, lines 1693-1697) | the condition for adding the search point to the GP, `u_search.size > 0 & self.search_es_hedge.count < self.options["search_n_try"]`: `&` binds before the comparisons, so Python reads `u_search.size > (0 & count) < search_n_try`, that is `u_search.size > 0 and 0 < search_n_try`, and the count limit is dropped: every non-empty search point is added. MATLAB (`bads.m:633`) adds it only when `~isempty(usearch) && optimState.searchcount < options.SearchNtry`, counting with `optimState.searchcount`, where the port names the hedge's own `count` | fixed in `91d3c33` (the port review's wave 1, W1-9) |
| `bads.py:1405` | assigns `self.best_u`, a name used nowhere else (the incumbent is `u_best`) | seen |
| `bads.py:500-507` | the half-bounds check tests all variables at once, rejecting any mix of bounded and unbounded variables, while the docstring allows per-variable infinite bounds | seen |
| `bads.py:2183` | appends the bound method `self.u_best.copy` instead of a copy | seen |
| `gaussian_process_train.py:1164` (at `676083d`, `add_and_update_gp`, line 1246) | the posterior update appends `sd_new` to `gp.s2`, where the initial fit stores `S**2` (`:1086`, at `676083d` line 1163) | fixed in `020d6a8` |
| `gaussian_process_train.py`, `get_grid_search_neighbors` and `local_gp_fitting` (at `676083d`, lines 1134 and 272) | the rebuild of the local GP stores `function_logger.S`, standard deviations, in `gp.s2`, a variance: the slip of the row above, at every rebuild. Only `specify_target_noise` reads `s2`; with an explicit `uncertainty_handling=True` at level 1, `S` is never filled and `gp.s2` holds NaN, which the noise function ignores. MATLAB squares the standard deviations in `likGaussHe`; the effect is in the section "The seed sweep behind the tolerances" | fixed in `020d6a8` |
| `bads.py`, `_get_target_from_gp_` (at `676083d`, line 2479) | recomputes the posterior of a copy of the GP under the best iteration's hyperparameters (`set_hyperparameters(hyp_best)`) to predict the target; MATLAB's `UpdateTarget` (`bads.m:1296-1302`) sets `gptemp.hyp = hyp` but keeps `gptemp.post`, which `gppred` passes on and `mygp` reuses (`mygp.m:123`), so MATLAB predicts from the current posterior with `hyp` in the mean and covariance functions, without refactorizing. The port's recomputation is why the call can fail, and it gives other targets at default options (`uncertain_incumbent`) | seen (MATLAB side read) |
| `bads.py`, `_get_target_from_gp_`, and MATLAB `UpdateTarget` | when the target prediction is not finite, both replace it by the incumbent's `fval` and `fsd` but compute the target from the failed variance, so the target is NaN, and the poll then treats the GP as unreliable. In MATLAB this follows a failed rebuild; in the port, a non-finite prediction from a consistent GP | seen (MATLAB side read) |
| `gaussian_process_train.py`, `local_gp_fitting` (at `676083d`, lines 520-529) | after a failed posterior update, retries with the previous hyperparameters on the new training set, which MATLAB's `gpupdate` does not do (it clears the posterior); without a refit the retry repeats the failed computation. Kept so that runs where it succeeds do not move. With a refit, `temporary_data["poll_scale"]`, `["len_scale"]` and `["effective_radius"]` come from the refit's hyperparameters while the retry puts back the previous ones, so the poll basis and the ES-ell search use a geometry the GP does not hold (before the guards too) | fixed in `9ac1a47` and `f65bc91` (the port review's wave 1, W1-10 and W1-11: no retry without a refit) |
| `gaussian_process_train.py`, `add_and_update_gp` (at `676083d`) | on a failed update the port leaves the point out of the GP until the next rebuild; MATLAB's `'add'` keeps it beside an empty posterior. It also recomputes every posterior in full, where MATLAB first tries a rank-1 update (skipped under `SpecifyTargetNoise`); `dev/TODO.md` | by design (`dev/plans/gp-update-guards.md`) |
| `bads.py`, `_re_evaluate_history_` (at `676083d`, line 2618) | rebuilds the GPs stored in `IterationHistory` in place, so the recorded GPs change after the fact; after a failed rebuild it records the restored GP's `fval` and `fsd`, where MATLAB would record NaN | seen |
| `bads.py`, `_poll_step_` with `stobads` (at `a83bd51`) | after a failed add in a noisy poll, `f_poll` is NaN and `_sto_success_improvement_` returns 0 (both of its comparisons are false); under `opp_stobads` (on by default) `sto_success > -1` then moves the poll to `u_poll_best`, never to the NaN point, and sets `reset_gp`: the NaN counts as uncertain, not as a failure. `stobads` is off by default | seen |
| `gaussian_process_train.py`, `local_gp_fitting`, and `bads.py`, the forced refit (at `a83bd51`) | after a failed rebuild the port restores the GP of the entry (old data, priors, hyperparameters and geometry) and refits at the next rebuild whatever `min_refit_time`; MATLAB keeps the new data and the failed rebuild's hyperparameters and `pollscale` beside `post = []` (`gpupdate.m`), and refits only when `gppredcheck` finds its NaN predictions unreliable, after `MinRefitTime` (`bads.m:1242-1244`) | by design (`dev/plans/gp-update-guards.md`, Open Question 7) |
| `bads.py`, `_search_step_` (at `a83bd51`) | after a failed rebuild, the search ranks its candidates by the LCB of the previous GP; MATLAB's `acqLCB` sums over the finite prediction samples, zero when none is, so MATLAB evaluates the first candidate. The poll treats such a GP as unreliable, as MATLAB does | seen (MATLAB side read) |
| `gaussian_process_train.py`, `init_and_train_gp` (at `a83bd51`, lines 161-209) | retries a failing initial fit without bound (from the fifth attempt, from random samples of the priors), so a fit that keeps failing loops forever; initialization only | fixed in `b6a4fd5` (the port review's wave 1, W1-20: ten tries, then `RuntimeError`) |
| `gaussian_process_train.py`, `_robust_gp_fit_`, and the forced refit (at `a83bd51`) | the refit forced after a failed rebuild sends a run that has met a failure into `_robust_gp_fit_`, whose fifth consecutive failed fit raises `ValueError` (section "Failed fits"); the stress run injects no failure into fits, so this is untested | fixed in `776b70d` and `446c443` (the port review's wave 1, W1-12 and W1-13: the bound rises by `noise_nudge[1]`, and a fit whose every try fails keeps its best start) |
| `function_logger/constraints_check.py`, `contraints_check` | removes no previously evaluated point (its `np.unique` keeps the first occurrences, which fall among the candidates), where MATLAB's `uCheck.m` removes them with `setdiff`; a repeat becomes a duplicate GP training input. Duplicates explain none of the four crashes of the section "Crashes on unguarded GP updates": one failing call holds three duplicate pairs, but removing most sets of three of its rows (36 of 40 draws) lets it succeed, and the other three hold none. One exact repeat was evaluated in `ellipsoid_D10` seed 7 on Linux at `8fc1dff` (section "Found while fixing the tests"; `dev/TODO.md`, "Previously evaluated points evaluated again") | seen (MATLAB side read) |
| `init_functions/init_sobol.py`, `init_sobol` | the seed of the initial Sobol design comes from the integer parts of `u0`, so every start point inside the plausible box gives the same design for a given `D`, and the cast is undefined at -1 and below (section "Found while fixing the tests") | seen (MATLAB side read; the saturation of MATLAB's `uint64` product not checked) |
| `gaussian_process_train.py:4` (at `a83bd51`) | imported its `logger` from `asyncio.log`, so the GP fit warnings went to the `asyncio` logger (section "Found while fixing the tests") | fixed at `8fc1dff` (the `BADS` logger) |
| `bads.py:722-770` | a non-empty `fun_values` option is reported to crash | not looked at |
| `optimize_result.py` | `success` is reported always `True`; `exit_flag`, `min_iter` and `min_fun_evals` are reported unread | not looked at |
| `examples/scripts/pybads_example_2_nonbox_constraints.py` | set `options["rng_seed"]`, not a valid option name, in options never passed to `BADS`; the notebook never had these lines, and the script generated from it (tooling plan, Phase 4) no longer has them | resolved |
| `search/es_search.py:239-253` | `ESSearchCMA` calls `ucov` with a signature that does not match its definition (`:294`); no option reaches the class | not looked at |
| `bads.py`, `_init_optim_state_` (at `4566acb`, lines 809-826) | with `specify_target_noise=True`, an `uncertainty_handling` of `None` is set to `False` and then raises `ValueError`, whose message says to leave `uncertainty_handling` empty; MATLAB's `private/setupoptions.m` sets it to `true`, so that `SpecifyTargetNoise` alone turns uncertainty handling on | fixed in `843cd76` (below the table) |
| `gaussian_process_train.py:979-983` (at `1cfe371`) | the fraction of the budget used divides by `min(max_fun_evals, n_train_max) - eff_starting_points`, zero when `max_fun_evals` equals the initial design, and the cubic next to it mixes `x_` and `x` | fixed in `e041de0` (the port review's wave 1, W1-19) |
| `bads.py:880-893` | the `gp_mean_fun` check accepts twelve names, of which only `zero`, `const` and `negquad` are reported to work | fixed in `43138f2` (the port review's wave 1, W1-34: only `zero` and `const`, `negquad` refused) |
| `gaussian_process_train.py`, `local_gp_fitting` (at `06badd3`, lines 296-346) | computes an empirical prior for the constant GP mean (`prior_mean`: centre at a percentile of `gp.y`, width `y_range ** (1/4)`) and never writes it into the priors it sets; MATLAB (`gpdef/gpdefBads.m`, "Update empirical prior for GP mean") updates the mean prior at each training, with variance `yrange.^2/4`, and with `gpFixedMean` also sets the mean hyperparameter to `ymean` under a delta prior, which the port leaves as a `TODO`. Applying the prior would change runs at default options. Applied at `8afbe16`, with MATLAB's percentile (`prctile1`, NumPy's `"hazen"`) and width; on `ellipsoid_D3_hetero` it lowers the median error of 90 seeds from 0.58 to 0.45, not significantly (`dev/experiments/population_ellipsoid_hetero_linux_20260925/`); over the default suite it improves the deterministic ellipsoids, whose median error falls by a factor of 6 to 110, and `rosenbrock_D6`, and worsens no configuration (`dev/experiments/population_linux_gpfixes_20260925/steps/8afbe16/`) | fixed in `8afbe16` |
| `gaussian_process_train.py`, `_robust_gp_fit_` (at `2f3d949`, lines 581-617) | after `tmp_gp.fit(X, Y, s2, ...)` raises, the retry reads `tmp_gp`: the bounds it nudges come from `tmp_gp.get_bounds()`, and with `use_slice_sampler=True` it samples hyperparameters on the data `tmp_gp` holds. What a failed fit leaves there depends on gpyreg: up to 1.3.2 the data of the failed fit and the bounds it filled in, from 1.3.3 the GP as it was before the call. At default options every bound the retry reads is set by `_gp_hyp`, so the retry does not depend on it; with the slice sampler, or the `negquad` mean (whose unset bounds are NaN until a fit fills them), it does. Passing the retry's `X`, `Y` and `s2` explicitly would remove the dependence | fixed in `5956c4b` (the port review's wave 1, W1-21) |
| `bads.py`, `_search_step_` and `_poll_step_` (at `894d205`, lines 1662-1669 and 2062-2069) | the fallback when the acquisition fails draws `index_acq` from `rng.integers(0, len(...) + 1)`, which can return one past the last index (the range of the `np.random.randint` it replaced); the branch is not reached, since `np.argmin(z)` is never `None`, empty or non-finite | fixed in `06badd3` |
| `search/search_hedge.py:72-74` (at `894d205`) | `np.argwhere(rand_uni < np.cumsum(self.prob))[0]` raises `IndexError` when nothing matches, before the `len(self.chosen_hedge) == 0` fallback that it guards can run | fixed in `06badd3` |
| `bads.py:39` (at `894d205`), the `BADS` class docstring | a `:math:` role holding `\mathtt` in a docstring that is not a raw string: Python 3.12 emits `SyntaxWarning: invalid escape sequence` for it when it compiles the module | fixed in `06badd3` |
| `bads.py`, the final estimate (at `1a21844`, lines 1491-1520) | with `specify_target_noise`, `fval` is the mean of the `noise_final_samples` samples and `fsd` their standard error, whatever standard deviations the target returns; MATLAB's `FinalEstimate` (`bads.m:1470-1476`) weights the samples by their precisions and takes `fsd = 1/sqrt(sum(1./ysd.^2))`. With one sample, the port adds `yval` and the `S` of the last logged row, where MATLAB adds nothing under `SpecifyTargetNoise`. The returned `x` does not depend on it | fixed in `4c4a213` (below the table) |
| `function_logger/function_logger.py`, `__call__` (at `1a21844`, line 193) | at level 2, a repeated point is merged into its row by precision weighting, and the call returns the merged value with the new observation's own standard deviation; `add_and_update_gp` then adds that pair beside the point's earlier row, so until the next rebuild the GP weighs the new observation by `(b/(a+b))**2` instead of `b/(a+b)` (`a`, `b` the two precisions). MATLAB's `funlogger` returns the observation itself and does not merge (reported by the review of #65). Returning the observation, with the fix of the next row, changes 20 of 90 runs of `ellipsoid_D3_hetero` and worsens 16 of them (p = 0.0019; `dev/experiments/population_ellipsoid_hetero_linux_20260925/`), so the port keeps the merged value | seen; MATLAB's form tested, not adopted |
| `function_logger/function_logger.py`, `_record` (at `1c8c71d`, line 408) | at level 2, the row of a repeated point was `np.argwhere(self.X == x)[0, 0]`, the first row that shares any one coordinate with `x`, not the row that matches it in every coordinate, so the new observation was merged into another point's value and noise, and that merged value went to the GP. Mesh points share coordinates often: on `ellipsoid_D3_hetero` at `1c8c71d`, 188 of the 227 repeats of seeds 0-89 went into another point's row, in 54 of the 90 runs, and the fix lowers the median error from 0.58 to 0.48 (p = 0.012). The code came with the comment "Like in PyVBMC" | fixed in `032dfcb` |
| `gaussian_process_train.py`, `_robust_gp_fit_` (at `1a21844`, line 680) | after each failed fit, raises the lower bound of the noise hyperparameter by the cumulative nudge, `noise_nudge[0]` per failure, and ignores `noise_nudge[1]`; MATLAB's `gpHyperOptimize.m` raises the starting point by the cumulative `nudge(1)` and the bound by `nudge(2)`, 0 by default (`NoiseNudge = [1 0]`), so its bound does not move. It cannot bear on `ellipsoid_D3_hetero`: none of the 2,530 fits of its seeds 0-89 at `1c8c71d` fails | fixed in `776b70d` (the port review's wave 1, W1-12) |
| `gaussian_process_train.py`, `local_gp_fitting` (at `8afbe16`, lines 298-306) | computes the mean of the prior of the noise hyperparameter, `log(noise_size) + mesh_noise_multiplier * log(mesh_size)`, and never writes it into the priors, as it did for the GP mean; MATLAB (`gpdefBads.m`, "Likelihood prior (scales with mesh size)") updates it at each training. A noisy run sets `mesh_noise_multiplier` to 0, so only deterministic runs differ: there MATLAB's prior centre falls by half the log of the mesh size, and the port's stays at `log(sqrt(tol_fun))` | fixed in `ee9d5d6` (the port review's wave 1, W1-22) |
| `search/es_search.py`, `ESSearch.__call__` (at `8afbe16`, line 216) | returns `us[0], z[0]` without checking that a candidate is left; when `contraints_check` removes every candidate (today only through `non_box_cons`), the run stops with `IndexError`. MATLAB's `searchES.m` returns an empty set (`if ~isempty(us); us = us(1,:); end`). With the evaluated points removed, as in MATLAB (`dev/TODO.md`), 10 of 90 runs of `ellipsoid_D3_hetero` stopped there (`dev/experiments/population_ellipsoid_hetero_linux_20260925/`) | seen (MATLAB side read) |
| `bads.py`, `_search_step_` (at `8afbe16`, lines 1760-1765 and 1869) | the branch for an empty search set assigns no `u_search`, which the step returns, so an empty set stops the run with `UnboundLocalError`; with the row above fixed, 8 of the same 90 runs did. MATLAB's search step keeps its variables across iterations and needs no value | seen (MATLAB side read) |
| `gaussian_process_train.py`, `_gp_hyp` (at `8afbe16`, lines 968-971) | bounds the constant GP mean, for the whole run, by gpyreg's defaults for the initial design's high-density points (`min(y) - h/2` to `max(y) + h/2`, `h` their spread); MATLAB's bounds are `[-Inf, Inf]` (`gpdefBads.m`). When the run descends far below the initial design relative to that spread, the mean, and the centre of its prior since `8afbe16`, can fall outside the bounds. Reached at default options: in the Windows reference at `ab4dded` (`dev/experiments/population_gpfixes_20260925/`), the prior of the mean lies so far below the lower bound of the mean that its mass inside the bounds underflows to 0, and the log prior is NaN in fits of all 30 runs of `ackley_D6`, 28 of `sphere_D10` and 9 of `sphere_nonbox_D3`; in `sphere_D10` seed 0 the fitted mean sits at its lower bound, 8.87, in 9 of the 15 GPs of the iteration history, above every training target. The median errors of the three configurations do not change significantly | fixed in `172df00` (the port review's wave 1, W1-23: the mean unbounded, as MATLAB); the infinite log prior of W1-24 fixed in gpyreg (acerbilab/gpyreg#57, not yet released) |
| `bads.py:853` (at `1a21844`) | `np.array(self.options["noise_size"] > 0)[0]` raises `IndexError` for a scalar `noise_size`, so `specify_target_noise=True` with a `noise_size` stops when `BADS` is created; the check is meant to warn that `noise_size` is ignored | fixed in `6dad7e4` |
| `bads.py:5` (at `1a21844`) | imports `logger` from `asyncio.log` and logs to it at line 2284, the slip fixed in `gaussian_process_train.py` at `8fc1dff`; `gaussian_process_train.py` logs four debug messages of failed GP updates with `logging.debug`, to the root logger | fixed in `6f28fc7` |
| `bads.py`, `_save_gp_stats_` calls (at `1a21844`, lines 1694-1696 and 2138-2140) and `_re_evaluate_history_` | the GP calibration statistics store the latent standard deviation, where MATLAB stores that of the observation, likelihood noise included; `_re_evaluate_history_` selects the neighbours of each stored GP with that GP's `len_scale` and `effective_radius`, where MATLAB uses the current GP's (`bads.m:1378-1388`) (reported by the review of #65) | first clause fixed in `d883cf9` (the port review's wave 1, W1-3); the second left to wave 2 (wave 1's ledger, "Found while verifying") |
| `bads.py`, the final estimate without target noise (at `4c4a213`) | `fsd` is `np.std(yval_vec) / sqrt(n)`, with NumPy's default normalization by `n`; MATLAB's `std` in `FinalEstimate` normalizes by `n - 1`, so the port's `fsd` is smaller by `sqrt((n - 1)/n)`: 0.95 at the default 10 samples, 0.71 with one sample, to which `yval` is added | fixed in `7b50a3a` (below the table) |
| `optimize_result.py:122` and `bads.py`, the final estimate (at `4c4a213`) | with noise and `noise_final_samples > 0`, `OptimizeResult` reads `optim_state["yval_vec"]`, which only the final re-evaluation sets, and that runs only from the second iteration on (`poll_iteration > 0`): a noisy run that ends within its first iteration, for instance with `max_iter=1`, raises `KeyError: 'yval_vec'`. MATLAB sets `yval_vec = yval` before the branch (`bads.m:1136`). Reproduced with the noisy sphere of `dev/scripts/fingerprint.py`, its noise drawn from a fresh `default_rng(0)`, `uncertainty_handling=True`, `random_seed=0` and `max_fun_evals=45`; at 35 and 40 the run stops earlier, in `_get_gp_training_options`, with `ValueError: cannot convert float NaN to integer` from the division by zero of the row at `1cfe371` | fixed in `843cd76` (below the table) |
| `gaussian_process_train.py`, `_gp_hyp` (at `4bde5e9`, lines 931-939) | the upper bound of the log length scales is `cov_range = min(100, 10 * (ub - lb) / scale)` itself, where MATLAB's `gpdefBads.m:90` takes `log(covrange(i))`: on the targets of the benchmark with its shifted box (`SHIFTED_BOUNDS` of `dev/scripts/benchmark_targets.py`), where `cov_range` is 80, the port lets a log length scale reach 80 and MATLAB stops at 4.38. At the failing calls of the four crashes of the section "Crashes on unguarded GP updates", the log length scales reach 19.3 in 3-D and 27.5 to 59.7 in 10-D, with the output scale at its upper bound, a plausible cause of their degenerate GPs (not checked). Fixing it moves results at default options. Fixed as in MATLAB in `97b2c66`: on `ellipsoid_D3_hetero` the log bound lowers the median error of 90 seeds from 0.58 to 0.38 and brings the error along the flat axis back to its level before `020d6a8`; with `032dfcb` and `8afbe16`, 0.25 (`dev/experiments/population_ellipsoid_hetero_linux_20260925/`); over the default suite it worsens no configuration (`dev/experiments/population_linux_gpfixes_20260925/`). Whether it would have prevented the crashes is a `dev/TODO.md` item | fixed in `97b2c66` |
| `bads.py`, `_poll_step_` (at `4bde5e9`, line 2115) | calls `np.seterr(divide="ignore")` when the root logger is above DEBUG and never restores it, so a run changes NumPy's error handling for the rest of the process; with the root logger at DEBUG, the division by the predicted SD below it warns `divide by zero` | seen |
| `gaussian_process_train.py`, `local_gp_fitting` (at `be84ff6`, lines 371-390) | the check for a high-noise GP reads `noise_size` whatever the noise mode, so `noise_size` is not ignored under `specify_target_noise`, as the warning of `_init_optim_state_` says it is; `noise_size=0`, which the warning proposes to silence it, gives `log(0) = -inf` and makes every refit a high-noise one. MATLAB's `private/gpupdate.m:379-381` reads `NoiseSize` the same way (reported by the review of #67, which saw such runs end at another `x`) | fixed in `7b50a3a` (below the table): under `specify_target_noise` the check takes the default base 1 (PI, 2026-09-25) |
| `gaussian_process_train.py:374` and `bads.py:1095-1098` (at `be84ff6`) | a list `noise_size` raises `TypeError` at the first refit (`len(options["noise_size"] == 1)`, where MATLAB has `numel(options.NoiseSize) == 1`), and an array of two elements, MATLAB's base value and prior SD, raises `ValueError` at `.item()`; the warning check at the creation of `BADS` accepts both (reported by the review of #67) | fixed in `7b50a3a` (below the table) |
| `bads.py`, the final estimate (at `be84ff6`, lines 1526-1529) | records the final `fval` and `fsd` in the iteration history at `poll_iteration`, the last iteration, while they describe the iterate `min_q_beta_idx`; MATLAB writes `iterList.fval(index)` (`bads.m:1158-1159`) (reported by the review of #67) | fixed in `7b50a3a` |
| `bads.py`, `optimize` (at `4745346`) | the iteration count, `optim_state["iter"]`, which `OptimizeResult` reports as `iterations`, starts at 0 where MATLAB's `iter` starts at 1 (`bads.m:482`, `private/bads_output.m:21`), so a run reports one iteration less than MATLAB's would: with `max_iter=1` a run polls once and reports 0 (reported by the second review of #67) | fixed in `7b50a3a` |
| `bads.py`, `optimize` (at `4745346`, lines 1208-1212 and 1554) | `output_fcn` is called only at the start, as `output_fcn(x, "init")`, where MATLAB calls `outputfun(x, optimState, state)` at `'init'`, `'iter'` and `'done'` (`bads.m:427`, 1038, 1165); one that returns `True` at the start stops the run with `UnboundLocalError` for `msg`, which only the loop assigns (reported by the second review of #67) | fixed in `7b50a3a` |
| `bads.py`, `_init_mesh_` (at `4745346`, lines 990-993) | with `max_fun_evals=1`, returns after evaluating `x0`, discarding its local `is_finished = True` and before setting `optim_state["eff_starting_points"]`, so the run stops in `_get_gp_training_options` with `KeyError: 'eff_starting_points'` (reported by the second review of #67) | fixed in `7b50a3a` |

Three of the rows marked "at `1a21844`" are fixed in pull request #67,
whose commits cited here are reachable from `refs/pull/67/head`: the
`noise_size` check in `6dad7e4`, the loggers in `6f28fc7` and the final
estimate in `4c4a213`. The fingerprint of `dev/scripts/fingerprint.py` is
`57241c985a68c78b` at `f11d2ee` and at `be84ff6`, whose package code is
that of `4c4a213` (Windows, gpyreg 1.3.3 at `98ab5a4`); none of its six
runs has target noise. `4c4a213` combines the final samples as MATLAB's
`FinalEstimate` does (`bads.m:1443-1476`). At `4c4a213`, `sphere_D3_hetero`
and `ellipsoid_D3_hetero` over seeds 0-29 give, in all 60 runs, the `x`,
evaluations, iterations and error of the Windows reference
`population_targetnoise_20260925` (at `c044fea`; the changes of the package
between the two that these runs reach are the final estimate, the loggers
of `6f28fc7`, which change no value, and a termination message). Their
`fval` moves by at most 4.4e-16: the benchmark's noise standard deviation,
`1 + sqrt(f - f_min)`, is the same for every sample at a point, so the
weighted mean is the plain mean up to rounding. Their `fsd`, which is that
standard deviation divided by `sqrt(10)`, differs from the reference's by
factors from 0.78 to 2.19 (medians 1.09 and 0.97). So from `4c4a213` on,
the `fval` and `fsd` of these two configurations differ from the
reference's records, while the error and the evaluations, which
`population.py compare` tests, do not.

Two more rows are fixed in `843cd76`, with the same fingerprint: with
`specify_target_noise=True`, an empty `uncertainty_handling` turns
uncertainty handling on, as in MATLAB's `setupoptions.m`; and a noisy run
that ends within its first iteration returns the incumbent's observation as
`yval_vec`, with `ysd_vec` set to `None`, as `bads.m:1136-1137` sets them.

Seven more rows, of `be84ff6` and `4745346`, are fixed in `7b50a3a`
(reachable from `refs/pull/71/head`), with the fingerprint
`f80abf397f44fc62` before (`ab4dded`) and after (Windows, gpyreg 1.3.3 at
`98ab5a4`); the fingerprint covers `x`, `fval`, `func_count`
and `yval_vec`, not `fsd` or `iterations`. Without target noise, the final
`fsd` uses the standard deviation of the samples normalized by `n - 1`, as
MATLAB's `std` does. `noise_size` takes MATLAB's forms, a scalar or a pair
of the base and the SD of the prior over its logarithm (`gpdefBads.m:147-152`,
`gpupdate.m:379-381`), and without target noise a value of 0 or less is
refused, as `setupoptions.m:80-81` refuses it; with gpyreg 1.3.3 such a
value already stopped the run, in the prior of the noise. Under
`specify_target_noise`, where MATLAB's high-noise check reads `NoiseSize`
although `setupoptions.m:100-101` warns that it is ignored, the port ignores
it: the check takes the base 1 that an empty `noise_size` gets, so a run
that leaves it empty is unchanged, and `noise_size=0`, which the warning
proposes, no longer makes every refit a second fit. `output_fcn` is called
as MATLAB calls it, with a copy of `optim_state`.

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
`test_he_noisy_sphere_opt` comes from a second sweep of that test at
`020d6a8`, which squares the target's noise standard deviations into the
GP's noise variances (below). `020d6a8` reaches only targets that return a
noise standard deviation, so the other rows stand.

| Test | Evaluations | Median error | Largest error | Tolerance | Before `9a99bae` | `runtest.m` |
|---|---|---|---|---|---|---|
| `test_ellipsoid_opt` | 67–93 | 8.7e-6 | 9.8e-5 | 1e-3 | 1 | 0.1 |
| `test_sphere_opt` | 65–100 | 2.0e-5 | 1.9e-4 | 5e-3 (2e-3 until `b536b8f`, below) | 1 | 0.1 |
| `test_noisy_sphere_opt` | 100 | 5.7e-3 | 0.10 | 1 | 1 | 1 |
| `test_he_noisy_sphere_opt` | 185–200 | 0.14 | 0.56 | 1 | 1 | 1 |
| `test_small_noisy_func` | 149–223 | 5.9e-6 | 2.5e-4 | 1e-2 | 0.1 | |
| `test_1D_opt_*` (three tests) | 30 | 5.8e-9 | 2.1e-7 | 5e-6 | 0.1 | |
| `test_univariate_input_and_opt` | 87–141 | 3.3e-5 | 1.4e-3 | 2e-2 | none | |
| `test_high_dim_opt` | 200 | 2.9e-2 | 5.5e-2 | 1 | none | |

On Linux (Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, gpyreg 1.3.3 from
its tag), the sweep ran at `1c8c71d`, at `8afbe16` (the GP mean prior) and
at `97b2c66` (the log bound of the GP length scales), seeds 0-99 each; no
run crashed or reached its tolerance. At `97b2c66` the largest error of
`test_sphere_opt` is 3.6e-4 (seed 68; 1.6e-4 at `1c8c71d`, 1.0e-4 at
`8afbe16`), so the rule gives it 5e-3 (`b536b8f`). The other tolerances
stand; the largest errors at `97b2c66` are 2.6e-5 (`test_ellipsoid_opt`),
1.4e-3 (`test_univariate_input_and_opt`), 6.2e-8 (`test_1D_opt_*`), 0.085
(`test_high_dim_opt`), 0.105 (`test_noisy_sphere_opt`), 1.4e-4
(`test_small_noisy_func`) and 0.65 (`test_he_noisy_sphere_opt`, 0.79 at
`1c8c71d`).

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
its noise (`specify_target_noise`), 2 at the minimum. Before `020d6a8`,
each rebuild of the local GP and each addition of a point stored these
standard deviations in `gp.s2`, which gpyreg reads as variances; only the
initial fit squared them. MATLAB BADS keeps them in `gpstruct.s` and
squares them in `likGaussHe` (`sn2 = exp(2*hyp) + s.^2`). Before the fix,
the test had a tolerance of 5, and its errors over seeds 0–99 exceeded the
tolerance of `runtest.m`, 1, in 7 seeds, with a median of 0.33 and a
largest error of 2.3. Rerun at `10d74a7`, the sweep reproduced the one at
`870fa35` run by run. With the code of `020d6a8`, in the same environment,
no error reaches 1: the median is 0.14 and the largest 0.56, and the error
is smaller in 70 of the 100 paired seeds (Wilcoxon signed-rank test,
p = 4e-8). The tolerance, 1, is 1.8 times the largest error, a narrower
margin than the other tests have. The logs of both sweeps are kept on the
machine that ran them (`dev/scripts/runs/LOCAL.md`).

In the benchmark, `020d6a8` changes only the two configurations with
target noise, whose noise standard deviation is `1 + sqrt(f - f_min)`, and
the comparison of
[`population_targetnoise_20260925`](../experiments/population_targetnoise_20260925/README.md)
with `population_gpyreg133_20260924`, 30 seeds each, flags neither. The
median error of `sphere_D3_hetero` falls from 0.20 to 0.10 (signed-rank
p = 0.008 before the Holm correction). That of `ellipsoid_D3_hetero`, a
3-D ellipsoid with condition number 1e6, rises from 0.26 to 0.37
(p = 0.10), and over 90 seeds
([`population_ellipsoid_hetero_20260925`](../experiments/population_ellipsoid_hetero_20260925/README.md))
from 0.21 to 0.54 (p = 8.5e-6), with 28 runs at or above 1 where there
were 4, and a largest error of 2,478. Most of the increase lies along the
flat axis of the ellipsoid. `dev/TODO.md` holds the diagnosis.

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
  one of the crashing seeds recorded above, whose crash on Windows at
  `2226883` came on a training set without duplicates (section "Crashes on
  unguarded GP updates"); `dev/TODO.md`, "Previously evaluated points
  evaluated again", holds the rest of the check. Fixing
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
  `D = 5`; the 64-bit product is 0 from `D = 8`. A product of 0 gives the
  seed 1. The two give different designs for `D` from 4 to 7, and the same
  at the dimensions of the tests (1, 2, 3 and 60). PyBADS requires NumPy 2
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
matplotlib 3.9.0 or later, as PyVBMC has required NumPy 2 since 1.0.4.
NumPy 1.x ran in no CI job, and on Windows its 32-bit default integer gives
`init_sobol` other designs at `D` from 4 to 7 (above). SciPy 1.13 is the
first release that works with NumPy 2. No CI job installs these minimums. Installed from the package, with
gpyreg 1.3.3, the suite passed on Windows in two environments: at `23adc75`
(159 tests) with Python 3.12 and these minimums, and at `10d74a7` (157
tests) with Python 3.11, NumPy 1.23.5, SciPy 1.9.3 and matplotlib 3.6.3.
The previous minimums (NumPy 1.22.1, SciPy 1.7.3 and matplotlib 3.5.1)
were not run. Under NumPy 2, `pybads.stats.kde1d` raised `AttributeError`
(`np.asfarray` and `np.product`, removed in NumPy 2.0) until `7f00fef`; no
test called it.
