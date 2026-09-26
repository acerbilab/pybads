<!-- Report of fix agent I of wave 1's fix pass (the gpyreg side: rows W1-25, the opt-in switch, and W1-24), working in two worktrees of acerbilab/gpyreg at main 51ebf95 (/home/user/gpyreg-wt/switch, branch pybads-w1-cholesky-switch; /home/user/gpyreg-wt/logmass, branch pybads-w1-log-prior-mass), in a cloud session; saved verbatim from its final message on 2026-09-26. Its commits are gpyreg's, reviewed by the orchestrator; the hashes it cites are those of its branches. -->
# Fix report: gpyreg W1-25, W1-24

Both rows are committed, one commit each, on their own worktrees. Nothing is pushed, and no branch was created, switched or rebased. Scripts and outputs are in `/tmp/claude-0/-home-user-pybads/4f4200d5-5a60-5de0-9b26-94c4f1e930f9/scratchpad/wave1fix/I/`. Main's code for the comparisons was exported there with `git archive main gpyreg`, into `main_src/`.

## W1-25: `pybads-w1-cholesky-switch`, `33dc2e7` "feat: opt-in switch that makes a failed Cholesky factorization an error"

**Files changed:** `gpyreg/gaussian_process.py`, `gpyreg/testing/test_gaussian_process.py`, `docsrc/source/release_notes.rst` (353 lines added, 28 removed).

**The switch**
- It is `raise_on_cholesky_failure: bool = False`, a keyword of `GP.__init__`, kept as the attribute `gp.raise_on_cholesky_failure`.
- It is read in one place, `GP.__cholesky_attempts()`, as `getattr(self, "raise_on_cholesky_failure", False)`. That returns 1 attempt when the switch is on and 10 when it is off.
- **Why the constructor:** the switch is a property of the model the GP computes. It is read everywhere the training covariance is factored: `update`, the objectives, and `predict` on old pickles. `fit` options would not reach those.
- A deepcopy or pickle keeps it. The `getattr` default covers GPs unpickled from 1.3.3 or earlier, which lack the attribute and keep inflating the noise.
- **Why the name:** it says what it does (raise), on what (the Cholesky factorization of the training covariance), and `False` is today's behaviour. I chose a boolean over a count mirroring `CholAttempts`: MATLAB's `kmax > 0` does a nearest-SPD repair, not gpyreg's ×10 scheme, so only 0 corresponds.

**What changed**
- `__training_cholesky` takes `attempts=10`; the loop is `range(0, attempts)`. `__core_computation` passes `attempts=self.__cholesky_attempts()`, and so does `__low_noise_factor`, which recomputes the factor of posteriors pickled by 1.3.1 or earlier.
- With the switch on, `update`, `set_hyperparameters`, the fit's final `update`, `log_likelihood`, `log_posterior` and the fit's objectives raise `LinAlgError` ("Singular matrix for L Cholesky decomposition") at the first failure. The GP is left as it was, through the existing state and restore.
- In `fit`, the design objective `objective_f_1` (a lambda before, now a small `def`) catches `LinAlgError` only when the switch is on and returns `np.inf`. A failing candidate is then ranked last, as `gpHyperOptimize.m:55-59` does. This applies to the `f_min_fill` design and also to the `init_N=0` ranking of `hyp0`.
- A failure in the optimizer, the slice sampler or the final `update` propagates out of `fit`, as today.
- Docstrings updated: the class Parameters section (the keyword's documentation), and the `LinAlgError` entries under Raises in `set_hyperparameters`, `update`, `fit`, `log_posterior`, `predict_full`, `predict`, `quad`, `random_function` and `__core_computation`. `log_likelihood` gets a new Raises section.

**Tests** (in `test_gaussian_process.py`, seeded; `_gp_1d` now forwards `**kwargs`):
- `test_failed_factorization_inflates_the_noise_by_default[high|low]`: the default and explicit `False` give the same posterior, `predict` and objective with gradient, bit for bit, with `sn2_mult > 1`. It covers both noise parametrizations (sn2 = 4e-6 and 1e-16).
- `test_failed_factorization_raises_where_the_gp_raises[high|low]`: `update(hyp=...)`, `set_hyperparameters`, `log_likelihood` and `log_posterior`, each with and without gradient, raise `LinAlgError`. After a failed update the GP holds the same objects and values, and its predictions are unchanged.
- `test_fit_ranks_starting_points_whose_factorization_fails_last`: the design has 2 failing starting points. With the switch on they are exactly the inf entries of y0 (off, all y0 are finite). The fit returns finite, in-bounds hyperparameters with `sn2_mult == 1`.
- `test_a_fit_whose_optimization_fails_raises_where_the_gp_raises`: with `init_N=0, opts_N=2`, the second optimization starts from a failing point. `fit` raises, and the GP is left as it was. Without the switch the same fit completes.
- `test_a_copy_and_a_pickle_keep_raise_on_cholesky_failure`: a deepcopy and a pickle keep the switch. After `del gp.raise_on_cholesky_failure` (as if unpickled from 1.3.3), the GP inflates.
- **Results:**
  - At the commit: `7 passed, 416 deselected in 1.29s`. The whole `test_gaussian_process.py`: `423 passed, 14 warnings in 56.84s`. `test_predict_cross_covariance.py` plus `test_gaussian_process_isotropic.py`: `73 passed`.
  - On main: `7 failed` (the keyword does not exist). The behavioural contrast is in the evidence below.

**Evidence the default is bit-identical to main** (`evidence_w125.py`, SHA-256 over every array):
- Cases:
  - Posteriors (all fields), `predict`, `predict_full`, `log_likelihood` and `log_posterior` with gradients, `random_function` and `quad`. These use problems with `sn2_mult` 10, 1e5, and [10, 100].
  - Single-point updates of those inflated posteriors.
  - Fits with (init_N, opts_N, n_samples) = (16, 2, 0), (0, 2, 0) and (64, 2, 4), their designs included.
  - A Rosenbrock-like 2-D RQ-ARD fit where 37 of 433 factorizations were inflated. In total, 57 of 949 factorizations were inflated.
- The hash `4c8f132314d84eb1dca8b7816fc81dfe69e0546d11a264689f531d1b1c0561ee` is identical for main, for the branch with the default, and for the branch with explicit `False` (`evidence_w125_{main,branch,branch_explicit,branch_33dc2e7}.out`).

**Proposed release note** (committed under 1.3.4 (unreleased)):
> :class:`gpyreg.GP` takes the keyword ``raise_on_cholesky_failure``, ``False`` by default. With ``True``, a Cholesky factorization of the training covariance that fails raises ``numpy.linalg.LinAlgError`` at its first attempt, in the posteriors and in the objective of :meth:`gpyreg.GP.fit` (:meth:`gpyreg.GP.log_likelihood` and :meth:`gpyreg.GP.log_posterior`), as in MATLAB BADS (``CholAttempts = 0``). By default such a factorization is tried again with the noise multiplied tenfold, up to ten times, and the posterior keeps the multiplier, which its predictions apply to the noise while :meth:`gpyreg.GP.get_hyperparameters` returns the noise as fitted. With the switch on, the space-filling design of ``fit`` ranks a starting point whose factorization fails last, as one of infinite value, and a failure in the optimization, in the sampling or at the fitted hyperparameters raises from ``fit``, which leaves the GP as it was. A copy or a pickle of the GP keeps the switch. Without it nothing changes, to the last bit.

**Proposed PR:** "feat: opt-in `raise_on_cholesky_failure`, a failed Cholesky factorization as an error". Description:
> gpyreg retries a failed factorization of the training covariance with the noise ×10, up to 10 times, and keeps the multiplier in the posterior used for prediction, while `get_hyperparameters` reports the fitted noise. MATLAB BADS raises at the first failure (`CholAttempts = 0`, `bads.m:272`, `infExact_fastrobust.m:79`), gives a failed starting point Inf (`gpHyperOptimize.m:55-59`), restarts a failed fit with the noise nudged, and empties the posterior (`gpupdate.m:340-350`). On Rosenbrock-like targets PyBADS hands on GPs with noise inflated 10 to 1e9 times.
>
> This adds an opt-in constructor keyword, off by default. With it on, the posteriors and objectives raise at the first failure, the fit's design ranks failing candidates last, and an optimizer or sampler failure raises from `fit` (the GP is left as it was). Off, everything is bit-identical to main (evidence: hashes of posteriors, predictions, objectives, rank-one updates and fits, including sampling and a Rosenbrock-like fit). There are tests for both states, restore, copy/pickle and old pickles.
>
> 🤖 Generated with [Claude Code](https://claude.com/claude-code)
>
> https://claude.ai/code/session_0197n9Ht4wrQPf3CfxPg4bs3

**Uncertain or to weigh**
- **Rosenbrock-like fit with the switch on** (`explore_w125_on.out`): the fit raises. The default completes, with all its inflations inside L-BFGS-B's steps. So with the switch on, PyBADS will rely much more on `_robust_gp_fit_` retries. This is as ruled, but it is the main thing the exploratory population will measure.
- **Second start with `opts_N >= 2`** (PyBADS's `second_fit`): the second start is the best of the lowest-noise 20% of the design. Those are the candidates likeliest to fail, and if all of them fail, the whole fit raises even though the first optimization succeeded.
- **The slice sampler** does not catch `LinAlgError`, so with the switch on a failure while sampling ends the fit. The ruling does not cover sampling; I left it propagating and documented that.
- **PyBADS calls `gp._GP__gp_obj_fun` directly** (`gaussian_process_train.py:829`, its own slice sampler). With the switch on that call raises.

## W1-24: `pybads-w1-log-prior-mass`, `890af19` "fix: take a prior's log mass in log space where it underflows"

**Files changed:** `gpyreg/gaussian_process.py`, `gpyreg/f_min_fill.py`, `gpyreg/testing/test_gaussian_process.py`, `docsrc/source/release_notes.rst` (287 lines added, 7 removed).

**What changed and why**
- **New helper `_log_gaussian_mass(lower, upper, mu, sigma)`** (module-level, private). It uses `logsf` or `logcdf` at the two bounds, with the same `lower > mu` switch as the linear mass, then adds `log(1 - exp(d))` computed through `expm1` or `log1p`.
- **`__recompute_normalization_constants`** stores the linear mass unchanged in `normalization_constants`. Only for a Gaussian prior whose linear mass `== 0` does it also store the log-space log in a new private array `_underflowed_log_masses` (NaN elsewhere; excluded from `repr`).
- **`__prior_masks`** computes `log_norm` exactly as before, `np.sum(np.log(constants))`, unless some constant is 0 and has a stored log. Then those entries use the stored log. `getattr` covers old pickles, and `fit` recomputes the constants anyway through `set_bounds`.
- **Scope of the trigger:** "where the linear one underflows" is taken as a mass of exactly 0, i.e. only where today's log is -inf. Subnormal masses (z ≈ 37.5–37.7) already give a finite and accurate log, so they are untouched.
- **Where the log is computed:** at recompute time rather than lazily, so it matches the constant's own df semantics. During `fit`, a NaN-df `student_t` reads as a t with `df_base`, while its constant was computed as a Gaussian.
- **`f_min_fill`:** in the Gaussian (df == 0) branch, any draw that comes out non-finite is redrawn with `sp.stats.truncnorm.ppf(S, (LB-mu)/sigma, (UB-mu)/sigma)*sigma+mu` and clipped to `[LB, UB]`. Every finite draw is kept as computed.

**Tests** (new: `test_prior_mass_far_outside_the_bounds[z×side]`, `test_space_filling_design_far_outside_the_bounds[z×side]`, `test_fit_with_a_prior_far_outside_its_bounds[side]`):
- **Mass:** z ∈ {37, 38, 60}, with the bounds of the constant mean above or below its N(0, 1) prior. `normalization_constants[3]` equals the linear sf or cdf difference bit for bit. It is > 0 exactly at z = 37, where `_log_gaussian_mass` is monkeypatched to fail the test if called.
- **Log prior:** it matches logpdf − (Mills-ratio asymptotic log tail) to rtol 1e-12, which is independent of scipy's logsf. It is finite with a finite gradient.
- **Design:** at 37 the draws equal the linear mapping bit for bit. At 38 and 60 they are finite, in bounds, and within 1e-2 of the Exp(rate z) quantiles (the truncated tail). The linear mapping there is shown to be infinite.
- **Fit:** the prior sits 40 SD outside the bounds. The design's mean values are finite and in bounds, all y0 are finite, and the returned hyperparameters are finite and in bounds, with a finite log posterior.
- **Results:**
  - At the commit: `14 passed`. `test_gaussian_process.py` plus `test_gaussian_process_isotropic.py`: `453 passed, 14 warnings in 93.57s`.
  - On main: `10 failed, 4 passed`. The four z = 37 cases pass there. The failures are `assert np.isfinite(log_prior)` (log prior inf) and `assert np.all(np.isfinite(X))` (design ±inf); in the fit test, y0 is -inf and the mean draws are inf.

**Evidence the result is bit-identical to main where nothing underflows** (`evidence_w124.py`, `evidence_w124.out`):
- 223 arrays go into one hash: constants, log posterior and gradient, and designs for a Gaussian with z from 0 to 37.6 on both sides (subnormal masses included); Student-t, smoothbox and smoothbox-t cases; one-sided bounds; and three fits with priors on every hyperparameter (one with the mean prior 25 SD out, one with sampling).
- The hash `670f05f67ee1df7d086676d85e858bed2bbd3b49aad37252ae403c478a5fecbe` is the same on main and on the branch.
- The changed cases:

| Case | log posterior on main | log posterior on branch | design on branch |
|---|---|---|---|
| z = 38 | inf | -17.51 | finite, in bounds |
| z = 60 | inf | -28.05 | finite, in bounds |
| z = 75 (the verifier's setup) | inf | -186.72 | not checked |

**Proposed release note** (committed):
> A Gaussian prior whose centre lies some 38 of its scales or more outside the bounds of its hyperparameter has a finite log prior, and the space-filling design of :meth:`gpyreg.GP.fit` draws finite values of that hyperparameter inside its bounds. The prior is renormalized by its mass inside the bounds, the difference of its cumulative distribution function (or of its survival function) at the two bounds, and so far in one tail both values underflowed to zero, and the mass with them: the log prior was infinite at every hyperparameter, so that :meth:`gpyreg.GP.log_posterior` was infinite and the objective of the fit minus infinity, and the design mapped its draws through the same zero values, to infinity. Where the mass underflows to zero its log is taken in log space, and a draw that would be infinite comes from the prior truncated to the bounds; every other computation is unchanged, to the last bit.

**Proposed PR:** "fix: log mass of a truncated Gaussian prior in log space where it underflows". Description:
> Each prior is renormalized by its mass inside its bounds, a linear sf or cdf difference. It underflows to 0 when the prior is about 38 SD or more outside the bounds, which makes the log prior +inf and the fit objective -inf at every hyperparameter, and gives infinite design draws. PyBADS reaches this at default options through its constant-mean prior.
>
> Only where the linear mass is exactly 0, its log is now taken in log space (logsf/logcdf plus a stable log1mexp). Only where a Gaussian design draw is non-finite, it is redrawn from `truncnorm` and clipped to the bounds. Everything that works today is bit-identical: an evidence hash over constants, log priors, gradients, designs and fits for z up to 37.6, the other prior families and prior-laden fits matches main. Tests at z = 37 (bit-identical path), 38 and 60, above and below, plus a fit.
>
> 🤖 Generated with [Claude Code](https://claude.com/claude-code)
>
> https://claude.ai/code/session_0197n9Ht4wrQPf3CfxPg4bs3

**Uncertain**
- **Old scipy:** the fallback draws rely on `scipy.stats.truncnorm.ppf` being accurate in the far tail. That holds on scipy 1.17 here. gpyreg's minimum is scipy 1.7.3; I believe truncnorm has handled extreme tails since about 1.4, but I could not test that version.
- **Smooth-box and Student-t priors:** the smooth-box priors have Gaussian tails beyond the box, so they have the same underflow about 38 σ past their box, in both the mass and the draws. I left them alone as outside the ruling, and the commit says so. The Student-t families underflow only at extreme z.

## Noticed outside my rows
- **Pre-commit could not run fully:** the isort 5.10.1 and pycln 1.2.5 hook environments fail to build here (Poetry metadata; a pyyaml/Cython build error). I ran pre-commit with `SKIP=isort,pycln` (whitespace, end-of-file, black 22.3.0) and `isort 9.0.1 --check-only` from the venv with the pyproject settings, which was clean. pycln was not run; the only import added is `pickle` in the tests, which is used.
- **The two branches conflict** only in `release_notes.rst`: both append a bullet after the same line (`git merge-tree` shows it). The resolution is to keep both bullets.
- **Latent mismatch in `__low_noise_factor`** (main): it discards the multiplier `__training_cholesky` returns. With the default, a recomputation that needed more inflation than the posterior's `sn2_mult` would hand back a factor that does not match the posterior.
- **Constant mismatch during `fit`** (main): for a `student_t` prior with NaN df, the fit's log prior uses `df_base`, while the normalization constant was computed as a Gaussian. It is a constant only, so the MAP is unaffected.
- **gpyreg's `AGENTS.md`** ("Jitter is multiplicative … retried up to 10 times") should mention the switch. I did not edit it. Proposed wording: "unless the GP was created with `raise_on_cholesky_failure=True`, where the first failure raises `LinAlgError` and the fit's design ranks such a candidate last".
- **`repr(gp)`** now lists `raise_on_cholesky_failure` (W1-25).
