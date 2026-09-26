<!-- Report of the verifier of wave 2, slice B1 (the two B1 reports and the items kept from the reviewers, given as B1-K1 to B1-K8), reading PyBADS at fef6c14 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), with the complete history, in a cloud session; saved verbatim from its final message on 2026-09-26. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave2/B1_verifier/. -->

# Wave 2 verification: B1

I checked PyBADS at `fef6c14` (`/home/user/pybads-review`) against MATLAB BADS at `74919c0` and gpyreg v1.3.3. Every script printed `pybads.__file__ = /home/user/pybads-review/pybads/__init__.py` and `gpyreg.__file__ = /home/user/gpyreg-v1.3.3/gpyreg/__init__.py`.

- **Where things are:** scripts and outputs are in `/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/review/B1_verifier/`, one `.out` per script. `common.py` holds a Python transcription of MATLAB's `boundscheck.m`, `setupvars.m:6-9, 58-99`, `transvars.m` and `force2grid.m`, used as the reference throughout.
- **Report labels:** "I-Fn" is `B1_internal.md`, "C-Fn" is `B1_comparison.md`.
- **Dates:** I dated every line myself with `git log -L` on the complete history. The comparison reviewer's history note agrees with my dates wherever we overlap. Both docstring sentences that I-F1 and I-F2 date to `ce3a0b3` are in fact from `c7c88ab`.

## 1. Summary

| Finding | Classification | Reached at default options | Dating | Confidence |
|---|---|---|---|---|
| **B1-K1** = I-F1 = C-F2 (a): a mix of bounded and unbounded variables is refused | confirmed port discrepancy | yes, for such inputs (no option needed); levels 0/1/2 | never agreed (Python `c7c88ab`, unchanged; MATLAB never had the check) | high |
| B1-K1 = I-F1 = C-F2 (b): half-bounded variables are refused | design question | yes, for such inputs; levels 0/1/2 | never agreed | high |
| I-F2 = C-F5: scalar bounds are not replicated when D > 1 | confirmed port discrepancy | yes, for such inputs; levels 0/1/2 | never agreed (MATLAB `42ae029` 2017; Python `c7c88ab`) | high |
| C-F1 = I-F4: effective bounds move `x0`, `plb` and `pub` by 1e-3·(ub−lb), or refuse the problem | confirmed port discrepancy | **yes**: bounded problems with `plb`/`pub` omitted or equal to `lb`/`ub`, or with `x0`/`plb` near a bound; levels 0/1/2 | never agreed (Python `c7c88ab`; MATLAB never had them) | high |
| I-F3: the "x0 inside the plausible box" test compares with the effective bounds | part of C-F1; the reviewer's proposed direction is not MATLAB's | as C-F1 | as C-F1 | high |
| I-F5: the transform's self-test uses an absolute 1e-6 tolerance | confirmed shared defect | only with \|bounds\| ≳ 1e10 (linear) or ub ≳ 1e9 (log) | always agreed (MATLAB `6c93629` 2017; Python `c7c88ab`) | high |
| **B1-K2** = I-F6 = C-F9: `fun_values` crashes | confirmed port discrepancy | no (default `{}` skips it) | never worked (`c7c88ab`, loop `8e59038`) | high |
| B1-K2 = I-F6 = C-F9: `f_vals` crashes | confirmed defect (PyBADS-only option) | no | `c7c88ab`/`8ff10f5` | high |
| I-F7 ⊂ C-F12: multi-row `x0` is accepted, then crashes | confirmed port discrepancy | no (input) | never agreed (`c7c88ab`) | high |
| C-F12: `x0=None` with only `lb`/`ub` is accepted | confirmed port discrepancy | no (input) | **matched at `c7c88ab`, diverged in `9037851`** | high |
| C-F12 / I-F10: the output check of `non_box_cons` is weak | confirmed port discrepancy | no | matched at `c7c88ab`, weakened in `8e59038` | high |
| I-F10: a random `x0` that violates `non_box_cons` is refused | confirmed shared defect | no (`x0=None` with `non_box_cons`) | both since Nov 2022 (Python `d466948`, MATLAB `019f0b4`) | high |
| I-F10: the docstring example is MATLAB syntax, and the constraint's contract is unstated | confirmed defect (docs) | — | `c7c88ab`/`cdc2e0f` | high |
| **B1-K3** = I-F8 = C-F7: `status` is never set | confirmed defect; **contradicts KD-B1-8** | yes, every run; levels 0/1/2 | `8ff10f5` (2022-11-04); MATLAB's exitflag was never ported | high |
| B1-K3: `success` is always `True` | design question (KD-B1-8 leaves it open) | yes, every run | `8ff10f5` | high |
| I-F9 = C-F6: the result deep-copies the target and the constraint | confirmed defect (Python-only) | every run copies; the crash needs a callable that cannot be copied | `8ff10f5` | high |
| I-F11 = C-F11: `display` "final", "notify", "none" and case variants act as "iter" | confirmed port discrepancy | no | never agreed (`c7c88ab`) | high |
| C-F4 (= I-F12, part): `search_factor_min` is never read | confirmed port discrepancy | **yes**, D ≥ 2; levels 0/1/2 | never agreed (MATLAB `6c93629`; Python `c7c88ab`) | high |
| I-F12: `min_fun_evals`, `min_iter` | confirmed, inert (unread; MATLAB has no such options); disposition left to B2 | no effect | `c7c88ab` | high |
| I-F13 = C-F8: user `None` and MATLAB-style strings | design question | no | `c7c88ab` | high |
| **B1-K4** = C-F3: `tol_noise` = eps·tol_fun, where MATLAB has sqrt(eps)·TolFun | confirmed port discrepancy | **yes** (`uncertainty_handling=None`): nearly deterministic targets go from level 0 to 1 | never agreed (MATLAB `fd3f7a2` 2017; Python `c7c88ab`) | high |
| C-F10: MATLAB's `MaxFunEvals` and `ImprovementQuantile` checks are missing | confirmed port discrepancy | no | never ported | high |
| C-F13: `overhead` leaves out the time of the final samples and of merged repeats | confirmed port discrepancy (informational) | yes, levels 1/2 (level 2 for merged repeats) | never agreed | high on the mechanism |
| **B1-K5**: the random `x0` is drawn in the original box | **no longer holds** (`0c56d86`, W0-5); the new draw is MATLAB's | — | — | high |
| **B1-K6**: option descriptions are cut at `=` or `:` | confirmed, inert (cosmetic); **8 options, not 3** | — | `c7c88ab` | high |
| **B1-K7**: options without descriptions, and stray `'` | confirmed, inert (cosmetic) | — | `c7c88ab` (mostly carried from MATLAB's defopts) | high |
| **B1-K8**: `search_n_try` is a float | confirmed, inert | — | `c7c88ab` | high |

## 2. Per finding

### V1. B1-K1 = I-F1 = C-F2: half-bounded and mixed variables

**Lines.**
- Python: `pybads/bads/bads.py:533-544`.
- MATLAB has no counterpart:
  - `boundscheck.m` checks nothing of the kind;
  - `setupvars.m:27-38` only prints a caution about infinite bounds;
  - `bads.m:15-16` documents `LB(i) = -Inf` and `UB(i) = Inf` separately.

**Check.** `v_bounds.py`:
```
mixed bounded/unbounded  lb=[0,-inf] ub=[1,inf]:  PY refused: ValueError bads:HalfBounds ...
                                                  MAT accepted ... u(lb) = [-1.25 -inf]
half-bounded (lb=0, ub=inf):                      PY refused ...  MAT accepted ... u(lb) = [-1.105]
all unbounded / all bounded (controls):           both accepted, identical
```
- **What the code does:** the test is `any(finite(lb)) and any(~finite(ub)) or any(~finite(lb)) and any(finite(ub))`, taken across all variables.
- **What its own message and docstring promise:** a per-variable test ("while other coordinates may be bounded", `bads.py:64-66`, a sentence from `c7c88ab`).
- **With the test made per variable** (in memory in `v_mixed_run.py`), the mixed problem runs:
  ```
  as is: ValueError: bads:HalfBounds ...
  per-variable test: ran; x [0.3 1.9995] fval 2.59e-08 evals 59
  ```
- **W1-1's code still applies:** it replaces infinite bounds variable by variable (`gaussian_process_train.py:557-564`), so it works for the unbounded variable of a mixed problem.

**Dating.** The Python lines are unchanged since `c7c88ab` (2022-06-02). MATLAB never had such a check. The two never agreed.

**Agreement with the reports.** I agree with both reports, with one split.
- **(a) The mixed case** is a slip of the kind AGENTS.md describes: `and`/`or` between two `any`, instead of an elementwise test.
- **(b) The half-bounded refusal** is what the code is written to do. Its error message ("bounded only below/above are not supported") and the docstring both say so. MATLAB accepts such variables, and no record decides the difference. So (b) is a design question, not a slip.
- **Side effect of lifting (b) alone:** the effective bounds (V3) would move a half-bounded variable by a whole unit. They substitute a range of 1e3 for the infinite one. For lb = 0, ub = inf, `v_bounds.out` shows the warnings "Moving the initial points", "Moving plausible bounds" and "Expanding" printed before the refusal (`LB_eff` = 1).

**Disposition.**
- (a) **Fix.** Make the test `np.any(np.isfinite(lb) != np.isfinite(ub))` in `bads.py:535-540`.
- (b) **Decide the design.** Either support half bounds as MATLAB does, which needs V3 removed first, or keep the refusal and add it to the sheet.
- **Gate:** the fix only accepts inputs that are refused today. The fingerprint of unchanged runs suffices, plus one test with mixed bounds.

### V2. I-F2 = C-F5: scalar bounds

**Lines.**
- Python: `bads.py:181-184` (`atleast_2d(scalar)` gives (1, 1)) and `380-400` (the shape test demands (1, D)); docstring `bads.py:63`, "If scalars, the bound is replicated", since `c7c88ab`.
- MATLAB: `boundscheck.m:7-10` expands scalars (`42ae029`, 2017-05-12).

**Check.** `v_bounds.py`:
- `-1.0, 1.0` with D = 3: PY refused ("need to be of the same dimension D=3"), MAT accepted.
- `-5, 5, -2, 2` with D = 3: PY refused, MAT accepted.
- D = 1 works on both sides.

**Dating.** Python `c7c88ab` (touched in `d466948` and `a029e63`, messages only). The two never agreed.

**Disposition.** **Fix.** Broadcast scalar bounds to (1, D) before the shape test. At construction only, so the gate is the fingerprint.

### V3. C-F1 = I-F4, with I-F3: effective bounds

**Lines.**
- Python: `bads.py:450-531`, together with `181-184`.
- MATLAB: none. `boundscheck.m:12-16` sets `PLB = LB` and moves nothing; `setupvars.m:79-99` only gridizes `x0`.
- `boundscheck.m` is unchanged since `8ea0ecd` (2018-02-13).

**Checks.**

`v_bounds.py`, with MATLAB from the transcription:

| Case | PyBADS | MATLAB |
|---|---|---|
| `[-5,5]`, plb omitted | plb/pub ±4.99, u(lb) = −1.002 | ±5, u(lb) = −1 |
| `[1,10]`, plb omitted | linear (9.991/1.009 < 10) | **log** |
| `[1e-3,1e3]`, plb omitted | plb 1.000999; x0 = 1 moved to 1.000999 | plb 0.001; x0 = 1 |
| `[1e-3,1e3]`, plb = 1e-2, x0 = 1e-2 | x0 and plb 1.000999 | x0 0.01 |
| x0 = lb = −5, plb = −2 | x0 and plb −4.99 | x0 = −5 |
| `[0,1]`, plb = 1e-4, pub = 5e-4 | refused (StrictBounds) | accepted |
| `[-1000,1]`, plb = 0, pub = 0.99 | refused | accepted |

`v_effrun.py` compares PyBADS as it is with "no-move", which restores the user's x0/plb/pub in memory after the checks. That is MATLAB's behaviour at this step; the rest of the run is PyBADS's.

| Problem | Variant | First evaluation | First eval with f < 1e-4 | Result |
|---|---|---|---|---|
| 1-D log, x0 = optimum 0.01 | pybads | 1.001 (f = 4) | 11 | fval 5e-12 in 31 evals |
| 1-D log, x0 = optimum 0.01 | no-move | 0.01 (f = 0) | 1 | 20 evals |
| 3-D log, plb omitted | pybads | — | 33 | fval 4.9e-06 in 74 evals |
| 3-D log, plb omitted | no-move | — | 44 | fval 2.6e-07 in 75 evals |
| 2-D linear, optimum at lb | pybads | — | — | identical evaluation counts |
| 2-D linear, optimum at lb | no-move | — | — | identical evaluation counts |

These are single seeds. The runs converge; what changes is the start, the plausible box, the log decision, and whether the problem is accepted at all.

**I-F3.** It holds as a description: the test at `bads.py:506-519` compares x0 with `LB_eff`/`UB_eff`, not with plb/pub as its comment says. **I disagree with the direction the report proposes** (test against plb/pub):
- MATLAB never expands the plausible box for an x0 that lies outside it.
- For x0 = 0.02, 1.0 and 9.0 in `[0,10]` with plausible box `[2,8]`, PyBADS already equals MATLAB exactly (`v_bounds.out`: same plb, same u0).
- PyBADS differs only for x0 inside the margin (0.005 and 9.995).
- So the discontinuity I-F3 describes comes entirely from the effective-bounds block. Correcting the comparison would add a new departure from MATLAB.

**Dating.** The logic dates from `c7c88ab`; `8bb3d59` and `cdc2e0f` changed messages only. The `plb`←`lb` default in `__init__` dates from `9037851`. The two never agreed. The block's identifiers (`TooCloseBounds`, `InitialPointsTooClosePB`) suggest it came from VBMC's `boundscheck`. That is unverified: neither VBMC nor PyVBMC is in the environment.

**Tests.** No test reaches the margin: `test_bads_optimization.py` uses ±100 with plausible box [−8, 12], and the 1-D tests ±10 with [−5, 5]. The random-x0 test in `test_run_control.py:207` uses bounds outside the margin too.

**Disposition.** **Fix.** Remove the effective-bounds block (`bads.py:450-531`) and keep MATLAB's checks: the order test and x0 within the hard bounds. Do not "correct" the I-F3 comparison.
- **Default runs:** this changes default runs of every bounded problem that omits `plb`/`pub` or has x0 or plb within the margin, at all levels.
- **Gate:** a population comparison with a configuration that reaches it. That means bounded problems with `plb`/`pub` omitted, a log-scaled variable, and x0 on a bound. It also needs the fingerprint, which should stay unchanged if its runs keep clear of the margin; I could not check that, since `dev/` is out of bounds for me.

### V4. I-F5: the transform's self-test tolerance

**Lines.**
- Python: `variables_transformer.py:207-231` (`numeps = 1e-6`).
- MATLAB: `transvars.m:30` (`NumEps = 1e-6`) and `169-178`, the same four absolute tests.

**Check.** `v_transform.py` and `v_transform2.py`. Over 200 random draws each, PyBADS and the transcription of `transvars.m` refuse the same bounds:
```
log, ub ~ 1e9   PY refused 83  MAT refused 83  disagree 0
lin |b|~1e10:   PY refused=18  MAT refused=18  disagree=0
lin |b|~1e11:   PY refused=36  MAT refused=36  disagree=0
x=9.53e+10: |ginv(g(x)) - x| = 1.53e-05 (tolerance 1e-6), spacing = 1.53e-05
```
- The reviewer's case (lb = −9.53e10, ub = 9.53e10, plb = −2.06, pub = −0.74) is refused on both sides.
- The two maps themselves are bit-identical on a mixed linear/log/unbounded case of 2000 points.

**Dating.** MATLAB `6c93629` (2017-03-14); Python `c7c88ab`. The two have always agreed. **The internal reviewer's "unsure" is settled as shared.**

**Disposition.** **Decide the design**: a relative tolerance would be a deliberate departure from MATLAB. If the decision is to change it, the fix is `variables_transformer.py:218-227`. It only accepts bounds that are refused today, so the gate is the fingerprint. Low priority.

### V5. B1-K2 = I-F6 = C-F9: `fun_values` and `f_vals`

**`fun_values`** (`bads.py:740-791`). MATLAB imports these evaluations: `setupvars.m:126-167` (last changed `4a5a3d7`, 2017), then `funlogger`.

- **Check** (`v_inputs.py`): either form of `fun_values` raises "The truth value of an array ... is ambiguous" at construction.
- **Three independent defects:**
  - `not np.isreal(X)` on an array (`bads.py:760-761`);
  - `range(len())` raises TypeError (`bads.py:787`);
  - `self.function_logger` is created at `bads.py:290`, after `_init_optim_state_` at 286.
- **Dating:** validation from `c7c88ab`, the loop from `8e59038` (2022-06-03). It never worked.
- **Default runs:** not reached; the default `{}` fails `len(fun_values) != 0`.

**`f_vals`** (PyBADS-only; KD-B1-4 lists it as "read by code").
- **Check** (`v_runs.py`):
  ```
  cache_active: True
  optimize raised ValueError: Unknown format code 'f' for object of type 'str'; evaluations 2
  ```
- **Why:** `cache_active` selects a format with **8** placeholders (the reports say 7), filled with 6 values at level 0 (`bads.py:2854-2894`).
- **The values are unused anyway:** they go to `optim_state["cache"]`, which nothing reads.
- **Dating:** `c7c88ab`/`8ff10f5`.

**Disposition.**
- **`fun_values`:** fix it, importing into the logger after the logger is created, as MATLAB does. Or refuse a non-empty value until it is ported.
- **`f_vals`:** remove or refuse it, since it duplicates `fun_values`, and correct KD-B1-4.
- **Gate:** neither is reached at default, so the fingerprint.

### V6. C-F12 and I-F7: input validation

**Multi-row `x0`** (I-F7).
- **Check** (`v_inputs.py`, `v_runs.py`): a (2, 2) x0 is accepted (u0 (2, 2), `self.u` (4,)). `optimize()` then raises "operands could not be broadcast together with shapes (1,2) (1,4)".
- **Without bounds**, `plb`/`pub` are estimated from the set (`bads.py:340-378`). MATLAB BADS has no such branch.
- **MATLAB refuses the input:** `boundscheck.m:18-27`.
- **Dating:** `c7c88ab`; never agreed.
- **Disposition:** fix. Refuse N0 > 1 and drop the estimation branch. Construction only, so the fingerprint.

**`x0=None` with only `lb`/`ub`.**
- PyBADS accepts it and draws in the moved box (plb −4.99). MATLAB errors (`bads.m:331-342`), and so does PyBADS's own "Raises" section.
- **Dating:** at `c7c88ab` the Python raised, as MATLAB does. `9037851` (2022-09-22) moved the `plb`←`lb` default ahead of the test, so the Python once matched MATLAB.
- **Disposition:** fix. Test `x0 is None` before the default. The gate is the fingerprint.
- **List-valued plb with `x0=None`:** raises AttributeError. The docstring types are `np.ndarray`, so this is minor.

**`non_box_cons` output check** (`bads.py:546-556`).
- **What it accepts:** (N,), (N,1) and (1,N).
- **What it mishandles:** a scalar or bool output gives AttributeError; an (N,2) output passes the check and fails at x0 with "truth value ... ambiguous".
- **MATLAB** requires 2×1 inside try/catch, with a clear error (`setupvars.m:11-25`, `e2920ce` 2017).
- **Dating:** MATLAB-like at `c7c88ab` (`shape[0] != 2 or shape[1] != 1`), weakened to `shape[0] != 2 and ndim == 1` in `8e59038`.
- **Disposition:** fix. Accept (N,) or (N,1), refuse anything else with a clear message. Fingerprint.

**Accepted as intended.** PyBADS also refuses a gridized x0 that violates the constraint (`bads.py:684-693`, `1bee482` "Check gridizied non-box-cons"). This is deliberate by its commit and has no MATLAB counterpart.

### V7. I-F10: a random `x0` that violates `non_box_cons`

**Check** (`v_inputs.py`): with the constraint sum(x²) > 1 on the plausible box [−1, 1]², construction is "refused for 4 of 20 seeds: [8, 10, 12, 13]; expected share 1 - pi/4 = 0.215".

**MATLAB errors in the same way.**
- `setupvars.m:83-85` draws the point and stores it as `optimState.x0`.
- `evalinitmesh.m:22-26` then raises "Initial starting point X0 does not satisfy non-bound constraints" (`5422a37`, 2017).
- So this is **a shared defect**, not the discrepancy the report proposes.
- **Dating:** MATLAB's random draw since `019f0b4` (2022-11-14); before it, the midpoint. Python's random draw since `d466948` (2022-11-10); the check at `bads.py:276-284` since `c7c88ab`.

**The documentation part is confirmed.**
- The docstring's example `lambda x: np.sum(x.^2,1)>1` (`bads.py:81`) is MATLAB syntax.
- The N×D-in, (N,)-out contract is not stated.

**Disposition.**
- **The refusal: decide the design.** Options: redraw within a bound on attempts, start from the best feasible point of the initial design, or keep MATLAB's error but document it. Only runs with `x0=None` and `non_box_cons` are affected; the gate is the fingerprint plus a test.
- **The docs: fix** the docstring.

### V8. B1-K3 = I-F8 = C-F7: `status` and `success`

**Lines.**
- Python: `optimize_result.py:62-84` (`"status"` whitelisted) and `159-162` (`success = True  # TODO`).
- `bads.py:1436-1464`: `exit_flag` is commented out at 0 and 1 and assigned, unused, at 2.
- MATLAB: `bads.m:1062-1083` (exitflag 0/1/2; `6c93629`/`0c2bb80`/`7dcdc2b`, 2017).

**Check** (`v_runs.py`):
```
max_fun_evals=30: success=True message="...reached maximum number of function evaluations..."
  'status' in r: False; ... keys missing from _keys: ['status']
  r.status -> AttributeError status
```

**Dating.** `8ff10f5` (2022-11-04) created `OptimizeResult` with `#'status',` commented out and the TODO. The exit flag's assignments have been commented out since `c7c88ab`. MATLAB's exitflag was never ported.

**The sheet.** **This contradicts KD-B1-8**, which lists `status` among the returned fields. `success` is left open by KD-B1-8.

**Disposition.**
- **`status`: fix**, and correct KD-B1-8. Set `status` to MATLAB's exitflag: 0 for the budget, 1 for the mesh, 2 for the stall, and a value of its own for the output function.
- **`success`: decide the design.** The natural option is `success = status > 0`, which is MATLAB's and scipy's convention: False at the budget or iteration limit. The alternative is to keep it `True`, which is uninformative, and document it.
- **Gate:** no numerics move. The fingerprint, provided it does not hash the result's key set, which I did not check.

### V9. I-F9 = C-F6: the deep copy of `fun` and `non_box_cons`

**Lines.**
- Python: `optimize_result.py:98-99` and `182-186` (`copy.deepcopy` on every value).
- MATLAB: `bads_output.m:4` (`func2str(fun)`), so there is no counterpart.

**Check** (`v_runs.py`):
```
optimize raised TypeError: cannot pickle '_thread.lock' object after 20 evaluations; b.x = [ 0.00537109 -0.00146484]
callable object: result['fun'] is c: False ; plain function kept by reference: True
```

**Dating.** `8ff10f5`.

**Disposition.** **Fix.** Store `fun` and `non_box_cons` by reference in `__setitem__`. There is no numeric effect, so the gate is the fingerprint.

### V10. I-F11 = C-F11: display levels

**Lines.**
- Python: `bads.py:224-232` compares exact strings.
- MATLAB: `bads.m:312-328` lower-cases the first three letters: notify → 1, none/off → 0, iter/all → 3, final → 2.

**Check** (`v_options.py`): 'final', 'notify', 'none', 'OFF' and 'Iter' all give logger level 20 (INFO), the full display. The `.ini` description lists "notify" and "final".

**Dating.** Python `c7c88ab`, MATLAB 2017; never agreed. KD-B2-3 settles the logger mechanism only.

**Disposition.** **Fix** the mapping. It is not numerical: fingerprint.

### V11. C-F4 and I-F12 (part): `search_factor_min` is never read

**Lines.**
- Python: `.ini:101`; `bads.py:2759-2767` multiplies by sqrt(0.5) on every failure with no floor. A grep finds 0 reads of the option.
- MATLAB: `bads.m:1366`, `max(SearchFactorMin, searchfactor*SearchScaleFailure)` (`6c93629`). The factor scales the ES covariance on both sides (`searchES.m:105`, `es_search.py:128`).

**Check** (`v_searchfactor.py`: seed 1, 200 evaluations; the floor is applied in memory by wrapping `_update_search_stats_`):
```
rosenbrock D=2   pybads  37 searches  11% <0.5  min 0.3536  fval 2.5177e-06
rosenbrock D=2   floor   37            0%       0.5000       1.3242e-06
ellipsoid D=6    pybads/floor identical (0% <0.5)
rosenbrock D=6   pybads 138           27%       0.1768       4.0228
rosenbrock D=6   floor  106            0%       0.5000       0.24675
```
These are single runs; the fval difference is not evidence of direction.

**Reach.** At default options for D ≥ 2: `search_n_try` ≥ 4, so the third failed search of a round runs below 0.5. At D = 1, the round resets first.

**Dating.** The update without the floor, and the unread option, both date from `c7c88ab`. The two never agreed. The option is not on KD-B1-5, and must not go there, since MATLAB reads it.

**Disposition.** **Fix.** Add the floor in `bads.py:2761-2764`. This changes default runs, so the gate is a population comparison at default options; the benchmark reaches it.

**`min_fun_evals` and `min_iter`** (I-F12). They have 0 reads, and MATLAB has no such options (grep of the MATLAB tree finds none). They are PyVBMC leftovers, missing from KD-B1-5(d). Per the brief, their disposition as termination options is left to B2's verifier.

### V12. I-F13 = C-F8: user `None` and wrong types

**Check** (`v_options.py`):
- `nonlinear_scaling=None` gives log flag `[False]`, where the default gives `[True]`.
- `nonlinear_scaling='off'` gives `[True]`.
- `uncertainty_handling='off'` or `'no'` gives level 1.
- `tol_mesh=None` raises TypeError at construction; `max_fun_evals=None` raises TypeError in `optimize`.
- MATLAB replaces an empty value with the default (`setupoptions.m:5-9`) and evaluates strings with `evalbool` (`22-50`), so 'off' is false there.

**Classification.** KD-B1-3 settles that user values are used verbatim, which covers the strings, and leaves `None` open. That makes the whole a **design question**:
- (i) treat `None` as "use the default", except for the options where `None` is a meaningful value;
- (ii) validate the types of boolean and numeric options and refuse the rest;
- or both.

**Dating.** `options.py:48-51` since `c7c88ab`.

**Disposition.** **Decide the design.** Not reached at default, so the fingerprint.

### V13. B1-K4 = C-F3: `tol_noise`

**Lines.**
- Python: `advanced_bads_options.ini:13`, read at `bads.py:1037`. Its value is 2.22e-19.
- MATLAB: `bads.m:195`, `sqrt(eps)*options.TolFun` = 1.49e-11 (`fd3f7a2`, 2017-03-29). It is read at `evalinitmesh.m:43`; `TolFun` precedes `TolNoise` in `evalfields`, so the product uses the user's `TolFun`.

**Check** (`v_tolnoise.py`, D = 2, 100 evaluations; the MATLAB threshold is set through the option):

| Target | Repeat difference | Threshold | Level | Evals | Outcome |
|---|---|---|---|---|---|
| shuffled sum | 2.22e-15 | PyBADS 2.22e-19 | 1 | 100 | \|x\| 7.2e-03 |
| shuffled sum | 2.22e-15 | MATLAB 1.49e-11 | 0 | 55 | \|x\| 8.0e-04 |
| jitter 1e-12 | 2.54e-12 | PyBADS 2.22e-19 | 1 | 100 | fval−1 2.3e-04 |
| jitter 1e-12 | 2.54e-12 | MATLAB 1.49e-11 | 0 | 55 | fval−1 1.9e-09 |

A target whose only nondeterminism is the order of a summation runs as noisy in PyBADS.

**Dating.** Python `c7c88ab` (`np.spacing(1.0) * self.get("tolfun")`, renamed in `cdc2e0f`). The two never agreed: this is a transcription slip of `sqrt(eps)`.

**Disposition.** **Fix.** Use `np.sqrt(np.spacing(1.0)) * self.get("tol_fun")`, and correct the description's typo "variabitility".
- **Which runs move:** only targets whose repeat at x0 differs by more than 2.2e-19 and at most 1.5e-11.
- **Gate:** the fingerprint (deterministic targets that repeat bit for bit and noisy ones are unaffected), plus a configuration with a nearly deterministic target. A population comparison only if the benchmark has such targets.

### V14. C-F10: missing option checks

**Check** (`v_options.py`):
- `max_fun_evals=0` or `−5`: "ValueError: cannot convert float NaN to integer".
- `max_fun_evals=30.5`: runs, 31 evaluations.
- `improvement_quantile=0.9`: runs with no warning.

MATLAB errors or warns at `setupoptions.m:71-78` (2017–2018). The Python never had these checks (`git log -S` finds none).

**Disposition.** **Fix** in `__init__`. Validation only, so the fingerprint.

### V15. C-F13: `overhead`

**Lines.**
- Python: `function_logger.py:384-398` (the `record_duplicate_data=False` path) and `424-431` (merged level-2 repeats) do not add to `total_fun_eval_time`; only `438-440` does.
- MATLAB: `funlogger.m:130` adds `t` in both 'iter' and 'single' (`603da99`, 2017-04-24). The noise test is excluded on both sides.

**Check** (`v_runs.py`, noisy target sleeping 2 ms per call, 60 evaluations):
```
total_fun_eval_time 0.1093 s, sum of call times 0.1310 s, of the last 10 calls 0.0219 s
overhead reported 2.586; with every call counted 1.990
```

**Dating.** Python accounting since `c7c88ab`; the duplicate paths touched in `8ff10f5` and `9915bbf` without adding to the total. The two never agreed.

**Disposition.** **Fix.** Add `fun_eval_time` to the total on both duplicate paths. Informational only; the gate is the fingerprint.

### V16. B1-K5: the random `x0`

**Not covered by a finding.** Both reports describe the current draw only in their answers to the first questions.

**No longer holds:** `0c56d86` (W0-5, 2026-09-26) replaced the draw in the original box.

**The replacement is MATLAB's draw.** `v_x0draw2.py` feeds both sides the same uniform numbers, with bounds that the effective bounds leave alone:
```
untouched: log flags [ True False  True] / [ True False  True]; evaluated start: u identical 200/200, x identical 200/200, max rel diff 0
plb moved by F1: ... u identical 200/200, x identical 0/200, max rel diff 97
   variable 1 of the evaluated start: PY range [1.019, 98.88], MAT range [0.01037, 97.78]
```
- **Where it still differs:** where V3 moves `plb`, the draw is made in the moved box.
- **`result["x0"]`** is the ungridized draw, not the evaluated point (e.g. 18.797 against 18.776, `v_x0draw.out`).
- **`v_x0draw.py`** is a first attempt whose bounds let V3 move `plb`; it is superseded by `v_x0draw2.py`.

**History.**
- At `c7c88ab` the Python took the midpoint of the original box, while MATLAB took the midpoint in u (`a3d7b70`, 2017). They agreed for linear variables only.
- Both moved to random draws in the same week: Python `d466948` (2022-11-10, original box), MATLAB `019f0b4` (2022-11-14, transformed box).

**Disposition.** **Correct the record**: mark B1-K5 as changed by `0c56d86`. No action beyond V3.

### V17. B1-K6: descriptions cut at `=` or `:`

**Partial coverage.** Both reports mention it in their answers to Q1, naming 3 options.

**Check** (`v_options.py`). **Eight** descriptions are cut, not three:
- `noise_size`, `periodic_vars`, `stobads_frame_size_scaling_power`;
- `gp_samples` ("Hyperparameters samples (0");
- `gp_sample_widths`, `stable_gp_samples`, `upper_gp_length_factor`, `temperature`.

**Where it shows.** Only `Options.descriptions` and `str(options)` (`options.py:118`, `200`; no other reader). The docs include the `.ini` files `:literal:` (`docsrc/source/api/options/bads_options.rst`), so the docs are unaffected. The parser dates from `c7c88ab`.

**Disposition.** **Fix** (cosmetic) in `_read_config_file` (`options.py:214-227`): use `delimiters=("=",)` and rejoin key, `=` and value for comment lines. No numerics.

### V18. B1-K7: missing descriptions and stray quotes

**Partial coverage.** The internal report mentions it under Q1 (cosmetic).

**Check** (`v_options.py`).
- **No description line:** `n_basis`, `search_factor_min`, `gp_cov_fun`, `use_effective_radius`, `gp_fixed_mean`, `hessian_method`, `hessian_alternate` (inline comments only), `hedge_beta`, `hedge_decay`.
- **Stray quotes:** 36 descriptions end in a stray `'` or `';`, MATLAB's closing string quote.
- **Carried from MATLAB:** its `defopts` also has no description for `Nbasis`, `SearchFactorMin`, `gpFixedMean`, `UseEffectiveRadius` (`'yes %'`) or the three `Hedge*`.

**Dating.** `c7c88ab`.

**Disposition.** **Fix** (cosmetic). The docs show these lines verbatim.

### V19. B1-K8: `search_n_try` is a float

**Coverage.** Both reports' defaults tables note it.

**Check** (`v_options.py`): `np.float64(3.0/4.0/6.0/20.0)` at D = 1/2/6/20, and `optim_state["search_count"]` starts as the same float.

**Why it is inert.** Every read compares it with an integer count (`bads.py:796-797`, `1352`, `1370-1374`, `1833-1834`, `2772`), which is exact for whole floats. MATLAB's value is a double too.

**Dating.** `c7c88ab`.

**Disposition.** Keep, or cast it with `int()` when the file is next touched. No numerics.

## 3. New items met while verifying (unverified beyond the small checks named)

- **N1. Grid rounding.** `force_to_grid` (`search/grid_functions.py:8-12`, `c7c88ab`) uses `np.round`, which rounds halves to even; MATLAB's `force2grid.m` uses `round`, which rounds them away from zero. `v_round.py`: for x0 = 1 in plausible box [−2048, 2048] (u = h/2 exactly), PyBADS starts at x = 0 and MATLAB at x = 2. Exact halves need dyadic inputs, and how often they occur for search or poll points inside a run is not measured. This is not on the sheet.
- **N2. `fsd` type.** `fsd` comes back as the `int` 0 in deterministic results, although `_init_optimization_` sets 0.0 (`v_runs.out`). It presumably comes from the incumbent's value read back from `iteration_history` (`bads.py:1511` and `1534`); I did not trace it.
- **N3. V1 and V3 together.** V1(b) cannot be lifted without V3: with a half-bounded variable, the effective bounds substitute 1e3 for the infinite range and move x0 and `plb` by a whole unit.
- **N4. Infinite `x0`.** An `x0` containing ±inf with a finite bound is refused by PyBADS (`bads.py:443`), where MATLAB replaces it by a random point (`setupvars.m:79`). This is a reading only, noted in C-F12's section 2; it is minor.

Scripts, in `/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/review/B1_verifier/`, each with its `.out`: `common.py`, `v_bounds.py`, `v_transform.py`, `v_transform2.py`, `v_options.py`, `v_searchfactor.py`, `v_inputs.py`, `v_runs.py`, `v_tolnoise.py`, `v_x0draw.py`, `v_x0draw2.py`, `v_effrun.py`, `v_mixed_run.py`, `v_round.py`.
