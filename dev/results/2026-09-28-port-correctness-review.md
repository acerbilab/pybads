# The port correctness review: consolidated ledger

The independent review of PyBADS against MATLAB BADS v1.1.3 (`74919c0`)
and against its own specification, from 2026-09-25 to 2026-09-28, planned
and logged in
[`plans/port-correctness-review.md`](../plans/port-correctness-review.md).
Fresh reviewer agents read the package in seven slices, on two tracks
(internal correctness, and a line-by-line comparison with MATLAB BADS),
with a reviewer of the MATLAB commits since the port began (M), one of
Sto-BADS (S) and a third reader of the improvement, the acquisition and the
geometry (O); a fresh verifier checked the reviewers' findings before they
entered the ledger of their wave, and the PI ruled on the rows before the
wave's fix pass. Rows found while fixing or doublechecking were checked by
the orchestrator or the doublecheck, and three rows of wave 0 were fixed by
#71 before the wave. This note
consolidates the five ledgers, `verification/wave0.md` to `wave4.md` under
[`experiments/port_review_20260925/`](../experiments/port_review_20260925/README.md),
which stay as written and hold the evidence, the dating from both
histories and the gates of every row. The deliberate differences that the
review settled are catalogued in
[`pybads/bads/README.md`](../../pybads/bads/README.md).

## Outcome

The five ledgers hold 173 rows. 121 changed the code: a defect or a
discrepancy fixed, most of them toward MATLAB BADS, or a value that failed
obscurely refused with a message when `BADS` is created (W1-24's fix is in
gpyreg, not yet released; W2-44, kept in wave 2, was fixed by W4-28, and
the options of W2-35 were removed by #87). 28 kept a behavior as it was,
most of them a deliberate difference from MATLAB BADS, in the catalogue, or
a behavior that PyBADS shares with MATLAB BADS, in
`matlab_side_defects.md`. 22 closed without a change of behavior: a record,
a comment or a description corrected, a finding that was not a defect, or
one that an earlier fix had removed. Two stay open, W0-12 and W0-13, the
design of Sto-BADS's success rule, off by default, which waits for a
population with `stobads=True` (`dev/TODO.md`).

The survey's candidate table has no open row: the review closed its 31 open
rows, which the ledgers count as 13 in wave 1, 6 in wave 2, 11 in wave 3
and 2 in wave 4, the row on the `_save_gp_stats_` calls closed in waves 1
and 2. On the MATLAB side, the review found 14 defects that PyBADS shared
and fixes, 8 behaviors that both share and PyBADS keeps, 4 defects of
MATLAB BADS that PyBADS does not share (one of them found by the
measurement of the rank-1 update, after wave 4), and one question that only
MATLAB can answer. On the benchmark, the deterministic configurations end
with an equal or smaller error, but for two spheres, whose errors grow far
below their tolerance, and the noisy configurations stop earlier ("Net
effect on the benchmark").

## How the review ran

| Wave | Slices | PyBADS read | Reviewers' findings | Ledger rows | Fixes on `dev-next` | Doublecheck |
|---|---|---|---|---|---|---|
| 0 | the preparatory agent (the sheet of known differences, the counterpart map); M; S | `ab4dded`, verified at `95da7f1` | M 10, S 7, the sheet's claims C1 to C8 | W0-1 to W0-21 | #71 (`95da7f1`), before the wave; #72 (`0c56d86`); W0-1 in #73 (`e004c79`) | — |
| 1 | B5 (the GP's training set and refits), B6 (the GP model) | `95da7f1`, the freeze | B5 14 and 12, B6 10 and 7 | W1-1 to W1-35 | #74 (`fef6c14`) | `4c38da8` |
| 2 | B1 (setup, options, bounds, transform, result), B2 (the main loop, termination, the final estimate) | `fef6c14` | B1 13 and 13, B2 13 and 13 | W2-1 to W2-47 | #76 (`8aecb6a`) | `68d4516` |
| 3 | B3 (the search), B4 (the poll, the mesh, the incumbent, the target) | `8aecb6a` | B3 9 and 10, B4 10 and 7 | W3-1 to W3-40 | #77 (`0d866e8`) | #79 (`6ceed6f`) |
| 4 | B7 (the function logger, the initial design, the utilities), O | `0d866e8` | B7 11 and 7, O 1 | W4-1 to W4-30 | #80 (`81385ac`) | #81 (`a4dcd65`) |

The findings of a slice's two reports are counted internal first; a row
of a ledger can merge several findings, and the later rows of each wave
were found while verifying, fixing or doublechecking. After wave 4, #84
(`7969783`) fixed the minor items of the review that needed no choice of
the PI, and moved no result, and #87 carries the PI's rulings on the rest
("Open ends"). Wave 1's gpyreg side is in gpyreg:
acerbilab/gpyreg#56 (W1-25's switch, off by default) and #57 (W1-24),
merged into gpyreg's `main` and not released, and #58, a fix of the switch
that wave 1's doublecheck found; PyBADS keeps gpyreg 1.3.3 as its minimum
and its CI pin.

Each pull request was squash-merged, so the commits of a wave's branch that
the ledgers cite are reachable from `refs/pull/<N>/head` on GitHub, and
this note cites them with their pull request. A fix is one commit per row,
with a regression test and a changelog line where a user of the last
release can notice it; a fix that must move nothing kept the fingerprint of
`dev/scripts/fingerprint.py`, and one that moves results was gated by the
population comparison against the reference of its platform (`AGENTS.md`,
"Numerical gates"; each ledger's "Fix pass").

## Net effect on the benchmark

The code before the review against the code after it, on the `default`
suite of `dev/scripts/benchmark_targets.py` (18 configurations at BADS's
default budget, 500 D), with gpyreg 1.3.3.

**Windows, 100 seeds**
([`population_wave4_20260928`](../experiments/population_wave4_20260928/README.md)
at `a4dcd65` against
[`population_prereview_20260927`](../experiments/population_prereview_20260927/README.md)
at `ab4dded`; the difference includes #71): nine configurations are
flagged.

| Configuration | Median error, before → after | Median evaluations | Solved | Median paired log10 error ratio [95% CI] | Flagged tests |
|---|---|---|---|---|---|
| `ackley_D6` | 3.4e-4 → 2.0e-4 | 400 → 388 | 1.00 → 1.00 | -0.27 [-0.34, -0.24] | error, evaluations, signed-rank |
| `rosenbrock_D2` | 7.4e-6 → 2.5e-6 | 95 → 94 | 1.00 → 1.00 | -0.51 [-0.86, -0.25] | error, signed-rank |
| `sphere_D2` | 5.5e-7 → 1.5e-6 | 55 → 55 | 1.00 → 1.00 | +0.33 [+0.17, +0.58] | error, signed-rank |
| `sphere_D10` | 7.0e-8 → 9.7e-8 | 454 → 449 | 1.00 → 1.00 | +0.20 [+0.04, +0.32] | error, signed-rank |
| `ellipsoid_D10` | 7.3e-7 → 4.7e-7 | 629 → 668 | 1.00 → 1.00 | -0.16 [-0.29, -0.07] | evaluations |
| `ellipsoid_D3_homo` | 0.076 → 0.081 | 380 → 315 | 0.64 → 0.55 | +0.11 [-0.10, +0.23] | evaluations |
| `ellipsoid_D3_hetero` | 0.29 → 0.31 | 366 → 304 | 0.19 → 0.16 | -0.00 [-0.24, +0.14] | evaluations |
| `sphere_D3_homo` | 0.0078 → 0.0136 | 313 → 226 | 1.00 → 1.00 | +0.17 [-0.02, +0.34] | evaluations |
| `multisensory_s1_D6_homo` | 0.20 → 0.15 | 668 → 580 | 0.96 → 0.97 | -0.02 [-0.15, +0.07] | evaluations |

The deterministic configurations end with an equal or smaller error, but
for the two spheres, whose median errors grow while staying more than two
orders of magnitude below their tolerance (0.001); `ellipsoid_D10` takes 6%
more evaluations for a smaller error. The five configurations with noise
stop earlier, with 12 to 28% fewer evaluations, as W0-1's gate first
showed, and none of their error tests is flagged.

**Linux, 30 seeds**
([`population_linux_wave4_20260927`](../experiments/population_linux_wave4_20260927/README.md)
at `46af65a`, whose default runs are those of `dev-next` at `7969783`,
against
[`population_linux_gpfixes_20260925`](../experiments/population_linux_gpfixes_20260925/README.md)
at `97b2c66`, `ab4dded`'s package code but for #67, so that the difference
includes #67 and #71;
[`verification/close/linux_net_comparison.md`](../experiments/port_review_20260925/verification/close/linux_net_comparison.md),
from the records): four configurations are flagged, in the directions of
Windows. `ackley_D6`'s error falls (median 4.3e-4 → 1.7e-4, paired log10
ratio -0.41 [-0.49, -0.27]), and so does `rosenbrock_D2`'s (1.9e-5 →
2.1e-6); `ellipsoid_D10` takes more evaluations (median 624 → 668) and
`ellipsoid_D3_homo` fewer (374 → 312), with a larger median error, unflagged
(0.070 → 0.119, above its tolerance of 0.1; solved 0.67 → 0.43; paired
log10 ratio +0.20 [+0.07, +0.50]). No run crashed in either population.

## The rows

One table per wave, in the order of its ledger. *Slice* is the slice of the
code (in wave 0: M, S, and C1 to C8, the claims of the sheet that did not
check out); *Classification* is the verifier's, in the ledger's terms,
"inert" meaning that no run reaches the code or that it changes no result;
rows found while fixing, which the ledgers do not classify, carry the
orchestrator's classification; *Disposition* is the PI's ruling, with what
later work changed; *Fix or open item* gives the row's commit on its wave's
branch with the pull request that carried it to `dev-next` (whose squash
commit is in "How the review ran"), or the `dev/TODO.md` item that holds
the row; "—" or "the sheet" marks a row settled in the records alone.
`KD-*` names an entry of the catalogue in `pybads/bads/README.md`; the
ledger of each wave holds the locations, the evidence, the dating from both
histories and the gate.

### Wave 0

| Row | Slice | What | Classification | Disposition | Fix or open item |
|---|---|---|---|---|---|
| W0-1 | M | In noisy runs the re-estimate at the end of an iteration puts a GP of the iteration history in place of the working GP and changes it in place, so later re-estimates use drifted hyperparameters | port discrepancy | fixed: each iterate re-estimated from a copy of the working GP under its recorded hyperparameters; gated alone and investigated (`w01_investigation/`) | `7c73704` (#73) |
| W0-2 | M | The final `fsd` without target noise is normalized by `n`, MATLAB's by `n - 1` | port discrepancy | fixed before the wave | `7b50a3a` (#71) |
| W0-3 | M | The final estimate is recorded in the last iterate's slot, not the chosen iterate's | port discrepancy | fixed before the wave | `7b50a3a` (#71) |
| W0-4 | M | `iterations` is one below MATLAB's; `yval_vec` is `None` for a deterministic run and with `noise_final_samples = 0` | port discrepancy (`iterations`); intentional, missing from the sheet (`yval_vec`) | `iterations` fixed before the wave; `yval_vec` kept, on the sheet (KD-B1-8) | `7b50a3a` (#71); `010eeb4` (#72) |
| W0-5 | M | A random `x0` is uniform in the original plausible box, MATLAB's in the transformed one | port discrepancy | fixed | `6c830c9` (#72) |
| W0-6 | M | The substitution of non-finite targets indexes the full array with an index into the finite subset, and `add_and_update_gp` has no penalty | port discrepancy, unreachable | fixed as MATLAB's `gpupdate.m` | `abf9814` (#72) |
| W0-7 | M | The starting GP mean is the median of the lowest `round(0.8 N)` values, MATLAB's of `ceil(0.8 N)` | port discrepancy | fixed in wave 1's pass | `3236c2f` (#74) |
| W0-8 | M | The nearest-neighbour training set is sorted with the unstable `np.argsort` | port discrepancy (row order only) | fixed in wave 1's pass | `d1caf6b` (#74) |
| W0-9 | M | With `uncertainty_handling=True` and no target noise, `gp.s2` holds NaN, which gpyreg ignores | confirmed, inert | fixed: the logger holds noise SDs only when the target returns them | `90d1101` (#72) |
| W0-10 | S | The Sto-BADS poll decides from the last evaluated point, so a success followed by another point is discarded | defect | fixed | `79c83a7` (#72) |
| W0-11 | S | A NaN estimate counts as uncertain, so `opp_stobads` can move the incumbent to a point with a NaN value | defect | fixed: a non-finite estimate is a failure | `491596e` (#72) |
| W0-12 | S | Sto-BADS's threshold takes the GP's SDs as epsilon, which do not shrink with the mesh as Sto-MADS requires | design question | open: decided after a population with `stobads=True` | `TODO.md`, "The uncertainty interval of Sto-BADS" |
| W0-13 | S | `opp_stobads` moves the search incumbent on any uncertain outcome, to worse estimates too, and widens the search | design question | open, with W0-12; since W4-15 an uncertain poll moves only to an improving point, and the search's move is not limited so | `TODO.md`, "The uncertainty interval of Sto-BADS" |
| W0-14 | S | `BADS.__init__` takes an undocumented 8th positional parameter before `options`, so MATLAB's argument order drops every option | defect | fixed: `gamma_uncertain_interval` keyword-only, options in MATLAB's order | `34ed21e` (#72) |
| W0-15 | S | An empty search set stops the run with `UnboundLocalError`, and an ES search left without candidates with `IndexError` | defect | fixed: a failed search on every path | `5d65c53` (#72) |
| W0-16 | S | A successful poll appends the bound method `u_best.copy` to `u_success` | confirmed, inert | fixed | `d634e09` (#72) |
| W0-17 | C1 | The noise test also runs with `uncertainty_handling=False`, where MATLAB tests only when the option is empty | code differs from MATLAB | fixed | `1bbd6d1` (#72) |
| W0-18 | C2 | The Sobol design is doubled when `2**ceil(log2(n))` equals `D`, which the records did not say | record wrong; unexplained code | records corrected; the doubling kept by the ruling on W4-3 (KD-B7-1) | `010eeb4` (#72) |
| W0-19 | C3 | `plans/tooling-and-rng.md` says that the Sobol seed keeps MATLAB's derivation from `u0`; it does not | record wrong | the plan corrected; the seed replaced by one draw of the run's generator (W4-1) | `010eeb4` (#72) |
| W0-20 | C4-C7 | Comments name the wrong kernel and retry, features of PyVBMC's GP code that BADS lacks, and stale MATLAB files | record wrong | comments corrected | `a73a844` (#72) |
| W0-21 | C8 | The changelog's "Failed GP updates" reads as if the forced refit were MATLAB's | record ambiguous | entry edited | `010eeb4` (#72) |

### Wave 1

| Row | Slice | What | Classification | Disposition | Fix or open item |
|---|---|---|---|---|---|
| W1-1 | B5 | `optim_state["pub"]` and `["plb"]` hold each other's transformed bound, so `poll_scale` is capped in every unbounded variable and ES-ell is isotropic | port discrepancy | fixed; moves only `ellipsoid_D3_unbounded` | `6e22d32` (#74) |
| W1-2 | B5 | `reset_gp` is cleared only at the end of a poll, so every later search and poll step rebuilds the GP | port discrepancy | fixed: cleared at the rebuild it asks for; amended by W3-29 (after a poll that moves, MATLAB rebuilds at every search until a poll that does not) | `d3640ab` (#74); `0b7add3` (#77) |
| W1-3 | B5 | The calibration statistics store the latent SD and replace near-zero SDs by 1e-6; MATLAB stores the observation's SD | port discrepancy | fixed, moves results | `d883cf9` (#74) |
| W1-4 | B5 | The statistics count is off by one: one statistic reads as none, and the periodic refit waits one evaluation | port discrepancy | fixed, moves results | `e420486` (#74) |
| W1-5 | B5 | The χ² bounds of the calibration test at n < 3 are half MATLAB's | port discrepancy | fixed | `bd49445` (#74) |
| W1-6 | B5 | SciPy's Shapiro-Wilk replaces `swtest.m`, which switches to Shapiro-Francia for leptokurtic samples | port discrepancy (substitution) | kept, on the sheet (KD-B5-7) | — |
| W1-7 | B5 | The calibration test for n ≥ 3 tests normality only, blind to scale and location, as MATLAB's | design question, shared with MATLAB | kept as MATLAB's; `matlab_side_defects.md` | — |
| W1-8 | B5 | With `poll_training` off, the poll records a refit that it then cancels, as MATLAB's | shared defect | fixed: the poll neither performs nor records it (KD-B5-9); `matlab_side_defects.md` | `c9ebdc7` (#74) |
| W1-9 | B5 | The condition for adding the search point to the GP is always true (`&` binds before the comparisons) | confirmed, inert | fixed: the round's `search_count` | `91d3c33` (#74) |
| W1-10 | B5 | After a failed update and a successful retry, the geometry stays that of the refit that the GP no longer holds | port discrepancy, inert | fixed: the geometry from the hyperparameters the GP holds | `9ac1a47` (#74) |
| W1-11 | B5 | The retry with the previous hyperparameters, which MATLAB lacks, cannot succeed without a refit | design question | fixed: the retry only after a refit (KD-B5-2) | `f65bc91` (#74) |
| W1-12 | B5 | After each failed fit the noise's lower bound rises by the cumulative nudge, `noise_nudge[1]` is unread, and a fifth failure raises | port discrepancy | fixed, moves results | `776b70d` (#74) |
| W1-13 | B5 | `_robust_gp_fit_` has no exit for a fit whose every try fails | port discrepancy | fixed: the best start, exit flag -1, as MATLAB | `446c443` (#74) |
| W1-14 | B5 | The retry cuts outliers with NumPy's linear percentile, has no stop below D points, and exits with 0 | port discrepancy (the percentile, the stop); confirmed, inert (the exit flag) | fixed, moves results | `3ddc047` (#74) |
| W1-15 | B5 | A refit starts from gpyreg's design of prior draws, not from MATLAB's local runs | design question | kept, on the sheet (KD-B5-6) | — |
| W1-16 | B5, B6 | The prior sampler exponentiates each prior's centre and SD, and a block without a prior raises | port discrepancy | fixed, moves results | `1c7200b` (#74) |
| W1-17 | B5 | At D = 1 the training set's distances are in units of 1, as MATLAB's, not of the fitted length scale | design question, PyBADS matches MATLAB | a defect shared with MATLAB (PI), fixed: the fitted length scale at every D (KD-B5-10); `matlab_side_defects.md` | `cfacb98` (#74) |
| W1-18 | B5 | `len_scale += len_scale + ...` doubles the running sum | confirmed, inert | fixed: MATLAB's weighted sum | `3cae0e1` (#74) |
| W1-19 | B5 | The `init_N` schedule divides 0 by 0 when the budget equals the initial design | defect (PyBADS only) | fixed | `e041de0` (#74) |
| W1-20 | B5 | The retry loop of the initial fit has no cap | defect | fixed: ten tries, then `RuntimeError` | `b6a4fd5` (#74) |
| W1-21 | B5 | After a failed fit the retry reads the failed GP's bounds and, with the slice sampler, its data | defect with the slice sampler, inert at default | fixed | `5956c4b` (#74) |
| W1-22 | B6 | The noise prior's new centre is computed at each rebuild and never written back | port discrepancy | fixed, moves results | `ee9d5d6` (#74) |
| W1-23 | B6 | The constant mean is bounded all run by the initial design's range while its prior is re-centred, which pins the mean at a bound | port discrepancy | fixed: unbounded, as MATLAB; moves results | `172df00` (#74) |
| W1-24 | B6 | gpyreg's normalization of a prior far outside its bounds underflows, and the log prior is infinite | defect (gpyreg's), reached through W1-23 | fixed in gpyreg, merged and not released; W1-23 closes PyBADS's route | acerbilab/gpyreg#57; `TODO.md`, "gpyreg releases after 1.3.3." |
| W1-25 | B6 | gpyreg's Cholesky retries multiply the noise and keep the multiplier in the posterior; MATLAB treats a failure as an error | port discrepancy (substituted library) | kept (KD-B6-6); gpyreg's switch (acerbilab/gpyreg#56), off in PyBADS after its measurement | `TODO.md`, "gpyreg's inflation of the GP noise (W1-25), after wave 1's fixes." |
| W1-26 | B6 | A plateau of the initial values stops the run in `_gp_hyp` (zero SD of the mean's prior) | port discrepancy; needs MATLAB (the rebuild case) | fixed: a positive fallback, and a rebuild keeps the previous prior (KD-B6-2) | `cd1831f` (#74) |
| W1-27 | B6 | PyBADS fits a GP on the initial design, where MATLAB only defines it | port discrepancy | kept (PI), on the sheet (KD-B6-5) | — |
| W1-28 | B6 | `gp_cov_prior="ard"` is not ported, and any value is accepted | port discrepancy (unported) | refused with a message (KD-B6-7) | `64616af` (#74); `TODO.md`, "`gp_cov_prior="ard"`." |
| W1-29 | B6 | The output scale's prior is centred with the SD of ddof 0 | port discrepancy, negligible | fixed | `1d03801` (#74) |
| W1-30 | B6 | The noise's upper bound is a log SD of 5, as MATLAB's, whatever the target's scale | design question, shared with MATLAB | kept, with a warning above e^5 (KD-B6-9); `matlab_side_defects.md` | `54e6424` (#74) |
| W1-31 | B6 | The effective radius is said not to match gpyreg's kernel | not a defect | a comment naming the convention | `2794e94` (#74) |
| W1-32 | B6 | `fit_lik=False` stops the run; MATLAB refuses fixed noise too | confirmed, inert | refused when `BADS` is created, with MATLAB's message (KD-B6-8) | `238afad` (#74) |
| W1-33 | B6 | `upper_gp_length_factor` sets bounds that the next lines overwrite | confirmed, inert | branch removed; the option removed by #87 (KD-B1-5) | `0889426` (#74); `b18382e` (#87) |
| W1-34 | B6 | 12 names of `gp_mean_fun` are accepted, 9 cannot be built, and `negquad` is wrong for a minimizer | defect (the names); design question (`negquad`) | only `zero` and `const` accepted | `43138f2` (#74) |
| W1-35 | B2 | W0-1's re-estimate crashes a noisy run whose rebuild fails (found while fixing) | defect | fixed: a past iterate gets NaN, the current keeps its estimate (KD-B2-4) | `463312f` (#74) |

### Wave 2

| Row | Slice | What | Classification | Disposition | Fix or open item |
|---|---|---|---|---|---|
| W2-1 | B1 | The half-bounds check tests every variable at once, refusing any mix of bounded and unbounded variables | port discrepancy | fixed: the test per variable | `f08fd6d` (#76) |
| W2-2 | B1 | A variable bounded on one side only is refused, where MATLAB accepts it with a caution | design question | supported, as MATLAB's | `8510ca8` (#76) |
| W2-3 | B1 | Scalar bounds are not replicated when D > 1 | port discrepancy | fixed | `4917bde` (#76) |
| W2-4 | B1 | `_bounds_check_` moves the plausible bounds and clamps `x0` into bounds 1e-3 inside the hard ones, which MATLAB never does | port discrepancy | fixed, moves results on the `bounds` suite (a worsening kept, reported to the PI); with N4, a start of ±inf | `a31a9be`, `a236eb7` (#76) |
| W2-5 | B1 | The transform's self-test has an absolute tolerance, which refuses valid bounds of large magnitude, as MATLAB's | shared defect | fixed: a relative tolerance (KD-B1-10); `matlab_side_defects.md` | `a2b8d38` (#76) |
| W2-6 | B1 | A non-empty `fun_values` stops `BADS()`: the option never worked | port discrepancy | refused with a message; the port left to `TODO.md` | `0ba1241` (#76); `TODO.md`, "Prior evaluations (`fun_values`)." |
| W2-7 | B1 | `f_vals`, PyBADS's own, sets a display format that the display cannot fill | defect | refused with a message (KD-B1-4); one without a finite value stands for `None` since the doublecheck | `652b25c` (#76); `68d4516` (`dev-next`) |
| W2-8 | B1 | A multi-row `x0` passes the checks and fails in `optimize()` | port discrepancy | refused, as MATLAB | `6250a3a`, `a1e8a93` (#76) |
| W2-9 | B1 | `x0=None` with only hard bounds is accepted, where MATLAB and PyBADS's own "Raises" section refuse it | port discrepancy | kept accepting it (revised proposal); the "Raises" section corrected (KD-B1-12) | `9b172fc` (#76) |
| W2-10 | B1 | The check of `non_box_cons`'s output accepts (N, k) and fails on a scalar; the docstring's example is MATLAB's syntax | port discrepancy; confirmed defect (the docs) | fixed: (N,) or (N, 1), anything else refused; the contract stated | `ab11b83` (#76) |
| W2-11 | B1 | A random start that violates `non_box_cons` stops the run, as in MATLAB | shared defect | fixed: drawn again, up to 1000 draws (KD-B1-11), tested on the mesh too since #87; `matlab_side_defects.md` | `c3d7815` (#76); `3a8db6c` (#87) |
| W2-12 | B1 | `status` is among the result's keys and never set | defect | fixed: MATLAB's exit flag (KD-B1-8) | `d964576` (#76) |
| W2-13 | B1 | `success` is `True` in every run | design question | `success` is `status > 0` | `877d63c` (#76) |
| W2-14 | B1 | The result deep-copies `fun` and `non_box_cons`, so a callable holding a lock makes `optimize()` raise | defect | fixed: kept by reference | `1fb162e` (#76) |
| W2-15 | B1 | `display` is compared as an exact string, where MATLAB reads its first three letters | port discrepancy | fixed: MATLAB's levels (KD-B2-3) | `37be0d9` (#76) |
| W2-16 | B3 | `search_factor_min` is read by nothing, so the search factor has no floor | port discrepancy | fixed, moves results (`ellipsoid_D10`'s evaluations flagged, 643 → 676, its error unchanged) | `3272bdd` (#76) |
| W2-17 | B1 | `tol_noise` is `eps · tol_fun` where MATLAB's is `sqrt(eps) · TolFun` | port discrepancy | fixed | `ba7de41` (#76) |
| W2-18 | B1 | MATLAB's checks of `MaxFunEvals` and `ImprovementQuantile` are not ported | port discrepancy | fixed; integers beyond 64 bits taken since #81 | `b29b9b5` (#76); `92a1d12` (#81) |
| W2-19 | B1 | A user's `None` replaces the default, and a string such as `'off'` is a true boolean | design question | `None` stands for the default, and the boolean options take only booleans (KD-B1-3) | `49ac8aa` (#76) |
| W2-20 | B1 | `overhead` leaves the final samples and the merged repeats out of the target's time | port discrepancy | fixed | `17e65ee` (#76) |
| W2-21 | B1 | A random `x0` drawn in the original plausible box | no longer holds (W0-5) | none | — |
| W2-22 | B1 | Option descriptions are cut at their first `=` or `:` | confirmed, inert | fixed | `93c86ee` (#76) |
| W2-23 | B1 | Nine options have no description, and 36 end in MATLAB's closing quote | confirmed, inert | descriptions corrected | `2d4304c` (#76) |
| W2-24 | B1 | `search_n_try` is a float | confirmed, inert | fixed | `2d4304c` (#76) |
| W2-25 | B2 | After the re-estimate, a better earlier iterate gives the incumbent its value but not its location, as in MATLAB | shared defect | (b): the incumbent moves with its value, a departure from MATLAB (KD-B2-7); moves the noisy runs, unflagged; `matlab_side_defects.md` | `a9fbb97` (#76) |
| W2-26 | B2 | The hyperparameters that the move sets reach only the next search's target, which nothing reads, as in MATLAB | confirmed, inert (shared) | kept | — |
| W2-27 | B2 | The design is capped before its rounding up, so small budgets are exceeded, and the noise test is not counted, as in MATLAB | port discrepancy; shared defect (the noise test) | fixed: the cap after the rounding, the noise test counted, the reserve floored at 0 (KD-B2-6); `matlab_side_defects.md` | `381bf32`, `f6f7f74` (#76) |
| W2-28 | B2 | `sloppy_improvement=False` stops every run at its first pass | port discrepancy | fixed | `f08b475` (#76) |
| W2-29 | B4 | The accelerated mesh reduction is tested one iteration later than in MATLAB | port discrepancy | fixed; no run of the benchmark changed | `c9a2cde` (#76) |
| W2-30 | B2 | The current iterate keeps its estimate when its re-estimate fails, where MATLAB records NaN | intentional, missing from the sheet | kept (KD-B2-4) | `9f65d73` (#76), the sheet |
| W2-31 | B2 | The display's action column shows a stale action | port discrepancy (display) | fixed | `765f13c` (#76) |
| W2-32 | B2 | A run that ends in its initialization reports 0 iterations, MATLAB 1 | design question | kept 0 (KD-B1-8) | `9f65d73` (#76), the sheet |
| W2-33 | B2 | The output function's stop message and final stop, and its `"init"` call after a noisy run's setup | intentional, missing from the sheet; design question (the timing) | kept (KD-B2-5) | `9f65d73` (#76), the sheet |
| W2-34 | B2 | With one final sample at level 1, `yval_vec` has shape (2, 1) | port discrepancy | fixed | `3476000` (#76) |
| W2-35 | B2 | `min_iter` and `min_fun_evals` are read by nothing, and MATLAB has no such options | confirmed, inert | on the sheet among the options without effect; removed by #87 (KD-B1-5) | `9f65d73` (#76), the sheet; `b18382e` (#87) |
| W2-36 | B2 | A noisy run's incumbent is the raw minimum of its design for two iterations, as in MATLAB | design question, shared | kept, as MATLAB's; `matlab_side_defects.md` | — |
| W2-37 | B2 | A feasible band thinner than the mesh can resolve ends the run at `x0` on the stall criterion | design question, shared; needs MATLAB | documented in `non_box_cons`'s description; no MATLAB run needed | `bdaef58` (#76); `TODO.md`, "The GP on a one-point training set." |
| W2-38 | B2 | `IterationHistory` deep-copies every stored GP whenever it grows | confirmed, inert (time) | fixed | `500526b` (#76) |
| W2-39 | B2 | The NaN estimates of past iterates whose re-estimate failed stay in `iteration_history` | confirmed, inert | kept; the documentation says so (KD-B2-4) | `763e21f` (#76) |
| W2-40 | B2 | The survey's rows on the re-estimate (stored GPs changed in place; the restored GP's value; their geometry) | no longer hold (W0-1, W1-35) | the survey's rows corrected | — |
| W2-41 | B2 | With 0 or 1 final samples, the final message prints the last observation, as MATLAB's | not a defect | kept | — |
| W2-42 | B2 | Termination is tested at every pass, so a round once begun counts toward `max_iter` | not a defect | kept; the descriptions of `max_iter` and `tol_stall_iters` say so | `2d4304c` (#76) |
| W2-43 | B2 | The move and the final choice skip the first iterate, as MATLAB's `75ec49f` | not a defect | kept | — |
| W2-44 | B2 | The main loop discards the GP that `_poll_step_` returns | confirmed, inert | kept; the return taken since W4-28 | `ffaf424` (#80) |
| W2-45 | B1 | `VariableTransformer` writes the logs of integer bounds into integer copies (found while fixing) | defect | fixed: bounds and `x0` cast to float | `dc7383a` (#76) |
| W2-46 | B1 | The self-test of a mixed transform warns of a spurious overflow (found while fixing) | confirmed, inert | fixed | `885fc33` (#76) |
| W2-47 | B1 | `VariableTransformer` used directly still truncates integer arrays (found while fixing) | defect | fixed | `a07be4e` (#76) |

### Wave 3

| Row | Slice | What | Classification | Disposition | Fix or open item |
|---|---|---|---|---|---|
| W3-1 | B3 | `contraints_check` removes no point already evaluated, so runs evaluate points again, where `uCheck.m` removes them | port discrepancy | fixed, moves results (no flag); its bins rounded as MATLAB's since W4-21 | `149d528` (#77); `86512c9` (#80) |
| W3-2 | B3 | `contraints_check` returns its candidates sorted by bin | not a defect (MATLAB's `setdiff` sorts too) | the comment corrected | `7e09887` (#77) |
| W3-3 | B3 | ES-wcm's covariance is the unweighted scatter of the best points, as `ucov.m`'s | shared defect | kept, as MATLAB's; comments corrected; `matlab_side_defects.md` | `7e09887` (#77) |
| W3-4 | B3 | ES-wcm takes one more best point than it has weights | port discrepancy | fixed, moves results | `81c6a15` (#77) |
| W3-5 | B3 | The ES selection mask is shifted by one rank (MATLAB's 1-based positions as 0-based indices) | port discrepancy | fixed, moves results | `115a922` (#77) |
| W3-6 | B3 | The hedge's expected reward takes `exp(-0.5*g**2/sqrt(2*pi))` for the normal density | port discrepancy | fixed, moves the noisy runs (KD-B3-3) | `d79ab75` (#77) |
| W3-7 | B3 | `hedge_gamma = 0` stops the run at the first search, on both sides at different places | shared defect | fixed: each search scored at the search point (KD-B3-8); `matlab_side_defects.md` | `4d357e4` (#77) |
| W3-8 | B3 | The fraction of new candidates behind the ES scale's update is miscounted, from `n_search_iter` 3 | port discrepancy | fixed, with a guard for 0/0 (KD-B3-6) | `c788617` (#77) |
| W3-9 | B3 | An ES generation emptied by the checks discards the earlier candidates with a false warning | port discrepancy | fixed: the generation skipped, the candidates kept (KD-B3-6); the message at DEBUG since W4-27 | `a77d95d` (#77); `6f673a2` (#80) |
| W3-10 | B3 | `acq_fcn_lcb` refuses a plain number as `sqrt_beta`, and takes neither names nor non-finite values | port discrepancy | `None`, a positive finite number or a callable, anything else refused (KD-B3-7); checked when `BADS` is created since W4-19 | `599115b` (#77); `36c9ec1` (#80) |
| W3-11 | B3 | An empty search set skips the hedge's update, where MATLAB decays the gains (and moves to a stale point, or stops) | design question | fixed: an empty set decays the gains; MATLAB's move and stop left out (KD-B3-5); `matlab_side_defects.md` | `4388e6d` (#77) |
| W3-12 | B3 | After a failed rebuild the search ranks its candidates by the restored GP, where MATLAB's scores are all 0 | design question | kept (KD-B5-2) | — |
| W3-13 | B3 | `ESSearchCMA` cannot run, and no option reaches it | confirmed, inert | removed (KD-B3-1) | `4865fad` (#77) |
| W3-14 | B3 | `force_to_grid` rounds halves to even, where `force2grid.m` rounds them away from zero | port discrepancy | fixed; moved 31 runs of the benchmark, unflagged | `1f7c8ee`, `bd110f0` (#77) |
| W3-15 | B3 | The ES search orders with the unstable `np.argsort`, where MATLAB's sort is stable | port discrepancy | fixed, moves results | `c276d79` (#77) |
| W3-16 | B3 | `search_factor_min` unread | no longer holds (W2-16) | none | — |
| W3-17 | B3 | `search_n_try` a float | no longer holds (W2-24) | none | — |
| W3-18 | B3 | The empty set's `f_sd_search` is an `int` | confirmed, inert | fixed | `4388e6d` (#77) |
| W3-19 | B4 | `p_less` is taken over D + 1 unsorted probabilities, where MATLAB sorts them and takes the D largest | port discrepancy | fixed, moves results | `8e28124` (#77) |
| W3-20 | B4 | `uncertain_incumbent=False` at level 0 stops the run at the first poll | port discrepancy | fixed | `5f31837` (#77) |
| W3-21 | B4 | The target under `hyp_best` is predicted from a copy's recomputed posterior, where MATLAB mixes the current posterior with `hyp` | design question | kept, (a) (KD-B4-2); since #84 the GP's own posterior when `hyp_best` is its own | — |
| W3-22 | B4 | The poll calls `np.seterr` and never restores it | defect (PyBADS only) | fixed: `np.errstate` | `f595f1b` (#77) |
| W3-23 | B4 | A non-finite prediction gives a NaN target, on both sides | shared defect | fixed: the target from the incumbent's SD (KD-B4-4); `matlab_side_defects.md` | `dac062e` (#77) |
| W3-24 | B4 | The poll's basis is the signed coordinate directions at every default state, as in MATLAB, whose bound of the basis inverts LTMADS's | design question, shared | (b), LTMADS's directions, fixed and reverted after its flagged gate (PI): MATLAB's poll kept (KD-B4-1); `matlab_side_defects.md` | `869a033`, reverted by `b03a320` (#77) |
| W3-25 | B4 | `poll_scale` does not shape the poll vectors, which are divided by it and multiplied back | not a defect (MATLAB's design) | records corrected | `fd8641d` (#77) |
| W3-26 | B4 | At level 0 the poll's GP does not take the poll's evaluations, as in MATLAB | design question, shared | kept; `matlab_side_defects.md` | — |
| W3-27 | B4 | `np.argmin` returns a NaN acquisition value, where MATLAB's `min` skips NaN | confirmed, inert | fixed: `nanargmin`, and a random choice when every value is NaN (KD-B4-5) | `a1bf658` (#77) |
| W3-28 | B4 | A zero predictive SD makes the GP unreliable and stops a good poll, as MATLAB's rule does | not a defect | kept; the description of `tol_poi` corrected | `fd8641d` (#77); `TODO.md`, "Zero predictive SDs at uncertainty level 0." |
| W3-29 | B4 | After a poll that moves the incumbent, MATLAB rebuilds at every search until a poll that does not move; PyBADS rebuilt once (W1-2's premise) | port discrepancy | fixed, moves results | `0b7add3` (#77) |
| W3-30 | B4 | With `poll_training` off, the poll neither records a refit nor clears the flag of an unreliable GP | intentional, missing from the sheet | kept (KD-B5-9) | `97bfc99` (#77), the sheet |
| W3-31 | B4 | An `improvement_quantile` outside (0, 1) gives NaN improvements, where MATLAB refuses it | port discrepancy | refused when `BADS` is created (KD-B4-6); any value that is not a real number since #79 and #81 | `ec1b2d0` (#77); `37cc649` (#79); `339e90e` (#81) |
| W3-32 | B4 | A successful poll appends a bound method | no longer holds (W0-16) | the survey's row corrected | — |
| W3-33 | B4 | After a re-estimate that moves nothing, `optim_state` keeps older values | confirmed, inert | kept in step (KD-B2-9) | `43ee8ed` (#77) |
| W3-34 | B4 | `np.vstack(u_poll, u_poll_new)` would raise in a branch that cannot run | confirmed, inert | the branch removed | `01ee524` (#77) |
| W3-35 | B4 | The poll discards `period_check`'s result | confirmed, inert | kept until periodic variables are ported | `TODO.md`, "Porting gaps" |
| W3-36 | B4 | `u_base` is computed and never used | confirmed, inert | removed | `e4b3bca` (#77) |
| W3-37 | B4 | The accelerated mesh reduction tested from the wrong iteration | no longer holds (W2-29) | none | — |
| W3-38 | B4 | Under `stobads`, a NaN estimate counts as uncertain | no longer holds (W0-11) | the survey's row corrected | — |
| W3-39 | B4 | An `accelerate_mesh_steps` below 1 stops the run, on both sides (from wave 2's doublecheck) | shared defect | refused unless a positive integer, `inf` included (KD-B4-6); the message names `accelerate_mesh=False` since #79; `matlab_side_defects.md` | `5d711bf` (#77); `d16cbba` (#79) |
| W3-40 | B6 | A rebuild on two distinct points gives the length scales' prior a zero width, which gpyreg refuses (found by W3-24's gate) | shared defect | fixed: the previous prior kept (KD-B6-2); `matlab_side_defects.md` | `a14524d` (#77) |

Also in wave 3, without a row: `ESSearch` no longer configures the root
logger (`d0c7178`, #77), and `acq_hedge=True`, which stopped a run at its
first improving search, is refused when `BADS` is created (`d16cbba`, #79;
KD-B3-3), both by the PI's rulings.

### Wave 4

| Row | Slice | What | Classification | Disposition | Fix or open item |
|---|---|---|---|---|---|
| W4-1 | B7 | The initial design is seeded from the integer parts of `u0`, so that it depends on neither the start inside the plausible box nor `random_seed` | needs MATLAB (MATLAB's own seed); PyBADS's facts confirmed | fixed, option (a): seeded by one draw of the run's generator (KD-B7-1); moves results; MATLAB's seed a question in `matlab_side_defects.md` | `efe5e95` (#80) |
| W4-2 | B7 | A start at or below -1 in `u` reaches an undefined cast to `uint64`, which x86 and arm64 resolve differently | defect | fixed with W4-1; wave 2's dating of the reach corrected | `efe5e95`, `3a8096b` (#80) |
| W4-3 | B7 | The design doubles when its size equals D, with no recorded reason (W0-18) | design question | kept at every D (PI), where the proposal was to remove it (KD-B7-1) | `a84a3dd` (#80), the records |
| W4-4 | B7 | `init_sobol` returns the exponent where its docstring says the number of samples, and its parameters are misdescribed | confirmed, inert | fixed; `lb` and `ub` required | `8daf7ad` (#80) |
| W4-5 | B7 | No run reaches the merge of a repeated point at level 2 since W3-1, and the records describe earlier runs | confirmed, inert | the merge kept (KD-B7-3), records corrected | `2dc5807`, `fba29cd` (#80) |
| W4-6 | B7 | The noise test, recorded nowhere, still adds 1 to the start's `n_evals` and its time to the start's row | defect (minor); the time inert | fixed, moves results; completed so that the fits' schedule leaves the test out of its budget | `e7bd01d`, `46af65a` (#80) |
| W4-7 | B7 | The untimed noise test counts as the optimizer's time in `overhead`, as MATLAB's | shared defect, negligible | kept, as MATLAB's; the description of `overhead` says so; `matlab_side_defects.md` | `e744ed9` (#80) |
| W4-8 | B7 | A malformed SD or a complex value does not raise the documented `ValueError` before the row is written | port discrepancy (minor) | fixed | `5dd92b7` (#80) |
| W4-9 | B7 | `finalize` trims every array but `n_evals`, and `reset_fun_eval_time` has no caller | confirmed, inert | fixed | `29a258a` (#80) |
| W4-10 | B7 | The logger's docstrings omit that `x` is in `u` space; `add`, which nothing calls, keeps checks of its own | confirmed, inert | docstrings corrected; `add` settled with the port of `fun_values` | `4ea665a` (#80); `TODO.md`, "Prior evaluations (`fun_values`)." |
| W4-11 | B7 | The poll discards `period_check`'s result, and the design takes the option's indices where the others take a mask | confirmed, inert (as W3-35) | kept until periodic variables are ported | `TODO.md`, "Porting gaps" |
| W4-12 | B7 | The log grows when full, where MATLAB's is a ring of `CacheSize` rows that never writes its last row | intentional difference, missing from the sheet | kept (KD-B7-4); the description of `cache_size` corrected; MATLAB's ring in `matlab_side_defects.md` | `f1247d0` (#80) |
| W4-13 | B7 | The noise test's second value goes through the logger's checks, where MATLAB reads NaN as deterministic and infinity as noisy | port discrepancy (benign) | kept (KD-B7-5) | `a84a3dd` (#80), the sheet |
| W4-14 | B7 | A noisy run that ends in its first iteration takes none of its reserved final samples, as MATLAB's, over more budgets | shared defect, widened by the design's size | fixed, option (a): the samples taken at the incumbent (KD-B2-8); `matlab_side_defects.md` | `b61a880` (#80) |
| W4-15 | O | Under Sto-BADS, an uncertain poll with no improving point moves the incumbent to itself and rebuilds | defect (Sto-BADS) | fixed: an uncertain poll moves only to a point that improves | `6c36782` (#80) |
| W4-16 | O | With several hyperparameter samples the poll scale sums unweighted and the effective radius is one per sample | confirmed, inert (unreachable) | fixed as MATLAB's | `fa5d842` (#80) |
| W4-17 | O | `acq_fcn_lcb`'s and `update_hedge`'s docstrings misdescribe them, and `acq_fcn_lcb` assigns an unread `n` | documentation | fixed | `2f15781` (#80) |
| W4-18 | O | `hedge_gamma` is not checked, on either side | shared defect (a missing check) | refused outside `[0, 1/n]` (KD-B3-8); a real number since #81 | `6e24519` (#80); `339e90e` (#81) |
| W4-19 | O | A `sqrt_beta` that is not valid is refused only at the first search, and a callable's value is not checked | defect | checked when `BADS` is created, and a callable's value at each call (KD-B3-7) | `36c9ec1` (#80) |
| W4-20 | O | Fig. 1, MATLAB's, draws the poll's steps of unequal length along the two axes | documentation, shared with MATLAB's README | caption corrected (and by #81) | `5442a6c` (#80) |
| W4-21 | B3 | `contraints_check` and the ES search's first split round halves to even, where `uCheck.m` rounds them away from zero (wave 3's doublecheck) | port discrepancy | fixed as MATLAB rounds; moves results | `86512c9` (#80) |
| W4-22 | B3, B4 | Three unused imports that pycln keeps | confirmed, inert | removed | `3b7e64c` (#80) |
| W4-23 | B3 | The empty search set's `search_dist` is an `int` | confirmed, inert | `0.0` | `65e2434` (#80) |
| W4-24 | B3 | `force_to_grid` has no docstring | confirmed, inert | docstring written | `4f535b8` (#80) |
| W4-25 | B3 | `n_search_iter` is not checked, on either side: 0, 0.5 or -1 stop the run at its first search | shared defect (a missing check) | refused unless a positive integer (KD-B4-6); `n_search` and large integers checked by #81 | `36e8b70` (#80); `92a1d12` (#81) |
| W4-26 | B2 | The noisy final estimate leaves `optim_state`'s values stale for the `"done"` call | confirmed, inert | `optim_state` kept in step (KD-B2-9) | `684d2e0` (#80) |
| W4-27 | B3 | The ES search logs "No candidate left" at WARNING, several times a run on a thin band | confirmed, inert (a message) | logged at DEBUG | `6f673a2` (#80) |
| W4-28 | B2 | The main loop discards the GP that `_poll_step_` returns | confirmed, inert (latent) | the return taken | `ffaf424` (#80) |
| W4-29 | B3 | `hedge_beta` and `hedge_decay` are not checked, on either side (found while verifying) | shared defect (missing checks) | refused outside their ranges (KD-B3-8), ruled during the pass; a real number since #81 | `bd793f2` (#80); `339e90e` (#81) |
| W4-30 | B2 | A noisy run stopped by `output_fcn` at `"init"` takes no final samples, and its `fsd` is not an estimate (found while fixing) | design question | kept, ruled during the pass; the description of `fsd` says what it is (KD-B2-8) | `4b84a2d` (#80) |

Also in wave 4, without a row: `periodic_vars` refused before the first
transform, and an empty value taken as `None` (`b78f782`, #80; KD-B1-6).

## The catalogue of deliberate differences

`pybads/bads/README.md` consolidates the 58 entries of the sheet of known
differences (`experiments/port_review_20260925/known_differences.md`,
which stays as the review left it) under their identifiers, citing PyBADS
by module and function and dropping the review's history. At the close
every entry was read again against `dev-next` and every MATLAB citation
against `74919c0`: each still describes a deliberate difference, and the
catalogue carries these corrections of the sheet:

- #84 and #85, after wave 4: the target's prediction and the search's part
  (KD-B4-2); `search_method` and `search_acq_fcn` refused when `BADS` is
  created (KD-B3-1, KD-B3-2); `f_vals` read only by its check (KD-B1-4);
  the rank-1 update measured and not adopted (KD-B5-1);
- the rulings of wave 4's doublecheck: real numbers for
  `improvement_quantile` and the hedge's options, one-element arrays for
  `sqrt_beta`, integers of any size and a `max_fun_evals` beyond 64 bits
  standing for `inf`, and `n_search` checked (KD-B1-3, KD-B3-7, KD-B3-8,
  KD-B4-6);
- errors that the sheet already had: KD-B1-9's test is in
  `_init_optim_state_`, not `_init_mesh_`; MATLAB fits samples of the
  hyperparameters at `gpSamples` above 1, not 0 (KD-B5-4); four options
  that KD-B1-4 lists as read are read only by code that no run reaches; the
  MATLAB lines of KD-B1-8 (`bads.m:1136`), KD-B2-5 and KD-B3-1
  (`searchES.m:39-101`); MATLAB BADS stops on an empty first search set at
  every quantile (KD-B3-5), and its search's random fallback fires when its
  acquisition raises (KD-B4-5).

Five entries are new: KD-B1-12, a missing `x0` with only the hard bounds
accepted (W2-9); KD-B1-13, the check of `tol_fun` (#84); KD-B3-9, the
floor of the ES search's number of parents (the rulings of wave 4's
doublecheck); and two that the doublecheck of the close found missing,
KD-B1-14, MATLAB's extra arguments to the target and its other calling
forms, and KD-B6-9, the warning of a `noise_size` above e^5 (W1-30).
KD-B4-3, LTMADS's directions, went with W3-24's revert. The
gpyreg citations of KD-B5-6 and KD-B6-6 were not read again.

## Open ends

**What `dev/TODO.md` holds.** Every row is fixed, kept or closed above but
for W0-12 and W0-13, the design of Sto-BADS's success rule. The rows and
items that a ruling left to later work are held by these items of
`dev/TODO.md`:

| `TODO.md` item | Rows and items |
|---|---|
| "gpyreg's inflation of the GP noise (W1-25), after wave 1's fixes." | W1-25, and W2-25's lower fraction solved on three of the five noisy configurations, unflagged at 30 seeds, which wave 2's doublecheck left to the same measurement |
| "The uncertainty interval of Sto-BADS." | W0-12, W0-13; W4-15's note on the search's move |
| "Zero predictive SDs at uncertainty level 0." | W3-28 and wave 3's "Found while verifying" |
| "Porting gaps" | W3-35, W4-11 (KD-B1-6) |
| "`gp_cov_prior="ard"`." | W1-28 (KD-B6-7) |
| "Prior evaluations (`fun_values`)." | W2-6, W4-10, `FunctionLogger.add`'s checks, and the final samples' bookkeeping in the log |
| "The GP on a one-point training set." | W2-37, W3-40, wave 1's "Found while fixing" and wave 2's "Found while verifying" |
| "The example notebooks' saved outputs." | wave 2's "Fix pass" and "Doublecheck" |
| "\"What's new\" at the next release." | W4-1, from wave 4's doublecheck; the release that `skills/pybads/SKILL.md` names |
| "`ellipsoid_D3_hetero` after `020d6a8`." | W3-1's effect on the configuration; W1-23, which fixed the bounds of the GP mean that the item listed as open |
| "gpyreg releases after 1.3.3." | W1-24 (acerbilab/gpyreg#57) and W1-25's switch, which reach PyBADS through a release |
| "Rank-1 GP update when adding a point: not adopted, to revisit if its terms change." | KD-B5-1 |

**The minor items.** "Found while fixing" of `verification/wave2.md` and
`wave4.md`, and "Doublecheck" of `wave2.md`, list minor items of slices B1,
B2, B7 and O. #84 fixed those that needed no choice of the PI. The PI
ruled on the rest at the close (2026-09-28), and #87 carries the rulings:

| Item | Ruling | In #87 |
|---|---|---|
| `test_options.ini` and `test_options2.ini` ship in the wheel, and nothing reads them | remove them | removed |
| A 0-d array for `max_fun_evals` or a boolean option is refused, where 1.1.0 took it; `tol_fun`'s check leaves other types through | keep refusing arrays, as for `improvement_quantile` and the hedge's options; refuse a `tol_fun` that is not a real number | `tol_fun` checked (KD-B1-13), the changelog's upgrading lines for all three |
| The reports of the log transform and of periodic variables are logged at INFO, and the caution for infinite bounds at WARNING, where MATLAB BADS prints all three from `"notify"` on | as MATLAB BADS | all three at the level of the opening message (KD-B2-3) |
| `__init__` fills missing plausible bounds without `bads:pbUnspecified` | warn, as MATLAB BADS | the warning (KD-B2-3) |
| The redraw of a random start tests it before it is put on the mesh (KD-B1-9) | test it on the mesh | the draw moved to where the start is put on the mesh (KD-B1-11) |
| The test of fixed variables leaves `x0` out (KD-B1-7) | no change: both sides refuse such a problem | KD-B1-7 says so |
| The floor of the ES search's `mu = n_search / n_search_iter`, kept by the rulings of wave 4's doublecheck | close as ruled | KD-B3-9 |
| `FunctionLogger.add`'s checks, and the final samples' bookkeeping in the log | with the port of `fun_values` | `TODO.md`'s item of that port |
| Elements beyond the pair in `search_acq_fcn` or in an entry of `search_method` are ignored | refuse them | refused (KD-B3-1, KD-B3-2) |
| 76 advanced options read by no code (78 on a closer count: `diagnostics` and `gp_cov_fun` too) | remove those without a MATLAB counterpart, keep and mark the MATLAB-named ones | 66 removed, 12 marked unused (KD-B1-5) |
| No module of PyBADS imports matplotlib, which `pyproject.toml` requires and gpyreg imports | keep the requirement, with a comment | the comment |
| `skills/pybads/SKILL.md` names no release | a step of the next release | `TODO.md`'s item on the release's "What's new" |
| `test_transform_inverse_largeN` built `np.ones((10 ^ 6, D))`, 12 rows, since `^` is XOR (wave 2, "Notes on the reports") | build a million | `10**6` |

**Noted, and ruled by no one.** The reports and the fix agents noted a few
observations outside their findings, most of which the ledgers left to the
docstrings and descriptions of the fix passes; the ones below held at
`dev-next` at the close, and no record took them up. None changes a
default run.

- ES-ell ignores the sum-rule flag of an entry of `search_method`, which
  only ES-wcm reads (non-default); `udist`'s periodic branch indexes the
  distance matrix's rows by variable (unreachable, KD-B1-6); the search
  step counts the points of the log, where MATLAB counts the GP's training
  set, equal in practice (wave 3, "Found while verifying").
- A 4-D ridge started on its valley stalls at `x0`, since the only descent
  direction is the exact diagonal, which the coordinate poll does not take
  (KD-B4-1); an `ESSearch` built directly with `n_search_iter = 0` returns
  its empty placeholders, since W4-25 checks the option when `BADS` is
  created (wave 3, "Fix pass").
- `VariableTransformer`'s inverse clips to the original bounds, where
  MATLAB's `transvars` does not, which absorbs only rounding; a target that
  returns `(f, sd)` without `specify_target_noise` is refused, where MATLAB
  drops `sd`; the function logger's `uncertainty_handling_level` keeps its
  value from construction after the noise test raises the run's level;
  `y_max` and `cache_count` are never read, and a merge does not update
  `Y_max` (wave 4, "Found while verifying").
- W2-36's measurement of a noisy run's first incumbent, which its ruling
  allowed "as a separate step" if W2-25 moved the noisy runs (it did), was
  not taken.
- Found at the close: `total_time` leaves out the creation of `BADS`,
  where MATLAB BADS times from the start of `bads()` (`bads.m:144`,
  `1186`).

The item "Loose ends of the port review" of `dev/TODO.md` points here.

**Needs MATLAB.** These questions of the review only MATLAB can settle:
MATLAB's seed of the initial design (W4-1, with the one call that settles
it, in `matlab_side_defects.md`), with the negative seed that
`i4_sobol.m:249-250` would clamp under one reading of its `mod`; what
MATLAB's fit does with the zero-variance prior of a plateau (W1-26, the
rebuild case), of a two-point training set (W3-40) and of a one-point
training set (W2-37). PyBADS's disposition of each is decided whatever
MATLAB computes, but for the priors of a GP on one point, an open item of
`dev/TODO.md` ("The GP on a one-point training set.") that a comparison
with MATLAB would inform; by the plan's rule no MATLAB run is written up. Two
items of `dev/TODO.md` would use one: a run of MATLAB BADS on
`ellipsoid_D3_hetero`, and MATLAB's prediction of a GP whose predictive SD
PyBADS computes as 0.

## The MATLAB side

What the review found wrong or questionable in MATLAB BADS itself, at
`74919c0`, whether or not PyBADS shares it, is collected in
[`experiments/port_review_20260925/matlab_side_defects.md`](../experiments/port_review_20260925/matlab_side_defects.md)
for a developer of MATLAB BADS, with the MATLAB lines, the evidence and
what PyBADS does; nothing there was run in MATLAB. Its items, as it stands:

| Item | MATLAB | Row | PyBADS |
|---|---|---|---|
| *Shared defects that PyBADS fixes* | | | |
| With `PollTraining` off, the poll records a refit that it then cancels | `bads.m:822-823` | W1-8 | neither performs nor records it (KD-B5-9) |
| A zero spread of the training targets gives a degenerate prior (needs MATLAB for what its fit then does) | `gpdef/gpdefBads.m:219-222`, `293-295` | W1-26 | a rebuild keeps the previous prior (KD-B6-2) |
| At D = 1 the training set's distances are in units of 1, not of the fitted length scale | `private/gpupdate.m:285-292` | W1-17 | the fitted length scale at every D (KD-B5-10) |
| The transform's self-test refuses valid bounds of large magnitude | `utils/transvars.m:30`, `169-178` | W2-5 | a relative tolerance (KD-B1-10) |
| A random start that violates the non-box constraints stops the run | `private/setupvars.m:83-85`, `private/evalinitmesh.m:22-26` | W2-11 | drawn again, up to 1000 times (KD-B1-11) |
| The noise test is left out of the budget | `private/evalinitmesh.m:37-42`, `98-104` | W2-27 | counted (KD-B2-6) |
| After the re-estimate, the move to an earlier iterate takes its value and not its location | `bads.m:1111-1118`, `769` | W2-25 | the incumbent moves with its value (KD-B2-7) |
| With `HedgeGamma` 0, the search hedge fails at its first update | `acq/acqPortfolio.m:40`, `47` | W3-7 | each search scored at the search point (KD-B3-8) |
| A non-finite target prediction gives a NaN target | `bads.m:1310-1311`, `1321` | W3-23 | the target from the incumbent's SD (KD-B4-4) |
| `AccelerateMeshSteps` below 1 stops the run | `bads.m:976-979` | W3-39 | refused when `BADS` is created (KD-B4-6) |
| The prior of the length scales on two points has a zero width | `gpdef/gpdefBads.m:240-251` | W3-40 | the previous prior kept (KD-B6-2) |
| A noisy run that ends in its first iteration takes none of the final samples it reserves | `bads.m:1138` | W4-14 | takes them at the incumbent (KD-B2-8) |
| The search hedge's parameters are not checked | `search/searchHedge.m:45-46`, `acq/acqPortfolio.m:69` | W4-18, W4-29 | refused outside their ranges (KD-B3-8) |
| `Nsearchiter` is not checked | `private/setupoptions.m:26`, `search/searchES.m:125` | W4-25 | refused unless a positive integer (KD-B4-6) |
| *Shared design observations (PyBADS keeps MATLAB's behavior)* | | | |
| The calibration test for three or more points tests normality only | `utils/gppredcheck.m:30` | W1-7 | as MATLAB |
| The GP's noise is bounded above at a log SD of 5 | `gpdef/gpdefBads.m:161` | W1-30 | as MATLAB, with a warning (KD-B6-9) |
| A noisy run's first incumbent is the raw minimum of its initial design | `bads.m:1097` | W2-36 | as MATLAB |
| A feasible region thinner than the mesh can resolve ends the run on its stall criterion (needs MATLAB for the GP on one point) | — | W2-37 | as MATLAB, documented |
| The covariance of ES-wcm is the unweighted scatter of the best points | `utils/ucov.m:19` | W3-3 | as MATLAB, commented |
| At uncertainty level 0 the poll's GP does not take the poll's evaluations | `bads.m:908-916`, `841` | W3-26 | as MATLAB |
| The poll's basis is bounded by the inverse of LTMADS's ratio, so the poll steps along the coordinates | `poll/pollMADS2N.m:7` | W3-24 | as MATLAB, after LTMADS's bound was tried and reverted (KD-B4-1) |
| The noise test's time counts as the optimizer's time | `private/evalinitmesh.m:41`, `private/funlogger.m:130` | W4-7 | as MATLAB, documented |
| *Defects that PyBADS does not share* | | | |
| An empty search set moves to a stale point, or stops the run | `bads.m:667-725`, `1257-1279` | W3-11 | a failed search (KD-B3-5) |
| The scale of the ES search becomes NaN after a generation without candidates | `search/searchES.m:170-193` | wave 3, "Found while verifying" | the scale kept (KD-B3-6) |
| The ring of evaluations never writes its last row, and reads it | `private/funlogger.m:120-121` | W4-12 | the log grows (KD-B7-4) |
| The rank-1 update takes a non-finite value before its penalty | `private/gpupdate.m:55`, `69-78` | the measurement of the rank-1 update ([`2026-09-28-where-pybads-spends-its-time.md`](2026-09-28-where-pybads-spends-its-time.md)) | the value replaced before a full recomputation (KD-B5-1) |
| *Questions that need MATLAB* | | | |
| The seed of the initial design: `mod(prod(uint64(num2str([0.25 -0.5]))), 997) + 1` is 966 for an exact remainder and 1 otherwise | `init/initSobol.m:9-15` | W4-1 | seeded from the run's generator, whatever MATLAB computes (KD-B7-1) |

## The records

- The plan and its worklog:
  [`plans/port-correctness-review.md`](../plans/port-correctness-review.md).
- Under
  [`experiments/port_review_20260925/`](../experiments/port_review_20260925/README.md):
  the sheet of known differences (`known_differences.md`), the counterpart
  map of MATLAB's files (`counterpart_map.md`), the preparatory agent's
  report (`prep_report.md`), the reviewers' reports (`reviews/`), the
  briefs (`briefs/`), the verifiers' reports and the per-wave ledgers
  (`verification/`), the fix agents' reports (`fixes/`), the scripts of the
  agents of waves 1 to 4 (`verification/scripts/`), the comparisons of the
  fix passes (`verification/wave<N>_fixpass/`), the doublechecks of waves 3
  and 4 (`verification/wave<N>_doublecheck_<scope>.md`), the investigation
  of W0-1 (`w01_investigation/`), the Linux comparison of the close
  (`verification/close/`), and the MATLAB side (`matlab_side_defects.md`).
  The check scripts of wave 0 are kept on the machine that ran them
  (`dev/scripts/runs/LOCAL.md`).
- The references of the benchmark that the fix passes left, one per wave on
  Linux (`experiments/population_linux_wave0_20260926/` to
  `population_linux_wave4_20260927/`), and on Windows the pre-review
  baseline and the reference after wave 4 (`dev/README.md`, "Index").
- The starting point, the
  [codebase survey](2026-09-23-codebase-survey.md), whose candidate table
  the review closed, each row with its ledger row, its fix or its entry of
  the catalogue in the status column.
