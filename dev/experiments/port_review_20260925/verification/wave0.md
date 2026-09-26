# Wave 0 ledger: M, S and the preparatory agent's claims

The verified findings of wave 0 of the port review
([plan](../../../plans/port-correctness-review.md)). The reports are
`reviews/M_comparison.md` and `reviews/S_internal.md`; the claims C1 to C8 are
the section "Claims that did not check out" of `known_differences.md`. Each
item was verified by a fresh Opus agent that had not written it, with checks
of its own, at the review's freeze `95da7f1` (and at `ab4dded`, where the
reports read the code): `wave0_S_verifier.md` for S, `wave0_M_verifier.md`
for M and the claims. Their scripts are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). Lines are at `95da7f1`, MATLAB lines at
`74919c0`. "Survey" names a row of the candidate table of
`dev/results/2026-09-23-codebase-survey.md` that describes the same
behavior. The dispositions are proposals until the PI rules; the gate is
the one of `AGENTS.md`, "Numerical gates", that a fix would need.

## The findings

| Id | Source | What | Classification | Default run | At `95da7f1` | Dating | Survey | Proposed disposition | Gate |
|---|---|---|---|---|---|---|---|---|---|
| W0-1 | M-F1, M-F2 | In noisy runs, the end-of-iteration re-evaluation replaces the working GP with a GP of the iteration history (`bads.py:1481`, `1505-1507`), and that GP is the history slot itself, which the next iteration's rebuilds and refits then change in place; later re-evaluations use the drifted hyperparameters. MATLAB changes only `fhyp` and re-evaluates each iterate with its recorded hyperparameters (`bads.m:1091-1120`, `1388`) | confirmed port discrepancy | yes (noisy targets, from the 2nd iteration) | present | never agreed (`c7c88ab`, `9037851`; MATLAB unchanged since 2017) | the rows on `_re_evaluate_history_` (in place; the geometry of each stored GP) | fix: drop the swaps, re-evaluate from a copy of the working GP with `gp_hyp_full[i]` | population comparison (noisy configurations change; deterministic runs must be identical) |
| W0-2 | M-F3 | final `fsd` without target noise normalized by `n` | confirmed port discrepancy | yes | fixed by #71 | never agreed until #71 | row fixed in #71 | none | — |
| W0-3 | M-F6 | final estimate recorded in the last iterate's slot | confirmed port discrepancy | yes | fixed by #71 | never agreed until #71 | row fixed in #71 | none | — |
| W0-4 | M-F7 | `iterations` one below MATLAB's; `yval_vec` None for deterministic runs and `noise_final_samples = 0`, where MATLAB returns `yval` | `iterations`: port discrepancy; `yval_vec`: intentional (the docstring), not on the sheet | yes | `iterations` fixed by #71; `yval_vec` as documented | — | row fixed in #71 | `yval_vec`: keep, add to KD-B1-8 | none |
| W0-5 | M-F4 | a random `x0` (missing or non-finite) is uniform in the original plausible box; MATLAB's is uniform in the transformed box (`setupvars.m:79-85`), log-uniform for a log-transformed variable | confirmed port discrepancy | no (random `x0` and a log-transformed variable) | present | never agreed (`d466948` in x, `019f0b4` in u) | — | fix: draw in the transformed box | fingerprint; a test of the distribution |
| W0-6 | M-F5 | the substitution of non-finite targets in `local_gp_fitting` indexes the full array with an index into the finite subset; its `s2` branch tests a key nothing writes; `add_and_update_gp` has no penalty | confirmed port discrepancy, unreachable (the function logger refuses non-finite values) | no | present | PyBADS since `c7c88ab`; MATLAB fixed its own index in `74919c0` | — | fix cheaply, as MATLAB | fingerprint |
| W0-7 | M-F8 | the starting GP mean is the median of the lowest `round(0.8 N)` values, MATLAB's of `ceil(0.8 N)` (`gpdefBads.m:164-165`) | confirmed port discrepancy, small | yes (N mod 5 in {3, 4}: D = 1, 4-7, 16-31 deterministic, every noisy run) | present | MATLAB changed after the port (`d4fead5`) | — | fix the starting mean alone (the high-density set feeds the prior and the bounds too) | population comparison |
| W0-8 | M-F9 | the nearest-neighbour training set sorts with the unstable `np.argsort`, MATLAB with a stable sort | confirmed port discrepancy (mechanism); no change of membership seen, only of row order | row order yes | present | never agreed | — | fix with `kind="stable"` | fingerprint; a population comparison if it moves |
| W0-9 | M-F10 | with `uncertainty_handling=True` and no target noise, `gp.s2` holds NaN, which gpyreg ignores | confirmed, inert | no | present | `c7c88ab`, `8ff10f5` | the row fixed in `020d6a8` notes it | keep, or hold `S` at level 2 only | fingerprint |
| W0-10 | S-F1 | the Sto-BADS poll decides from the last evaluated point: a success followed by another point is discarded (no move, no mesh expansion), where Sto-MADS succeeds if some point succeeds | confirmed defect | no (`stobads=True`) | present | `9037851` | — | fix | fingerprint; population with `stobads=True` |
| W0-11 | S-F2 | a NaN estimate (after a failed GP rebuild or add) counts as "uncertain", so with `opp_stobads` the incumbent moves to a point with NaN value | confirmed defect | no (`stobads=True` and a GP failure) | present | rule `9037851`, NaN estimates `685da15` | the row on `stobads` NaN estimates | fix: a non-finite estimate is a failure | fingerprint |
| W0-12 | S-F3 | the uncertainty threshold `gamma * epsilon * mesh_size**2`, with epsilon the GP's SDs, which do not shrink with the mesh as Sto-MADS's accuracy requirement makes its estimates do: the "certain" outcomes are nearly coin flips at small meshes, and the rule is looser than BADS's own sufficient improvement | design question | no | present | `9037851` | — | PI: what epsilon means (current; power 0, a z-test; a fixed epsilon with the mesh squared, as Sto-MADS) | fingerprint; population with `stobads=True` |
| W0-13 | S-F4 | `opp_stobads` moves the search incumbent on any uncertain outcome, including to worse estimates, and widens the search as for an incremental improvement | design question | no | present | `9037851` | — | PI, with W0-12: which uncertain moves, and what they do to the search factor | as W0-12 |
| W0-14 | S-F5 | `BADS.__init__` takes an undocumented 8th positional parameter, `gamma_uncertain_interval`, before `options`: a call in MATLAB's order, `BADS(fun, x0, lb, ub, plb, pub, None, options)`, drops every option, the seed included | confirmed defect | yes (that call form) | present | `9037851` | — | fix: `gamma_uncertain_interval` keyword-only after `options`, or an option, documented | fingerprint; changelog and an "Upgrading from" line |
| W0-15 | S-F6 | an empty search set stops the run with `UnboundLocalError`; an ES search with no candidate with `IndexError` | confirmed defect, narrow reach (a non-deterministic `non_box_cons`) | no | present | `c7c88ab`; MATLAB has the same branch and continues | the two rows on the empty search set | fix: count it as a failed search on every path; the ES returns an empty set as MATLAB's | fingerprint |
| W0-16 | S-F7 | a successful poll appends the bound method `self.u_best.copy` to `optim_state["u_success"]`, which since #71 reaches `output_fcn` | confirmed, inert | yes | present | `c7c88ab` | the row on `bads.py:2183` | fix | fingerprint |
| W0-17 | C1 | the noise test also runs with `uncertainty_handling=False`, where MATLAB tests only when the option is empty (`evalinitmesh.m:17`, `38-50`); a noisy target then switches a run declared deterministic to level 1 | record right in `AGENTS.md`, code differs from MATLAB | no (`None` is the default) | present | code since `c7c88ab`; MATLAB since 2017 | — | fix the code | fingerprint |
| W0-18 | C2 | the Sobol design is doubled when `2**ceil(log2(n))` equals `D` (`init_sobol.py:72-76`): 2, 4, 8, 16, 32 points at D = 1, 2, 4, 8, 16; MATLAB draws `Ninit` | record wrong (`AGENTS.md`, the option's description); unexplained code | yes (those D) | present | `c7c88ab` | — | PI: correct the record, or remove the doubling | none, or a population comparison at those D |
| W0-19 | C3 | the Sobol seed does not keep MATLAB's derivation, as `dev/plans/tooling-and-rng.md` says: MATLAB skips points of the unscrambled sequence by a number from the digits of `u0`, PyBADS seeds scipy's scrambling from the integer parts of `u0` | record wrong | yes | present | `c7c88ab` | the row on the Sobol seed (whether MATLAB's `uint64` product saturates needs MATLAB) | correct the plan's sentence | none |
| W0-20 | C4-C7 | comments that name the wrong kernel, the wrong retry, features of PyVBMC's GP code that BADS lacks, and stale MATLAB pointers | record wrong | — | present | `c7c88ab`, `9037851` | — | correct the comments | fingerprint |
| W0-21 | C8 | the changelog's "Failed GP updates" entry reads as if the forced refit were MATLAB's; "(in the poll, only with `poll_training` on)" misses the refit forced at iteration 0 | record right, ambiguous | — | present | `685da15` | — | edit the entry | none |

## Found while verifying

Reproduced by a verifier but outside the reports, or not verified; each
belongs to a later slice, whose wave checks it.

- `search_factor_min` is never read, where MATLAB floors the search factor
  after a failed search (`bads.m:1366`); default runs reach it (S verifier,
  by reading). One of the differences the preparatory agent saw in passing
  and kept from the reviewers; slice B3, wave 3.
- A noisy run whose budget is below its power-of-two initial design exceeds
  the budget (`max_fun_evals=30`, D = 2: 34 evaluations, `noise_final_samples`
  -4), or stops with `ValueError` from the division by zero of the survey's
  row at `1cfe371` (M verifier, reproduced). MATLAB caps the design at
  `MaxFunEvals - 1`. One of the differences seen in passing; slice B2,
  wave 2.
- A thin feasible region (`|x1 - x2| <= 0.005` as `non_box_cons`) stops the
  first GP fit with `ValueError` from gpyreg's `set_priors`, after
  `log(np.std(gp.y))` of a single point (S verifier, reproduced, not
  analysed); slices B1 and B6.
- After a failed rebuild in a noisy run, the swap of W0-1 hands on a slot
  whose `needs_rebuild` and `needs_refit` markers the re-evaluation removed
  (M verifier, not verified); the fix of W0-1 removes it.
- `IterationHistory._expand_array` deep-copies every stored GP when an
  iteration is added (M verifier; performance).
- `bads.py:2116` calls `np.vstack(u_poll, u_poll_new)` with two arguments,
  reachable only when the poll basis is empty (S verifier); slice B4.
- Unstable `np.argsort` also at `es_search.py:190`, `240` and
  `get_hpd.py:34` (M verifier, not checked); slices B3 and B6.
- Needs MATLAB: whether `prod(uint64(strseed))` in `initSobol.m` saturates.
