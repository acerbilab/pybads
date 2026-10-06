<!-- Report of reviewer (a) of the doublecheck of wave 3, the fixes of B3 (briefs/wave3_doublecheck.md), reading PyBADS at 0d866e8 in /home/user/pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave3/doublecheck/a_B3/. -->


# Wave 3 doublecheck: the fixes of B3

Scripts and outputs: `/tmp/claude-0/-home-user-pybads/cec8ef42-ff6e-5789-9637-a276f362470b/scratchpad/dc/a_B3/` (`c1`–`c15`, each `.py` with its `.out`, plus the helpers `hdr.py`, `ucheck_ref.py` and `cc_variants.py`). Every script ran from that directory with one BLAS thread and `PYTHONPATH=/home/user/pybads-review:/home/user/gpyreg-v1.3.3`, and prints `pybads.__file__` (the review worktree) and `gpyreg.__file__` (the v1.3.3 clone). No tracked file was touched; the worktree is clean.

## 1. Coverage

- **Read completely:**
  - `verification/wave3.md`: the B3 rows, the notes, "Found while verifying", "Rulings" with "After the gates", and "Fix pass" with its choices and "Found while fixing".
  - Fix report A, fix report C, and the header and W3-14 part of report B. Also `briefs/wave3_fix_common.md`.
  - The diffs of all 15 commits in scope, plus `bd110f0` and `6281663`.
  - The code at `0d866e8`:
    - `search/es_search.py`, `search_hedge.py`, `grid_functions.py` (`force_to_grid`) and `search/__init__.py`;
    - `acq_fcn_lcb.py` and `constraints_check.py`;
    - in `bads.py`: `_search_step_` (1747-2082), `_update_search_bounds_`, `_update_search_stats_`, the logger setup, and the call sites of `contraints_check` (init 1141, poll 2232).
  - The tests: `test_search.py` and `test_empty_search.py`.
  - MATLAB at `74919c0`: `searchES.m`, `ESupdate.m`, `ucov.m`, `uCheck.m`, `force2grid.m`, `searchHedge.m`, `acqPortfolio.m`, `acqLCB.m`, `gppred.m` 1-60, and in `bads.m` lines 550-760, 1257-1285 and 1342-1426.
  - The gate files: `batch1_vs_reference.md`, `w3-14_attribution.txt`, `es_vs_batch1.md`, `w3-1_vs_es.md`, `geometry_w3-1_vs_es.md`, `geometry_es_summary.md` and `w3-1_repeats.txt`, with their `_fields.txt`. Also `count_repeats.py`, `same_fields.py`, `gate.sh` and `fp_all.out`.
  - The sheet entries KD-B3-1, KD-B3-3, KD-B3-5, KD-B3-6, KD-B5-2 and KD-B2-3, and the `matlab_side_defects.md` entries for W3-3, W3-7, W3-11 and the ES's NaN scale.
- **Skimmed:**
  - the CHANGELOG entries of the rows;
  - `w3-6_vs_w3-1.md` and `medians_default.md`;
  - the verifier's `v_ucheck.py` and fix C's `check_ucheck.py`;
  - the survey rows 230, 233, 239 and 253.
- **Not reached:**
  - the B3 reviewers' reports (grep only), and the verifier's report beyond B3-K8;
  - rechecks of W3-16 and W3-17 beyond the `.ini` line;
  - running the new tests (not allowed).

## 2. What holds

- **W3-1** (`149d528`), except F1:
  - On grids at least as coarse as the bin (`tol_mesh/2` = 2^-20), `contraints_check` equals a transcription of `uCheck.m` in 3000 of 3000 random sets with evaluated points among them, order included (c2).
  - `bads.py`'s calls (the initial design, the search step and the poll) equalled `uCheck.m` in 409 of 409 calls in 8 seeded runs (c3).
  - The test is corrected and seeded.
  - The gates compare the right commits and suites and reached the change:
    - default suite, `c276d79` to `149d528`: 54 tests, no flag, 325 runs with another fval;
    - geometry suite: 21 tests, no flag, 124 of 210 runs changed;
    - `w3-1_repeats.txt` matches the ledger's counts.
- **W3-2:** the output is sorted by bin, as `setdiff` sorts (c2), and the comment at `constraints_check.py:29-30` is true.
- **W3-3:** `ucov` computes SᵀS·Σw, the unweighted scatter, as `ucov.m:19` does. The comments at `es_search.py:252-254` and `313-315` are true, and the `matlab_side_defects.md` entry is present.
- **W3-4:** ⌊μ⌋ rows and ⌊μ⌋ weights.
  - The ES matches a transcription of `searchES.m` (methods 1 and 2) on the same inputs and draws: same point, value and scale in 64 of 64 cases (`n_search_iter` 2-5, 4 seeds, 2 search factors; c5).
- **W3-5:**
  - The mask equals `ESupdate.m`'s selectmask − 1 on every (μ, λ) in {1..79, 100, 683, 1024, 1365, 2048, 4096}² (0 mismatches).
  - 885072 is MATLAB's 1-based sum at (1, 2048) and 883024 the 0-based one.
  - At (2048, 2048) the offspring counts are 17, 12, 10, 9, 8, 7 (the old mask gave 1, 17, 12, 10, 9, 8) (c5, c6).
- **W3-6:** φ(γ) is exact. `update_hedge` equals a transcription of `acqPortfolio.m` 'upd' with difference 0 over 600 states, and the ratios 2.51 (γ = 0) and 424 (γ = −3) recompute (c7).
- **W3-7:**
  - At `hedge_gamma = 0` each search not chosen is scored at the search point, taken as a row. This matches `acqPortfolio.m:40` with `gppred`'s latent mean and variance (c7).
  - The new description of `hedge_gamma` is true for γ ≤ 1/n; above that it is recorded under "Found while fixing".
  - The `matlab_side_defects.md` entry is right: the assignment of `gpstructnew` was commented out since 2017 and removed in `d4fead5`.
- **W3-8:** the count of new candidates equals `searchES.m`'s:
  - the scale matches in c5;
  - generation by generation with a quantized LCB (many ties), the candidates of every generation are identical at `n_search_iter` 3, 4 and 5 (c9b).
  - The untrimmed pool selects MATLAB's set even at ties, because the kept ties precede the trimmed ones in the stable order. Agent C's "uncertain" note on this does not materialize.
- **W3-9:**
  - An emptied last generation returns MATLAB's point (n = 2 with generation 2 emptied, and n = 3 with generation 3).
  - At n ≥ 3 with an earlier generation emptied, the port keeps its scale where MATLAB's goes to NaN. The transcription reproduces MATLAB's corner `UBsearch` = [4, 4, 4], which confirms KD-B3-6 and the `matlab_side_defects.md` entry.
  - The warning reads "No candidate left in generation k of the search, once the points already evaluated or violating the constraints are removed", which is accurate.
  - The commit comes before W3-1.
- **W3-10:**
  - Accepted: `None`, callables, and positive finite int, uint or float values of size 1.
  - Refused with a `ValueError`: zero, negative, NaN, infinite, boolean, complex, string and empty values (c10).
  - The docstring now calls the third output an SD.
  - The "Upgrading from" line matches 1.1.0's check (`~np.isfinite(...) or .size > 1`).
- **W3-11 and W3-18:**
  - At γ > 0 the decay equals MATLAB's update at the stale point (the chosen search's er = 0, the others' er/Inf = 0): difference 0 over 200 states (c7).
  - The comment at `bads.py:2016-2021` holds against `bads.m:667-725` and `1266-1279`, with `SloppyImprovement` on by default.
  - The KD-B3-5 and `matlab_side_defects.md` entries are present, and `f_sd_search = 0.0`.
  - At γ = 0 PyBADS decays every gain, where MATLAB errors; this is off default.
- **W3-12:** KD-B5-2 is extended. MATLAB's z ≡ 0 path holds by reading `gppred.m:38-59` and `acqLCB.m:35`.
- **W3-13:** `git grep` finds no `ESSearchCMA` or `active_flag` in the package, the docs or `AGENTS.md`. KD-B3-1 is updated, and the "Upgrading from" line is present.
- **W3-14:**
  - `force_to_grid` equals MATLAB's round, computed exactly in rationals, on 479,936 random doubles and on the special cases: halves, `nextafter(0.5, 0)`, ±(2^51 + 0.5), 2^52 ± 1, ±inf and NaN. The one exception is that −0.0 becomes +0.0, which nothing can observe (c1).
  - The gate is batch 1: reference (`8510ca8`) against `a1bf658`, 54 tests, no flag, with 31 of 31 changed runs attributed to W3-14. Its numbers are in F2.
- **W3-15:** both sorts are stable. No other sort of the search is unstable: `np.unique` with `return_index` uses mergesort, and `nanargmin` returns the first minimum. The ES batch gate compares `a1bf658` with `c276d79`: 54 tests, no flag, every run changed.
- **The root logger:** creating and running `ESSearchWM`, `ESSearchELL` and `ESSearchHedge` adds no handler. `BADS()` adds one through `basicConfig`, and the `BADS` logger's levels follow KD-B2-3 (c8).
- **Fingerprints:** `fp_all.out` matches the ledger's table for every commit in scope.

## 3. Findings

### F1. contraints_check rounds its bins half to even, uCheck.m rounds them away from zero: on a search mesh finer than tol_mesh/2 it merges and removes other candidates than setdiff does
- Where: `pybads/function_logger/constraints_check.py:39,41` (`np.round`) and the comment at 34-36 ("as MATLAB's setdiff(u1, u2, 'rows')"), at `0d866e8`. The records: `verification/wave3.md:471-473` records only the representative within a bin; `dev/results/2026-09-23-codebase-survey.md:233` says "as `uCheck.m`".
- Kind: MATLAB comparison does not hold.
- Severity: substantial. The code does not do what the ruling says ("as uCheck.m"), the records say it does, and the rounding is the same one W3-14 was ruled to fix. The numerical effect is below `tol_mesh/2`.
- **Why it happens.** `uCheck.m:64,66` bins with MATLAB's `round`. The search mesh is 2^(2k−10) at the poll mesh 2^k, so from mesh 2^-6 (search mesh 2^-22) it is finer than the 2^-20 bin. The grid's coordinates then fall on exact halves of a bin: a quarter of them at 2^-6.
- **Where the bins differ.** Take an evaluated point at 0 and a candidate half a bin away (2^-21):
  - the port removes the candidate (`np.round(0.5) = 0`), while `uCheck.m` keeps it;
  - with the evaluated point one bin away instead, the port keeps the candidate and `uCheck.m` removes it (c2).
  - On random sets with grids of 2^-21 to 2^-24, the output equals `uCheck.m` in only 1289 of 3000 sets (3000 of 3000 on coarser grids).
- **In seeded runs** (c3, with a null check of 0 in c3b):
  - the ES's `contraints_check` output changes with MATLAB's round in its bins in 2/32, 10/50, 10/58, 16/66, 116/186 and 71/202 calls;
  - the point the ES returns changes in 2 and 3 of 25 searches (edge sphere, D = 2, seeds 0 and 1) and in 1 of 93 (noisy sphere, D = 2).
  - Of the noisy run's 116 changed calls, 1 changes which candidates are removed as evaluated; the rest change which candidates are merged into one bin (c4).
  - Whole runs with MATLAB's round in the bins evaluate other points in 5 of 9 runs, with the same final states in all 9 (c14b).
- **Why no check caught it.** The equality checks cited for W3-1 never had a half: fix C's `check_ucheck.py` used grids of 2^-2 to 2^-6, and the verifier's `v_ucheck.py` transcribes with `np.round`, with the comment "no halves here".
- Would the correction move results: yes, at default, in any run whose mesh reaches 2^-6, and also behind `force_poll_mesh`. How many final states move needs `population.py run --suite default` and `--suite geometry` (seeds 0-29) at the fix, compared with `population_linux_wave3_20260927` and the geometry population at `a14524d`. The orchestrator decides.
- Proposed correction, one of two:
  - **Fix it:** bin `u1` and `u2` with MATLAB's round, the exact rule `force_to_grid` uses (`frac, r = np.modf(q); r + np.sign(frac) * (np.abs(frac) >= 0.5)`), under that population gate.
  - **Or keep it and record it:** the comment becomes "keep the first vector, in input order, of each bin that holds no evaluated vector, the bins sorted as MATLAB's setdiff(u1, u2, 'rows') sorts them; the bins round halves to even, where uCheck.m's round takes them away from zero, which differs once the search mesh is finer than tol_mesh / 2". Add a sheet entry that also carries the representative choice, and qualify the survey's "as uCheck.m".

### F2. Batch 1's row: "31 of the 540 runs end at other points"; its fields file has another final x in 29
- Where: `verification/wave3.md:419` (the CHANGELOG line from `bd110f0` repeats it).
- Kind: number does not recompute.
- Severity: minor.
- **The count.** `batch1_vs_reference_fields.txt` has `final.x` in 29 runs, `final.fval` in 30, `fsd` in 2 and `min_noise_var` in 1. `w3-14_attribution.txt` lists 31 changed runs, so 2 of the 31 end at the same point with other estimates.
- **The parenthetical.** "x / tol is about 1e12, where the fraction of a double comes in steps of 2^-11" is off: at q ≈ 1e12 the spacing is 2^-13; 2^-11 holds for q in [2^41, 2^42), that is |x| ≥ 1/2 (c13).
- Would the correction move results: no.
- Proposed correction: "31 of the 540 runs change, 29 of them ending at other points", and "(at 2^-42, x / tol reaches 4.4e12, where the fractional part of a double comes in steps of 2^-11 for |x| ≥ 1/2, finer below)".

### F3. The tests of W3-9 and W3-11 stand in for the configurations the rulings name, unrecorded
- Where: `pybads/testing/bads/search/test_empty_search.py:146-196` (W3-9) and `96-126` (W3-11); `wave3.md:235-237` (W3-9: "a test with a thin band") and `:56`, gate column (W3-11: "a test with a `non_box_cons` that empties a set").
- Kind: ruling not implemented (in its wording).
- Severity: minor.
- What the tests do instead:
  - W3-9's test uses a constraint that rejects every candidate on its second call;
  - W3-11's test patches the hedge to return an empty set.
  - Both reach the code the rows changed, but "Choices within the rulings" records neither. Agent A says a brief told it to patch the ES; no recorded brief says so.
- Would the correction move results: no.
- Proposed correction: one line under "Choices within the rulings" giving both stand-ins and why.

### F4. Two items of "Found while fixing" (agent C) are false at 0d866e8
- Where: `verification/wave3.md:571-576`.
- Kind: false statement.
- Severity: minor. These items are handed to a later wave.
- The two statements:
  - "Other tests of `test_search.py` draw from NumPy's global stream": none does. Every draw there comes from a seeded `Generator` or from `bads.rng` of a seeded `BADS`. At `8aecb6a` only `test_incumbent_constraint_check` drew from the global stream, and C seeded it.
  - "With `n_search_iter = 0` the search returns its `np.empty` placeholders": in a run, `ESSearchHedge.__init__` divides by it (`search_hedge.py:58`) and raises `ZeroDivisionError` at the first search. Only an `ESSearch` built directly returns the placeholders; a value of 0.5 raises `TypeError` at `es_search.py:137` (c12).
- Would the correction move results: no.
- Proposed correction: drop the first clause. Replace the second with "with `n_search_iter = 0` a run stops with `ZeroDivisionError` at its first search (`search_hedge.py:58`), and an `ESSearch` built directly returns its `np.empty` placeholders; no check refuses a value below 1".

## 4. Outside my scope

- `es_search.py:33-35` splits the initial population with `np.round(linspace(0, mu, 3))`, where `searchES.m:111` uses MATLAB's round. The two differ when μ is odd, which happens only off default: `n_search` 1002 gives [250, 251] against MATLAB's [251, 250] (c11).
- KD-B3-5 calls `bads.m:1257-1279` "`SearchStep`'s improvement tests"; those lines are `EvalImprovement`. Its Python citation `bads.py:1956-2004` at `8aecb6a` starts at the comment, not at the empty branch (1902). This is for reviewer (d).
- The CHANGELOG entry "Rounding to the search grid" repeats F2's "31 of 540 runs end at other points". "Points evaluated again … as in MATLAB BADS" depends on F1. This is for reviewer (c).
