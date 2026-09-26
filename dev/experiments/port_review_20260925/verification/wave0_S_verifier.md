<!-- Report of the verifier of wave 0, slice S (Sto-BADS), reading PyBADS at 95da7f1 in ../pybads-review (and ab4dded for comparison); saved verbatim from its final message on 2026-09-26. Its check scripts are kept on the orchestrator's machine only (dev/scripts/runs/LOCAL.md). -->

# Wave 0 verification: S (Sto-BADS)

I read the code at `95da7f1` (`C:\Users\luigi\Documents\GitHub\pybads-review`) and checked the two Sto-MADS algorithm pages of arXiv 1911.01012 myself (Def. 1–2, Algorithm 1, Assumption 2, Remark 1). I also read MATLAB BADS at `C:\Users\luigi\Documents\GitHub\bads`. My checks and logs are in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pybads\10994841-8958-4975-b288-1fd2bcd6013b\scratchpad\review\verify_S\`. Every script printed `pybads-review\pybads\__init__.py` and `gpyreg\v1.3.3\gpyreg\__init__.py`, used one BLAS thread and was seeded. No run went over 200 evaluations.

Between `ab4dded` and `95da7f1`, PR #71 touched none of the lines of these findings. `git diff` shows this, and `v2_instrument.py sphere 3 22 1` gives the same output on both worktrees except the reported iteration count (11 vs 12), which #71 changed on purpose.

## 1. Summary

| Finding | Classification | Reached at default options | Still present at 95da7f1 | Confidence |
|---|---|---|---|---|
| F1 Sto poll decides from the last point | confirmed defect | no (`stobads=True`, noisy target) | yes | high |
| F2 NaN estimate counts as "uncertain" | confirmed defect | no (`stobads=True` plus a failed GP rebuild or add) | yes | high |
| F3 threshold γ·ε·δ² | design question | no | yes | high on the facts |
| F4 `opp_stobads` moves uphill and widens the search | design question | no | yes | high |
| F5 `gamma_uncertain_interval` sits before `options` | confirmed defect | yes (a MATLAB-order positional call) | yes | high |
| F6 empty search set gives `UnboundLocalError` | confirmed defect, narrow reach | only with a non-deterministic `non_box_cons` | yes | high |
| F7 `u_success.append(self.u_best.copy)` | confirmed, inert | yes (every successful poll) | yes | high |

## 2. Per finding

### F1. The Sto poll decides from the last evaluated point: confirmed defect

**Lines at 95da7f1:**
- `bads.py:2284-2294`: for every point, `sto_success = self._sto_success_improvement_(...)` and then `certain_good_poll = sto_success == 1`. This overwrites the flag at each point.
- On the default path (2279-2282) the flag is set only when a new best improvement appears, so once it is true it stays true.
- `2313-2327`: the move decision uses the last point's `sto_success`, and the move goes to `u_poll_best` (the point with the largest mean improvement).
- `2329-2346`: the mesh update, driven by `certain_good_poll`.
- `_is_poll_stop_` (2523-2550) takes the same flag.

**History:** written in `9037851` (2022-09-22, "Full PyBADS and first Sto-BADS version (#5)"). Since then only `157bd09` (formatting) and `685da15` (#64, which moved the stop test into `_is_poll_stop_` without changing its logic) touched it.

**Paper:** Algorithm 1, lines 18-23, says success if the condition holds "for some" poll point, and certain failure only "for all" of them.

**Check:** `v2_instrument.py`, 12 runs: sphere D=2/3/4, Rosenbrock D=2/3, noise SD 1, and one sphere D=3 at SD 0.1, each with `opp_stobads` on and off.
- 19 polls had at least one point with outcome 1. In 9 of them the final outcome was not 1, and all 9 ended with no move and a contracted mesh.
- Example (`v2_sphere3_s22_opp1_sd1.log`): `poll 4: frame=0.0625 outs=[-1, 1, -1, -1, -1, -1] moved=False dmsi=-1`. After the success the stop check was `(2, True, 0.1854, False)`: p_less was below `1 - tol_poi`, so the poll went on, and the next point reset the flag.
- 119 of 126 polls evaluated all 2D directions, because `min_failed_poll_steps = inf` under noise (`bads.py:1130`).
- In no run would a flag that stays true have stopped the poll earlier (0 of 9). The lost early stop that the reviewer mentions is real in the code but did not matter in these runs. What matters is the lost moves and the lost mesh expansions.

**Agree / disagree:** I agree with the mechanism and the classification.

**Disposition: fix.**
- What changes: keep an "any point succeeded" flag that stays true (it drives `_is_poll_stop_` and the mesh expansion), and an "every point was a certain failure" flag.
- With opp on, move whenever the poll is not a certain failure. Otherwise move only on success.
- Default runs are unchanged, so the gate is the fingerprint of an unchanged default run, plus a before/after population comparison with `stobads=True` on noisy targets. Sto-BADS turns itself off on deterministic targets (`bads.py:1167-1169`).

### F2. A NaN estimate counts as "uncertain", and with `opp_stobads` the incumbent moves to a point with NaN value: confirmed defect

**Lines at 95da7f1:**
- `1823-1828` (search) and `2252-2257` (poll) set the estimate to NaN after a failed rebuild or a failed add. Both came in `685da15` (#64, 2026-09-25), and the comment says "so the search counts as failed".
- The rule (`2007-2028`) returns 0 for NaN, because both comparisons are false. It dates from `9037851`.
- In the search, `1899-1901` accepts outcome 0 (`is_search_improved = sto_success > -1`), then `1927-1933` moves the incumbent and sets `gp = new_gp`.

**Checks:**
- `v1_rule.py`: `(1.0, nan, 0.2, nan, 1.0) -> 0` and `(nan, 0.0, nan, 0.2, 1.0) -> 0`.
- `v3_nan.py search N stobads opp` detects the `new_gp` call from the caller's locals and injects a failed rebuild of the search GP.
  - Default path, and Sto with opp off: `moved=False`, counted as a failure (`0.0`).
  - Sto with opp on: `estimate=(nan, nan); incumbent fval -0.4184 -> nan, fsd 0.32 -> nan, moved=True, search_stats tail=0.5, reset_gp=True`.
  - The next search then compared against `f_base=nan ... -> 0` and was accepted with no real comparison.
  - With N=4, the last search of the round, the following poll went `f_base=nan ... -> 0` on all 4 points and ended `moved=False, dmsi=-1`.
- `v3_nan.py poll` makes every add after a good point fail, in a poll that continues:
  - opp off: `moved=False, dmsi=-2`;
  - opp on: `moved=True, dmsi=-1` (the mesh contracts instead of expanding).
  - On the default path the flag stays true, so a later NaN cannot erase a success.

**Where I disagree with the reviewer:** the NaN state lasts less long than the report says.
- It also clears at the next search of the round, because the unconditional acceptance moves the incumbent to a point with a finite estimate.
- "Not at all in iteration 0" cannot happen: iteration 0 has no search, since `search_count` starts at `search_n_try` (`bads.py:802-804`). So a search NaN always falls in iteration 1 or later, and the re-evaluation at the end of the iteration always runs.
- The NaN written to the iteration history is overwritten by that re-evaluation: 0 NaNs remain in every run.
- The poll variant never makes the incumbent NaN (a NaN point never becomes `u_poll_best`). It is F1's mechanism with NaN mapped to 0.

**Disposition: fix.**
- In `_sto_success_improvement_`, return −1 when `mu` or `epsilon` is not finite. This makes the Sto path agree with the comment at 1824-1826. Fixing F1 covers the poll variant.
- Add a `stobads=True` variant of `test_noisy_search_after_failed_rebuild_counts_as_failure`.
- Gate: fingerprint only. Default runs are untouched, and a run changes only when a GP failure occurs.

### F3. The uncertainty threshold multiplies the GP SDs by frame_size²: design question

**Lines at 95da7f1:** `2007-2018`, which is `gamma * epsilon * frame_size ** power` with ε = sqrt(s_base² + s_new²). The call sites pass `self.mesh_size` (1896, 2291). The option comment is at `advanced_bads_options.ini:62-63`. All of it dates from `9037851`.

**The code does what the option documents.** The reviewer's point about the mathematics holds.
- In Sto-MADS, ε_f is "a fixed constant" (Def. 1), and resampling makes the estimates ε_f δ²-accurate (Assumption 2(ii); Remark 1: δ_p "adaptively controls the variance").
- Here ε is the current GP SD, which is not made to shrink like δ². So the "certain" labels carry no certainty at small δ.

**Checks:**
- `v1_rule.py` gives the probabilities of each outcome when the two true values are equal:

  | mesh | P(1) | P(0) |
  |---|---|---|
  | 2⁰ | 0.025 | 0.950 |
  | 2⁻¹ | 0.312 | 0.376 |
  | 2⁻² | 0.451 | 0.097 |
  | 2⁻³ | 0.488 | 0.024 |

- `v2_instrument.py`, 12 runs:
  - The rule was called at frame ≤ 1/4 in 83–92% of calls.
  - Among outcomes 1, the median of mu/ε was 0.04–0.16, and 97–100% had mu/ε < 1.96.
  - Among outcomes −1, 67–94% had mu/ε > −1.96, so they were not certain failures at 95%.
- Against BADS's own success test (mu > max(δ^1.5, tol_fun)), the Sto rule is almost always the looser one:
  - "outcome 1 but default test fails" in 1–8 calls per run at SD 1, and in 33–34 of 36–37 at SD 0.1;
  - "default test passes but outcome ≠ 1" in 0 calls (one run: 1).
- The three classes also do not reach the mesh. Certain and uncertain failures both contract by one step. The extra step for a certain failure has been commented out since `9037851` (2348-2352). So the classes matter only for success and for the opp move (F4).

**The decision:** what ε means.
- **(i) Keep power 2 and the GP SD (current).** The rule is a sufficient-decrease test scaled by the noise, close to a sign test at small δ, and looser than BADS's own test. The rule's docstring should stop calling it Sto-MADS's certain/uncertain interval.
- **(ii) Power 0.** A z-test at level γ at every δ; the classes keep a probabilistic meaning. Successes and moves become rarer, and opp's uncertain band grows to ±1.96ε at every δ (see F4).
- **(iii) A fixed ε_f (for example `noise_size`) with δ².** This is Sto-MADS's form, but still without its accuracy requirement.

Any change behind `stobads` needs the fingerprint of a default run plus a population comparison with `stobads=True` on noisy targets. Decide F3 together with F4.

### F4. `opp_stobads` accepts worse estimates and widens the search: design question

**Lines at 95da7f1:**
- `1899-1901` accepts any outcome > −1.
- `1914-1924` logs such a move as "incremental".
- `_update_search_stats_` (`2695-2700`) multiplies `search_factor` by `search_scale_incremental = 2` (ini:98). MATLAB uses the same factors (bads.m:235-236).
- The option comment (ini:60-61): "Move incumbent even for the uncertain unsuccess". Sto-MADS's "unsuccessful and uncertain" (Def. 2) covers both sides of zero, so the code does literally what the comment says.
- Sto-MADS itself never moves on an uncertain failure (Alg. 1, line 23). All of this dates from `9037851`.

**Check:** `v2_instrument.py`, 6 runs with opp on.
- 29 searches came out uncertain; all 29 moved, and 17 of them moved to a worse estimate.
- The uphill moves were small, with mu/ε between −0.34 and 0.00, all at frame ≤ 1/2, because of F3's δ².
- The search factor doubled after each of them, for example `(1.0, 2.0)`, `(2, 4)`, `(1.414, 2.828)`, unless the round ended and reset it to 1.
- `adaptive_incumbent_shift` is False by default, so the search factor is the only thing affected.

**Agree / disagree:** I agree with the facts. At δ=1 the band reaches ±1.96ε, but I saw no uncertain search at δ=1, since the first search round follows a poll that usually contracts the mesh.

**The decision:**
- (a) Which moves opp allows: the whole uncertain band (current), only mu > 0 (the analogue of `sloppy_improvement`), or none (Sto-MADS, which is `opp_stobads=False`).
- (b) What an uncertain move does to the search factor: ×2 as "incremental" (current), ×1, or the failure factor.

Gate: as in F3.

### F5. `gamma_uncertain_interval` is an undocumented 8th positional parameter before `options`: confirmed defect

**Lines at 95da7f1:**
- Signature at `154-165`. The docstring (`79-86`) lists `options` right after `non_box_cons` and never mentions γ.
- The attribute is set at `251`. The rule (2009-2012) tests the attribute for None but uses the argument as the value.
- MATLAB's order is `bads(fun,x0,LB,UB,PLB,PUB,nonbcon,options,...)` (bads.m:1).
- History: γ was inserted in `9037851`, and `user_options` was renamed `options` in `f9e9326` (2022-11-02). The docs and examples pass `options=` by keyword.

**Check (`v1_rule.py`):**
- `BADS(*args, None, opts)` gives `gamma_uncertain_interval = {'display': 'off', 'max_fun_evals': 30, 'random_seed': 3, ...}` and `options: max_fun_evals = 1000 stobads = False display = iter random_seed = None uncertainty_handling = None`. Every option is silently dropped, including the seed.
- A dict passed as γ with Sto on raises `TypeError: unsupported operand type(s) for *: 'dict' and 'float'`.
- The attribute/argument mismatch is inert, because both call sites pass the attribute. Called directly it gives `attr=100, arg=None -> TypeError` and `attr=None, arg=100 -> 1` (the argument is ignored).

**Disposition: fix.**
- Make γ keyword-only after `options`, or turn it into an option, and document it. Use the attribute in the rule.
- A script that passes γ as the 8th positional argument would change, so this needs a CHANGELOG entry and an "Upgrading from" line.
- Gate: fingerprint.

### F6. An empty search set ends in `UnboundLocalError`: confirmed defect, narrow reach

**Lines at 95da7f1:**
- The empty-set branch `1852-1857` never assigns `u_search`.
- The crash comes at the return (`1962`) on the default path and on Sto with opp off, and at `_update_incumbent_` (`1927-1929`) on Sto with opp on.
- `optimize()` never uses the returned `u_search` (1325-1331).
- The branch dates from `c7c88ab` (initial port). MATLAB has the same branch (bads.m:667-672) and does not assign `usearch` either. The reviewer's "MATLAB: no counterpart" is wrong.

**Can a configuration reach it?**
- **Periodic variables: no.** They are refused at construction (`bads.py:617-620`), and `period_check` is an identity stub (`pybads/utils/period_check.py`).
- **A deterministic `non_box_cons`: no.** The ES (`es_search.py:145-156`) already puts every candidate through the same `force_to_grid(search_mesh_size)` and the same `contraints_check` (same bounds, tolerance, logger, projection and constraint). Nothing is evaluated in between, so the second filter keeps the point.
- **A non-deterministic `non_box_cons`: yes.** In `v4_empty_search.py random 0.05`, each point is infeasible with probability 0.05, and the run is otherwise at default options with noise. Output: `UnboundLocalError ... bads.py:1962 in _search_step_: return u_search, ...` after 51 evaluations. With Sto+opp: `... bads.py:1928`.
- **An ES with no candidate: a different crash.** `v5_es_empty.py` gives `IndexError ... es_search.py:216: return us[0], z[0]`. There MATLAB's `searchES` returns an empty set (`searchES.m:209`) and `bads.m` goes on. I found no natural deterministic trigger for this: a thin feasible band crashed earlier, see §3.

**Disposition: fix.**
- In the empty branch, count the search as a failure on every path and skip `_update_incumbent_`.
- Optionally have the ES return an empty set as MATLAB does.
- No run that completes today changes, so the gate is the fingerprint.

### F7. A successful poll appends a bound method to `optim_state["u_success"]`: confirmed, inert

**Lines at 95da7f1:** `2340` appends `self.u_best.copy` without the call. It dates from `c7c88ab`; the search path (1918) appends an array. Nothing in `pybads/`, `docsrc/` or `examples/` reads `u_success`.

**Check:** `v6b_u_success.py`, Rosenbrock D=3 at default options.
- `entry: <built-in method copy of numpy.ndarray ...>`, and calling it returns the coordinates.
- `deepcopy gives the same object: True`.
- `output_fcn's copy holds methods: 1`.

**Where I disagree with the reviewer:**
- MATLAB has a counterpart: `optimState.usuccess = [optimState.usuccess; ubest]` (bads.m:968), a list that starts at the initial point (bads.m:460). PyBADS splits it into `optim_state["usuccess"]` (set once at 1173) and a list without the initial point (1177).
- Since #71, `output_fcn` gets a deep copy of `optim_state`, so the method now reaches user code. It still has no effect on results.

**Disposition: fix** (the call, and optionally make the key match MATLAB's). Gate: fingerprint. Results do not change, since nothing reads the key.

## 3. Met while verifying (unverified unless stated)

1. **`search_factor_min = 0.5` (ini:101) is never read** (seen by reading and grep; no run). MATLAB floors the factor after a failed search: `max(options.SearchFactorMin, ...)` (bads.m:1366). PyBADS's failure branch (`bads.py:2706-2711`) has no floor. This affects default runs: 3 failed searches in a row give √0.5³ ≈ 0.35 < 0.5. The known-differences sheet does not mention it.
2. **A thin feasible region crashes the first GP fit** (reproduced, not analysed). With `|x1-x2| <= 0.005`, `v4_empty_search.py band` evaluated only x0 before a `ValueError` from gpyreg `set_priors` ("covariance_log_outputscale has an infinite mu"), after `log(np.std(gp.y))` of 0 at `gaussian_process_train.py:357`.
3. **`bads.py:1501` assigns `self.best_u`, which nothing reads.** MATLAB's matching block (bads.m:1111-1117) also updates only `u`, and both reset `u` to the best point after the next search (bads.py:1372, bads.m:769). It looks port-faithful. I did not check whether the re-evaluated `fval` then describes a different point.
4. **`bads.py:2116` calls `np.vstack(u_poll, u_poll_new)` with two arguments,** which would raise if reached. It is reachable only when `poll_mads_2n` returns an empty basis.
