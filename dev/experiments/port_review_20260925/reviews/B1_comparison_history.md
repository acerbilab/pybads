<!-- Addendum of the B1 comparison reviewer to reviews/B1_comparison.md, wave 2 of the port review: its reply to the orchestrator's note that the sandbox's clone, shallow when it began (oldest commit ce3a0b3), now held the complete history. Saved verbatim from its message on 2026-09-26. It re-dates the report's Python lines with git log -L on the full history and changes no finding. -->

History corrections to the B1 comparison report; no finding, classification or conclusion changes.

I re-dated every Python line cited in the findings with `git log -L` on the full history now in /home/user/pybads-review.

Three dates in the report were wrong:
- F4 (`search_factor_min` unread): the failure branch of `_update_search_stats_`, which has no floor, dates from c7c88ab (2022-06-02), not 8e59038. The option has sat unread in the .ini since c7c88ab too.
- F9 (`fun_values`, `f_vals`): the `for i in range(len()):` loop dates from 8e59038 (2022-06-03), not 9037851. The `fun_values` validation, `f_vals` and `cache_active` with its 7-field display format date from c7c88ab.
- F12 (input validation): the `non_box_cons` check was MATLAB-like in c7c88ab (`y.shape[0] != 2 or y.shape[1] != 1`). It was weakened to `y.shape[0] != 2 and y.ndim == 1` in 8e59038 (2022-06-03), not "by 9037851", and last touched in f9e9326 (2022-11-02).

Three dates were imprecise and are now exact:
- F6 (deep copy in `OptimizeResult`): written in 8ff10f5 (2022-11-04), the commit that created `OptimizeResult`, and reformatted in 157bd09.
- F7 (`status` never set, `success` always True): 8ff10f5 created `OptimizeResult` with 'status' in the key list, the line that would set it commented out (`#'status',`), and `self['success'] = True  # TODO`. Later commits touched these lines without changing either behaviour: 2650aef and d466948 (2022-11) for the keys, 060dd7e (2023-02-17) for the success line. `status` has never been set.
- F13 (`overhead`): the logger's accounting dates from c7c88ab. The lines that leave repeats out of `total_fun_eval_time` were touched in 8ff10f5.

These were already correct:
- F1: effective bounds from c7c88ab, last touched in 8bb3d59. The plb←lb default from 9037851 (2022-09-22), touched in a029e63 (2024-02-22).
- F2: the half-bounds test from c7c88ab, unchanged since.
- F3: `tol_noise` (advanced_bads_options.ini:13) from c7c88ab.
- F5: the (1, D) shape check without scalar expansion from c7c88ab, touched in d466948 (2022-11-10).
- F8: options.py:48-51 from c7c88ab.
- F11: `display` handling from c7c88ab.
- `cache_size` = 500 from c7c88ab.

Every cited MATLAB line predates 2022-02-11. For all 13 findings, the Python never matched MATLAB, and MATLAB did not change after the Python was written.
