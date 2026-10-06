# Slice part: B7, the function logger, the initial design and the utilities

Title of the report: `# B7 <track> review: function logger, initial design, utilities`, with `<track>` "internal" or "comparison".

## The slice

Your slice is **B7, the function logger, the initial design and the utilities**: how every evaluation of the target is made and recorded (the call of the target in the original space, the checks of what it returns, the rows of the log and their growth, the merge of a repeated point, the evaluations that are not recorded, the counts and the evaluation times, the final trimming of the log, the import of an evaluation made elsewhere); how the initial design is drawn (its seed, its size, its generator, its map onto the plausible box); and the stub for periodic variables. The main loop and the setup that call these, `_init_mesh_` among them, are slice B2 and B1, the search and `contraints_check` B3, the poll B4, and the GP's training set, which reads the log, B5, all reviewed in earlier waves: read them where you need them to judge this slice (what they pass to the logger and to `init_sobol`, what they read back), but report on them only where they depend on this slice.

Python (`{PYBADS_REVIEW}`):
- `pybads/function_logger/function_logger.py` (`FunctionLogger`: `__init__`, `__call__`, `add`, `finalize`, `reset_fun_eval_time`, `_expand_arrays`, `_record`); `function_logger/constraints_check.py`, in the same package, is slice B3's;
- `pybads/init_functions/init_sobol.py` (`init_sobol`), and its call in `pybads/bads/bads.py`, `_init_mesh_` (read as far as the design depends on it);
- `pybads/utils/period_check.py` (`period_check`) and its call sites;
- the construction of the logger in `BADS.__init__` and the reads of its fields elsewhere (grep `function_logger`), as far as they depend on what the logger holds;
- the options these read (grep the code), in `pybads/bads/option_configs/*.ini`.

MATLAB counterparts (`{BADS}`), for the comparison track:
- `private/funlogger.m` (`'init'`, `'iter'`, `'single'`, `'done'`), and its calls in `bads.m` and `private/evalinitmesh.m`;
- `init/initSobol.m`, `init/private/i4_sobol_generate.m`, `init/private/i4_sobol.m`, `init/private/i4_bit_hi1.m`, `init/private/i4_bit_lo0.m`, and the call of `options.InitFcn` in `private/evalinitmesh.m`;
- `utils/periodCheck.m`;
- unported: `init/initLHS.m` (also `initSobol.m`'s fallback), `init/initRand.m`, `init/private/lhs.m`; unused: `init/private/tau_sobol.m` (the sheet has the entries).

## How PyBADS reaches this code at default options

Every run. `BADS.__init__` creates the logger with the transform of the variables and `cache_size` rows (500), holding the noise SDs only at uncertainty level 2. Every evaluation of the target goes through `FunctionLogger.__call__`, which maps the point from `u` space to the original space and calls the target with a 1-D array: the start `x0`; a second evaluation of `x0` as the noise test when `uncertainty_handling` is left empty (the default), with `record_duplicate_data=False` and its time kept out of the total; the initial design; each search and poll point; and, in a noisy run, the `noise_final_samples` (10) re-evaluations of the returned point, with `record_duplicate_data=False`. At level 2 the target returns a tuple `(f, sd)`, and a point evaluated again is merged into its row. The log's `X`, `Y`, `S`, `X_flag`, `Xn`, `n_evals` and `func_count` are read by the GP's training set, by `contraints_check`, by the choice of the start after the design and by the loop's counts; `total_fun_eval_time` gives the result's `overhead`. `init_sobol` is called once, in `_init_mesh_`, with the start `u0` in `u` space on the search grid and `fun_eval_start` (D by default; in a run that the noise test finds noisy, `max(20, D)` capped at `max_fun_evals`), itself capped at `max_fun_evals - 1`; `_init_mesh_` then keeps the design's first points within the evaluations left, puts them through `period_check`, onto the search grid and through `contraints_check`, evaluates them in order, and starts from the lowest value. `period_check` is a stub called at the initial design, the search and the poll; periodic variables are refused (the sheet's KD-B1-6). Say for every finding whether a default run reaches it, at which uncertainty level (0 deterministic, 1 noise inferred, 2 `specify_target_noise`), and which option or input reaches it otherwise.

## First questions

Answer each under its own heading:
1. **The evaluation.** Does `FunctionLogger.__call__` evaluate the target and check what it returns as `funlogger.m`'s `'iter'` and `'single'` do: the point the target receives (the transform back to the original space, its shape), the outputs at each uncertainty level, the checks of the value and of the SD and what they accept, the handling of an error raised by the target, and the count of evaluations? On the internal track: is it what the class's docstrings, the descriptions of `uncertainty_handling` and `specify_target_noise`, and the documentation under `docsrc/source/` (the target function in `api/advanced_docs.rst`, the class in `api/classes/function_logger.rst`) say?
2. **The record.** Does `_record` store each evaluation as `funlogger.m` does: a new row per evaluation, the growth of the arrays against MATLAB's fixed cache, the merge of a repeated point at level 2 (the sheet's KD-B7-3: does the code match the entry?), the paths that do not record (which evaluations take them, and what they leave in the log, in `n_evals` and in the times), `X_flag`, `Xn` and `Y_max`, `fun_eval_time` and `total_fun_eval_time`, and `finalize` against `'done'`? Does `add` do what its docstring says, and what MATLAB's import of earlier evaluations does (the sheet's KD-B1-4 on `fun_values`)? Do the readers of the log in the other slices get what they expect from each field, at each level?
3. **The initial design.** Apart from the substitution the sheet settles (KD-B7-1), does `init_sobol` do what `initSobol.m` does: the seed (from `u0` when it is finite, and from the generator otherwise), the number of points and the rounding to a power of two (and when it is raised further), the map onto the plausible box, what it returns and what the caller uses; and does `_init_mesh_`'s handling of the design follow `evalinitmesh.m` (the cap, the periodic check, the grid, `uCheck`, the order of evaluation, the choice of the start)? On the internal track: is the design what the docstring and the comment on Owen (2020) describe, a balanced space-filling set of the plausible box; how does it depend on the start, on `random_seed` and on the platform; and is every input it can receive at default options handled?
4. **The utilities.** Does `period_check` do what `periodCheck.m` does for the inputs it can receive today? Is the sheet's KD-B1-6 (periodic variables refused) what the code does?

---

The rest of the prompt is `wave4_common.md` (from "You are a reviewer") and the part of your track.
