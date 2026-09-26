# Slice part: B2, main loop, termination, noisy re-evaluation and final estimate

Title of the report: `# B2 <track> review: main loop, termination, noisy re-evaluation and final estimate`, with `<track>` "internal" or "comparison".

## The slice

Your slice is **B2, the course of a run**: the initial evaluations, the changes a noisy run makes to the options, the main loop that alternates the search and the poll (when each runs, the mesh and the sufficient improvement of each iteration, what is recorded), the termination criteria, the re-estimation of the iterates in noisy runs, the choice of the returned point and its final estimate. The search step and the poll step themselves are slices B3 and B4, and the setup before `_init_mesh_` is B1, reviewed at the same time by others: read them where you need them to judge the course of the run (what they return, what state they change), but report on them only where the loop depends on them.

Python (`{PYBADS_REVIEW}`):
- `pybads/bads/bads.py`: `_init_mesh_`, `_init_optimization_`, `optimize` (the whole method), `_re_evaluate_history_`, `_check_mesh_overflow_`, and the display (`_log_column_headers`, `_setup_logging_display_format`, `_display_function_log_`, the messages of `optimize`);
- `pybads/utils/iteration_history.py` (`IterationHistory`);
- the options these read (grep the code), in `pybads/bads/option_configs/*.ini`.

MATLAB counterparts (`{BADS}`), for the comparison track:
- `bads.m` from the initial evaluations (`%% Initial function evaluations`) to the end of the main function: the changes under uncertainty, the incumbent and the GP definition, the head of the loop (mesh, search mesh, sufficient improvement), the decisions between search and poll (the bodies of the search and poll stages are B3's and B4's), `%% Finalize iteration` (termination, `iterList`, the re-evaluation), the final re-evaluation and estimate, the output function and the outputs;
- the subfunctions of `bads.m`: `reevaluateIterList`, `FinalEstimate`, `meshOverflowCheck`, and `EvalImprovement` as the loop and the final choice use it;
- `private/evalinitmesh.m`;
- unported: `private/fixedbads.m` and `expandvars` (fixed variables, which PyBADS refuses), `private/scatterplot.m` (plotting; the sheet has the entries).

## How PyBADS reaches this code at default options

Every run: `_init_mesh_` evaluates `x0`, evaluates it again as a test of noise when `uncertainty_handling` is left empty (the default), and evaluates a Sobol initial design (`init_sobol`, slice B7) put on the search grid; `_init_optimization_` changes several options when the target is noisy and trains the first GP (`init_and_train_gp`, slice B6). The loop then runs until `max_fun_evals` (500·D), `max_iter` (200·D), `tol_mesh` or the stall criterion (`tol_stall_iters`, `tol_fun`, `improvement_quantile`) ends it; each pass runs at most one search and, at the end of a round of up to `search_n_try` searches, a poll; the iteration number counts the polls. At uncertainty level 1 or 2 (a noisy target), from the second iteration, every poll is followed by a re-estimation of the recorded iterates from the current data (`_re_evaluate_history_`), and the run ends with a choice among the iterates by `final_quantile` and `noise_final_samples` evaluations of the chosen point. Say for every finding whether a default run reaches it, at which uncertainty level (0 deterministic, 1 noise inferred, 2 `specify_target_noise`), and which option or input reaches it otherwise.

## First questions

Answer each under its own heading:
1. **The initialization.** Does `_init_mesh_`, with the first part of `_init_optimization_`, do what `evalinitmesh.m` and `bads.m` do from the initial evaluations to the loop: the evaluations of `x0` and the test of noise (when it runs, its threshold `tol_noise`, what a noisy result changes), the number of points of the initial design as the options and the budget set it, the incumbent chosen after the design, the options a noisy run changes (which, by how much, in which order, and the evaluations reserved for the final samples), and the incumbent's `fval` and `fsd` at the start? On the internal track: is it what the docstrings and the option descriptions say?
2. **The loop and its termination.** Does `optimize` sequence an iteration as `bads.m`'s loop does: the mesh size and the search mesh size of each pass, the sufficient improvement, when a search runs and when a poll runs (`search_n_try`, `skip_poll_after_search`, the spree of successful searches and the mesh expansion it brings), what is recorded in `iteration_history` and when, and each termination criterion (`max_fun_evals`, `max_iter`, `tol_mesh`, the stall criterion, `output_fcn`): its condition, which iterations it compares, and its message? On the internal track: is each criterion what its option's description and the BADS paper say, and does each index into `iteration_history` read the iteration it means?
3. **Noisy runs: the re-estimation, the incumbent and the final estimate.** Does `_re_evaluate_history_` re-estimate each iterate as `reevaluateIterList` does (which GP, which hyperparameters, which training set, which iterates, what a failure leaves), and does the end of an iteration then move the incumbent as `bads.m` does? Does the end of the run choose the returned point and estimate it as `bads.m` and `FinalEstimate` do (the quantile, which iterates are eligible, the samples and how they are combined at levels 1 and 2, what is recorded, and what reaches the result)? On the internal track: do these steps do what the docstrings and the descriptions of `final_quantile` and `noise_final_samples` say, and is the final estimate a correct estimate of what the result documents?

---

The rest of the prompt is `wave2_common.md` (from "You are a reviewer") and the part of your track.
