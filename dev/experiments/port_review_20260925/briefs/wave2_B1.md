# Slice part: B1, setup, options, defaults, bounds, transform and result

Title of the report: `# B1 <track> review: setup, options, defaults, bounds, transform and result`, with `<track>` "internal" or "comparison".

## The slice

Your slice is **B1, the setup of a run and its result**: how a `BADS` object is created from the user's arguments and options, the default options, the checks of the bounds and the starting point, the transformation of the variables, the state that the run starts from, the random generator, and the result that the run returns. The main loop, the initial evaluations and the final estimate are slice B2, reviewed at the same time by others: read them where you need them to judge the setup (what reads an option, what reads the state the setup leaves), but report on them only where the setup depends on them.

Python (`{PYBADS_REVIEW}`):
- `pybads/bads/bads.py`: `__init__`, `_bounds_check_`, `_init_optim_state_`, `_variable_transformer_`, `_init_rng_`;
- `pybads/bads/options.py` (the `Options` class and `_read_config_file`), `pybads/bads/option_configs/` (the two option files `basic_bads_options.ini` and `advanced_bads_options.ini`, and `options_confs.py`; `test_options.ini` and `test_options2.ini` are test fixtures shipped in the package);
- `pybads/variable_transformer/` (`VariableTransformer`, `maskindex`), `pybads/search/grid_functions.py` (`grid_units` only), `pybads/rng.py`, `pybads/bads/optimize_result.py`;
- the package's `__init__.py` files (what they export);
- the documentation of the interface: the docstring of `BADS`, `docsrc/source/api/` (the pages of `BADS`, `OptimizeResult` and the options) and `docsrc/source/quickstart.rst`.

MATLAB counterparts (`{BADS}`), for the comparison track:
- `bads.m` from its start to the initial evaluations: the help text, `defopts` (the basic options and the advanced ones), and the setup up to the call of `evalinitmesh` (inputs, display level, starting point, fixed variables, options, variables, logger); and `bads.m`'s output arguments at the end;
- `private/setupoptions.m`, `private/setupvars.m`, `private/boundscheck.m`, `private/bads_output.m`;
- `utils/transvars.m`, `utils/origunits.m`, `utils/gridunits.m`, `utils/maskindex.m`, and `utils/evalbool.m`, which has no counterpart (the option files hold Python literals);
- unported: `private/fixedbads.m` and `expandvars` (fixed variables, which PyBADS refuses; the sheet has the entry).

## How PyBADS reaches this code at default options

Every run goes through all of it once. `BADS.__init__` reads the basic option file, the user's `options` dict and the advanced file (which skips the keys the user set), evaluating the files' expressions with `D` bound; creates the generator `self.rng` from `random_seed`; sets the level of the `BADS` logger from `display`; checks and adjusts the bounds and the starting point in `_bounds_check_`; draws a random `x0` when it is missing or not finite; and builds `optim_state` in `_init_optim_state_`: the transformation of the variables (a log transform where `nonlinear_scaling` is on and the bounds allow it), the transformed bounds, the starting point put on the search grid, the mesh sizes and `tol_mesh` in the transformed space, the uncertainty level from `uncertainty_handling`, `specify_target_noise` and `noise_size`, and the GP settings. At the end of `optimize()`, `OptimizeResult` reads the object's state. Several options are rewritten later, at the start of `optimize()` in a noisy run (B2's): judge a default by what the code does with it. Say for every finding whether a default run reaches it, at which uncertainty level (0 deterministic, 1 noise inferred, 2 `specify_target_noise`), and which option or input reaches it otherwise.

## First questions

Answer each under its own heading:
1. **The options and their defaults.** On the comparison track, the full defaults table: every `defopts` entry of `bads.m`, basic and advanced, against its counterpart in the option files, evaluated at D = 1, 2, 6 and 20 (the value on each side, and whether PyBADS's code reads the option at all, so that a difference in an option nothing reads is marked as such); then the processing of `setupoptions.m` (the evaluation of string values, the combinations of the noise options and what each side refuses or warns about). On the internal track: is each default what its description says, does each description say what the code does with the option, and does the layering of the option files and the user's dict do what the documentation says, including for a user value of `None`, of the wrong type or given as an expression?
2. **The bounds and the starting point.** Does `_bounds_check_`, with the handling of missing bounds and of a missing or non-finite `x0` in `__init__`, do what `boundscheck.m` and the setup of `bads.m` do: each check and its order, which inputs are refused, how `x0` and the plausible bounds are moved, the effective bounds, the plausible bounds estimated from a set of starting points, and `non_box_cons` at the start? On the internal track: is what is accepted and refused what the docstring of `BADS` and the documentation say?
3. **The transform and the grid.** Does `VariableTransformer` compute what `transvars.m` and `origunits.m` do: which variables get a log transform and why, the map of the plausible box onto `[-1, 1]`, the transformed hard bounds (infinite ones included), the inverse, and `maskindex`; and are `grid_units`, the gridization of the starting point, the search bounds and `tol_mesh` as in `setupvars.m`? Check the transform and its inverse on random inputs, with and without log-transformed variables and with infinite bounds.
4. **The result and the seed.** Does `OptimizeResult` report what its docstring, the documentation and `bads_output.m` with `bads.m`'s outputs say, field by field (among them `x`, `fval`, `fsd`, `iterations`, `func_count`, `success`, `status`, `message`)? Does `_init_rng_` with `rng.py` do what the `random_seed` option's description says for each kind of value it accepts?

---

The rest of the prompt is `wave2_common.md` (from "You are a reviewer") and the part of your track.
