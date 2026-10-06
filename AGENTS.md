## About this file

This file states what holds in this repository, for an agent who would not
meet it at the point of need: couplings that span files, procedures that gate
a change, traps that fail silently or at a cost, and conventions that nothing
enforces. It is not a record of changes: what a change did belongs to its
commit and pull request. What an agent meets where it matters stays there, in
a module's docstring, an option's description or the user documentation. Add
a line only when an agent without it would go wrong, and write it as a fact
about the repository as it stands.

## The project

PyBADS is the Python port of the MATLAB BADS toolbox (Bayesian Adaptive
Direct Search, `acerbilab/bads`, the reference implementation): optimization
of black-box, possibly noisy, mildly expensive objectives with up to about
20 parameters, under bound and optional non-box constraints. Plain
NumPy/SciPy. The GP layer is the lab's `gpyreg` (`acerbilab/gpyreg`), a
sibling repository.

PyBADS does not import PyVBMC. It carries its own copies of classes that
PyVBMC also has (`Options`, `FunctionLogger`, `IterationHistory`, `Timer`,
`stats/`), which have diverged from PyVBMC's: a fix in one repository does
not reach the other. `VariableTransformer` is PyBADS's own and is not
PyVBMC's `ParameterTransformer`, although its documentation page is
`parameter_transformer.rst`.

- `dev/` holds the developer notes, plans, results and tooling.
  `dev/README.md` says where each kind of record goes; `dev/TODO.md` lists
  the open work.
- `pybads/bads/README.md` is the catalogue of the deliberate differences
  between PyBADS and MATLAB BADS v1.1.3, with the open porting work.
  `dev/results/2026-09-28-port-correctness-review.md`, the consolidated
  ledger of the review that compared the port with MATLAB BADS, gives
  every finding its verdict and its fix or `dev/TODO.md` item. Check both
  before treating an oddity in the numerical code as intended or as a
  defect; a change that adds or removes a deliberate difference updates
  the catalogue.
- `docsrc/` is the Sphinx source. `docs/` is its gitignored build output;
  the published site lives on the `gh-pages` branch.

## Setup and commands

The development environment is a venv at `.venv` (gitignored), with gpyreg
installed from a sibling checkout:

```console
python -m venv .venv                                    # then activate it
git clone https://github.com/acerbilab/gpyreg ../gpyreg
pip install -e "../gpyreg[dev]"
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

The version comes from git tags through setuptools_scm, which writes the
gitignored `pybads/_version.py`. `test_init_conf.py` reads the installed
package metadata, so the suite needs the editable install, not only the
source on `sys.path`.

```console
python -m pytest -x -vv                                 # what CI runs
python -m pytest pybads/testing/bads/test_bads_optimization.py::test_sphere_opt
```

The tests live in `pybads/testing/`, mirroring the package, and default
discovery is limited to them (`testpaths` in `pyproject.toml`); the checks
under `dev/scripts/` run only when named by path.

The test job is defined once, in `.github/workflows/test-matrix.yml`. It
installs PyBADS, which brings gpyreg from PyPI, and then gpyreg's checkout
at the commit pinned as `GPYREG_PIN` there, which replaces it whatever
version the checkout reads from its tags (pip only reports a conflict with
the minimum in `pyproject.toml`). The pin is the tagged commit of the
release that `pyproject.toml` names as the minimum, and a change that
needs a newer gpyreg moves both.
`merge-tests.yml` runs the full matrix
(Ubuntu, Windows, macOS × Python 3.10–3.12) on a PR to `main` or to a
`dev*` branch, only when its changes against that base touch `pybads/`,
`pyproject.toml` or `setup.py`; a PR that changes anything else, the
workflows included, runs no tests. `tests.yml` runs the
full matrix on dispatch and on the 13th and 28th of each month, the
scheduled run against gpyreg's `main` instead of the pin (the drift
detector), and a smoke run (Ubuntu, Python 3.12) on each push to a `dev*`
branch that touches the package. `docs.yml` rebuilds the docs on every push
to `main` and commits them to `gh-pages`.

A release is a tag `vX.Y.Z` on `main` and a GitHub release published from
it. Before the pull request that carries it to `main`, the changelog's
`Unreleased` section becomes `[X.Y.Z] - <date>` under a new, empty
`Unreleased`, and the GitHub release takes that section as its notes,
without its heading and with each paragraph and list item joined onto one
line: a release's notes render every newline as a line break, and the
changelog wraps its lines at 79 characters.
The same pull request sets `RELEASE_DATE` in `pybads/_release.py` to that
date, from which the old-release reminder tells the age of the installed
release. `pybads/testing/bads/test_release_reminder.py` fails while the two
disagree; CI runs it on a pull request whose changes touch `pybads/`, as
those that carry a release to `main` do.
Before that pull request, too, the example notebooks, whose outputs ship
with the release and which nothing else runs, are rerun and their outputs
committed: `make -C examples/scripts run`, with the venv's Python first on
`PATH` (the target calls `python`) and gpyreg at the release that
`pyproject.toml` names as the minimum. The target prints the locations and
installed versions of the PyBADS and gpyreg that it runs, and the commit of
a gpyreg checkout, whose installed version can belong to another
installation. Example 5 shows the result's `version`, which PyBADS reads
from the installed package's metadata, and the run comes before the tag, so
for it PyBADS is installed under the release's version,
`SETUPTOOLS_SCM_PRETEND_VERSION=X.Y.Z pip install -e . --no-deps`, and `pip
install -e . --no-deps` after it takes the version from the tags again.
`release.yml` builds the package with `build.yml` and uploads it to PyPI by
trusted publishing, through the `pypi` environment, which admits only `v*`
tags. No token is stored. After the upload, the version bot of
`conda-forge/pybads-feedstock` opens an update PR and merges it once its
CI passes (the `bot` settings in the feedstock's `conda-forge.yml`;
`conda-forge/gpyreg-feedstock` has the same). The recipe keeps its own
copy of the requirements, which the bot is not to be relied on to update:
for gpyreg 1.4.0 it changed the version and the hash alone. A release that
changes the dependencies, or what its tests need, changes the recipe by
hand, in a PR opened before the bot's or pushed to it before it merges;
removing `[bot-automerge]` from the bot's title keeps it from merging
itself.

Formatting is enforced by the pre-commit hooks alone (black at line length
79 on every Python file and the notebooks' code cells, isort with the black
profile, pycln); no CI job checks it, and the whole tree passes them. The
commit that first formatted the tree is listed in `.git-blame-ignore-revs`;
`git config blame.ignoreRevsFile .git-blame-ignore-revs` hides it from
`git blame`.

`pyproject.toml` is authoritative; `setup.py` is a shim. The `packages` of
`pyproject.toml` lists every directory of the package, those that hold
only data (the `.ini` option files, the oracles' fixtures) included, so a
new subpackage is added there; the build warns "Package would be ignored"
for one that is missing. A listed package's modules reach the sdist and
the wheel through `packages`, which takes every `.py` file of its
directory, committed or not; its other files come through
`include-package-data` and setuptools_scm's file finder, which takes only
files tracked by git (the notebooks of `pybads.examples` also through the
pattern of `package-data`). A release ships the files of its commit, with
the generated `pybads/_version.py`, because `build.yml` builds from a
fresh checkout; a build from a working tree, `python -m build` included,
also takes an uncommitted module. The tests ship in the wheel: the
conda-forge recipe runs them from the installed package
(`pytest --pyargs pybads`). What they need is the `test` extra, which CI
installs and `dev` includes; no package module imports pytest.

Docstrings are numpydoc, with `OptimizeResult`
(`pybads/bads/optimize_result.py`) as the reference style. Nothing generates
the API pages: a new public class or function needs a hand-written `.rst`
under `docsrc/source/api/` and an entry in the toctree that owns it
(`documentation.rst` for a headline page, `api/classes/classes.rst` or
`api/functions/functions.rst` otherwise). The options page includes the two
`.ini` files verbatim, so the comment above each option is its user
documentation. The FAQ, `docsrc/source/faq.md`, quotes option names and
defaults, messages, display labels and result fields, and nothing runs its
snippets or checks them against the code: a change to one of those is made
in the FAQ by hand. Its table of contents is written out by hand too,
`skills/pybads/SKILL.md` names its sections and questions by their titles,
and other files link its labels, so a question added or renamed, or a
label changed, is updated there as well:

- `faq-which-kind-of-problems-is-pybads-suited-for`: Example 1;
- `faq-what-do-i-do-if-pybads-is-not-suited-for-my-problem`: `README.md`
  and `index.rst`;
- `faq-how-do-i-know-whether-a-newer-version-of-pybads-exists`:
  `installation.rst`;
- `faq-conda-installs-an-older-version-of-pybads-why`: `README.md` and
  `installation.rst`;
- `faq-my-objective-function-requires-additional-datainputs-how-do-i-pass-them-to-pybads`:
  `quickstart.rst`;
- `faq-how-do-i-run-pybads-from-several-starting-points`,
  `faq-how-do-i-choose-plb-and-pub` and
  `faq-on-some-problems-pybads-seems-to-get-stuck-and-stop-too-early-is-there-a-way-to-tune-pybads-to-optimize-towards-a-higher-precision-result-or-to-have-it-optimize-for-longer`:
  the runtime tips;
- `faq-noisy-objective-function`: Examples 3 and 4, and the runtime tips;
- `faq-can-pybads-handle-any-arbitrary-amount-of-noise-in-the-objective`:
  Example 3;
- `faq-does-pybads-support-periodic-variables-such-as-angles`: Example 6,
  the changelog's entry "Periodic variables", "What's new" in `README.md`
  and `index.rst`, and the runtime tips;
- `faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value`: the
  changelog's entry "Fixed variables", "What's new" in `README.md` and
  `index.rst`, and the runtime tips;
- `faq-how-do-i-silence-pybads-or-send-its-output-elsewhere`: the API
  page of `check_for_updates`;
- `faq-how-do-i-make-a-run-reproducible`: Examples 3 and 4, the Quick
  start of `README.md` and `quickstart.rst`;
- `faq-i-used-bads-in-matlab-what-is-different-in-pybads`: "What's new" in
  `README.md` and `index.rst`;
- `faq-i-have-run-pybads-on-my-problem-how-do-i-run-pyvbmc`: the runtime
  tips;
- `faq-pybads-crashes-saying-that-the-returned-function-value-must-be-a-finite-real-valued-scalar-what-do-i-do`:
  the error of `FunctionLogger` for a value that is not a finite real
  scalar (`pybads/function_logger/function_logger.py`).

The runtime tips (`pybads/bads/_tip_catalog.py`) link the published FAQ,
which is built from `main`, and an installed release keeps its links: a
label that a released tip or message links is not renamed. Each tip also restates in
brief the advice of the answer that it links, with its quantities, so a
change to that advice is made in the catalogue as well. The tip on PyVBMC
also links the lab's page of its model-fitting tools,
https://acerbilab.org/model-fitting/ (the repository
`acerbilab/model-fitting`), at its root, and restates what that page says
of PyVBMC, its limit of about 10 parameters included; nothing in this
repository sees a change to that page.

Advice that the FAQ gives is linked from elsewhere rather than written out
again, but for the runtime tips (above) and its list of the problems that
PyBADS suits: `index.rst`
includes the list between the markers `suited-for`, which it matches
exactly (a changed marker drops the list from the index, and the build
does not fail), and `README.md` copies it, so a change to the list is made
in `README.md` too.
Build with `make github` in `docsrc/` (`.\make.bat github` from cmd
on Windows), which copies the result into `docs/`.

The notebooks in `examples/` ship in the wheel as `pybads.examples`
(`python -m pybads` opens them) and are rendered without execution by the
docs build; no CI job runs them, so a change that breaks one goes
unnoticed. `examples/scripts/*.py` are generated from the notebooks by
`examples/scripts/Makefile` (GNU Make, with nbconvert, IPython, and black
and isort at the pre-commit hook versions, in the environment `python`
names); regenerate them with `make -B -C examples/scripts`, do not edit
them. `make -C examples/scripts run`, which needs ipykernel too, reruns the
notebooks in place; nbconvert called directly saves each line that PyBADS
prints as an output of its own, and the times of each cell's execution in
its metadata.

## Architecture

`BADS._optimize_()` in `pybads/bads/bads.py`, which the public
`BADS.optimize()` runs under the run's stage timer, holds nearly all of
the algorithm. Initialization (`_init_mesh_`) evaluates `x0`, and a second time
as a noise test when `uncertainty_handling` is `None`, then a Sobol initial
design of `2**ceil(log2(fun_eval_start))` points, twice as many when that
number equals `D`, cut to the evaluations that `max_fun_evals` leaves, the
noise test counted (`init_functions/`; MATLAB draws `fun_eval_start`
points, at most `MaxFunEvals - 1`, and does not count the noise test), and
trains the first GP (`init_and_train_gp`). The main loop then interleaves
two stages:

1. **SEARCH** (`_search_step_`, `search/`): at most one evaluation per pass
   of the loop. `ESSearchHedge` picks an evolution-strategy search (ES-wcm
   or ES-ell) by a Hedge portfolio; the chosen strategy scores its
   candidates by LCB (`acquisition_functions/acq_fcn_lcb.py`) and returns
   the best one.
2. **POLL** (`_poll_step_`, `poll/poll_mads_2n.py`): runs at the end of each
   round of up to `search_n_try` searches. It evaluates 2D directions in
   LCB order, the signed coordinate directions at default (`poll_mads_2n`
   draws an LTMADS basis under MATLAB BADS's bound, which is 1 at every
   default state), and stops early when the GP's probability of
   improvement drops below `tol_poi`. The mesh
   (`poll_mesh_multiplier ** mesh_size_integer`) grows after a successful
   poll and shrinks after a failed one. An iteration is a round of
   searches that a poll ends, and a run can end within one: the reported
   `iterations` and the displayed iteration number the current round from
   1, as MATLAB does, `max_iter` ends the run when that number reaches it,
   and `optim_state["iter"]`, the index into `iteration_history`, is one
   less.

The GP (`bads/gaussian_process_train.py` → `gpyreg.GP`) is local:
`local_gp_fitting` rebuilds its training set from the nearest neighbours of
the incumbent, re-optimizes hyperparameters only when `_is_gp_refit_time_`
says so, and `add_and_update_gp` updates the posterior after each new
evaluation. The run ends on `max_fun_evals`, `max_iter`, `mesh_size <
tol_mesh` or a stall over `tol_stall_iters`, and returns an
`OptimizeResult`, a dict with a fixed whitelist of keys.

## What spans files

- **Two coordinate spaces.** The algorithm runs in `u` space, where
  `VariableTransformer` maps the plausible box to `[-1, 1]^D` (with a log
  transform for a variable whose bounds are all positive and whose
  `pub/plb >= 10`, unless it is periodic); the target and `non_box_cons`
  see the original space.
  After `_init_optim_state_`, `self.lower_bounds` and its siblings hold the
  transformed bounds, and so do `optim_state["lb"]`, `["ub"]`, `["plb"]`
  and `["pub"]`, which `gaussian_process_train.py` reads; the original
  ones are in `optim_state["*_orig"]`.
- **Fixed variables** (four equal bounds) are left out of the run.
  `BADS.__init__` finds them (`_find_fixed_values`) before the options are
  evaluated, so that `D` (`self.D`, and the `D` of the options) counts the
  other variables. `self._fixed_values`, a row of one value per variable,
  NaN at the free ones (all NaN when none is fixed), goes to
  `VariableTransformer` as `fixed_values`: its `inverse_transf` returns
  points of all the variables and its `__call__` takes them. So every point
  that leaves the run through it (to the target, `non_box_cons` and
  `output_fcn`, into the log's `X_orig`, `iteration_history["x"]` and the
  result) has them all, as `self.x0` has, while `optim_state` (its
  `*_orig` bounds and its `periodic_vars` mask too), the `u` points and the
  GP cover the free ones. `options["periodic_vars"]` keeps the user's
  indices, which `_run_indices` maps to the run's variables where the run
  reads them; `_user_indices` maps back the indices that a message names.
  New code that hands the user a point takes it through the transformer.
- **The GP shapes the geometry.** `gp.temporary_data["poll_scale"]`,
  `["len_scale"]` and `["effective_radius"]` are set in
  `gaussian_process_train.py`. `poll_scale` shapes the ES-ell search and
  not the poll: `poll_mads_2n` returns directions divided by it, and
  `_poll_step_` multiplies them back, as MATLAB BADS does. `len_scale`
  scales the distances of `udist` (the neighbours of the local training
  set, the length of a search's step), and `effective_radius` the radius
  of the training set.
- **Periodic variables** (`periodic_vars`, stored as a sorted list of
  indices; `optim_state["periodic_vars"]` is their `(1, D)` mask) wrap
  around their hard bounds, in `u` space. Code that proposes points wraps
  them with `force_to_grid_periodic` where it puts them on the grid, and
  with `period_check` where it does not: the start (`start_on_mesh`), the
  initial design (`_init_mesh_`), the search set (`_search_step_`), each
  generation of the ES search (`ESSearch.__call__`) and the poll
  (`_poll_step_`); a new source of candidates needs the same, and
  `test_every_source_of_candidates_wraps_them` names them; the evaluations
  made before the run are logged as given. `udist` and `ucov` take a
  periodic difference the shorter way round, and the GP's kernel takes the
  periods from `_gp_periods` (gpyreg's `periods`), only in a run that has
  periodic variables: without them the kernel gets no `periods`. The gate
  of a change to this code is the `default` suite, which holds the six
  configurations that set `periodic_vars` (the `periodic` suite).
- **Options** are layered: `bads/option_configs/basic_bads_options.ini`,
  then the `options=` dict, then `advanced_bads_options.ini`, which skips
  any key the user set. `.ini` values are `eval`'d with `D` bound by `exec`
  into the module globals of `options.py`, and may read earlier options
  through `self.get(...)`; the dict is used verbatim, so a user's `"200*D"`
  stays a string, except that a user value of `None` leaves the option at
  its default (`Options.unset_user_options` records its name): an `is None`
  test on an option whose default is not `None` never fires. An option
  whose default in an `.ini` is `True` or `False` takes only a boolean
  (`Options.validate_boolean_options`, called in `BADS.__init__` with
  `uncertainty_handling` added and `plot` exempted), so a new such option
  that takes other values too goes into its `excluded_names` there. An
  unknown name raises `ValueError`, so a new option starts as a
  `# description` line followed by `name = <expr>` in the right `.ini`; the
  description is the last comment line above the option, so it fits on one
  line. Options stay mutable: a noisy run rewrites several of them
  (`tol_stall_iters`, `n_train_min`, `n_train_max`, `max_fun_evals` and
  others) at the start of `optimize()`, so a `BADS` object runs once: a
  second call of `optimize()` raises `RuntimeError`.
- **Some options do nothing.** Twelve are named after MATLAB BADS's
  options and read by no code (`poll_method`, `poll_acq_fcn`, `gp_samples`,
  among others), kept so that a user's setting is not an error, with
  descriptions that say so; an option that no code reads and that MATLAB
  BADS does not have is removed rather than kept. The GP's kernel is a
  hard-coded rational-quadratic ARD kernel (`optim_state["gp_cov_fun"] =
  1`; periodic along the periodic variables), a few options are read only
  by code that no run reaches (KD-B1-4 in `pybads/bads/README.md`), and
  `_init_optim_state_` reads `gpintmeanfun`, which no `.ini` defines, as
  `None`. Grep for an option's reads before relying on it.
- **Extension points are hard-coded.** `ESSearchHedge.__call__` chooses a
  search by string comparison, and `ESSearch.__call__` its acquisition
  function (`"acq_LCB"` or an error), LCB is called directly at the search
  and poll call sites, and the initial design is selected by
  `init_fun == "init_sobol"`. A new search method is an
  `ESSearch` subclass, an `elif` in the hedge, its name among those that
  the check of `search_method` in `_init_optim_state_` takes, and an entry
  in the `search_method` option; a search acquisition function other than
  the LCB, likewise, needs the check of `search_acq_fcn` there and, if it
  reads the optimization target, its name in
  `_SEARCH_ACQ_FCNS_READING_TARGET` (`bads.py`), which has the search
  compute the target; the poll computes its own.
- **Noise.** `optim_state["uncertainty_handling_level"]` is 0
  (deterministic), 1 (noise inferred) or 2 (`specify_target_noise`: the
  target returns exactly a `tuple` `(f, sd)`). In noisy runs the incumbent's
  `fval`/`fsd` are GP predictions. `_re_evaluate_history_` recomputes those
  of every iteration from a copy of the working GP, under the
  hyperparameters recorded at that iteration, and leaves the stored GPs as
  recorded; a past iterate whose rebuild fails gets NaN. The returned point
  is re-evaluated `noise_final_samples` times, reserved from
  `max_fun_evals`, with those samples kept out of the training set
  (`record_duplicate_data=False`).
- **`FunctionLogger`** calls the target with a 1-D `x` in the original space
  and raises `ValueError` on a NaN, infinite or non-scalar value; it
  preallocates its arrays, and `X_flag` marks the filled rows. The log can
  open with evaluations made before the run (`precomputed_evaluations`,
  added through `FunctionLogger.add` in `BADS.__init__`), which `func_count`
  leaves out. Code that takes the log's rows for the run's own evaluations
  leaves them out: `_init_mesh_` chooses the first incumbent among the rows
  that the start and the initial design returned (`_init_rows`) and counts
  `eff_starting_points` from `func_count`; `init_and_train_gp` fits the
  first GP on those rows; at level 2 `_init_optimization_` reads the first
  incumbent's `fsd` from its row, `_init_incumbent_row`; and
  `_get_gp_training_options` subtracts `optim_state["precomputed_n_evals"]`
  from `n_eff`. New code that reads `Xn`, `X_flag` or `n_evals` as the
  run's evaluations does the same.
  The evaluations made before the run join the GP at the loop's first
  rebuild, at the first poll, among the incumbent's neighbours.
  A repeated point at level 2 is merged into its row by precision weighting,
  which a run reaches only through those evaluations: `contraints_check`
  removes the candidates already evaluated, and the noise test and the final
  samples take `record_duplicate_data=False`, which leaves the log as it is.
- **Randomness goes through one `numpy.random.Generator`, `bads.rng`.**
  `BADS.__init__` creates it from `random_seed` (`pybads/rng.py: get_rng`)
  before its first draw, the random `x0`, and passes it as `rng` to
  everything that draws: the GP functions of `gaussian_process_train.py`,
  which pass it on to `gp.fit` and `SliceSampler` (a gpyreg call without
  `rng` draws from NumPy's global stream), `ESSearchHedge` and the
  `ESSearch` classes, `init_sobol` and `poll_mads_2n`; each resolves
  `rng=None` through `get_rng`. No draw of a run goes through the global
  stream, which `test_seeded_run_leaves_global_state_untouched` checks;
  `random_seed=None` derives the generator from four draws of it, so that
  `np.random.seed` before construction fixes a run, and nothing reseeds it.
  Nothing that is deep-copied holds the generator (the `OptimizeResult`,
  what `IterationHistory` records, the GP and its `temporary_data`), and
  neither does `optim_state`: a copy would be a second generator in the same
  state. The scrambling of the Sobol design is seeded by one draw of the
  generator in `init_sobol`, so that the seed decides the design, whatever
  `x0` is given (a random `x0` is drawn from the generator before it);
  MATLAB BADS derives its design from the start alone, with no random draw.
  The order of the runtime tips (`pybads/bads/_runtime_tips.py`) is not a
  draw of a run: a private `random.Random` of the session shuffles them,
  which no seed fixes and which touches neither `bads.rng` nor the state
  of NumPy or of the `random` module.
- **gpyreg internals.** `gaussian_process_train.py` calls the name-mangled
  private `gp._GP__gp_obj_fun`, so a change to gpyreg's private interface
  can break PyBADS.
- **A GP update can fail, and the GP handed on must stay consistent.**
  gpyreg (from 1.3.3) puts a GP whose `fit`, `update` or
  `set_hyperparameters` raises back as it was when the call started. So new
  data go through `gp.update(X_new=..., y_new=...)`. If `gp.X` or `gp.y` is
  assigned before an update that fails, the new data sit beside the old
  posteriors, and `predict` then raises, or silently predicts wrong values
  when the sizes are equal. `local_gp_fitting`, which replaces the training
  set, snapshots the GP and restores it when the rebuild fails: at once
  without a refit, and after a refit when its retry with the previous
  hyperparameters fails too. A GP that could not take a
  point carries `temporary_data["needs_rebuild"]`; a restored one also
  carries `["needs_refit"]`. The markers are set in
  `gaussian_process_train.py` and read by the search and the poll in
  `bads.py`, which rebuild at their next step, and refit (the poll only
  with `poll_training` on).
  `local_gp_fitting` removes both once it leaves a posterior on its new
  training set. `test_gp_update_failures.py` injects the failures.
- **The oracles' stored states.** The oracles of `pybads/testing/oracles/`
  rebuild states stored by an earlier commit, so a key that the code newly
  reads from `optim_state` or a GP's `temporary_data` gets a default in
  `STATE_DEFAULTS` of `pybads/testing/oracles/_state.py` in the same
  commit; `test_every_case_computes` fails without it, on the GP refit
  too, whose outputs no test compares.
- **`IterationHistory`** deep-copies what it records, including the GP,
  every iteration.
- **Stage times.** `optimize` charges each second of a run to the
  innermost open stage of a private `StageTimer`
  (`pybads/utils/timer/stage_timer.py`), and the target's evaluations to
  `target`, which together make `total_time`. `target` follows the
  function logger's `total_fun_eval_time`, the time that `overhead`
  compares with the run's, which leaves out the noise test at `x0`: that
  evaluation counts in `init`. `dev/scripts/profile_run.py` reads their
  plain snapshots, `optim_state["stage_times"]` and
  `iteration_history["timer"]`. The timer lives on the `BADS` object and
  goes to the GP functions as `timer=`, as `rng` does, never into
  `optim_state` or a GP's `temporary_data`, which are deep-copied.

## Numerical gates

A change that can move results is gated by the population comparison of
`dev/scripts/population.py` against the current reference of the platform
under `dev/experiments/` (its `README.md` holds the command, the
provenance, the null check, the positive control and what "no flag" can
detect at its number of seeds). There is one reference for Windows and one
for Linux, since pairing by seed holds only on one platform and set of
versions; `dev/README.md` names both. `compare` tests only the
configurations that both populations hold and lists the others on one
line, outside its verdict and its exit code: a configuration added to a
suite goes ungated until the reference holds it too. A gate is evidence
only if it reaches the changed code: the benchmark exercises the default
options, so a change behind a non-default option needs a configuration
that sets it. Every
evidence run selects gpyreg explicitly, with `PYTHONPATH` naming a clone at
the release tag (`dev/scripts/runs/LOCAL.md` lists them): the editable
install follows `../gpyreg`, which other work moves. PyBADS is selected in
two ways. `benchmark_targets.py`, and every script that imports it
(`population.py`, `gp_update_failures.py` and the others that run the
benchmark), puts the checkout that holds it first on `sys.path`: a commit
is measured by the scripts of a worktree at that commit, run from the main
checkout's root, and a worktree put on `PYTHONPATH` changes nothing (the
records of `population.py` name the package that ran,
`meta.pybads_source`). `fingerprint.py` and `tolerance_sweep.py` import
PyBADS from `PYTHONPATH`, or else from the editable install, the main
checkout. Evidence runs call the venv's Python by its path
(`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere). An
agent's shell does not activate the venv, so a bare `python` can be another
installation. If that installation has NumPy and SciPy, the scripts under
`dev/scripts/` run there without error, on its versions. A gpyreg release
is a change to PyBADS's numerics; its gate is the comparison run with that
release's clone, beside the test suite, with `gpyreg.__file__` printed. A
change that must move nothing shows the same hash of
`dev/scripts/fingerprint.py` before and after, on one machine, with the
same gpyreg and the same number of BLAS threads: one thread
(`OMP_NUM_THREADS=1` and its kin) and the default can give different
hashes of the same commit, and a recorded hash names its setting. Two
SciPy versions can give different hashes as well: SciPy 1.18, which needs
Python 3.12, takes its Cholesky factor from another LAPACK routine than
1.17. A recorded hash names its SciPy, and a change to gpyreg that must
move nothing is gated under the newest SciPy as well as under the venv's.
On one machine, a change that must move nothing also shows
`dev/scripts/replay.py check` identical against the parent commit (the
first evaluation and GP computation at which two commits' runs part), and
`dev/scripts/make_oracle_fixtures.py --check --exact --against` a
`--dump` of the parent commit identical (the oracles of
`pybads/testing/oracles/`, PyBADS's components on stored states);
`dev/README.md` gives the procedures. A change to gpyreg that must move
nothing shows the fingerprint and the replay identical with its clone and
with its parent's, and `dev/scripts/gpyreg_bitwise.py compare` identical
between dumps of the two, which reach gpyreg's paths that PyBADS's runs do
not; the oracles' `--against` refuses a dump made under another gpyreg.
The four, the fingerprint, the replay, the oracles' `--against` and
`gpyreg_bitwise.py`, need a machine that repeats a computation bit for
bit, which macOS arm64 is not: there Accelerate's
results depend on the alignment of the arrays, so that two runs of one
seed need not match (`dev/results/2026-09-28-macos-arm64-repeatability.md`).
On every platform, the tests pin the initial design, which involves no BLAS
work (`test_initial_design_pin.py`, whose fixture a change that moves the
design on purpose regenerates in the same commit), and the oracles' stored
references, under tolerances measured across BLAS settings. Never loosen an
oracle's tolerance or regenerate the fixtures to make a change pass: a
change that moves an oracle on purpose replaces that oracle's references
alone, in the same commit, with `--rebaseline ORACLE --reason TEXT`.
`--write --reason TEXT`, from a clean checkout, makes a new baseline of
every oracle for a change of the recipes or of what the snapshots hold,
and when `--rebaseline` refuses a decision at its bound, since it alone
re-chooses the size of the reduced training sets; the fixtures record its
reason.

## Tests and their traps

- `pybads/testing/bads/test_bads_optimization.py` runs whole optimizations
  (a few hundred evaluations at most, one of them 60-D) and dominates the
  runtime of the suite.
- Every test whose outcome depends on random draws is seeded, including
  the noise of a noisy target, so a failing test fails again on each rerun,
  and CI runs each test once. A test that fails and then passes when rerun
  depends on something unseeded, in the test or in the package, which is a
  bug to fix, except on macOS arm64: there Accelerate's results depend on
  the alignment of the arrays, so two runs of one seed need not match bit
  for bit, and a test that compares two runs exactly can fail without any
  unseeded draw (`dev/results/2026-09-28-macos-arm64-repeatability.md`).
  There a test compares two runs on what the seed alone decides, as
  `_same` in `test_bads_seed.py` does, or checks its property directly.
  The tolerances of
  `test_bads_optimization.py` hold over a sweep of seeds, not only at the
  seed each test runs at: when a change that moves results fails one,
  measure the errors over the seeds again with
  `dev/scripts/tolerance_sweep.py` before reseeding the test or loosening
  the tolerance.
- `pybads/testing/bads/scripts/` holds manual scripts that pytest does not
  collect.
- The runtime tips are the state of the Python session
  (`pybads/bads/_runtime_tips.py`): a test that reads a run's messages at
  the iteration display sees a tip or not depending on the tests run
  before it. `show_tips=False`, or `_reset_runtime_tip_state` in a
  fixture, isolates it.

## Conventions

- **Commits** follow conventional commits. A `Co-Authored-By:` line is fine;
  a `Claude-Session:` trailer is not, even where the session's own
  attribution instructions ask for one. Changes reach `main` through pull
  requests, which run the full test matrix, and are squash-merged, titled
  `<type>: <summary> (#NN)`. Work collects on the long-lived branch
  `dev-next`; after its pull request is squash-merged, `dev-next` is reset
  onto `main`, keeping only the commits made after the merged head, and
  force-pushed, or the next pull request lists the merged commits again.
- **Changelog.** A change to results, to the interface, or to what a
  script sees or has to handle gets an entry in `CHANGELOG.md` under
  `Unreleased`, in the commit that makes it, and so does a speed-up that a
  user notices, with its measured size and a link to its record under
  `dev/`; a message's wording, a minor speed-up or a change to the tests
  gets none. An entry is one or
  two sentences, written for users, on what a user notices, relative to
  the last release: a fix to a change that no release has shipped edits
  that change's entry, and the reasons and the comparison with MATLAB BADS
  belong in the catalogue (`pybads/bads/README.md`) and the records under
  `dev/`, to which an entry can point. The fixes are grouped under a few
  themes (`####` headings under Fixed), and changes of one kind, such as
  new checks of the options' values, share one entry. A change that can
  stop a script written for the last release, or change its results, also
  has a line in the "Upgrading from" list that opens the section: the
  change's only mention when that line says all a user needs, and
  otherwise a pointer to its entry, kept in step with it; changes of one
  kind share a line.
- **Modules.** No general `util`/`misc` modules: a general-purpose function
  goes into the module that fits it or into a module of its own. A new
  subpackage is listed in the `packages` of `pyproject.toml`.
- **Network access.** The package opens a network connection only in
  `pybads.check_for_updates()` (`pybads/_update_check.py`), which the user
  calls, and which imports its networking modules inside the function. The
  old-release reminder at the start of a run
  (`pybads/bads/_release_reminder.py`) takes the release date from
  `pybads/_release.py` and its past showings from its state file,
  `update_reminder.json` in the user's cache directory, and writes no file
  but that one.
- **Links to the lab.** In what ships or is published, a link that names
  Luigi Acerbi goes to his personal page, https://lacerbi.github.io/, and
  one that names the lab or its members to the group's page,
  https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence.
  A paragraph that sends the reader to another of the lab's methods
  (PyVBMC, PyIBS, MATLAB BADS) links the lab's page of them,
  https://acerbilab.org/model-fitting/, with the text "tools for fitting
  models to data", and `README.md` and `index.rst` link it under their
  title.
- **MATLAB logicals.** Where MATLAB has `~`, `&` or `|` on logicals, use
  `not`, `and`, `or`: on a Python `bool`, `~` gives `-1` or `-2` (always
  truthy, and deprecated since Python 3.12), and `&` binds tighter than a
  comparison. The condition for adding the search point to the GP in
  `_search_step_` was once such a slip: `size > 0 & count < n_try`, which
  is always true. On NumPy arrays the converse holds: a test per variable
  stays elementwise, with `&`, `|` or `!=` inside a single `np.any`;
  written as `np.any(a) and np.any(b)`, it tests all the variables at once,
  a slip that once made a check of `_bounds_check_` refuse every problem
  that mixed bounded and unbounded variables.
