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
- `dev/results/2026-09-23-codebase-survey.md` records the observed failures
  and the candidate defects of the port, not yet verified against MATLAB
  BADS. Check it before treating an oddity in the numerical code as
  intended, and record a fix or a verdict in its entry.
- `pybads/bads/README.md` lists the open porting work.
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

The test job is defined once, in `.github/workflows/test-matrix.yml`, and
installs gpyreg at the commit pinned as `GPYREG_PIN` there: the tagged
commit of the release that `pyproject.toml` names as the minimum (CI reads
gpyreg's version from its tags, and an untagged commit reads lower, so pip
would install gpyreg from PyPI over the pinned checkout). A change that
needs a newer gpyreg moves both. `merge-tests.yml` runs the full matrix
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
`Unreleased`, and the GitHub release takes that section as its notes. The
example scripts are worth a headless run first, since nothing else runs
them. `release.yml` builds the package with `build.yml` and uploads it to
PyPI by trusted publishing, through the `pypi` environment, which admits
only `v*` tags. No token is stored. conda-forge follows by itself: after
the upload, the version bot of `conda-forge/pybads-feedstock` opens an
update PR, takes its dependencies from the PyPI metadata and merges it
once its CI passes (the `bot` settings in the feedstock's
`conda-forge.yml`; `conda-forge/gpyreg-feedstock` has the same). That PR
fails when a dependency is missing from conda-forge, for instance a new
gpyreg minimum that the gpyreg feedstock has not published yet; then a
feedstock maintainer takes over.

Formatting is enforced by the pre-commit hooks alone (black at line length
79 on every Python file and the notebooks' code cells, isort with the black
profile, pycln); no CI job checks it, and the whole tree passes them. The
commit that first formatted the tree is listed in `.git-blame-ignore-revs`;
`git config blame.ignoreRevsFile .git-blame-ignore-revs` hides it from
`git blame`.

`pyproject.toml` is authoritative; `setup.py` is a shim. It names only
`pybads` and `pybads.examples` as packages: the subpackages and the `.ini`
option files reach the wheel through `include-package-data` and
setuptools_scm's file finder, which takes only files tracked by git, so a
new module or data file ships only once committed. The tests ship in the
wheel: the conda-forge recipe runs them from the installed package
(`pytest --pyargs pybads`). What they need is the `test` extra, which CI
installs and `dev` includes; no package module imports pytest.

Docstrings are numpydoc, with `OptimizeResult`
(`pybads/bads/optimize_result.py`) as the reference style. Nothing generates
the API pages: a new public class or function needs a hand-written `.rst`
under `docsrc/source/api/` and an entry in the toctree that owns it
(`documentation.rst` for a headline page, `api/classes/classes.rst` or
`api/functions/functions.rst` otherwise). The options page includes the two
`.ini` files verbatim, so the comment above each option is its user
documentation. Build with `make github` in `docsrc/` (`.\make.bat github`
from cmd on Windows), which copies the result into `docs/`.

The notebooks in `examples/` ship in the wheel as `pybads.examples`
(`python -m pybads` opens them) and are rendered without execution by the
docs build; nothing runs them, so a change that breaks one goes unnoticed.
`examples/scripts/*.py` are generated from the notebooks by
`examples/scripts/Makefile` (GNU Make, with nbconvert, and black and isort
at the pre-commit hook versions, in the environment `python` names);
regenerate them with `make -B -C examples/scripts`, do not edit them.

## Architecture

`BADS.optimize()` in `pybads/bads/bads.py` holds nearly all of the
algorithm. Initialization (`_init_mesh_`) evaluates `x0`, and a second time
as a noise test when `uncertainty_handling` is `None`, then a Sobol initial
design of `2**ceil(log2(fun_eval_start))` points, twice as many when that
number equals `D` (`init_functions/`; MATLAB draws `fun_eval_start`
points), and trains the first GP (`init_and_train_gp`). The main loop then interleaves
two stages:

1. **SEARCH** (`_search_step_`, `search/`): at most one evaluation per pass
   of the loop. `ESSearchHedge` picks an evolution-strategy search (ES-wcm
   or ES-ell) by a Hedge portfolio; the chosen strategy scores its
   candidates by LCB (`acquisition_functions/acq_fcn_lcb.py`) and returns
   the best one.
2. **POLL** (`_poll_step_`, `poll/poll_mads_2n.py`): runs at the end of each
   round of up to `search_n_try` searches. It evaluates the 2D LTMADS
   directions in LCB order and stops early when the GP's probability of
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
  `pub/plb >= 10`); the target and `non_box_cons` see the original space.
  After `_init_optim_state_`, `self.lower_bounds` and its siblings hold the
  transformed bounds, and so do `optim_state["lb"]`, `["ub"]`, `["plb"]`
  and `["pub"]`, which `gaussian_process_train.py` reads; the original
  ones are in `optim_state["*_orig"]`.
- **The GP shapes the geometry.** `gp.temporary_data["poll_scale"]`,
  `["len_scale"]` and `["effective_radius"]`, set in
  `gaussian_process_train.py`, drive the poll basis and the ES-ell search.
  `poll_mads_2n` returns directions divided by `poll_scale`, and
  `_poll_step_` multiplies them back.
- **Options** are layered: `bads/option_configs/basic_bads_options.ini`,
  then the `options=` dict, then `advanced_bads_options.ini`, which skips
  any key the user set. `.ini` values are `eval`'d with `D` bound by `exec`
  into the module globals of `options.py`, and may read earlier options
  through `self.get(...)`; the dict is used verbatim, so a user's `"200*D"`
  stays a string. An unknown name raises `ValueError`, so a new option
  starts as a `# description` line followed by `name = <expr>` in the right
  `.ini`; the description is the last comment line above the option, so it
  fits on one line. Options stay mutable: a noisy run rewrites several of them
  (`tol_stall_iters`, `n_train_min`, `n_train_max`, `max_fun_evals` and
  others) at the start of `optimize()`, so a `BADS` object runs once.
- **Many options do nothing.** Some are PyVBMC or MATLAB leftovers that no
  code reads (`warp_*`, `variational_sampler`, `poll_method`,
  `poll_acq_fcn`, among others); `gp_cov_fun` is overridden by a hard-coded
  rational-quadratic ARD kernel (`optim_state["gp_cov_fun"] = 1`); and
  `_init_optim_state_` reads `gpintmeanfun`, which no `.ini` defines, as
  `None`. Grep for an option's reads before relying on it.
- **Extension points are hard-coded.** `ESSearchHedge.__call__` chooses a
  search by string comparison (`ESSearchCMA` is unreachable), LCB is called
  directly at the search and poll call sites, and the initial design is
  selected by `init_fun == "init_sobol"`. A new search method is an
  `ESSearch` subclass, an `elif` in the hedge, and an entry in the
  `search_method` option.
- **Noise.** `optim_state["uncertainty_handling_level"]` is 0
  (deterministic), 1 (noise inferred) or 2 (`specify_target_noise`: the
  target returns exactly a `tuple` `(f, sd)`). In noisy runs the incumbent's
  `fval`/`fsd` are GP predictions, `_re_evaluate_history_` recomputes them
  from every stored GP, and the returned point is re-evaluated
  `noise_final_samples` times, reserved from `max_fun_evals`, with those
  samples kept out of the training set (`record_duplicate_data=False`).
- **`FunctionLogger`** calls the target with a 1-D `x` in the original space
  and raises `ValueError` on a NaN, infinite or non-scalar value; it
  preallocates its arrays, and `X_flag` marks the filled rows. A repeated
  point at level 2 is merged into its row by precision weighting.
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
  state. The Sobol design takes its seed from `u0`, not from the generator:
  from the integer part of each of its first 11 coordinates, so every start
  point inside the plausible box gives the same design for a given `D`,
  whatever the seed (a candidate defect, in the survey).
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
- **`IterationHistory`** deep-copies what it records, including the GP,
  every iteration.

## Numerical gates

A change that can move results is gated by the population comparison of
`dev/scripts/population.py` against the current reference of the platform
under `dev/experiments/` (its `README.md` holds the command, the
provenance, the null check, the positive control and what "no flag" can
detect at its number of seeds). There is one reference for Windows and one
for Linux, since pairing by seed holds only on one platform and set of
versions; `dev/README.md` names both. A gate is evidence only if it reaches
the changed code: the benchmark exercises the default options, so a change
behind a non-default option needs a configuration that sets it. Every
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
`dev/scripts/fingerprint.py` before and after, on one machine and with the
same gpyreg.

## Tests and their traps

- `pybads/testing/bads/test_bads_optimization.py` runs whole optimizations
  (a few hundred evaluations at most, one of them 60-D) and dominates the
  runtime of the suite.
- Every test whose outcome depends on random draws is seeded, including
  the noise of a noisy target, so a failing test fails again on each rerun,
  and CI runs each test once. A test that fails and then passes when rerun
  depends on something unseeded, in the test or in the package, which is a
  bug to fix. The tolerances of
  `test_bads_optimization.py` hold over a sweep of seeds, not only at the
  seed each test runs at: when a change that moves results fails one,
  measure the errors over the seeds again with
  `dev/scripts/tolerance_sweep.py` before reseeding the test or loosening
  the tolerance.
- `pybads/testing/bads/scripts/` holds manual scripts that pytest does not
  collect.

## Conventions

- **Commits** follow conventional commits. A `Co-Authored-By:` line is fine;
  a `Claude-Session:` trailer is not, even where the session's own
  attribution instructions ask for one. Changes reach `main` through pull
  requests, which run the full test matrix, and are squash-merged, titled
  `<type>: <summary> (#NN)`. Work collects on the long-lived branch
  `dev-next`; after its pull request is squash-merged, `dev-next` is reset
  onto `main`, keeping only the commits made after the merged head, and
  force-pushed, or the next pull request lists the merged commits again.
- **Changelog.** A change that a user can notice is listed in `CHANGELOG.md`
  under `Unreleased`, in the commit that makes it, in a sentence written for
  users and relative to the last release (a fix to a feature that no
  release has shipped belongs to that feature's entry). A change that can
  stop a script written for the last release, or change what it returns,
  also has one line in the "Upgrading from" list that opens the section,
  kept in step with its entry.
- **Modules.** No general `util`/`misc` modules: a general-purpose function
  goes into the module that fits it or into a module of its own.
- **MATLAB logicals.** Where MATLAB has `~`, `&` or `|` on logicals, use
  `not`, `and`, `or`: on a Python `bool`, `~` gives `-1` or `-2` (always
  truthy, and deprecated since Python 3.12), and `&` binds tighter than a
  comparison. The condition for adding the search point to the GP in
  `_search_step_` was once such a slip: `size > 0 & count < n_try`, which
  is always true. A check of `_bounds_check_` that refused a variable
  bounded on one side only, since removed, was once one the other way
  round, a test per variable written as `any(...) and any(...)` across all
  of them: on arrays, an elementwise test stays elementwise (`&`, `|`,
  `!=`) inside one `np.any`.
