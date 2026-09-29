---
name: pybads
description: Find and apply PyBADS documentation when optimizing a black-box, possibly noisy objective under bound or non-box constraints, troubleshooting a run, or interpreting its result.
---

# PyBADS

Use the documentation below to answer the user's question or work on their
optimization. Read the sections relevant to the task; follow links for
details as needed.

Check the installed PyBADS version (`importlib.metadata.version("pybads")`)
before using version-specific features: the
[changelog](https://github.com/acerbilab/pybads/blob/main/CHANGELOG.md) says
what changed in each release, and its "Upgrading from" lists say what to
check in a script written for the release before. Prefer documentation from
the user's checkout when available. The source links below point to the
[`main` branch](https://github.com/acerbilab/pybads/tree/main), from which
the published [documentation](https://acerbilab.github.io/pybads/) is built.
For another version, use the corresponding Git tag and check API signatures
and docstrings in that version.

## What to read

Paths are relative to the PyBADS repository root. The links also work when
this skill folder has been copied elsewhere.

| Task | Read |
| --- | --- |
| Decide whether PyBADS fits the problem; install it | [README.md](https://github.com/acerbilab/pybads/blob/main/README.md): “When should I use PyBADS?” and “Installation”. |
| Set up an optimization: the target, the starting point, the hard and plausible bounds | [docsrc/source/quickstart.rst](https://github.com/acerbilab/pybads/blob/main/docsrc/source/quickstart.rst), [Example 1](https://github.com/acerbilab/pybads/blob/main/examples/pybads_example_1_basic_usage.ipynb), the FAQ's “Input arguments” sections, and the docstring of `BADS` in [pybads/bads/bads.py](https://github.com/acerbilab/pybads/blob/main/pybads/bads/bads.py), which gives what each argument takes and what `BADS` refuses. |
| Add constraints beyond the bounds | [Example 2](https://github.com/acerbilab/pybads/blob/main/examples/pybads_example_2_nonbox_constraints.ipynb), the FAQ's “How do I prevent PyBADS from evaluating certain inputs or regions of input space?”, and `non_box_cons` in the docstring of `BADS`, which says how to reparametrize a feasible region thinner than the mesh. |
| Optimize a noisy target | The FAQ's “Noisy objective function” section; [Example 3](https://github.com/acerbilab/pybads/blob/main/examples/pybads_example_3_noisy_objective.ipynb) for noise that BADS estimates, [Example 4](https://github.com/acerbilab/pybads/blob/main/examples/pybads_example_4_user_provided_noise.ipynb) for a target that returns its own noise estimate (`specify_target_noise`). |
| Run several starts; reproduce a run | The FAQ's “How do I run PyBADS from several starting points?” and “How do I make a run reproducible?”; [Example 5](https://github.com/acerbilab/pybads/blob/main/examples/pybads_example_5_extended_usage.ipynb) for a multi-start run and the result's fields; `rng` in the docstring of `BADS` for `random_seed`. |
| Set options | The [options page](https://acerbilab.github.io/pybads/api/options/bads_options.html), which shows [basic_bads_options.ini](https://github.com/acerbilab/pybads/blob/main/pybads/bads/option_configs/basic_bads_options.ini) and [advanced_bads_options.ini](https://github.com/acerbilab/pybads/blob/main/pybads/bads/option_configs/advanced_bads_options.ini): the comment above each option describes it. Some advanced options are not read by PyBADS, and not all of them say so: before relying on one, search `pybads/` for where it is read. |
| Interpret a result; troubleshoot a run | The FAQ's “Output arguments”, “Display” and “Troubleshooting” sections; the docstring of `OptimizeResult` in [pybads/bads/optimize_result.py](https://github.com/acerbilab/pybads/blob/main/pybads/bads/optimize_result.py) (`fval`, `fsd`, `success`, `status`, `message`, `overhead`); the README's “How does it work?” and “Troubleshooting and contact”. |
| Port a script or an analysis from MATLAB BADS | The FAQ's “I used BADS in MATLAB. What is different in PyBADS?”. |
| Compare with an earlier version of PyBADS | The [changelog](https://github.com/acerbilab/pybads/blob/main/CHANGELOG.md): its “Upgrading from” lists, then the entries they point to. |
| Look up exact arguments, options or result fields | The [API reference](https://acerbilab.github.io/pybads/documentation.html), with sources under `docsrc/source/api/` and implementation docstrings under `pybads/`. |

The FAQ is [docsrc/source/faq.md](https://github.com/acerbilab/pybads/blob/main/docsrc/source/faq.md),
published at <https://acerbilab.github.io/pybads/faq.html>.

Before evaluating the target or starting more runs, establish the user's
evaluation budget (`max_fun_evals`, `500 * D` evaluations by default) and
account for any diagnostic calls within it. PyBADS finds a local minimum:
plan several runs from different starting points when the target may have
more than one (Example 5). When working on an existing analysis, inspect
its setup and saved results before deciding whether another run is needed.
Link the relevant documentation in your explanation so the user can check
the reasoning.
