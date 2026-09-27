# PyBADS: Bayesian Adaptive Direct Search in Python
![Version](https://img.shields.io/badge/dynamic/json?label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpybads%2Fjson)
[![Conda](https://img.shields.io/conda/v/conda-forge/pybads)](https://anaconda.org/conda-forge/pybads)
[![PyPI](https://img.shields.io/pypi/v/pybads)](https://pypi.org/project/pybads/)
<br />
[![Discussion](https://img.shields.io/badge/-discussion-blue?logo=github)](https://github.com/orgs/acerbilab/discussions)
[![tests](https://img.shields.io/github/actions/workflow/status/acerbilab/pybads/tests.yml?branch=main&label=tests)](https://github.com/acerbilab/pybads/actions/workflows/tests.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/acerbilab/pybads/docs.yml?branch=main&label=docs)](https://github.com/acerbilab/pybads/actions/workflows/docs.yml)
[![build](https://img.shields.io/github/actions/workflow/status/acerbilab/pybads/build.yml?branch=main&label=build)](https://github.com/acerbilab/pybads/actions/workflows/build.yml)

## What is it?

PyBADS is a Python implementation of the Bayesian Adaptive Direct Search (BADS) algorithm for solving difficult and mildly expensive optimization problems, originally implemented [in MATLAB](https://github.com/acerbilab/bads). BADS has been intensively tested for fitting a variety of computational models, and is currently being used in many computational labs around the world (see [Google Scholar](https://scholar.google.co.uk/scholar?cites=7209174494000095753&as_sdt=2005&sciodt=0,5&hl=en) for many example applications).

In a benchmark with real model-fitting problems from computational and cognitive neuroscience, BADS performed on par or better than many other common and state-of-the-art optimizers, as shown in the original *NeurIPS* paper [[2](#references-and-citation)].

PyBADS requires no specific tuning and runs off-the-shelf like other Python optimizers (e.g., `scipy.optimize.minimize`).

*Note*: If you are interested in estimating posterior distributions (i.e., uncertainty and error bars) over model parameters, and not just point estimates, you might also want to check out Variational Bayesian Monte Carlo for Python ([PyVBMC](https://github.com/acerbilab/pyvbmc)), a package for Bayesian posterior and model inference which can be used in synergy with PyBADS.

## What's new in PyBADS 1.1

- **Reproducible runs.** Every random draw of a run comes from one NumPy random generator, created from the `random_seed` option, so a seeded run gives the same result every time on the same machine and leaves NumPy's global random state untouched (see [Quick start](#quick-start)).
- **More precise results with gpyreg 1.3.3.** PyBADS requires gpyreg 1.3.3, whose Gaussian process predictions are more accurate when the noise is very small; on several benchmark problems with deterministic targets, runs end closer to the minimum.
- **A fix for user-specified noise.** A run with `specify_target_noise=True` no longer stops with a `ValueError` when a point is evaluated a second time.
- **Requirements.** PyBADS needs Python 3.10 or newer; the test dependencies are an optional extra, `pybads[test]`.

The [changelog](CHANGELOG.md) lists what changed since PyBADS 1.0.6, including why results differ from earlier versions and what to check in an existing script.

## Documentation

The full documentation is available at: https://acerbilab.github.io/pybads/

## When should I use PyBADS?

BADS is effective when:

- the objective function landscape is rough (nonsmooth), typically due to numerical approximations or noise;
- the objective function is at least moderately expensive to compute (e.g., more than 0.1 second per function evaluation);
- the gradient is unavailable (black-box function);
- the number of input parameters is up to about `D = 20` or so.

## Installation

PyBADS is available via `pip` and `conda-forge`.

1. Install with:
    ```console
    python -m pip install pybads
    ```
    or:
    ```console
    conda install --channel=conda-forge pybads
    ```
    PyBADS requires Python version 3.10 or newer.

2. (Optional): Install [Jupyter Notebook](https://jupyter.org/install) to run the examples. You can skip this step if your environment already has Jupyter Notebook, but be aware that if the wrong `jupyter` executable is found on your path then import errors may arise.
   ```console
   python -m pip install notebook
   ```
   or, with Conda:
   ```console
   conda install --channel=conda-forge jupyter
   ```
   The example notebooks can then be accessed by running
   ```console
   python -m pybads
   ```

If you wish to install directly from latest source code, please see the [instructions for developers and contributors](https://acerbilab.github.io/pybads/development.html#installation-instructions-for-developers).

## Quick start

The typical workflow of PyBADS follows four steps:

1. Define the target (or objective) function;
2. Setup the problem configuration (optimization bounds, starting point, possible constraint violation function);
3. Initialize and run the optimization;
4. Examine and visualize the results.

Running the optimizer in step 3 only involves a couple of lines of code:

```python
from pybads import BADS
# ...
bads = BADS(target, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds)
optimize_result = bads.optimize()
```

with input arguments:

- ``target``: the target function, it takes as input a vector and returns its function evaluation;
- ``x0``: the starting point of the optimization problem. If it is not given, the starting point is drawn uniformly at random within the plausible bounds;
- ``lower_bounds`` and ``upper_bounds``: hard lower and upper bounds for the optimization region (can be ``-inf`` and ``inf``, or bounded);
- ``plausible_lower_bounds`` and ``plausible_upper_bounds``: *plausible* lower and upper bounds, that represent our best guess at bounding the region where the solution might lie;
- ``non_box_cons`` (optional): a callable function that denotes non-box constraint violations.

The outputs are:

- ``optimize_result``: an ``OptimizeResult`` object which presents relevant information about the solution and the optimization problem. In particular:
  - ``"x"``: the minimum point found by the optimizer;
  - ``"fval"``: the value of the function at the given solution.

For a full list and description of the entries of the ``optimize_result`` object, see the [OptimizeResult](https://acerbilab.github.io/pybads/api/classes/optimize_result.html) class documentation.

For a reproducible run, pass an integer seed when creating the `BADS` object, e.g. `BADS(..., options={"random_seed": 42})`; the seed is read when the object is created. For independent runs, leave `random_seed` unset or use different seeds. The seed controls only PyBADS's own random draws: if your target is noisy (e.g., simulation-based), seed its random number generator separately.

## Next steps

Once installed, example Jupyter notebooks can be found in the `pybads/examples` directory. They can also be [viewed statically](https://acerbilab.github.io/pybads/index.html#examples) on the [main documentation pages](https://acerbilab.github.io/pybads/index.html). These examples represent a full tutorial that will walk you through the basic usage of PyBADS as well as some of its more advanced features, such as [noisy targets](examples/pybads_example_3_noisy_objective.ipynb).

For practical recommendations, such as how to set `lower_bounds`, `upper_bounds` and the plausible bounds, check out the FAQ on the [BADS wiki](https://github.com/acerbilab/bads/wiki). Even though the FAQ refers to the MATLAB version of BADS, most of the concepts apply equally to PyBADS.

## How does it work?

PyBADS/BADS follows a [mesh adaptive direct search](http://epubs.siam.org/doi/abs/10.1137/040603371) (MADS) procedure for function minimization that alternates **poll** steps and **search** steps (see **Fig 1**).

- In the **poll** stage, points are evaluated on a mesh by taking steps in one direction at a time, until an improvement is found or all directions have been tried. The step size is doubled in case of success, halved otherwise.
- In the **search** stage, a [Gaussian process](https://distill.pub/2019/visual-exploration-gaussian-processes/) (GP) is fit to a (local) subset of the points evaluated so far. Then, we iteratively choose points to evaluate according to a *lower confidence bound* strategy that trades off between exploration of uncertain regions (high GP uncertainty) and exploitation of promising solutions (low GP mean).

**Fig 1: BADS procedure** ![BADS procedure](https://raw.githubusercontent.com/acerbilab/pybads/main/docsrc/source/_static/bads-cartoon.png)

See [here](https://github.com/lacerbi/optimviz) for a visualization of several optimizers at work, including BADS.

See the original BADS paper for more details ([Acerbi and Ma, 2017](#references-and-citation)).

## Troubleshooting and contact

PyBADS is under active development. The original BADS algorithm has been extensively tested in several benchmarks and published papers, and some of the benchmarks have been replicated with PyBADS. However, as with any optimization method, you should double-check your results.

If you have trouble doing something with PyBADS, spot bugs or strange behavior, or you simply have some questions, please feel free to:
- Post in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions) with questions or comments about PyBADS, your problems & applications;
- [Open an issue](https://github.com/acerbilab/pybads/issues/new) on GitHub;
- Contact the project lead at <luigi.acerbi@helsinki.fi>, putting 'PyBADS' in the subject of the email.

## References and citation

1. Singh, G. S. & Acerbi, L. (2024). PyBADS: Fast and robust black-box optimization in Python. *Journal of Open Source Software*, 9(94), 5694, [https://doi.org/10.21105/joss.05694](https://doi.org/10.21105/joss.05694)

2. Acerbi, L. & Ma, W. J. (2017). Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. In *Advances in Neural Information Processing Systems 30*: 1834-1844. ([paper + supplement on arXiv](https://arxiv.org/abs/1705.04405), [NeurIPS Proceedings](https://papers.nips.cc/paper/2017/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html))


Please cite both references if you use PyBADS in your work (the 2017 paper introduced the framework, and the latest one is its Python library). You can cite PyBADS in your work with something along the lines of

> We optimized the log likelihoods of our models using Bayesian adaptive direct search (BADS; Acerbi and Ma, 2017), via the PyBADS software (Singh and Acerbi, 2024). PyBADS alternates between a series of fast, local Bayesian optimization steps and a systematic, slower exploration of a mesh grid.

Besides formal citations, you can demonstrate your appreciation for PyBADS in the following ways:

- *Star :star:* the PyBADS repository on GitHub;
- [Subscribe](http://eepurl.com/idcvc9) to the lab's newsletter for news and updates (new features, bug fixes, new releases, etc.);
- Follow Luigi Acerbi on [X](https://x.com/AcerbiLuigi) or [Bluesky](https://bsky.app/profile/lacerbi.bsky.social) for updates about BADS/PyBADS and other projects;
- Tell us about your model-fitting problem and your experience with PyBADS (positive or negative) in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions).

*Note*: If you are interested in estimating posterior distributions (i.e., uncertainty and error bars) over model parameters, and not just point estimates, you might also want to check out Variational Bayesian Monte Carlo for Python ([PyVBMC](https://github.com/acerbilab/pyvbmc)), a package for Bayesian posterior and model inference which can be used in synergy with PyBADS.

### BibTeX

```BibTeX

@article{singh2024pybads,
  title={{PyBADS}: {F}ast and robust black-box optimization in {P}ython},
  author={Gurjeet Sangra Singh and Luigi Acerbi},
  publisher = {The Open Journal},
  journal = {Journal of Open Source Software},
  year = {2024},
  volume = {9},
  number = {94},
  pages = {5694},
  url = {https://doi.org/10.21105/joss.05694},
  doi = {10.21105/joss.05694},
}

@article{acerbi2017practical,
    title={Practical {B}ayesian Optimization for Model Fitting with {B}ayesian Adaptive Direct Search},
    author={Acerbi, Luigi and Ma, Wei Ji},
    journal={Advances in Neural Information Processing Systems},
    volume={30},
    pages={1834--1844},
    year={2017}
  }
```

### License

PyBADS is released under the terms of the [BSD 3-Clause License](LICENSE).

### Acknowledgments

PyBADS is developed by [members](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people) (past and current) of the [Machine and Human Intelligence Group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/) at the University of Helsinki and [ELLIS Institute Finland](https://www.ellisinstitute.fi/). Starting from version 1.1, development of PyBADS has been assisted by coding agents, including Anthropic's [Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5).
Work on the PyBADS package is supported by the Research Council of Finland (grants 356498 and 358980 to Luigi Acerbi) and its Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/).
