
# PyBADS: Bayesian Adaptive Direct Search in Python

## What is it?
`PyBads` is a Python implementation of the Bayesian Adaptive Direct Search (BADS) algorithm for solving difficult and moderate expensive optimization problems, originally implemented [in MATLAB](https://github.com/lacerbi/bads). BADS has been intensively tested for fitting a variety of computational models, and is currently being used in many computational labs around the world (see `Google Scholar <https://scholar.google.co.uk/scholar?cites=7209174494000095753&as_sdt=2005&sciodt=0,5&hl=en>`__ for many example applications).

In our benchmark with real model-fitting problems, BADS performed on par or better than many other common and state-of-the-art optimizers, such as `fminsearch`, `fmincon`, and `cmaes` as shown in the original paper presented at NeurIPS in 2017 [[1](#references-and-citation)].

BADS requires no specific tuning and runs off-the-shelf like other built-in MATLAB optimizers such as `fminsearch`.

## Documentation
The full documentation is available at: https://acerbilab.github.io/pybads/

## When should I use PyBADS?

BADS is effective when:

- the objective function landscape is rough (nonsmooth), typically due to numerical approximations or noise;
- the objective function is at least moderately expensive to compute (e.g., more than 0.1 second per function evaluation);
- the gradient is unavailable (black-box function);
- the number of input parameters is up to about `D = 20` or so.

*Note*: If you are interested in estimating posterior distributions (i.e., uncertainty and error bars) over model parameters, and not just point estimates, you might also want to check out Variational Bayesian Monte Carlo for Python ([PyVBMC](https://github.com/acerbilab/pyvbmc)), a package for Bayesian posterior and model inference which can be used in synergy with PyBADS.

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
    PyBADS requires Python version 3.9 or newer.
2. (Optional): Install Jupyter to view the example Notebooks. You can skip this step if you're working from a Conda environment which already has Jupyter, but be aware that if the wrong `jupyter` executable is found on your path then import errors may arise.
   ```console
   conda install jupyter
   ```
   If you are running Python 3.11 and get an `UnsatisfiableError` you may need to install Jupyter from `conda-forge`:
   ```console
   conda install --channel=conda-forge jupyter
   ```
   The example notebooks can then be accessed by running
   ```console
   python -m pybads
   ```

If you wish to install directly from latest source code, please see the [instructions for developers and contributors](/docs/development.html#installation-instructions-for-developers).

### Quick start

The typical workflow of PyBADS follows four steps:

1. Define the target (or objective) function;
2. Setup the problem configuration (optimization bounds, starting point, possible constraint violation function);
3. Initialize and run the optimization;
4. Examine and visualize the results.
   
Running the optimizer in step 3 only involves a couple of lines of code:

```
from pybads import BADS
  # ...
  bads = BADS(target, x0, LB, UB, PLB, PUB)
  optimize_result = bads.optimize()
```

with input arguments:

- ``target``: the target function, it takes as input a vector and returns its function evaluation;
- ``x0``: the starting point of the optimization problem. If it is not given, the starting point is randomly drawn from the problems bounds;
- ``LB`` and ``UB``: hard lower and upper bounds for the optimization region (can be ``-inf`` and ``inf``, or bounded);
- ``PLB`` and ``PUB``: *plausible* lower and upper bounds, that represent our best guess at bounding the region where the solution might lie;
- ``non_box_cons`` (optional): a callable non-bound constraints function.

The outputs are:

- ``optimize_result``: a ``OptimizeResult`` which presents the most important information about the solution and the optimization problem.

  - ``"x"``: the minimum point found by the optimizer;
  - ``"fval"``: the value of the function at the given solution.

The ``optimize_result`` object can be manipulated in various ways, see the [OptimizeResult](https://acerbilab.github.io/pybads/api/classes/optimize_result.html) class documentation.

Examples of usages of PyBADS are present in the directory `examples` of the repository, which provides a Tutorial from simple to more complex problems like noisy targets (see [this example notebook](examples/pybads_example_3_noisy_objective.ipynb)).

In order to run any of these examples ensure you have followed the installation guideline, then you can run them using the following commands:

1. Activate the environment: `conda activate pybads-dev`
2. From the `pybads` folder, run: `python examples/example_file`

In addition, checkout the [BADS FAQ](https://github.com/acerbilab/bads/wiki#bads-frequently-asked-questions) page for practical recommendations, such as how to set `LB` and `UB`, and other practical insights. Although, the FAQ refers to the MATLAB version of BADS, most of the concepts still apply for PyBADS. 


## References and citation

1. Acerbi, L. (2017). Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. In *Advances in Neural Information Processing Systems 31*: 8222-8232. ([paper + supplement on arXiv](<https://arxiv.org/abs/1705.04405>), [NeurIPS Proceedings](<https://papers.nips.cc/paper/2017/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html))

You can cite BADS in your work with something along the lines of

> We optimized the log likelihoods of our models using Bayesian adaptive direct search (BADS; Acerbi and Ma, 2017). BADS alternates between a series of fast, local Bayesian optimization steps and a systematic, slower exploration of a mesh grid.

Besides formal citations, you can demonstrate your appreciation for PyBADS in the following ways:

- *Star :star:* the BADS repository on GitHub;
- [Subscribe](http://eepurl.com/idcvc9) to the lab's newsletter for news and updates (new features, bug fixes, new releases, etc.);
- [Follow Luigi Acerbi on Twitter](https://twitter.com/AcerbiLuigi) for updates about BADS/PyBADS and other projects;
- Tell us about your model-fitting problem and your experience with PyBADS (positive or negative) in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions).

*Note*: If you are interested in estimating posterior distributions (i.e., uncertainty and error bars) over model parameters, and not just point estimates, you might also want to check out Variational Bayesian Monte Carlo for Python ([PyVBMC](https://github.com/acerbilab/pyvbmc)), a package for Bayesian posterior and model inference which can be used in synergy with PyBADS.

### BibTeX

```BibTeX
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

PyBADS was developed from the original MATLAB toolbox by [members](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people) (past and current) of the [Machine and Human Intelligence Lab](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/) at the University of Helsinki. The ongoing Python port is being supported by the Academy of Finland Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/).
