
# PyBADS: Bayesian Adaptive Direct Search in Python

## What is it?

PyBads is a Python implementation of the Bayesian Adaptive Direct Search (BADS) algorithm for solving difficult and mildly expensive optimization problems, originally implemented [in MATLAB](https://github.com/lacerbi/bads). BADS has been intensively tested for fitting a variety of computational models, and is currently being used in many computational labs around the world (see [Google Scholar](https://scholar.google.co.uk/scholar?cites=7209174494000095753&as_sdt=2005&sciodt=0,5&hl=en) for many example applications).

In a benchmark with real model-fitting problems, BADS performed on par or better than many other common and state-of-the-art optimizers, as shown in the original paper presented at NeurIPS in 2017 [[1](#references-and-citation)].

PyBADS requires no specific tuning and runs off-the-shelf like other Python optimizers (e.g., `scipy.optimize.minimize`).

*Note*: If you are interested in estimating posterior distributions (i.e., uncertainty and error bars) over model parameters, and not just point estimates, you might also want to check out Variational Bayesian Monte Carlo for Python ([PyVBMC](https://github.com/acerbilab/pyvbmc)), a package for Bayesian posterior and model inference which can be used in synergy with PyBADS.

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
bads = BADS(target, x0, lb, ub, plb, pub)
optimize_result = bads.optimize()
```

with input arguments:

- ``target``: the target function, it takes as input a vector and returns its function evaluation;
- ``x0``: the starting point of the optimization problem. If it is not given, the starting point is randomly drawn from the problems bounds;
- ``lb`` and ``ub``: hard lower and upper bounds for the optimization region (can be ``-inf`` and ``inf``, or bounded);
- ``plb`` and ``pub``: *plausible* lower and upper bounds, that represent our best guess at bounding the region where the solution might lie;
- ``non_box_cons`` (optional): a callable non-bound constraints function.

The outputs are:

- ``optimize_result``: a ``OptimizeResult`` which presents relevant information about the solution and the optimization problem. In particular:
  - ``"x"``: the minimum point found by the optimizer;
  - ``"fval"``: the value of the function at the given solution.

For a full list and description of the entries of the ``optimize_result`` object, see the [OptimizeResult](https://acerbilab.github.io/pybads/api/classes/optimize_result.html) class documentation.

Examples of usages of PyBADS are present in the directory `examples` of the repository, which provides a tutorial going from simpler to more complex problems, such as noisy targets (see [this example notebook](examples/pybads_example_3_noisy_objective.ipynb)).

In addition, checkout the [BADS FAQ](https://github.com/acerbilab/bads/wiki#bads-frequently-asked-questions) page for practical recommendations, such as how to set the bounds `LB` and `UB`, and other practical insights. Even though the FAQ refers to the MATLAB version of BADS, most of the concepts still apply to PyBADS.

## Next steps

Once installed, example Jupyter notebooks can be found in the `pybads/examples` directory. They can also be [viewed statically](https://acerbilab.github.io/pybads/index.html#examples) on the [main documentation pages](https://acerbilab.github.io/pybads/index.html). These examples will walk you through the basic usage of PyBADS as well as some if its more advanced features.

For practical recommendations, such as how to set `lb` and `ub` and the plausible bounds, check out the FAQ on the [BADS wiki](https://github.com/acerbilab/bads/wiki). The wiki was written with the MATLAB toolbox in mind, but the general advice applies to the Python version as well.

## How does it work?

PyBADS/BADS follows a [mesh adaptive direct search](http://epubs.siam.org/doi/abs/10.1137/040603371) (MADS) procedure for function minimization that alternates **poll** steps and **search** steps (see **Fig 1**). 

- In the **poll** stage, points are evaluated on a mesh by taking steps in one direction at a time, until an improvement is found or all directions have been tried. The step size is doubled in case of success, halved otherwise. 
- In the **search** stage, a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) (GP) is fit to a (local) subset of the points evaluated so far. Then, we iteratively choose points to evaluate according to a *lower confidence bound* strategy that trades off between exploration of uncertain regions (high GP uncertainty) and exploitation of promising solutions (low GP mean).

**Fig 1: BADS procedure** ![BADS procedure](https://raw.githubusercontent.com/acerbilab/bads/master/docs/bads-cartoon.png)

See [here](https://github.com/lacerbi/optimviz) for a visualization of several optimizers at work, including BADS.

See our paper for more details ([Acerbi and Ma, 2017](#references-and-citation)).

## Troubleshooting and contact

PyBADS is under active development. The original BADS algorithm has been extensively tested in several benchmarks and published papers, and the some benchmarks have been replicated using PyBADS. But as with any optimization method, you should double-check your results.

If you have trouble doing something with PyBADS, spot bugs or strange behavior, or you simply have some questions, please feel free to:
- Post in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions) with questions or comments about PyBADS, your problems & applications;
- [Open an issue](https://github.com/acerbilab/pybads/issues/new) on GitHub;
- Contact the project lead at <luigi.acerbi@helsinki.fi>, putting 'PyBADS' in the subject of the email.


## References and citation

1. Acerbi, L. & Ma, W. J. (2017). Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. In *Advances in Neural Information Processing Systems 31*: 8222-8232. ([paper + supplement on arXiv](https://arxiv.org/abs/1705.04405), [NeurIPS Proceedings](https://papers.nips.cc/paper/2017/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html))

You can cite PyBADS in your work with something along the lines of

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

PyBADS was developed from the original MATLAB toolbox by [members](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people) (past and current) of the [Machine and Human Intelligence Lab](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/) at the University of Helsinki. The Python port was partially supported by the Academy of Finland Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/).
