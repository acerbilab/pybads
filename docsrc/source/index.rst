******
PyBADS
******

PyBADS is a Python implementation of the Bayesian Adaptive Direct Search (BADS) algorithm for solving difficult and moderately expensive optimization problems, originally implemented :labrepos:`in MATLAB <bads>`.

What is it?
###########

BADS is a fast hybrid Bayesian optimization algorithm designed to solve difficult optimization problems, in particular related to fitting computational models (e.g., `via maximum likelihood estimation <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>`__).

BADS has been intensively tested for fitting a variety of computational models, and is currently being used in many computational labs around the world (see `Google Scholar <https://scholar.google.co.uk/scholar?cites=7209174494000095753&as_sdt=2005&sciodt=0,5&hl=en>`__ for many example applications).

In our benchmark with real model-fitting problems, BADS performed on par or better than many other common and state-of-the-art optimizers, as shown in the original BADS paper `(Acerbi, 2017) <#references>`_.

BADS requires no specific tuning and runs off-the-shelf similarly to other Python optimizers, such as those in `scipy.optimize.minimize`.

*Note*: If you are interested in estimating posterior distributions (i.e., uncertainty and error bars) over model parameters, and not just point estimates, you might also want to check out Variational Bayesian Monte Carlo for Python (:labrepos:`PyVBMC <pyvbmc>`), a package for Bayesian posterior and model inference which can be used in synergy with PyBADS.

Example run
-----------
TODO: Put a Gif here showing a BADS run on a simple problem (e.g on the Rosenbrock's banana function).

Should I use PyBADS?
--------------------

BADS is particularly recommended when:

- the objective function landscape is rough (nonsmooth), typically due to numerical approximations or noise;
- the objective function is at least moderately expensive to compute (e.g., more than 0.1 second per function evaluation);
- the gradient is unavailable (black-box function);
- the number of input parameters is up to about `D = 20` or so.

Documentation
#############
.. toctree::
   :maxdepth: 1
   :titlesonly:

   installation
   quickstart
   api/classes/bads
   api/classes/optimize_result
   api/advanced_docs

Examples
########
.. toctree::
   :maxdepth: 1
   :titlesonly:
   :glob:

   _examples/*

Contributing
############
.. toctree::
   :maxdepth: 1
   :titlesonly:

   development

References
###############

1. Acerbi, L. (2017). Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. In *Advances in Neural Information Processing Systems 31*: 8222-8232. (`paper + supplement on arXiv <https://arxiv.org/abs/1705.04405>`__, `NeurIPS Proceedings <https://papers.nips.cc/paper/2017/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html>`__)

You can cite BADS in your work with something along the lines of

    We optimized the log likelihoods of our models using Bayesian adaptive direct search (BADS; Acerbi and Ma, 2017). BADS alternates between a series of fast, local Bayesian optimization steps and a systematic, slower exploration of a mesh grid.

BibTeX
------
::

  @article{acerbi2017practical,
    title={Practical {B}ayesian Optimization for Model Fitting with {B}ayesian Adaptive Direct Search},
    author={Acerbi, Luigi and Ma, Wei Ji},
    journal={Advances in Neural Information Processing Systems},
    volume={30},
    pages={1834--1844},
    year={2017}
  }

License and source
------------------

PyBADS is released under the terms of the :mainbranch:`BSD 3-Clause License <LICENSE>`.
The Python source code is on :labrepos:`GitHub <pybads>`.
You may also want to check out the original :labrepos:`MATLAB toolbox <bads>`.


Acknowledgments:
################
Work on the PyBADS package was supported by the Academy of Finland Flagship programme: `Finnish Center for Artificial Intelligence FCAI <https://fcai.fi/>`_.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   about_us
