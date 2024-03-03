******
PyBADS
******

PyBADS is a Python implementation of the Bayesian Adaptive Direct Search (BADS) algorithm for solving difficult and moderately expensive optimization problems, previously implemented :labrepos:`in MATLAB <bads>`.

What is it?
###########

BADS is a fast hybrid Bayesian optimization algorithm designed to solve difficult optimization problems, in particular related to fitting computational models (e.g., `via maximum likelihood estimation <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>`__).

BADS has been intensively tested for model fitting in science and engineering and is currently used in many computational labs around the world (see `Google Scholar <https://scholar.google.co.uk/scholar?cites=7209174494000095753&as_sdt=2005&sciodt=0,5&hl=en>`__ for many example applications).

In our benchmark with real model-fitting problems from computational and cognitive neuroscience, BADS performed on par or better than many other common and state-of-the-art optimizers, as shown in the original BADS paper (`Acerbi and Ma, 2017 <#references>`_).

BADS requires no specific tuning and runs off-the-shelf similarly to other Python optimizers, such as those in ``scipy.optimize.minimize``.

*Note*: If you are interested in estimating posterior distributions (i.e., uncertainty and error bars) over model parameters, and not just point estimates, you should check out Variational Bayesian Monte Carlo for Python (:labrepos:`PyVBMC <pyvbmc>`), a package for Bayesian posterior and model inference which can be used in synergy with PyBADS.

How does it work?
-----------------

PyBADS/BADS follows a `mesh adaptive direct search <http://epubs.siam.org/doi/abs/10.1137/040603371>`__ (MADS) procedure for function minimization that alternates **poll** steps and **search** steps (see **Fig 1**). 

- In the **poll** stage, points are evaluated on a mesh by taking steps in one direction at a time, until an improvement is found or all directions have been tried. The step size is doubled in case of success, halved otherwise. 
- In the **search** stage, a `Gaussian process <https://distill.pub/2019/visual-exploration-gaussian-processes/>`__ (GP) is fit to a (local) subset of the points evaluated so far. Then, we iteratively choose points to evaluate according to a *lower confidence bound* strategy that trades off between exploration of uncertain regions (high GP uncertainty) and exploitation of promising solutions (low GP mean).

.. image:: _static/bads-cartoon.png
    :align: center
    :alt: Fig 1: BADS procedure

Fig 1: BADS procedure

See `here <https://github.com/lacerbi/optimviz>`__ for a visualization of several optimizers at work, including BADS.

See our paper for more details (`Acerbi and Ma, 2017 <#references>`_).

.. Example run
   -----------
   TODO: Put a Gif here showing a BADS run on a simple problem (e.g on the Rosenbrock's banana function).

Should I use PyBADS?
--------------------

BADS is particularly recommended when:

- the objective function landscape is rough (nonsmooth), typically due to numerical approximations or noise;
- the objective function is at least moderately expensive to compute (e.g., more than 0.1 second per function evaluation);
- the gradient is unavailable (black-box function);
- the number of input parameters is up to about `D = 20` or so.

How-to
#############
.. toctree::
   :maxdepth: 2
   :titlesonly:

   installation
   quickstart
   examples
   documentation

Contributing
############
.. toctree::
   :maxdepth: 1
   :titlesonly:

   development

References
###############

1. Singh, S. G. & Acerbi, L. (2024). PyBADS: Fast and robust black-box optimization in Python. Journal of Open Source Software, 9(94), 5694. (`paper on JOSS <https://doi.org/10.21105/joss.05694>`__).

2. Acerbi, L. & Ma, W. J. (2017). Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. In *Advances in Neural Information Processing Systems 31*: 8222-8232. (`paper + supplement on arXiv <https://arxiv.org/abs/1705.04405>`__, `NeurIPS Proceedings <https://papers.nips.cc/paper/2017/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html>`__)

Please cite both references if you use PyBADS in your work (the 2017 paper introduced the framework, and the latest one is its Python library). You can cite PyBADS in your work with something along the lines of

    We optimized the log likelihoods of our models using Bayesian adaptive direct search (BADS; Acerbi and Ma, 2017), via the PyBADS software (Singh and Acerbi, 2024). PyBADS alternates between a series of fast, local Bayesian optimization steps and a systematic, slower exploration of a mesh grid.

BibTeX
------
::

  @article{singh2023pybads,
    title={{PyBADS}: {F}ast and robust black-box optimization in {P}ython}, 
    author={Gurjeet Sangra Singh and Luigi Acerbi},
    publisher = {The Open Journal},
    journal = {Journal of Open Source Software},
    year = {2024},
    volume = {9},
    number = {94},
    pages = {5694},
    url = {https://doi.org/10.21105/joss.05694},
    doi = {10.21105/joss.05694}
  }

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
Work on the PyBADS package was supported by the Research Council of Finland Flagship programme: `Finnish Center for Artificial Intelligence FCAI <https://fcai.fi/>`_.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   about_us
