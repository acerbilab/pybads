******
PyBADS
******

PyBADS is a Python implementation of the Bayesian Adaptive Direct Search (BADS) algorithm for solving difficult and moderately expensive optimization problems, previously implemented :labrepos:`in MATLAB <bads>`.

PyBADS is one of the `open-source tools for fitting models to data <https://acerbilab.org/model-fitting/>`__ from `Luigi Acerbi's group <https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence>`__ at the University of Helsinki. Check out our other tools, such as :labrepos:`PyVBMC <pyvbmc>` for the posterior and the model evidence and :labrepos:`PyIBS <pyibs>` for models that can only be simulated.

What is it?
###########

BADS is a fast hybrid Bayesian optimization algorithm designed to solve difficult optimization problems, in particular related to fitting computational models (e.g., `via maximum likelihood estimation <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>`__).

BADS has been intensively tested for model fitting in science and engineering and is currently used in many computational labs around the world (see `Google Scholar <https://scholar.google.co.uk/scholar?cites=7209174494000095753&as_sdt=2005&sciodt=0,5&hl=en>`__ for many example applications).

In our benchmark with real model-fitting problems from computational and cognitive neuroscience, BADS performed on par or better than many other common and state-of-the-art optimizers, as shown in the original BADS paper (`Acerbi and Ma, 2017 <#references>`_).

BADS requires no specific tuning and runs off-the-shelf similarly to other Python optimizers, such as those in ``scipy.optimize.minimize``.

*Note*: If you are interested in estimating posterior distributions (i.e., uncertainty and error bars) over model parameters, and not just point estimates, you should check out Variational Bayesian Monte Carlo for Python (:labrepos:`PyVBMC <pyvbmc>`), a package for Bayesian posterior and model inference which can be used in synergy with PyBADS. PyBADS and PyVBMC are among the lab's `tools for fitting models to data <https://acerbilab.org/model-fitting/>`__.

What's new in PyBADS 1.5
------------------------

- **Faster, with equal or better results.** On our benchmark problems,
  PyBADS's own computations run almost **twice as fast** as in PyBADS 1.1.0,
  because each step is faster and runs need fewer evaluations, and it finds
  equal or better minima.
- **Periodic variables.** The option ``periodic_vars`` names the variables
  that are periodic, such as angles: BADS wraps each of them around its hard
  bounds, and the Gaussian process that models the objective is periodic
  along it (see
  :doc:`Example 6 <_examples/pybads_example_6_periodic_variables>` and the
  :ref:`FAQ <faq-does-pybads-support-periodic-variables-such-as-angles>`).
- **Fixed variables.** A variable whose four bounds, hard and plausible, are
  equal is fixed at that value, and BADS optimizes the others (see the
  :ref:`FAQ <faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value>`).
- **Evaluations made before the run.**
  ``BADS(..., precomputed_evaluations=(X, y))`` gives a run evaluations of the
  target made before it, for instance by an earlier run. The run still starts
  from ``x0`` and its initial design, but skips the points of the design that
  they hold; those nearest the incumbent join its Gaussian process from the
  first poll on, and none counts against ``max_fun_evals`` (see the
  :doc:`BADS reference <api/classes/bads>`).
- **Seeded initial design.** ``random_seed`` decides the initial design of a
  run, as it decides every other random draw; in 1.1.0 the design did not
  depend on the seed. The FAQ says
  :ref:`how to make a run reproducible <faq-how-do-i-make-a-run-reproducible>`.
- **Closer to MATLAB BADS.** PyBADS was checked line by line against MATLAB
  BADS 1.1.3, the reference implementation, and follows it more closely in
  many details, above all with noisy targets. The FAQ lists
  :ref:`what differs from MATLAB BADS <faq-i-used-bads-in-matlab-what-is-different-in-pybads>`.
- **Requirements.** PyBADS needs gpyreg 1.4.0 or later, NumPy 2.0 or later,
  SciPy 1.13 or later and matplotlib 3.9 or later.
- **FAQ, tips and a coding-agent skill.** The documentation has a
  :doc:`page of frequently asked questions <faq>`, adapted from the MATLAB
  BADS FAQ with further questions on PyBADS; a run occasionally prints a
  short tip with a link to the documentation
  (``options={"show_tips": False}`` turns them off); and the
  :mainbranch:`PyBADS skill <skills/pybads/SKILL.md>` points a coding agent
  to the documentation relevant to its task: give the agent that file, or
  copy the ``skills/pybads`` folder into its skill directory.

The :mainbranch:`changelog <CHANGELOG.md>` lists what changed since PyBADS
1.1.0. Results differ from 1.1.0, also with a fixed seed. ``BADS`` checks the
values of many options when it is created and refuses some that 1.1.0
accepted, and some calls and returned fields change: the changelog's list
"Upgrading from 1.1.0" says what to check in an existing script.

How does it work?
-----------------

PyBADS/BADS follows a `mesh adaptive direct search <http://epubs.siam.org/doi/abs/10.1137/040603371>`__ (MADS) procedure for function minimization that alternates **poll** steps and **search** steps (see **Fig 1**).

- In the **poll** stage, points are evaluated on a mesh by taking steps in one direction at a time, until an improvement is found or all directions have been tried. The step size is doubled in case of success, halved otherwise.
- In the **search** stage, a `Gaussian process <https://distill.pub/2019/visual-exploration-gaussian-processes/>`__ (GP) is fit to a (local) subset of the points evaluated so far. Then, we iteratively choose points to evaluate according to a *lower confidence bound* strategy that trades off between exploration of uncertain regions (high GP uncertainty) and exploitation of promising solutions (low GP mean).

.. image:: _static/bads-cartoon.png
    :align: center
    :alt: Fig 1: BADS procedure

Fig 1: BADS procedure. The poll's steps are equal in the normalized coordinates in which BADS works, where the plausible box spans [-1, 1] in every variable, and scale with the plausible box in the original coordinates, drawn here; in a variable that BADS maps through a log (positive bounds, and a plausible box that spans a factor of 10 or more), they grow with its value.

See `here <https://github.com/lacerbi/optimviz>`__ for a visualization of several optimizers at work, including BADS.

See our paper for more details (`Acerbi and Ma, 2017 <#references>`_).

.. Example run
   -----------
   TODO: Put a Gif here showing a BADS run on a simple problem (e.g on the Rosenbrock's banana function).

Should I use PyBADS?
--------------------

BADS is particularly recommended for problems in which:

.. include:: faq.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- suited-for: start -->
   :end-before: <!-- suited-for: end -->

The :ref:`FAQ <faq-what-do-i-do-if-pybads-is-not-suited-for-my-problem>`
says what to use for other problems.

How-to
#############
.. toctree::
   :maxdepth: 2
   :titlesonly:

   installation
   quickstart
   faq
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

1. Singh, G. S. & Acerbi, L. (2024). PyBADS: Fast and robust black-box optimization in Python. *Journal of Open Source Software*, 9(94), 5694. (`paper on JOSS <https://doi.org/10.21105/joss.05694>`__).

2. Acerbi, L. & Ma, W. J. (2017). Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. In *Advances in Neural Information Processing Systems 30*: 1834-1844. (`paper + supplement on arXiv <https://arxiv.org/abs/1705.04405>`__, `NeurIPS Proceedings <https://papers.nips.cc/paper/2017/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html>`__)

Please cite both references if you use PyBADS in your work (the 2017 paper introduced the framework, and the latest one is its Python library). You can cite PyBADS in your work with something along the lines of

    We optimized the log likelihoods of our models using Bayesian adaptive direct search (BADS; Acerbi and Ma, 2017), via the PyBADS software (Singh and Acerbi, 2024). PyBADS alternates between a series of fast, local Bayesian optimization steps and a systematic, slower exploration of a mesh grid.

BibTeX
------
::

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
You may also want to check out the original :labrepos:`MATLAB toolbox <bads>`, and the lab's other `tools for fitting models to data <https://acerbilab.org/model-fitting/>`__.


Acknowledgments:
################

PyBADS is developed by members (past and current) of the `Machine and Human Intelligence Group <https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence>`_ at the University of Helsinki and `ELLIS Institute Finland <https://www.ellisinstitute.fi/>`_. Work on the PyBADS package is supported by the Research Council of Finland (grants 356498 and 358980 to Luigi Acerbi) and its Flagship programme: `Finnish Center for Artificial Intelligence FCAI <https://fcai.fi/>`_.

Starting from version 1.1, development of PyBADS has been assisted by coding agents, including Anthropic's `Claude Opus 5.5 <https://www.anthropic.com/claude-opus-5-5>`_.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   about_us
