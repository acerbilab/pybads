---
title: 'PyBADS: Fast and robust black-box optimization in Python'
tags:
  - Python
  - Bayesian optimization
  - Black-box optimization
  - Optimization
  - Machine learning
  - Gaussian Processes
authors:
  - name: Gurjeet Sangra Singh
    equal-contrib: false
    corresponding: true
    affiliation: 1, 3
    orcid: 0009-0008-2340-5867
  - name: Luigi Acerbi
    equal-contrib: false
    corresponding: true
    affiliation: 2
    orcid: 0000-0001-7471-7336
affiliations:
  - name: University of Geneva
    index: 1
  - name: University of Helsinki
    index: 2
  - name: University of Applied Sciences and Arts Western Switzerland (HES-SO)
    index: 3
date: 10 June 2023
bibliography: PyBADS_JoSS.bib
---

# Summary
PyBADS is a Python implementation of the Bayesian Adaptive Direct Search (BADS) algorithm for fast and robust *black-box* optimization [@acerbi2017practical]. BADS is an optimization algorithm designed to efficiently solve difficult optimization problems where the objective function is rough (non-convex, non-smooth), mildly expensive (e.g., the function evaluation requires more than 0.1 seconds), possibly noisy, and gradient information is unavailable. With BADS, these issues are well addressed, making it an excellent choice for fitting computational models using methods such as maximum-likelihood estimation.
The algorithm scales efficiently to black-box functions with up to $D \approx 20$ continuous input parameters and supports bounds or no constraints. PyBADS comes along with an easy-to-use Pythonic interface for running the algorithm and inspecting its results. PyBADS only requires the user to provide a Python function for evaluating the target function, and optionally other constraints.

Extensive benchmarks on both artificial test problems and large real model-fitting problems models drawn from cognitive, behavioral and computational neuroscience, show that BADS performs on par with or better than many other common and state-of-the-art optimizers [@acerbi2017practical], making it a general model-fitting tool which provides fast and robust solutions. 

# Statement of need
Many optimization problems in science and engineering involve complex and expensive simulations or numerical approximations such that the target function can only be evaluated at a point with moderate to high cost, possibly yielding stochastic outcomes, and gradients are unavailable (or exceedingly expensive) -- the typical *black-box* scenario. There is a large landscape of derivative-free optimization algorithms for tackling black-box problems  [@Rios2013], many of which follow variants of direct-search methods [@MADS; @stoMADS; @orthoMADS; @deng2006adaptation].
Despite their theoretical guarantees, direct search methods require a large number of function evaluations and have limited support for handling stochastic targets. 

Conversely, Bayesian Optimization (BayesOpt) is a recently popular family of methods that has shown effectiveness in solving *very* costly black-box problems in machine learning and engineering with very few, possibly noisy, function evaluations [@garnett_bayesoptbook_2023; @reviewBO; @agnihotri2020exploring]. However, classical BayesOpt requires specific technical knowledge to be implemented or tuned beyond simple tasks, since vanilla BayesOpt applied to complex real-world problems can be strongly affected by deviations from the algorithm's assumptions (model misspecification), a problem rarely dealt with in current implementations. Moreover, traditional BayesOpt methods assume *highly expensive* target functions (e.g., with evaluation costs of hours or more), whereas many computational models might only have a *moderate* evaluation cost (e.g., from a fraction of a second to a few seconds), meaning that the optimization algorithm should add only a relatively small overhead. 

PyBADS addresses all these problems as a fast hybrid algorithm that combines the strengths of BayesOpt and the Mesh Adaptive Direct Search [@MADS] method. In contrast to other black-box optimization algorithms, PyBADS is both *fast* in terms of wall-clock time and *sample-efficient* in terms of the number of target evaluations (typically of the order of a few hundred), with support for noisy targets. Moreover, PyBADS does not require any specific tuning and runs off-the-shelf with its well-modularized Python API.

## Method
PyBADS follows the Mesh Adaptive Direct Search [MADS, @MADS] schema for minimizing the given objective function. The algorithm alternates between a series of fast local Bayesian Optimization (BayesOpt) steps, referred to as *search stage*, and systematic exploration of the mesh space in a neighborhood of the current point, known as *poll stage*, based on the MADS poll method [@MADS]; see Figure \ref{fig:example}.
Briefly:

  - In the poll stage, points are evaluated on a mesh by taking steps in one (non-orthogonal) direction at a time, until an improvement is found or all directions have been tried. The step size is doubled in case of success, halved otherwise.
  - In the search stage, a Gaussian process (GP) surrogate model [@rasmussen_gaussian_2006] of the target function is fit to a local subset of the points evaluated so far. New points to evaluate are quickly chosen according to a *lower confidence bound* strategy that trades off between exploration of uncertain regions (high GP uncertainty) and exploitation of promising solutions (low GP mean). The search switches back to the poll stage after repeated failures to find an improvement over the current point.

![Contour plots and PyBADS exploration of a two-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) function. Lines represent the contours of the true target (left) and of the GP surrogate model built during the search stage (right). The solid black diamonds indicate new points chosen by the poll method (here for simplicity a simple orthogonal poll), the grey circles represent the previously sampled points, and the orange solid star represents the global minimum of the function. When switching to the search stage, the blue diamond describes the point selected by the active sampling method based on the lower confidence bound obtained from the GP surrogate model (green region). \label{fig:example}](bads-optimization.png)

This alternation between the two stages makes BADS uniquely robust and effective. The poll stage follows a slow but steady "model-free" optimization with theoretical guarantees. Conversely, the search stage exploits a powerful "model-based" GP surrogate to propose potentially large steps, which can be extremely effective if the surrogate is able to approximate the target well. Notably, this strategy is fail-safe in that if the GP fails to locally model the target, the search will fail and PyBADS will fall back to the safer poll method. The points acquired during the poll will afford a construction of a better surrogate at the next search, and so on.
In addition, when the target is noisy, BADS follows some effective heuristics for calibrating the surrogate model, checking the reliability of the predictions, and reassessing the estimated value of the current point in light of the new points. 

Thanks to these techniques, our algorithm has demonstrated high robustness and effectiveness in solving optimization problems with noisy and complex non-convex objective functions.

## Related work
Similarly to PyBADS, relevant libraries have been developed over the years in the area of Bayesian Optimization, such as BoTorch [@balandat2020botorch],  GPflowOpt [@GPflowOpt2017], Spearmint [@pmlr-v32-snoek14], among others. Instead, NOMAD [@nomad4paper] is the main reference library for *pattern search* algorithms, and it implements several variants of MADS in C++, by providing Python and Julia interface bindings.

Differently to these algorithms, PyBADS comes with a unique hybrid, fast and robust combination of direct search (MADS) and Bayesian Optimization. This combination of strategies protects against failures of the GP surrogate models – whereas vanilla BayesOpt does not have such fail-safe mechanisms, and can be strongly affected by misspecification of the surrogate GP model. PyBADS has also been designed to avoid problem-specific tuning, making it a generic tool for model fitting. Compared to other approaches, PyBADS also has the advantage of natively accommodating  target functions with heteroskedastic (input-dependent) observation noise. The results of our approach demonstrate that a hybrid Bayesian approach to optimization can be beneficial beyond the domain of costly black-box functions.
Finally, unlike most other BayesOpt packages, targeted to an audience of machine learning researchers, PyBADS comes with a neat API library and well-structured, user-friendly documentation.

PyBADS was developed in parallel to PyVBMC, a new software for sample-efficient Bayesian inference [@huggins2023pyvbmc; @acerbi2018variational; @acerbi2020variational; @acerbi2019exploration]. PyBADS can be used in combination with PyVBMC, by providing an effective way of initializing the inference algorithm at the maximum-a-posteriori (MAP) solution.

## Applications and usage

The BADS algorithm, in its MATLAB implementation, has already been applied in multiple fields, especially in neuroscience where it finds a broad audience by efficiently solving difficult model-fitting problems [@cao2019causal; @Tajima2019; @Li2020; @DAUBE20191924]. Other fields in which BADS has been successfully applied include control engineering [@stenger2022benchmark], electrical engineering [@li2022topology], material engineering [@ren2021novel], robotics [@REN2020575], petroleum science [@FENG20222879], environmental economics [@Wildfires2020], and cognitive science [@STENGARD2022105160; @vanOpheusden2023]. Moreover, BADS has been shown to perform best in most settings of a black-box optimization benchmark for control engineering [@stenger2022benchmark], highlighting the effectiveness of our algorithm compared to other BayesOpt and direct-search approaches.
With PyBADS, we bring the same sample-efficient and robust optimization to the wider open-source Python community, while improving the interface, test coverage, and documentation.

The package is available on both PyPI (``pip install pybads``) and ``conda-forge``, and provides an idiomatic and accessible interface, only depending on standard, widely available scientific Python packages [@harris_array_2020; @scipy]. The user only needs to give a few basic details about the objective function and its parameter space, and PyBADS handles the rest of the optimization task. PyBADS includes automatic handling of bounded variables, robust termination conditions, sensible default settings, and does not need tunable parameters. At the same time, experienced users can easily supply their own options. We have extensively tested the algorithm and implementation details for correctness and performance. We provide detailed [tutorials](https://github.com/acerbilab/pybads/tree/main/examples), so that PyBADS may be accessible to those not already familiar with black-box optimization, and our comprehensive [documentation](https://acerbilab.github.io/pybads) will aid not only new users but future contributors as well.

# Acknowledgments

We thank Bobby Huggins, Chengkun Li, Marlon Tobaben and Mikko Aarnos for helpful comments and feedback.
Work on the PyBADS package is supported by the Academy of Finland Flagship programme: Finnish Center for Artificial Intelligence FCAI.

# References
