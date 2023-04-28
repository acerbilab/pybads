***************
Getting started
***************

The best way to get started with PyBADS is via the tutorials and worked examples.
In particular, start with :ref:`PyBADS Example 1: Basic usage` and continue from there.

If you are already familiar with optimization problems and optimizers, you can find a summary usage below.

Summary usage
=============

The typical usage pipeline of PyBADS follows four steps:

1. Define the target (or objective) function;
2. Setup the problem configuration (optimization bounds, starting point, possible constraint violation function);
3. Initialize and run the optimization;
4. Examine and visualize the results.

Running the optimizer in step 3 only involves a couple of lines of code:

.. code-block:: python

  from pybads import BADS
  # ...
  bads = BADS(target, x0, lb, ub, plb, pub)
  optimize_result = bads.optimize()

with input arguments:

- ``target``: the target function, it takes as input a vector and returns its function evaluation;
- ``x0``: the starting point of the optimization problem. If it is not given, the starting point is randomly drawn from the problems bounds;
- ``lb`` and ``ub``: hard lower and upper bounds for the optimization region (can be ``-inf`` and ``inf``, or bounded);
- ``plb`` and ``pub``: *plausible* lower and upper bounds, that represent our best guess at bounding the region where the solution might lie;
- ``non_box_cons`` (optional): a callable non-bound constraints function.

The outputs are:

- ``optimize_result``: an ``OptimizeResult`` which presents the most important information about the solution and the optimization problem. In particular:

  - ``"x"``: the minimum point found by the optimizer;
  - ``"fval"``: the value of the function at the given solution.

The ``optimize_result`` object contains more information about the optimization solution, see the :ref:`\`\`OptimizeResult\`\`` class documentation.

See the examples for more detailed information. The :ref:`Basic options` may also be useful.

In addition, checkout the `BADS FAQ <https://github.com/acerbilab/bads/wiki#bads-frequently-asked-questions>`__ page for practical recommendations, such as how to set `lb` and `ub`, and other practical insights. Even though the FAQ refers to the MATLAB version of BADS, most of the concepts still apply to PyBADS.
