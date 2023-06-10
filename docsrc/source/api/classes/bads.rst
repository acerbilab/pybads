:tocdepth: 2

========
``BADS``
========

.. note::

  The ``BADS`` class implements the Bayesian Adaptive Direct Search (BADS) algorithm.

  BADS attempts to solve an unbounded, bounded or nonlinearly constrained optimization (minimization) problem, and is compatible with both noiseless and noisy target functions.

  To perform the optimization, first initialize a ``BADS`` object and then call ``bads.optimize()`` on the instance.

  See below for more details on the ``BADS`` class methods and interface. The primary entry-points for users are the ``BADS`` class, which initializes the algorithm, and the :ref:`\`\`OptimizeResult\`\`` class, which represents the returned optimization solution. The :ref:`Basic options` may also be useful.

.. autoclass:: pybads.bads.BADS
   :members:
