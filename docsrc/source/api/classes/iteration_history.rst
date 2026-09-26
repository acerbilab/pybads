====================
``IterationHistory``
====================

.. note::

  The ``iteration_history`` of a ``BADS`` object holds the incumbent at the end of each iteration of its run.

  In a noisy run, the ``fval`` and ``fsd`` of every iterate are re-estimated from the current data at the end of each iteration and of the run, as in MATLAB BADS. A past iterate whose last re-estimate failed (its Gaussian process could not be rebuilt around it) is left with NaN for both, as in MATLAB BADS's ``iterList``, and out of the choice of the returned point. The current iterate keeps its estimate, and the result's ``fval`` and ``fsd`` are finite.

.. autoclass:: pybads.utils.IterationHistory
   :members:
