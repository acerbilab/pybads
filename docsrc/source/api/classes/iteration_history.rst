====================
``IterationHistory``
====================

.. note::

  The ``iteration_history`` of a ``BADS`` object holds the incumbent at the end of each iteration of its run.

  Its ``timer`` holds, at the end of each iteration, the seconds that the run has spent so far in each of its stages and in the target's evaluations, a record for PyBADS's developer tools whose format can change in any release.

  In a noisy run, the ``fval`` and ``fsd`` of every iterate are re-estimated from the current data at the end of each iteration and of the run, as in MATLAB BADS. A past iterate whose last re-estimate failed (its Gaussian process could not be rebuilt around it) is left with NaN for both, as in MATLAB BADS's ``iterList``, and out of the choice of the returned point. The current iterate keeps its estimate, and the result's ``fval`` and ``fsd`` are finite.

.. autoclass:: pybads.utils.IterationHistory
   :members:
