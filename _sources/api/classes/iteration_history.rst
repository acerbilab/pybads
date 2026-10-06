====================
``IterationHistory``
====================

.. note::

  The ``iteration_history`` of a ``BADS`` object holds the incumbent at the end of each iteration of its run.

  Its ``timer`` holds, at the end of each iteration, the seconds that the run has spent so far in each of its stages and in the target's evaluations, a record for PyBADS's developer tools whose format can change in any release.

  In a noisy run, the ``fval`` and ``fsd`` of every past iterate are re-estimated at the end of each iteration and of the run, so the recorded values of earlier iterations change as the run goes on. A past iterate whose re-estimate fails holds NaN for both and is not returned as the solution; the result's ``fval`` and ``fsd`` are finite.

.. autoclass:: pybads.utils.IterationHistory
   :members:
