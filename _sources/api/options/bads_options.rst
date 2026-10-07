============
BADS options
============
The options are divided into two groups:

- **Basic options:** Commonly adjusted settings, such as evaluation budgets,
  display and noise handling.
- **Advanced options:** Specialized features and settings that control the
  optimization algorithm.

You can find the default options for both groups below. ``D`` in the options
below is the number of variables that BADS optimizes: all of them but the fixed
ones, whose four bounds are equal.

Basic options
=====================
We expect these options to be routinely changed by many users.

.. include:: ./../../../../pybads/bads/option_configs/basic_bads_options.ini
   :literal:

Advanced options
=====================
This group includes ``periodic_vars`` for
:ref:`periodic variables <faq-does-pybads-support-periodic-variables-such-as-angles>`
and ``output_fcn`` for
:ref:`monitoring or stopping a run <faq-can-i-monitor-or-stop-a-run-while-it-is-running>`.
Set these options when your problem calls for them.

Most other options control the search, polling and Gaussian process model.
Keep their defaults unless a documented recommendation or a specific
algorithmic requirement calls for a change.

.. include:: ./../../../../pybads/bads/option_configs/advanced_bads_options.ini
   :literal:
