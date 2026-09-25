************
Installation
************

PyBADS is available via ``pip`` and ``conda-forge``.

1. Install with::

     python -m pip install pybads

   or::

     conda install --channel=conda-forge pybads

   PyBADS requires Python version 3.10 or newer.

2. (Optional): Install `Jupyter Notebook <https://jupyter.org/install>`__ to run the examples. You can skip this step if your environment already has Jupyter Notebook, but be aware that if the wrong ``jupyter`` executable is found on your path then import errors may arise. ::

     python -m pip install notebook

   or, with Conda::

     conda install --channel=conda-forge jupyter

   The example notebooks can then be accessed by running ::

     python -m pybads

You can run PyBADS's internal tests with ::

  python -m pip install "pybads[test]"
  pytest --pyargs pybads --reruns=3

The `--reruns=3` argument allows re-trying a failed test up to 3 times, as many of PyBADS's tests are stochastic in nature.

If you wish to install directly from latest source code, please see the :ref:`installation instructions for developers`.
