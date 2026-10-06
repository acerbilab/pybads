************
Installation
************

PyBADS is available via ``pip`` and ``conda-forge``.

1. Install with::

     python -m pip install pybads

   or::

     conda install --channel=conda-forge pybads

   PyBADS requires Python version 3.10 or newer.

   PyBADS 1.5 requires NumPy 2.0 or newer, and its conda-forge package
   requires Python 3.11 or newer. In an environment that holds NumPy 1.x or Python
   3.10, ``conda`` installs an older PyBADS instead, without a warning: ask
   it for ``"pybads>=1.5"``, or see the
   :ref:`FAQ <faq-conda-installs-an-older-version-of-pybads-why>`.

   To learn whether a newer release exists and how to update, see the
   :ref:`FAQ <faq-how-do-i-know-whether-a-newer-version-of-pybads-exists>`.

2. (Optional): Install `Jupyter Notebook <https://jupyter.org/install>`__ to run the examples. You can skip this step if your environment already has Jupyter Notebook, but be aware that if the wrong ``jupyter`` executable is found on your path then import errors may arise. ::

     python -m pip install notebook

   or, with Conda::

     conda install --channel=conda-forge jupyter

   The example notebooks can then be accessed by running ::

     python -m pybads

You can run PyBADS's internal tests with ::

  python -m pip install "pybads[test]"
  pytest --pyargs pybads

If you wish to install directly from latest source code, please see the :ref:`installation instructions for developers`.
