********************************************
Instructions for developers and contributors
********************************************

PyBADS is the port of the MATLAB BADS algorithm to Python 3.x (development has targeted version 3.9 and up).

The reference code is the :labrepos:`MATLAB toolbox <bads>`.

The documentation is available at: https://acerbilab.github.io/pybads/

Installation instructions for developers
########################################

Release versions of PyBADS are available via ``pip`` and ``conda-forge``, but developers will need to work with the latest source code. They should follow these steps to install:

1. (Optional, but recommended for development): Create a new environment in Conda and activate it. Requires Python 3.9 or newer::

     conda create --name pybads-env python=3.9
     conda activate pybads-env

2. Clone the PyBADS and GPyReg GitHub repos locally::

     git clone https://github.com/acerbilab/pybads
     git clone https://github.com/acerbilab/gpyreg

   (PyBADS depends on :labrepos:`GPyReg <gpyreg>`, which is a package for lightweight Gaussian process regression in Python.)
3. Install the packages and their optional development dependencies::

     cd ./gpyreg
     pip install -e ".[dev]"
     cd ../pybads
     pip install -e ".[dev]"

4. Install Jupyter to view the examples You can skip this step if you're working from a Conda environment which already has Jupyter, but be aware that if the wrong ``jupyter`` executable is found on your path then import errors may arise. ::

     conda install jupyter

We are using the dependencies listed in ``pyproject.toml``. Please list all used dependencies there. Dependencies are separated into basic dependencies, and optional development dependencies included under ``dev``.

The necessary packages can be installed with `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_ or `pip <https://pypi.org/project/pip/>`_.

Coding conventions
##################

We try to follow common conventions whenever possible. Some useful reading:

- `PEP 8 -- Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`__
- `Code style in The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`__

These basic rules should be followed to ensure coherence and to make it easy for third parties to contribute. In the following, we list more detailed conventions. Please read carefully if you are contributing to PyBADS.

Code formatting
---------------

The code is formatted using `Black <https://pypi.org/project/black/>`__ with a line length of 79, with the help of pre-commit hooks. To install and use::

    pip install pre-commit
    pre-commit install
    pre-commit run -a  # run for all files, optional

After installation, when you try to commit the staged files, git will automatically check the files and modify them for meeting the requirements of the hooks in ``.pre-commit-config.yaml``. The settings of the hooks are specified in ``pyproject.toml``. You need to restage the file if it gets modified by the hooks.

If you want, you can also check with `ruff <https://beta.ruff.rs/docs/>`__ or `Pylint <https://www.pylint.org/>`__ for more detailed errors, warnings, and suggestions.

Docstrings
----------

The docstrings are generated following the `NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.
There are add-ons to generate docstring blueprints using IDEs.

- See an example for a correct docstring from NumPy `here <https://numpydoc.readthedocs.io/en/latest/example.html>`__.
- In PyBADS, the ``OptimizeResult`` class can be taken as an example of (mostly) correct docstring structure, see :mainbranch:`here <pybads/bads/optimize_result.py>`.
- In particular, see how the single quotes and double quotes are used; the math notation is used; full stops are added at the end of each sentence, etc.

Code documentation
------------------

The documentation is currently hosted on :doc:`github.io <index>`. We build the PyBADS documentation using `Sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_. The source code of the documentation is in the :mainbranch:`docsrc folder <docsrc>` and the build version is in the :labrepos:`gh-pages <pybads/tree/gh-pages>` branch. When the documentation is re-built, it should be pushed to the ``gh-pages`` instead of committing it on the ``main`` branch.

To setup an existing PyBADS repository for building documentation, please follow the steps below:

1. One-time setup:

   a. Remove the ``docs/`` folder from the root of your existing PyBADS repo, if it is present.
   b. From the root of the PyBADS repo, run::

       git clone -b gh-pages --single-branch https://github.com/acerbilab/pybads docs

      This will clone *only* the ``gh-pages`` branch inside ``docs/``, so that changes to the docs can now be pushed directly to ``gh-pages`` from within ``docs/``.
2. From the ``main`` branch render new documentation::

    cd /docsrc (navigate to documentation source folder)
    conda active pybads-env (activate conda environment)
    make github  (this builds the doc and copies the build version to ./docs)

   (If you are using Windows, run ``.\make.bat github`` with ``cmd`` instead.)
3. Change into the ``docs/`` directory::

     cd ../docs

4. Commit the new documentation and push. `github.io <https://acerbilab.github.io/pybads/>`_ will detect the changes and rebuild the website (possibly after a few minutes). Only documentation that was built from the ``main`` branch should be committed to ``gh-pages``.

If it seems that the documentation does not update correctly (e.g., items not appearing in the sidebar or table of content), try deleting the ``./docs`` folder and the cached folder ``./docsrc/_build`` before compiling the documentation. There is a command for that::

    make clean

(If you are using Windows, run ``.\make.bat clean`` with ``cmd`` instead.)

General structure
.................

For each new class, function, etc. a ``.rst`` file needs to be created in an appropriate folder. The folder names are arbitrary, for now we have ``functions``, ``classes``, etc.
The ``.rst`` file contains the text in `reStructuredText format <https://en.wikipedia.org/wiki/ReStructuredText>`__, a lightweight markup language with special commands that tell Sphynx where to compile the documentation, for example::

    .. autoclass:: pybads.bads.BADS
      :members:

Refer to existing documentation for an overview of the file structure. So far the documentation includes the following:

- Status of the port (what is missing?);
- Reference to the respective file of the original :labrepos:`MATLAB <bads>` implementation;
- Known issues (if something is currently suboptimal in PyBADS);
- The documentation of the Python code (generated from the docstrings).

For each new file, a link needs to be added manually to the :doc:`index page </index>`.
Please keep the documentation up to date. (Sphinx logs possible issues when compiling the documentation.)

Exceptions
----------

Please use standard Python exceptions whenever it is sensible. Here is a list of those `exceptions <https://docs.python.org/3/library/exceptions.html>`__.

``git`` commits
---------------

Commits follow the `conventional commits <https://www.conventionalcommits.org/en/v1.0.0/>`__ style. This makes it easier to collaborate on the project. A cheat sheet is can be found `here <https://cheatography.com/albelop/cheat-sheets/conventional-commits/>`__.

Please do not submit pull requests with unfinished code or code which does not pass all tests. Work on feature branches whenever possible and sensible. All PRs must be approved by another developer before being merged to the main branch. `Read this <https://martinfowler.com/bliki/FeatureBranch.html>`__ ::

    git checkout -b <new-feature>
    [... do stuff and commit ...]
    git push -u origin <new-feature>
    [... when finished created pull request on github ...]

If you switch to an existing branch using ``git checkout``, remember to ``pull`` before making any change as it is not done automatically.

Modules and code organization
-----------------------------

We have decided against general util/misc modules. This means that general-purpose functions should be included in a fitting existing module or in their own module. This approach encourages us to keep functions general and coherent to their scope. Furthermore, it improves readability for new collaborators. See some reading about that `here <https://breadcrumbscollector.tech/stop-naming-your-python-modules-utils/>`__.

Testing
-------

The testing is done using ``pytest`` with unit tests for each class in the respective folder.
Tests can be run with::

    pytest test_filename.py
    pytest
    pytest --reruns 5 --cov=. --cov-report html:cov_html

The final command creates an html folder with a full report on coverage -- double-check it from time to time. Some tests are stochastic and occasionally fail: Tests can be automatically rerun by specifying e.g. ``--reruns 3``.

A few comments about testing:

- Testing is mandatory! The full suite of tests is automatically run before any pull request can be merged into ``main``. The tests run on Windows, Linux, and macOS. Automatic tests are skipped for PRs which do not change the source code or tests (e.g., changes to the documentation only).
- Still, as a good practice, please rerun all tests before major commits and pull requests. This might take a while, but it is worth it to avoid surprises.
- Please try to keep the total runtime of the tests minimal for the task at hand.
- As a good practice, please rerun all tests before major commits and pull requests (might take a while, but it is worth it to avoid surprises).
- A nice way of proceeding is 'test first': write a test first, make it fail, write the code until the test is passed.
- Many methods are tested against test cases produced with the original :labrepos:`MATLAB implementation <bads>`.
- The ``pytest-mock`` library is very useful for testing. It allows you to replace parts of your system under test with mock objects and make assertions about how they have been used. (Perhaps we should switch to ``unittest.mock`` in the future, which is part of the Python standard library.)
