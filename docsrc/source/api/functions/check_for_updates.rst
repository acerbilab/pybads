=====================
``check_for_updates``
=====================

Check for a newer release
-------------------------

Ask PyPI whether a newer release of PyBADS is available:

.. code-block:: python

   import pybads

   check = pybads.check_for_updates()

The function prints one line. When PyPI has a newer release, the line names
it and gives the command that installs it, for example:

.. code-block:: text

   PyBADS 1.6.0 is available; you have 1.5.0. Update with: python -m pip install --upgrade pybads

The command follows the installer recorded with your installation:
``python -m pip install --upgrade pybads`` for pip,
``conda update --channel=conda-forge pybads`` for conda (the conda-forge
package can follow PyPI by a few days), and both when the installer is
another or unknown. The other messages are listed below, with the returned
named tuple, which gives a script the same answer. A network failure raises
no error.

Network access
--------------

PyBADS contacts PyPI only when you call this function, and makes no other
network request. The request names the installed version of PyBADS and
nothing else about your installation, and the call writes nothing to disk.

The old-release reminder
------------------------

When a run starts in an interactive session (a terminal or a Jupyter
notebook) and the installed release is more than a year old, ``BADS``
suggests calling this function, in place of the
:ref:`tip <faq-how-do-i-silence-pybads-or-send-its-output-elsewhere>` before
the first iteration line:

.. code-block:: text

   Note: PyBADS 1.5.0 was released more than a year ago. Run pybads.check_for_updates() to see whether a newer version is available.
   https://pypi.org/project/pybads/

The reminder makes no network request: it compares the release date shipped
with PyBADS with the date of the run. It appears at most once per Python
session and records the dates of its showings in ``update_reminder.json``
in PyBADS's cache directory:
``%LOCALAPPDATA%\pybads`` on Windows, ``~/Library/Caches/pybads`` on macOS,
``~/.cache/pybads`` on Linux (or under ``$XDG_CACHE_HOME``), or the directory
that ``PYBADS_CACHE_DIR`` names.

The saved dates limit reminders to three showings for each installed version,
at least 90 days apart. If the state file cannot be read or updated, these
limits may not hold across sessions; any dates that can be read still count.

To turn it off, pass ``options={"show_tips": False}`` to ``BADS``, which also
turns off the tips, or set the environment variable
``PYBADS_NO_UPDATE_REMINDER=1``.

.. autofunction:: pybads.check_for_updates
