from importlib.metadata import PackageNotFoundError, version

import pytest


def test_version():
    __version__ = version("pybads")
