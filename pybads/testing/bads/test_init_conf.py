from importlib.metadata import version, PackageNotFoundError
import pytest

def test_version():
    __version__ = version("pybads")
