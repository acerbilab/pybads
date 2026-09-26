import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)
