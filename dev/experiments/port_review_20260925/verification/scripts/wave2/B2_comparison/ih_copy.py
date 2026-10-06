import time

import gpyreg
import numpy as np

import pybads
from pybads.utils.iteration_history import IterationHistory

print(pybads.__file__)
print(gpyreg.__file__)
ncopies = [0]


class Tracked:
    def __init__(self, a):
        self.a = a

    def __deepcopy__(self, memo):
        ncopies[0] += 1
        return Tracked(self.a.copy())


ih = IterationHistory(["gp"])
payload = np.zeros((150, 150))
t = time.perf_counter()
for i in range(200):
    ih.record("gp", Tracked(payload), i)
print(
    "records: 200, deep copies of stored objects:",
    ncopies[0],
    "time %.2fs" % (time.perf_counter() - t),
)
