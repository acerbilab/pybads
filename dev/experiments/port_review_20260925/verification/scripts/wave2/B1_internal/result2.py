import threading
import traceback

from common import *


class Model:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0

    def nll(self, x):
        self.calls += 1
        return quad(x)


m = Model()
b = BADS(
    m.nll,
    np.array([[0.3, -0.2]]),
    np.array([[-2.0, -2.0]]),
    np.array([[2.0, 2.0]]),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 20},
)
try:
    b.optimize()
except TypeError:
    tb = traceback.format_exc().splitlines()
    print("\n".join(l for l in tb if "pybads" in l or "Error" in l))
print("evaluations made before the error:", m.calls)
