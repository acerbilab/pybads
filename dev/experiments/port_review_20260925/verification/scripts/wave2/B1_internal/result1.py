import threading

from common import *


class Model:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0

    def nll(self, x):
        with self.lock:
            self.calls += 1
        return quad(x)

    __call__ = nll


x0 = np.array([[0.3, -0.2]])
lb = np.array([[-2.0, -2.0]])
ub = np.array([[2.0, 2.0]])
m = Model()


def run(fun):
    b = BADS(
        fun,
        x0.copy(),
        lb,
        ub,
        options={"display": "off", "random_seed": 0, "max_fun_evals": 20},
    )
    r = b.optimize()
    return r["fun"] is fun, type(r["fsd"]).__name__, r["fsd"]


trycall("bound method with lock", lambda: run(m.nll))
trycall("callable object with lock", lambda: run(m))


class Big:
    def __init__(self):
        self.data = np.zeros(10)
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return quad(x)


bg = Big()


def run2():
    b = BADS(
        bg,
        x0.copy(),
        lb,
        ub,
        options={"display": "off", "random_seed": 0, "max_fun_evals": 20},
    )
    r = b.optimize()
    return r["fun"] is bg, r["fun"].n, bg.n


trycall("callable object: result['fun'] is the user's object?", run2)
