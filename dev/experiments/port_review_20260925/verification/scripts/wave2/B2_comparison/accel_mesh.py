import logging
import sys

import gpyreg
import numpy as np

import pybads
from pybads import BADS
from pybads.bads import bads as bads_mod

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)

checks = []  # iterations (0-based) where the accelerated-mesh test ran
orig_eval = BADS._eval_improvement_


def wrapped(self, f_base, f_new, s_base, s_new, q):
    fr = sys._getframe(1)
    if fr.f_code.co_name == "_poll_step_" and "u_base" in fr.f_locals:
        checks.append(self.optim_state["iter"])
    return orig_eval(self, f_base, f_new, s_base, s_new, q)


BADS._eval_improvement_ = wrapped

failed = []  # (0-based iter) of failed polls
orig_poll = BADS._poll_step_


def poll(self, gp):
    msi = self.mesh_size_integer
    out = orig_poll(self, gp)
    if self.mesh_size_integer < msi:
        failed.append(self.optim_state["iter"])
    return out


BADS._poll_step_ = poll

D = 2
f = lambda x: float(np.sum(np.abs(x)) ** 1.5 + 0.3 * np.sum(np.cos(3 * x)))
b = BADS(
    f,
    np.array([1.3, -0.7]),
    np.full(D, -5.0),
    np.full(D, 5.0),
    np.full(D, -2.0),
    np.full(D, 2.0),
    options=dict(display="off", random_seed=1, max_fun_evals=150),
)
r = b.optimize()
steps = b.options["accelerate_mesh_steps"]
print("accelerate_mesh_steps", steps)
print("failed polls at 0-based iter:", failed)
print("accelerate test ran at 0-based iter:", checks)
print(
    "MATLAB would test at the failed polls with 1-based iter > steps, i.e. 0-based >=",
    steps,
    ":",
    [i for i in failed if i + 1 > steps],
)
