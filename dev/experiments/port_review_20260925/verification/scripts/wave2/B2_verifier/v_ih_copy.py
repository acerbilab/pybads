"""K7: IterationHistory._expand_array reassigns the grown array through
__setitem__, which deep-copies it: every record at a new index copies every
stored GP again."""
import copy
import time

import common
import gpyreg as gpr
import numpy as np

from pybads.utils.iteration_history import IterationHistory

n_copies = {"n": 0}
orig_dc = gpr.GP.__deepcopy__ if hasattr(gpr.GP, "__deepcopy__") else None


class CountedGP(gpr.GP):
    def __deepcopy__(self, memo):
        n_copies["n"] += 1
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new


D = 3
N = 150
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (N, D))
y = np.sum(X**2, 1, keepdims=True) + 0.1 * rng.standard_normal((N, 1))
gp = CountedGP(
    D=D,
    covariance=gpr.covariance_functions.RationalQuadraticARD(),
    mean=gpr.mean_functions.ConstantMean(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
hyp = np.zeros((1, len(gp.get_hyperparameters(as_array=True).ravel())))
hyp[0, -2] = np.log(0.1)
gp.update(X_new=X, y_new=y, hyp=hyp)
t0 = time.perf_counter()
copy.deepcopy(gp)
t1 = time.perf_counter()
print(f"one deep copy of a GP with {N} points: {1e3*(t1-t0):.2f} ms")
for n_rec in [50, 100, 200]:
    ih = IterationHistory(["gp", "fval"])
    n_copies["n"] = 0
    t0 = time.perf_counter()
    for i in range(n_rec):
        ih.record("gp", gp, i)
        ih.record("fval", 0.0, i)
    t = time.perf_counter() - t0
    print(
        f"{n_rec} records: {n_copies['n']} GP deep copies (n(n+1)/2 + n = {n_rec*(n_rec+1)//2 + n_rec}), {t:.2f} s"
    )
