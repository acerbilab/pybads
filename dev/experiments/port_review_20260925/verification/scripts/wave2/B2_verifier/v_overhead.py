"""I-F6: overhead = total_time / total_fun_eval_time - 1. Which evaluations
reach total_fun_eval_time? MATLAB (funlogger.m:130) adds every 'iter' and
'single' call (final samples included); its noise test calls funwrapper
directly (evalinitmesh.m:41), so it is not timed."""
import time

import common
import numpy as np

from pybads import BADS


class Timed:
    def __init__(self, f, dt):
        self.f, self.dt, self.t, self.n = f, dt, 0.0, 0

    def __call__(self, x):
        t0 = time.perf_counter()
        time.sleep(self.dt)
        y = self.f(x)
        self.t += time.perf_counter() - t0
        self.n += 1
        return y


for noisy in [False, True]:
    nrng = np.random.default_rng(0)
    base = (
        (
            lambda x: float(
                np.sum(np.ravel(x) ** 2) + 0.3 * nrng.standard_normal()
            )
        )
        if noisy
        else (lambda x: float(np.sum(np.ravel(x) ** 2)))
    )
    f = Timed(base, 0.05)
    b = BADS(
        f,
        np.array([1.0, 1.0]),
        np.full(2, -5.0),
        np.full(2, 5.0),
        np.full(2, -2.0),
        np.full(2, 2.0),
        options=dict(random_seed=0, max_fun_evals=60, display="off"),
    )
    r = b.optimize()
    fl = b.function_logger
    print(
        f"{'noisy' if noisy else 'det  '}: calls {f.n}, func_count {r['func_count']}, logged rows {fl.Xn+1}, "
        f"target time {f.t:.3f} s, logger total_fun_eval_time {fl.total_fun_eval_time:.3f} s, total_time {r['total_time']:.3f} s, "
        f"reported overhead {r['overhead']:.3f}, overhead from target time {r['total_time']/f.t - 1:.3f}, "
        f"MATLAB-style (all but the noise test) {r['total_time']/(f.t - f.t/f.n) - 1:.3f}"
    )
