import importlib.util
import sys

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


A = load("bt_before/dev/scripts/benchmark_targets.py", "bta")
B = load("bt_after/dev/scripts/benchmark_targets.py", "btb")
ca = A.suite_configs("default")
cb = B.suite_configs("default")
print(
    "configs", len(ca), len(cb), [c.label for c in ca] == [c.label for c in cb]
)
rng = np.random.default_rng(1)
bad = 0
for a, b in zip(ca, cb):
    assert a == b or a.label == b.label, (a, b)
    for seed in range(30):
        pa, pb = a.make(seed), b.make(seed)
        same = (
            np.array_equal(pa.x0, pb.x0)
            and pa.options == pb.options
            and pa.f_min == pb.f_min
            and np.array_equal(pa.x_min, pb.x_min)
        )
        for arr in ("lb", "ub", "plb", "pub"):
            same &= np.array_equal(getattr(pa, arr), getattr(pb, arr))
        if seed < 3:
            X = pa.plb + rng.random((50, pa.D)) * (pa.pub - pa.plb)
            same &= np.array_equal(pa.f_vec(X), pb.f_vec(X))
            ya = [pa.fun(x) for x in X[:10]]
            yb = [pb.fun(x) for x in X[:10]]
            same &= ya == yb
            if pa.non_box_cons is not None:
                same &= np.array_equal(pa.non_box_cons(X), pb.non_box_cons(X))
        bad += not same
print("default configs x seeds that differ:", bad)
