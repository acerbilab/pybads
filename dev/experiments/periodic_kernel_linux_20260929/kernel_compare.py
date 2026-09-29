"""Compare gpyreg's periodic kernels between two checkouts: speed and agreement.

    python kernel_compare.py OLD_GPYREG NEW_GPYREG
"""
import importlib.util
import sys
import timeit

import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(
        name, f"{path}/gpyreg/covariance_functions.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old = load(sys.argv[1], "cf_old")
new = load(sys.argv[2], "cf_new")


def t(f, n=20):
    return min(timeit.repeat(f, number=n, repeat=5)) / n * 1e3


rng = np.random.default_rng(0)
print(
    "timing (ms), RQ-ARD, D=3 with periods [2, 2, inf] (periodic_D3's layout)"
)
for N, M in [(50, 2048), (100, 2048), (200, 2048)]:
    X = rng.uniform(-1, 1, (N, 3))
    Xs = rng.uniform(-1, 1, (M, 3))
    per = np.array([2.0, 2.0, np.inf])
    hyp = np.array([np.log(0.3), np.log(0.5), np.log(0.4), 0.2, 0.1])
    ko, kn, k0 = (
        old.RationalQuadraticARD(per),
        new.RationalQuadraticARD(per),
        new.RationalQuadraticARD(None),
    )
    print(
        f"  cross N={N} M={M}: old {t(lambda: ko.compute(hyp, X, Xs)):.3f}"
        f"  new {t(lambda: kn.compute(hyp, X, Xs)):.3f}"
        f"  no periods {t(lambda: k0.compute(hyp, X, Xs)):.3f}"
    )
    print(
        f"  grad  N={N}:        old {t(lambda: ko.compute(hyp, X, compute_grad=True)):.3f}"
        f"  new {t(lambda: kn.compute(hyp, X, compute_grad=True)):.3f}"
        f"  no periods {t(lambda: k0.compute(hyp, X, compute_grad=True)):.3f}"
    )

print(
    "agreement, new against old: max |K_new - K_old| / sf2, and of dK / max|dK|"
)
worst = {}
for name in [
    "SquaredExponential",
    "Matern1",
    "Matern3",
    "Matern5",
    "RationalQuadraticARD",
]:
    wk = wg = 0.0
    for seed in range(200):
        r = np.random.default_rng(seed)
        D = int(r.integers(1, 7))
        per = np.where(
            r.random(D) < 0.6, r.choice([2.0, 2 * np.pi, 1.5, 7.3], D), np.inf
        )
        if np.all(np.isinf(per)):
            per[0] = 2.0
        N = int(r.integers(2, 60))
        lo = -per / 2
        lo[np.isinf(lo)] = -1.0
        width = np.where(np.isinf(per), 2.0, per)
        X = lo + width * r.random((N, D))
        # some close pairs, and some exact duplicates
        k = N // 4
        X[:k] = (
            X[N - k :]
            + r.normal(scale=10.0 ** r.uniform(-12, -2), size=(k, D)) * width
        )
        Xs = lo + width * r.random((30, D))
        mk = lambda m: (
            getattr(m, name[:-1] if name.startswith("Matern") else name)(
                int(name[-1]), per
            )
            if name.startswith("Matern")
            else getattr(m, name)(per)
        )
        ko, kn = mk(old), mk(new)
        hyp = r.normal(scale=0.7, size=ko.hyperparameter_count(D))
        sf2 = np.exp(2 * hyp[D])
        Ko, dKo = ko.compute(hyp, X, compute_grad=True)
        Kn, dKn = kn.compute(hyp, X, compute_grad=True)
        Co, Cn = ko.compute(hyp, X, Xs), kn.compute(hyp, X, Xs)
        wk = max(
            wk, np.max(np.abs(Kn - Ko)) / sf2, np.max(np.abs(Cn - Co)) / sf2
        )
        wg = max(wg, np.max(np.abs(dKn - dKo)) / np.max(np.abs(dKo)))
        assert np.array_equal(Kn, Kn.T)
    worst[name] = (wk, wg)
    print(f"  {name:22s} K {wk:.2e}  dK {wg:.2e}")
