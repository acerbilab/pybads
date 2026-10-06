import numpy as np

import pybads
from pybads.variable_transformer import VariableTransformer

print(pybads.__file__)


def errs(lb, ub, plb, pub):
    lb, ub, plb, pub = (
        np.atleast_2d(np.asarray(v, dtype=float)) for v in (lb, ub, plb, pub)
    )
    D = lb.shape[1]
    try:
        VariableTransformer(D, lb, ub, plb, pub, np.full((1, D), np.nan))
        ok = True
    except ValueError as e:
        ok = str(e)[:40]
    # errors of the round trips, with the transform built without the test
    return ok


cases = {
    "reviewer": (-9.53e10, 9.53e10, -2.06, -0.74),
    "log 1e9": (1e-3, 1e9, 1.0, 10.0),
    "log 1.3e9": (1e-3, 1.3e9, 1.0, 10.0),
    "log 2e9": (1e-3, 2e9, 1.0, 10.0),
    "log 5e8": (1e-3, 5e8, 1.0, 10.0),
}
for k, v in cases.items():
    print(k, errs(*v))
# the round-trip errors, relative
for k, (lb, ub, plb, pub) in cases.items():
    log = lb > 0 and pub / plb >= 10
    f = np.log if log else (lambda x: x)
    finv = np.exp if log else (lambda x: x)
    mu = 0.5 * (f(plb) + f(pub))
    gam = 0.5 * (f(pub) - f(plb))
    for name, b in [("lb", lb), ("ub", ub), ("plb", plb), ("pub", pub)]:
        r = finv(gam * ((f(b) - mu) / gam) + mu)
        print(
            f"  {k} {name}: |err| {abs(r - b):.3g}, rel {abs(r - b) / max(1, abs(b)):.3g}"
        )
