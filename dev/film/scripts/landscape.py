"""The film's landscape: a long diagonal valley from the start down to A, and a narrow trench that leaves A
along +x1 and deepens to the minimum B. The valley is easy for a surrogate and slow for coordinate steps; the
trench is too narrow for a smooth surrogate and lies along a parameter axis."""
import math

import numpy as np

LB, UB = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
PLB, PUB = np.array([-4.0, -4.0]), np.array([4.0, 4.0])
X0 = np.array([-4.0, -4.0])
D = np.array([1.0, 1.0]) / math.sqrt(2.0)  # along the valley
N = np.array([1.0, -1.0]) / math.sqrt(2.0)  # across it


def make(a, b, A, tstar, w=0.2):
    """a, b: curvature along and across the valley; A: its bottom; tstar: distance from A to the minimum along
    +x1; w: the trench's width (SD of its cross-section)."""
    A = np.asarray(A, dtype=float)
    S = (
        a + b
    ) * tstar  # the trench's slope, which puts the minimum at A + tstar e1

    def f(x):
        r = np.asarray(x, dtype=float) - A
        return float(
            a * (D @ r) ** 2
            + b * (N @ r) ** 2
            - S * max(0.0, r[0]) * math.exp(-0.5 * (r[1] / w) ** 2)
        )

    xmin = A + np.array([tstar, 0.0])
    return (
        f,
        xmin,
        f(xmin),
        dict(a=a, b=b, A=A.tolist(), tstar=tstar, w=w, S=S),
    )


def mads(
    f,
    x0,
    step0=2.0,
    step_max=4.0,
    tol=4.0 * 2**-7,
    order=((1, 0), (-1, 0), (0, 1), (0, -1)),
    budget=600,
):
    """Plain coordinate direct search: opportunistic poll in a fixed order, the step doubled (up to step_max)
    after a success and halved after a failure, until it is below tol."""
    x, fx, step = np.array(x0, float), f(x0), step0
    seen = {tuple(np.round(x, 9)): fx}
    log = [(x.copy(), fx, "start", step)]
    while step >= tol and len(log) < budget:
        moved = False
        for d in order:
            y = x + step * np.array(d, dtype=float)
            if np.any(y < LB) or np.any(y > UB):
                continue
            key = tuple(np.round(y, 9))
            if key in seen:
                fy = seen[key]
            else:
                fy = f(y)
                seen[key] = fy
                log.append((y.copy(), fy, "poll", step))
            if fy < fx - 1e-12:
                x, fx, moved = y, fy, True
                break
        step = min(2 * step, step_max) if moved else step / 2
    return log


def first_within(vals, fmin, margin):
    best = np.minimum.accumulate(np.asarray(vals))
    hit = np.nonzero(best <= fmin + margin)[0]
    return int(hit[0]) + 1 if len(hit) else None
