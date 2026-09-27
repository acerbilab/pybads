"""Runs of 150 evaluations with values of hedge_beta and hedge_decay that
BADS does not check: the hedge's probabilities, gains and choices."""
import numpy as np

import pybads
from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge

records = []
original = ESSearchHedge.__call__


def call(self, *args, **kwargs):
    out = original(self, *args, **kwargs)
    records.append(
        (self.prob.copy(), self.g.copy(), int(self.chosen_hedge.item()))
    )
    return out


ESSearchHedge.__call__ = call


def f(x):
    x = np.atleast_2d(x)
    return float(np.sum((x - 0.3) ** 2) + np.sum(np.cos(3 * x)))


for name, value in [
    ("default", None),
    ("hedge_beta", 0.0),
    ("hedge_beta", -1.0),
    ("hedge_beta", -1e3),
    ("hedge_beta", np.inf),
    ("hedge_beta", np.nan),
    ("hedge_decay", 1.0),
    ("hedge_decay", 2.0),
    ("hedge_decay", 50.0),
    ("hedge_decay", -1.0),
    ("hedge_decay", np.nan),
]:
    records.clear()
    o = {"display": "off", "random_seed": 0, "max_fun_evals": 150}
    if value is not None:
        o[name] = value
    try:
        r = BADS(
            f,
            np.zeros(3),
            -5 * np.ones(3),
            5 * np.ones(3),
            -3 * np.ones(3),
            3 * np.ones(3),
            options=o,
        ).optimize()
        end = f"fval {r['fval']:.4g}, func_count {r['func_count']}"
    except Exception as e:  # noqa: BLE001
        end = f"{type(e).__name__}: {e}"
    probs = np.array([p for p, _, _ in records])
    gains = np.array([g for _, g, _ in records])
    chosen = np.array([c for _, _, c in records])
    nan_p = int(np.sum(np.any(~np.isfinite(probs), axis=1)))
    print(
        f"{name}={value}: {len(records)} searches, non-finite probabilities "
        f"in {nan_p}, choices of ES-wcm {int(np.sum(chosen == 0))}, "
        f"max |g| {np.nanmax(np.abs(gains)) if gains.size else None:.3g}, "
        f"non-finite g in {int(np.sum(np.any(~np.isfinite(gains), axis=1)))}; "
        f"{end}",
        flush=True,
    )
print(pybads.__file__)
