"""W4-14: a noisy run that ends within its first iteration takes its
reserved final samples at the incumbent, whatever ends it (the budget,
max_iter, the mesh, output_fcn at "iter"), at levels 1 and 2, and reports
them as the final estimate does after a later iteration. Also the cases
with no sample reserved."""
import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
from pybads import BADS

D = 2


def make(level, seed=0, pts=None):
    rng = np.random.default_rng(seed)

    def f(x):
        x = np.ravel(x)
        if pts is not None:
            pts.append(x.copy())
        y = float(np.sum((x - 0.1) ** 2)) + 0.5 * rng.normal()
        return (y, 0.5) if level == 2 else y

    return f


def run(name, level, **opts):
    pts = []
    o = {"display": "off", "random_seed": 0, "uncertainty_handling": True}
    if level == 2:
        o["specify_target_noise"] = True
    o.update(opts)
    b = BADS(
        make(level, 0, pts),
        0.3 * np.ones(D),
        -5 * np.ones(D),
        5 * np.ones(D),
        -2 * np.ones(D),
        2 * np.ones(D),
        options=o,
    )
    r = b.optimize()
    nfs = b.options["noise_final_samples"]
    y = r["yval_vec"]
    line = (
        f"{name:34s} L{level} it={r['iterations']} fc={r['func_count']} pts={len(pts)} "
        f"status={r['status']} nfs={nfs} yval_vec={None if y is None else y.shape} "
    )
    if y is not None and nfs > 0 and r["iterations"] > 0:
        at_x = all(np.array_equal(p, np.ravel(r["x"])) for p in pts[-nfs:])
        if level == 2:
            prec = 1 / r["ysd_vec"] ** 2
            fv, fs = np.sum(y * prec) / np.sum(prec), 1 / np.sqrt(np.sum(prec))
        else:
            fv, fs = np.mean(y), np.std(y, ddof=1) / np.sqrt(y.size)
        h = b.iteration_history
        line += (
            f"samples at x={at_x} fval ok={np.isclose(r['fval'], fv, rtol=1e-12)} "
            f"fsd ok={np.isclose(r['fsd'], fs, rtol=1e-12)} hist[0]==result="
            f"{h.get('fval')[0] == r['fval'] and h.get('fsd')[0] == r['fsd']}"
        )
    else:
        line += f"fsd={r['fsd']!r} noise_size={b.options['noise_size']!r}"
    print(line, flush=True)
    return b, r


import sys

levels = (1, 2) if len(sys.argv) < 2 else (int(sys.argv[1]),)
for level in levels:
    if level == 2 or len(sys.argv) < 2:
        pass
    if not (len(sys.argv) > 1 and sys.argv[1] == "1b"):
        pass
for level in levels:
    if level == 1 and len(sys.argv) > 2:
        run(
            "output_fcn stops at init",
            level,
            output_fcn=lambda x, s, st: st == "init",
        )
        continue
    run("budget 38", level, max_fun_evals=38)
    run("max_iter 1", level, max_iter=1)
    run("mesh (tol_mesh=2)", level, tol_mesh=2)
    run(
        "output_fcn stops at first iter",
        level,
        output_fcn=lambda x, s, st: st == "iter",
    )
    run("budget 33 (none reserved)", level, max_fun_evals=33)
    run("budget 34 (one reserved)", level, max_fun_evals=34)
    run(
        "noise_final_samples 0, max_iter 1",
        level,
        noise_final_samples=0,
        max_iter=1,
    )
    run(
        "output_fcn stops at init",
        level,
        output_fcn=lambda x, s, st: st == "init",
    )
# noise found by the test
b, r = run(
    "noise test finds noise, budget 38",
    1,
    max_fun_evals=38,
    uncertainty_handling=None,
)
b, r = run(
    "noise test finds noise, budget 34",
    1,
    max_fun_evals=34,
    uncertainty_handling=None,
)
