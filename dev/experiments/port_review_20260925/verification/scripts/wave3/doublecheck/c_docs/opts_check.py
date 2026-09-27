"""Run PyBADS (the one on PYTHONPATH) with the option values that the
changelog's wave-3 entries describe, and report what happens."""
import logging
import sys
import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)


def sphere(x):
    x = np.atleast_2d(x)
    return float(np.sum(x**2))


def run(label, options, fun=sphere, x0=None, D=2, **kw):
    if x0 is None:
        x0 = np.full(D, 2.0)
    base = dict(display="off", random_seed=0, max_fun_evals=100)
    base.update(options)
    try:
        b = BADS(
            fun,
            x0,
            lower_bounds=np.full(D, -5.0),
            upper_bounds=np.full(D, 5.0),
            plausible_lower_bounds=np.full(D, -4.0),
            plausible_upper_bounds=np.full(D, 4.0),
            options=base,
            **kw,
        )
    except Exception as e:
        print(f"{label}: at BADS(): {type(e).__name__}: {e}", flush=True)
        return None
    try:
        r = b.optimize()
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        where = [f"{f.name}:{f.lineno}" for f in tb if "pybads" in f.filename][
            -3:
        ]
        it = b.optim_state.get("iter")
        print(
            f"{label}: at optimize(): {type(e).__name__}: {str(e)[:120]} "
            f"(iter {it}, func_count {b.function_logger.func_count}; {where})",
            flush=True,
        )
        return None
    print(
        f"{label}: ran; fval {r['fval']:.3g}, x {np.round(r['x'], 4)}, "
        f"func_count {r['func_count']}, iterations {r['iterations']}",
        flush=True,
    )
    return r


which = sys.argv[1] if len(sys.argv) > 1 else "all"

if which in ("all", "acc"):
    for v in [0, -1, 3.0, 2.5, np.inf, True, 1000.0, 1]:
        run(f"accelerate_mesh_steps={v!r}", {"accelerate_mesh_steps": v})

if which in ("all", "iq"):
    for v in [0, 1, -0.5, 1.5, 0.3]:
        run(
            f"improvement_quantile={v!r} (level 0)",
            {"improvement_quantile": v},
        )

    def noisy(x, rng=np.random.default_rng(1)):
        return sphere(x) + rng.normal()

    for v in [0, 1, 1e-12, 1 - 1e-12]:
        # a fresh noise stream per run
        rng = np.random.default_rng(1)
        run(
            f"improvement_quantile={v!r} (noisy)",
            {
                "improvement_quantile": v,
                "uncertainty_handling": True,
                "noise_size": 1.0,
                "max_fun_evals": 150,
            },
            fun=lambda x, rng=rng: sphere(x) + rng.normal(),
        )

if which in ("all", "beta"):
    vals = [
        2.0,
        2,
        np.float64(2.0),
        np.float64(0.0),
        np.float64(-1.0),
        np.bool_(True),
        np.complex128(1.0 + 0j),
        np.array([0.0]),
        np.array([2.0]),
        np.inf,
        True,
        "x",
    ]
    for v in vals:
        run(f"sqrt_beta={v!r}", {"search_acq_fcn": ("acq_LCB", v)})

if which in ("all", "misc"):
    run("hedge_gamma=0", {"hedge_gamma": 0})
    run("uncertain_incumbent=False", {"uncertain_incumbent": False})
    before = np.geterr()
    run("default (geterr check)", {})
    print("np.geterr before", before, "after", np.geterr(), flush=True)
