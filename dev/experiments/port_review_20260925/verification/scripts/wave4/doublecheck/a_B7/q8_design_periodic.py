"""init_sobol's size at each level and D, and the cut in _init_mesh_; the
refusal of periodic_vars before the transform, and an empty one as None."""
import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
import pybads.bads.bads as bads_mod
from pybads import BADS

seen = []
orig = bads_mod.init_sobol


def hook(u0, lb, ub, plb, pub, fes, rng=None):
    out = orig(u0, lb, ub, plb, pub, fes, rng=rng)
    seen.append((fes, out[1], out[0].shape[0]))
    return out


bads_mod.init_sobol = hook


def f(x):
    return float(np.sum(np.ravel(x) ** 2))


def fn(seed):
    rng = np.random.default_rng(seed)
    return lambda x: float(np.sum(np.ravel(x) ** 2)) + rng.normal()


def expect(fes, D):
    m = int(np.ceil(np.log2(fes)))
    return 2 ** (m + 1) if 2**m == D else 2**m


for D in (1, 2, 3, 4, 8, 16):
    for level, opts, fun in (
        (0, {}, f),
        (1, {"uncertainty_handling": True}, fn(0)),
    ):
        for mfe in (5, 40):
            seen.clear()
            o = {
                "display": "off",
                "random_seed": 0,
                "max_fun_evals": mfe,
                "max_iter": 1,
            }
            o.update(opts)
            b = BADS(
                fun,
                0.5 * np.ones(D),
                -5 * np.ones(D),
                5 * np.ones(D),
                -2 * np.ones(D),
                2 * np.ones(D),
                options=o,
            )
            b.optimize()
            ((fes, n, rows),) = seen
            evaluated = b.optim_state["eff_starting_points"] - 1
            print(
                f"D={D:2d} L{level} mfe={mfe:3d}: asked {fes} -> {rows} points (n_samples {n}, formula {expect(fes, D)}), evaluated {evaluated}"
            )
print("periodic_vars:")
for pv in ([1], [5], [], np.array([], dtype=int), (), None, 0, False):
    for x0 in (np.array([0.5, 0.0]), None):
        try:
            b = BADS(
                f,
                x0,
                -5 * np.ones(2),
                5 * np.ones(2),
                -3 * np.ones(2),
                3 * np.ones(2),
                options={
                    "display": "off",
                    "random_seed": 1,
                    "periodic_vars": pv,
                },
            )
            out = f"accepted, option now {b.options['periodic_vars']!r}, mask {b.optim_state['periodic_vars']}"
        except Exception as e:
            out = f"{type(e).__name__}: {str(e)[:60]}"
        print(
            f"   {pv!r:26} x0={'given' if x0 is not None else 'random'}: {out}"
        )
