"""PI's question on W4-1: the run's generator advances by one draw at the
design whatever scipy draws; nothing draws from NumPy's global stream; two
seeds give two designs and one seed one design whatever the start; W4-2's
cast gone."""
import copy

import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
import pybads.bads.bads as bads_mod
from pybads import BADS
from pybads.init_functions import init_sobol

# (a) init_sobol alone: one draw of rng, whatever D and size
bad = 0
for D in (1, 2, 3, 5, 8, 12, 20, 40):
    for fes in (1, 2, 3, D, 20, 33):
        for s in range(3):
            rng = np.random.default_rng(1000 * D + s)
            ref = copy.deepcopy(rng)
            init_sobol(
                np.zeros(D), None, None, -np.ones(D), np.ones(D), fes, rng=rng
            )
            ref.integers(2**63)
            if rng.bit_generator.state != ref.bit_generator.state:
                bad += 1
print(
    "(a) init_sobol advances rng by one integers(2**63) draw: mismatches", bad
)

# (b) through BADS: hook init_sobol, check one draw and the global state
seen = []
orig = bads_mod.init_sobol


def hook(u0, lb, ub, plb, pub, fes, rng=None):
    before = copy.deepcopy(rng)
    out = orig(u0, lb, ub, plb, pub, fes, rng=rng)
    before.integers(2**63)
    seen.append(
        dict(
            one_draw=before.bit_generator.state == rng.bit_generator.state,
            design=out[0].copy(),
            n=out[1],
            rng_is_bads=None,
        )
    )
    return out


bads_mod.init_sobol = hook
D = 3


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


def design_of(seed, x0, **opts):
    seen.clear()
    np.random.seed(7)
    g0 = np.random.get_state()
    o = {"display": "off", "max_fun_evals": 12, "random_seed": seed}
    o.update(opts)
    b = BADS(
        sphere,
        x0,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=o,
    )
    b.optimize()
    g1 = np.random.get_state()
    same_global = (
        g0[0] == g1[0] and np.array_equal(g0[1], g1[1]) and g0[2:] == g1[2:]
    )
    (s,) = seen
    return s["design"], s["one_draw"], same_global


starts = {
    "x0=4": np.ones(D) * 4,
    "(-5,0,10)": np.array([-5.0, 0.0, 10.0]),
    "on plb": -8 * np.ones(D),
    "below plb (-20)": -20 * np.ones(D),
    "on lb-ish (-99)": -99 * np.ones(D),
    "above pub (50)": 50 * np.ones(D),
    "random (None)": None,
}
for seed in (42, 43):
    designs = {}
    for name, x0 in starts.items():
        d, one, g = design_of(seed, None if x0 is None else x0.copy())
        designs[name] = d
        print(
            f"(b) seed {seed} start {name:16s}: one draw={one} global untouched={g} first row={np.round(d[0], 6)}"
        )
    ref = designs["x0=4"]
    print(
        f"    seed {seed}: given starts equal to x0=4:",
        {k: bool(np.array_equal(v, ref)) for k, v in designs.items()},
    )
d42, _, _ = design_of(42, np.ones(D) * 4)
d43, _, _ = design_of(43, np.ones(D) * 4)
print(
    "(c) seeds 42 and 43 give different designs:", not np.array_equal(d42, d43)
)

# (d) noisy and level-2 paths, seed decides the 32-point design
d1, one1, g1 = design_of(
    5, np.ones(D) * 4, uncertainty_handling=True, max_fun_evals=40
)
d2, one2, g2 = design_of(
    5, np.ones(D) * 4, uncertainty_handling=True, max_fun_evals=40
)
print(
    "(d) level 1 repeat same design:",
    np.array_equal(d1, d2),
    d1.shape,
    one1,
    g1,
)

# (e) the cast
import subprocess

print(
    "(e) grep uint64 in pybads/init_functions:",
    subprocess.run(
        [
            "grep",
            "-rn",
            "uint64",
            "/home/user/pybads-review/pybads/init_functions",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    or "none",
)
