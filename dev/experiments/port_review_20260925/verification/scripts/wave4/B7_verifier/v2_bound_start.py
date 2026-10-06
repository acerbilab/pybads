"""B7 verifier, K3: a start on a lower hard bound with the plausible bounds
omitted. Which u0 the run starts from, on the grid, what init_sobol's seed
cast gives for it here, and whether a default run reaches it.
Run with the review worktree (full check) or with v1.1.0 (argument 'old':
the start only)."""
import sys
import warnings

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
OLD = len(sys.argv) > 1 and sys.argv[1] == "old"


def target(x):
    x = np.atleast_2d(x)
    return float(np.sum((x - 0.3) ** 2))


cases = {
    "linear, x0 = lb, plb/pub omitted": dict(
        x0=np.array([[-3.0, 0.5]]),
        lb=np.array([[-3.0, -3.0]]),
        ub=np.array([[3.0, 3.0]]),
        plb=None,
        pub=None,
    ),
    "linear, x0 = lb = plb given": dict(
        x0=np.array([[-3.0, 0.5]]),
        lb=np.array([[-3.0, -3.0]]),
        ub=np.array([[3.0, 3.0]]),
        plb=np.array([[-3.0, -3.0]]),
        pub=np.array([[3.0, 3.0]]),
    ),
    "log, x0 = lb, plb/pub omitted": dict(
        x0=np.array([[0.1, 1.0]]),
        lb=np.array([[0.1, 0.1]]),
        ub=np.array([[10.0, 10.0]]),
        plb=None,
        pub=None,
    ),
    "log, x0 = lb = 1e-3, [1e-3, 1e3], D=3": dict(
        x0=np.array([[1e-3, 1.0, 2.0]]),
        lb=np.array([[1e-3] * 3]),
        ub=np.array([[1e3] * 3]),
        plb=None,
        pub=None,
    ),
    "linear, x0 between lb and plb": dict(
        x0=np.array([[-2.5, 0.5]]),
        lb=np.array([[-3.0, -3.0]]),
        ub=np.array([[3.0, 3.0]]),
        plb=np.array([[-2.0, -2.0]]),
        pub=np.array([[2.0, 2.0]]),
    ),
    "interior start": dict(
        x0=np.array([[-2.9, 0.5]]),
        lb=np.array([[-3.0, -3.0]]),
        ub=np.array([[3.0, 3.0]]),
        plb=None,
        pub=None,
    ),
}

for name, c in cases.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            target,
            c["x0"],
            c["lb"],
            c["ub"],
            c["plb"],
            c["pub"],
            options={"display": "off", "random_seed": 0, "max_fun_evals": 200},
        )
    print(
        f"{name}: x0 {b.x0.ravel().tolist()}, plb (orig) "
        f"{b.optim_state['plb_orig'].ravel().tolist()}, lb (u) "
        f"{b.lower_bounds.ravel().tolist()}, u0 on the grid "
        f"{b.u.tolist()}, u0[0] == -1 exactly: {b.u[0] == -1.0}",
        flush=True,
    )

if OLD:
    sys.exit(0)

import pybads.bads.bads as bads_module  # noqa: E402

orig_init_sobol = bads_module.init_sobol
seen = []


def spy(u0, *args, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cast = u0[: min(11, len(u0))].astype(np.uint64)
    s = np.array2string(cast)[1:-1]
    seed = int(np.mod(np.prod(np.array([ord(ch) for ch in s])), 997) + 1)
    out = orig_init_sobol(u0, *args, **kwargs)
    seen.append(
        dict(
            u0=u0.copy(),
            cast=cast,
            seed=seed,
            design=out[0],
            warnings=[str(x.message) for x in w],
        )
    )
    return out


bads_module.init_sobol = spy
for name in (
    "linear, x0 = lb, plb/pub omitted",
    "interior start",
    "log, x0 = lb, plb/pub omitted",
):
    c = cases[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            target,
            c["x0"],
            c["lb"],
            c["ub"],
            c["plb"],
            c["pub"],
            options={"display": "off", "random_seed": 0, "max_fun_evals": 200},
        )
        r = b.optimize()
    s = seen[-1]
    print(
        f"\n{name}: default run reached init_sobol with u0 "
        f"{s['u0'].tolist()}; cast {s['cast'].tolist()} (warnings "
        f"{s['warnings']}); seed {s['seed']}; design rows "
        f"{s['design'].shape[0]}; run: {r['func_count']} evaluations, "
        f"fval {r['fval']:.3g}",
        flush=True,
    )
print(
    "\nlinear on-bound design equals interior design:",
    np.array_equal(seen[0]["design"], seen[1]["design"]),
    flush=True,
)
# The seed that a saturating cast (negative -> 0) would give for the
# on-bound start: that of '0 0'
print(
    "seed of '0 0' (what a saturating cast gives):",
    int(np.mod(np.prod(np.array([ord(ch) for ch in "0 0"])), 997) + 1),
    flush=True,
)
