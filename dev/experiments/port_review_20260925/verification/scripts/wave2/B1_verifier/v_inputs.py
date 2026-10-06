"""B1 verifier, construction-only checks: fun_values (K2, I-F6, C-F9),
multi-row x0 and x0=None (I-F7, C-F12), the non_box_cons checks (I-F10,
C-F12)."""
import common  # noqa: F401
import numpy as np

from pybads import BADS

f = lambda x: float(np.sum(np.asarray(x) ** 2))
O = {"display": "off", "random_seed": 0}


def try_(label, thunk):
    try:
        r = thunk()
        print(f"  {label}: accepted {r if r is not None else ''}")
        return r
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e).splitlines()[0][:90]}")


print("-- K2 fun_values")
for fv in [
    {"X": np.array([[1.0, 1.0]]), "Y": np.array([[2.0]])},
    {"X": [[1.0, 1.0]], "Y": [[2.0]]},
]:
    try_(
        f"fun_values={fv}",
        lambda: BADS(
            f,
            np.zeros(2),
            -5 * np.ones(2),
            5 * np.ones(2),
            -2 * np.ones(2),
            2 * np.ones(2),
            options={**O, "fun_values": fv},
        )
        and None,
    )
# with the first ambiguous test bypassed: what the next lines do
X = np.array([[1.0, 1.0]])
print(
    "  np.isreal(X) =",
    np.isreal(X),
    "-> 'not np.isreal(X)' on a (1,2) "
    "array raises; range(len()) raises TypeError:",
    end=" ",
)
try:
    range(len())
except TypeError as e:
    print(e)

print("\n-- multi-row x0")
b = try_(
    "x0 (2,2) with bounds",
    lambda: BADS(
        f,
        np.array([[0.1, 0.2], [0.3, -0.4]]),
        -np.ones(2),
        np.ones(2),
        options=O,
    ),
)
if b is not None:
    print(
        "    D =",
        b.D,
        " u0 shape",
        b.optim_state["u"].shape,
        " self.u shape",
        b.u.shape,
    )
b = try_(
    "x0 (3,2) without bounds (plb estimated)",
    lambda: BADS(
        f, np.array([[0.1, 0.2], [0.3, -0.4], [0.0, 0.5]]), options=O
    ),
)
if b is not None:
    print(
        "    plb_orig",
        b.var_transf.orig_plb,
        "pub_orig",
        b.var_transf.orig_pub,
    )

print("\n-- x0=None")
b = try_(
    "x0=None, lb/ub only (MATLAB: error)",
    lambda: BADS(
        f, None, -5 * np.ones((1, 2)), 5 * np.ones((1, 2)), options=O
    ),
)
if b is not None:
    print("    plb_orig", b.var_transf.orig_plb, " x0", b.x0)
try_(
    "x0=None, list plb/pub",
    lambda: BADS(f, None, [-5, -5], [5, 5], [-2, -2], [2, 2], options=O)
    and None,
)
try_(
    "x0=None, 1-D array plb/pub",
    lambda: BADS(
        f,
        None,
        np.array([-5, -5.0]),
        np.array([5, 5.0]),
        np.array([-2, -2.0]),
        np.array([2, 2.0]),
        options=O,
    )
    and None,
)

print("\n-- non_box_cons output checks (MATLAB: N x 1 required)")
for label, nbc in [
    ("returns (N,)", lambda x: np.sum(np.atleast_2d(x) ** 2, 1) > 100),
    (
        "returns (N,1)",
        lambda x: (np.sum(np.atleast_2d(x) ** 2, 1) > 100)[:, None],
    ),
    ("returns scalar", lambda x: float(np.sum(np.atleast_2d(x) ** 2) > 100)),
    ("returns bool", lambda x: bool(np.sum(np.atleast_2d(x) ** 2) > 100)),
    ("returns (N,2)", lambda x: np.zeros((np.atleast_2d(x).shape[0], 2))),
    (
        "returns (1,N)",
        lambda x: (np.sum(np.atleast_2d(x) ** 2, 1) > 100)[None, :],
    ),
]:
    try_(
        label,
        lambda: BADS(
            f,
            np.zeros(2) + 0.3,
            -5 * np.ones(2),
            5 * np.ones(2),
            -2 * np.ones(2),
            2 * np.ones(2),
            non_box_cons=nbc,
            options=O,
        )
        and None,
    )

print("\n-- random x0 against non_box_cons: sum(x^2) > 1 on [-1,1]^2")
nbc = lambda x: np.sum(np.atleast_2d(x) ** 2, 1) > 1
refused = []
for s in range(20):
    try:
        BADS(
            f,
            None,
            -np.ones((1, 2)) * 2,
            np.ones((1, 2)) * 2,
            -np.ones((1, 2)),
            np.ones((1, 2)),
            non_box_cons=nbc,
            options={**O, "random_seed": s},
        )
    except ValueError:
        refused.append(s)
print(
    f"  refused for {len(refused)} of 20 seeds: {refused}; "
    f"expected share 1 - pi/4 = {1 - np.pi / 4:.3f}"
)
