import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
f = lambda x: float(np.sum(np.ravel(x) ** 2))


def mk(seed, x0=None):
    return BADS(
        f,
        x0=x0,
        lower_bounds=-5 * np.ones(2),
        upper_bounds=5 * np.ones(2),
        plausible_lower_bounds=-2 * np.ones(2),
        plausible_upper_bounds=2 * np.ones(2),
        options={"random_seed": seed, "display": "off"},
    )


g = np.random.default_rng(7)
ss = np.random.SeedSequence(7)
bg = np.random.PCG64(7)
cases = [
    ("int 3", 3),
    ("np.int64 3", np.int64(3)),
    ("float 3.0", 3.0),
    ("np.float32 3.0", np.float32(3.0)),
    ("float 3.5", 3.5),
    ("str '3'", "3"),
    ("negative -1", -1),
    ("bool True", True),
    ("list [1,2]", [1, 2]),
    ("SeedSequence", ss),
    ("Generator", g),
    ("BitGenerator PCG64", bg),
    ("None", None),
    ("float nan", float("nan")),
    ("float inf", float("inf")),
    ("big int 2**70", 2**70),
]
for label, s in cases:
    try:
        b = mk(s)
        extra = ""
        if isinstance(s, np.random.Generator):
            extra = f" same object={b.rng is s}"
        print(
            f"{label:22s}: OK rng={type(b.rng).__name__} optim_state random_seed={b.optim_state['random_seed']!r}{extra} x0={np.round(b.x0.ravel(),4)}"
        )
    except Exception as e:
        print(f"{label:22s}: {type(e).__name__}: {str(e)[:90]}")
# same seed -> same random x0; int vs float equal
print("int 3 vs float 3.0 same x0:", np.array_equal(mk(3).x0, mk(3.0).x0))
# None: global state
np.random.seed(11)
a = mk(None).x0
np.random.seed(11)
b2 = mk(None).x0
print("None with np.random.seed fixed -> same x0:", np.array_equal(a, b2))
np.random.seed(11)
st0 = np.random.get_state()[2]
mk(None)
st1 = np.random.get_state()[2]
np.random.seed(11)
np.random.randint(0, 2**32, size=4, dtype=np.uint32)
st4 = np.random.get_state()[2]
print(
    "global state pos after BADS(None):",
    st1,
    " after 4 uint32 draws:",
    st4,
    " start:",
    st0,
)
np.random.seed(11)
st_a = np.random.get_state()[1].copy()
mk(5)
st_b = np.random.get_state()[1]
print("seeded BADS leaves global state untouched:", np.array_equal(st_a, st_b))
# result field
b = mk(3.0, x0=np.zeros(2))
b.options["max_fun_evals"] = 30
r = b.optimize()
print("result random_seed for 3.0:", repr(r["random_seed"]))
b = mk(g, x0=np.zeros(2))
b.options["max_fun_evals"] = 30
r = b.optimize()
print("result random_seed for Generator:", repr(r["random_seed"]))
# changing option after creation
b = mk(1, x0=None)
x1 = b.x0.copy()
b.options["random_seed"] = 2
print(
    "option changed after creation has no effect on rng (by design):",
    b._random_seed,
)
