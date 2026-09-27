"""What the checks of W4-18 (hedge_gamma), W4-19 (sqrt_beta), W4-25
(n_search_iter) and W4-29 (hedge_beta, hedge_decay) refuse and accept when
BADS is created, and the type of the stored value."""
import warnings

import hdr  # noqa
import numpy as np

from pybads import BADS

warnings.simplefilter("ignore")


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


x0 = np.array([[0.5, -0.5]])
lb, ub = -5 * np.ones((1, 2)), 5 * np.ones((1, 2))
plb, pub = -2 * np.ones((1, 2)), 2 * np.ones((1, 2))


def make(opts):
    o = {"display": "off"}
    o.update(opts)
    return BADS(f, x0, lb, ub, plb, pub, options=o)


common = [
    ("True", True),
    ("False", False),
    ("np.True_", np.True_),
    ("None", None),
    ("py complex 0.1+0j", 0.1 + 0j),
    ("np.complex128(0.1)", np.complex128(0.1)),
    ("np.complex128(0.1+0.1j)", np.complex128(0.1 + 0.1j)),
    ("np.complex128(0.1-5j)", np.complex128(0.1 - 5j)),
    ("np.int64(0)", np.int64(0)),
    ("np.int64(1)", np.int64(1)),
    ("np.float64(0.1)", np.float64(0.1)),
    ("np.float32(0.1)", np.float32(0.1)),
    ("0.0", 0.0),
    ("1.0", 1.0),
    ("2.0", 2.0),
    ("0", 0),
    ("1", 1),
    ("2", 2),
    ("-0.0", -0.0),
    ("np.array(0.1) 0-d", np.array(0.1)),
    ("np.array([0.1])", np.array([0.1])),
    ("np.array([[0.1]])", np.array([[0.1]])),
    ("[0.1] list", [0.1]),
    ("np.array([0.1, 0.2])", np.array([0.1, 0.2])),
    ("inf", np.inf),
    ("-inf", -np.inf),
    ("nan", np.nan),
    ("1e308", 1e308),
    ("-1e-12", -1e-12),
    ("0.5", 0.5),
    ("0.5+1e-16", 0.5 + 1e-16),
    ("np.nextafter(0.5,1)", np.nextafter(0.5, 1)),
    ("1+1e-12", 1 + 1e-12),
    ("np.nextafter(1,2)", np.nextafter(1.0, 2.0)),
    ("1/3", 1 / 3),
    ("np.bool_ arr", np.array([True])),
    ("np.uint8(3)", np.uint8(3)),
    ("Fraction(1,4)", __import__("fractions").Fraction(1, 4)),
    ("Decimal('0.25')", __import__("decimal").Decimal("0.25")),
    ("2**70", 2**70),
    ("4096", 4096),
    ("4097", 4097),
]
strings = {
    "hedge_gamma": "0.1",
    "hedge_beta": "1",
    "hedge_decay": "0.5",
    "n_search_iter": "2",
}

for name in ["hedge_gamma", "hedge_beta", "hedge_decay", "n_search_iter"]:
    print(f"\n== {name}")
    for label, v in common + [(f"str {strings[name]!r}", strings[name])]:
        try:
            b = make({name: v})
            s = b.options[name]
            print(f"  {label:26s} ACCEPT stored {type(s).__name__} {s!r}")
        except ValueError as e:
            print(f"  {label:26s} refuse ValueError")
        except Exception as e:
            print(f"  {label:26s} refuse {type(e).__name__}: {str(e)[:70]}")

print("\n== search_acq_fcn[1] (sqrt_beta)")
extra = [
    ("callable", lambda t, d: 2.0),
    ("str 'acq'", "acq"),
    ("str '2'", "2"),
    ("-1", -1),
    ("1e-300", 1e-300),
]
for label, v in common + extra:
    try:
        b = make({"search_acq_fcn": ("acq_LCB", v)})
        s = b.options["search_acq_fcn"][1]
        print(f"  {label:26s} ACCEPT stored {type(s).__name__} {s!r}")
    except ValueError as e:
        print(f"  {label:26s} refuse ValueError")
    except Exception as e:
        print(f"  {label:26s} refuse {type(e).__name__}: {str(e)[:70]}")

print("\n== hedge_gamma with search_method of 1 and 3")
for sm in ([("ES-wcm", 1)], [("ES-wcm", 1), ("ES-ell", 1), ("ES-wcm", 1)]):
    for v in [1 / len(sm), np.nextafter(1 / len(sm), 1), 0.5]:
        try:
            make({"search_method": sm, "hedge_gamma": v})
            print(f"  n={len(sm)} gamma={v!r}: ACCEPT")
        except ValueError:
            print(f"  n={len(sm)} gamma={v!r}: refuse")
print("\n== tol_fun and the default hedge_beta")
for tf in [1e-3, -1e-3, np.float64(0.0), 1e-320]:
    try:
        b = make({"tol_fun": tf})
        print(
            f"  tol_fun={tf!r}: ACCEPT hedge_beta={b.options['hedge_beta']!r}"
        )
    except Exception as e:
        print(f"  tol_fun={tf!r}: {type(e).__name__}: {str(e)[:100]}")
