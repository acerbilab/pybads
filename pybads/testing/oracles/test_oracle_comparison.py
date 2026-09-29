"""The comparison of the oracles (``_oracles.compare``) on synthetic arrays:
the per-element rule ``|out - ref| <= rtol * max(|ref|, q25) + atol``, the
NaN and infinite patterns, the shapes and the keys, and the tolerance
classes of the registered oracles."""

import numpy as np
import pytest

from pybads.testing.oracles._oracles import (
    ORACLES,
    TOLERANCES,
    Oracle,
    compare,
)


def _ok(ref, out, tolerance):
    rows = compare({"x": np.asarray(ref)}, {"x": np.asarray(out)}, tolerance)
    assert len(rows) == 1
    return rows[0][3]


def test_identical_arrays_pass_exactly():
    ref = np.array([1.0, -2.5, 0.0, 1e-300])
    assert _ok(ref, ref.copy(), (0.0, 0.0))


def test_exact_comparison_detects_one_ulp():
    ref = np.array([1.0, 2.0])
    out = ref.copy()
    out[1] = np.nextafter(out[1], np.inf)
    assert not _ok(ref, out, (0.0, 0.0))
    assert _ok(ref, out, (1e-15, 0.0))


def test_relative_tolerance_is_per_element():
    """Entries far below the largest are held to their own size: a change
    that a global scale would allow is caught."""
    ref = np.array([1e6, 1.0, 2.0, 3.0, 4.0])
    out = ref.copy()
    out[1] = 1.0 + 1e-3
    assert not _ok(ref, out, (1e-6, 0.0))
    out = ref * (1 + 5e-7)
    assert _ok(ref, out, (1e-6, 0.0))


def test_lower_quartile_floors_the_denominator():
    """An entry near zero, among entries of order one, is held to the
    lower quartile of the magnitudes: its rounding follows the scale of the
    terms it is a difference of."""
    ref = np.array([1.0, 2.0, 3.0, 4.0, 1e-12])
    floor = np.quantile(np.abs(ref), 0.25)
    out = ref.copy()
    out[4] += 0.5e-6 * floor
    assert _ok(ref, out, (1e-6, 0.0))
    out[4] = ref[4] + 2e-6 * floor
    assert not _ok(ref, out, (1e-6, 0.0))


def test_absolute_tolerance_adds():
    ref = np.zeros(3)
    assert _ok(ref, np.full(3, 1e-9), (1e-3, 1e-8))
    assert not _ok(ref, np.full(3, 1e-7), (1e-3, 1e-8))


def test_nonfinite_patterns_must_match():
    ref = np.array([np.inf, -np.inf, np.nan, 1.0])
    assert _ok(ref, ref.copy(), (0.0, 0.0))
    for i, value in ((0, -np.inf), (1, np.inf), (2, 0.0), (3, np.nan)):
        out = ref.copy()
        out[i] = value
        assert not _ok(ref, out, (1.0, 1.0)), i


def test_all_nonfinite_arrays_compare_patterns():
    ref = np.array([np.nan, np.inf])
    assert _ok(ref, ref.copy(), (0.0, 0.0))
    assert not _ok(ref, np.array([np.inf, np.nan]), (0.0, 0.0))


def test_shapes_and_keys_must_match():
    ref = {"a": np.zeros(3), "b": np.ones(2)}
    assert not compare(ref, {"a": np.zeros((3, 1)), "b": np.ones(2)}, (1, 1))[
        0
    ][3]
    rows = compare(ref, {"a": np.zeros(3)}, (1.0, 1.0))
    assert [r[0] for r in rows if not r[3]] == ["b"]
    rows = compare(ref, {**ref, "c": np.zeros(1)}, (1.0, 1.0))
    assert [r[0] for r in rows if not r[3]] == ["c"]


def test_empty_arrays_pass():
    assert _ok(np.empty((0, 3)), np.empty((0, 3)), (0.0, 0.0))


def test_per_key_tolerances():
    orc = Oracle(
        "demo",
        lambda state, seed: {},
        {"default": "gp_free", "*_rows": "exact", "fs2": "gp_var"},
    )
    assert orc.tolerance("fs2") == TOLERANCES["gp_var"]
    assert orc.tolerance("small_worst_rows") == TOLERANCES["exact"]
    assert orc.tolerance("fmu") == TOLERANCES["gp_free"]
    ref = {"fs2": np.array([1.0]), "a_rows": np.array([3.0])}
    rows = compare(
        ref,
        {"fs2": np.array([1.0 + 1e-4]), "a_rows": np.array([3.0])},
        orc.tolerance,
    )
    assert all(r[3] for r in rows)
    rows = compare(
        ref, {"fs2": np.array([1.0]), "a_rows": np.array([4.0])}, orc.tolerance
    )
    assert [r[0] for r in rows if not r[3]] == ["a_rows"]


def test_unknown_tolerance_class_is_refused():
    with pytest.raises(ValueError, match="tolerance class"):
        Oracle("demo", lambda state, seed: {}, {"default": "loose"})


def test_registered_oracles_use_known_classes():
    for orc in ORACLES.values():
        assert set(orc.tol.values()) <= set(TOLERANCES), orc.name
