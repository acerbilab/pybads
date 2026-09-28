"""The initial design of seeded runs, pinned: the points that BADS passes to
the target before its first search, in the original space and in order,
against ``initial_design_pin.json``, for four configurations. The points are
the start (on the mesh; drawn from the run's generator when ``x0`` is
omitted), the noise test's repeat of it when ``uncertainty_handling`` is
left ``None``, and the Sobol design scaled to the plausible box and put on
the mesh, 32 points when the noise test finds noise.

The initial design involves no BLAS work: the scrambling of the Sobol design
is seeded by one draw of the run's generator, and the rest is elementwise
arithmetic, so the points repeat exactly on every platform and every BLAS
setting, and a change to them is a change to the code. A log-transformed
variable is mapped back by ``np.exp`` from an offset and a scale computed by
``np.log``, whose results can differ by an ulp or two between platforms
(NumPy's SIMD loops, the C library). At the bounds used here, one ulp in
each of those logs and in the ``exp`` moves a point by up to 11 ulps, two
ulps by up to 18, so those variables are compared to 32 ulps, the others
exactly; a change to the design moves a point by a step of the mesh, about
1e-3 of its plausible range.

Each run stops at the first call of its output function (``"init"``), which
BADS makes after the initialization and its first GP fit, before the first
search: the run is the one that BADS makes with these options up to that
point, and every call of the target until then belongs to the
initialization.

A change that moves the initial design on purpose (the order or the number
of the generator's draws before the design, the design's size, its scaling,
the mesh) regenerates the fixture in the same commit, from the repository
root, and the commit message says why::

    python -m pybads.testing.bads.test_initial_design_pin --regenerate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from pybads import BADS

FIXTURE = Path(__file__).resolve().with_name("initial_design_pin.json")
MAX_ULP_LOG = 32


def _box(D, lb=-20.0, ub=20.0, plb=-5.0, pub=5.0):
    return tuple(np.full(D, v) for v in (lb, ub, plb, pub))


CONFIGS = {
    # No noise test; at D = 4 the design of 2**ceil(log2(D)) points doubles
    "deterministic_D4": dict(
        x0=np.array([1.5, -2.0, 0.5, 3.0]),
        bounds=_box(4),
        options={"uncertainty_handling": False, "random_seed": 11},
    ),
    # The noise test finds noise: the design grows to 32 points
    "noise_inferred_D3": dict(
        x0=np.array([1.0, 2.0, -1.5]),
        bounds=_box(3),
        options={"random_seed": 12},
        noise_sd=1.0,
    ),
    # Two log-transformed variables (bounds all positive, pub / plb >= 10)
    # and a linear one; the noise test finds no noise
    "log_transform_D3": dict(
        x0=np.array([0.5, 2.0, 1.0]),
        bounds=(
            np.array([1e-3, 1e-3, -20.0]),
            np.array([1e3, 1e3, 20.0]),
            np.array([1e-2, 1e-1, -5.0]),
            np.array([1e2, 10.0, 5.0]),
        ),
        options={"random_seed": 13},
    ),
    # x0 omitted: drawn from the run's generator in the plausible box
    "random_x0_D2": dict(
        x0=None,
        bounds=_box(2),
        options={"uncertainty_handling": False, "random_seed": 14},
    ),
}


def run_initialization(name):
    """The points passed to the target by the initialization of the run
    ``name``: ``{"x0": ..., "noise_test": ..., "design": ...}``, the second
    with one row or none."""
    cfg = CONFIGS[name]
    noise_sd = cfg.get("noise_sd")
    noise = np.random.default_rng(0) if noise_sd else None
    calls = []

    def target(x):
        x = np.array(x, dtype=float).ravel()
        calls.append(x)
        y = float(np.sum(x**2))
        if noise_sd:
            y += noise_sd * noise.standard_normal()
        return y

    n_init = []

    def stop_at_init(x, optim_state, state):
        if state == "init":
            n_init.append(len(calls))
        return True

    options = {"display": "off", "output_fcn": stop_at_init}
    options.update(cfg["options"])
    lb, ub, plb, pub = cfg["bounds"]
    BADS(target, cfg["x0"], lb, ub, plb, pub, options=options).optimize()
    points = np.array(calls[: n_init[0]])
    n_test = 1 if options.get("uncertainty_handling") is None else 0
    return {
        "x0": points[0],
        "noise_test": points[1 : 1 + n_test],
        "design": points[1 + n_test :],
    }


def _log_variables(name):
    lb, ub, plb, pub = CONFIGS[name]["bounds"]
    positive = (lb > 0) & (ub > 0) & (plb > 0) & (pub > 0)
    return positive & (pub / np.where(positive, plb, 1.0) >= 10)


@pytest.fixture(scope="module")
def pins():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_covers_the_configurations(pins):
    assert set(pins) - {"about"} == set(CONFIGS)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_initial_design_is_pinned(name, pins):
    got = run_initialization(name)
    log = _log_variables(name)
    D = log.size
    for part in ("x0", "noise_test", "design"):
        expected = np.array(pins[name][part], dtype=float)
        if part != "x0":
            expected = expected.reshape(-1, D)
        actual = got[part]
        assert actual.shape == expected.shape, part
        np.testing.assert_array_equal(
            actual[..., ~log], expected[..., ~log], err_msg=part
        )
        if log.any():
            np.testing.assert_array_max_ulp(
                actual[..., log], expected[..., log], maxulp=MAX_ULP_LOG
            )


def test_configurations_reach_what_they_pin():
    """The noise test runs, and repeats the start, where
    ``uncertainty_handling`` is left ``None``; the design has 8 points at
    D = 4 and 32 when the noise test finds noise; one configuration has
    both log-transformed and linear variables."""
    runs = {name: run_initialization(name) for name in CONFIGS}
    assert len(runs["deterministic_D4"]["noise_test"]) == 0
    assert len(runs["deterministic_D4"]["design"]) == 8
    assert len(runs["noise_inferred_D3"]["design"]) == 32
    for name in ("noise_inferred_D3", "log_transform_D3"):
        assert runs[name]["noise_test"].tolist() == [runs[name]["x0"].tolist()]
    assert _log_variables("log_transform_D3").tolist() == [True, True, False]


def _dump(pins):
    """JSON with one point per line; floats as ``repr`` writes them, which
    ``json.loads`` reads back exactly."""
    lines = ["{", f' "about": {json.dumps(pins["about"])},']
    names = [n for n in pins if n != "about"]
    for i, name in enumerate(names):
        lines.append(f" {json.dumps(name)}: {{")
        parts = list(pins[name].items())
        for j, (part, value) in enumerate(parts):
            end = "," if j < len(parts) - 1 else ""
            if part == "x0":
                lines.append(f"  {json.dumps(part)}: {json.dumps(value)}{end}")
                continue
            rows = [f"   {json.dumps(row)}" for row in value]
            if rows:
                body = ",\n".join(rows)
                lines.append(f"  {json.dumps(part)}: [\n{body}\n  ]{end}")
            else:
                lines.append(f"  {json.dumps(part)}: []{end}")
        lines.append(" }" + ("," if i < len(names) - 1 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def regenerate():
    pins = {
        "about": (
            "The points of the initialization of the runs of"
            " test_initial_design_pin.py, which says when and how to"
            " regenerate them."
        )
    }
    for name in CONFIGS:
        pins[name] = {
            k: v.tolist() for k, v in run_initialization(name).items()
        }
    text = _dump(pins)
    assert json.loads(text) == pins
    FIXTURE.write_text(text, encoding="utf-8")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regenerate", action="store_true", required=True)
    ap.parse_args()
    regenerate()
