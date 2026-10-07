"""Checks of ``make_oracle_fixtures.py`` on a copy of the stored fixtures:
``--rebaseline`` refuses new references whose decisions lie within their
margin of a threshold, names the remedy and leaves every file as it was,
and otherwise replaces one oracle's references and records why. Run by
path, from the repository root::

    python -m pytest dev/scripts/test_make_oracle_fixtures.py

It computes the oracles of one fixture a few times, in a few seconds.
"""

import shutil

import make_oracle_fixtures as mof
import numpy as np
import pytest

FIXTURE = "sphere_D2_init"


@pytest.fixture
def fixtures(tmp_path, monkeypatch):
    """A copy of the stored fixtures, which the script reads and writes in
    place of the package's, with the checkout's provenance as it records
    it for them."""
    here = mof.checkout_info()
    copy = tmp_path / "fixtures"
    shutil.copytree(mof.FIXTURES, copy)
    monkeypatch.setattr(mof, "FIXTURES", copy)
    monkeypatch.setattr(mof, "checkout_info", lambda: here)
    return copy


def _contents(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


@pytest.mark.parametrize("oracle", ["gp_training_set", "improvement"])
def test_rebaseline_refuses_a_decision_within_its_margin(
    fixtures, monkeypatch, oracle
):
    """Under a margin that no decision clears, the new references could
    flip on another platform: the fixture keeps its references, and the
    message names ``--write``, which alone re-chooses the snapshot's
    reduced training sets."""
    before = _contents(fixtures)
    monkeypatch.setattr(mof, "MARGIN", np.inf)
    with pytest.raises(SystemExit) as info:
        mof.rebaseline([FIXTURE], oracle, "a test")
    message = str(info.value)
    assert message.startswith(f"{FIXTURE}: a ")
    assert f"could flip this decision of {oracle}" in message
    assert "not replaced" in message and "--write --reason TEXT" in message
    assert _contents(fixtures) == before


def test_rebaseline_replaces_one_oracle(fixtures, monkeypatch):
    """The selected references match the current computation exactly and
    the historical values within their portable tolerances. Every other
    array stays exact, and the rebaseline records its reason."""
    before = mof.load_arrays(fixtures / FIXTURE)
    references = mof.load_snapshot(fixtures / FIXTURE)["ref"]
    expected = {}
    compute_outputs = mof.compute_outputs

    def record_outputs(snap, cases):
        outputs = compute_outputs(snap, cases)
        for case, name, view in cases:
            assert name == "gp_training_set"
            expected[case] = {
                key: value.copy()
                for key, value in mof.portable_outputs(
                    snap, view, name, outputs[case]
                ).items()
            }
        return outputs

    monkeypatch.setattr(mof, "compute_outputs", record_outputs)
    mof.rebaseline([FIXTURE], "gp_training_set", "a test")
    after = mof.load_arrays(fixtures / FIXTURE)
    assert set(after) == set(before)
    assert expected
    replaced = {
        f"ref/{case}/{key}": value
        for case, outputs in expected.items()
        for key, value in outputs.items()
    }
    assert replaced.keys() <= after.keys()
    for key, value in before.items():
        assert np.array_equal(
            replaced.get(key, value), after[key], equal_nan=True
        ), key
    tolerance = mof.ORACLES["gp_training_set"].tolerance
    for case, outputs in expected.items():
        rows = mof.compare(references[case], outputs, tolerance)
        assert all(row[3] for row in rows), mof.format_rows(rows)
    (entry,) = mof.load_tree(fixtures / FIXTURE)["meta"]["rebaselined"]
    assert (entry["oracle"], entry["reason"]) == ("gp_training_set", "a test")
