"""Checks of ``replay.py``: the comparison of two recordings on synthetic
traces (identical, parted at a known evaluation, generator states that
differ, a GP computation that differs, a shorter run, platform keys that
differ, the repeats of one recording), the loud failure on a missing
private name, and one short real recording repeated in one process. Run by
path, from the repository root::

    python -m pytest dev/scripts/test_replay.py

The real recording runs ``sphere_D2`` at 20 evaluations twice, in a child
process, in a few seconds.
"""

import copy
import json

import numpy as np
import pytest
import replay as rp

D = 2
N_EVALS = 12
N_INIT = 6  # the start, the noise test and a design of four points
PLATFORM = {
    "os": "Linux",
    "machine": "x86_64",
    "cpu": "a CPU",
    "numpy": "2.4.6",
    "openblas_runtime": [{"library": "libopenblas.so", "corename": "Haswell"}],
    "env": {"OPENBLAS_CORETYPE": "Haswell", "OMP_NUM_THREADS": "1"},
}


def _digest(i):
    return f"{i:016x}"


def _synthetic(name="sphere_D2_seed0", repeat=0):
    """The arrays and sidecar of a made-up run, in the layout of
    ``Recorder.arrays``."""
    g = np.random.default_rng(0)
    rec = rp.Recorder()
    for k in range(N_EVALS):
        if k < N_INIT:
            stage, it = "init", -1
        else:
            stage, it = ("search" if k % 2 else "poll"), (k - N_INIT) // 2
        rec.evals.append(
            (g.random(D), float(g.random()), np.nan, stage, it, _digest(k))
        )
    for i in range(4):
        rec.steps.append(
            ("search", i, 6 + i, 7 + i, "ES-wcm", "failure", -i, _digest(i))
        )
    for i in range(3):
        rec.fits.append(
            (
                "local_gp_fitting",
                "search",
                i,
                6 + i,
                6 + i,
                0,
                0.0,
                0,
                g.random((1, 5)),
                _digest(i),
            )
        )
    arrays = rec.arrays(D)
    arrays["hist_fval"] = g.random((3, 1))
    arrays.update(rp._ragged("hist_hyp", [g.random((1, 5)) for _ in range(3)]))
    arrays["res_x"] = g.random(D)
    sidecar = {
        "name": name,
        "label": "sphere_D2",
        "seed": 0,
        "repeat": repeat,
        "platform": copy.deepcopy(PLATFORM),
        "provenance": {"git": {"sha": "abc1234", "dirty": False}},
        "result": {"fval": 0.5, "func_count": N_EVALS, "iterations": 3},
        "crash": None,
        "counts": {"evals": N_EVALS, "steps": 4, "fits": 3, "iterations": 3},
        "wall_s": 0.1,
    }
    return arrays, sidecar


def _write(d, arrays, sidecar):
    d.mkdir(exist_ok=True)
    rp.write_trace(d, sidecar["name"], arrays, sidecar)
    return d


def _pair(tmp_path, change=None):
    """Two recordings of the synthetic run, ``change(arrays, sidecar)``
    applied to the second; the comparison of their one run."""
    a, s = _synthetic()
    base = _write(tmp_path / "base", a, s)
    a, s = _synthetic()
    if change is not None:
        change(a, s)
    new = _write(tmp_path / "new", a, s)
    (pair,), lonely = rp.pair_traces(base, new)
    assert lonely == []
    return base, new, rp.compare_traces(*pair)


def test_identical(tmp_path, capsys):
    base, new, c = _pair(tmp_path)
    assert c["identical"] and all(c["streams"].values())
    assert rp.main(["check", str(base), str(new)]) == 0
    out = capsys.readouterr().out
    assert "platform: identical" in out
    assert "identical  (12 evaluations, 4 steps, 3 GP computations)" in out


def test_parted_at_a_known_evaluation(tmp_path, capsys):
    k = 8

    def change(a, s):  # the value moves; the same draws came before
        a["eval_x"][k, 1] += 1e-10
        a["eval_y"][k + 1 :] += 1.0

    base, new, c = _pair(tmp_path, change)
    e = c["eval"]
    assert not c["identical"] and not c["streams"]["evals"]
    assert c["streams"]["steps"] and c["streams"]["fits"]
    assert (e["k"], e["short"], e["stage"], e["iter"]) == (
        k,
        False,
        ("poll", "poll"),
        (1, 1),
    )
    assert e["dx"] == pytest.approx(1e-10, rel=1e-3) and e["dy"] == 0.0
    assert e["rng_agree"] and e["rng_agree_before"]
    assert c["horizons"] == {1e-12: k, 1e-8: k + 1}
    assert c["fit_hyp"] is None and c["step"] is None
    assert rp.main(["check", str(base), str(new)]) == 1
    out = capsys.readouterr().out
    assert f"PARTED at evaluation {k} (iteration 2, poll)" in out
    assert "generator states agree" in out
    assert "0 identical, 1 differ" in out


def test_generator_states_that_differ(tmp_path, capsys):
    k = 7

    def change(a, s):  # a draw added before evaluation k: another branch
        a["eval_rng"][k:] = "ffffffffffffffff"
        a["eval_x"][k:] += 0.5

    base, new, c = _pair(tmp_path, change)
    e = c["eval"]
    assert e["k"] == k and not e["rng_agree"] and e["rng_agree_before"]
    assert e["dx"] == pytest.approx(0.5)
    rp.main(["check", str(base), str(new)])
    assert "generator states differ" in capsys.readouterr().out


def test_generator_states_alone(tmp_path):
    def change(a, s):  # the same points and values, other draws
        a["eval_rng"][5] = "ffffffffffffffff"

    _, _, c = _pair(tmp_path, change)
    assert not c["identical"]
    assert c["eval"]["k"] == 5 and c["eval"]["dx"] == 0.0
    assert not c["eval"]["rng_agree"]


def test_earliest_gp_computation_that_differs(tmp_path, capsys):
    def change(a, s):
        off = a["fit_hyp_off"]
        a["fit_hyp_flat"][off[1] + 2] += 3e-15
        a["fit_hyp_flat"][off[2]] += 1.0

    base, new, c = _pair(tmp_path, change)
    f = c["fit_hyp"]
    assert c["eval"] is None and not c["streams"]["fits"]
    assert f["i"] == 1 and f["dhyp"] == pytest.approx(3e-15, rel=0.2)
    assert f["base"]["kind"] == "local_gp_fitting" and f["base"]["fc"] == 7
    assert rp.main(["check", str(base), str(new)]) == 1
    out = capsys.readouterr().out
    assert "DIFFERS, with identical evaluations" in out
    assert "first GP hyperparameters that differ: computation 1" in out


def test_shorter_run(tmp_path):
    def change(a, s):
        for key in [k for k in a if k.startswith("eval_")]:
            a[key] = a[key][:10]

    _, _, c = _pair(tmp_path, change)
    assert c["eval"] == {"k": 10, "short": True, "n_base": 12, "n_new": 10}


def test_platform_keys_that_differ(tmp_path, capsys):
    def change(a, s):
        s["platform"]["env"]["OPENBLAS_CORETYPE"] = "Sandybridge"
        s["platform"]["openblas_runtime"][0]["corename"] = "Sandybridge"

    base, new, c = _pair(tmp_path, change)
    assert rp.main(["check", str(base), str(new)]) == 2
    out = capsys.readouterr().out
    assert "env.OPENBLAS_CORETYPE: 'Haswell' vs 'Sandybridge'" in out
    assert "openblas_runtime[0].corename" in out
    assert "refused" in out and "identical  (" not in out
    # the runs themselves are the same
    assert rp.main(["check", str(base), str(new), "--force"]) == 0


def test_runs_on_one_side_only(tmp_path, capsys):
    a, s = _synthetic()
    base = _write(tmp_path / "base", a, s)
    a, s = _synthetic(name="sphere_D2_seed1")
    _write(base, a, dict(s, seed=1))
    a, s = _synthetic()
    new = _write(tmp_path / "new", a, s)
    assert rp.main(["check", str(base), str(new)]) == 1
    out = capsys.readouterr().out
    assert "sphere_D2_seed1" in out and "on one side only" in out


def test_repeats_within_one_recording(tmp_path, capsys):
    d = tmp_path / "rec"
    a, s = _synthetic()
    _write(d, a, s)
    a, s = _synthetic(name="sphere_D2_seed0_rep1", repeat=1)
    a["eval_y"][9] += 1.0
    _write(d, a, s)
    pairs, lonely = rp.pair_traces(d)
    assert [(b.name, n.name) for b, n in pairs] == [
        ("sphere_D2_seed0", "sphere_D2_seed0_rep1")
    ]
    assert rp.main(["check", str(d)]) == 1
    assert "PARTED at evaluation 9" in capsys.readouterr().out


def test_missing_private_name_fails_loudly(monkeypatch):
    import pybads.bads.bads as bads_module

    monkeypatch.delattr(bads_module, "local_gp_fitting")
    with pytest.raises(rp.MissingName, match="local_gp_fitting"):
        rp.record_run("sphere_D2", 0, 0.02)
    monkeypatch.undo()
    monkeypatch.delattr(bads_module.BADS, "_poll_step_")
    with pytest.raises(rp.MissingName, match="_poll_step_"):
        rp.record_run("sphere_D2", 0, 0.02)


def test_record_twice_in_one_process(tmp_path, capsys):
    out = tmp_path / "rec"
    args = ["record", "--out", str(out), "--configs", "sphere_D2"]
    args += ["--budget-scale", "0.02", "--repeat", "2"]
    assert rp.main(args) == 0
    t = rp.Trace(out / "sphere_D2_seed0.json")
    assert (out / "sphere_D2_seed0_rep1.npz").exists()
    # every call of the target, in the order BADS made them
    assert t.n_evals == t.meta["result"]["func_count"] == 20
    stages = list(t.arrays["eval_stage"])
    assert stages[:N_INIT] == ["init"] * N_INIT
    assert set(stages[N_INIT:]) <= {"search", "poll"}
    assert t.arrays["eval_x"][1].tolist() == t.arrays["eval_x"][0].tolist()
    assert t.arrays["fit_kind"][0] == "init_and_train_gp"
    assert t.meta["platform"]["env"]["OMP_NUM_THREADS"] == "1"
    assert t.meta["provenance"]["pin"]["threads"] == 1
    assert rp.main(["check", str(out)]) == 0
    assert "identical  (20 evaluations" in capsys.readouterr().out
    # a second recording never goes into the same directory
    with pytest.raises(SystemExit, match="already holds"):
        rp.main(args)


def test_trace_sidecar_is_json(tmp_path):
    a, s = _synthetic()
    d = _write(tmp_path / "x", a, s)
    meta = json.loads((d / "sphere_D2_seed0.json").read_text())
    assert meta["platform"] == PLATFORM
