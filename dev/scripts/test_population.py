"""Checks of ``population.py``: the record schema, the reference minima of
the real-data targets, resumability, and the statistics of ``compare`` on
synthetic records. Run by path, from the repository root::

    python -m pytest dev/scripts/test_population.py

The tests of real records run short BADS optimizations (20 to 60
evaluations), the resumability test in spawned processes; the tests of
``compare`` write synthetic records.
"""

import json

import benchmark_targets as bt
import numpy as np
import population as pp
import pytest

FAST = {"max_fun_evals": 20}


# --------------------------------------------------------------------------
# Records of real runs
# --------------------------------------------------------------------------


def test_record_schema(tmp_path):
    row = pp.run_task("sphere_D2", 3, FAST, 1.0, str(tmp_path))
    assert row["status"] == "ok"
    rec = json.loads((tmp_path / "sphere_D2_seed3.json").read_text())
    for key in (
        "label",
        "seed",
        "problem",
        "D",
        "noise",
        "x0",
        "tolerance",
        "requested_options",
        "effective_options",
        "final",
        "meta",
    ):
        assert key in rec
    assert (rec["label"], rec["seed"], rec["problem"], rec["D"]) == (
        "sphere_D2",
        3,
        "sphere",
        2,
    )
    assert rec["noise"] == "none"
    assert rec["requested_options"]["random_seed"] == 3
    assert rec["requested_options"]["max_fun_evals"] == 20
    assert rec["effective_options"]["max_fun_evals"] == 20
    final = rec["final"]
    assert set(final) == {
        "x",
        "fval",
        "fsd",
        "true_error",
        "func_count",
        "iterations",
        "message",
        "wall_s",
        "crashed",
        "exception",
        "min_noise_var",
    }
    assert final["crashed"] is False and final["exception"] is None
    assert len(final["x"]) == 2 and final["func_count"] == 20
    prob = bt.find_config("sphere_D2").make(seed=3)
    assert final["true_error"] == prob.f_true(np.array(final["x"])) - 0.0
    assert rec["x0"] == prob.x0.tolist()
    assert final["true_error"] >= 0 and final["wall_s"] > 0
    assert 0 < final["min_noise_var"] < np.inf
    meta = rec["meta"]
    for key in (
        "git",
        "python",
        "numpy",
        "scipy",
        "pybads",
        "gpyreg",
        "gpyreg_source",
        "threads",
        "started",
        "finished",
    ):
        assert key in meta
    assert set(meta["git"]) == {"sha", "dirty"}
    assert meta["gpyreg_source"]["path"]


def test_seed_fixes_run(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    opts = {"max_fun_evals": 60}
    for d in (a, b):
        pp.run_task("sphere_D3_homo", 1, opts, 1.0, str(d))
    ra, rb = (
        json.loads((d / "sphere_D3_homo_seed1.json").read_text())["final"]
        for d in (a, b)
    )
    for key in ("x", "fval", "true_error", "func_count", "min_noise_var"):
        assert ra[key] == rb[key]


def test_crash_is_an_outcome(tmp_path):
    # specify_target_noise=True with uncertainty_handling=False: BADS raises
    row = pp.run_task(
        "sphere_D3_hetero",
        0,
        {"uncertainty_handling": False},
        1.0,
        str(tmp_path),
    )
    assert row["status"] == "crash"
    rec = json.loads((tmp_path / "sphere_D3_hetero_seed0.json").read_text())
    assert rec["final"]["crashed"] is True
    assert rec["final"]["exception"]["type"] == "ValueError"
    assert rec["final"]["true_error"] is None
    pop = pp.load_population(tmp_path)
    assert pop["sphere_D3_hetero"]["crashed"].tolist() == [True]
    assert np.isnan(pop["sphere_D3_hetero"]["true_error"][0])


def test_real_targets_reference_and_pins():
    for name, D in bt.REAL_TARGETS.items():
        prob = bt.make_problem(name, D, seed=0)
        assert bt._close(prob.f_true(prob.x_min), prob.f_min)
        assert np.all(prob.lb <= prob.x_min) and np.all(prob.x_min <= prob.ub)
        assert prob.tolerance == bt.TOL_REAL and prob.pins
        for x, expected, kind, tol in prob.pins:
            assert abs(bt._pin_value(prob, x, kind) - expected) <= tol


def test_real_target_record(tmp_path):
    # the error of a real-data target is measured from its reference minimum
    opts = {"max_fun_evals": 60}
    row = pp.run_task("multisensory_s1_D6_homo", 0, opts, 1.0, str(tmp_path))
    assert row["status"] == "ok"
    path = tmp_path / "multisensory_s1_D6_homo_seed0.json"
    rec = json.loads(path.read_text())
    ref = bt.reference_optimum("multisensory_s1", 6)
    assert rec["f_min"] == ref["f_min"]
    assert rec["tolerance"] == bt.TOL_REAL
    assert rec["effective_options"]["uncertainty_handling"] is True
    prob = bt.find_config("multisensory_s1_D6_homo").make(seed=0)
    x = np.array(rec["final"]["x"])
    assert rec["final"]["true_error"] == prob.f_true(x) - ref["f_min"]


def test_run_resumes(tmp_path, capsys):
    out = tmp_path / "pop"
    base = ["run", "--suite", "smoke", "--only", "sphere_D2", "--out"]
    args = base + [str(out), "--options", json.dumps(FAST)]
    assert pp.main(args + ["--seeds", "0-1"]) == 0
    files = sorted(out.glob("*.json"))
    assert [f.name for f in files] == [
        "sphere_D2_seed0.json",
        "sphere_D2_seed1.json",
    ]
    before = {f.name: (f.stat().st_mtime_ns, f.read_text()) for f in files}
    # an interrupted write does not count as a finished run
    (out / "sphere_D2_seed2.json.tmp").write_text("{")
    capsys.readouterr()
    assert pp.main(args + ["--seeds", "0-2"]) == 0
    assert "1 runs (2 already done)" in capsys.readouterr().out
    for name, state in before.items():
        f = out / name
        assert (f.stat().st_mtime_ns, f.read_text()) == state
    assert (out / "sphere_D2_seed2.json").exists()
    assert pp.main(args + ["--seeds", "0-2"]) == 0
    assert "0 runs (3 already done)" in capsys.readouterr().out
    rec = json.loads((out / "sphere_D2_seed0.json").read_text())
    assert rec["meta"]["threads"] == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


# --------------------------------------------------------------------------
# compare on synthetic records
# --------------------------------------------------------------------------


def write_population(d, label, errors, seeds=None, crashed=None, evals=None):
    """Minimal records with what ``load_population`` reads."""
    d.mkdir(parents=True, exist_ok=True)
    n = len(errors)
    seeds = list(range(n)) if seeds is None else list(seeds)
    crashed = [False] * n if crashed is None else list(crashed)
    evals = [100] * n if evals is None else list(evals)
    for s, e, c, fc in zip(seeds, errors, crashed, evals):
        rec = {
            "label": label,
            "seed": int(s),
            "tolerance": 1e-3,
            "x0": [float(s), 0.0],
            "final": {
                "true_error": None if c else float(e),
                "func_count": int(fc),
                "crashed": bool(c),
                "wall_s": 1.0,
                "min_noise_var": 1e-7,
            },
            "meta": {"git": {"sha": "abc", "dirty": False}},
        }
        (d / f"{label}_seed{s}.json").write_text(json.dumps(rec))


def lognormal_errors(rng, n=30, mean=-4.0, sd=1.0):
    return 10 ** rng.normal(mean, sd, n)


def compare(ref_dir, new_dir, **kw):
    return pp.compare_populations(
        pp.load_population(ref_dir), pp.load_population(new_dir), **kw
    )


def test_identical_populations_do_not_flag(tmp_path, capsys):
    rng = np.random.default_rng(1)
    for label in ("a_D2", "b_D3", "c_D6"):
        err = lognormal_errors(rng)
        evals = rng.integers(50, 150, 30)
        for side in ("ref", "new"):
            write_population(tmp_path / side, label, err, evals=evals)
    text, flagged = compare(tmp_path / "ref", tmp_path / "new")
    assert flagged == set()
    assert "FLAG" not in text and "WARNING" not in text
    assert (
        pp.main(["compare", str(tmp_path / "ref"), str(tmp_path / "new")]) == 0
    )


def test_shifted_population_flags_ks_and_paired(tmp_path):
    rng = np.random.default_rng(2)
    err = lognormal_errors(rng)
    write_population(tmp_path / "ref", "a_D2", err)
    write_population(tmp_path / "new", "a_D2", err * 1e3)
    text, flagged = compare(tmp_path / "ref", tmp_path / "new")
    assert flagged == {"a_D2"}
    rows = [line for line in text.splitlines() if "FLAG" in line]
    assert any("| KS | true_error |" in r for r in rows)
    assert any("| signed-rank |" in r for r in rows)
    assert "+3.000 [+3.000, +3.000]" in text
    assert (
        pp.main(["compare", str(tmp_path / "ref"), str(tmp_path / "new")]) == 1
    )


def test_small_paired_shift_flags_only_signed_rank(tmp_path):
    rng = np.random.default_rng(3)
    log_err = rng.normal(-4.0, 1.0, 30)
    new_log = log_err + 0.1 + rng.normal(0.0, 0.02, 30)
    write_population(tmp_path / "ref", "a_D2", 10**log_err)
    write_population(tmp_path / "new", "a_D2", 10**new_log)
    text, flagged = compare(tmp_path / "ref", tmp_path / "new")
    assert flagged == {"a_D2"}
    ks_row = next(r for r in text.splitlines() if "| KS | true_error |" in r)
    sr_row = next(r for r in text.splitlines() if "| signed-rank |" in r)
    assert ks_row.endswith("| ok |") and sr_row.endswith("| FLAG |")


def test_func_count_shift_flags_ks(tmp_path):
    rng = np.random.default_rng(4)
    err = lognormal_errors(rng)
    write_population(
        tmp_path / "ref", "a_D2", err, evals=rng.integers(80, 100, 30)
    )
    write_population(
        tmp_path / "new", "a_D2", err, evals=rng.integers(120, 140, 30)
    )
    text, flagged = compare(tmp_path / "ref", tmp_path / "new")
    assert flagged == {"a_D2"}
    ks_row = next(r for r in text.splitlines() if "| KS | func_count |" in r)
    assert ks_row.endswith("| FLAG |")


def test_crash_rise_from_zero_flags(tmp_path):
    rng = np.random.default_rng(5)
    err = lognormal_errors(rng)
    crash1 = [False] * 29 + [True]
    crash2 = [False] * 28 + [True, True]
    write_population(tmp_path / "ref", "a_D2", err)
    write_population(tmp_path / "new", "a_D2", err, crashed=crash1)
    write_population(tmp_path / "ref", "b_D2", err, crashed=crash1)
    write_population(tmp_path / "new", "b_D2", err, crashed=crash2)
    text, flagged = compare(tmp_path / "ref", tmp_path / "new")
    assert flagged == {"a_D2"}
    assert "Crash count rising from zero: ['a_D2']" in text


def test_split_is_a_null_check(tmp_path):
    rng = np.random.default_rng(6)
    for label in ("a_D2", "b_D3"):
        write_population(tmp_path / "ref", label, lognormal_errors(rng))
    pop = pp.load_population(tmp_path / "ref")
    even, odd = pp.split_population(pop)
    assert even["a_D2"]["seeds"].tolist() == list(range(0, 30, 2))
    assert odd["a_D2"]["seeds"].tolist() == list(range(1, 30, 2))
    text, flagged = pp.compare_populations(
        even, odd, paired=False, crash_flag=False
    )
    assert flagged == set() and "signed-rank" not in text
    assert pp.main(["compare", str(tmp_path / "ref"), "--split"]) == 0


def test_unpaired_start_points_warn(tmp_path):
    rng = np.random.default_rng(7)
    err = lognormal_errors(rng)
    write_population(tmp_path / "ref", "a_D2", err)
    write_population(tmp_path / "new", "a_D2", err)
    rec_path = tmp_path / "new" / "a_D2_seed4.json"
    rec = json.loads(rec_path.read_text())
    rec["x0"] = [99.0, 0.0]
    rec_path.write_text(json.dumps(rec))
    text, _ = compare(tmp_path / "ref", tmp_path / "new")
    assert "WARNING" in text and "'a_D2': 1" in text


def test_summary(tmp_path):
    err = [1e-4] * 10 + [1e-2] * 10
    write_population(tmp_path, "a_D2", err, crashed=[False] * 19 + [True])
    assert pp.main(["summary", str(tmp_path)]) == 0
    text = (tmp_path / "summary.md").read_text()
    row = next(r for r in text.splitlines() if r.startswith("| a_D2 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:3] == ["20", "1"]  # runs, crashed
    assert cells[5] == "0.50"  # solved: 10 of 20, the crash unsolved


# --------------------------------------------------------------------------
# Holm and the KS threshold
# --------------------------------------------------------------------------


def test_holm():
    reject, adj = pp.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert reject.tolist() == [True, False, False, True]
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    # step-down: rejects what Bonferroni (0.025 each) would not
    assert pp.holm([0.01, 0.04])[0].tolist() == [True, True]
    # and stops at the first acceptance
    assert pp.holm([0.03, 0.04])[0].tolist() == [False, False]
    assert pp.holm([0.5, 1.0])[1].tolist() == [1.0, 1.0]
    assert pp.holm([])[0].tolist() == []


def test_ks_threshold():
    from scipy import stats

    thr = pp.ks_threshold(30, 30, 0.05)
    k = round(thr * 30)
    assert thr == pytest.approx(k / 30)
    x = np.arange(30.0)
    assert stats.ks_2samp(x, x + k).pvalue <= 0.05
    assert stats.ks_2samp(x, x + k - 1).pvalue > 0.05
    assert pp.ks_threshold(30, 30, 0.001) > thr
