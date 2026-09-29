"""Checks of ``harness.py``: its import, which loads no NumPy and leaves
``sys.path`` as it is; the seeds it parses; the run of a configuration given
evaluations made before it, and of a ``benchmark_targets.py`` whose
problems have none; ``git_info``, inside and outside a checkout and under a
directory; ``module_source`` and ``module_identity``; and the thread
variables that it sets and that the platform key records. Run by path, from
the repository root::

    python -m pytest dev/scripts/test_harness.py

The run given evaluations made before it makes their earlier run, of 45
evaluations, in a few seconds.
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import benchmark_targets as bt
import harness
import numpy as np

import pybads


def test_import_loads_no_numpy_and_leaves_sys_path():
    code = (
        "import sys; path = list(sys.path); import harness;"
        " assert 'numpy' not in sys.modules, 'numpy';"
        " assert sys.path == path, 'sys.path'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=harness.HERE, check=True)


def test_parse_seeds():
    assert harness.parse_seeds("0-2,5, 4,2,") == [0, 1, 2, 4, 5]
    assert harness.parse_seeds(3) == [3]


def test_run_given_earlier_evaluations():
    run = harness.build_run(
        "sphere_D3_rerun", 2, budget_scale=0.02, extra_options={"max_iter": 3}
    )
    assert run.cfg == bt.find_config("sphere_D3_rerun")
    assert run.options["max_fun_evals"] == run.cfg.max_fun_evals(0.02)
    assert run.options["max_iter"] == run.requested["max_iter"] == 3
    assert run.options["random_seed"] == run.requested["random_seed"] == 2
    given = run.kwargs["precomputed_evaluations"]
    X, y = run.prob.precomputed
    assert len(given) == 2
    assert np.array_equal(given[0], X) and given[0] is not X
    assert np.array_equal(given[1], y) and given[1] is not y
    assert run.precomputed == harness.precomputed_summary(run.cfg, run.prob)
    assert run.precomputed["kind"] == "rerun"
    assert run.precomputed["rows"] == len(X)
    # the same log for the seed at each call
    again = harness.build_run("sphere_D3_rerun", 2, budget_scale=0.02)
    assert again.precomputed == run.precomputed
    assert harness.build_run("sphere_D2", 0).precomputed is None


def test_run_of_an_older_benchmark():
    """A ``benchmark_targets.py`` from before 2822c561, whose problems have
    neither ``bads_kwargs`` nor ``precomputed``: its runs are given no
    evaluations made before them, and records keep none."""
    prob = bt.find_config("sphere_D2").make(seed=0)
    older = types.SimpleNamespace(bads_args=prob.bads_args)
    config = types.SimpleNamespace(make=lambda seed, budget_scale: older)
    run = harness.build_run(config, 0)
    assert run.kwargs == {}
    assert run.precomputed is None
    assert run.args[0] == prob.fun


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_git_info(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    assert harness.git_info(outside) == {"sha": None, "dirty": None}
    assert harness.git_info(outside, describe=True) == dict.fromkeys(
        ("sha", "describe", "dirty")
    )
    repo = tmp_path / "repo"
    for d in ("a", "b"):
        (repo / d).mkdir(parents=True)
        (repo / d / "f.txt").write_text("0")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c")
    info = harness.git_info(repo, describe=True)
    assert list(info) == ["sha", "describe", "dirty"]
    assert info["sha"] and info["describe"] == info["sha"]
    assert info["dirty"] is False
    # tracked files alone, under the directory given, but the excluded
    (repo / "a" / "new.txt").write_text("untracked")
    assert harness.git_info(repo)["dirty"] is False
    (repo / "b" / "f.txt").write_text("1")
    assert harness.git_info(repo)["dirty"] is True
    assert harness.git_info(repo / "a")["dirty"] is False
    assert harness.git_info(repo, exclude=("b",))["dirty"] is False


def test_module_sources():
    source = harness.module_source("pybads")
    assert Path(source["path"]) == Path(pybads.__file__).resolve().parent
    assert set(source["git"]) == {"sha", "dirty"}
    assert harness.module_source("no_such_package") is None
    identity = harness.module_identity(pybads, "pybads")
    assert identity["source"] == source["path"]
    assert set(identity["git"]) == {"sha", "describe", "dirty"}
    assert identity["installed_version"] == harness.pkg_version("pybads")
    # an installed copy, outside a checkout that tracks it
    assert harness.module_identity(np, "numpy")["git"] is None


def test_thread_variables(monkeypatch):
    for k in harness.THREAD_VARS + ("MPLBACKEND",):
        monkeypatch.delenv(k, raising=False)
    harness.single_thread_env()
    assert harness.thread_env() == dict.fromkeys(harness.THREAD_VARS, "1")
    assert os.environ["MPLBACKEND"] == "Agg"
    key = harness.platform_key()
    assert set(key["env"]) == {"OPENBLAS_CORETYPE", *harness.THREAD_VARS}
    assert all(key["env"][k] == "1" for k in harness.THREAD_VARS)
    assert key["cpu_count"] == os.cpu_count()
