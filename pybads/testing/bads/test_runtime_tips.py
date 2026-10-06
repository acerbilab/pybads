"""The runtime tips (`pybads/bads/_runtime_tips.py`): the first eligible run
of a session shows a tip, then every third; each tip shows at most once,
low-frequency tips are spaced, a run that is not eligible neither counts nor
uses a tip, and a tip changes nothing in the run that shows it."""

import logging
import os
import platform
import random
import re
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from pybads import BADS
from pybads.bads import _runtime_tips as rt
from pybads.bads._tip_catalog import TIPS, Tip

D = 3
# Apple's Accelerate makes two runs of one seed differ in their last bits on
# macOS arm64, where the seed alone decides the start and the initial design
# (dev/results/2026-09-28-macos-arm64-repeatability.md)
_REPEATS_BIT_FOR_BIT = not (
    sys.platform == "darwin" and platform.machine() == "arm64"
)
_REPO = Path(__file__).resolve().parents[3]
_FAQ_MD = _REPO / "docsrc" / "source" / "faq.md"
_BLOB = "https://github.com/acerbilab/pybads/blob/main/"


@pytest.fixture(autouse=True)
def _isolated_tip_state():
    """Each test starts a session of its own, with a seeded shuffle, and
    leaves a fresh one."""
    rt._reset_runtime_tip_state(rng=random.Random(0))
    yield
    rt._reset_runtime_tip_state()


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def tip_logger():
    logger = logging.getLogger("pybads.testing.runtime_tips")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records = _Records()
    logger.addHandler(records)
    yield logger, records.messages
    logger.removeHandler(records)


class _InOrder(random.Random):
    """A generator whose shuffle keeps the catalogue's order."""

    def shuffle(self, x):
        pass


def _catalog(*frequencies):
    return tuple(
        Tip(id=f"tip{i}", text=f"Text {i}.", frequency=frequency)
        for i, frequency in enumerate(frequencies)
    )


def _shown(logger, n, catalog, rng=None):
    return [
        rt.consider_runtime_tip(
            logger=logger, enabled=True, catalog=catalog, rng=rng
        )
        for _ in range(n)
    ]


def test_first_eligible_run_then_every_third(tip_logger):
    logger, messages = tip_logger
    shown = _shown(logger, 7, _catalog(*["normal"] * 5))
    assert [tip is not None for tip in shown] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    ids = [tip.id for tip in shown if tip is not None]
    assert len(set(ids)) == 3
    assert len(messages) == 3


def test_each_tip_once_then_none(tip_logger):
    logger, messages = tip_logger
    catalog = _catalog("normal", "normal")
    shown = [tip for tip in _shown(logger, 12, catalog) if tip is not None]
    assert sorted(tip.id for tip in shown) == ["tip0", "tip1"]
    assert len(messages) == 2


def test_the_shuffle_decides_the_order(tip_logger):
    logger, _ = tip_logger
    catalog = _catalog(*["normal"] * 6)
    orders = []
    for seed in (1, 2):
        rt._reset_runtime_tip_state()
        shown = _shown(logger, 16, catalog, rng=random.Random(seed))
        orders.append([tip.id for tip in shown if tip is not None])
    assert (
        sorted(orders[0])
        == sorted(orders[1])
        == sorted(tip.id for tip in catalog)
    )
    assert orders[0] != orders[1]


def test_low_frequency_tips_are_spaced(tip_logger):
    """After a low-frequency tip, the next is an ordinary one while one
    remains; a catalogue of low-frequency tips alone still shows them all."""
    logger, _ = tip_logger
    shown = _shown(
        logger,
        7,
        _catalog("low_frequency", "low_frequency", "normal"),
        rng=_InOrder(),
    )
    assert [tip.id for tip in shown if tip is not None] == [
        "tip0",
        "tip2",
        "tip1",
    ]
    rt._reset_runtime_tip_state()
    shown = _shown(
        logger, 4, _catalog("low_frequency", "low_frequency"), rng=_InOrder()
    )
    assert [tip.id for tip in shown if tip is not None] == ["tip0", "tip1"]


@pytest.mark.parametrize("cause", ["tips_off", "logger_above_info"])
def test_runs_that_are_not_eligible_neither_count_nor_use_a_tip(
    tip_logger, cause
):
    logger, messages = tip_logger
    catalog = _catalog("normal", "normal")
    if cause == "logger_above_info":
        logger.setLevel(logging.WARNING)
    for _ in range(5):
        assert (
            rt.consider_runtime_tip(
                logger=logger, enabled=cause != "tips_off", catalog=catalog
            )
            is None
        )
    assert messages == []
    assert rt._ELIGIBLE_STARTS == 0
    assert rt._SEEN_IDS == set()
    logger.setLevel(logging.INFO)
    assert (
        rt.consider_runtime_tip(logger=logger, enabled=True, catalog=catalog)
        is not None
    )


def test_a_tip_is_one_record_with_its_urls_and_an_empty_line():
    tip = Tip(
        id="t",
        text="Text.",
        frequency="normal",
        urls=("https://a", "https://b"),
    )
    assert rt.format_tip(tip) == "Tip: Text.\nhttps://a\nhttps://b\n"
    assert rt.format_tip(Tip(id="u", text="T.", frequency="normal")) == (
        "Tip: T.\n"
    )


def test_invalid_catalogues_are_refused(tip_logger):
    logger, _ = tip_logger
    for catalog in [
        _catalog("normal") + _catalog("normal"),  # repeated id
        (Tip(id="t", text="", frequency="normal"),),
        (Tip(id="t", text="T.", frequency="often"),),
        (Tip(id="t", text="T.", frequency="normal", urls=["https://a"]),),
    ]:
        rt._reset_runtime_tip_state()
        with pytest.raises(ValueError):
            rt.consider_runtime_tip(
                logger=logger, enabled=True, catalog=catalog
            )


def test_the_shipped_catalogue():
    """The shipped tips are valid, plain ASCII, and link the published
    documentation or the lab's page of its tools for fitting models to
    data; the FAQ's labels and the repository's files that they link exist
    (checked where the sources are at hand, as in a checkout)."""
    rt._validate_catalog(TIPS)
    labels = (
        set(
            re.findall(r"^\((faq-[^)]+)\)=$", _FAQ_MD.read_text("utf-8"), re.M)
        )
        if _FAQ_MD.is_file()
        else None
    )
    for tip in TIPS:
        assert tip.text.isascii() and tip.text == tip.text.strip()
        assert tip.urls
        for url in tip.urls:
            assert url.isascii()
            assert url == "https://acerbilab.org/model-fitting/" or (
                url.startswith(
                    (
                        "https://acerbilab.github.io/pybads/",
                        "https://github.com/acerbilab/pybads/",
                    )
                )
            )
            faq_url = "https://acerbilab.github.io/pybads/faq.html#"
            if labels is not None and url.startswith(faq_url):
                assert url[len(faq_url) :] in labels
            if labels is not None and url.startswith(_BLOB):
                assert (_REPO / url[len(_BLOB) :]).is_file()


def test_a_forked_child_reshuffles(tip_logger):
    """The hook that `os.register_at_fork` runs in a forked child gives it a
    lock and a generator of its own, a count of eligible runs from zero and
    a new shuffle of the tips not shown yet; what the parent has shown stays
    shown."""
    logger, _ = tip_logger
    catalog = _catalog("normal", "normal", "normal")
    shown = rt.consider_runtime_tip(
        logger=logger, enabled=True, catalog=catalog
    )
    rt.consider_runtime_tip(logger=logger, enabled=True, catalog=catalog)
    lock, generator = rt._STATE_LOCK, rt._RNG
    rt._after_fork_in_child()
    assert rt._ORDER is None
    assert rt._RNG is not generator
    assert rt._STATE_LOCK is not lock
    assert isinstance(rt._STATE_LOCK, type(threading.Lock()))
    assert rt._ELIGIBLE_STARTS == 0
    assert rt._SEEN_IDS == {shown.id}
    # the child's first eligible run shows one of the tips left
    after = rt.consider_runtime_tip(
        logger=logger, enabled=True, catalog=catalog
    )
    assert after is not None and after.id != shown.id


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_the_fork_hook_is_registered(tip_logger):
    """A real fork runs the hook: the child's state is its own."""
    logger, _ = tip_logger
    rt.consider_runtime_tip(
        logger=logger, enabled=True, catalog=_catalog("normal", "normal")
    )
    parent_generator = rt._RNG
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:  # the child reports and leaves at once
        try:
            reset = (
                rt._ORDER is None
                and rt._RNG is not parent_generator
                and rt._ELIGIBLE_STARTS == 0
            )
            os.write(write_end, b"1" if reset else b"0")
        finally:
            os._exit(0)
    os.close(write_end)
    report = os.read(read_end, 1)
    os.close(read_end)
    os.waitpid(pid, 0)
    assert report == b"1"


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _make_bads(**options):
    opts = {"display": "iter", "max_fun_evals": 30, "random_seed": 3}
    opts.update(options)
    return BADS(
        _sphere,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    )


def _bads_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "BADS"]


def test_a_run_shows_a_tip_before_the_column_headers(caplog):
    with caplog.at_level(logging.INFO, logger="BADS"):
        _make_bads().optimize()
    messages = _bads_messages(caplog)
    tips = [i for i, m in enumerate(messages) if m.startswith("Tip: ")]
    assert len(tips) == 1
    opening = next(
        i
        for i, m in enumerate(messages)
        if m.startswith("Beginning optimization")
    )
    headers = next(
        i for i, m in enumerate(messages) if m.startswith(" Iteration")
    )
    assert opening < tips[0] < headers
    assert messages[tips[0]].endswith("\n")


@pytest.mark.parametrize(
    "options",
    [{"show_tips": False}, {"display": "final"}, {"display": "off"}],
    ids=["tips_off", "final", "off"],
)
def test_no_tip_without_the_iteration_display_or_with_tips_off(
    caplog, options
):
    with caplog.at_level(logging.INFO, logger="BADS"):
        _make_bads(**options).optimize()
    assert not any(m.startswith("Tip: ") for m in _bads_messages(caplog))
    assert rt._ELIGIBLE_STARTS == 0


def test_the_logger_when_the_run_starts_decides(caplog):
    """The `BADS` logger is shared, and the last `BADS` object created sets
    its level: a run whose lines another object has silenced shows no tip
    and counts as no eligible run."""
    with caplog.at_level(logging.INFO, logger="BADS"):
        bads = _make_bads()
        _make_bads(display="off")
        bads.optimize()
    assert not any(m.startswith("Tip: ") for m in _bads_messages(caplog))
    assert rt._ELIGIBLE_STARTS == 0


def test_show_tips_takes_only_a_boolean():
    with pytest.raises(ValueError, match="show_tips"):
        _make_bads(show_tips="off")


def _result_summary(bads, result):
    log = bads.function_logger
    X, Y = log.X[log.X_flag].copy(), log.Y[log.X_flag].copy()
    if not _REPEATS_BIT_FOR_BIT:
        n = bads.optim_state["eff_starting_points"]
        return (X[:n], Y[:n])
    return (
        np.asarray(result["x"]).copy(),
        result["fval"],
        result["func_count"],
        X,
        Y,
    )


def test_a_tip_changes_nothing_in_its_run(caplog):
    """A seeded run that shows a tip draws and computes what the same run
    without tips does, and leaves NumPy's global random state and that of
    the `random` module as it found them, the session's own generator of
    the tips included."""
    np_saved, random_saved = np.random.get_state(), random.getstate()
    try:
        np.random.seed(7)
        random.seed(7)
        np_before, random_before = np.random.get_state(), random.getstate()
        rt._reset_runtime_tip_state()  # the generator that a session uses
        with caplog.at_level(logging.INFO, logger="BADS"):
            with_tip = _make_bads()
            summary_with = _result_summary(with_tip, with_tip.optimize())
        assert any(m.startswith("Tip: ") for m in _bads_messages(caplog))
        np_after, random_after = np.random.get_state(), random.getstate()
        assert np_before[0] == np_after[0]
        assert np.array_equal(np_before[1], np_after[1])
        assert np_before[2:] == np_after[2:]
        assert random_before == random_after
        without = _make_bads(show_tips=False)
        summary_without = _result_summary(without, without.optimize())
        for a, b in zip(summary_with, summary_without):
            assert np.array_equal(a, b)
    finally:
        np.random.set_state(np_saved)
        random.setstate(random_saved)
