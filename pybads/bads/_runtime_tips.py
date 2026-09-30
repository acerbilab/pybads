"""The scheduling of the runtime tips within a Python session.

A run with the iteration display considers one tip before its first
iteration line (``BADS._init_mesh_``). The first eligible run of a session
shows a tip, then every third; each tip of the catalogue (``_tip_catalog``)
shows at most once per session, in an order shuffled once per session by a
private ``random.Random``, which touches no draw of a run and no other
random state, and which no seed fixes. After a low-frequency tip, the next
is an ordinary one while one remains. A run that is not eligible (tips off,
the ``BADS`` logger above INFO, or no tip left) neither advances the count
nor uses a tip. The state is the process's own: a forked child counts its
eligible runs afresh and reshuffles the tips that its parent has not shown.
"""

import logging
import os
import random
import threading

from ._tip_catalog import TIPS, Tip

_TIP_FREQUENCIES = frozenset({"normal", "low_frequency"})
# One tip every _CADENCE eligible runs, from the first
_CADENCE = 3

_STATE_LOCK = threading.Lock()
_RNG = random.Random()
_ORDER = None
_SEEN_IDS = set()
_ELIGIBLE_STARTS = 0
_LAST_FREQUENCY = None


def _validate_catalog(catalog):
    """Check the fields of the catalogue that the scheduler reads."""
    seen_ids = set()
    for tip in catalog:
        if not isinstance(tip, Tip):
            raise TypeError("Runtime tips must be Tip objects.")
        if not tip.id or tip.id in seen_ids:
            raise ValueError("Runtime tip ids must be nonempty and unique.")
        if not tip.text:
            raise ValueError("Runtime tip text must be nonempty.")
        if tip.frequency not in _TIP_FREQUENCIES:
            raise ValueError(
                "Runtime tip frequency must be 'normal' or 'low_frequency'."
            )
        if not isinstance(tip.urls, tuple) or not all(
            isinstance(url, str) and url for url in tip.urls
        ):
            raise ValueError("Runtime tip URLs must be nonempty strings.")
        seen_ids.add(tip.id)


def _remaining_order(catalog, rng):
    """The session's order of the tips not shown yet, shuffled at the first
    eligible run."""
    global _ORDER
    if _ORDER is None:
        _validate_catalog(catalog)
        _ORDER = list(catalog)
        rng.shuffle(_ORDER)
    return [tip for tip in _ORDER if tip.id not in _SEEN_IDS]


def _next_tip(remaining):
    """The first remaining tip, or, after a low-frequency tip, the first
    ordinary one while one remains."""
    first = remaining[0]
    if (
        _LAST_FREQUENCY == "low_frequency"
        and first.frequency == "low_frequency"
    ):
        for tip in remaining[1:]:
            if tip.frequency == "normal":
                return tip
    return first


def format_tip(tip):
    """The message of a tip: its text, each URL on a line of its own, and an
    empty line before the display that follows."""
    return "".join(
        [f"Tip: {tip.text}"] + [f"\n{url}" for url in tip.urls] + ["\n"]
    )


def consider_runtime_tip(*, logger, enabled, catalog=TIPS, rng=None):
    """
    Consider one tip for a run that starts, and log it if it is due.

    Parameters
    ----------
    logger : logging.Logger
        The logger of the run, which shows the tip at INFO.
    enabled : bool
        ``options['show_tips']``.
    catalog : sequence of Tip, optional
        The tips, by default those of ``_tip_catalog``.
    rng : random.Random, optional
        The generator that shuffles the catalogue, by default the session's
        own.

    Returns
    -------
    tip : Tip or None
        The tip logged, or ``None``.
    """
    global _ELIGIBLE_STARTS, _LAST_FREQUENCY

    if not enabled or not logger.isEnabledFor(logging.INFO):
        return None

    with _STATE_LOCK:
        remaining = _remaining_order(catalog, _RNG if rng is None else rng)
        if not remaining:
            return None

        opportunity = _ELIGIBLE_STARTS
        _ELIGIBLE_STARTS += 1
        if opportunity % _CADENCE != 0:
            return None

        tip = _next_tip(remaining)
        logger.info(format_tip(tip))
        _SEEN_IDS.add(tip.id)
        _LAST_FREQUENCY = tip.frequency
        return tip


def _reset_runtime_tip_state(*, rng=None):
    """Reset the session's state, for tests that need it isolated."""
    global _RNG, _ORDER, _ELIGIBLE_STARTS, _LAST_FREQUENCY
    with _STATE_LOCK:
        _RNG = random.Random() if rng is None else rng
        _ORDER = None
        _SEEN_IDS.clear()
        _ELIGIBLE_STARTS = 0
        _LAST_FREQUENCY = None


def _after_fork_in_child():
    """In a forked child, a lock and a generator of its own, a count of its
    eligible runs from zero, so that its first run is eligible for a tip, and
    a new shuffle of the tips not shown yet, so that children forked together
    draw their tips independently; the tips that the parent has shown stay
    shown."""
    global _STATE_LOCK, _RNG, _ORDER, _ELIGIBLE_STARTS
    _STATE_LOCK = threading.Lock()
    _RNG = random.Random()
    _ORDER = None
    _ELIGIBLE_STARTS = 0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
