"""Exclusive timing of the stages of a PyBADS run, for developer profiling.

``StageTimer`` is private: it has no page in the API documentation, and its
interface can change in any release. ``BADS.optimize`` creates one per run
and stores what it measured, as plain numbers, in ``optim_state`` and
``iteration_history``; the developer scripts under ``dev/scripts/`` read
them there.
"""

import math
import time

TARGET = "target"
"""The key of the pseudo-stage that holds the target's evaluations."""


class _Frame:
    """An open stage: the key its time is charged to while it is the
    innermost stage (``charge``), the prefix of the keys of the stages it
    encloses, and, for an attempt, the key its time moves to if it raises
    (``failed``), the exceptions that count as a failure and the time it
    charged."""

    __slots__ = ("charge", "prefix", "failed", "errors", "own")

    def __init__(self, charge, prefix, failed=None, errors=None):
        self.charge = charge
        self.prefix = prefix
        self.failed = failed
        self.errors = errors
        self.own = 0.0


class _Stage:
    """The context manager of ``StageTimer.stage`` and
    ``StageTimer.attempt``."""

    __slots__ = ("_timer", "_name", "_failed", "_errors", "_frame")

    def __init__(self, timer, name, failed=False, errors=None):
        self._timer = timer
        self._name = name
        self._failed = failed
        self._errors = errors
        self._frame = None

    def __enter__(self):
        self._frame = self._timer._enter(
            self._name, self._failed, self._errors
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._timer._exit(self._frame, exc_type)
        return False


class StageTimer:
    """Wall-clock time of a run, each second charged to one stage.

    Private to PyBADS: it has no page in the API documentation, and its
    interface can change in any release.

    The stages nest: ``with timer.stage(name):`` enters a stage, which
    pauses the stage that encloses it, so that each second is charged to
    the innermost stage open at that moment. The outermost stage, the
    root, opened by ``start`` and closed by ``stop``, takes the time spent
    outside every other stage. A stage's key is its path: the names of the
    stages that enclose it, the root's left out, and its own, joined by
    ``/``, as ``search/gp_rebuild/gp_fit``. The same name at two places has
    two keys. Each key holds its seconds and the number of times the stage
    was entered.

    The target's evaluations form a pseudo-stage, ``"target"``: at every
    transition the timer reads the target's cumulative evaluation time and
    charges its increase to ``"target"`` rather than to the stage that was
    open. The other stages then hold the optimizer's own time, and the
    seconds of all the keys, ``"target"`` included, add up to the time
    between ``start`` and ``stop``.

    ``with timer.attempt(name):`` marks a block whose time stays with the
    enclosing stage, unless the block raises one of the given exceptions:
    then its time moves to the key ``name`` under the enclosing stage, which
    counts one entry. The stages entered inside an attempt are keyed under
    the enclosing stage and keep their time.

    The timer never raises and asserts nothing about the times it reads (a
    target timer replaced in a test can report any value; a value that is
    not a finite number is ignored). A stage entered while the timer is not
    running is not timed. An exception that leaves a stage closes it, since
    the ``with`` statement exits it.

    Parameters
    ----------
    target_time : callable, optional
        Called without arguments, returns the target's cumulative
        evaluation time in seconds. If ``None``, ``"target"`` stays at 0.
    root : str, optional
        The name of the root stage, ``"loop"`` by default.
    """

    def __init__(self, target_time=None, root="loop"):
        self._target_time = target_time
        self._root = root
        self._stack = []
        self._seconds = {}
        self._calls = {}
        self._last = 0.0
        self._last_target = 0.0

    @property
    def running(self):
        """True between ``start`` and ``stop``."""
        return bool(self._stack)

    @property
    def depth(self):
        """The number of open stages, the root included."""
        return len(self._stack)

    def start(self):
        """Open the root stage; a timer already running is left as it
        is."""
        try:
            if self._stack:
                return
            self._last_target = self._read_target(0.0)
            self._seconds.setdefault(self._root, 0.0)
            self._seconds.setdefault(TARGET, 0.0)
            self._count(self._root)
            self._stack.append(_Frame(self._root, ""))
            self._last = time.perf_counter()
        except Exception:  # noqa: BLE001  (the timer never raises)
            pass

    def stop(self):
        """Charge the time up to now and close every open stage, the root
        included; a timer that is not running is left as it is."""
        try:
            if self._stack:
                self._charge()
        except Exception:  # noqa: BLE001
            pass
        self._stack.clear()

    def stage(self, name):
        """A context manager that times its block as the stage ``name``.

        Parameters
        ----------
        name : str
            The name of the stage, without ``/``.
        """
        return _Stage(self, name)

    def attempt(self, name, errors=Exception):
        """A context manager whose block's time moves to the stage ``name``
        if the block raises one of ``errors``.

        Parameters
        ----------
        name : str
            The name of the stage of a failed attempt, without ``/``.
        errors : type or tuple of type, optional
            The exceptions that make the attempt a failure; ``Exception``
            by default.
        """
        return _Stage(self, name, True, errors)

    def snapshot(self):
        """The seconds and the entries of every key, up to now.

        Returns
        -------
        snapshot : dict
            ``"seconds"``, a dict from each key to its seconds (a float),
            ``"target"`` included, and ``"calls"``, a dict from each key but
            ``"target"`` to the number of times its stage was entered (an
            int). Both are new dicts of plain Python numbers.
        """
        try:
            if self._stack:
                self._charge()
            return {
                "seconds": dict(self._seconds),
                "calls": dict(self._calls),
            }
        except Exception:  # noqa: BLE001
            return {"seconds": {}, "calls": {}}

    def _read_target(self, default):
        if self._target_time is None:
            return default
        try:
            value = float(self._target_time())
        except Exception:  # noqa: BLE001
            return default
        return value if math.isfinite(value) else default

    def _count(self, key):
        self._calls[key] = self._calls.get(key, 0) + 1

    def _charge(self):
        """Charge the time since the last transition: the increase of the
        target's time to ``"target"``, the rest to the innermost stage."""
        now = time.perf_counter()
        target = self._read_target(self._last_target)
        d_target = target - self._last_target
        d_own = (now - self._last) - d_target
        top = self._stack[-1]
        seconds = self._seconds
        seconds[top.charge] = seconds.get(top.charge, 0.0) + d_own
        seconds[TARGET] = seconds.get(TARGET, 0.0) + d_target
        if top.failed is not None:
            top.own += d_own
        self._last = now
        self._last_target = target

    def _enter(self, name, is_attempt, errors):
        """Open a stage (an attempt if ``is_attempt``); return its frame,
        or None when the timer is not running."""
        try:
            if not self._stack:
                return None
            self._charge()
            parent = self._stack[-1]
            if is_attempt:
                frame = _Frame(
                    parent.charge,
                    parent.prefix,
                    failed=parent.prefix + name,
                    errors=errors,
                )
            else:
                key = parent.prefix + name
                frame = _Frame(key, key + "/")
                self._count(key)
            self._stack.append(frame)
            return frame
        except Exception:  # noqa: BLE001
            return None

    def _exit(self, frame, exc_type):
        """Close the stage of ``frame``, and any stage left open inside it;
        move a failed attempt's time to its key."""
        try:
            if frame is None or not any(f is frame for f in self._stack):
                return
            self._charge()
            while self._stack.pop() is not frame:
                pass
            if (
                frame.failed is not None
                and exc_type is not None
                and issubclass(exc_type, frame.errors)
            ):
                self._seconds[frame.charge] -= frame.own
                self._seconds[frame.failed] = (
                    self._seconds.get(frame.failed, 0.0) + frame.own
                )
                self._count(frame.failed)
        except Exception:  # noqa: BLE001
            pass


class _NullStage:
    """A context manager that does nothing."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


_NULL_STAGE = _NullStage()


class NullStageTimer:
    """A stage timer that times nothing: the timer of the functions that
    take ``timer=None``, and of a ``BADS`` object's steps called outside
    ``optimize``."""

    running = False
    depth = 0

    def start(self):
        pass

    def stop(self):
        pass

    def stage(self, name):
        return _NULL_STAGE

    def attempt(self, name, errors=Exception):
        return _NULL_STAGE

    def snapshot(self):
        return {"seconds": {}, "calls": {}}


NULL_STAGE_TIMER = NullStageTimer()


def get_stage_timer(timer=None):
    """Return ``timer``, or the timer that times nothing if it is None."""
    return NULL_STAGE_TIMER if timer is None else timer
