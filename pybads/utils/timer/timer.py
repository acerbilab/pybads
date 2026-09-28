import time


class Timer:
    """
    A small Timer class used to time the different parts of PyBADS.

    It measures with ``time.perf_counter``, the clock of the highest
    resolution, where ``time.time`` has a resolution of about 15.6 ms on
    Windows before Python 3.13, so that it timed most evaluations of a fast
    target as 0.
    """

    def __init__(self, eps_t=1e-9):
        """
        Initialize a new timer.
        """
        self._start_times = dict()
        self._durations = dict()
        self.eps_t = eps_t

    def start_timer(self, name: str):
        """
        Start the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """
        if name not in self._start_times:
            if name in self._durations:
                self._durations.pop(name)
            self._start_times[name] = time.perf_counter()

    def stop_timer(self, name: str):
        """
        Stop the specified timer

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """

        if name in self._start_times:
            end_time = time.perf_counter()
            self._durations[name] = (
                end_time - self._start_times[name]
            ) + self.eps_t
            self._start_times.pop(name)

    def get_duration(self, name: str):
        """
        Return the duration of the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer which time should be returned.

        Returns
        -------
        duration : float
            The duration of the timer or None when the timer is not existing.
        """
        return self._durations.get(name)
