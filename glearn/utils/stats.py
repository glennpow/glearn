import time
import numpy as np


class RunningAverage(object):
    """
    This class keeps a running average of values, with an optional sliding-window.
    :param window: Either an int specifying how many of the most recent values should be
        averaged, or None (default), which will average over an infinite horizon.
    """

    def __init__(self, window=None):
        self.window = window

        self._count = 0
        self._value = None
        if self.window is None:
            self._average = 0
        else:
            assert isinstance(window, int)
            self._buffer = [0] * window

    def add(self, value):
        # add another value to be averaged
        if self.window is None:
            if self._count > 0:
                self._average = (self._average * self._count + value) / (self._count + 1)
            else:
                self._average = value
        else:
            index = self._count % self.window
            self._buffer[index] = value
        self._count += 1

        # update current average
        if self.window is None:
            self._value = self._average
        else:
            effective_count = min(self._count, self.window)
            self._value = np.mean(self._buffer[:effective_count])

    @property
    def count(self):
        # get number of values averaged
        return self._count

    @property
    def value(self):
        return self._value


class RunningTimer(RunningAverage):
    """
    This class keeps a running average of time intervals (deltas).  You can also query the
        rate_per_second using this average.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_time = None

    def tick(self):
        # call this each time you want to mark a time interval
        t = time.time()
        if self._last_time is not None:
            self.add(t - self._last_time)
        self._last_time = t

    @property
    def rate_per_second(self):
        # returns rate per second using average time intervals
        return 1 / self._value if self._value else None
