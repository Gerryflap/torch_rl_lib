import time


class Timer:
    """
        Simple timer class for easily keeping track of time
    """
    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        # Reset timer start time to current time
        self.start_time = time.time()

    def get_duration(self) -> float:
        # Get elapsed time since timer reset
        return time.time() - self.start_time

    def get_duration_and_reset(self) -> float:
        # Get elapsed time since timer reset, then reset the timer
        delta = self.get_duration()
        self.reset()
        return delta
