import time


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def get_duration(self) -> float:
        return time.time() - self.start_time

    def get_duration_and_reset(self) -> float:
        delta = self.get_duration()
        self.reset()
        return delta
