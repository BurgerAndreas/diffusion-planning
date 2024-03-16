import time

class Timer:

    def __init__(self):
        self.init_time = time.time()
        self._start = time.time() # overwrite this to reset the timer

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff

    def time_passed_since_init(self):
        return time.time() - self.init_time
