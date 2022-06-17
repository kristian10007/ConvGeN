import time


class timing:
    def __init__(self, name="?"):
        self.name = name
        self.duration = 0.0
        self.startTime = None
        self.runCount = 0

    def start(self):
        self.startTime = time.process_time()

    def stop(self):
        if self.startTime is not None:
            self.duration += time.process_time() - self.startTime
            self.runCount += 1
        self.startTime = None

    def __str__(self):
        return f"{self.name}: #{self.runCount} {self.duration:.4f}s"
