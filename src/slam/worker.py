"""Keyframe queue processed inline (deterministic) or on a background thread (ORB-SLAM style)."""

from __future__ import annotations

import threading
import traceback
from collections import deque


class Worker:
    def __init__(self, threaded: bool):
        self.queue: deque = deque()
        self.busy = False
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True) if threaded else None
        if self._thread is not None:
            self._thread.start()

    def process(self, kf) -> None:
        raise NotImplementedError

    def insert(self, kf) -> None:
        with self._cv:
            self.queue.append(kf)
            self._cv.notify()
        if self._thread is None:
            while self.queue and not self.busy:
                self.busy = True
                try:
                    self.process(self.queue.popleft())
                finally:
                    self.busy = False

    def close(self) -> None:
        """Drain the queue and stop the thread."""
        if self._thread is None:
            return
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self.queue and not self._stop:
                    self._cv.wait()
                if not self.queue:
                    return
                kf = self.queue.popleft()
                self.busy = True
            try:
                self.process(kf)
            except Exception:
                traceback.print_exc()  # a dead worker would silently stall the map
            finally:
                self.busy = False
