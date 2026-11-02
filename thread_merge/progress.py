"""Progress reporting for the batched pipeline stages."""

from __future__ import annotations

import logging
import time

log = logging.getLogger("thread_merge")


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


class Progress:
    """Logs "label: done/total (pct), ~time left" after every step of a stage.

    The estimate is the average time per step so far times the steps left, so it includes
    throttling and retry waits. Cached responses make early steps fast, so the estimate can
    rise once uncached requests start.
    """

    def __init__(self, label: str, total: int, unit: str = "batches") -> None:
        self.label = label
        self.total = total
        self.unit = unit
        self.done = 0
        self._start = time.monotonic()

    def step(self, n: int = 1) -> None:
        self.done += n
        elapsed = time.monotonic() - self._start
        left = max(self.total - self.done, 0)
        pct = 100 * self.done // self.total if self.total else 100
        if left:
            eta = elapsed / self.done * left
            log.info("%s: %d/%d %s (%d%%), ~%s left", self.label, self.done, self.total, self.unit, pct,
                     format_duration(eta))
        else:
            log.info("%s: %d/%d %s done in %s", self.label, self.done, self.total, self.unit,
                     format_duration(elapsed))
