"""Per-stage wall-clock accounting (decode, detect, track, embed ...).

Cheap enough to leave on in production; Day 13 profiles on top of it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - t0
            self.calls[name] += 1

    @property
    def total(self) -> float:
        return sum(self.seconds.values())

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {"seconds": round(self.seconds[name], 4), "calls": self.calls[name]}
            for name in sorted(self.seconds, key=self.seconds.__getitem__, reverse=True)
        }
