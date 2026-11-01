"""Domain types shared across ingest, search and evaluation.

Frame indexing convention (matches the legacy code and the test-case labels):

* ``frame_index`` - 1-based index into the *raw* video (what the legacy loop
  calls ``frame_idx``).
* ``sample_index`` - 0-based index into the *sampled* stream (the 12 fps frames
  that are processed). This is the index used by ``output_annotated.mp4`` and
  therefore by ``evaluation.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# Names of the vectors stored per detection. One Qdrant point carries both.
OSNET = "osnet"
IRRA = "irra"


class BBox(NamedTuple):
    """Integer pixel box in ``left, top, right, bottom`` order (exclusive right/bottom)."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def clip(self, width: int, height: int) -> BBox:
        """Clamp to an image of size ``width`` x ``height``."""
        return BBox(
            max(0, self.left),
            max(0, self.top),
            min(width, self.right),
            min(height, self.bottom),
        )

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.top : self.bottom, self.left : self.right]


@dataclass(frozen=True, slots=True)
class Detection:
    """A person detection in original-frame pixel coordinates (``xywh`` floats)."""

    x: float
    y: float
    w: float
    h: float
    confidence: float
    label: str = "person"


@dataclass(frozen=True, slots=True)
class TrackState:
    """A confirmed track's box in the current frame, as reported by the tracker."""

    track_id: str
    ltrb: tuple[float, float, float, float]
    time_since_update: int = 0


@dataclass(slots=True)
class Observation:
    """One tracked person in one sampled frame.

    ``embeddings`` maps a vector name (``"osnet"``, ``"irra"``) to an
    L2-normalised vector. It is empty on frames where embeddings are not
    extracted (the pipeline embeds every N-th sampled frame).
    """

    track_id: str
    frame_index: int
    sample_index: int
    bbox: BBox
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def has_embeddings(self) -> bool:
        return bool(self.embeddings)


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
