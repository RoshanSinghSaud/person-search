"""The vector index interface and the types that cross it.

One *point* per tracked person per sampled frame (an :class:`Observation`).
Points on embedding frames (every 3rd sampled frame) carry two named vectors,
``osnet`` and ``irra``; the others carry none. The vector-less points are kept
because the paper's frame selection forward-fills similarities over every frame
of a track, so indexed search needs them to return the same frames as the
legacy pipeline.

Payload per point::

    video_id     content hash of the video (16 hex chars)
    video        file name, for display
    camera       camera label (free text; "unknown" when not given)
    track_id     tracker ID, unique within a video only
    frame_index  1-based raw frame index
    sample_index 0-based index in the 12 fps sampled stream
    t_seconds    frame_index / video fps
    bbox         [left, top, right, bottom] in pixels
    embedded     True when the point carries vectors

Implementations: :class:`~person_search.index.memory.InMemoryIndex` (tests) and
:class:`~person_search.index.qdrant.QdrantIndex` (local mode on disk, or a
server).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Point:
    id: str
    payload: dict[str, Any]
    vectors: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """Restrict a search to some videos, cameras or a time window (seconds)."""

    video_ids: tuple[str, ...] | None = None
    cameras: tuple[str, ...] | None = None
    t_min: float | None = None
    t_max: float | None = None

    def matches(self, payload: dict[str, Any]) -> bool:
        if self.video_ids is not None and payload.get("video_id") not in self.video_ids:
            return False
        if self.cameras is not None and payload.get("camera") not in self.cameras:
            return False
        t = payload.get("t_seconds")
        if self.t_min is not None and (t is None or t < self.t_min):
            return False
        return not (self.t_max is not None and (t is None or t > self.t_max))


TrackKey = tuple[str, str]  # (video_id, track_id)


@runtime_checkable
class VectorIndex(Protocol):
    """What search needs from a vector store."""

    #: encoder versions whose vectors this index holds, e.g. {"osnet": "osnet-v1", ...}
    model_versions: dict[str, str]

    def upsert(self, points: Sequence[Point]) -> None:
        """Insert or overwrite points by ID (re-ingesting a video adds nothing)."""
        ...

    def search(
        self,
        vector: str,
        query: np.ndarray,
        *,
        limit: int,
        score_threshold: float | None = None,
        flt: SearchFilter | None = None,
    ) -> list[Hit]:
        """Points with the highest dot product on ``vector``, best first."""
        ...

    def fetch_tracks(self, keys: Iterable[TrackKey], *, with_vectors: bool = True) -> list[Point]:
        """Every point of the given tracks (embedded or not)."""
        ...

    def count(self, flt: SearchFilter | None = None) -> int: ...

    def videos(self) -> dict[str, dict[str, Any]]:
        """``{video_id: {"video": name, "camera": ..., "points": n}}``."""
        ...
