"""Vector index: one point per tracked detection, two named vectors (Day 10).

* :mod:`~person_search.index.base`   - ``VectorIndex``, ``Point``, ``Hit``, ``SearchFilter``
* :mod:`~person_search.index.points` - ingest observations <-> points, uuid5 point IDs
* :mod:`~person_search.index.memory` - in-memory fake for tests
* :mod:`~person_search.index.qdrant` - Qdrant (local mode or server)
"""

from person_search.index.base import Hit, Point, SearchFilter, VectorIndex
from person_search.index.memory import InMemoryIndex
from person_search.index.points import point_id, points_from_ingest, video_fingerprint

__all__ = [
    "Hit",
    "InMemoryIndex",
    "Point",
    "SearchFilter",
    "VectorIndex",
    "point_id",
    "points_from_ingest",
    "video_fingerprint",
]
