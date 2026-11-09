"""Conversion between ingest observations and index points.

Point IDs are ``uuid5(video_id, frame_index, track_id)``: the same detection in
the same video always gets the same ID, so re-ingesting a video overwrites its
points instead of adding copies.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from person_search.index.base import Point
from person_search.ingest.pipeline import IngestResult
from person_search.types import BBox, Observation

# Fixed namespace so IDs are stable across machines and releases.
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/RoshanSinghSaud/person-search")


def video_fingerprint(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of a video file (first 16 hex chars of SHA-256)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def point_id(video_id: str, frame_index: int, track_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{video_id}:{frame_index}:{track_id}"))


def points_from_ingest(
    result: IngestResult,
    video_id: str,
    *,
    video_name: str | None = None,
    camera: str = "unknown",
) -> list[Point]:
    fps = result.video.fps or 0.0
    name = video_name or Path(result.video.path).name
    points = []
    for seq, o in enumerate(result.observations):
        payload = {
            "video_id": video_id,
            "video": name,
            "camera": camera,
            "track_id": o.track_id,
            "frame_index": o.frame_index,
            "sample_index": o.sample_index,
            "t_seconds": round(o.frame_index / fps, 3) if fps else None,
            "bbox": list(o.bbox),
            "embedded": o.has_embeddings,
            "seq": seq,  # position in the ingest output; restores its exact order
        }
        vectors = {k: np.asarray(v, dtype=np.float32) for k, v in o.embeddings.items()}
        points.append(Point(point_id(video_id, o.frame_index, o.track_id), payload, vectors))
    return points


def observation_from_point(point: Point) -> Observation:
    p = point.payload
    return Observation(
        track_id=str(p["track_id"]),
        frame_index=int(p["frame_index"]),
        sample_index=int(p["sample_index"]),
        bbox=BBox(*(int(v) for v in p["bbox"])),
        embeddings=dict(point.vectors),
    )


def observations_by_video(points: Iterable[Point]) -> dict[str, list[Observation]]:
    """Group points by video and turn them back into observations.

    Observations come back in the order the ingest produced them (``seq``), so
    tracks are grouped in order of first appearance, exactly as in the
    in-memory search (tied track scores keep that order).
    """
    out: dict[str, list[Observation]] = {}
    for pt in sorted(points, key=lambda p: (p.payload["video_id"], p.payload["seq"])):
        out.setdefault(pt.payload["video_id"], []).append(observation_from_point(pt))
    return out
