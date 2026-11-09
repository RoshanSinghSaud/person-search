"""In-memory :class:`VectorIndex` for tests: a dict and brute-force dot products."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from person_search.index.base import Hit, Point, SearchFilter, TrackKey


class InMemoryIndex:
    def __init__(self, model_versions: dict[str, str] | None = None) -> None:
        self.model_versions = dict(model_versions or {})
        self._points: dict[str, Point] = {}

    def upsert(self, points: Sequence[Point]) -> None:
        for p in points:
            self._points[p.id] = p

    def search(
        self,
        vector: str,
        query: np.ndarray,
        *,
        limit: int,
        score_threshold: float | None = None,
        flt: SearchFilter | None = None,
    ) -> list[Hit]:
        q = np.asarray(query, dtype=np.float32).ravel()
        cands = [
            p
            for p in self._points.values()
            if vector in p.vectors and (flt is None or flt.matches(p.payload))
        ]
        if not cands:
            return []
        scores = np.stack([p.vectors[vector] for p in cands]) @ q
        order = np.argsort(-scores, kind="stable")[:limit]
        hits = [Hit(cands[i].id, float(scores[i]), dict(cands[i].payload)) for i in order]
        if score_threshold is not None:
            hits = [h for h in hits if h.score >= score_threshold]
        return hits

    def fetch_tracks(
        self,
        keys: Iterable[TrackKey],
        *,
        with_vectors: bool = True,
        vectors: Sequence[str] | None = None,
        embedded_only: bool = False,
    ) -> list[Point]:
        wanted = set(keys)
        out = []
        for p in self._points.values():
            if (p.payload["video_id"], p.payload["track_id"]) not in wanted:
                continue
            if embedded_only and not p.payload.get("embedded"):
                continue
            if not with_vectors:
                out.append(Point(p.id, p.payload))
            elif vectors:
                out.append(
                    Point(p.id, p.payload, {k: v for k, v in p.vectors.items() if k in vectors})
                )
            else:
                out.append(p)
        return out

    def count(self, flt: SearchFilter | None = None) -> int:
        return sum(1 for p in self._points.values() if flt is None or flt.matches(p.payload))

    def videos(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for p in self._points.values():
            v = out.setdefault(
                p.payload["video_id"],
                {"video": p.payload.get("video"), "camera": p.payload.get("camera"), "points": 0},
            )
            v["points"] += 1
        return out
