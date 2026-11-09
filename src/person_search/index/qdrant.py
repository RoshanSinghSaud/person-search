"""Qdrant implementation of :class:`VectorIndex`.

* Local mode (``QdrantIndex.local(path)``) keeps the collection in a folder and
  needs no server: that is what runs inside the container. The same class talks
  to a Qdrant server through ``QdrantIndex(QdrantClient(url=...), ...)``.
* Distance is DOT on L2-normalised vectors (= cosine). DOT stores the vectors
  unchanged, so a vector read back is bit-identical to the one written and
  indexed search can reproduce the in-memory scores exactly. (COSINE would
  re-normalise on insert and shift scores by ~1e-7.)
* The collection name carries the encoder versions
  (``detections__osnet-v1__irra-v1``), so an index can never be searched with
  encoders it was not built with.
* In local mode Qdrant evaluates payload filters in Python, point by point
  (~250 ms for 16k points, against ~10 ms to score them all). So in local mode:

  - ``search`` ranks without a filter and keeps the hits that match it, looking
    deeper until it has ``limit`` matches or has seen every point: the same
    result as a filtered search, an order of magnitude faster;
  - ``fetch_tracks`` uses a ``(video, track) -> point IDs`` map kept in memory
    (built with one scan on first use, updated on every upsert) and retrieves by
    ID. That assumes this process is the only writer, which local mode enforces
    anyway (it locks its folder).

  Against a Qdrant server both are switched off: filters run on payload indexes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from person_search.errors import ConfigurationError
from person_search.index.base import Hit, Point, SearchFilter, TrackKey

log = logging.getLogger(__name__)

PAYLOAD_KEYWORDS = ("video_id", "track_id", "camera")


def collection_name(model_versions: dict[str, str], prefix: str = "detections") -> str:
    return "__".join([prefix, *(model_versions[k] for k in sorted(model_versions))])


def _to_filter(
    flt: SearchFilter | None, extra: list[qm.Condition] | None = None
) -> qm.Filter | None:
    must: list[qm.Condition] = list(extra or [])
    if flt is not None:
        if flt.video_ids is not None:
            must.append(
                qm.FieldCondition(key="video_id", match=qm.MatchAny(any=list(flt.video_ids)))
            )
        if flt.cameras is not None:
            must.append(qm.FieldCondition(key="camera", match=qm.MatchAny(any=list(flt.cameras))))
        if flt.t_min is not None or flt.t_max is not None:
            must.append(
                qm.FieldCondition(key="t_seconds", range=qm.Range(gte=flt.t_min, lte=flt.t_max))
            )
    return qm.Filter(must=must) if must else None


class QdrantIndex:
    def __init__(
        self,
        client: QdrantClient,
        model_versions: dict[str, str],
        dims: dict[str, int] | None = None,
        *,
        create: bool = True,
        local: bool = False,
        batch_size: int = 256,
        track_cache: bool | None = None,
    ) -> None:
        if not model_versions:
            raise ConfigurationError("QdrantIndex needs the encoder versions it stores")
        self.client = client
        self.model_versions = dict(model_versions)
        self.collection = collection_name(self.model_versions)
        self.batch_size = batch_size
        self.track_cache = local if track_cache is None else track_cache
        self.postfilter = local
        # (video, track) -> (all point IDs, IDs of points that carry vectors)
        self._tracks: dict[TrackKey, tuple[set[str], set[str]]] | None = None
        if not client.collection_exists(self.collection):
            if not create:
                existing = [c.name for c in client.get_collections().collections]
                raise ConfigurationError(
                    f"no index for {self.model_versions} (collection {self.collection!r}); "
                    f"existing: {existing}"
                )
            dims = dims or dict.fromkeys(self.model_versions, 512)
            client.create_collection(
                self.collection,
                vectors_config={
                    name: qm.VectorParams(size=dim, distance=qm.Distance.DOT)
                    for name, dim in dims.items()
                },
            )
            if not local:  # payload indexes are a server feature; local mode ignores them
                for key in PAYLOAD_KEYWORDS:
                    client.create_payload_index(self.collection, key, qm.PayloadSchemaType.KEYWORD)
                client.create_payload_index(
                    self.collection, "t_seconds", qm.PayloadSchemaType.FLOAT
                )
            log.info("created collection %s", self.collection)

    # ------------------------------------------------------------ constructors
    @classmethod
    def local(
        cls,
        path: str | Path,
        model_versions: dict[str, str],
        dims: dict[str, int] | None = None,
        *,
        create: bool = True,
    ) -> QdrantIndex:
        """On-disk local mode (one process at a time may open a folder)."""
        Path(path).mkdir(parents=True, exist_ok=True)
        return cls(QdrantClient(path=str(path)), model_versions, dims, create=create, local=True)

    @classmethod
    def in_memory(
        cls, model_versions: dict[str, str], dims: dict[str, int] | None = None
    ) -> QdrantIndex:
        return cls(QdrantClient(":memory:"), model_versions, dims, local=True)

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ writes
    def upsert(self, points: Sequence[Point]) -> None:
        for i in range(0, len(points), self.batch_size):
            chunk = points[i : i + self.batch_size]
            self.client.upsert(
                self.collection,
                points=[
                    qm.PointStruct(
                        id=p.id,
                        vector={k: v.tolist() for k, v in p.vectors.items()},
                        payload=p.payload,
                    )
                    for p in chunk
                ],
                wait=True,
            )
        if self._tracks is not None:
            for p in points:
                self._remember(self._tracks, p)

    # ------------------------------------------------------------------- reads
    def search(
        self,
        vector: str,
        query: np.ndarray,
        *,
        limit: int,
        score_threshold: float | None = None,
        flt: SearchFilter | None = None,
    ) -> list[Hit]:
        q = np.asarray(query, dtype=np.float32).ravel().tolist()
        if flt is None or not self.postfilter:
            res = self.client.query_points(
                self.collection,
                query=q,
                using=vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=_to_filter(flt),
                with_payload=True,
            )
            return [Hit(str(p.id), float(p.score), dict(p.payload or {})) for p in res.points]
        depth = max(4 * limit, 256)
        while True:
            res = self.client.query_points(
                self.collection,
                query=q,
                using=vector,
                limit=depth,
                score_threshold=score_threshold,
                with_payload=True,
            )
            hits = [
                Hit(str(p.id), float(p.score), dict(p.payload or {}))
                for p in res.points
                if flt.matches(p.payload or {})
            ]
            if len(hits) >= limit or len(res.points) < depth:
                return hits[:limit]
            depth *= 4

    def _scroll(self, flt: qm.Filter | None, with_vectors: bool | list[str]) -> list[Point]:
        out: list[Point] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=with_vectors,
                limit=4096,
                offset=offset,
            )
            for r in records:
                vecs = r.vector if isinstance(r.vector, dict) else {}
                out.append(
                    Point(
                        str(r.id),
                        dict(r.payload or {}),
                        {k: np.asarray(v, dtype=np.float32) for k, v in vecs.items()},
                    )
                )
            if offset is None:
                return out

    @staticmethod
    def _remember(tracks: dict[TrackKey, tuple[set[str], set[str]]], p: Point) -> None:
        key = (p.payload["video_id"], str(p.payload["track_id"]))
        all_ids, embedded = tracks.setdefault(key, (set(), set()))
        all_ids.add(p.id)
        if p.payload.get("embedded"):
            embedded.add(p.id)

    def _track_map(self) -> dict[TrackKey, tuple[set[str], set[str]]]:
        if self._tracks is None:
            tracks: dict[TrackKey, tuple[set[str], set[str]]] = {}
            for p in self._scroll(None, with_vectors=False):
                self._remember(tracks, p)
            self._tracks = tracks
        return self._tracks

    def fetch_tracks(
        self,
        keys: Iterable[TrackKey],
        *,
        with_vectors: bool = True,
        vectors: Sequence[str] | None = None,
        embedded_only: bool = False,
    ) -> list[Point]:
        want: bool | list[str] = list(vectors) if (with_vectors and vectors) else with_vectors
        if self.track_cache:
            track_map = self._track_map()
            empty: tuple[set[str], set[str]] = (set(), set())
            ids = sorted(
                {i for k in keys for i in track_map.get(k, empty)[1 if embedded_only else 0]}
            )
            records = self.client.retrieve(
                self.collection, ids=ids, with_payload=True, with_vectors=want
            )
            return [
                Point(
                    str(r.id),
                    dict(r.payload or {}),
                    {k: np.asarray(v, dtype=np.float32) for k, v in (r.vector or {}).items()}
                    if isinstance(r.vector, dict)
                    else {},
                )
                for r in records
            ]
        by_video: dict[str, set[str]] = {}
        for video_id, track_id in keys:
            by_video.setdefault(video_id, set()).add(track_id)
        out: list[Point] = []
        for video_id, tracks in by_video.items():
            must: list[qm.Condition] = [
                qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id)),
                qm.FieldCondition(key="track_id", match=qm.MatchAny(any=sorted(tracks))),
            ]
            if embedded_only:
                must.append(qm.FieldCondition(key="embedded", match=qm.MatchValue(value=True)))
            out.extend(self._scroll(qm.Filter(must=must), want))
        return out

    def count(self, flt: SearchFilter | None = None) -> int:
        return int(
            self.client.count(self.collection, count_filter=_to_filter(flt), exact=True).count
        )

    def videos(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for p in self._scroll(None, with_vectors=False):
            v = out.setdefault(
                p.payload["video_id"],
                {"video": p.payload.get("video"), "camera": p.payload.get("camera"), "points": 0},
            )
            v["points"] += 1
        return out
