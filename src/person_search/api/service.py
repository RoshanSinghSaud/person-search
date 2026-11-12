"""What the routes call: models and the index, loaded once and shared by every request.

``SearchService`` wraps the Week 2 pieces without changing them: queries go
through :class:`~person_search.search.indexed.IndexedSearch` (exactly a full
scan, decision 0008) and ingest through
:class:`~person_search.ingest.pipeline.IngestPipeline`, so the API returns the
same answers as ``scripts/eval_search.py``.

Concurrency: requests run in FastAPI's thread pool and ingest in one worker
thread (``jobs.py``). Encoders are only read, so they are shared. The index is
guarded by one lock: Qdrant local mode is not thread-safe, and ``QdrantIndex``
keeps a track map that ``upsert`` updates. Ingest holds the lock only while it
writes its points, not while it decodes and embeds.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from person_search import __version__
from person_search.api.schemas import (
    HealthResponse,
    IndexInfo,
    PhotoVideoResult,
    SearchResponse,
    SegmentOut,
    TrackOut,
    VideoOut,
)
from person_search.config import Settings
from person_search.errors import ConfigurationError, QueryError
from person_search.index.base import SearchFilter, VectorIndex
from person_search.index.points import points_from_ingest
from person_search.ingest.pipeline import IngestPipeline
from person_search.ingest.video import probe, sample_schedule
from person_search.interfaces import ImageEncoder, TextEncoder
from person_search.search.indexed import IndexedSearch, RankedTrack, SearchStats
from person_search.types import IRRA, OSNET

log = logging.getLogger(__name__)


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _preview(values: list[str], n: int = 10) -> str:
    more = f" (+{len(values) - n} more)" if len(values) > n else ""
    return ", ".join(values[:n]) + more if values else "(none)"


class SearchService:
    def __init__(
        self,
        settings: Settings,
        photo_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        index: VectorIndex,
        new_ingest: Callable[[], IngestPipeline],
    ) -> None:
        expected = {OSNET: photo_encoder.version, IRRA: text_encoder.version}
        if dict(index.model_versions) != expected:
            raise ConfigurationError(
                f"index holds vectors of {index.model_versions}, "
                f"but the loaded encoders are {expected}: rebuild the index or fix the versions"
            )
        self.settings = settings
        self.photo_encoder = photo_encoder
        self.text_encoder = text_encoder
        self.index = index
        self.searcher = IndexedSearch(index, settings.scoring, settings.fusion)
        self._new_ingest = new_ingest
        self._lock = threading.Lock()  # the index
        self._catalog_lock = threading.Lock()  # _videos, read by requests, written by ingest
        with self._lock:
            self._videos: dict[str, dict[str, Any]] = index.videos()
        log.info(
            "search service ready: %s, %d videos, %d points",
            expected,
            len(self._videos),
            self.n_points,
        )

    # ------------------------------------------------------------------ state
    @property
    def model_versions(self) -> dict[str, str]:
        return dict(self.index.model_versions)

    def _catalog(self) -> dict[str, dict[str, Any]]:
        with self._catalog_lock:
            return dict(self._videos)

    @property
    def n_points(self) -> int:
        return sum(int(v["points"]) for v in self._catalog().values())

    def has_video(self, video_id: str) -> bool:
        return video_id in self._catalog()

    def videos(self) -> list[VideoOut]:
        return [
            VideoOut(video_id=vid, video=v.get("video"), camera=v.get("camera"), points=v["points"])
            for vid, v in sorted(self._catalog().items(), key=lambda kv: str(kv[1].get("video")))
        ]

    def filter_errors(
        self, video_ids: tuple[str, ...] | None, cameras: tuple[str, ...] | None
    ) -> list[str]:
        """What a filter names that the index does not hold (a typo would otherwise
        return an empty result that looks like "nobody found")."""
        catalog = self._catalog()
        errors = []
        unknown = [v for v in video_ids or () if v not in catalog]
        if unknown:
            errors.append(
                f"unknown video_ids {unknown}; indexed: {_preview(sorted(catalog))} (GET /videos)"
            )
        known = sorted({str(v.get("camera")) for v in catalog.values()})
        unknown = [c for c in cameras or () if c not in known]
        if unknown:
            errors.append(f"unknown cameras {unknown}; indexed: {_preview(known)}")
        return errors

    def health(self, jobs: dict[str, int]) -> HealthResponse:
        return HealthResponse(
            version=__version__,
            models=self.model_versions,
            index=IndexInfo(
                collection=getattr(self.index, "collection", None),
                points=self.n_points,
                videos=len(self._catalog()),
            ),
            jobs=jobs,
        )

    def close(self) -> None:
        close = getattr(self.index, "close", None)
        if callable(close):
            close()

    # ----------------------------------------------------------------- search
    def search(
        self,
        *,
        photo: np.ndarray | None = None,
        description: str | None = None,
        top_k: int = 5,
        alpha: float | None = None,
        flt: SearchFilter | None = None,
    ) -> SearchResponse:
        if photo is None and not description:
            raise QueryError("give a photo, a description or both")
        timing: dict[str, float] = {}
        t0 = time.perf_counter()
        p = t = None
        if photo is not None:
            p = self.photo_encoder.encode_images([photo])[0]
            timing["encode_photo"] = _ms(time.perf_counter() - t0)
        if description:
            t1 = time.perf_counter()
            t = self.text_encoder.encode_texts([description])[0]
            timing["encode_text"] = _ms(time.perf_counter() - t1)

        if t is None:
            assert p is not None
            out = self._photo(p, top_k, flt)
        elif p is None:
            with self._lock:
                ranked, stats = self.searcher.text(t, top_k=top_k, flt=flt)
            out = self._ranked("text", ranked, stats)
        else:
            a = self.settings.fusion.alpha if alpha is None else alpha
            with self._lock:
                ranked, stats = self.searcher.hybrid(p, t, top_k=top_k, flt=flt, alpha=a)
            out = self._ranked("hybrid", ranked, stats)
            out.alpha = a
        out.timing_ms = {**timing, **out.timing_ms, "total": _ms(time.perf_counter() - t0)}
        return out

    def _ranked(
        self, mode: Literal["text", "hybrid"], ranked: list[RankedTrack], stats: SearchStats
    ) -> SearchResponse:
        return SearchResponse(
            mode=mode,
            model_versions=self.model_versions,
            tracks=[self._track(r) for r in ranked],
            timing_ms=self._stats_ms(stats),
        )

    def _photo(self, query: np.ndarray, top_k: int, flt: SearchFilter | None) -> SearchResponse:
        with self._lock:
            res = self.searcher.photo(query, flt)
        catalog = self._catalog()
        videos: list[PhotoVideoResult] = []
        tracks: list[TrackOut] = []
        for vid, r in res.per_video.items():
            meta = catalog.get(vid, {})
            best = r.decision.best
            videos.append(
                PhotoVideoResult(
                    video_id=vid,
                    video=meta.get("video"),
                    camera=meta.get("camera"),
                    matched=r.matched,
                    best_score=best.score if best else None,
                    effective_threshold=r.decision.effective_threshold,
                    accepted_tracks=list(r.decision.accepted),
                    selected_frames=len(r.hits),
                    segments=[
                        SegmentOut(
                            start_frame=s.start_frame,
                            end_frame=s.end_frame,
                            start_sample=s.start_sample,
                            end_sample=s.end_sample,
                            n_samples=s.n_samples,
                        )
                        for s in r.segments()
                    ],
                )
            )
            accepted = set(r.decision.accepted)
            tracks.extend(
                TrackOut(
                    video_id=vid,
                    video=meta.get("video"),
                    camera=meta.get("camera"),
                    track_id=ts.track_id,
                    score=ts.score,
                    n_frames=ts.n_scored,
                    accepted=ts.track_id in accepted,
                )
                for ts in r.decision.ranked
            )
        videos.sort(key=lambda v: (v.matched, v.best_score or 0.0), reverse=True)
        tracks.sort(key=lambda tr: tr.score, reverse=True)
        return SearchResponse(
            mode="photo",
            model_versions=self.model_versions,
            tracks=tracks[:top_k],
            matched=res.matched,
            videos=videos,
            timing_ms=self._stats_ms(res.stats),
        )

    def _track(self, r: RankedTrack) -> TrackOut:
        meta = self._catalog().get(r.video_id, {})
        return TrackOut(
            video_id=r.video_id,
            video=meta.get("video"),
            camera=meta.get("camera"),
            track_id=r.track_id,
            score=r.score,
            n_frames=r.n_scored,
        )

    @staticmethod
    def _stats_ms(stats: SearchStats) -> dict[str, float]:
        return {k: round(v, 1) for k, v in stats.ms.items()}

    # ----------------------------------------------------------------- ingest
    def ingest(
        self,
        path: str | Path,
        *,
        video_id: str,
        name: str,
        camera: str,
        progress: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Run the ingest pipeline on one video and write its points. Called by the job thread."""
        if self.has_video(video_id):  # an identical upload finished while this one queued
            return self.indexed_summary(video_id)
        t0 = time.perf_counter()
        sampling = self.settings.sampling
        info = probe(path, sampling.fallback_fps)
        expected = max(1, len(sample_schedule(info.frame_count, info.fps, sampling.target_fps)))

        def on_progress(n_samples: int, _frame_count: int) -> None:
            if progress is not None:
                progress(min(n_samples / expected, 0.99))

        result = self._new_ingest().run(path, on_progress=on_progress)
        if result.encoder_versions != self.model_versions:
            raise ConfigurationError(
                f"ingest produced {result.encoder_versions}, index holds {self.model_versions}"
            )
        points = points_from_ingest(result, video_id, video_name=name, camera=camera)
        t_write = time.perf_counter()
        with self._lock:
            self.index.upsert(points)
            with self._catalog_lock:
                self._videos[video_id] = {"video": name, "camera": camera, "points": len(points)}
        log.info("indexed %s (%s): %d points", name, video_id, len(points))
        return {
            "video_id": video_id,
            "skipped": False,
            "points": len(points),
            "embedded_points": sum(bool(pt.vectors) for pt in points),
            "tracks": len(result.track_ids),
            "samples": result.n_samples,
            "ingest_seconds": round(result.wall_seconds, 2),
            "write_seconds": round(time.perf_counter() - t_write, 2),
            "total_seconds": round(time.perf_counter() - t0, 2),
        }

    def indexed_summary(self, video_id: str) -> dict[str, Any]:
        return {
            "video_id": video_id,
            "skipped": True,
            "points": self._catalog()[video_id]["points"],
            "reason": "this video is already indexed",
        }


def open_index(path: Path, versions: dict[str, str], dims: dict[str, int]) -> VectorIndex:
    """Open the index for these model versions in a Qdrant local-mode folder.

    An empty (or new) folder gets an empty index. A folder that holds an index
    for other versions is refused: serving would return nothing, and silently
    starting a second collection next to the old one hides that it needs a rebuild.
    """
    from qdrant_client import QdrantClient

    from person_search.index.qdrant import QdrantIndex, collection_name

    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))
    existing = sorted(c.name for c in client.get_collections().collections)
    wanted = collection_name(versions)
    if existing and wanted not in existing:
        client.close()
        raise ConfigurationError(
            f"{path} holds {existing}, not {wanted!r}: rebuild the index for {versions} "
            "or set PS_API__INDEX_PATH to an empty folder"
        )
    return QdrantIndex(client, versions, dims, local=True)


def build_service(settings: Settings) -> SearchService:
    """Load YOLO, DeepSORT, OSNet and IRRA, and open (or create) the index."""
    from person_search.factory import build_components

    components = build_components(settings, encoders=[OSNET, IRRA])
    osnet, irra = components.osnet, components.irra
    index = open_index(
        settings.api.index_path,
        {OSNET: osnet.version, IRRA: irra.version},
        {OSNET: osnet.dim, IRRA: irra.dim},
    )
    return SearchService(settings, osnet, irra, index, components.ingest_pipeline)
