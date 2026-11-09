"""Search over the vector index: retrieve candidates, then rerank whole tracks.

Two stages for every query type:

1. **Retrieve** detections from the index by one vector (``osnet`` for photos,
   ``irra`` for text, both for hybrid) and collect the tracks they belong to.
2. **Rerank** by fetching *every* point of those tracks and running the same
   scoring as the in-memory search (``search/photo.py``, ``search/text.py``,
   ``search/hybrid.py``).

Each stage-1 cut-off comes with a bound that proves stage 2 sees every track
that could change the answer; if the bound does not hold, the cut-off grows
and retrieval repeats. So indexed search returns exactly what scanning the
whole video returns (``tests/unit/test_indexed_search.py`` and the Day 11/12
runs check this on the 4 test cases):

* **photo** - the paper accepts a track only if its top-50% mean is at least
  tau_base, which needs at least one detection scoring >= tau_base. Retrieval
  takes every detection above tau_base (minus a 1e-4 float margin), so every
  track that can be accepted is a candidate. A few extra top tracks are added so
  "no match" answers still show the closest tracks.
* **text** - a track's top-50% mean never exceeds its best detection. A track
  with no detection among the top N therefore scores below the N-th hit, so the
  top-k is exact once the k-th candidate track scores at least that much.
* **hybrid** - min-max normalisation needs each modality's min and max over the
  searched scope: the max is the top hit, the min is the top hit of the negated
  query. A track with no detection in either top-N list scores below
  ``alpha*norm(p_N) + (1-alpha)*norm(t_N)``, which gives the same stopping rule.
  (z-score normalisation needs the whole score distribution, so it scans.)

Results are grouped per video: the paper's dynamic threshold is defined for one
query against one video, so a photo query over several videos applies it to
each video separately.

Photo queries fetch every point of a candidate track (frame selection
forward-fills over frames without vectors) but only the ``osnet`` vector; text
and hybrid queries fetch only the points that carry vectors, and only the
vectors they score.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from person_search.config import FusionSettings, ScoringSettings
from person_search.core.scoring import FrameSim, TrackScore, rank_tracks
from person_search.index.base import Hit, SearchFilter, TrackKey, VectorIndex
from person_search.index.points import observations_by_video
from person_search.search.photo import as_query, photo_search, similarities
from person_search.search.results import PhotoSearchResult
from person_search.types import IRRA, OSNET, Observation

FLOAT_MARGIN = 1e-4


@dataclass(frozen=True, slots=True)
class RankedTrack:
    """A text/hybrid result. ``n_frames`` counts the track's scored (embedded) frames."""

    video_id: str
    track_id: str
    score: float
    n_frames: int
    n_scored: int


@dataclass
class SearchStats:
    retrieved_points: int = 0
    candidate_tracks: int = 0
    fetched_points: int = 0
    rounds: int = 0
    ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.ms.values())


@dataclass
class IndexedPhotoResult:
    per_video: dict[str, PhotoSearchResult]
    stats: SearchStats

    @property
    def matched(self) -> bool:
        return any(r.matched for r in self.per_video.values())


def _keys(hits: Iterable[Hit]) -> set[TrackKey]:
    return {(h.payload["video_id"], str(h.payload["track_id"])) for h in hits}


class _Clock:
    def __init__(self, stats: SearchStats) -> None:
        self.stats = stats
        self.t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.stats.ms[name] = self.stats.ms.get(name, 0.0) + (now - self.t) * 1000
        self.t = now


class IndexedSearch:
    def __init__(
        self,
        index: VectorIndex,
        scoring: ScoringSettings | None = None,
        fusion: FusionSettings | None = None,
        *,
        candidate_limit: int = 256,
        context_tracks: int = 10,
        max_limit: int = 1 << 20,
    ) -> None:
        self.index = index
        self.scoring = scoring or ScoringSettings()
        self.fusion = fusion or FusionSettings()
        self.candidate_limit = candidate_limit
        self.context_tracks = context_tracks
        self.max_limit = max_limit

    # ---------------------------------------------------------------- helpers
    def _fetch(
        self,
        keys: set[TrackKey],
        stats: SearchStats,
        vectors: Sequence[str],
        embedded_only: bool = False,
    ) -> dict[str, list[Observation]]:
        points = self.index.fetch_tracks(keys, vectors=vectors, embedded_only=embedded_only)
        stats.candidate_tracks = len(keys)
        stats.fetched_points = len(points)
        return observations_by_video(points)

    # ------------------------------------------------------------------ photo
    def photo(self, query: np.ndarray, flt: SearchFilter | None = None) -> IndexedPhotoResult:
        """The paper's photo query (top-50% score, dynamic threshold) per video."""
        q = as_query(query)
        stats = SearchStats()
        clock = _Clock(stats)
        floor = self.scoring.base_threshold - FLOAT_MARGIN
        limit = self.candidate_limit
        while True:
            stats.rounds += 1
            above = self.index.search(OSNET, q, limit=limit, score_threshold=floor, flt=flt)
            if len(above) < limit or limit >= self.max_limit:
                break
            limit *= 4
        context = self.index.search(OSNET, q, limit=self.context_tracks, flt=flt)
        stats.retrieved_points = len(above) + len(context)
        clock.lap("retrieve")
        by_video = self._fetch(_keys(above) | _keys(context), stats, [OSNET])
        clock.lap("fetch")
        per_video = {
            vid: photo_search(obs, q, self.scoring, OSNET) for vid, obs in by_video.items()
        }
        clock.lap("score")
        return IndexedPhotoResult(per_video, stats)

    # ------------------------------------------------------------------- text
    def text(
        self, query: np.ndarray, top_k: int = 5, flt: SearchFilter | None = None
    ) -> tuple[list[RankedTrack], SearchStats]:
        """Tracks ranked by top-50% IRRA score; no threshold (a person reviews)."""
        q = as_query(query)
        stats = SearchStats()
        clock = _Clock(stats)
        limit = self.candidate_limit
        while True:
            stats.rounds += 1
            hits = self.index.search(IRRA, q, limit=limit, flt=flt)
            clock.lap("retrieve")
            by_video = self._fetch(_keys(hits), stats, [IRRA], embedded_only=True)
            clock.lap("fetch")
            ranked = self._rank({vid: similarities(obs, q, IRRA) for vid, obs in by_video.items()})
            clock.lap("score")
            exhausted = len(hits) < limit or limit >= self.max_limit
            if exhausted or (len(ranked) >= top_k and ranked[top_k - 1].score >= hits[-1].score):
                stats.retrieved_points = len(hits)
                return ranked[:top_k], stats
            limit *= 4

    # ----------------------------------------------------------------- hybrid
    def hybrid(
        self,
        photo_query: np.ndarray,
        text_query: np.ndarray,
        top_k: int = 5,
        flt: SearchFilter | None = None,
        alpha: float | None = None,
    ) -> tuple[list[RankedTrack], SearchStats]:
        """Late fusion of min-max-normalised OSNet and IRRA scores per detection."""
        if self.fusion.normalization != "minmax":
            raise NotImplementedError("indexed hybrid search supports minmax normalisation only")
        a = self.fusion.alpha if alpha is None else alpha
        p, t = as_query(photo_query), as_query(text_query)
        stats = SearchStats()
        clock = _Clock(stats)
        # Range of each modality's scores over the searched scope.
        ranges = {}
        for name, q in ((OSNET, p), (IRRA, t)):
            top = self.index.search(name, q, limit=1, flt=flt)
            bottom = self.index.search(name, -q, limit=1, flt=flt)
            if not top:
                return [], stats
            ranges[name] = (-bottom[0].score, top[0].score)
        clock.lap("retrieve")

        def norm(name: str, s: float) -> float:
            lo, hi = ranges[name]
            return 0.0 if hi - lo < 1e-12 else (s - lo) / (hi - lo)

        limit = self.candidate_limit
        while True:
            stats.rounds += 1
            hp = self.index.search(OSNET, p, limit=limit, flt=flt)
            ht = self.index.search(IRRA, t, limit=limit, flt=flt)
            clock.lap("retrieve")
            by_video = self._fetch(_keys(hp) | _keys(ht), stats, [OSNET, IRRA], embedded_only=True)
            clock.lap("fetch")
            per_video: dict[str, dict[str, list[FrameSim]]] = {}
            for vid, obs in by_video.items():
                both = [o for o in obs if OSNET in o.embeddings and IRRA in o.embeddings]
                if not both:
                    continue
                sp = np.array([np.dot(p, o.embeddings[OSNET]) for o in both])
                st = np.array([np.dot(t, o.embeddings[IRRA]) for o in both])
                fused = a * np.array([norm(OSNET, s) for s in sp]) + (1 - a) * np.array(
                    [norm(IRRA, s) for s in st]
                )
                tracks: dict[str, list[FrameSim]] = {}
                for o, s in zip(both, fused, strict=True):
                    tracks.setdefault(o.track_id, []).append((o.frame_index, float(s)))
                per_video[vid] = tracks
            ranked = self._rank(per_video)
            clock.lap("score")
            # One exhausted list already holds every detection, so every track is a candidate.
            exhausted = len(hp) < limit or len(ht) < limit or limit >= self.max_limit
            bound = a * norm(OSNET, hp[-1].score) + (1 - a) * norm(IRRA, ht[-1].score)
            if exhausted or (len(ranked) >= top_k and ranked[top_k - 1].score >= bound):
                stats.retrieved_points = len(hp) + len(ht)
                return ranked[:top_k], stats
            limit *= 4

    def _rank(self, per_video: dict[str, dict[str, list[FrameSim]]]) -> list[RankedTrack]:
        out = []
        for vid, tracks in per_video.items():
            for ts in rank_tracks(tracks, self.scoring.top_fraction):
                out.append(RankedTrack(vid, ts.track_id, ts.score, ts.n_frames, ts.n_scored))
        # stable: ties keep per-video first-appearance order
        return sorted(out, key=lambda r: r.score, reverse=True)


def ranked_from_scores(video_id: str, scores: list[TrackScore]) -> list[RankedTrack]:
    return [RankedTrack(video_id, s.track_id, s.score, s.n_frames, s.n_scored) for s in scores]


__all__ = [
    "IndexedPhotoResult",
    "IndexedSearch",
    "RankedTrack",
    "SearchStats",
    "ranked_from_scores",
]
