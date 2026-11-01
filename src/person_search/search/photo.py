"""Photo query: the paper's top-50% track score and dynamic threshold.

Takes observations already in memory. Day 11 feeds it the candidate tracks
retrieved from Qdrant instead of a whole video; the scoring does not change,
which is what the "indexed search == pipeline.run" test checks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from person_search.config import ScoringSettings
from person_search.core.scoring import FrameSim, decide, rank_tracks, select_frames
from person_search.errors import QueryError
from person_search.search.results import FrameHit, PhotoSearchResult
from person_search.types import OSNET, Observation


def _group(observations: Iterable[Observation]) -> dict[str, list[Observation]]:
    out: dict[str, list[Observation]] = {}
    for obs in observations:
        out.setdefault(obs.track_id, []).append(obs)
    return out


def as_query(vec: np.ndarray) -> np.ndarray:
    q = np.asarray(vec, dtype=np.float32).ravel()
    if q.size == 0 or not np.all(np.isfinite(q)):
        raise QueryError("query vector is empty or not finite")
    return q


def similarities(
    observations: Iterable[Observation], query: np.ndarray, vector: str
) -> dict[str, list[FrameSim]]:
    """Per-track ``(frame_index, sim | None)`` lists; sim = dot(query, vector)."""
    q = as_query(query)
    out: dict[str, list[FrameSim]] = {}
    for tid, obs_list in _group(observations).items():
        out[tid] = [
            (
                o.frame_index,
                float(np.dot(q, o.embeddings[vector])) if vector in o.embeddings else None,
            )
            for o in obs_list
        ]
    return out


def photo_search(
    observations: Sequence[Observation],
    query: np.ndarray,
    scoring: ScoringSettings | None = None,
    vector: str = OSNET,
) -> PhotoSearchResult:
    """The paper's photo query: top-50% track score, dynamic threshold, frames."""
    cfg = scoring or ScoringSettings()
    sims = similarities(observations, query, vector)
    decision = decide(rank_tracks(sims, cfg.top_fraction), cfg)
    selected = select_frames(sims, decision)
    lookup = {(o.track_id, o.frame_index): o for o in observations}
    hits = tuple(
        FrameHit(f, lookup[(tid, f)].sample_index, tid, lookup[(tid, f)].bbox)
        for f, tid in sorted(selected.items())
    )
    return PhotoSearchResult(decision, hits)
