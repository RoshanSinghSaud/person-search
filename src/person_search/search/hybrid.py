"""Hybrid query: late fusion of normalised OSNet and IRRA per-detection scores."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from person_search.config import FusionSettings, ScoringSettings
from person_search.core.fusion import late_fusion
from person_search.core.scoring import FrameSim, TrackScore, rank_tracks
from person_search.search.photo import as_query
from person_search.types import IRRA, OSNET, Observation


def hybrid_search(
    observations: Sequence[Observation],
    photo_query: np.ndarray,
    text_query: np.ndarray,
    top_k: int = 5,
    fusion: FusionSettings | None = None,
    scoring: ScoringSettings | None = None,
) -> list[TrackScore]:
    """Late fusion per detection, then top-50% track scores, ranked.

    Only detections that carry both vectors take part.
    """
    fcfg = fusion or FusionSettings()
    cfg = scoring or ScoringSettings()
    p, t = as_query(photo_query), as_query(text_query)
    both = [o for o in observations if OSNET in o.embeddings and IRRA in o.embeddings]
    if not both:
        return []
    s_photo = np.array([np.dot(p, o.embeddings[OSNET]) for o in both])
    s_text = np.array([np.dot(t, o.embeddings[IRRA]) for o in both])
    fused = late_fusion(s_photo, s_text, fcfg.alpha, fcfg.normalization)
    per_track: dict[str, list[FrameSim]] = {}
    for o, s in zip(both, fused, strict=True):
        per_track.setdefault(o.track_id, []).append((o.frame_index, float(s)))
    return rank_tracks(per_track, cfg.top_fraction)[:top_k]
