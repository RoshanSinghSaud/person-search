"""Text query: IRRA text vector -> ranked tracks for a person to review."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from person_search.config import ScoringSettings
from person_search.core.scoring import TrackScore, rank_tracks
from person_search.search.photo import similarities
from person_search.types import IRRA, Observation


def text_search(
    observations: Sequence[Observation],
    query: np.ndarray,
    top_k: int = 5,
    scoring: ScoringSettings | None = None,
    vector: str = IRRA,
) -> list[TrackScore]:
    """Text query: rank tracks by top-50% score; no threshold (a person reviews)."""
    cfg = scoring or ScoringSettings()
    return rank_tracks(similarities(observations, query, vector), cfg.top_fraction)[:top_k]
