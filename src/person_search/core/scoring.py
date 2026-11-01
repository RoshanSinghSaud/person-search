"""The paper's track-level robust scoring and dynamic threshold.

Extracted from repo 2 ``pole/reid_main.py`` (``compute_track_score`` and the
selection block after it) and kept exactly:

* track score = mean of the top 50 % of the track's per-detection similarities
  (at least one); a track with no similarities scores 0.0;
* tau_eff = max(tau_base, best_score - delta); every track scoring >= tau_eff
  is accepted (handles one person split over several track IDs);
* "no match found" when the best score is below tau_base;
* per-frame floor = 0.9 * tau_eff, applied to a forward-filled similarity
  (frames before the first similarity take the first one);
* a frame claimed by a higher-scoring track is not reassigned.

Pure functions over plain Python data: no torch, no video, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from person_search.config import ScoringSettings

# (frame key, similarity or None when not computed on that frame)
FrameSim = tuple[int, float | None]


@dataclass(frozen=True, slots=True)
class TrackScore:
    track_id: str
    score: float
    n_frames: int
    n_scored: int


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    ranked: tuple[TrackScore, ...]
    effective_threshold: float
    frame_threshold: float
    accepted: tuple[str, ...]

    @property
    def matched(self) -> bool:
        return bool(self.accepted)

    @property
    def best(self) -> TrackScore | None:
        return self.ranked[0] if self.ranked else None


def track_score(similarities: Sequence[float], top_fraction: float = 0.5) -> float:
    """Mean of the top ``top_fraction`` of ``similarities`` (at least one value)."""
    if not similarities:
        return 0.0
    ordered = sorted(similarities, reverse=True)
    k = max(1, int(len(ordered) * top_fraction))
    return float(np.mean(ordered[:k]))


def effective_threshold(best_score: float, base: float, margin: float) -> float:
    return max(base, best_score - margin)


def rank_tracks(
    track_frames: Mapping[str, Sequence[FrameSim]], top_fraction: float = 0.5
) -> list[TrackScore]:
    """Score every track and sort by score, descending.

    The sort is stable, so tied tracks keep their input order (first
    appearance), which is what the legacy ``list.sort`` did.
    """
    scores = []
    for tid, frames in track_frames.items():
        sims = [s for _, s in frames if s is not None]
        scores.append(TrackScore(tid, track_score(sims, top_fraction), len(frames), len(sims)))
    return sorted(scores, key=lambda t: t.score, reverse=True)


def decide(ranked: Sequence[TrackScore], cfg: ScoringSettings) -> ThresholdDecision:
    """Apply the dynamic threshold to ranked track scores."""
    if not ranked:
        return ThresholdDecision(
            (), cfg.base_threshold, cfg.base_threshold * cfg.frame_threshold_ratio, ()
        )
    tau = effective_threshold(ranked[0].score, cfg.base_threshold, cfg.relative_margin)
    accepted = tuple(t.track_id for t in ranked if t.score >= tau)
    return ThresholdDecision(tuple(ranked), tau, tau * cfg.frame_threshold_ratio, accepted)


def forward_fill(frames: Sequence[FrameSim]) -> list[tuple[int, float]]:
    """Fill missing similarities with the last seen value.

    Leading gaps take the first available value; a track with no values at
    all is filled with 0.0 (legacy ``last_valid_sim = 0.0``).
    """
    first = next((s for _, s in frames if s is not None), 0.0)
    out, last = [], first
    for key, sim in frames:
        if sim is not None:
            last = sim
        out.append((key, last))
    return out


def select_frames(
    track_frames: Mapping[str, Sequence[FrameSim]], decision: ThresholdDecision
) -> dict[int, str]:
    """Frames kept for the accepted tracks: ``{frame key: track_id}``."""
    selected: dict[int, str] = {}
    for tid in decision.accepted:
        for key, sim in forward_fill(track_frames[tid]):
            if sim >= decision.frame_threshold and key not in selected:
                selected[key] = tid
    return selected
