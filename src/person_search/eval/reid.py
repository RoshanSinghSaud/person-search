"""Retrieval metrics: CMC (Rank-k) and mAP.

Two protocols:

* :func:`evaluate_reid` - image ReID, Market-1501 protocol (the one torchreid's
  ``evaluate_rank`` uses for the campus 75.4% mAP / 81.7% Rank-1 numbers): for
  each query, gallery images with the same identity *and* the same camera are
  removed; queries with no remaining true match are skipped.
* :func:`evaluate_text_to_image` - text-to-image retrieval (IRRA's protocol for
  the college 65.22% Rank-1): every gallery image is a candidate, no camera
  filtering.

Both take a similarity matrix ``(n_query, n_gallery)``: higher = more similar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalMetrics:
    cmc: np.ndarray  # cmc[k-1] = Rank-k accuracy
    mAP: float
    n_valid_queries: int

    def rank(self, k: int) -> float:
        return float(self.cmc[min(k, len(self.cmc)) - 1])

    def as_dict(self, ranks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
        return {"mAP": self.mAP, **{f"rank{k}": self.rank(k) for k in ranks}}


def _average_precision(matches: np.ndarray) -> float:
    """AP of one ranked boolean match list (all true matches are in the list)."""
    n_pos = matches.sum()
    if n_pos == 0:
        return 0.0
    hits = np.cumsum(matches)
    precision_at_hit = hits[matches] / (np.flatnonzero(matches) + 1)
    return float(precision_at_hit.sum() / n_pos)


MatchFn = Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _evaluate(sim: np.ndarray, matches_fn: MatchFn, max_rank: int) -> RetrievalMetrics:
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != 2:
        raise ValueError(f"similarity matrix must be 2-D, got shape {sim.shape}")
    order = np.argsort(-sim, axis=1, kind="stable")
    cmcs, aps = [], []
    for qi in range(sim.shape[0]):
        matches, keep = matches_fn(qi, order[qi])
        matches = matches[keep]
        if not matches.any():
            continue
        cmc = np.cumsum(matches)
        cmc[cmc > 1] = 1
        cmcs.append(cmc[:max_rank].astype(np.float64))
        aps.append(_average_precision(matches))
    if not cmcs:
        raise ValueError("no query has a valid match in the gallery")
    width = max(len(c) for c in cmcs)
    padded = np.array([np.pad(c, (0, width - len(c)), constant_values=c[-1]) for c in cmcs])
    return RetrievalMetrics(padded.mean(axis=0), float(np.mean(aps)), len(cmcs))


def evaluate_reid(
    sim: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> RetrievalMetrics:
    q_pids, g_pids = np.asarray(q_pids), np.asarray(g_pids)
    q_camids, g_camids = np.asarray(q_camids), np.asarray(g_camids)

    def matches_fn(qi: int, ranked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        same_pid = g_pids[ranked] == q_pids[qi]
        same_cam = g_camids[ranked] == q_camids[qi]
        return same_pid, ~(same_pid & same_cam)

    return _evaluate(sim, matches_fn, max_rank)


def evaluate_text_to_image(
    sim: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray, max_rank: int = 10
) -> RetrievalMetrics:
    q_pids, g_pids = np.asarray(q_pids), np.asarray(g_pids)

    def matches_fn(qi: int, ranked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = g_pids[ranked] == q_pids[qi]
        return m, np.ones_like(m, dtype=bool)

    return _evaluate(sim, matches_fn, max_rank)
