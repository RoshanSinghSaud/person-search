"""Combining photo (OSNet) and text (IRRA) evidence for hybrid queries.

Repo 1 built the hybrid query as ``alpha * image_emb + (1 - alpha) * text_emb``
where ``image_emb`` came from the CLIP-ReID image model and ``text_emb`` from
IRRA, then compared the sum with IRRA image vectors (AUDIT.md F2). The two
embeddings live in different spaces, so the sum is not meaningful.
:func:`legacy_embedding_sum` keeps that behaviour only so Day 12 can measure it.

The replacement is late fusion: score each detection in each model's own space,
normalise the two score sets so they are comparable, then mix them.
Which normalisation and which alpha work best is a Day 12 experiment.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Normalization = Literal["minmax", "zscore"]


def l2n(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def legacy_embedding_sum(image_emb: np.ndarray, text_emb: np.ndarray, alpha: float) -> np.ndarray:
    """Repo 1 ``_build_query_feature`` hybrid branch (the bug, kept for comparison)."""
    _check_alpha(alpha)
    return l2n(alpha * np.asarray(image_emb).ravel() + (1.0 - alpha) * np.asarray(text_emb).ravel())


def normalize_scores(scores: np.ndarray, method: Normalization = "minmax") -> np.ndarray:
    """Map a set of similarity scores to a comparable range.

    ``minmax`` -> [0, 1]; ``zscore`` -> mean 0, std 1. A constant score set maps
    to zeros (no evidence either way) rather than NaN.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    if method == "minmax":
        lo, hi = s.min(), s.max()
        return np.zeros_like(s) if hi - lo < 1e-12 else (s - lo) / (hi - lo)
    if method == "zscore":
        sd = s.std()
        return np.zeros_like(s) if sd < 1e-12 else (s - s.mean()) / sd
    raise ValueError(f"unknown normalization {method!r}")


def late_fusion(
    photo_scores: np.ndarray,
    text_scores: np.ndarray,
    alpha: float,
    method: Normalization = "minmax",
) -> np.ndarray:
    """``alpha * norm(photo) + (1 - alpha) * norm(text)``, per detection."""
    _check_alpha(alpha)
    p, t = np.asarray(photo_scores), np.asarray(text_scores)
    if p.shape != t.shape:
        raise ValueError(f"score arrays differ in shape: {p.shape} vs {t.shape}")
    return alpha * normalize_scores(p, method) + (1.0 - alpha) * normalize_scores(t, method)


def _check_alpha(alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
