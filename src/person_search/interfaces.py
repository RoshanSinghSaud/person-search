"""Adapter interfaces.

Concrete implementations live in ``ingest/`` and ``models/``. Everything else
depends on these protocols, so a model can be swapped (ONNX, INT8, LoRA) or
faked in tests without touching the pipeline or the search code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from person_search.types import Detection, TrackState


@runtime_checkable
class Detector(Protocol):
    """Finds people in a full-resolution BGR frame."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Associates detections over time. One instance per video."""

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> list[TrackState]:
        """Return the *confirmed* tracks after this frame."""
        ...

    def reset(self) -> None:
        """Forget all tracks (call before a new video)."""
        ...


@runtime_checkable
class ImageEncoder(Protocol):
    """Embeds BGR person crops into an L2-normalised vector space."""

    name: str
    dim: int
    version: str

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return a float32 array of shape ``(len(images), dim)``."""
        ...


@runtime_checkable
class TextEncoder(Protocol):
    """Embeds free-text descriptions into the same space as its image encoder."""

    name: str
    dim: int
    version: str

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)``."""
        ...
