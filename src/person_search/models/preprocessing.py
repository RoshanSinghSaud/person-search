"""Crop preprocessing shared by the encoders.

Each crop goes through exactly the legacy per-image transform
(BGR -> RGB -> PIL -> Resize -> ToTensor -> Normalize), then crops are stacked
into one batch. Keeping the per-crop transform identical is what makes the
batched encoders match the legacy single-crop path to cosine > 0.9999.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transform(
    size: tuple[int, int], mean: Sequence[float], std: Sequence[float]
) -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def to_batch(images: Sequence[np.ndarray], transform: T.Compose) -> torch.Tensor:
    """BGR uint8 crops -> normalised float tensor ``(N, 3, H, W)``."""
    if len(images) == 0:
        raise ValueError("to_batch() needs at least one image")
    tensors = []
    for img in images:
        if img is None or img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
            shape = None if img is None else img.shape
            raise ValueError(f"expected a non-empty HxWx3 BGR image, got shape {shape}")
        tensors.append(transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return torch.stack(tensors)


def l2_normalize(feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x / (||x|| + eps)`` - the legacy normalisation (not ``F.normalize``)."""
    return feats / (feats.norm(dim=-1, keepdim=True) + eps)


def batched(items: Sequence, size: int) -> list[Sequence]:
    return [items[i : i + size] for i in range(0, len(items), size)]
