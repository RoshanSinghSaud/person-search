"""OSNet image encoder for photo queries (the published paper's model).

Extracted from repo 2 ``pole/reid_main.py`` -> ``FeatureExtractor``:

* same architecture (``osnet_x1_0``, vendored from torchreid),
* same preprocessing (256x128, ImageNet mean/std),
* same output (the 512-d feature OSNet returns in eval mode, divided by
  ``norm + 1e-6``).

Removed: the ImageNet ResNet50 fallback. A missing file or a checkpoint whose
backbone keys do not match raises instead (AUDIT.md F7).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from person_search.errors import ModelLoadError
from person_search.models import osnet as osnet_arch
from person_search.models.checkpoint import drop_prefixes, load_strict, read_state_dict
from person_search.models.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    batched,
    build_transform,
    l2_normalize,
    to_batch,
)
from person_search.types import OSNET

log = logging.getLogger(__name__)

# The ID classifier is training-only: the campus model has 70 classes, the
# Market-1501 model 751. Eval-mode OSNet never calls it.
_CLASSIFIER_PREFIX = "classifier."


class OSNetEncoder:
    name = OSNET
    dim = 512

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "cpu",
        input_size: tuple[int, int] = (256, 128),
        batch_size: int = 32,
        version: str = "osnet-v1",
    ) -> None:
        self.weights = Path(weights)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.version = version
        self.transform = build_transform(input_size, IMAGENET_MEAN, IMAGENET_STD)
        self.model = self._load()

    def _load(self) -> nn.Module:
        # torchreid checkpoints pickle optimizer/scheduler state -> trusted load.
        state = read_state_dict(self.weights, trusted=True)
        state = drop_prefixes(state, [_CLASSIFIER_PREFIX])
        if not any(k.startswith("conv1.") for k in state):
            raise ModelLoadError(f"{self.weights} does not look like an OSNet checkpoint")

        model = osnet_arch.osnet_x1_0(num_classes=1, pretrained=False, loss="softmax")
        model.classifier = nn.Identity()  # no parameters; never called in eval mode
        load_strict(model, state, model_name="OSNet x1_0", source=self.weights)
        return model.to(self.device).eval()

    @torch.inference_mode()
    def encode_images(self, images: Sequence[np.ndarray], *, normalize: bool = True) -> np.ndarray:
        """``normalize=False`` returns the raw features (torchreid's test protocol)."""
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for chunk in batched(images, self.batch_size):
            x = to_batch(chunk, self.transform).to(self.device)
            feats = self.model(x)
            out.append((l2_normalize(feats) if normalize else feats).cpu())
        return torch.cat(out).numpy().astype(np.float32, copy=False)
