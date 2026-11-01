"""Checkpoint reading and strict state-dict matching.

The legacy loaders filtered checkpoint keys down to "whatever fits" and kept
running when nothing fit (repo 1 fell back to base CLIP, repo 2 to an ImageNet
ResNet50). Here a checkpoint either matches its architecture, with an explicit
allow-list of keys that may be missing or ignored, or loading raises
:class:`CheckpointMismatchError`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from person_search.errors import CheckpointMismatchError, ModelLoadError

log = logging.getLogger(__name__)

StateDict = dict[str, torch.Tensor]


def read_state_dict(path: str | Path, *, trusted: bool = False) -> StateDict:
    """Load a checkpoint file and return its tensor state dict.

    Handles both raw state dicts and torchreid-style ``{"state_dict": ...}``
    wrappers, and strips a ``module.`` prefix left by DataParallel.

    ``trusted=False`` uses ``weights_only=True`` (no arbitrary pickle code).
    torchreid checkpoints also pickle optimizer/scheduler objects and need
    ``trusted=True``; only use that for files from the team's own registry.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(f"checkpoint not found: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=not trusted)
    except Exception as exc:
        raise ModelLoadError(f"cannot read checkpoint {path}: {exc}") from exc

    if isinstance(obj, Mapping) and isinstance(obj.get("state_dict"), Mapping):
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping):
        raise ModelLoadError(f"{path} does not contain a state dict (got {type(obj).__name__})")

    state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if not state:
        raise ModelLoadError(f"{path} contains no tensors")
    if all(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def drop_prefixes(state: Mapping[str, torch.Tensor], prefixes: Iterable[str]) -> StateDict:
    prefixes = tuple(prefixes)
    return {k: v for k, v in state.items() if not k.startswith(prefixes)}


@dataclass(frozen=True)
class MatchReport:
    loaded: int
    allowed_missing: tuple[str, ...]


def load_strict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    model_name: str,
    source: str | Path,
    allow_missing: Iterable[str] = (),
) -> MatchReport:
    """Copy ``state`` into ``model``; raise unless keys and shapes match exactly.

    ``allow_missing`` lists model keys that may be absent from the checkpoint
    (e.g. CLIP's ``logit_scale``, which inference never uses).
    """
    allow = set(allow_missing)
    model_state = model.state_dict()
    missing = set(model_state) - set(state) - allow
    unexpected = set(state) - set(model_state)
    shape_mismatch = {
        f"{k} ckpt{tuple(state[k].shape)} != model{tuple(model_state[k].shape)}"
        for k in set(state) & set(model_state)
        if state[k].shape != model_state[k].shape
    }
    if missing or unexpected or shape_mismatch:
        raise CheckpointMismatchError(model_name, str(source), missing, unexpected, shape_mismatch)

    result = model.load_state_dict(dict(state), strict=False)
    # Belt and braces: torch agrees with our own accounting.
    if set(result.missing_keys) - allow or result.unexpected_keys:
        raise CheckpointMismatchError(
            model_name, str(source), result.missing_keys, result.unexpected_keys
        )
    report = MatchReport(loaded=len(state), allowed_missing=tuple(sorted(allow & set(model_state))))
    log.info(
        "loaded %s from %s (%d tensors, allowed missing: %s)",
        model_name,
        source,
        report.loaded,
        ", ".join(report.allowed_missing) or "none",
    )
    return report
