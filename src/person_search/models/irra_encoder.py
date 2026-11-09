"""IRRA encoder for text queries (model 2), image and text towers.

Extracted from the IRRA half of repo 1 ``clip_inference.py`` ->
``ClipFeatureExtractor`` (``_load_text_checkpoint``,
``extract_image_embedding_for_text_query``, ``extract_text_embedding``,
``_encode_text_with_prompts``). Behavioural parity:

* image crops: CLIP mean/std, resized to the grid of the checkpoint's visual
  positional embedding (129 tokens -> 16x8 patches -> 256x128),
* text: CLIP BPE tokens with the learned prompt tokens (``ctx``) inserted after
  SOS, the last ``n_ctx`` tokens dropped, and the global feature read at the
  EOT position shifted by ``n_ctx``,
* outputs divided by ``norm + 1e-6``.

What changed:

* The model is built from the checkpoint's own shapes, so no base OpenAI CLIP
  download is needed. The IRRA checkpoint holds every CLIP tensor except
  ``logit_scale``, which inference never reads, so the result is identical.
* Loading is strict. Keys that are neither CLIP tensors nor known IRRA
  training heads raise :class:`CheckpointMismatchError` instead of falling back
  to base CLIP (AUDIT.md F7).
* The CLIP-ReID image model from repo 1 is not ported: photo queries use OSNet.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from person_search.errors import ModelLoadError
from person_search.models.checkpoint import StateDict, drop_prefixes, load_strict, read_state_dict
from person_search.models.preprocessing import (
    CLIP_MEAN,
    CLIP_STD,
    batched,
    build_transform,
    l2_normalize,
    to_batch,
)
from person_search.types import IRRA

try:  # OpenAI CLIP: model definition + BPE tokenizer (MIT)
    import clip
    from clip.model import CLIP
except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
    raise ImportError(
        "OpenAI CLIP is required: pip install git+https://github.com/openai/CLIP.git"
    ) from exc

log = logging.getLogger(__name__)

# IRRA training-only heads (cross-modal MLM, ID classifiers, BN necks).
IRRA_TRAINING_PREFIXES = (
    "cross_attn.",
    "cross_attn_norm.",
    "mlm_head.",
    "img_bn.",
    "txt_bn.",
    "img_classifier.",
    "txt_classifier.",
)
# Bookkeeping entries in OpenAI's original state dicts (vanilla CLIP = irra v0).
CLIP_META_KEYS = ("input_resolution", "context_length", "vocab_size")
PROMPT_KEY = "ctx"


def _count_blocks(state: StateDict, prefix: str) -> int:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    idx = {int(m.group(1)) for k in state if (m := pat.match(k))}
    return max(idx) + 1 if idx else 0


def grid_from_tokens(n_tokens: int) -> tuple[int, int]:
    """Patch grid ``(rows, cols)`` for a ViT positional embedding with a CLS token.

    Square grids (197 -> 14x14) and the 2:1 ReID grids (129 -> 16x8) are
    recognised, matching the legacy ``_infer_image_size``.
    """
    n = n_tokens - 1
    side = math.isqrt(n)
    if side * side == n:
        return side, side
    rows = math.isqrt(2 * n)
    if rows * rows == 2 * n and rows % 2 == 0:
        return rows, rows // 2
    raise ModelLoadError(f"cannot infer a patch grid for {n_tokens} positional tokens")


def build_clip_from_state(state: StateDict) -> tuple[CLIP, tuple[int, int]]:
    """Instantiate a CLIP ViT whose shapes match ``state``.

    Returns the model and the image input size ``(H, W)``.
    """
    try:
        conv1 = state["visual.conv1.weight"]
        vis_pos = state["visual.positional_embedding"]
        width = state["ln_final.weight"].shape[0]
        model = CLIP(
            embed_dim=state["text_projection"].shape[1],
            image_resolution=224,  # placeholder; the positional embedding is replaced below
            vision_layers=_count_blocks(state, "visual.transformer.resblocks."),
            vision_width=conv1.shape[0],
            vision_patch_size=conv1.shape[-1],
            context_length=state["positional_embedding"].shape[0],
            vocab_size=state["token_embedding.weight"].shape[0],
            transformer_width=width,
            transformer_heads=width // 64 or 1,
            transformer_layers=_count_blocks(state, "transformer.resblocks."),
        )
    except KeyError as exc:
        raise ModelLoadError(f"not a CLIP ViT checkpoint: missing {exc}") from exc

    rows, cols = grid_from_tokens(vis_pos.shape[0])
    patch = conv1.shape[-1]
    model.visual.positional_embedding = nn.Parameter(torch.empty_like(vis_pos))
    model.visual.input_resolution = (rows * patch, cols * patch)
    return model, (rows * patch, cols * patch)


def _read_clip_checkpoint(path: Path) -> StateDict:
    if not path.is_file():
        raise ModelLoadError(f"checkpoint not found: {path}")
    try:
        return read_state_dict(path)
    except ModelLoadError as first:
        # OpenAI's released .pt files (vanilla CLIP = irra v0) are TorchScript archives.
        try:
            return dict(torch.jit.load(str(path), map_location="cpu").state_dict())
        except Exception:
            raise ModelLoadError(
                f"{path} is neither a plain state dict nor a TorchScript CLIP archive ({first})"
            ) from first


class IRRAEncoder:
    """Image and text encoders that share IRRA's joint embedding space."""

    name = IRRA

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        precision: str = "fp32",
        version: str = "irra-v1",
    ) -> None:
        self.weights = Path(weights)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.version = version
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be fp32 or fp16, got {precision!r}")
        self.dtype = torch.float16 if precision == "fp16" else torch.float32

        state = _read_clip_checkpoint(self.weights)
        ctx = state.pop(PROMPT_KEY, None)
        state = drop_prefixes(state, IRRA_TRAINING_PREFIXES)
        for key in CLIP_META_KEYS:
            state.pop(key, None)

        model, self.image_size = build_clip_from_state(state)
        load_strict(
            model, state, model_name="IRRA/CLIP", source=self.weights, allow_missing=["logit_scale"]
        )
        self.model = model.to(self.device, dtype=self.dtype).eval()
        self.dim = int(self.model.text_projection.shape[1])
        self.context_length = int(self.model.positional_embedding.shape[0])

        self.ctx: torch.Tensor | None = None
        if ctx is not None:
            if ctx.ndim != 2 or ctx.shape[1] != self.model.token_embedding.weight.shape[1]:
                raise ModelLoadError(f"prompt tokens have shape {tuple(ctx.shape)}")
            self.ctx = ctx.to(self.device, dtype=self.dtype)
        self.n_ctx = 0 if self.ctx is None else int(self.ctx.shape[0])
        log.info(
            "IRRA ready: image %sx%s, dim %d, %s",
            *self.image_size,
            self.dim,
            f"{self.n_ctx} learned prompt tokens"
            if self.n_ctx
            else "no prompt tokens (plain CLIP text path)",
        )
        self.transform = build_transform(self.image_size, CLIP_MEAN, CLIP_STD)

    # ------------------------------------------------------------------ image
    @torch.inference_mode()
    def encode_images(self, images: Sequence[np.ndarray], *, normalize: bool = True) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for chunk in batched(images, self.batch_size):
            x = to_batch(chunk, self.transform).to(self.device, dtype=self.dtype)
            feats = self.model.encode_image(x).float()
            out.append((l2_normalize(feats) if normalize else feats).cpu())
        return torch.cat(out).numpy()

    # ------------------------------------------------------------------- text
    def tokenize(self, texts: Sequence[str]) -> torch.Tensor:
        return clip.tokenize(
            [str(t) for t in texts], context_length=self.context_length, truncate=True
        )

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if any(not str(t).strip() for t in texts):
            raise ValueError("text queries must be non-empty")
        out = []
        for chunk in batched(list(texts), self.batch_size):
            tokens = self.tokenize(chunk).to(self.device)
            feats = (
                self._encode_with_prompts(tokens) if self.n_ctx else self.model.encode_text(tokens)
            )
            out.append(l2_normalize(feats.float()).cpu())
        return torch.cat(out).numpy()

    def _encode_with_prompts(self, tokens: torch.Tensor) -> torch.Tensor:
        """IRRA text path with learned prompts: ``[SOS] ctx_1..ctx_n w_1..w_k [EOT] ...``.

        The sequence keeps CLIP's length, so the last ``n_ctx`` token slots are
        dropped. A description long enough to push EOT past the end is read at
        the last position (legacy behaviour); that is logged.
        """
        assert self.ctx is not None
        m = self.model
        batch, length = tokens.shape
        x = m.token_embedding(tokens).to(self.dtype)
        ctx = self.ctx.unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat([x[:, :1], ctx, x[:, 1 : length - self.n_ctx]], dim=1)
        x = x + m.positional_embedding.to(self.dtype)
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = m.ln_final(x).to(self.dtype)

        eot = tokens.argmax(dim=-1) + self.n_ctx
        if bool((eot > length - 1).any()):
            log.warning(
                "text query too long: EOT truncated by the prompt tokens; consider shortening"
            )
        eot = eot.clamp(max=length - 1)
        return x[torch.arange(batch, device=x.device), eot] @ m.text_projection.to(self.dtype)
