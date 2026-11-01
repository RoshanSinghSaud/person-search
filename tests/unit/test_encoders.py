"""Encoder loading rules, on synthetic checkpoints (no real weights needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from person_search.errors import CheckpointMismatchError, ModelLoadError
from person_search.models import osnet as osnet_arch
from person_search.models.irra_encoder import IRRAEncoder, grid_from_tokens
from person_search.models.osnet_encoder import OSNetEncoder

RNG = np.random.default_rng(0)


def crops(n: int, h: int = 120, w: int = 50) -> list[np.ndarray]:
    return [RNG.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


# ------------------------------------------------------------------ OSNet
@pytest.fixture(scope="module")
def osnet_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    model = osnet_arch.osnet_x1_0(num_classes=70, pretrained=False)
    path = tmp_path_factory.mktemp("osnet") / "osnet.pth.tar"
    # torchreid layout: weights wrapped with training metadata
    torch.save({"state_dict": model.state_dict(), "epoch": 60, "rank1": 0.8}, path)
    return path


def test_osnet_encodes_normalised_512d(osnet_ckpt):
    enc = OSNetEncoder(osnet_ckpt, batch_size=2)
    out = enc.encode_images(crops(5))
    assert out.shape == (5, 512) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_osnet_batching_does_not_change_vectors(osnet_ckpt):
    enc = OSNetEncoder(osnet_ckpt, batch_size=8)
    xs = crops(4)
    batched = enc.encode_images(xs)
    single = np.concatenate([enc.encode_images([x]) for x in xs])
    np.testing.assert_allclose((batched * single).sum(1), 1.0, atol=1e-5)


def test_osnet_empty_input(osnet_ckpt):
    assert OSNetEncoder(osnet_ckpt).encode_images([]).shape == (0, 512)


def test_osnet_missing_file_raises(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        OSNetEncoder(tmp_path / "missing.pth")


def test_osnet_missing_backbone_key_raises(osnet_ckpt, tmp_path):
    state = torch.load(osnet_ckpt, weights_only=False)["state_dict"]
    del state["conv5.conv.weight"]
    bad = tmp_path / "bad.pth"
    torch.save(state, bad)
    with pytest.raises(CheckpointMismatchError, match="missing"):
        OSNetEncoder(bad)


def test_osnet_wrong_architecture_raises(tmp_path):
    small = osnet_arch.osnet_x0_25(num_classes=10, pretrained=False)
    bad = tmp_path / "x025.pth"
    torch.save(small.state_dict(), bad)
    with pytest.raises(CheckpointMismatchError, match="shape mismatch"):
        OSNetEncoder(bad)


def test_osnet_rejects_non_osnet_file(tmp_path):
    bad = tmp_path / "other.pth"
    torch.save({"visual.conv1.weight": torch.zeros(1)}, bad)
    with pytest.raises(ModelLoadError, match="OSNet"):
        OSNetEncoder(bad)


# ------------------------------------------------------------------- IRRA
def tiny_clip_state(n_ctx: int = 4) -> dict[str, torch.Tensor]:
    """A 2-layer CLIP with a 16x8 patch grid (256x128 input), IRRA-style keys."""
    from clip.model import CLIP

    torch.manual_seed(0)
    m = CLIP(
        embed_dim=32,
        image_resolution=224,
        vision_layers=2,
        vision_width=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=2,
    )
    state = {k: v.clone() for k, v in m.state_dict().items() if k != "logit_scale"}
    state["visual.positional_embedding"] = torch.randn(129, 64) * 0.02
    if n_ctx:
        state["ctx"] = torch.randn(n_ctx, 64) * 0.02
    # IRRA training heads that inference must ignore
    state["mlm_head.0.weight"] = torch.zeros(4, 4)
    state["cross_attn.in_proj_weight"] = torch.zeros(4, 4)
    state["img_classifier.weight"] = torch.zeros(3, 32)
    return state


@pytest.fixture(scope="module")
def irra_ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("irra") / "irra.pth"
    torch.save(tiny_clip_state(), path)
    return path


def test_irra_builds_from_checkpoint_shapes(irra_ckpt):
    enc = IRRAEncoder(irra_ckpt)
    assert enc.image_size == (256, 128)
    assert enc.dim == 32 and enc.n_ctx == 4


def test_irra_image_and_text_vectors(irra_ckpt):
    enc = IRRAEncoder(irra_ckpt, batch_size=2)
    img = enc.encode_images(crops(3))
    txt = enc.encode_texts(["a man in a red jacket", "a woman with a backpack", "a child"])
    assert img.shape == (3, 32) and txt.shape == (3, 32)
    np.testing.assert_allclose(np.linalg.norm(txt, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(img, axis=1), 1.0, atol=1e-5)


def test_irra_prompt_tokens_change_text_vectors(irra_ckpt, tmp_path):
    no_prompt = tiny_clip_state(n_ctx=0)
    p = tmp_path / "plain.pth"
    torch.save(no_prompt, p)
    with_ctx = IRRAEncoder(irra_ckpt).encode_texts(["a man in a red jacket"])
    plain = IRRAEncoder(p).encode_texts(["a man in a red jacket"])
    assert float((with_ctx * plain).sum()) < 0.999


def test_irra_rejects_empty_text(irra_ckpt):
    with pytest.raises(ValueError):
        IRRAEncoder(irra_ckpt).encode_texts(["  "])


def test_irra_unknown_key_raises_instead_of_falling_back(tmp_path):
    state = tiny_clip_state()
    state["some_new_head.weight"] = torch.zeros(2)
    p = tmp_path / "weird.pth"
    torch.save(state, p)
    with pytest.raises(CheckpointMismatchError, match="unexpected"):
        IRRAEncoder(p)


def test_irra_missing_layer_raises(tmp_path):
    state = tiny_clip_state()
    del state["visual.ln_post.weight"]
    p = tmp_path / "partial.pth"
    torch.save(state, p)
    with pytest.raises(CheckpointMismatchError, match="missing"):
        IRRAEncoder(p)


def test_irra_rejects_non_clip_file(tmp_path):
    p = tmp_path / "osnet.pth"
    torch.save({"conv1.conv.weight": torch.zeros(1)}, p)
    with pytest.raises(ModelLoadError, match="not a CLIP"):
        IRRAEncoder(p)


@pytest.mark.parametrize(
    ("tokens", "grid"), [(197, (14, 14)), (129, (16, 8)), (513, (32, 16)), (50, (7, 7))]
)
def test_grid_from_tokens(tokens, grid):
    assert grid_from_tokens(tokens) == grid


def test_grid_from_tokens_rejects_odd_shapes():
    with pytest.raises(ModelLoadError):
        grid_from_tokens(100)
