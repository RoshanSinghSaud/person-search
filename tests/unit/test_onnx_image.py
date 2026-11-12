"""ONNX export and INT8 quantization of a CLIP image tower, on a tiny random ViT.

Same architecture class as IRRA's (clip.model.VisionTransformer), with a
non-square 2:1 patch grid like IRRA's 256x128, so the export path is the real
one; only the size is small enough for CI.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from clip.model import VisionTransformer
from torch import nn

from person_search.errors import ModelLoadError
from person_search.models.onnx_image import OnnxImageSession, export_image_tower, quantize_int8
from person_search.models.preprocessing import CLIP_MEAN, CLIP_STD, build_transform, to_batch

SIZE = (64, 32)  # H x W, a 4x2 patch grid with patch 16


@pytest.fixture(scope="module")
def tiny_vit() -> VisionTransformer:
    torch.manual_seed(0)
    vit = VisionTransformer(
        input_resolution=32, patch_size=16, width=64, layers=2, heads=2, output_dim=32
    )
    grid = (SIZE[0] // 16) * (SIZE[1] // 16)
    vit.positional_embedding = nn.Parameter(0.02 * torch.randn(grid + 1, 64))
    return vit.eval()


@pytest.fixture(scope="module")
def exported(tiny_vit, tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    fp32 = export_image_tower(tiny_vit, SIZE, d / "fp32.onnx", metadata={"version": "test"})
    int8 = quantize_int8(fp32, d / "int8.onnx")
    return fp32, int8


def _crops(n: int, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 255, (int(rng.integers(40, 90)), 30, 3), dtype=np.uint8) for _ in range(n)
    ]


def _torch_reference(vit, crops) -> np.ndarray:
    x = to_batch(crops, build_transform(SIZE, CLIP_MEAN, CLIP_STD))
    with torch.inference_mode():
        f = vit(x)
    return (f / (f.norm(dim=-1, keepdim=True) + 1e-6)).numpy()


def test_fp32_export_matches_pytorch(tiny_vit, exported):
    sess = OnnxImageSession(exported[0], batch_size=4)
    assert sess.image_size == SIZE and sess.dim == 32
    assert sess.metadata["version"] == "test"
    crops = _crops(7)  # 2 batches of 4 and 3: the batch axis is dynamic
    got, ref = sess.encode_images(crops), _torch_reference(tiny_vit, crops)
    assert got.shape == (7, 32)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-4)
    assert (got * ref).sum(axis=1).min() > 0.99999


def test_int8_is_smaller_and_close(exported, tiny_vit):
    fp32, int8 = exported
    assert int8.stat().st_size < fp32.stat().st_size
    crops = _crops(6, seed=2)
    a = OnnxImageSession(int8).encode_images(crops)
    b = _torch_reference(tiny_vit, crops)
    assert (a * b).sum(axis=1).min() > 0.98


def test_raw_features_and_empty_input(exported):
    sess = OnnxImageSession(exported[0])
    assert sess.encode_images([]).shape == (0, 32)
    raw = sess.encode_images(_crops(2), normalize=False)
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        OnnxImageSession(tmp_path / "nope.onnx")
