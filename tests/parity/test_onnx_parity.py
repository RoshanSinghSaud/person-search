"""ONNX image tower vs the PyTorch IRRA encoder on real crops (``make parity``).

* FP32 export: must match PyTorch (cosine > 0.9999, identical text-to-image top-10).
* INT8: reported, with a loose floor; whether INT8 is good enough is decided by
  the gate set (scripts/eval_models.py), not here.

Needs ``models/irra_v1_image_fp32.onnx`` and ``models/irra_v2_image_int8.onnx``
(``python scripts/export_onnx.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from person_search.models.irra_encoder import IRRAEncoder
from person_search.models.onnx_image import OnnxImageSession
from tests.conftest import MODELS_DIR, WEIGHTS, require
from tests.parity.conftest import DESCRIPTIONS, cosine_rows

pytestmark = pytest.mark.parity

FP32 = MODELS_DIR / "irra_v1_image_fp32.onnx"
INT8 = MODELS_DIR / "irra_v2_image_int8.onnx"


@pytest.fixture(scope="module")
def torch_encoder() -> IRRAEncoder:
    require(WEIGHTS["irra"])
    return IRRAEncoder(WEIGHTS["irra"])


def _top10(text: np.ndarray, images: np.ndarray) -> list[list[int]]:
    return np.argsort(-(text @ images.T), axis=1, kind="stable")[:, :10].tolist()


def test_onnx_fp32_matches_pytorch(torch_encoder, person_crops):
    require(FP32)
    ref = torch_encoder.encode_images(person_crops)
    got = OnnxImageSession(FP32).encode_images(person_crops)
    assert cosine_rows(got, ref).min() > 0.9999
    text = torch_encoder.encode_texts(DESCRIPTIONS)
    assert _top10(text, got) == _top10(text, ref)


def test_onnx_int8_stays_close(torch_encoder, person_crops):
    require(INT8)
    ref = torch_encoder.encode_images(person_crops)
    got = OnnxImageSession(INT8).encode_images(person_crops)
    cos = cosine_rows(got, ref)
    print(f"INT8 vs FP32 cosine: min {cos.min():.4f}, mean {cos.mean():.4f}")
    assert cos.min() > 0.95


def test_encoder_with_onnx_backend_keeps_the_text_path(torch_encoder):
    require(FP32)
    enc = IRRAEncoder(WEIGHTS["irra"], image_onnx=FP32, version="irra-v1-onnx")
    assert enc.onnx is not None
    assert np.array_equal(enc.encode_texts(DESCRIPTIONS), torch_encoder.encode_texts(DESCRIPTIONS))
