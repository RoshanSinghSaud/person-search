"""IRRAEncoder vs repo 1 ``ClipFeatureExtractor`` (IRRA half).

The legacy extractor downloads base OpenAI ViT-B/16 (~335 MB, cached in
``~/.cache/clip``) before overwriting it with the IRRA checkpoint; ours builds
the model from the checkpoint alone. Matching outputs show nothing from the
base model survives in the legacy path except the unused ``logit_scale``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from person_search.models.irra_encoder import IRRAEncoder
from tests.conftest import WEIGHTS, require
from tests.parity.conftest import DESCRIPTIONS, cosine_rows, load_legacy_module

pytestmark = [pytest.mark.parity, pytest.mark.slow]


@pytest.fixture(scope="module")
def legacy_extractor(legacy_root):
    require(WEIGHTS["irra"])
    module = load_legacy_module(
        legacy_root / "person-reid-clip" / "clip_inference.py", "legacy_repo1_clip_inference"
    )
    ext = module.ClipFeatureExtractor(
        device=torch.device("cpu"), image_ckpt_path=None, text_ckpt_path=str(WEIGHTS["irra"])
    )
    assert ext.text_ckpt_loaded and ext.n_ctx == 8
    return ext


@pytest.fixture(scope="module")
def encoder():
    require(WEIGHTS["irra"])
    return IRRAEncoder(WEIGHTS["irra"], batch_size=8)


def test_irra_image_matches_legacy(legacy_extractor, encoder, person_crops):
    ref = np.concatenate(
        [legacy_extractor.extract_image_embedding_for_text_query(c) for c in person_crops]
    )
    ours = encoder.encode_images(person_crops)
    cos = cosine_rows(ref, ours)
    assert cos.min() > 0.9999, f"min cosine {cos.min():.6f}"


def test_irra_text_matches_legacy(legacy_extractor, encoder):
    # the last description must exercise the EOT clamp (AUDIT.md F15)
    eot = int(encoder.tokenize([DESCRIPTIONS[-1]]).argmax(dim=-1)[0])
    assert eot + encoder.n_ctx > encoder.context_length - 1
    ref = np.concatenate([legacy_extractor.extract_text_embedding(t) for t in DESCRIPTIONS])
    ours = encoder.encode_texts(DESCRIPTIONS)
    cos = cosine_rows(ref, ours)
    assert cos.min() > 0.9999, f"min cosine {cos.min():.6f}"


def test_irra_text_to_image_ranking_unchanged(legacy_extractor, encoder, person_crops):
    ref_img = np.concatenate(
        [legacy_extractor.extract_image_embedding_for_text_query(c) for c in person_crops]
    )
    ref_txt = np.concatenate([legacy_extractor.extract_text_embedding(t) for t in DESCRIPTIONS])
    ours = encoder.encode_texts(DESCRIPTIONS) @ encoder.encode_images(person_crops).T
    np.testing.assert_array_equal(
        np.argsort(-(ref_txt @ ref_img.T), axis=1)[:, :10], np.argsort(-ours, axis=1)[:, :10]
    )
