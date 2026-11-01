"""OSNetEncoder vs repo 2 ``FeatureExtractor`` on the same 20 crops."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from person_search.models.osnet_encoder import OSNetEncoder
from tests.conftest import WEIGHTS, require
from tests.parity.conftest import cosine_rows, load_legacy_module

pytestmark = [pytest.mark.parity, pytest.mark.slow]


@pytest.fixture(scope="module")
def legacy_repo2(legacy_root):
    return load_legacy_module(
        legacy_root / "amrit-pbvs" / "pole" / "reid_main.py", "legacy_repo2_reid_main"
    )


@pytest.mark.parametrize("weights_key", ["osnet", "osnet_market"])
def test_osnet_matches_legacy(legacy_repo2, person_crops, weights_key):
    weights = WEIGHTS[weights_key]
    require(weights)
    legacy_repo2.OSNET_MODEL_PATH = str(weights)
    legacy = legacy_repo2.FeatureExtractor(torch.device("cpu"))
    # The legacy loader swallows errors and swaps in ResNet50; make sure it did not.
    assert type(legacy.model).__name__ == "OSNet"

    ref = np.concatenate([legacy.extract(c) for c in person_crops])
    ours = OSNetEncoder(weights, batch_size=8).encode_images(person_crops)

    cos = cosine_rows(ref, ours)
    assert cos.min() > 0.9999, f"min cosine {cos.min():.6f}"
    # Ranking of the gallery against each query photo is unchanged.
    np.testing.assert_array_equal(
        np.argsort(-(ref[:4] @ ref.T), axis=1), np.argsort(-(ours[:4] @ ours.T), axis=1)
    )
