from __future__ import annotations

import numpy as np
import pytest

from person_search.core.fusion import late_fusion, legacy_embedding_sum, normalize_scores


def test_minmax_maps_to_unit_range():
    np.testing.assert_allclose(normalize_scores(np.array([0.1, 0.2, 0.3])), [0.0, 0.5, 1.0])


def test_zscore_has_zero_mean_unit_std():
    z = normalize_scores(np.array([1.0, 2.0, 3.0, 4.0]), "zscore")
    assert z.mean() == pytest.approx(0.0)
    assert z.std() == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_constant_scores_carry_no_evidence(method):
    np.testing.assert_array_equal(normalize_scores(np.full(4, 0.7), method), np.zeros(4))


def test_empty_scores():
    assert normalize_scores(np.array([])).size == 0


def test_late_fusion_endpoints_select_one_model():
    photo = np.array([0.2, 0.9, 0.5])
    text = np.array([0.15, 0.10, 0.12])  # IRRA scores live on a compressed scale
    np.testing.assert_allclose(late_fusion(photo, text, 1.0), normalize_scores(photo))
    np.testing.assert_allclose(late_fusion(photo, text, 0.0), normalize_scores(text))


def test_late_fusion_is_scale_invariant_per_model():
    photo = np.array([0.2, 0.9, 0.5])
    text = np.array([0.15, 0.10, 0.12])
    a = late_fusion(photo, text, 0.5)
    b = late_fusion(photo * 3 + 1, text * 10 - 2, 0.5)
    np.testing.assert_allclose(a, b)


def test_late_fusion_rejects_bad_input():
    with pytest.raises(ValueError):
        late_fusion(np.zeros(3), np.zeros(2), 0.5)
    with pytest.raises(ValueError):
        late_fusion(np.zeros(3), np.zeros(3), 1.5)


def test_legacy_embedding_sum_reproduces_repo1():
    img = np.array([1.0, 0.0, 0.0])
    txt = np.array([0.0, 1.0, 0.0])
    fused = legacy_embedding_sum(img, txt, alpha=0.5)
    np.testing.assert_allclose(fused, np.array([1.0, 1.0, 0.0]) / np.sqrt(2), atol=1e-6)
