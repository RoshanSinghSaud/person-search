"""mAP / CMC on toy problems solved by hand."""

from __future__ import annotations

import numpy as np
import pytest

from person_search.eval.reid import evaluate_reid, evaluate_text_to_image


def test_text_to_image_hand_solved():
    # 2 queries, 4 gallery items. Similarities already in the ranking order we want.
    sim = np.array(
        [
            [0.9, 0.8, 0.7, 0.6],  # q0 (pid 1): ranks -> g0(1) g1(2) g2(1) g3(3)
            [0.1, 0.9, 0.3, 0.8],  # q1 (pid 3): ranks -> g1(2) g3(3) g2(1) g0(1)
        ]
    )
    q = np.array([1, 3])
    g = np.array([1, 2, 1, 3])
    m = evaluate_text_to_image(sim, q, g, max_rank=4)
    # q0: matches [1,0,1,0] -> AP = (1/1 + 2/3) / 2 = 5/6 ; rank-1 hit
    # q1: matches [0,1,0,0] -> AP = 1/2 ; first hit at rank 2
    assert m.mAP == pytest.approx((5 / 6 + 1 / 2) / 2)
    np.testing.assert_allclose(m.cmc, [0.5, 1.0, 1.0, 1.0])
    assert m.rank(1) == pytest.approx(0.5)
    assert m.n_valid_queries == 2


def test_reid_removes_same_camera_same_identity():
    # q0 = pid 1 cam 0. g0 is pid 1 on cam 0 (junk, removed); g1 pid 1 cam 1 is the true match.
    sim = np.array([[0.99, 0.5, 0.9]])
    m = evaluate_reid(sim, [1], [1, 1, 2], [0], [0, 1, 1], max_rank=3)
    # after removing g0: ranking g2 (wrong) then g1 (right) -> AP 1/2, rank-1 miss
    assert m.mAP == pytest.approx(0.5)
    assert m.rank(1) == pytest.approx(0.0)
    assert m.rank(2) == pytest.approx(1.0)


def test_query_without_valid_match_is_skipped():
    sim = np.array([[0.9, 0.1], [0.2, 0.8]])
    # q1 (pid 9) has no match anywhere
    m = evaluate_text_to_image(sim, [1, 9], [1, 2])
    assert m.n_valid_queries == 1
    assert m.mAP == pytest.approx(1.0)


def test_all_queries_invalid_raises():
    with pytest.raises(ValueError):
        evaluate_text_to_image(np.array([[0.5]]), [1], [2])


def test_matches_torchreid_reference():
    rank = pytest.importorskip("torchreid.reid.metrics.rank")
    rng = np.random.default_rng(0)
    nq, ng = 12, 60
    q_pids = rng.integers(0, 8, nq)
    g_pids = rng.integers(0, 8, ng)
    q_cam = rng.integers(0, 3, nq)
    g_cam = rng.integers(0, 3, ng)
    sim = rng.normal(size=(nq, ng))
    ours = evaluate_reid(sim, q_pids, g_pids, q_cam, g_cam, max_rank=20)
    cmc, mAP = rank.evaluate_rank(
        -sim, q_pids, g_pids, q_cam, g_cam, max_rank=20, use_metric_cuhk03=False, use_cython=False
    )
    assert ours.mAP == pytest.approx(mAP, abs=1e-6)
    np.testing.assert_allclose(ours.cmc[:20], cmc, atol=1e-6)
