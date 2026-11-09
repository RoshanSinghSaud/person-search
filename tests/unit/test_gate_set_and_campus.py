"""Cached gate crops round-trip to the ingest's vectors; campus split loading and protocols."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from person_search.eval.campus import load_market_split, similarity
from person_search.eval.gate_set import export_case, load_case
from person_search.eval.reid import evaluate_reid
from person_search.ingest.pipeline import IngestPipeline
from tests.unit.test_pipeline_and_search import FakeDetector, FakeEncoder, FakeTracker


@pytest.fixture
def textured_video(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48))
    for _ in range(9):
        w.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    w.release()
    return path


def test_cached_case_reproduces_the_ingest_vectors(tmp_path, textured_video):
    enc = FakeEncoder("osnet")
    ingest = IngestPipeline(FakeDetector(), FakeTracker(), {"osnet": enc}).run(textured_video)
    path = export_case("c1", textured_video, ingest, tmp_path / "c1.npz", extra_meta={"note": "x"})
    case = load_case(path)
    assert case.case == "c1" and case.meta["note"] == "x"
    assert [(o.track_id, o.sample_index, o.bbox) for o in case.observations] == [
        (o.track_id, o.sample_index, o.bbox) for o in ingest.observations
    ]
    assert len(case.crops) == sum(o.has_embeddings for o in ingest.observations)
    again = case.embed({"osnet": FakeEncoder("osnet")})
    for a, b in zip(again, ingest.observations, strict=True):
        assert set(a.embeddings) == set(b.embeddings)
        if "osnet" in b.embeddings:
            assert np.array_equal(a.embeddings["osnet"], b.embeddings["osnet"])


def test_load_market_split_parses_ids_and_cameras(tmp_path):
    img = np.zeros((8, 4, 3), np.uint8)
    for folder, names in {
        "query": ["0071_c1s1_000001_00.jpg", "0072_c2s1_000001_00.jpg"],
        "bounding_box_test": [
            "0071_c2s1_000010_00.jpg",
            "-1_c1s1_000000_00.jpg",
            "0072_c3s1_000020_00.jpg",
        ],
    }.items():
        (tmp_path / folder).mkdir()
        for n in names:
            cv2.imwrite(str(tmp_path / folder / n), img)
    split = load_market_split(tmp_path)
    assert split.q_pids.tolist() == [71, 72] and split.q_camids.tolist() == [0, 1]
    assert split.g_pids.tolist() == [71, 72]  # junk (-1) dropped
    assert split.g_camids.tolist() == [1, 2]
    assert split.n_ids == 2


def test_protocols_rank_identically_on_unit_vectors():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(5, 8))
    g = rng.normal(size=(9, 8))
    qn, gn = (
        q / np.linalg.norm(q, axis=1, keepdims=True),
        g / np.linalg.norm(g, axis=1, keepdims=True),
    )
    cos = similarity(qn, gn, "cosine")
    euc = similarity(qn, gn, "euclidean_raw")
    assert (np.argsort(-cos, axis=1) == np.argsort(-euc, axis=1)).all()
    q_p, g_p = np.arange(5) % 3, np.arange(9) % 3
    a = evaluate_reid(cos, q_p, g_p, np.zeros(5), np.ones(9))
    b = evaluate_reid(euc, q_p, g_p, np.zeros(5), np.ones(9))
    assert a.mAP == pytest.approx(b.mAP)


def test_vectors_can_be_reused_from_the_ingest_after_a_spot_check(tmp_path, textured_video):
    ingest = IngestPipeline(FakeDetector(), FakeTracker(), {"osnet": FakeEncoder("osnet")}).run(
        textured_video
    )
    case = load_case(export_case("c", textured_video, ingest, tmp_path / "c.npz"))
    assert case.spot_check(ingest, "osnet", FakeEncoder("osnet")) > 0
    reused = case.from_ingest(ingest, "osnet")
    fresh = case.embed({"osnet": FakeEncoder("osnet")})
    for a, b in zip(reused, fresh, strict=True):
        assert a.embeddings.keys() == b.embeddings.keys()
        for k in a.embeddings:
            assert np.array_equal(a.embeddings[k], b.embeddings[k])

    class Off(FakeEncoder):
        def encode_images(self, images):
            return super().encode_images(images) + 1e-3

    with pytest.raises(AssertionError, match="differs"):
        case.spot_check(ingest, "osnet", Off("osnet"))
