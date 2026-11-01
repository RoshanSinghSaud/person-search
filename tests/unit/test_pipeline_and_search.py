"""Ingest pipeline and search, wired with fakes (no models, no real video)."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import pytest

from person_search.config import SamplingSettings, ScoringSettings
from person_search.ingest.pipeline import IngestPipeline
from person_search.search import hybrid_search, photo_search, text_search, to_segments
from person_search.search.results import FrameHit
from person_search.types import IRRA, OSNET, BBox, Detection, Observation, TrackState


class FakeDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(5, 5, 20, 30, 0.9)]


class FakeTracker:
    """Two confirmed tracks per frame; 'tiny' is below the min box size."""

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def update(self, detections, frame) -> list[TrackState]:
        return [
            TrackState("1", (5.6, 5.2, 25.9, 35.1)),
            TrackState("2", (-4.0, 10.0, 30.0, 44.0)),  # clipped at the left edge
            TrackState("tiny", (0.0, 0.0, 5.0, 5.0)),
        ]


class FakeEncoder:
    def __init__(self, name: str, dim: int = 4) -> None:
        self.name, self.dim, self.version = name, dim, f"{name}-test"
        self.batches: list[int] = []

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        self.batches.append(len(images))
        v = np.array([[img.mean(), 1.0, 0.0, 0.0] for img in images], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48))
    for i in range(9):
        w.write(np.full((48, 64, 3), 20 * i, dtype=np.uint8))
    w.release()
    return path


def test_ingest_records_every_sample_and_embeds_every_third(video):
    osnet, irra = FakeEncoder(OSNET), FakeEncoder(IRRA)
    tracker = FakeTracker()
    pipe = IngestPipeline(
        FakeDetector(), tracker, {OSNET: osnet, IRRA: irra}, sampling=SamplingSettings()
    )
    res = pipe.run(video)
    assert tracker.resets == 1
    assert res.n_samples == 9
    assert res.track_ids == ["1", "2"]  # "tiny" dropped
    assert len(res.observations) == 18
    embedded = sorted({o.sample_index for o in res.observations if o.has_embeddings})
    assert embedded == [2, 5, 8]  # legacy: processed_frames % 3 == 0 (1-based)
    # one batch per embedding frame, holding both tracks' crops
    assert osnet.batches == [2, 2, 2] and irra.batches == [2, 2, 2]
    obs = res.by_track()["2"][0]
    assert obs.bbox == BBox(0, 10, 30, 44)
    assert obs.frame_index == 1 and obs.sample_index == 0
    assert set(res.encoder_versions) == {OSNET, IRRA}
    assert "detect" in res.timings and "embed:osnet" in res.timings


def _obs(tid: str, frame: int, osnet: float | None, irra: float | None = None) -> Observation:
    emb = {}
    if osnet is not None:
        emb[OSNET] = np.array([osnet, np.sqrt(1 - osnet**2)], dtype=np.float32)
    if irra is not None:
        emb[IRRA] = np.array([irra, np.sqrt(1 - irra**2)], dtype=np.float32)
    return Observation(tid, frame, frame - 1, BBox(0, 0, 10, 20), emb)


QUERY = np.array([1.0, 0.0], dtype=np.float32)


def test_photo_search_end_to_end():
    observations = [
        _obs("a", 1, None), _obs("a", 2, None), _obs("a", 3, 0.9),
        _obs("a", 4, None), _obs("a", 5, None), _obs("a", 6, 0.92),
        _obs("b", 3, 0.3), _obs("b", 6, 0.35),
    ]  # fmt: skip
    res = photo_search(observations, QUERY, ScoringSettings())
    assert res.matched and res.decision.accepted == ("a",)
    assert res.selected_frames == [1, 2, 3, 4, 5, 6]
    assert [(s.start_frame, s.end_frame) for s in res.segments()] == [(1, 6)]


def test_photo_search_no_match():
    res = photo_search([_obs("a", 3, 0.4)], QUERY, ScoringSettings(base_threshold=0.65))
    assert not res.matched and res.hits == ()


def test_text_search_ranks_tracks():
    observations = [_obs("a", 3, None, 0.10), _obs("b", 3, None, 0.14), _obs("c", 3, None, 0.12)]
    ranked = text_search(observations, QUERY, top_k=2)
    assert [t.track_id for t in ranked] == ["b", "c"]


def test_hybrid_search_uses_both_spaces():
    observations = [
        _obs("a", 3, 0.90, 0.10),  # photo says a
        _obs("b", 3, 0.50, 0.15),  # text says b
    ]
    from person_search.config import FusionSettings

    photo_heavy = hybrid_search(observations, QUERY, QUERY, fusion=FusionSettings(alpha=0.9))
    text_heavy = hybrid_search(observations, QUERY, QUERY, fusion=FusionSettings(alpha=0.1))
    assert photo_heavy[0].track_id == "a"
    assert text_heavy[0].track_id == "b"


def test_segments_bridge_small_gaps():
    hits = [FrameHit(f, f - 1, "a", BBox(0, 0, 1, 1)) for f in (1, 2, 3, 5, 6, 10)]
    segs = to_segments(hits, max_gap=1)
    assert [(s.start_sample, s.end_sample) for s in segs] == [(0, 2), (4, 5), (9, 9)]
    assert [(s.start_sample, s.end_sample) for s in to_segments(hits, max_gap=2)] == [
        (0, 5),
        (9, 9),
    ]


def test_ingest_result_round_trips_through_npz(video, tmp_path):
    pipe = IngestPipeline(FakeDetector(), FakeTracker(), {OSNET: FakeEncoder(OSNET)})
    res = pipe.run(video)
    loaded = type(res).load(res.save(tmp_path / "obs"))
    assert loaded.video == res.video and loaded.n_samples == res.n_samples
    assert len(loaded.observations) == len(res.observations)
    for a, b in zip(res.observations, loaded.observations, strict=True):
        assert (a.track_id, a.frame_index, a.sample_index, a.bbox) == (
            b.track_id,
            b.frame_index,
            b.sample_index,
            b.bbox,
        )
        assert a.embeddings.keys() == b.embeddings.keys()
        for k in a.embeddings:
            np.testing.assert_array_equal(a.embeddings[k], b.embeddings[k])
