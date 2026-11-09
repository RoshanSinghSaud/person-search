"""Vector index (Day 10) and indexed search (Days 11-12) on synthetic data.

Both index implementations run the same tests: the in-memory fake and Qdrant's
in-process mode. Indexed search must return exactly what a full scan of the
observations returns (``search/photo.py``, ``text.py``, ``hybrid.py``), including
when the first candidate cut-off is too small and retrieval has to repeat.
"""

from __future__ import annotations

import numpy as np
import pytest

from person_search.config import FusionSettings, ScoringSettings
from person_search.errors import ConfigurationError
from person_search.index import InMemoryIndex, Point, SearchFilter, point_id, points_from_ingest
from person_search.index.points import observations_by_video
from person_search.index.qdrant import QdrantIndex, collection_name
from person_search.ingest.pipeline import IngestResult
from person_search.search import hybrid_search, photo_search, text_search
from person_search.search.indexed import IndexedSearch
from person_search.types import IRRA, OSNET, BBox, Observation, VideoInfo

VERSIONS = {OSNET: "osnet-test", IRRA: "irra-test"}
DIM = 16
DIMS = {OSNET: DIM, IRRA: DIM}


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def make_ingest(seed: int, n_tracks: int = 12, n_frames: int = 45, path: str = "v.mp4"):
    """Tracks with their own appearance centre; every 3rd frame embedded."""
    rng = np.random.default_rng(seed)
    centres = {
        OSNET: _unit(rng.normal(size=(n_tracks, DIM))),
        IRRA: _unit(rng.normal(size=(n_tracks, DIM))),
    }
    obs = []
    for f in range(n_frames):
        for t in range(n_tracks):
            if (f + t) % 5 == 0 and f > 3:  # tracks drop out now and then
                continue
            o = Observation(str(t + 1), f + 1, f, BBox(t, t, t + 20, t + 40))
            if (f + 1) % 3 == 0:
                for name in (OSNET, IRRA):
                    o.embeddings[name] = _unit(centres[name][t] + 0.35 * rng.normal(size=DIM))
            obs.append(o)
    info = VideoInfo(path, 12.0, n_frames, 64, 48)
    return IngestResult(info, obs, n_frames, dict(VERSIONS)), centres


@pytest.fixture(params=["memory", "qdrant"])
def index(request):
    if request.param == "memory":
        yield InMemoryIndex(VERSIONS)
    else:
        idx = QdrantIndex.in_memory(VERSIONS, DIMS)
        yield idx
        idx.close()


def test_point_ids_are_stable_and_distinct():
    a = point_id("abc", 10, "7")
    assert a == point_id("abc", 10, "7")
    assert (
        len({a, point_id("abc", 11, "7"), point_id("abc", 10, "8"), point_id("abd", 10, "7")}) == 4
    )


def test_points_keep_every_observation_and_only_embedded_ones_have_vectors():
    ingest, _ = make_ingest(0)
    pts = points_from_ingest(ingest, "vid0", camera="cam1")
    assert len(pts) == len(ingest.observations)
    assert len({p.id for p in pts}) == len(pts)
    emb = [p for p in pts if p.payload["embedded"]]
    assert emb and all(set(p.vectors) == {OSNET, IRRA} for p in emb)
    assert all(not p.vectors for p in pts if not p.payload["embedded"])
    p = pts[5]
    assert p.payload["camera"] == "cam1"
    assert p.payload["t_seconds"] == pytest.approx(p.payload["frame_index"] / 12.0, abs=1e-3)


def test_round_trip_restores_observations_in_ingest_order(index):
    ingest, _ = make_ingest(1)
    index.upsert(points_from_ingest(ingest, "vid1"))
    keys = {("vid1", t) for t in ingest.track_ids}
    back = observations_by_video(index.fetch_tracks(keys))["vid1"]
    assert [(o.track_id, o.frame_index) for o in back] == [
        (o.track_id, o.frame_index) for o in ingest.observations
    ]
    for a, b in zip(back, ingest.observations, strict=True):
        assert a.bbox == b.bbox and set(a.embeddings) == set(b.embeddings)
        for k in a.embeddings:
            assert np.array_equal(a.embeddings[k], b.embeddings[k])  # bit-identical


def test_reingesting_a_video_adds_zero_points(index):
    ingest, _ = make_ingest(2)
    pts = points_from_ingest(ingest, "vid2")
    index.upsert(pts)
    n = index.count()
    index.upsert(points_from_ingest(ingest, "vid2"))
    assert index.count() == n == len(pts)


def test_filters_by_video_camera_and_time(index):
    a, _ = make_ingest(3, path="a.mp4")
    b, _ = make_ingest(4, path="b.mp4")
    index.upsert(points_from_ingest(a, "va", camera="gate"))
    index.upsert(points_from_ingest(b, "vb", camera="library"))
    assert index.count(SearchFilter(video_ids=("va",))) == len(a.observations)
    assert index.count(SearchFilter(cameras=("library",))) == len(b.observations)
    q = np.ones(DIM, np.float32) / np.sqrt(DIM)
    hits = index.search(OSNET, q, limit=50, flt=SearchFilter(cameras=("gate",)))
    assert hits and all(h.payload["video_id"] == "va" for h in hits)
    late = index.search(OSNET, q, limit=500, flt=SearchFilter(t_min=2.0))
    assert late and all(h.payload["t_seconds"] >= 2.0 for h in late)
    assert set(index.videos()) == {"va", "vb"}


def test_search_scores_are_dot_products_best_first(index):
    ingest, centres = make_ingest(5)
    index.upsert(points_from_ingest(ingest, "v"))
    q = centres[OSNET][3]
    hits = index.search(OSNET, q, limit=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    best = next(
        o for o in ingest.observations if point_id("v", o.frame_index, o.track_id) == hits[0].id
    )
    assert hits[0].score == pytest.approx(float(np.dot(q, best.embeddings[OSNET])), abs=1e-6)
    assert all(h.score >= 0.5 for h in index.search(OSNET, q, limit=100, score_threshold=0.5))


def test_qdrant_collection_is_named_for_its_model_versions(tmp_path):
    assert collection_name({IRRA: "irra-v1", OSNET: "osnet-v1"}) == "detections__irra-v1__osnet-v1"
    idx = QdrantIndex.local(tmp_path / "q", VERSIONS, DIMS)
    idx.upsert([Point(point_id("v", 1, "1"), {"video_id": "v", "track_id": "1"}, {})])
    idx.close()
    again = QdrantIndex.local(tmp_path / "q", VERSIONS, create=False)
    assert again.count() == 1
    again.close()
    with pytest.raises(ConfigurationError, match="no index"):
        QdrantIndex.local(tmp_path / "q", {OSNET: "osnet-v0", IRRA: "irra-test"}, create=False)


# --------------------------------------------------------------- indexed search
SCORING = ScoringSettings(base_threshold=0.6)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_indexed_photo_search_equals_full_scan(index, seed):
    ingest, centres = make_ingest(seed)
    index.upsert(points_from_ingest(ingest, "v"))
    searcher = IndexedSearch(index, SCORING, candidate_limit=4)  # forces repeated retrieval
    for t in (0, 4, 9):
        q = centres[OSNET][t]
        full = photo_search(ingest.observations, q, SCORING, OSNET)
        got = searcher.photo(q).per_video["v"]
        assert got.matched == full.matched
        assert got.decision.accepted == full.decision.accepted
        assert got.selected_samples == full.selected_samples
        assert [h.track_id for h in got.hits] == [h.track_id for h in full.hits]


def test_indexed_photo_search_with_no_match_still_lists_closest_tracks(index):
    ingest, _ = make_ingest(13)
    index.upsert(points_from_ingest(ingest, "v"))
    q = _unit(np.random.default_rng(99).normal(size=DIM))
    strict = ScoringSettings(base_threshold=0.99)
    res = IndexedSearch(index, strict).photo(q)
    assert not res.matched
    assert res.per_video["v"].decision.ranked  # context tracks for display
    assert not photo_search(ingest.observations, q, strict, OSNET).matched


@pytest.mark.parametrize("seed", [20, 21])
def test_indexed_text_search_equals_full_scan(index, seed):
    ingest, centres = make_ingest(seed)
    index.upsert(points_from_ingest(ingest, "v"))
    searcher = IndexedSearch(index, SCORING, candidate_limit=3)
    for t in (1, 6):
        q = centres[IRRA][t]
        full = text_search(ingest.observations, q, top_k=5, scoring=SCORING)
        got, stats = searcher.text(q, top_k=5)
        assert [r.track_id for r in got] == [s.track_id for s in full]
        assert [r.score for r in got] == pytest.approx([s.score for s in full], abs=1e-6)
        assert stats.rounds >= 1


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_indexed_hybrid_search_equals_full_scan(index, alpha):
    ingest, centres = make_ingest(30)
    index.upsert(points_from_ingest(ingest, "v"))
    fusion = FusionSettings(alpha=alpha)
    searcher = IndexedSearch(index, SCORING, fusion, candidate_limit=3)
    p, t = centres[OSNET][2], centres[IRRA][2]
    full = hybrid_search(ingest.observations, p, t, top_k=5, fusion=fusion, scoring=SCORING)
    got, _ = searcher.hybrid(p, t, top_k=5)
    assert [r.track_id for r in got] == [s.track_id for s in full]
    assert [r.score for r in got] == pytest.approx([s.score for s in full], abs=1e-5)


def test_indexed_search_respects_the_video_filter(index):
    a, ca = make_ingest(40, path="a.mp4")
    b, _ = make_ingest(41, path="b.mp4")
    index.upsert(points_from_ingest(a, "va"))
    index.upsert(points_from_ingest(b, "vb"))
    s = IndexedSearch(index, SCORING)
    only_a = SearchFilter(video_ids=("va",))
    assert set(s.photo(ca[OSNET][0], only_a).per_video) <= {"va"}
    ranked, _ = s.text(ca[IRRA][0], top_k=5, flt=only_a)
    assert {r.video_id for r in ranked} == {"va"}
    full = text_search(a.observations, ca[IRRA][0], top_k=5, scoring=SCORING)
    assert [r.track_id for r in ranked] == [x.track_id for x in full]


def test_qdrant_track_cache_and_filter_scan_fetch_the_same_points():
    from qdrant_client import QdrantClient

    ingest, _ = make_ingest(50)
    client = QdrantClient(":memory:")
    cached = QdrantIndex(client, VERSIONS, DIMS, local=True, track_cache=True)
    scan = QdrantIndex(client, VERSIONS, DIMS, local=True, track_cache=False)
    cached.fetch_tracks([("v", "1")])  # build the map before the write: upserts must update it
    cached.upsert(points_from_ingest(ingest, "v"))
    keys = [("v", "2"), ("v", "7"), ("v", "nope")]
    a = {p.id: p for p in cached.fetch_tracks(keys)}
    b = {p.id: p for p in scan.fetch_tracks(keys)}
    assert a.keys() == b.keys() and a
    assert all(np.array_equal(a[i].vectors.get(OSNET, 0), b[i].vectors.get(OSNET, 0)) for i in a)
