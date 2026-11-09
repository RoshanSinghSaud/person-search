"""Text end-to-end metrics (Day 9) on hand-solved examples, plus the committed test set."""

from __future__ import annotations

import clip
import pytest

from person_search.core.scoring import rank_tracks
from person_search.eval.text_e2e import (
    QueryResult,
    is_target,
    load_targets,
    load_text_queries,
    rank_with_evidence,
    summarize,
)
from person_search.types import BBox, Observation
from tests.conftest import GOLDEN_ROOT

CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")


def _obs(track: str, sample: int) -> Observation:
    return Observation(track, sample + 1, sample, BBox(0, 0, 10, 20))


def test_is_target_uses_inclusive_segments():
    seg = {"4": [(2, 5), (9, 9)]}
    assert [is_target(seg, "4", s) for s in (1, 2, 5, 6, 9)] == [False, True, True, False, True]
    assert not is_target(seg, "5", 3)


def test_rank_with_evidence_hand_solved():
    # Track A: target in samples 0-1 only (an identity switch at sample 2).
    # Track B: never the target.
    obs = [_obs("A", 0), _obs("A", 1), _obs("A", 2), _obs("A", 3), _obs("B", 0), _obs("B", 1)]
    scores = [0.9, 0.8, 0.1, None, 0.5, 0.6]
    seg = {"A": [(0, 1)]}
    ranked = rank_with_evidence(obs, scores, seg)
    a, b = ranked
    # A: scored [0.9, 0.8, 0.1] -> top 1 of 3 = 0.9; evidence 1/1; purity 2/3
    assert (a.track_id, a.score, a.evidence_share, a.purity) == (
        "A",
        0.9,
        1.0,
        pytest.approx(2 / 3),
    )
    assert a.target_samples == {0, 1}
    # B: [0.5, 0.6] -> top 1 of 2 = 0.6
    assert (b.track_id, b.score, b.evidence_share, b.purity) == ("B", 0.6, 0.0, 0.0)
    assert a.hit() and a.hit("purity") and not b.hit()


def test_ranking_matches_the_papers_track_score():
    obs = [_obs(t, s) for t in "XYZ" for s in range(6)]
    scores = [0.1 * ((i * 7) % 10) if i % 3 != 1 else None for i in range(len(obs))]
    mine = [r.track_id for r in rank_with_evidence(obs, scores, {})]
    frames: dict[str, list] = {}
    for o, s in zip(obs, scores, strict=True):
        frames.setdefault(o.track_id, []).append((o.frame_index, s))
    assert mine == [t.track_id for t in rank_tracks(frames)]


def test_query_result_metrics():
    obs = [_obs("A", 0), _obs("B", 0), _obs("C", 0)]
    ranked = rank_with_evidence(obs, [0.2, 0.9, 0.5], {"A": [(0, 0)]})  # order B, C, A
    r = QueryResult("q", "case", "short", "text", ranked, gt_frames=4)
    assert r.first_hit() == 3 and not r.hit_at(1) and r.hit_at(5)
    assert r.frame_recall_at1 == 0.0
    hit_first = QueryResult(
        "q2", "case", "short", "text", rank_with_evidence(obs, [0.9, 0.2, 0.5], {"A": [(0, 0)]}), 4
    )
    assert hit_first.frame_recall_at1 == 0.25
    s = summarize([r, hit_first])
    assert (s.n, s.hit1, s.hit5) == (2, 1, 2)
    assert s.mrr == pytest.approx((1 / 3 + 1) / 2)


def test_committed_text_queries_are_well_formed():
    queries = load_text_queries(GOLDEN_ROOT / "text_queries.yaml")
    assert len(queries) == 12
    assert {q.case for q in queries} == set(CASES)
    assert len({q.id for q in queries}) == 12
    for case in CASES:
        assert sorted(q.level for q in queries if q.case == case) == ["detailed", "medium", "short"]
    # IRRA inserts 8 prompt tokens into CLIP's 77-token context: EOT must stay inside it.
    tokens = clip.tokenize([q.text for q in queries], context_length=77, truncate=True)
    eot = tokens.argmax(dim=-1)
    assert int(eot.max()) + 8 <= 76


def test_committed_targets_cover_every_case():
    targets = load_targets(GOLDEN_ROOT / "text_targets.yaml")
    assert set(targets) == set(CASES)
    for segs in targets.values():
        assert segs
        for ranges in segs.values():
            assert all(a <= b for a, b in ranges)
