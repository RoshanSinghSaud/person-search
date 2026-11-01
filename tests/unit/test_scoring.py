"""The paper's scoring, on hand-made similarity lists."""

from __future__ import annotations

import pytest

from person_search.config import ScoringSettings
from person_search.core.scoring import (
    decide,
    effective_threshold,
    forward_fill,
    rank_tracks,
    select_frames,
    track_score,
)

CFG = ScoringSettings(base_threshold=0.65, relative_margin=0.1, frame_threshold_ratio=0.9)


class TestTrackScore:
    def test_top_half_of_even_list(self):
        # top 2 of 4 -> mean(0.9, 0.8)
        assert track_score([0.1, 0.9, 0.8, 0.2]) == pytest.approx(0.85)

    def test_odd_list_floors_k(self):
        # 5 // 2 = 2 -> mean(0.9, 0.7)
        assert track_score([0.5, 0.9, 0.7, 0.1, 0.3]) == pytest.approx(0.8)

    def test_single_value_keeps_it(self):
        assert track_score([0.42]) == pytest.approx(0.42)

    def test_empty_scores_zero(self):
        assert track_score([]) == 0.0

    def test_outliers_from_occlusion_are_ignored(self):
        clean = [0.8] * 6
        occluded = [*clean, 0.1, 0.05, 0.0]
        # 9 // 2 = 4 values, all 0.8
        assert track_score(occluded) == pytest.approx(0.8)

    def test_fraction_is_configurable(self):
        assert track_score([1.0, 0.0, 0.0, 0.0], top_fraction=0.25) == pytest.approx(1.0)
        assert track_score([1.0, 0.0, 0.0, 0.0], top_fraction=1.0) == pytest.approx(0.25)


class TestThreshold:
    def test_base_dominates_when_best_is_low(self):
        assert effective_threshold(0.70, 0.65, 0.1) == pytest.approx(0.65)

    def test_margin_dominates_when_best_is_high(self):
        assert effective_threshold(0.90, 0.65, 0.1) == pytest.approx(0.80)

    def test_accepts_fragments_within_margin(self):
        tracks = {
            "a": [(1, 0.90), (2, 0.90)],
            "b": [(3, 0.85), (4, 0.85)],  # same person, broken track
            "c": [(5, 0.70), (6, 0.70)],  # above base, outside margin
        }
        d = decide(rank_tracks(tracks), CFG)
        assert d.accepted == ("a", "b")
        assert d.effective_threshold == pytest.approx(0.80)
        assert d.frame_threshold == pytest.approx(0.72)

    def test_no_match_when_best_below_base(self):
        tracks = {"a": [(1, 0.60)], "b": [(2, 0.50)]}
        d = decide(rank_tracks(tracks), CFG)
        assert not d.matched
        assert d.best is not None and d.best.track_id == "a"
        assert select_frames(tracks, d) == {}

    def test_no_tracks(self):
        d = decide([], CFG)
        assert not d.matched and d.best is None

    def test_ties_keep_first_appearance_order(self):
        tracks = {"late": [(1, 0.7)], "early": [(2, 0.7)]}
        assert [t.track_id for t in rank_tracks(tracks)] == ["late", "early"]


class TestFrameSelection:
    def test_forward_fill_backfills_leading_gap(self):
        assert forward_fill([(1, None), (2, 0.5), (3, None), (4, 0.9), (5, None)]) == [
            (1, 0.5),
            (2, 0.5),
            (3, 0.5),
            (4, 0.9),
            (5, 0.9),
        ]

    def test_forward_fill_all_missing_is_zero(self):
        assert forward_fill([(1, None), (2, None)]) == [(1, 0.0), (2, 0.0)]

    def test_drops_low_segments_inside_accepted_track(self):
        # embeddings every 3rd frame; target leaves at frame 6
        tracks = {
            "t": [
                (1, None), (2, None), (3, 0.90),
                (4, None), (5, None), (6, 0.30),
                (7, None), (8, None), (9, 0.95),
            ]
        }  # fmt: skip
        d = decide(rank_tracks(tracks), CFG)
        assert d.accepted == ("t",)
        # frame floor 0.9 * max(0.65, 0.925 - 0.1) = 0.7425
        assert sorted(select_frames(tracks, d)) == [1, 2, 3, 4, 5, 9]

    def test_frame_claimed_by_higher_track_is_not_reassigned(self):
        tracks = {
            "best": [(10, 0.95), (11, 0.95)],
            "frag": [(11, 0.90), (12, 0.90)],
        }
        d = decide(rank_tracks(tracks), CFG)
        assert select_frames(tracks, d) == {10: "best", 11: "best", 12: "frag"}
