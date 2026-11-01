from __future__ import annotations

import json
from pathlib import Path

import pytest

from person_search.eval.frame_metrics import (
    FrameLabels,
    FrameMetrics,
    hit_rate,
    load_cases,
    metrics_for_selection,
    metrics_from_labels,
    system_metrics,
)

REPORT_LABELS = Path(__file__).resolve().parents[1] / "golden" / "data" / "labels"


def _write(tmp: Path, case: str, tp, fp, fn) -> None:
    d = tmp / case
    d.mkdir(parents=True)
    (d / "evaluation.json").write_text(
        json.dumps({"true_positives": tp, "false_positives": fp, "false_negatives": fn})
    )


def test_metrics_arithmetic():
    m = FrameMetrics(tp=8, fp=2, fn=8)
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(2 * 0.8 * 0.5 / 1.3)
    assert m.contamination == pytest.approx(0.2)
    assert m.hit


def test_empty_selection_is_a_miss():
    m = FrameMetrics(0, 0, 5)
    assert (m.precision, m.recall, m.f1, m.hit) == (0.0, 0.0, 0.0, False)


def test_system_is_micro_averaged(tmp_path):
    _write(tmp_path, "test_case_001", [1, 2, 3], [4], [])
    _write(tmp_path, "test_case_002", [1], [], [2, 3, 4])
    cases = {k: metrics_from_labels(v) for k, v in load_cases(tmp_path).items()}
    s = system_metrics(cases)
    assert (s.tp, s.fp, s.fn) == (4, 1, 3)
    assert hit_rate(cases) == 1.0


def test_new_selection_scored_against_ground_truth():
    labels = FrameLabels(frozenset({1, 2, 3}), frozenset({9}), frozenset({4}))
    m = metrics_for_selection([2, 3, 4, 7], labels)  # GT = {1,2,3,4}
    assert (m.tp, m.fp, m.fn) == (3, 1, 1)


def test_reproduces_report_f1():
    """The 4 committed evaluation.json files give the report's 89.05% system F1."""
    if not REPORT_LABELS.exists():
        pytest.skip("labels not vendored")
    cases = {k: metrics_from_labels(v) for k, v in load_cases(REPORT_LABELS).items()}
    assert len(cases) == 4
    s = system_metrics(cases)
    assert (s.tp, s.fp, s.fn) == (1073, 74, 190)
    assert s.f1 == pytest.approx(0.8904564315352698)
    assert hit_rate(cases) == 1.0
