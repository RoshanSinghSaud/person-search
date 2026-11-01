"""Frame-level precision / recall / F1 / contamination on the test cases.

Ported from repo 2 ``utilities/calculate_system_metrics.py`` and
``utilities/generate_evaluation_report.py``. System numbers are micro-averaged
(summed TP/FP/FN), which is how the report's 89.05% F1 is computed.

Two ways to score:

* :func:`metrics_from_labels` - the counts in an ``evaluation.json``
  (reproduces the report's table exactly);
* :func:`metrics_for_selection` - a new selection of sampled frames against the
  ground truth in an ``evaluation.json`` (TP + FN = frames the target is in).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameLabels:
    true_positives: frozenset[int]
    false_positives: frozenset[int]
    false_negatives: frozenset[int]

    @classmethod
    def load(cls, path: str | Path) -> FrameLabels:
        data = json.loads(Path(path).read_text())
        return cls(
            frozenset(data.get("true_positives", [])),
            frozenset(data.get("false_positives", [])),
            frozenset(data.get("false_negatives", [])),
        )

    @property
    def ground_truth(self) -> frozenset[int]:
        """Sampled frames where the target is present."""
        return self.true_positives | self.false_negatives

    @property
    def labelled_selection(self) -> frozenset[int]:
        """Frames the labelled run selected (TP + FP)."""
        return self.true_positives | self.false_positives


@dataclass(frozen=True)
class FrameMetrics:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def contamination(self) -> float:
        """Share of output frames that show the wrong person (FP / (TP + FP))."""
        return self.fp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def hit(self) -> bool:
        return self.tp > 0

    def __add__(self, other: FrameMetrics) -> FrameMetrics:
        return FrameMetrics(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self) | {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "contamination": self.contamination,
            "hit": self.hit,
        }


def metrics_from_labels(labels: FrameLabels) -> FrameMetrics:
    return FrameMetrics(
        len(labels.true_positives), len(labels.false_positives), len(labels.false_negatives)
    )


def metrics_for_selection(selected: Iterable[int], labels: FrameLabels) -> FrameMetrics:
    sel, gt = set(selected), labels.ground_truth
    return FrameMetrics(len(sel & gt), len(sel - gt), len(gt - sel))


def system_metrics(per_case: Mapping[str, FrameMetrics]) -> FrameMetrics:
    total = FrameMetrics(0, 0, 0)
    for m in per_case.values():
        total = total + m
    return total


def hit_rate(per_case: Mapping[str, FrameMetrics]) -> float:
    return sum(m.hit for m in per_case.values()) / len(per_case) if per_case else 0.0


def load_cases(cases_dir: str | Path, pattern: str = "test_case_*") -> dict[str, FrameLabels]:
    """``{case_name: FrameLabels}`` for every ``<case>/evaluation.json`` under ``cases_dir``."""
    root = Path(cases_dir)
    return {
        p.parent.name: FrameLabels.load(p) for p in sorted(root.glob(f"{pattern}/evaluation.json"))
    }


def format_table(per_case: Mapping[str, FrameMetrics]) -> str:
    rows = [f"{'case':<16} {'P':>7} {'R':>7} {'F1':>7} {'contam':>7}  hit"]
    for name, m in per_case.items():
        rows.append(
            f"{name:<16} {m.precision:7.2%} {m.recall:7.2%} {m.f1:7.2%} {m.contamination:7.2%}  "
            f"{'yes' if m.hit else 'no'}"
        )
    s = system_metrics(per_case)
    rows.append(
        f"{'SYSTEM':<16} {s.precision:7.2%} {s.recall:7.2%} {s.f1:7.2%} {s.contamination:7.2%}  "
        f"{hit_rate(per_case):.0%}"
    )
    return "\n".join(rows)
