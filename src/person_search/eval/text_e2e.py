"""End-to-end evaluation of text (and hybrid) queries on the 4 test videos (Day 9).

Inputs, all committed under ``tests/golden/data/``:

* ``text_queries.yaml`` - 3 witness descriptions per case (short / medium /
  detailed), written from ``query.jpg`` before any text search ran;
* ``text_targets.yaml`` - for each case, the tracks that show the target and
  the sample ranges in which they do (DeepSORT switches identities, so a track
  can be the target for only part of its life);
* ``labels/<case>/evaluation.json`` - frame-level ground truth (TP + FN).

A query is scored against one video. Its tracks are ranked by the paper's
top-50% mean of per-detection scores (whatever the scores are: IRRA text
similarity, a fused hybrid score, ...). Then for the ranked tracks:

* **evidence share** - of the detections that produced a track's score (its
  top 50%), the share that show the target. A reviewer looking at a returned
  track sees those best-matching detections first.
* **purity** - the share of all the track's scored detections that show the
  target.

Metrics per query:

* ``Hit@k`` - some track in the top k has evidence share > 0.5 (the target is
  what matched). This is the headline metric.
* ``Hit@k (pure)`` - stricter: some track in the top k has purity > 0.5.
* ``first_hit`` - rank of the first such track (1-based; None if not in the
  ranking), giving the mean reciprocal rank.
* ``frame_recall@1`` - ground-truth frames (TP + FN) in which the top-ranked
  track is on the target, over all ground-truth frames of the case.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from person_search.types import Observation

Segments = dict[str, list[tuple[int, int]]]  # track_id -> [(first_sample, last_sample)]


@dataclass(frozen=True, slots=True)
class TextQuery:
    case: str
    id: str
    level: str
    text: str


def load_text_queries(path: str | Path) -> list[TextQuery]:
    data = yaml.safe_load(Path(path).read_text())
    out = []
    for case, block in data["cases"].items():
        for q in block["queries"]:
            out.append(TextQuery(case, q["id"], q["level"], q["text"]))
    return out


def load_targets(path: str | Path) -> dict[str, Segments]:
    data = yaml.safe_load(Path(path).read_text())
    return {
        case: {str(t): [(int(a), int(b)) for a, b in segs] for t, segs in tracks.items()}
        for case, tracks in data["cases"].items()
    }


def is_target(segments: Segments, track_id: str, sample_index: int) -> bool:
    return any(a <= sample_index <= b for a, b in segments.get(track_id, ()))


@dataclass(frozen=True)
class RankedEvidence:
    track_id: str
    score: float
    n_scored: int
    evidence_share: float
    purity: float
    target_samples: frozenset[int]

    def hit(self, rule: str = "evidence") -> bool:
        share = self.evidence_share if rule == "evidence" else self.purity
        return share > 0.5


def rank_with_evidence(
    observations: Sequence[Observation],
    scores: Sequence[float | None],
    segments: Segments,
    top_fraction: float = 0.5,
) -> list[RankedEvidence]:
    """Rank tracks by top-``top_fraction`` mean score and attach target evidence.

    ``scores[i]`` belongs to ``observations[i]`` (None where nothing was
    embedded). Ranking matches :func:`person_search.core.scoring.rank_tracks`:
    stable sort, so tied tracks keep their order of first appearance.
    """
    if len(scores) != len(observations):
        raise ValueError("one score per observation is required")
    per_track: dict[str, list[tuple[float, bool]]] = {}
    target_samples: dict[str, set[int]] = {}
    for o, s in zip(observations, scores, strict=True):
        tgt = is_target(segments, o.track_id, o.sample_index)
        target_samples.setdefault(o.track_id, set())
        if tgt:
            target_samples[o.track_id].add(o.sample_index)
        lst = per_track.setdefault(o.track_id, [])
        if s is not None:
            lst.append((float(s), tgt))
    ranked = []
    for tid, lst in per_track.items():
        if not lst:
            ranked.append(RankedEvidence(tid, 0.0, 0, 0.0, 0.0, frozenset(target_samples[tid])))
            continue
        ordered = sorted(lst, key=lambda x: x[0], reverse=True)
        k = max(1, int(len(ordered) * top_fraction))
        top = ordered[:k]
        ranked.append(
            RankedEvidence(
                tid,
                float(np.mean([s for s, _ in top])),
                len(lst),
                sum(t for _, t in top) / k,
                sum(t for _, t in lst) / len(lst),
                frozenset(target_samples[tid]),
            )
        )
    return sorted(ranked, key=lambda r: r.score, reverse=True)


@dataclass
class QueryResult:
    query_id: str
    case: str
    level: str
    method: str
    ranked: list[RankedEvidence]
    ground_truth: frozenset[int]  # sampled frames where the target is (labels' TP + FN)
    extra: dict[str, Any] = field(default_factory=dict)

    def first_hit(self, rule: str = "evidence") -> int | None:
        return next((i + 1 for i, r in enumerate(self.ranked) if r.hit(rule)), None)

    def hit_at(self, k: int, rule: str = "evidence") -> bool:
        fh = self.first_hit(rule)
        return fh is not None and fh <= k

    @property
    def frame_recall_at1(self) -> float:
        if not self.ranked or not self.ground_truth:
            return 0.0
        return len(self.ranked[0].target_samples & self.ground_truth) / len(self.ground_truth)

    def as_dict(self, top: int = 5) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "case": self.case,
            "level": self.level,
            "method": self.method,
            "hit@1": self.hit_at(1),
            "hit@5": self.hit_at(5),
            "hit@1_pure": self.hit_at(1, "purity"),
            "hit@5_pure": self.hit_at(5, "purity"),
            "first_hit": self.first_hit(),
            "frame_recall@1": round(self.frame_recall_at1, 4),
            "top": [
                {
                    "track": r.track_id,
                    "score": round(r.score, 4),
                    "evidence": round(r.evidence_share, 3),
                    "purity": round(r.purity, 3),
                }
                for r in self.ranked[:top]
            ],
            **self.extra,
        }


@dataclass(frozen=True)
class Summary:
    method: str
    n: int
    hit1: int
    hit5: int
    hit1_pure: int
    hit5_pure: int
    mrr: float
    frame_recall1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n": self.n,
            "hit@1": self.hit1 / self.n,
            "hit@5": self.hit5 / self.n,
            "hit@1_pure": self.hit1_pure / self.n,
            "hit@5_pure": self.hit5_pure / self.n,
            "mrr": self.mrr,
            "frame_recall@1": self.frame_recall1,
            "hit@1_count": f"{self.hit1}/{self.n}",
            "hit@5_count": f"{self.hit5}/{self.n}",
        }


def summarize(results: Sequence[QueryResult], method: str | None = None) -> Summary:
    if not results:
        raise ValueError("no results to summarise")
    fh = [r.first_hit() for r in results]
    return Summary(
        method or results[0].method,
        len(results),
        sum(r.hit_at(1) for r in results),
        sum(r.hit_at(5) for r in results),
        sum(r.hit_at(1, "purity") for r in results),
        sum(r.hit_at(5, "purity") for r in results),
        float(np.mean([1.0 / f if f else 0.0 for f in fh])),
        float(np.mean([r.frame_recall_at1 for r in results])),
    )


def summary_table(rows: Mapping[str, Summary]) -> str:
    lines = [
        "| Method | Hit@1 | Hit@5 | MRR | Frame recall@1 | Hit@1 (pure) | Hit@5 (pure) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in rows.items():
        lines.append(
            f"| {name} | {s.hit1}/{s.n} ({s.hit1 / s.n:.0%}) | {s.hit5}/{s.n} ({s.hit5 / s.n:.0%}) "
            f"| {s.mrr:.3f} | {s.frame_recall1:.1%} | {s.hit1_pure}/{s.n} | {s.hit5_pure}/{s.n} |"
        )
    return "\n".join(lines)
