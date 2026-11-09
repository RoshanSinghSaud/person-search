"""Scoring a model version on the gate set (Day 8; the CI gate reuses this on Day 24).

* :func:`campus_metrics` - image ReID on the campus test split (OSNet; IRRA's
  image tower as a secondary check).
* :func:`photo_metrics` - the paper's photo query on the 4 cached test cases:
  frame P/R/F1 against the labels, and whether each case is found.
* :func:`text_metrics` - the 12 witness descriptions on the cached test cases:
  Hit@1, Hit@5, MRR (``eval/text_e2e.py``).

All three run on cached crops, so no video decoding or detection is involved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from person_search.config import ScoringSettings
from person_search.eval.campus import FeatureEncoder, ReidSplit, ScoreProtocol, evaluate_encoder
from person_search.eval.frame_metrics import (
    FrameLabels,
    hit_rate,
    metrics_for_selection,
    system_metrics,
)
from person_search.eval.gate_set import CachedCase
from person_search.eval.text_e2e import (
    QueryResult,
    Segments,
    TextQuery,
    rank_with_evidence,
    summarize,
)
from person_search.interfaces import ImageEncoder
from person_search.models.irra_encoder import IRRAEncoder
from person_search.pipeline import read_query_image
from person_search.search import photo_search
from person_search.types import IRRA, OSNET


def campus_metrics(
    encoder: FeatureEncoder, split: ReidSplit, protocols: Sequence[ScoreProtocol] = ("cosine",)
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for proto, m in evaluate_encoder(encoder, split, protocols).items():
        for k, v in m.as_dict((1, 5, 10, 20)).items():
            out[f"campus_{k}_{proto}"] = float(v)
    return out


def photo_metrics(
    encoder: ImageEncoder,
    cases: Mapping[str, CachedCase],
    labels: Mapping[str, FrameLabels],
    query_dir: Path,
    scoring: ScoringSettings,
) -> dict[str, Any]:
    per_case, selections = {}, {}
    for name, case in cases.items():
        obs = case.embed({OSNET: encoder})
        q = encoder.encode_images([read_query_image(query_dir / name / "query.jpg")])[0]
        res = photo_search(obs, q, scoring, OSNET)
        per_case[name] = metrics_for_selection(res.selected_samples, labels[name])
        selections[name] = res.selected_samples
    s = system_metrics(per_case)
    return {
        "photo_f1": s.f1,
        "photo_precision": s.precision,
        "photo_recall": s.recall,
        "photo_hit_rate": hit_rate(per_case),
        "per_case": {
            k: {"f1": m.f1, "tp": m.tp, "fp": m.fp, "fn": m.fn} for k, m in per_case.items()
        },
        "selected_samples": selections,
    }


def text_metrics(
    encoder: IRRAEncoder,
    cases: Mapping[str, CachedCase],
    queries: Sequence[TextQuery],
    targets: Mapping[str, Segments],
    labels: Mapping[str, FrameLabels],
) -> dict[str, Any]:
    embedded = {name: case.embed({IRRA: encoder}) for name, case in cases.items()}
    tvecs = encoder.encode_texts([q.text for q in queries])
    results = []
    for q, t in zip(queries, tvecs, strict=True):
        obs = embedded[q.case]
        scores = [
            float(np.dot(t, o.embeddings[IRRA])) if IRRA in o.embeddings else None for o in obs
        ]
        ranked = rank_with_evidence(obs, scores, targets[q.case])
        results.append(
            QueryResult(q.id, q.case, q.level, "text", ranked, len(labels[q.case].ground_truth))
        )
    s = summarize(results)
    return {
        "text_hit1": s.hit1 / s.n,
        "text_hit5": s.hit5 / s.n,
        "text_mrr": s.mrr,
        "text_frame_recall1": s.frame_recall1,
        "text_hit1_count": s.hit1,
        "text_hit5_count": s.hit5,
        "text_n": s.n,
        "per_query": [r.as_dict(top=3) for r in results],
    }
