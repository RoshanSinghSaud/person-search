"""How much do the text metrics move if the target labels are off by a few frames?

The target segments in ``text_targets.yaml`` were read off crop strips and are
accurate to about +-4 sampled frames (decision 0007). This shrinks and grows
every segment by ``--shift`` samples at both ends and re-scores the text query
(IRRA, full scan), reporting Hit@1 / Hit@5 for each variant.

Usage::

    python tools/analysis/text_label_sensitivity.py --ingest-dir ../legacy/runs/ingest_v1 \
        --out ../legacy/runs/week2/text_label_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from person_search.config import load_settings
from person_search.eval.frame_metrics import FrameLabels
from person_search.eval.text_e2e import (
    QueryResult,
    load_targets,
    load_text_queries,
    rank_with_evidence,
    summarize,
)
from person_search.factory import build_irra
from person_search.ingest.pipeline import IngestResult
from person_search.types import IRRA

DATA = Path(__file__).resolve().parents[2] / "tests" / "golden" / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shift", type=int, nargs="+", default=[-8, -4, 0, 4, 8])
    args = ap.parse_args()

    queries = load_text_queries(DATA / "text_queries.yaml")
    targets = load_targets(DATA / "text_targets.yaml")
    cases = sorted({q.case for q in queries})
    ingests = {c: IngestResult.load(args.ingest_dir / f"{c}.npz") for c in cases}
    labels = {c: FrameLabels.load(DATA / "labels" / c / "evaluation.json") for c in cases}
    irra = build_irra(load_settings())
    tvecs = irra.encode_texts([q.text for q in queries])
    out = {}
    for d in args.shift:  # d > 0 grows each segment, d < 0 shrinks it
        shifted = {
            c: {t: [(a - d, b + d) for a, b in segs if b + d >= a - d] for t, segs in tr.items()}
            for c, tr in targets.items()
        }
        results = []
        for q, t in zip(queries, tvecs, strict=True):
            obs = ingests[q.case].observations
            scores = [
                float(np.dot(t, o.embeddings[IRRA])) if IRRA in o.embeddings else None
                for o in obs
            ]
            ranked = rank_with_evidence(obs, scores, shifted[q.case])
            results.append(
                QueryResult(q.id, q.case, q.level, "text", ranked, labels[q.case].ground_truth)
            )
        s = summarize(results)
        out[str(d)] = {"hit@1": f"{s.hit1}/{s.n}", "hit@5": f"{s.hit5}/{s.n}", "mrr": s.mrr}
        print(d, out[str(d)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
