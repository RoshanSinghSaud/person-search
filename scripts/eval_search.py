"""Days 11-12: photo, text and hybrid search over the Qdrant index, on the 4 test cases.

Photo (Day 11)
    Each case's query.jpg is searched in the index (filtered to its video). The
    selected frames must equal the golden frames (repo 2's run) and the full
    in-memory scan; frame P/R/F1 is computed against the labels. Query time is
    the median of ``--repeats`` runs after one warm-up.

Text and hybrid (Day 12)
    The 12 witness descriptions are scored against their case's video with:

    * ``text``                - IRRA text vector vs IRRA image vectors (the service's text query)
    * ``hybrid a=0.3/0.5/0.7`` - late fusion of min-max-normalised OSNet (photo) and
      IRRA (text) scores per detection; a = weight of the photo
    * ``embedding sum (repo 1)`` - repo 1's hybrid: CLIP-ReID image embedding + IRRA
      text embedding, summed (a = 0.5, repo 1's default), vs IRRA image vectors
    * ``embedding sum, one space`` - control: the same sum with the photo embedded
      by IRRA's own image tower, so both halves share a space
    * ``photo only`` - OSNet track ranking for the case's photo (4 queries), reference

    Each method's ranking is computed by a full scan; the indexed text and
    hybrid searches must return the same top 5 (checked for every query).

Usage::

    python scripts/eval_search.py --ingest-dir ../legacy/runs/ingest_v1 \
        --cases-dir ../legacy/amrit-pbvs --index ../legacy/runs/week2/index \
        --clipreid ../legacy/runs/week2/clipreid_query_embeddings.npz --out-dir ../legacy/runs/week2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from person_search.config import load_settings
from person_search.core.fusion import l2n, late_fusion, legacy_embedding_sum
from person_search.eval.frame_metrics import FrameLabels, metrics_for_selection, system_metrics
from person_search.eval.golden import GoldenCase, compare
from person_search.eval.text_e2e import (
    QueryResult,
    load_targets,
    load_text_queries,
    rank_with_evidence,
    summarize,
    summary_table,
)
from person_search.factory import build_irra, build_osnet
from person_search.index.base import SearchFilter
from person_search.index.qdrant import QdrantIndex
from person_search.ingest.pipeline import IngestResult
from person_search.pipeline import read_query_image
from person_search.search import photo_search
from person_search.search.indexed import IndexedSearch
from person_search.types import IRRA, OSNET

CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")
DATA = Path(__file__).resolve().parents[1] / "tests" / "golden" / "data"


def timed(fn, repeats: int) -> tuple[Any, list[float]]:
    out = fn()  # warm-up
    ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ms.append((time.perf_counter() - t0) * 1000)
    return out, ms


def pct(ms: list[float], q: float) -> float:
    return float(np.percentile(ms, q))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", type=Path, required=True)
    ap.add_argument("--cases-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--clipreid", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--golden-dir", type=Path, default=DATA / "linux-x86_64")
    ap.add_argument("--labels-dir", type=Path, default=DATA / "labels")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--legacy-alpha", type=float, default=0.5)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    args = ap.parse_args()
    cases = tuple(args.cases)

    settings = load_settings()
    osnet, irra = build_osnet(settings), build_irra(settings)
    ingests = {c: IngestResult.load(args.ingest_dir / f"{c}.npz") for c in cases}
    versions = ingests[cases[0]].encoder_versions
    index_info = json.loads((args.index.parent / "index.json").read_text())
    vid = {c: index_info["videos"][c]["video_id"] for c in cases}
    index = QdrantIndex.local(args.index, versions, create=False)
    searcher = IndexedSearch(index, settings.scoring, settings.fusion)
    labels = {c: FrameLabels.load(args.labels_dir / c / "evaluation.json") for c in cases}
    clipreid = np.load(args.clipreid)
    out: dict[str, Any] = {"model_versions": versions, "scoring": settings.scoring.model_dump()}

    # ------------------------------------------------------------- Day 11 photo
    photo_rows, frame_metrics, pq = {}, {}, {}
    for c in cases:
        img = read_query_image(args.cases_dir / c / "query.jpg")
        osnet.encode_images([img])  # warm-up: the first forward pass is slow
        t0 = time.perf_counter()
        q = osnet.encode_images([img])[0]
        encode_ms = (time.perf_counter() - t0) * 1000
        pq[c] = q
        flt = SearchFilter(video_ids=(vid[c],))
        res, ms = timed(lambda q=q, flt=flt: searcher.photo(q, flt), args.repeats)
        got = res.per_video[vid[c]]
        full = photo_search(ingests[c].observations, q, settings.scoring, OSNET)
        golden = GoldenCase.load(args.golden_dir / f"{c}.json")
        diff = compare(golden, got.selected_samples)
        m = metrics_for_selection(got.selected_samples, labels[c])
        frame_metrics[c] = m
        photo_rows[c] = {
            "matched": got.matched,
            "best_track": got.decision.best.track_id if got.decision.best else None,
            "best_score": round(got.decision.best.score, 4) if got.decision.best else None,
            "accepted_tracks": list(got.decision.accepted),
            "selected_frames": len(got.selected_samples),
            "equals_golden": diff.ok,
            "golden_diff": str(diff),
            "equals_full_scan": got.selected_samples == full.selected_samples,
            "f1": round(m.f1, 4),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "encode_query_ms": round(encode_ms, 1),
            "search_ms_p50": round(pct(ms, 50), 1),
            "search_ms_p95": round(pct(ms, 95), 1),
            "stage_ms_last": {k: round(v, 2) for k, v in res.stats.ms.items()},
            "candidate_tracks": res.stats.candidate_tracks,
            "fetched_points": res.stats.fetched_points,
            "index_points_in_video": index_info["videos"][c]["points"],
        }
        print(
            c,
            {
                k: photo_rows[c][k]
                for k in ("equals_golden", "f1", "search_ms_p50", "candidate_tracks")
            },
        )
    s = system_metrics(frame_metrics)
    out["photo"] = {
        "per_case": photo_rows,
        "system": {
            "f1": s.f1,
            "precision": s.precision,
            "recall": s.recall,
            "tp": s.tp,
            "fp": s.fp,
            "fn": s.fn,
        },
        "all_equal_golden": all(r["equals_golden"] for r in photo_rows.values()),
    }

    # ------------------------------------------------------------- Day 12 text
    queries = [q for q in load_text_queries(DATA / "text_queries.yaml") if q.case in cases]
    targets = load_targets(DATA / "text_targets.yaml")
    irra.encode_texts([queries[0].text])  # warm-up
    t0 = time.perf_counter()
    for q in queries:
        irra.encode_texts([q.text])
    encode_text_ms = (time.perf_counter() - t0) * 1000 / len(queries)
    tvecs = dict(
        zip([q.id for q in queries], irra.encode_texts([q.text for q in queries]), strict=True)
    )
    irra_photo = {
        c: irra.encode_images([read_query_image(args.cases_dir / c / "query.jpg")])[0]
        for c in cases
    }
    results: dict[str, list[QueryResult]] = {}
    checks: list[dict[str, Any]] = []
    text_ms: list[float] = []
    hybrid_ms: list[float] = []

    def add(method: str, qr: QueryResult) -> None:
        results.setdefault(method, []).append(qr)

    for q in queries:
        obs = ingests[q.case].observations
        gt = labels[q.case].ground_truth
        segs = targets[q.case]
        emb = [i for i, o in enumerate(obs) if IRRA in o.embeddings and OSNET in o.embeddings]
        irra_m = np.stack([obs[i].embeddings[IRRA] for i in emb])
        osn_m = np.stack([obs[i].embeddings[OSNET] for i in emb])
        t = tvecs[q.id]
        flt = SearchFilter(video_ids=(vid[q.case],))

        n_obs = len(obs)

        def scores_from(vals: np.ndarray, emb=emb, n=n_obs) -> list[float | None]:
            full: list[float | None] = [None] * n
            for i, v in zip(emb, vals, strict=True):
                full[i] = float(v)
            return full

        # text
        st = irra_m @ t
        ranked = rank_with_evidence(obs, scores_from(st), segs)
        add("text", QueryResult(q.id, q.case, q.level, "text", ranked, gt))
        (got, _), ms = timed(lambda t=t, flt=flt: searcher.text(t, 5, flt), 5)
        text_ms += ms
        checks.append(
            {
                "query": q.id,
                "method": "text",
                "indexed_equals_full": [r.track_id for r in got]
                == [r.track_id for r in ranked[:5]],
            }
        )

        # hybrid late fusion
        sp = osn_m @ pq[q.case]
        for a in (0.3, 0.5, 0.7):
            fused = late_fusion(sp, st, a, "minmax")
            ranked = rank_with_evidence(obs, scores_from(fused), segs)
            name = f"hybrid late fusion a={a}"
            add(name, QueryResult(q.id, q.case, q.level, name, ranked, gt))
            (got, _), ms = timed(
                lambda a=a, t=t, flt=flt, c=q.case: searcher.hybrid(pq[c], t, 5, flt, alpha=a), 3
            )
            hybrid_ms += ms
            checks.append(
                {
                    "query": q.id,
                    "method": name,
                    "indexed_equals_full": [r.track_id for r in got]
                    == [r.track_id for r in ranked[:5]],
                }
            )

        # repo 1's embedding sum (two spaces) and the same sum in one space
        qs = legacy_embedding_sum(clipreid[q.case], t, args.legacy_alpha)
        ranked = rank_with_evidence(obs, scores_from(irra_m @ qs), segs)
        add(
            "embedding sum (repo 1)",
            QueryResult(q.id, q.case, q.level, "embedding sum (repo 1)", ranked, gt),
        )
        qs1 = l2n(args.legacy_alpha * irra_photo[q.case] + (1 - args.legacy_alpha) * t)
        ranked = rank_with_evidence(obs, scores_from(irra_m @ qs1), segs)
        add(
            "embedding sum, one space",
            QueryResult(q.id, q.case, q.level, "embedding sum, one space", ranked, gt),
        )

    for c in cases:  # photo-only reference: track ranking by OSNet (one query per case)
        obs = ingests[c].observations
        sc = [
            float(np.dot(pq[c], o.embeddings[OSNET])) if OSNET in o.embeddings else None
            for o in obs
        ]
        ranked = rank_with_evidence(obs, sc, targets[c])
        add(
            "photo only (OSNet)",
            QueryResult(
                f"{c}-photo", c, "photo", "photo only (OSNet)", ranked, labels[c].ground_truth
            ),
        )

    summaries = {m: summarize(rs, m) for m, rs in results.items()}
    by_level = {
        m: {
            lvl: summarize([r for r in rs if r.level == lvl], m).as_dict()
            for lvl in ("short", "medium", "detailed")
        }
        for m, rs in results.items()
        if m != "photo only (OSNet)"
    }
    out["text"] = {
        "n_queries": len(queries),
        "summary": {m: s.as_dict() for m, s in summaries.items()},
        "by_level": by_level,
        "per_query": {m: [r.as_dict() for r in rs] for m, rs in results.items()},
        "indexed_equals_full_scan": all(c["indexed_equals_full"] for c in checks),
        "indexed_checks": checks,
        "query_ms": {
            "encode_text_mean": round(encode_text_ms, 1),
            "text_p50": round(pct(text_ms, 50), 1),
            "text_p95": round(pct(text_ms, 95), 1),
            "hybrid_p50": round(pct(hybrid_ms, 50), 1),
            "hybrid_p95": round(pct(hybrid_ms, 95), 1),
        },
    }
    index.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "search_eval.json").write_text(json.dumps(out, indent=2))
    md = ["## Photo search over the index (Day 11)", ""]
    md.append(
        "| Case | Equals golden | F1 | Best track (score) | Accepted | Search p50 ms "
        "| Candidate tracks |"
    )
    md.append("|---|---|---:|---|---|---:|---:|")
    for c, r in photo_rows.items():
        same = "yes" if r["equals_golden"] else "NO: " + r["golden_diff"]
        md.append(
            f"| {c} | {same} | {r['f1']:.2%} | {r['best_track']} ({r['best_score']}) "
            f"| {', '.join(r['accepted_tracks'])} | {r['search_ms_p50']} "
            f"| {r['candidate_tracks']} |"
        )
    md += ["", f"System frame F1 {s.f1:.2%} (P {s.precision:.2%}, R {s.recall:.2%})", ""]
    md += [
        "## Text and hybrid search, 12 witness descriptions (Day 12)",
        "",
        summary_table(summaries),
        "",
    ]
    md.append("By description level (Hit@1, Hit@5):")
    md.append("")
    md.append("| Method | short | medium | detailed |")
    md.append("|---|---|---|---|")
    for m, lv in by_level.items():
        md.append(
            "| "
            + m
            + " | "
            + " | ".join(
                f"{lv[x]['hit@1_count']}, {lv[x]['hit@5_count']}"
                for x in ("short", "medium", "detailed")
            )
            + " |"
        )
    qms = out["text"]["query_ms"]
    md += [
        "",
        "Indexed text/hybrid top-5 equals the full scan for every query: "
        f"{out['text']['indexed_equals_full_scan']}",
        f"Query time p50: text {qms['text_p50']} ms, hybrid {qms['hybrid_p50']} ms",
    ]
    (args.out_dir / "search_eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if out["photo"]["all_equal_golden"] and out["text"]["indexed_equals_full_scan"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
