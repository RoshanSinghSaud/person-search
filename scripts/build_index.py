"""Day 10: write ingested videos into the Qdrant index (local mode) and check idempotency.

Reads ``<ingest-dir>/<case>.npz`` (``person-search ingest`` output with both
vectors), hashes each video, writes one point per observation, then writes
everything a second time and checks that the point count did not change.

Usage::

    python scripts/build_index.py --ingest-dir ../legacy/runs/ingest_v1 \
        --cases-dir ../legacy/amrit-pbvs --index ../legacy/runs/week2/index \
        --out ../legacy/runs/week2/index.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from person_search.index.points import points_from_ingest, video_fingerprint
from person_search.index.qdrant import QdrantIndex
from person_search.ingest.pipeline import IngestResult
from person_search.types import IRRA, OSNET

CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", type=Path, required=True)
    ap.add_argument("--cases-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True, help="Qdrant local-mode folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    args = ap.parse_args()

    results = {c: IngestResult.load(args.ingest_dir / f"{c}.npz") for c in args.cases}
    versions = {k: v for k, v in next(iter(results.values())).encoder_versions.items()}
    if set(versions) != {OSNET, IRRA}:
        raise SystemExit(f"ingest must carry both vectors, got {versions}")
    for c, r in results.items():
        if r.encoder_versions != versions:
            raise SystemExit(f"{c}: encoder versions {r.encoder_versions} != {versions}")

    index = QdrantIndex.local(args.index, versions)
    summary: dict = {"collection": index.collection, "model_versions": versions, "videos": {}}
    for case, r in results.items():
        video = args.cases_dir / case / "test_video.mp4"
        vid = video_fingerprint(video)
        pts = points_from_ingest(r, vid, video_name=f"{case}/test_video.mp4", camera=case)
        t0 = time.perf_counter()
        index.upsert(pts)
        summary["videos"][case] = {
            "video_id": vid,
            "points": len(pts),
            "embedded_points": sum(p.payload["embedded"] for p in pts),
            "tracks": len(r.track_ids),
            "upsert_seconds": round(time.perf_counter() - t0, 2),
        }
        print(case, summary["videos"][case])
    first = index.count()
    t0 = time.perf_counter()
    for case, r in results.items():
        vid = summary["videos"][case]["video_id"]
        index.upsert(points_from_ingest(r, vid, video_name=f"{case}/test_video.mp4", camera=case))
    second = index.count()
    summary.update(
        {
            "points_after_first_write": first,
            "points_after_second_write": second,
            "reingest_added_points": second - first,
            "second_write_seconds": round(time.perf_counter() - t0, 2),
        }
    )
    index.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "videos"}, indent=2))
    return 0 if second == first else 1


if __name__ == "__main__":
    raise SystemExit(main())
