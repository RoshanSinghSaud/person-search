"""Step 2: smoke-test a running API with the real models against Week 2's numbers.

Start the server on the Week 2 index first (``make api-index serve``), then::

    python scripts/smoke_api.py --cases-dir ../legacy/amrit-pbvs \
        --ingest /tmp/clip_tc4_10s.mp4 --out ../legacy/runs/week3/api_smoke.json

Checks, per test case and filtered to that case's video (as Week 2 searched):

* photo: the accepted tracks equal docs/results/week2.md (Day 11);
* text: the 12 descriptions' top track and score equal Week 2 (Day 12);
* hybrid (alpha 0.5): the top track is one of the photo query's accepted tracks;
* unfiltered photo: which other videos also "match" (the threshold is per video);
* ``--ingest``: upload a clip, poll its job, then photo-search it with
  ``--ingest-photo`` (default: test case 4's query).

Every request goes over HTTP; latencies are wall times seen by the client.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")
# docs/results/week2.md, Day 11 (accepted tracks) and Day 12 (text top track, score)
WEEK2_PHOTO = {
    "test_case_001": ["12", "8"],
    "test_case_002": ["2", "49", "7"],
    "test_case_003": ["4", "162"],
    "test_case_004": ["7", "42"],
}
WEEK2_TEXT_TOP = {
    "tc1-short": ("3", 0.1588), "tc1-medium": ("8", 0.1957), "tc1-detailed": ("8", 0.182),
    "tc2-short": ("5", 0.1836), "tc2-medium": ("2", 0.2483), "tc2-detailed": ("2", 0.2234),
    "tc3-short": ("96", 0.1622), "tc3-medium": ("162", 0.1842), "tc3-detailed": ("96", 0.2124),
    "tc4-short": ("3", 0.1488), "tc4-medium": ("7", 0.2496), "tc4-detailed": ("7", 0.2728),
}  # fmt: skip
SCORE_TOL = 6e-4  # Week 2 printed 4 decimals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--cases-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=Path("tests/golden/data/text_queries.yaml"))
    ap.add_argument("--ingest", type=Path, help="a video to upload through POST /ingest")
    ap.add_argument("--ingest-photo", type=Path, help="photo to search the ingested video with")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    c = httpx.Client(base_url=args.url, timeout=300, trust_env=False)
    health = c.get("/health").raise_for_status().json()
    by_name = {v["video"]: v["video_id"] for v in c.get("/videos").raise_for_status().json()}
    video_ids = {case: by_name[f"{case}/test_video.mp4"] for case in CASES}
    lat: dict[str, list[float]] = {"photo": [], "text": [], "hybrid": []}

    def search(kind: str | None, files: dict | None = None, **data: Any) -> dict:
        t0 = time.perf_counter()
        r = c.post("/search", files=files, data=data).raise_for_status()
        if kind:
            lat[kind].append((time.perf_counter() - t0) * 1000)
        return r.json()

    def photo(case: str) -> dict:
        return {"photo": ("query.jpg", (args.cases_dir / case / "query.jpg").read_bytes())}

    search(None, photo(CASES[0]), description="a person")  # warm-up: first passes are slow

    out: dict[str, Any] = {"health_before": health, "photo": {}, "photo_all_videos": {}}
    for case in CASES:
        body = search("photo", photo(case), video_ids=video_ids[case])
        v = body["videos"][0]
        out["photo"][case] = {
            "accepted": v["accepted_tracks"],
            "week2": WEEK2_PHOTO[case],
            "same": v["accepted_tracks"] == WEEK2_PHOTO[case],
            "selected_frames": v["selected_frames"],
            "server_ms": body["timing_ms"],
        }
        body = search(None, photo(case))
        out["photo_all_videos"][case] = {
            v["video"]: [v["accepted_tracks"], round(v["best_score"], 4)]
            for v in body["videos"]
            if v["matched"]
        }

    out["text"], out["hybrid"] = {}, {}
    queries = yaml.safe_load(args.queries.read_text())["cases"]
    for case in CASES:
        for q in queries[case]["queries"]:
            body = search("text", description=q["text"], video_ids=video_ids[case])
            top = body["tracks"][0]
            tid, score = WEEK2_TEXT_TOP[q["id"]]
            out["text"][q["id"]] = {
                "top": [top["track_id"], round(top["score"], 4)],
                "week2": [tid, score],
                "same": top["track_id"] == tid and abs(top["score"] - score) < SCORE_TOL,
            }
            body = search("hybrid", photo(case), description=q["text"], video_ids=video_ids[case])
            top_id = body["tracks"][0]["track_id"]
            out["hybrid"][q["id"]] = {"top": top_id, "photo_accepted": top_id in WEEK2_PHOTO[case]}

    out["latency_ms"] = {
        k: {"p50": round(statistics.median(v), 1), "max": round(max(v), 1), "n": len(v)}
        for k, v in lat.items()
    }

    def count(section: str, key: str) -> str:
        return f"{sum(r[key] for r in out[section].values())}/{len(out[section])}"

    out["summary"] = {
        "photo_same_as_week2": count("photo", "same"),
        "text_top_same_as_week2": count("text", "same"),
        "hybrid_top_photo_accepted": count("hybrid", "photo_accepted"),
    }

    if args.ingest:
        t0 = time.perf_counter()
        r = c.post("/ingest", files={"video": (args.ingest.name, args.ingest.read_bytes())})
        accepted_ms = (time.perf_counter() - t0) * 1000
        job = r.raise_for_status().json()
        while job["status"] in ("queued", "running"):
            time.sleep(2)
            job = c.get(f"/jobs/{job['job_id']}").raise_for_status().json()
        found = None
        if job["status"] == "succeeded":
            q = args.ingest_photo or args.cases_dir / CASES[3] / "query.jpg"
            body = search(None, {"photo": ("q.jpg", q.read_bytes())}, video_ids=job["video_id"])
            v = body["videos"][0] if body["videos"] else None
            found = v and {
                "matched": v["matched"],
                "accepted": v["accepted_tracks"],
                "best_score": round(v["best_score"], 4),
                "segments_frames": [(s["start_frame"], s["end_frame"]) for s in v["segments"]],
            }
        out["ingest"] = {
            "http_status": r.status_code,
            "response_ms": round(accepted_ms, 1),
            "job": job,
            "photo_search": found,
        }
        out["health_after"] = c.get("/health").raise_for_status().json()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("summary", "latency_ms")}, indent=2))
    if "ingest" in out:
        print(json.dumps({k: v for k, v in out["ingest"].items() if k != "job"}, indent=2))
        print(json.dumps(out["ingest"]["job"]["result"], indent=2))
    ok = all(r["same"] for r in [*out["photo"].values(), *out["text"].values()])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
