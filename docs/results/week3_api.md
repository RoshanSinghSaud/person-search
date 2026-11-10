# Step 2: the web API, checked with the real models

Machine: Linux-6.18.44-x86_64, 2 vCPU, `PS_TORCH_THREADS=2`, one uvicorn worker
(`make api-index serve`). Index: a copy of the Week 2 index (4 test videos,
16,483 points, irra-v1 + osnet-v1). Every number below comes from
`scripts/smoke_api.py` (`make smoke-api CLIP=...`), which calls the API over
HTTP; its output is `../legacy/runs/week3/api_smoke.json`. Decision 0012.

## Same answers as Week 2

Each query searched its own case's video (`video_ids`), as Week 2 did.

| Check | Result |
|---|---|
| Photo query: accepted tracks equal Week 2 (Day 11) | **4/4** (case 1: 12, 8; case 2: 2, 49, 7; case 3: 4, 162; case 4: 7, 42) |
| Text query: top track and score equal Week 2 (Day 12) | **12/12** (all scores equal to 4 decimals) |
| Hybrid (alpha 0.5): top track is one of the photo query's accepted tracks | **12/12** |

## Latency over HTTP (client wall time, after one warm-up query)

| Query | p50 | max | n |
|---|---:|---:|---:|
| photo | 260 ms | 550 ms | 4 |
| text | 275 ms | 581 ms | 12 |
| photo + text | 932 ms | 1,196 ms | 12 |

Week 2 measured the same searches without HTTP: photo 244-519 ms, text 382 ms,
hybrid 952 ms (p50). Server-side, a photo query is about 140 ms of OSNet on the
photo plus 20 ms of retrieval and 30-380 ms fetching candidate tracks.

## Ingest through the API

The first 10 s of test case 4, re-encoded to H.264 (`ffmpeg -t 10`), so it is a
new video to the index.

| Step | Result |
|---|---|
| `POST /ingest` answered | 202 in 63 ms, job `running` |
| Job finished | 59 s later: 120 sampled frames, 6 tracks, 526 points (177 with vectors) |
| `/health` after | 5 videos, 17,009 points |
| Photo search on the clip (case 4's photo) | match: track 6, score 0.660, frames 14-29, 33-35, 39-50 |
| The full case 4 video uploaded again | 202 in 0.1 s, `succeeded`, `skipped: true`, 0 points added |

The golden run selects samples 13-58 of the full video for the target (track
7); the clip's frames 14-50 are samples 13-49, the same person. Its track ID and
score differ because re-encoding changed the pixels the detector and tracker see.

A text query sent while the clip was being ingested took 0.35 s (2 vCPU shared).

## Searching all videos at once (AUDIT.md F29)

Without `video_ids`, the paper's threshold is applied to each video separately,
so a photo also "matches" in other videos:

| Photo of | Its own video | Also accepted elsewhere (best score) |
|---|---|---|
| case 1 | 12, 8 (0.900) | case 4 video: track 13 (0.687) |
| case 2 | 2, 49, 7 (0.771) | case 3 video: track 162 (0.654) |
| case 3 | 4, 162 (0.738) | case 2 video: 73, 74, 1 (0.726); case 1 video: 14 (0.707) |
| case 4 | 7, 42 (0.683) | none |

Not yet checked whether these are lookalikes or the same students walking
through another camera's view.
